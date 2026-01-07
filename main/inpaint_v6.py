import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels"
OUTPUT_DIR = "./dataset/MFS/processed/"

# PROMPTS: "Flat" style to match simulator footage
# We ask for "video game texture" to avoid photorealistic sun glare.
CLASS_CONFIG = {
    0: { # Oil
        "prompt": "stain on asphalt, dark tar texture, flat lighting, video game asset, top down",
        "mask_type": "blob" 
    },
    1: { # Water
        "prompt": "puddle on asphalt, wet road texture, dark reflection, flat lighting, top down",
        "mask_type": "blob"
    },
    2: { # Ice
        "prompt": "white frost patch on asphalt, frozen road, winter texture, flat lighting",
        "mask_type": "blob"
    },
    3: { # Crack
        "prompt": "asphalt crack, road damage, fissure, dark line, flat texture",
        "mask_type": "crack"
    }
}

NEGATIVE_PROMPT = "3d render, pebbles, high contrast, sun glare, shadows, borders, frame, blur, noise"

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def load_yolo_boxes(label_path, w, h):
    defects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid not in CLASS_CONFIG: continue
                    xc, yc, wb, hb = map(float, parts[1:])
                    x1 = int((xc - wb/2) * w)
                    y1 = int((yc - hb/2) * h)
                    x2 = int((xc + wb/2) * w)
                    y2 = int((yc + hb/2) * h)
                    defects.append({"class": cid, "box": [x1, y1, x2, y2]})
    return defects

def generate_procedural_mask(shape, mask_type):
    """
    Creates a random black/white mask (blob or crack) to constrain the AI.
    """
    w, h = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    
    if mask_type == "blob":
        # Draw random ellipses to make an irregular blob
        num_blobs = random.randint(3, 6)
        for _ in range(num_blobs):
            axes = (random.randint(w//4, w//2), random.randint(h//4, h//2))
            angle = random.randint(0, 360)
            offset_x = random.randint(-w//4, w//4)
            offset_y = random.randint(-h//4, h//4)
            pt = (center[0] + offset_x, center[1] + offset_y)
            cv2.ellipse(mask, pt, axes, angle, 0, 360, 255, -1)
            
    elif mask_type == "crack":
        # Draw a jagged lightning bolt
        pts = []
        curr_x, curr_y = w//2, 0  # Start top middle
        pts.append([curr_x, curr_y])
        
        segments = 8
        step_y = h // segments
        
        for i in range(segments):
            curr_y += step_y
            curr_x += random.randint(-w//5, w//5) # Jitter X
            # Clamp to borders
            curr_x = max(0, min(w, curr_x))
            pts.append([curr_x, curr_y])
            
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Draw thick line
        cv2.polylines(mask, [pts], False, 255, thickness=random.randint(3, 8))

    # Convert to PIL and Blur edges for blending
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=5))
    return mask_pil

def process_defect(pipe, image_pil, defect):
    w, h = image_pil.size
    x1, y1, x2, y2 = defect["box"]
    
    # 1. PADDING (Context is King)
    padding = 64
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    
    # Crop the area
    crop = image_pil.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size
    
    # 2. GENERATE MASK
    # We create a mask that fits inside the YOLO box area of the crop
    # Rel coords of the box inside the crop
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    
    box_w = rx2 - rx1
    box_h = ry2 - ry1
    
    if box_w <= 0 or box_h <= 0: return image_pil

    # Generate the shape (blob/crack)
    config = CLASS_CONFIG[defect["class"]]
    shape_mask = generate_procedural_mask((box_w, box_h), config["mask_type"])
    
    # Place this shape mask onto a full black mask
    full_mask = Image.new("L", (cw, ch), 0)
    full_mask.paste(shape_mask, (rx1, ry1))
    
    # 3. RESIZE FOR SD (512x512)
    crop_512 = crop.resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = full_mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 4. INPAINT
    with torch.autocast("cuda"):
        generated = pipe(
            prompt=config["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            image=crop_512,
            mask_image=mask_512,
            strength=1.0, # Replace pixels fully (but only inside the mask!)
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
    # 5. PASTE BACK
    generated = generated.resize((cw, ch), Image.Resampling.LANCZOS)
    # We use the mask again to blend only the painted pixels
    image_pil.paste(generated, (cx1, cy1), mask=full_mask)
    
    return image_pil

def main():
    ensure_dir(OUTPUT_DIR)

    print("⏳ Loading Inpainting Model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    pipe.safety_checker = None
    print("✅ Model Loaded.")

    files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith('.jpg')]
    
    for f in files:
        base = os.path.splitext(f)[0]
        lbl_path = os.path.join(INPUT_LABELS_DIR, base + ".txt")
        if not os.path.exists(lbl_path): continue
        
        img_path = os.path.join(INPUT_IMAGES_DIR, f)
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        
        defects = load_yolo_boxes(lbl_path, w, h)
        if not defects: continue
        
        print(f"Processing {f}...")
        for d in defects:
            img_pil = process_defect(pipe, img_pil, d)
            
        img_pil.save(os.path.join(OUTPUT_DIR, f))
        print(f"Saved: {f}")

if __name__ == "__main__":
    main()