import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels"
OUTPUT_DIR = "./dataset/MFS/processed/"

# PROMPTS: Focused purely on surface detail
CLASS_CONFIG = {
    0: { # Oil: Needs to be dark and glossy
        "prompt": "glossy oil texture, black liquid, wet reflection, high viscosity",
        "base_color": (20, 20, 20), # Almost black
        "opacity": 0.85 
    },
    1: { # Water: Transparent but reflective
        "prompt": "water puddle, wet asphalt, mirror reflection, ripples",
        "base_color": (50, 60, 70), # Dark Grey/Blue
        "opacity": 0.4
    },
    2: { # Ice: White/Blueish and matte/shiny
        "prompt": "frozen ice sheet, white frost, rough texture, winter road",
        "base_color": (220, 230, 255), # White-ish Blue
        "opacity": 0.7
    },
    3: { # Crack: Dark and sharp
        "prompt": "deep asphalt crack, fissure, broken pavement, dark gap",
        "base_color": (10, 10, 10), # Pitch black
        "opacity": 0.9
    }
}

NEGATIVE_PROMPT = "grass, dirt, bright lights, cartoon, 3d render, frame, border, blur"

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

def generate_noise_mask(shape, defect_class):
    """
    Generates an organic, irregular mask using Perlin-like noise
    """
    w, h = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Simple blob approach for oil/water/ice
    if defect_class in [0, 1, 2]:
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        angle = random.randint(0, 360)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        # Erode edges to make it irregular
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        
        # Add noise to edges
        noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        mask = cv2.bitwise_and(mask, mask, mask=cv2.threshold(noise, 100, 255, cv2.THRESH_BINARY)[1])
        # Close holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Lightning bolt approach for cracks
    elif defect_class == 3:
        pts = []
        curr_x, curr_y = w//2, 0
        pts.append([curr_x, curr_y])
        steps = 10
        for i in range(steps):
            curr_y += h // steps
            curr_x += random.randint(-w//4, w//4)
            curr_x = max(0, min(w, curr_x))
            pts.append([curr_x, curr_y])
        
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, thickness=random.randint(2, 5))

    return Image.fromarray(mask)

def process_defect(pipe, image_pil, defect):
    w, h = image_pil.size
    x1, y1, x2, y2 = defect["box"]
    
    padding = 64
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    
    crop = image_pil.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size
    
    # 1. Generate Shape Mask
    bx1, by1 = x1 - cx1, y1 - cy1
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0: return image_pil
    
    shape_mask_small = generate_noise_mask((bw, bh), defect["class"])
    full_mask = Image.new("L", (cw, ch), 0)
    full_mask.paste(shape_mask_small, (bx1, by1))
    
    # 2. PRE-COMPOSITE: Darken the area BEFORE AI
    # This forces the "ghost" to become a solid object
    config = CLASS_CONFIG[defect["class"]]
    
    # Create a solid color layer
    color_layer = Image.new("RGB", (cw, ch), config["base_color"])
    
    # Blend it onto the crop using the mask and opacity
    # We create a dimmer mask for the blending
    blend_mask = full_mask.point(lambda p: p * config["opacity"])
    prepped_crop = Image.composite(color_layer, crop, blend_mask)
    
    # 3. AI Refinement
    # Now the AI sees a dark blob and just needs to add texture
    crop_512 = prepped_crop.resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = full_mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Blur mask slightly for AI guidance
    mask_512 = mask_512.filter(ImageFilter.GaussianBlur(3))

    with torch.autocast("cuda"):
        generated = pipe(
            prompt=config["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            image=crop_512,
            mask_image=mask_512,
            strength=0.65, # LOWER strength = Keep the dark color we added!
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
    # 4. Paste Back
    generated = generated.resize((cw, ch), Image.Resampling.LANCZOS)
    
    # Final blend mask (soft edges)
    final_mask = full_mask.filter(ImageFilter.GaussianBlur(5))
    image_pil.paste(generated, (cx1, cy1), mask=final_mask)
    
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
        
        img_pil = Image.open(os.path.join(INPUT_IMAGES_DIR, f)).convert("RGB")
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