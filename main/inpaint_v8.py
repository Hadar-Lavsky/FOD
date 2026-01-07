import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageStat
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels"
OUTPUT_DIR = "./dataset/MFS/processed/"

# Prompts: We ask for "neutral" backgrounds to make blending easier
CLASS_CONFIG = {
    0: { # Oil
        "prompt": "top down view of a black oil spill on grey asphalt, tar texture, hyperrealistic",
        "mask_type": "blob" 
    },
    1: { # Water
        "prompt": "top down view of a water puddle on grey asphalt, wet road, dark reflection",
        "mask_type": "blob"
    },
    2: { # Ice
        "prompt": "top down view of white ice patch on grey asphalt, frost texture, frozen road",
        "mask_type": "blob"
    },
    3: { # Crack
        "prompt": "top down view of deep asphalt crack, road fissure, line of damage",
        "mask_type": "crack"
    }
}

NEGATIVE_PROMPT = "colors, sunset, grass, borders, frame, blur, 3d render, watermark"

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
    w, h = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    
    if mask_type == "blob":
        num_blobs = random.randint(3, 5)
        for _ in range(num_blobs):
            axes = (random.randint(w//5, w//2), random.randint(h//5, h//2))
            angle = random.randint(0, 360)
            offset_x = random.randint(-w//5, w//5)
            offset_y = random.randint(-h//5, h//5)
            pt = (center[0] + offset_x, center[1] + offset_y)
            cv2.ellipse(mask, pt, axes, angle, 0, 360, 255, -1)
            
    elif mask_type == "crack":
        pts = []
        curr_x, curr_y = w//2, 0
        pts.append([curr_x, curr_y])
        steps = 15
        for i in range(steps):
            curr_y += h // steps
            curr_x += random.randint(-w//5, w//5)
            curr_x = max(0, min(w, curr_x))
            pts.append([curr_x, curr_y])
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, thickness=random.randint(3, 6))

    return Image.fromarray(mask)

def color_transfer(source, reference):
    """
    Matches the mean and standard deviation of the 'source' image 
    to match the 'reference' image (Reinhard Color Transfer).
    This forces the AI generation to match the simulator's lighting.
    """
    s = np.array(source).astype(np.float32)
    r = np.array(reference).astype(np.float32)
    
    # Calculate stats for each channel (R, G, B)
    mu_s, std_s = cv2.meanStdDev(s)
    mu_r, std_r = cv2.meanStdDev(r)
    
    mu_s = mu_s.flatten()
    std_s = std_s.flatten()
    mu_r = mu_r.flatten()
    std_r = std_r.flatten()
    
    res = np.zeros_like(s)
    for i in range(3):
        # Avoid division by zero
        if std_s[i] == 0: std_s[i] = 1e-5
        
        # (Pixel - Mean_Src) * (Std_Ref / Std_Src) + Mean_Ref
        res[:,:,i] = (s[:,:,i] - mu_s[i]) * (std_r[i] / std_s[i]) + mu_r[i]
    
    # Clip to valid range
    res = np.clip(res, 0, 255).astype(np.uint8)
    return Image.fromarray(res)

def process_defect(pipe, image_pil, defect):
    w, h = image_pil.size
    x1, y1, x2, y2 = defect["box"]
    
    # PADDING: We need context to know the "Reference Color"
    padding = 64
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    
    crop = image_pil.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size
    
    # Generate Mask
    bx1, by1 = x1 - cx1, y1 - cy1
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0: return image_pil
    
    config = CLASS_CONFIG[defect["class"]]
    shape_mask = generate_procedural_mask((bw, bh), config["mask_type"])
    full_mask = Image.new("L", (cw, ch), 0)
    full_mask.paste(shape_mask, (bx1, by1))
    
    # Resize for SD
    crop_512 = crop.resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = full_mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Blur mask for AI
    mask_512_blurred = mask_512.filter(ImageFilter.GaussianBlur(5))

    # INPAINTING (Strength 1.0 -> New Texture)
    with torch.autocast("cuda"):
        generated = pipe(
            prompt=config["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            image=crop_512,
            mask_image=mask_512_blurred,
            strength=1.0, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    generated = generated.resize((cw, ch), Image.Resampling.LANCZOS)
    
    # COLOR CORRECTION
    # We force the generated image to match the original crop's color palette
    corrected_generated = color_transfer(generated, crop)
    
    # BLEND
    # Use the mask to paste ONLY the defect
    final_mask = full_mask.filter(ImageFilter.GaussianBlur(3))
    image_pil.paste(corrected_generated, (cx1, cy1), mask=final_mask)
    
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