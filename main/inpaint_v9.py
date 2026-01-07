import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline  # <--- Standard Pipeline (Text-to-Image)

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels"
OUTPUT_DIR = "./dataset/MFS/processed/"

# PROMPTS: We ask for clean, high-contrast texture maps
CLASS_CONFIG = {
    0: { # Oil: Darken the road (Multiply)
        "prompt": "texture of black oil, dark liquid surface, glossy, high contrast, seamless",
        "blend_mode": "MULTIPLY",
        "opacity": 0.9,
        "mask_type": "blob"
    },
    1: { # Water: Darken the road (Multiply)
        "prompt": "texture of water ripples, dark liquid, wet reflection, high contrast",
        "blend_mode": "MULTIPLY",
        "opacity": 0.6,
        "mask_type": "blob"
    },
    2: { # Ice: Lighten the road (Screen)
        "prompt": "texture of white frost, ice sheet, frozen snow, rough detail",
        "blend_mode": "SCREEN",
        "opacity": 0.8,
        "mask_type": "blob"
    },
    3: { # Crack: Standard Paste (Normal)
        "prompt": "texture of asphalt crack, dark fissure, road damage, gap, white background",
        "blend_mode": "NORMAL",
        "opacity": 0.95,
        "mask_type": "crack"
    }
}

NEGATIVE_PROMPT = "colors, grass, blur, 3d render, frame, border, watermark, text"

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
                    # YOLO to Rect
                    x1 = int((xc - wb/2) * w)
                    y1 = int((yc - hb/2) * h)
                    x2 = int((xc + wb/2) * w)
                    y2 = int((yc + hb/2) * h)
                    defects.append({"class": cid, "box": [x1, y1, x2, y2]})
    return defects

def generate_procedural_mask(shape, mask_type):
    """ Generates Organic High-Res Masks """
    w, h = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    
    if mask_type == "blob":
        # Overlapping ellipses for organic liquid shape
        num_blobs = random.randint(3, 7)
        for _ in range(num_blobs):
            axes = (random.randint(w//4, w//2), random.randint(h//4, h//2))
            angle = random.randint(0, 360)
            shift_x = random.randint(-w//4, w//4)
            shift_y = random.randint(-h//4, h//4)
            pt = (center[0] + shift_x, center[1] + shift_y)
            cv2.ellipse(mask, pt, axes, angle, 0, 360, 255, -1)
        # Heavy blur for soft liquid edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
    elif mask_type == "crack":
        # Jagged Lightning Bolt
        pts = []
        cx, cy = w//2, 0
        pts.append([cx, cy])
        steps = 12
        for i in range(steps):
            cy += h // steps
            cx += random.randint(-w//6, w//6)
            cx = max(0, min(w, cx))
            pts.append([cx, cy])
        
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, thickness=random.randint(2, 5))
        # Cracks need sharper edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return Image.fromarray(mask)

def apply_blending(background, overlay, mask, mode, opacity):
    """ VFX Compositing Logic """
    bg = background.convert("RGBA")
    fg = overlay.convert("RGBA")
    mask_l = mask.convert("L")
    
    if fg.size != bg.size:
        fg = fg.resize(bg.size)
        mask_l = mask_l.resize(bg.size)

    bg_arr = np.array(bg).astype(float)
    fg_arr = np.array(fg).astype(float)
    mask_arr = np.array(mask_l).astype(float) / 255.0
    
    # Global Opacity
    mask_arr = mask_arr * opacity

    out_arr = bg_arr.copy()
    
    if mode == "MULTIPLY":
        # Standard Multiply (Darkens)
        # Result = (BG * FG) / 255
        multiplied = (bg_arr[:,:,:3] * fg_arr[:,:,:3]) / 255.0
        for c in range(3):
            out_arr[:,:,c] = (multiplied[:,:,c] * mask_arr) + (bg_arr[:,:,c] * (1 - mask_arr))
            
    elif mode == "SCREEN":
        # Standard Screen (Lightens)
        # Result = 1 - (1-BG)*(1-FG)
        screened = 255 - ((255 - bg_arr[:,:,:3]) * (255 - fg_arr[:,:,:3]) / 255.0)
        for c in range(3):
            out_arr[:,:,c] = (screened[:,:,c] * mask_arr) + (bg_arr[:,:,c] * (1 - mask_arr))
            
    elif mode == "NORMAL":
        # Standard Paste
        for c in range(3):
            out_arr[:,:,c] = (fg_arr[:,:,c] * mask_arr) + (bg_arr[:,:,c] * (1 - mask_arr))

    return Image.fromarray(np.uint8(out_arr)).convert("RGB")

def process_defect(pipe, image_pil, defect):
    w, h = image_pil.size
    x1, y1, x2, y2 = defect["box"]
    
    # Padding for seamless integration
    padding = 64
    cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
    cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
    
    crop = image_pil.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size
    
    config = CLASS_CONFIG[defect["class"]]
    
    # 1. Generate Procedural Mask
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5: return image_pil
    
    shape_mask_small = generate_procedural_mask((bw, bh), config["mask_type"])
    full_mask = Image.new("L", (cw, ch), 0)
    full_mask.paste(shape_mask_small, (x1 - cx1, y1 - cy1))

    # 2. Generate Texture Patch (Text-to-Image)
    # We generate a large square texture first
    with torch.autocast("cuda"):
        texture_patch = pipe(
            prompt=config["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]
    
    texture_patch = texture_patch.resize((cw, ch), Image.Resampling.LANCZOS)
    
    # 3. Blend using VFX Modes
    final_crop = apply_blending(
        crop, 
        texture_patch, 
        full_mask, 
        config["blend_mode"], 
        config["opacity"]
    )
    
    # 4. Paste back
    image_pil.paste(final_crop, (cx1, cy1))
    return image_pil

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading Standard SD 1.5 Model...")
    # NOTE: Switched to StableDiffusionPipeline (Standard Text-to-Image)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
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