import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels"
OUTPUT_DIR = "./dataset/MFS/processed/"

# Prompts focused ONLY on the texture (no background descriptions)
CLASS_PROMPTS = {
    0: "texture of black oil stain on asphalt, dark liquid, top down view",
    1: "texture of water puddle on asphalt, wet road, reflection, top down view",
    2: "texture of white ice patch on asphalt, frost, frozen road, top down view",
    3: "texture of cracked asphalt, road damage, fissure, concrete cracks, top down view"
}

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
                    if cid not in CLASS_PROMPTS: continue
                    xc, yc, wb, hb = map(float, parts[1:])
                    # YOLO to Pixel Center/Size
                    center_x = int(xc * w)
                    center_y = int(yc * h)
                    width_px = int(wb * w)
                    height_px = int(hb * h)
                    defects.append({
                        "class": cid, 
                        "center": (center_x, center_y), 
                        "size": (width_px, height_px)
                    })
    return defects

def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Use Standard Stable Diffusion (Faster, better textures)
    print("⏳ Loading Standard SD Model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    pipe.safety_checker = None
    print("✅ Model Loaded.")

    files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    
    for f in files:
        base = os.path.splitext(f)[0]
        lbl_path = os.path.join(INPUT_LABELS_DIR, base + ".txt")
        if not os.path.exists(lbl_path): continue
        
        # Load Image in OpenCV (BGR format is required for seamlessClone)
        img_path = os.path.join(INPUT_IMAGES_DIR, f)
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape
        
        defects = load_yolo_boxes(lbl_path, w, h)
        if not defects: continue
        
        print(f"Processing {f} ({len(defects)} defects)...")

        for d in defects:
            # --- STEP 1: Generate Texture Patch ---
            # We ask AI for a 512x512 square of PURE texture
            prompt = CLASS_PROMPTS[d["class"]]
            with torch.autocast("cuda"):
                texture_patch = pipe(
                    prompt=prompt, 
                    negative_prompt="border, frame, cartoon, drawing, out of frame, low res",
                    num_inference_steps=25,
                    guidance_scale=7.5
                ).images[0]
            
            # Convert to OpenCV format
            patch_cv = cv2.cvtColor(np.array(texture_patch), cv2.COLOR_RGB2BGR)
            
            # Resize patch to fit the YOLO box (plus a little extra for blending)
            target_w, target_h = d["size"]
            # Enlarge slightly so we blend OUTSIDE the box
            scale = 1.2 
            patch_w = int(target_w * scale)
            patch_h = int(target_h * scale)
            
            # Safety check: Don't resize to 0
            if patch_w < 10 or patch_h < 10: continue
            
            patch_resized = cv2.resize(patch_cv, (patch_w, patch_h))

            # --- STEP 2: Create a Mask ---
            # White in middle, Black borders (Ellipse shape is best for blending)
            mask = 255 * np.ones(patch_resized.shape, patch_resized.dtype)
            
            # --- STEP 3: Poisson Blending (The Magic) ---
            center = d["center"]
            
            # seamlessClone requires the center to be inside the image.
            # If the defect is on the very edge, skip it or clamp it.
            if (center[0] < patch_w//2 or center[0] > w - patch_w//2 or 
                center[1] < patch_h//2 or center[1] > h - patch_h//2):
                # Simple paste fallback for edge cases
                print("Skipping edge blending for safety.")
                continue

            try:
                # MIXED_CLONE is often better for textures (transparency)
                # NORMAL_CLONE is better for solid objects
                mode = cv2.MIXED_CLONE if d["class"] in [0, 1] else cv2.NORMAL_CLONE
                
                img_cv = cv2.seamlessClone(
                    patch_resized, 
                    img_cv, 
                    mask, 
                    center, 
                    mode
                )
            except Exception as e:
                print(f"Blend failed: {e}")

        # Save Final
        cv2.imwrite(os.path.join(OUTPUT_DIR, f), img_cv)
        print(f"Saved: {f}")

if __name__ == "__main__":
    main()