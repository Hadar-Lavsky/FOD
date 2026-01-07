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

# PROMPTS: Texture focused
CLASS_PROMPTS = {
    0: "oil spill on asphalt, dark liquid, shiny reflection, high detail, realistic texture",
    1: "water puddle on asphalt, wet, reflection, high detail, realistic",
    2: "ice patch on asphalt, white frost, frozen texture, slippery",
    3: "cracked asphalt, road damage, fissure, rough concrete texture"
}

# CONFIG
PADDING = 100       # How much context (pixels) around the box the AI gets to see
STRENGTH = 0.85     # 0.0 = Keep drawing, 1.0 = New image. 0.85 is the sweet spot.

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def load_yolo_boxes(label_path, w, h):
    """Reads YOLO file, converts to pixel coordinates [x1, y1, x2, y2]."""
    defects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid not in CLASS_PROMPTS: continue
                    xc, yc, wb, hb = map(float, parts[1:])
                    x1 = (xc - wb/2) * w
                    y1 = (yc - hb/2) * h
                    x2 = (xc + wb/2) * w
                    y2 = (yc + hb/2) * h
                    defects.append({"class": cid, "box": [x1, y1, x2, y2]})
    return defects

def draw_rough_guide(image_cv, box, class_id):
    """
    STEP 1: The "Paste" Logic.
    Draws a crude shape on the image to guide the AI.
    """
    x1, y1, x2, y2 = map(int, box)
    w_box, h_box = x2-x1, y2-y1
    center = (int(x1 + w_box/2), int(y1 + h_box/2))
    
    # Create a mask for this specific shape
    # We draw on the MAIN image directly (destructive edit), 
    # but we will rely on AI to make it look good.
    
    if class_id == 0: # Oil (Dark Blob)
        axes = (int(w_box/2 * 0.8), int(h_box/2 * 0.8))
        angle = random.randint(0, 180)
        cv2.ellipse(image_cv, center, axes, angle, 0, 360, (20, 20, 20), -1) # Dark Grey
        
    elif class_id == 1: # Water (Blue-ish Blob)
        axes = (int(w_box/2 * 0.9), int(h_box/2 * 0.7))
        angle = random.randint(0, 180)
        cv2.ellipse(image_cv, center, axes, angle, 0, 360, (100, 100, 120), -1) # Blue-Grey

    elif class_id == 2: # Ice (White Patch)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (220, 220, 240), -1) # White-ish
        
    elif class_id == 3: # Crack (Black Line)
        # Draw a jagged line
        pts = []
        for i in range(5):
            px = x1 + (w_box * i / 4)
            py = y1 + (h_box/2) + random.randint(-int(h_box/4), int(h_box/4))
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_cv, [pts], False, (10, 10, 10), thickness=int(min(w_box, h_box)/3))

    return image_cv

def process_defect(pipe, image_pil, defect):
    """
    STEP 2: The "Refine" Logic.
    Crops the rough drawing and asks AI to fix it.
    """
    w, h = image_pil.size
    x1, y1, x2, y2 = defect["box"]
    
    # 1. Add Context Padding
    cx1 = max(0, int(x1 - PADDING))
    cy1 = max(0, int(y1 - PADDING))
    cx2 = min(w, int(x2 + PADDING))
    cy2 = min(h, int(y2 + PADDING))
    
    # 2. Crop
    crop = image_pil.crop((cx1, cy1, cx2, cy2))
    crop_w, crop_h = crop.size
    
    # 3. Create Mask (The Area AI is allowed to change)
    # We give it the full box + a little bit of the edge to blend
    mask = Image.new("L", (crop_w, crop_h), 0)
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(mask)
    
    # Coordinates relative to crop
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    
    draw.rectangle([rx1, ry1, rx2, ry2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(10)) # Soft edges
    
    # 4. Resize for SD
    crop_512 = crop.resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 5. Run Inpainting
    # STRENGTH is key here. 0.85 means "Change pixels a lot, but keep structure"
    with torch.autocast("cuda"):
        generated = pipe(
            prompt=CLASS_PROMPTS[defect["class"]],
            negative_prompt="cartoon, drawing, 3d render, blur, frame, border",
            image=crop_512,
            mask_image=mask_512,
            strength=STRENGTH,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
    # 6. Paste Back
    generated = generated.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    image_pil.paste(generated, (cx1, cy1), mask=mask)
    
    return image_pil

def main():
    ensure_dir(OUTPUT_DIR)

    # We use Inpainting Pipeline because it is best for blending textures
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
        
        # Load Image
        img_path = os.path.join(INPUT_IMAGES_DIR, f)
        
        # --- PHASE 1: PASTE ROUGH GUIDES (OpenCV) ---
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape
        defects = load_yolo_boxes(lbl_path, w, h)
        
        if not defects: continue
        print(f"Processing {f}...")

        # Draw "Ugly" Guides
        for d in defects:
            img_cv = draw_rough_guide(img_cv, d["box"], d["class"])
            
        # Convert to PIL for Phase 2
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # --- PHASE 2: AI REFINE (Diffusers) ---
        for d in defects:
            img_pil = process_defect(pipe, img_pil, d)
            
        img_pil.save(os.path.join(OUTPUT_DIR, f))
        print(f"Saved: {f}")

if __name__ == "__main__":
    main()