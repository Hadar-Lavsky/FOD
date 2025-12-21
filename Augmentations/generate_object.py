import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

# --- 1. SETTINGS & PATHS ---
SOURCE_IMG = "source_images/image_1.avif"
DATASET_DIR = "yolo_dataset" 

# YOLOv12 standard directory structure
IMG_OUT = os.path.join(DATASET_DIR, "images/train")
LBL_OUT = os.path.join(DATASET_DIR, "labels/train")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# --- 2. LOAD MODEL ---
print("--- Loading Stable Diffusion Inpainting ---")
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    pipe.safety_checker = None 
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def generate_yolo_sample():
    if not os.path.exists(SOURCE_IMG):
        print(f"File {SOURCE_IMG} not found!")
        return

    # Load and prepare image
    init_img = Image.open(SOURCE_IMG).convert("RGB").resize((512, 512))
    w, h = init_img.size

    # --- 3. DYNAMIC MASK (Elliptical and Blurred for realism) ---
    # Centered on the asphalt
    x_c, y_c, w_rel, h_rel = 0.5, 0.8, 0.25, 0.12 
    
    x1, y1 = int((x_c - w_rel/2) * w), int((y_c - h_rel/2) * h)
    x2, y2 = int((x_c + w_rel/2) * w), int((y_c + h_rel/2) * h)

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

    # --- 4. GENERATION ---
    print("Generating jet-black industrial oil spill...")
    prompt = "A thick, opaque pitch-black industrial oil spill, wet reflective liquid, high contrast on gray asphalt"
    
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            guidance_scale=25.0, # High force to override runway markings
            strength=1.0
        ).images[0]

    # --- 5. SAVE FOR YOLOv12 ---
    file_id = "runway_oil_01"
    
    # Save Image
    result.save(os.path.join(IMG_OUT, f"{file_id}.jpg"))
    
    # Save Label: class_id x_center y_center width height
    with open(os.path.join(LBL_OUT, f"{file_id}.txt"), "w") as f:
        f.write(f"0 {x_c:.6f} {y_c:.6f} {w_rel:.6f} {h_rel:.6f}")

    print(f"SUCCESS: Data saved to {DATASET_DIR}")
    os.startfile(os.path.abspath(os.path.join(IMG_OUT, f"{file_id}.jpg")))

if __name__ == "__main__":
    generate_yolo_sample()