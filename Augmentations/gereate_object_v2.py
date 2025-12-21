import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
import random

# --- 1. SETTINGS & PATHS ---
SOURCE_IMG = "source_images/image_1.avif"
# This mask should be Black (background) and White (valid area/road)
VALID_MASK_PATH = "source_images/road_mask_1.png" 

DATASET_DIR = "yolo_dataset" 

# YOLOv12/Standard YOLO directory structure
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

def find_valid_location(mask_pil, img_w, img_h):
    """
    Analyzes the valid_mask to find a random spot for the spill.
    Returns: center_x, center_y, spill_w, spill_h (in pixels)
    """
    # Convert mask to numpy array (255 = valid area, 0 = invalid)
    mask_arr = np.array(mask_pil.convert("L").resize((img_w, img_h)))
    
    # Find all coordinates where the mask is white (valid)
    # np.argwhere returns (row, col) -> (y, x)
    valid_coords = np.argwhere(mask_arr > 128)
    
    if len(valid_coords) == 0:
        raise ValueError("The provided mask has no valid white areas!")

    # Pick a random valid center point
    # We add a buffer so we don't pick a pixel right on the edge of the mask
    idx = random.randint(0, len(valid_coords) - 1)
    cy, cx = valid_coords[idx]

    # Define random spill size (e.g., between 5% and 15% of image width)
    # You can tune these ranges
    spill_w = random.randint(int(img_w * 0.10), int(img_w * 0.25))
    spill_h = random.randint(int(img_h * 0.05), int(img_h * 0.15))

    return cx, cy, spill_w, spill_h

def generate_yolo_sample():
    if not os.path.exists(SOURCE_IMG):
        print(f"File {SOURCE_IMG} not found!")
        return
    if not os.path.exists(VALID_MASK_PATH):
        print(f"File {VALID_MASK_PATH} not found!")
        return

    # Load and prepare image
    init_img = Image.open(SOURCE_IMG).convert("RGB").resize((512, 512))
    w_img, h_img = init_img.size
    
    # Load Valid Area Mask
    valid_area_mask = Image.open(VALID_MASK_PATH)

    # --- 3. DYNAMIC LOCATION FINDING ---
    try:
        cx_px, cy_px, w_px, h_px = find_valid_location(valid_area_mask, w_img, h_img)
    except ValueError as e:
        print(e)
        return

    # Calculate bounding box coordinates
    x1 = int(cx_px - w_px / 2)
    y1 = int(cy_px - h_px / 2)
    x2 = int(cx_px + w_px / 2)
    y2 = int(cy_px + h_px / 2)

    # Create the Inpainting Mask (The specific hole for the AI to fill)
    inp_mask = Image.new("L", (w_img, h_img), 0)
    draw = ImageDraw.Draw(inp_mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    
    # Blur the mask slightly for better blending
    inp_mask = inp_mask.filter(ImageFilter.GaussianBlur(radius=5))

    # --- 4. NORMALIZE FOR YOLO ---
    # YOLO Format: x_center, y_center, width, height (all 0.0 to 1.0 relative to image size)
    norm_xc = cx_px / w_img
    norm_yc = cy_px / h_img
    norm_w = w_px / w_img
    norm_h = h_px / h_img

    # Clamp values to ensure 0-1 range (safety)
    norm_xc = max(0.0, min(1.0, norm_xc))
    norm_yc = max(0.0, min(1.0, norm_yc))

    # --- 5. GENERATION ---
    print(f"Generating spill at: {norm_xc:.2f}, {norm_yc:.2f}...")
    prompt = "A thick, opaque pitch-black industrial oil spill, wet reflective liquid, high contrast on gray asphalt"
    
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            image=init_img,
            mask_image=inp_mask,
            guidance_scale=20.0, 
            strength=1.0
        ).images[0]

    # --- 6. SAVE FOR YOLOv12 ---
    # Using a random ID so you can run this multiple times without overwriting
    file_id = f"oil_spill_{random.randint(1000, 9999)}"
    
    # Save Image
    result.save(os.path.join(IMG_OUT, f"{file_id}.jpg"))
    
    # Save Label: class_id x_center y_center width height
    # Assuming Class ID 0 for Oil Spill
    with open(os.path.join(LBL_OUT, f"{file_id}.txt"), "w") as f:
        f.write(f"0 {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"SUCCESS: {file_id} saved to {DATASET_DIR}")
    # Optional: Open the image to verify
    # os.startfile(os.path.abspath(os.path.join(IMG_OUT, f"{file_id}.jpg")))

if __name__ == "__main__":
    generate_yolo_sample()