import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLImg2ImgPipeline

# ==========================================
#               CONFIGURATION
# ==========================================

OUTPUT_DIR = "./main/results"                    # Final dir
OUTPUT_FILENAME_PREFIX = 'oil'
NUM_IMAGES = 299

YOLO_LABEL_INDEX = 0 

# OIL PROMPT
PROMPT = (
    "raw photograph, satellite view of asphalt airport runway surface, "
    "heavy grain texture, dirt, weathered concrete, black oil puddle, "
    "detailed ground texture, daylight"
)

NEGATIVE_PROMPT = "smooth, digital painting, cartoon, drawing, 3d render, blur, low res, clean"

AI_STRENGTH = 0.55

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def add_noise_and_blur(img_cv):
    """ Adds grit to the perfect OpenCV drawing so AI doesn't make it cartoonish. """
    # 1. Slight Blur to soften razor-sharp computer edges
    img_blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
    
    # 2. Add random noise grain
    noise = np.random.randint(0, 50, img_blurred.shape, dtype='uint8')
    # Blend noise into image (add grain texture)
    img_noisy = cv2.addWeighted(img_blurred, 0.8, noise, 0.2, 0)
    
    return img_noisy

# ==========================================
#        NEW BOUNDING BOX FUNCTIONS
# ==========================================

def get_ellipse_bbox(center, axes, angle):
    """
    Calculates the axis-aligned bounding box for a rotated ellipse.
    center: (cx, cy)
    axes: (width_radius, height_radius) -> Note: OpenCV uses (width/2, height/2) roughly
    angle: rotation angle in degrees
    """
    cx, cy = center
    a, b = axes # Semi-axes
    theta = np.radians(angle)

    # Parametric equation calculation for BBox extent
    # Width extent (x)
    ux = a * np.cos(theta)
    vx = b * np.sin(theta)
    bbox_halfwidth = np.sqrt(ux*ux + vx*vx)
    
    # Height extent (y)
    uy = a * np.sin(theta)
    vy = b * np.cos(theta)
    bbox_halfheight = np.sqrt(uy*uy + vy*vy)

    min_x = cx - bbox_halfwidth
    max_x = cx + bbox_halfwidth
    min_y = cy - bbox_halfheight
    max_y = cy + bbox_halfheight

    return (min_x, min_y, max_x, max_y)

def save_yolo_label(path, bbox, img_w, img_h, class_id):
    """
    Saves a .txt file with YOLO format: class_id center_x center_y width height
    All coordinates are normalized (0 to 1).
    """
    min_x, min_y, max_x, max_y = bbox
    
    # Calculate center and width/height
    w = max_x - min_x
    h = max_y - min_y
    center_x = min_x + (w / 2)
    center_y = min_y + (h / 2)
    
    # Normalize
    norm_cx = center_x / img_w
    norm_cy = center_y / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    # Clamp values to 0-1 just in case
    norm_cx = min(max(norm_cx, 0), 1)
    norm_cy = min(max(norm_cy, 0), 1)
    norm_w = min(max(norm_w, 0), 1)
    norm_h = min(max(norm_h, 0), 1)
    
    line = f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
    
    with open(path, 'w') as f:
        f.write(line)

# ==========================================

def create_base_layout(w=512, h=512):
    # 1. Base Asphalt (Slightly varied grey)
    base_color = random.randint(70, 90)
    img = np.full((h, w, 3), (base_color, base_color, base_color), dtype=np.uint8)
    
    # 2. Draw Runway Markings (Off-White, not pure white)
    mark_color = (210, 210, 210)
    bar_width = w // 15
    bar_height = h // 10
    spacing = w // 20
    
    # Threshold bars
    start_x = spacing
    for i in range(4):
        cv2.rectangle(img, (start_x, h - bar_height), (start_x + bar_width, h), mark_color, -1)
        end_x = w - start_x - bar_width
        cv2.rectangle(img, (end_x, h - bar_height), (end_x + bar_width, h), mark_color, -1)
        start_x += bar_width + spacing

    # Centerline
    dash_h = h // 8
    dash_w = w // 40
    curr_y = 0
    while curr_y < h - bar_height - spacing:
        cv2.rectangle(img, (w//2 - dash_w//2, curr_y), (w//2 + dash_w//2, curr_y + dash_h), mark_color, -1)
        curr_y += dash_h * 2

    # 3. Draw Oil Blob (Pure Black)
    # Note: cv2.ellipse takes axes as (half_width, half_height)
    blob_w = random.randint(w//8, w//4)
    blob_h = random.randint(h//8, h//4)
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    angle = random.randint(0, 180)
    
    cv2.ellipse(img, (center_x, center_y), (blob_w, blob_h), angle, 0, 360, (5, 5, 5), -1)
    
    # Calculate BBox for the ellipse
    bbox = get_ellipse_bbox((center_x, center_y), (blob_w, blob_h), angle)
    
    # --- NEW STEP: Add realism noise before AI sees it ---
    img_gritty = add_noise_and_blur(img)
    
    # Return both image and bbox
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB)), bbox

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading Stable Diffusion XL Img2Img...")
    # UPDATED: Using SDXL Base 1.0
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    # Disable safety checker mostly handled by bypassing or setting to None if attribute exists
    # Standard SDXL pipe usually doesn't have the same safety_checker attribute exposed by default like v1.5
    # but we proceed with standard loading.

    print(f"✅ Model Loaded. Applying photorealistic texture (Strength: {AI_STRENGTH})...")
    print(f"generating {NUM_IMAGES} images.")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
        # Get image and bbox
        base_image, bbox = create_base_layout(512, 512)
        
        with torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=40,
                guidance_scale=8.0
            ).images[0]

        filename_base = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}"
        img_filename = f"{filename_base}.png"
        txt_filename = f"{filename_base}.txt"
        
        save_path_img = os.path.join(OUTPUT_DIR, img_filename)
        save_path_txt = os.path.join(OUTPUT_DIR, txt_filename)
        
        final_image.save(save_path_img)
        
        # Save Label
        width, height = final_image.size
        save_yolo_label(save_path_txt, bbox, width, height, YOLO_LABEL_INDEX)
        
        print(f"Saved Image: {save_path_img}")
        print(f"Saved Label: {save_path_txt}")
        
    print(f"Done")

if __name__ == "__main__":
    main()