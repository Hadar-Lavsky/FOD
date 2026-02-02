import os
import cv2
import torch
import random
import numpy as np
import math
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLImg2ImgPipeline

# ==========================================
#               CONFIGURATION
# ==========================================

OUTPUT_DIR = "./dataset/raw/hole"
OUTPUT_FILENAME_PREFIX = 'hole'
NUM_IMAGES = 200

YOLO_LABEL_INDEX = 2  # The class number for the hole

# HOLE PROMPT
PROMPT = (
    "raw photograph, satellite view of asphalt airport runway surface, "
    "large jagged pothole, missing chunk of asphalt, exposed brown dirt inside hole, "
    "rubble, cracked pavement edges, heavy grain texture, weathered concrete, daylight"
)

NEGATIVE_PROMPT = "smooth, digital painting, cartoon, drawing, 3d render, blur, low res, clean, water, liquid, reflection"

# Keep strength high to allow AI to texture the dirt properly
AI_STRENGTH = 0.55

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def add_noise_and_blur(img_cv):
    """ Adds grit to the perfect OpenCV drawing so AI doesn't make it cartoonish. """
    img_blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
    noise = np.random.randint(0, 50, img_blurred.shape, dtype='uint8')
    img_noisy = cv2.addWeighted(img_blurred, 0.8, noise, 0.2, 0)
    return img_noisy

# ==========================================
#        NEW BOUNDING BOX FUNCTIONS
# ==========================================

def get_bounding_box(pts):
    """
    Calculates the axis-aligned bounding box from a list of polygon points.
    Returns: (min_x, min_y, max_x, max_y)
    """
    # pts shape is (N, 1, 2)
    x_coords = pts[:, 0, 0]
    y_coords = pts[:, 0, 1]
    
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    
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

def draw_jagged_hole(img, center_x, center_y, base_radius):
    """ 
    Draws an irregular, jagged polygon filled with brown to simulate 
    a broken chunk of asphalt exposing dirt.
    Returns the bounding box of the drawn hole.
    """
    num_points = 12 # Number of vertices in the hole
    points = []

    for i in range(num_points):
        # Calculate angle
        angle = (2 * math.pi * i) / num_points
        # Randomize radius to make it jagged (0.7x to 1.3x variance)
        r = base_radius * random.uniform(0.7, 1.4)
        
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        points.append([x, y])

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Color: Brown/Dirt in BGR format
    dirt_color = (35, 65, 95) 

    # Fill the polygon
    cv2.fillPoly(img, [pts], dirt_color)
    
    # Optional: Add a thin dark outline to simulate the shadow of the edge
    cv2.polylines(img, [pts], True, (10, 10, 10), 2)
    
    # NEW: Calculate and return bbox using the new function
    return get_bounding_box(pts)

def create_base_layout(w=512, h=512):
    # 1. Base Asphalt (Grey)
    base_color = random.randint(70, 90)
    img = np.full((h, w, 3), (base_color, base_color, base_color), dtype=np.uint8)
    
    # 2. Draw Runway Markings
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

    # 3. Draw The Hole with Dirt
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    radius = random.randint(w//12, w//8)
    
    # Capture bbox here
    bbox = draw_jagged_hole(img, center_x, center_y, radius)
    
    # 4. Add realism noise
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

    # Disable safety checker mechanism if it exists (standard SDXL pipelines usually don't have the same checker attribute, 
    # but strictly speaking we just leave it standard or None if supported. 
    # SDXL normally doesn't block 'dirt' textures as aggressively).
        
    print(f"✅ Model Loaded. Generating Potholes with Dirt (Strength: {AI_STRENGTH})...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
        # Get image and the approximate bounding box
        base_image, bbox = create_base_layout(512, 512)
        
        with torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=40, # SDXL works well with 30-50 steps
                guidance_scale=8.0
            ).images[0]

        # Save Image
        filename_base = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}"
        img_filename = f"{filename_base}.png"
        txt_filename = f"{filename_base}.txt"
        
        save_path_img = os.path.join(OUTPUT_DIR, img_filename)
        save_path_txt = os.path.join(OUTPUT_DIR, txt_filename)
        
        final_image.save(save_path_img)
        
        # Save Label using the new function
        width, height = final_image.size
        save_yolo_label(save_path_txt, bbox, width, height, YOLO_LABEL_INDEX)
        
        print(f"Saved Image: {save_path_img}")
        print(f"Saved Label: {save_path_txt}")
    
    print("Done")

if __name__ == "__main__":
    main()