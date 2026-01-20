import os
import cv2
import torch
import random
import numpy as np
import math
from PIL import Image, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

OUTPUT_DIR = "./dataset/raw/hole"
OUTPUT_FILENAME_PREFIX = 'hole' # Changed filename
NUM_IMAGES = 200

# HOLE/DIRT PROMPT
# We emphasize "exposed earth", "cracked edges", and "jagged"
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

def draw_jagged_hole(img, center_x, center_y, base_radius):
    """ 
    Draws an irregular, jagged polygon filled with brown to simulate 
    a broken chunk of asphalt exposing dirt.
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

    # Color: Brown/Dirt in BGR format (OpenCV uses BGR, not RGB)
    # Blue: ~30, Green: ~60, Red: ~90 => Dark Dirt Brown
    dirt_color = (35, 65, 95) 

    # Fill the polygon
    cv2.fillPoly(img, [pts], dirt_color)
    
    # Optional: Add a thin dark outline to simulate the shadow of the edge
    cv2.polylines(img, [pts], True, (10, 10, 10), 2)

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

    # 3. Draw The Hole with Dirt (Instead of Ellipse)
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    radius = random.randint(w//12, w//8)
    
    draw_jagged_hole(img, center_x, center_y, radius)
    
    # 4. Add realism noise
    img_gritty = add_noise_and_blur(img)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB))

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading Stable Diffusion Img2Img...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")

    # Disable safety checker to prevent false positives on "dirt/messy" textures
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
        
    print(f"✅ Model Loaded. Generating Potholes with Dirt (Strength: {AI_STRENGTH})...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        base_image = create_base_layout(512, 512)
        
        with torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=50,
                guidance_scale=8.0
            ).images[0]

        filename = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        final_image.save(save_path)
        print(f"Saved: {save_path}")
    
    print("Done")

if __name__ == "__main__":
    main()