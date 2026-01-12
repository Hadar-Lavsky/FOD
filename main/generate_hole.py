import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

OUTPUT_DIR = "./main/results/"
OUTPUT_FILENAME_PREFIX = 'shallow_sinkhole'
NUM_IMAGES = 1

# PROMPT: Updated to describe a shallow, shadowed depression with asphalt texture inside
PROMPT = (
    "photorealistic top-down view of asphalt airport runway, "
    "shallow sunken hole, collapsed pavement, "
    "cracked asphalt edges, inside of the hole is the same asphalt color but shadowed, "
    "structural failure, rubble, 8k, sharp focus, daylight, depth perception"
)

# NEGATIVE: Ban deep black holes and smooth surfaces
NEGATIVE_PROMPT = "smooth, digital painting, cartoon, drawing, 3d render, blur, low res, clean, deep black hole, bottomless pit"

# STRENGTH: Lowered slightly to preserve the grey color and texture of the depression from the base image
AI_STRENGTH = 0.55

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def draw_cracked_hole(img, w, h):
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    
    # 1. Draw Stress Cracks - MADE THICKER
    # Thicker lines (thickness=2 or 3) ensure SD sees them as geometry, not dirt.
    num_cracks = random.randint(6, 14)
    for _ in range(num_cracks):
        angle_deg = random.randint(0, 360)
        length = random.randint(50, 110)
        
        end_x = int(center_x + length * np.cos(np.deg2rad(angle_deg)))
        end_y = int(center_y + length * np.sin(np.deg2rad(angle_deg)))
        
        mid_x = (center_x + end_x) // 2 + random.randint(-10, 10)
        mid_y = (center_y + end_y) // 2 + random.randint(-10, 10)
        
        # Dark cracks
        cv2.line(img, (center_x, center_y), (mid_x, mid_y), (30, 30, 30), 3)
        cv2.line(img, (mid_x, mid_y), (end_x, end_y), (30, 30, 30), 2)

    # 2. Sinkhole Shape
    radius = random.randint(35, 65)
    num_points = random.randint(8, 14)
    points = []
    
    for i in range(num_points):
        r_var = radius + random.randint(-15, 15)
        angle = (2 * np.pi / num_points) * i
        x = int(center_x + r_var * np.cos(angle))
        y = int(center_y + r_var * np.sin(angle))
        points.append([x, y])
    
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    
    # 3. Fill Hole - DARKER and TEXTURED
    # Darker Grey (50) ensures high contrast against the road (110)
    hole_color = (50, 50, 50) 
    cv2.fillPoly(img, [pts], hole_color) 

    # 4. ADD "RUBBLE" NOISE INSIDE THE HOLE
    # This creates little light and dark spots inside the hole area only.
    # It stops the AI from treating it as a flat shadow.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    
    # Generate rubble noise
    rubble_noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    img_int = img.astype(np.int16)
    
    # Apply noise only where the mask is white
    mask_bool = mask > 0
    img_int[mask_bool] = img_int[mask_bool] + rubble_noise[mask_bool]
    
    # Clip back to valid range
    img[:] = np.clip(img_int, 0, 255).astype(np.uint8)
    
    # 5. Rim Highlight
    cv2.polylines(img, [pts], True, (160, 160, 160), 2)

def create_base_layout(w=512, h=512):
    # 1. Base Asphalt
    base_color = random.randint(100, 120)
    img = np.full((h, w, 3), (base_color, base_color, base_color), dtype=np.uint8)
    
    # 2. Markings
    mark_color = (220, 220, 220)
    bar_width, bar_height = w // 15, h // 10
    spacing = w // 20
    
    start_x = spacing
    for i in range(4):
        cv2.rectangle(img, (start_x, h - bar_height), (start_x + bar_width, h), mark_color, -1)
        end_x = w - start_x - bar_width
        cv2.rectangle(img, (end_x, h - bar_height), (end_x + bar_width, h), mark_color, -1)
        start_x += bar_width + spacing

    dash_h, dash_w = h // 8, w // 40
    curr_y = 0
    while curr_y < h - bar_height - spacing:
        cv2.rectangle(img, (w//2 - dash_w//2, curr_y), (w//2 + dash_w//2, curr_y + dash_h), mark_color, -1)
        curr_y += dash_h * 2

    # 3. Draw Sinkhole (With new Rubble logic)
    draw_cracked_hole(img, w, h)
    
    # 4. Global Noise (lighter than before so we don't bury the hole)
    noise = np.random.randint(0, 30, (h, w, 3), dtype='uint8')
    img = cv2.addWeighted(img, 0.85, noise, 0.15, 0)

    # 5. Blur
    img_gritty = cv2.GaussianBlur(img, (3, 3), 0)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB))

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading Stable Diffusion...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
        
    print(f"✅ Model Loaded. Generating {NUM_IMAGES} FIXED SINKHOLE images...")

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
        
    print(f"Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()