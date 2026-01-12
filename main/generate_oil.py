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
OUTPUT_FILENAME_PREFIX = 'sinkhole_damage'
NUM_IMAGES = 1

# PROMPT: Focus on depth, collapse, and broken edges
PROMPT = (
    "photorealistic top-down view of asphalt airport runway, "
    "small sinkhole in the middle of the road, collapsed pavement, "
    "cracked asphalt edges, deep dark hole, bottomless pit, "
    "structural failure, rubble, 8k, sharp focus, daylight"
)

# NEGATIVE: Ban water, reflections, and flat paint
NEGATIVE_PROMPT = (
    "water, puddle, reflection, oil, wet ground, shiny, "
    "painting, drawing, cartoon, geometric, square, manhole cover, "
    "smooth edges, blur, low resolution"
)

# STRENGTH: 0.60 - Higher strength needed to render the "depth" inside the black blob
AI_STRENGTH = 0.60

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def draw_cracked_hole(img, w, h):
    """
    Draws a jagged hole with stress fractures radiating outward.
    """
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    
    # 1. Draw Stress Cracks (Thin dark lines radiating out)
    # We draw these BEFORE the hole so the hole covers their origin
    num_cracks = random.randint(5, 12)
    for _ in range(num_cracks):
        angle_deg = random.randint(0, 360)
        length = random.randint(40, 100)
        
        # Calculate end point
        end_x = int(center_x + length * np.cos(np.deg2rad(angle_deg)))
        end_y = int(center_y + length * np.sin(np.deg2rad(angle_deg)))
        
        # Draw irregular jagged line (crack)
        # We simulate a jagged line by drawing 2 segments
        mid_x = (center_x + end_x) // 2 + random.randint(-5, 5)
        mid_y = (center_y + end_y) // 2 + random.randint(-5, 5)
        
        cv2.line(img, (center_x, center_y), (mid_x, mid_y), (40, 40, 40), 2)
        cv2.line(img, (mid_x, mid_y), (end_x, end_y), (40, 40, 40), 1)

    # 2. Create the Sinkhole Shape (Jagged Polygon)
    radius = random.randint(30, 60)
    num_points = random.randint(8, 14)
    points = []
    
    for i in range(num_points):
        # High variance in radius creates the "collapsed" look
        r_var = radius + random.randint(-15, 15)
        angle = (2 * np.pi / num_points) * i
        x = int(center_x + r_var * np.cos(angle))
        y = int(center_y + r_var * np.sin(angle))
        points.append([x, y])
    
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    
    # 3. Draw the Hole (Pitch Black)
    # (0,0,0) is best here because it represents "no light/depth"
    cv2.fillPoly(img, [pts], (0, 0, 0)) 
    
    # 4. Optional: Draw a slight "inner rim" highlight to suggest thickness
    # This helps the AI see it as a 3D hole, not a 2D spot
    cv2.polylines(img, [pts], True, (90, 90, 90), 1)

def create_base_layout(w=512, h=512):
    # 1. Base Asphalt (Grey with noise)
    base_color = random.randint(100, 120)
    img = np.full((h, w, 3), (base_color, base_color, base_color), dtype=np.uint8)
    noise = np.random.randint(0, 40, (h, w, 3), dtype='uint8')
    img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
    
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

    # 3. Draw Sinkhole
    draw_cracked_hole(img, w, h)
    
    # 4. Blur slightly to blend lines
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
        
    print(f"✅ Model Loaded. Generating {NUM_IMAGES} SINKHOLE images...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
        base_image = create_base_layout(512, 512)
        
        # base_image.save(f"debug_sinkhole_{i}.png") # Debug if needed

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
        
    print(f"Done")

if __name__ == "__main__":
    main()