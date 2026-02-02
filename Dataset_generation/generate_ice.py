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

OUTPUT_DIR = "./dataset/raw/ice"
OUTPUT_FILENAME_PREFIX = 'ice'
NUM_IMAGES = 200

YOLO_LABEL_INDEX = 1  

# ICE PROMPT
PROMPT = (
    "raw photograph, satellite view of asphalt airport runway surface, "
    "large solid chunk of ice sitting on the ground, frozen block of ice, "
    "translucent white and blue ice, frost, smashed ice debris, "
    "winter, cold, slippery texture, sharp focus, daylight"
)

NEGATIVE_PROMPT = (
    "hole, pit, dirt, brown, fire, hot, liquid water, melted, "
    "smooth, digital painting, cartoon, drawing, 3d render, blur, low res"
)

AI_STRENGTH = 0.60

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
#        BOUNDING BOX FUNCTIONS
# ==========================================

def get_bounding_box(pts):
    """ Calculates the axis-aligned bounding box from a list of polygon points. """
    x_coords = pts[:, 0, 0]
    y_coords = pts[:, 0, 1]
    
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    
    return (min_x, min_y, max_x, max_y)

def save_yolo_label(path, bbox, img_w, img_h, class_id):
    """ Saves a .txt file with YOLO format: class_id center_x center_y width height """
    min_x, min_y, max_x, max_y = bbox
    
    w = max_x - min_x
    h = max_y - min_y
    center_x = min_x + (w / 2)
    center_y = min_y + (h / 2)
    
    norm_cx = min(max(center_x / img_w, 0), 1)
    norm_cy = min(max(center_y / img_h, 0), 1)
    norm_w = min(max(w / img_w, 0), 1)
    norm_h = min(max(h / img_h, 0), 1)
    
    line = f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
    
    with open(path, 'w') as f:
        f.write(line)

# ==========================================

def draw_ice_chunk(img, center_x, center_y, base_radius):
    """ 
    Draws an irregular, jagged polygon filled with white/light blue 
    to simulate a chunk of ice.
    Returns the bounding box.
    """
    num_points = 10  # Fewer points than a hole makes it look more "blocky"
    points = []

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        # High variance (0.6 to 1.5) for jagged, blocky shape
        r = base_radius * random.uniform(0.6, 1.5)
        
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        points.append([x, y])

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Color: Very light blue/white for Ice (BGR format)
    # B: ~255, G: ~250, R: ~240
    ice_color = (255, 250, 240) 

    # Fill the polygon
    cv2.fillPoly(img, [pts], ice_color)
    
    # Add a slightly darker outline to help SD find the edge
    cv2.polylines(img, [pts], True, (200, 200, 220), 2)
    
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
    
    start_x = spacing
    for i in range(4):
        cv2.rectangle(img, (start_x, h - bar_height), (start_x + bar_width, h), mark_color, -1)
        end_x = w - start_x - bar_width
        cv2.rectangle(img, (end_x, h - bar_height), (end_x + bar_width, h), mark_color, -1)
        start_x += bar_width + spacing

    dash_h = h // 8
    dash_w = w // 40
    curr_y = 0
    while curr_y < h - bar_height - spacing:
        cv2.rectangle(img, (w//2 - dash_w//2, curr_y), (w//2 + dash_w//2, curr_y + dash_h), mark_color, -1)
        curr_y += dash_h * 2

    # 3. Draw The Ice Chunk
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    radius = random.randint(w//14, w//9) # Slightly smaller base than the hole
    
    bbox = draw_ice_chunk(img, center_x, center_y, radius)
    
    # 4. Add realism noise
    img_gritty = add_noise_and_blur(img)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB)), bbox

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("⏳ Loading Stable Diffusion XL Img2Img...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    print(f"✅ Model Loaded. Generating Ice Chunks (Strength: {AI_STRENGTH})...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
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

        # Save Image & Label
        filename_base = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}"
        img_filename = f"{filename_base}.png"
        txt_filename = f"{filename_base}.txt"
        
        save_path_img = os.path.join(OUTPUT_DIR, img_filename)
        save_path_txt = os.path.join(OUTPUT_DIR, txt_filename)
        
        final_image.save(save_path_img)
        
        width, height = final_image.size
        save_yolo_label(save_path_txt, bbox, width, height, YOLO_LABEL_INDEX)
        
        print(f"Saved: {save_path_img}")
    
    print("Done")

if __name__ == "__main__":
    main()