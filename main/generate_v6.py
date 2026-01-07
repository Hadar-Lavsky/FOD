import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

OUTPUT_DIR = "./dataset/MFS/processed/"
OUTPUT_FILENAME = "runway_oil_final.png"

# The prompt just asks for realism, not scene composition
PROMPT = "satellite view photograph of asphalt airport runway surface, realistic tarmac texture, black oil puddle, detailed concrete, daylight"
NEGATIVE_PROMPT = "blur, cartoon, drawing, 3d render, low res, objects, vehicles"

# Strength 0.4 means: "Keep 60% of my drawing structure, add 40% AI texture"
# Do not go higher than 0.5 or it will start distorting shapes.
AI_STRENGTH = 0.4

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def create_base_layout(w=512, h=512):
    """
    Procedurally draws a perfect top-down runway scene using OpenCV.
    """
    # 1. Fill background with base asphalt color (dark grey)
    # OpenCV uses BGR, so (80, 80, 80) is grey
    img = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
    
    # 2. Draw Runway Markings (White)
    # Threshold bars at the bottom
    bar_width = w // 15
    bar_height = h // 10
    spacing = w // 20
    start_x = spacing
    for i in range(4): # Draw 8 bars total (4 pairs)
        # Left side
        cv2.rectangle(img, (start_x, h - bar_height), (start_x + bar_width, h), (220, 220, 220), -1)
        # Right side
        end_x = w - start_x - bar_width
        cv2.rectangle(img, (end_x, h - bar_height), (end_x + bar_width, h), (220, 220, 220), -1)
        start_x += bar_width + spacing

    # Centerline dashes
    dash_h = h // 8
    dash_w = w // 40
    curr_y = 0
    while curr_y < h - bar_height - spacing:
        cv2.rectangle(img, (w//2 - dash_w//2, curr_y), (w//2 + dash_w//2, curr_y + dash_h), (220, 220, 220), -1)
        curr_y += dash_h * 2

    # 3. Draw the Oil Blob (Pure Black)
    # Random size and position near center
    blob_w = random.randint(w//8, w//4)
    blob_h = random.randint(h//8, h//4)
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    angle = random.randint(0, 180)
    
    # Draw solid black ellipse
    cv2.ellipse(img, (center_x, center_y), (blob_w, blob_h), angle, 0, 360, (10, 10, 10), -1)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def main():
    ensure_dir(OUTPUT_DIR)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # 1. Generate the perfect base layout
    print("🔨 Procedurally creating base layout...")
    base_image = create_base_layout(512, 512)
    # Uncomment the next line if you want to see the "ugly" base drawing before AI
    # base_image.save("debug_base.png") 

    # 2. Load Img2Img Pipeline (Not text-to-image)
    print("⏳ Loading Stable Diffusion Img2Img...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
        
    print("✅ Model Loaded. Applying photorealistic texture...")

    # 3. Run AI Polish
    with torch.autocast("cuda"):
        final_image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_image, # We feed it our drawing
            strength=AI_STRENGTH, # Low strength keeps shapes intact
            num_inference_steps=40,
            guidance_scale=7.5
        ).images[0]

    # 4. Save
    final_image.save(save_path)
    print(f"🎉 Final image saved to: {save_path}")

if __name__ == "__main__":
    main()