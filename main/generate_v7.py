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

OUTPUT_DIR = "./dataset/MFS/processed/"
OUTPUT_FILENAME = "runway_oil_realism_v7.png"

# UPDATED PROMPT: Emphasizing raw, gritty reality
PROMPT = (
    "raw photograph, satellite view of asphalt airport runway surface, "
    "heavy grain texture, dirt, weathered concrete, black oil puddle, "
    "detailed ground texture, daylight"
)
# UPDATED NEGATIVE PROMPT: Banning smooth/digital looks
NEGATIVE_PROMPT = "smooth, digital painting, cartoon, drawing, 3d render, blur, low res, clean"

# INCREASED STRENGTH: Give AI more freedom to create grit (0.55 - 0.60 range)
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
    blob_w = random.randint(w//8, w//4)
    blob_h = random.randint(h//8, h//4)
    center_x = random.randint(w//3, 2*w//3)
    center_y = random.randint(h//3, 2*h//3)
    angle = random.randint(0, 180)
    cv2.ellipse(img, (center_x, center_y), (blob_w, blob_h), angle, 0, 360, (5, 5, 5), -1)
    
    # --- NEW STEP: Add realism noise before AI sees it ---
    img_gritty = add_noise_and_blur(img)
    
    return Image.fromarray(cv2.cvtColor(img_gritty, cv2.COLOR_BGR2RGB))

def main():
    ensure_dir(OUTPUT_DIR)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("🔨 Procedurally creating GRITTY base layout...")
    base_image = create_base_layout(512, 512)
    # Uncomment to see the "dirty" base image
    # base_image.save("debug_gritty_base.png") 

    print("⏳ Loading Stable Diffusion Img2Img...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
        
    print(f"✅ Model Loaded. Applying photorealistic texture (Strength: {AI_STRENGTH})...")

    with torch.autocast("cuda"):
        final_image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_image,
            strength=AI_STRENGTH,
            num_inference_steps=50,
            guidance_scale=8.0 # High guidance for strict texture adherence
        ).images[0]

    final_image.save(save_path)
    print(f"🎉 Final photorealistic image saved to: {save_path}")

if __name__ == "__main__":
    main()