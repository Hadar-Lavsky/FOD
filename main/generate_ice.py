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
OUTPUT_FILENAME_PREFIX = 'water_puddle_realistic'
NUM_IMAGES = 1

# PROMPT: We focus on "wet asphalt stain" and "reflection"
PROMPT = (
    "photorealistic top-down view of airport runway asphalt, "
    "a single rain water puddle on the ground, "
    "mirror reflection of clouds in the puddle, "
    "wet dark concrete, high texture, 8k, sharp focus, daylight"
)

# NEGATIVE: Explicitly ban symbols and solid shapes
NEGATIVE_PROMPT = (
    "painting, drawing, cartoon, geometric, square, diamond shape, "
    "road sign, white markings inside puddle, "
    "hole, black pit, oil, tar, 3d render"
)

# STRENGTH: 0.55 allows SD to texture the "paint" into "water"
AI_STRENGTH = 0.55

# ==========================================

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def generate_organic_puddle_mask(w, h):
    """
    Creates a natural, jagged, organic liquid shape.
    It does this by upscaling random noise and thresholding it.
    """
    # 1. Generate low-res noise
    scale = 16
    noise_small = np.random.randint(0, 255, (h // scale, w // scale), dtype=np.uint8)
    
    # 2. Upscale smoothly (Bilinear) -> Creates cloud-like blobs
    noise_big = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Blur heavily to smooth edges further
    noise_blur = cv2.GaussianBlur(noise_big, (45, 45), 0)
    
    # 4. Threshold to get a "Puddle" shape
    # We pick a random threshold to vary puddle size
    thresh_val = random.randint(160, 190)
    _, mask = cv2.threshold(noise_blur, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 5. Isolate ONE blob (We only want 1 puddle, not 50 tiny ones)
    # Find contours and keep only the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(mask)
    if contours:
        # Sort by area and keep largest
        largest_contour = max(contours, key=cv2.contourArea)
        # Only keep if it's a decent size (not a speck)
        if cv2.contourArea(largest_contour) > 500:
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
            
            # Move the puddle to a random spot on the image
            # (Calculate center, shift to new random center)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                shift_x = random.randint(w//4, 3*w//4) - cx
                shift_y = random.randint(h//4, 3*h//4) - cy
                
                M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                clean_mask = cv2.warpAffine(clean_mask, M_trans, (w, h))

    return clean_mask

def create_base_layout(w=512, h=512):
    # 1. Base Asphalt (Grey with heavy noise)
    # Noise is critical so the "wet" part has texture to darken
    base_color = random.randint(100, 120)
    img = np.full((h, w, 3), (base_color, base_color, base_color), dtype=np.uint8)
    noise = np.random.randint(0, 40, (h, w, 3), dtype='uint8')
    img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
    
    # 2. Markings (Draw first)
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

    # ==========================================
    # 3. APPLY THE PUDDLE (Stain Method)
    # ==========================================
    
    mask = generate_organic_puddle_mask(w, h)
    
    # If mask is empty (generation failed), just return (rare)
    if cv2.countNonZero(mask) == 0:
        return Image.fromarray(img)

    # A. DARKEN THE ASPHALT (Wetness)
    # Convert to float
    img_float = img.astype(np.float32)
    mask_factor = mask.astype(np.float32) / 255.0
    
    # Where mask is white, darken image by multiplying by 0.4
    # Where mask is black, keep original (multiply by 1.0)
    darkness = 1.0 - (mask_factor * 0.6) 
    
    # Apply to all channels
    img_float[:,:,0] *= darkness
    img_float[:,:,1] *= darkness
    img_float[:,:,2] *= darkness
    
    # B. ADD REFLECTIONS (Clouds)
    # This is the secret sauce. We add bright white blobs INSIDE the mask.
    # Without this, AI thinks it's a black hole.
    reflections = np.zeros((h, w), dtype=np.float32)
    
    # Draw random soft blobs for clouds
    for _ in range(5):
        rx = random.randint(0, w)
        ry = random.randint(0, h)
        r_rad = random.randint(20, 60)
        cv2.circle(reflections, (rx, ry), r_rad, 180, -1) # Intensity 180 (grey-white)
    
    # Blur reflections heavily
    reflections = cv2.GaussianBlur(reflections, (61, 61), 0)
    
    # Only show reflections INSIDE the puddle
    reflections_masked = reflections * mask_factor
    
    # Add reflections to the dark water (Blueish tint for sky)
    img_float[:,:,0] += reflections_masked * 1.1 # Blue
    img_float[:,:,1] += reflections_masked * 1.0 # Green
    img_float[:,:,2] += reflections_masked * 1.0 # Red

    # Clip and convert
    img_final = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))

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
        
    print(f"✅ Model Loaded. Generating {NUM_IMAGES} realistic puddles...")

    for i in range(NUM_IMAGES):
        print(f"[{i+1}/{NUM_IMAGES}] Generative Step...")
        
        base_image = create_base_layout(512, 512)
        
        # DEBUG: Save this to see the "stain" before AI touches it
        # base_image.save(f"debug_puddle_{i}.png") 

        with torch.autocast("cuda"):
            final_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=base_image,
                strength=AI_STRENGTH,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

        filename = f"{OUTPUT_FILENAME_PREFIX}_{i+1:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        final_image.save(save_path)
        
    print(f"Done")

if __name__ == "__main__":
    main()