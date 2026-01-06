import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting

# ==========================================
#              CONFIGURATION
# ==========================================

# DIRECTORIES
INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_MASKS_DIR = "./dataset/MFS/raw_images_masks/"
OUTPUT_DIR = "./dataset/MFS/processed/"

# MODEL SETTINGS
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" 
SEED = 42 

# CLASS PROMPTS
PROMPTS = {
    'oil': "A photorealistic dark oil spill on an airport runway, iridescent sheen, toxic liquid on asphalt, high contrast, 8k",
    'water': "A photorealistic puddle of water on an airport runway, wet asphalt, clear reflections of the sky, 8k",
    'ice': "A photorealistic patch of slippery ice on an airport runway, frozen white frost texture, dangerous conditions, 8k",
    'hole': "A photorealistic pothole in the asphalt runway, cracked pavement, jagged edges, deep hole, damage, 8k"
}

# NEGATIVE PROMPT
NEGATIVE_PROMPT = "square, geometric, straight lines, blur, cartoon, drawing, painting, illustration, low quality, distorted"

# GENERATION SETTINGS
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_mask_organic(mask_path):
    """
    Reads a rectangular mask and converts the rectangles into 
    organic, irregular blobs that fit inside the original box.
    """
    # 1. Read the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    # Check if empty
    if cv2.countNonZero(mask) == 0:
        return None

    # 2. Create a blank canvas for the new organic mask
    organic_mask = np.zeros_like(mask)

    # 3. Find the rectangular boxes in the original mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # skip tiny noise
        if w < 5 or h < 5: 
            continue

        # 4. Generate Random Points INSIDE this box
        # We generate random points to create a "Convex Hull" (a rock/puddle shape)
        # We leave a small margin (10%) so it doesn't touch the box edges perfectly
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        
        num_points = random.randint(8, 15) # More points = smoother, Fewer = jagged
        points = []
        
        for _ in range(num_points):
            px = random.randint(x + margin_w, x + w - margin_w)
            py = random.randint(y + margin_h, y + h - margin_h)
            points.append([px, py])
            
        points = np.array(points, dtype=np.int32)
        
        # 5. Create the Organic Shape (Convex Hull)
        hull = cv2.convexHull(points)
        
        # Draw the filled blob onto our new mask
        cv2.drawContours(organic_mask, [hull], -1, 255, thickness=-1)

    # Optional: Add a slight blur to soften edges (makes blending better)
    organic_mask = cv2.GaussianBlur(organic_mask, (15, 15), 0)
    
    # Threshold back to binary (soft edges become hard edges for the mask)
    _, organic_mask = cv2.threshold(organic_mask, 127, 255, cv2.THRESH_BINARY)
    
    return Image.fromarray(organic_mask).convert("RGB")

def load_pipeline():
    print(f"Loading model: {MODEL_ID}...")
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    pipe.to(device)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    return pipe

def main():
    ensure_dir(OUTPUT_DIR)
    
    pipe = load_pipeline()
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} base images. Starting generation...")

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        
        original_image = Image.open(img_path).convert("RGB")
        
        for suffix, prompt_text in PROMPTS.items():
            mask_filename = f"{base_name}_mask_{suffix}.jpg"
            mask_path = os.path.join(INPUT_MASKS_DIR, mask_filename)

            if not os.path.exists(mask_path):
                continue

            # === CHANGED HERE ===
            # Instead of loading the mask directly, we process it to be organic
            organic_mask_image = make_mask_organic(mask_path)
            
            if organic_mask_image is None:
                print(f"Skipping {base_name} [{suffix}]: Mask is empty.")
                continue

            print(f"Generating {suffix} for {base_name} (Organic Shape)...")

            output = pipe(
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
                image=original_image,
                mask_image=organic_mask_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                strength=0.99 
            ).images[0]

            output_filename = f"{base_name}_gen_{suffix}.jpg"
            save_path = os.path.join(OUTPUT_DIR, output_filename)
            output.save(save_path)
            
    print("Batch generation complete!")

if __name__ == "__main__":
    main()