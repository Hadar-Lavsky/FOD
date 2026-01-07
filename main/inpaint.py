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
DEBUG_MASKS_DIR = "./dataset/MFS/debug/"

# MODEL SETTINGS
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" 
SEED = 42 

# === PROMPTS FOR AERIAL/SIMULATOR VIEW ===
COMMON_STYLE = "top down aerial view, drone footage, flat surface, looking down, 8k resolution, highly detailed"

PROMPTS = {
    'oil': f"A dark black oil spill on concrete pavement, {COMMON_STYLE}, shiny liquid texture, irregular shape, industrial waste, contrast against grey asphalt",
    
    'water': f"A flat puddle of water on concrete pavement, {COMMON_STYLE}, wet surface, dark reflection of grey sky, rain accumulation, no horizon",
    
    'ice': f"A patch of white frost and ice on asphalt, {COMMON_STYLE}, frozen road surface, slippery texture, winter conditions, white coating on grey ground",
    
    'hole': f"A distressed pothole in the asphalt, {COMMON_STYLE}, cracked concrete, broken pavement, hole in the ground, dark depth, damage texture"
}

# === NEGATIVE PROMPT ===
NEGATIVE_PROMPT = (
    "tilted, perspective, horizon, sky, trees, buildings, grass, car, people, "
    "3d extrusion, side view, isometric, low quality, blur, watermark, text"
)

# SETTINGS
NUM_INFERENCE_STEPS = 40  
GUIDANCE_SCALE = 8.5      

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_mask_organic_filled(mask_path, debug_save_path=None):
    """
    Creates an organic mask that GUARANTEES filling most of the box.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or cv2.countNonZero(mask) == 0:
        return None

    organic_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5: continue

        # Define key points along the perimeter
        points = [
            (x, y), (x + w//2, y), (x + w, y),           
            (x + w, y + h//2), (x + w, y + h),           
            (x + w//2, y + h), (x, y + h),               
            (x, y + h//2)                                
        ]

        jittered_points = []
        # Jitter amount: 15% of the dimension
        jitter_x = max(2, int(w * 0.15))
        jitter_y = max(2, int(h * 0.15))

        for px, py in points:
            nx = px + random.randint(-jitter_x, jitter_x)
            ny = py + random.randint(-jitter_y, jitter_y)
            
            # Clamp to image boundaries
            nx = max(0, min(organic_mask.shape[1]-1, nx))
            ny = max(0, min(organic_mask.shape[0]-1, ny))
            
            jittered_points.append([nx, ny])

        pts_array = np.array(jittered_points, dtype=np.int32)
        cv2.fillPoly(organic_mask, [pts_array], 255)

    organic_mask = cv2.GaussianBlur(organic_mask, (21, 21), 0)
    _, organic_mask = cv2.threshold(organic_mask, 127, 255, cv2.THRESH_BINARY)
    
    if debug_save_path:
        cv2.imwrite(debug_save_path, organic_mask)

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
    ensure_dir(DEBUG_MASKS_DIR)
    
    pipe = load_pipeline()
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} base images. Starting generation...")

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        
        # Load image
        original_image = Image.open(img_path).convert("RGB")
        
        for suffix, prompt_text in PROMPTS.items():
            # Matches format from previous scripts: "imageName_mask_oil.jpg"
            mask_filename = f"{base_name}_mask_{suffix}.jpg"
            mask_path = os.path.join(INPUT_MASKS_DIR, mask_filename)

            if not os.path.exists(mask_path):
                continue

            # Debug filename
            debug_name = f"{base_name}_debugmask_{suffix}.jpg"
            debug_path = os.path.join(DEBUG_MASKS_DIR, debug_name)

            # Generate organic mask
            organic_mask_image = make_mask_organic_filled(mask_path, debug_save_path=debug_path)
            
            if organic_mask_image is None:
                continue

            print(f"Generating {suffix} for {base_name}...")

            output = pipe(
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
                image=original_image,
                mask_image=organic_mask_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                strength=1.0 
            ).images[0]

            output_filename = f"{base_name}_gen_{suffix}.jpg"
            save_path = os.path.join(OUTPUT_DIR, output_filename)
            output.save(save_path)
            
    print(f"Batch generation complete! Processed images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()