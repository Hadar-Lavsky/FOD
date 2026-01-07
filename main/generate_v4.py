import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ==========================================
#              CONFIGURATION
# ==========================================

# Directory to save the image
OUTPUT_DIR = "./dataset/MFS/processed/"

# Name of the file
OUTPUT_FILENAME = "runway_oil_puddle_flat_topdown.png"

# --- NEW, RIGID PROMPT ---
# Using technical terms to force a flat, map-like perspective and high contrast.
PROMPT = (
    "Satellite photograph, straight down 90-degree nadir view of a section of light gray asphalt airport runway. "
    "A distinct, dark black, glossy oil spill stain is prominent on the gray pavement surface. "
    "High contrast between the black liquid and the gray road. "
    "Flat lighting, no shadows, map view, orthographic, highly detailed texture."
)

# Negative prompt to absolutely forbid angled views
NEGATIVE_PROMPT = (
    "angled view, horizon, sky, clouds, perspective, depth, tilt, isometric, "
    "3d render, cartoon, drawing, low resolution, blurry, colorful, iridescent, vehicles"
)

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 Created directory: {directory}")

def main():
    # 1. Setup Output Directory
    ensure_dir(OUTPUT_DIR)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("⏳ Loading Stable Diffusion Model...")
    
    # 2. Load Model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
    
    print("✅ Model Loaded. Generating image...")
    print(f"Prompt: {PROMPT}")

    # 3. Generate
    with torch.autocast("cuda"):
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,
            # Increased guidance scale to force it to listen to the strict prompt
            guidance_scale=9.0, 
            height=512,
            width=512
        ).images[0]

    # 4. Save
    image.save(save_path)
    print(f"🎉 Image saved successfully to: {save_path}")

if __name__ == "__main__":
    main()