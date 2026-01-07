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
OUTPUT_FILENAME = "runway_with_oil_puddle.png"

# The text prompt
PROMPT = (
    "top-down aerial view of a worn asphalt airport runway with a large, dark, "
    "iridescent oil puddle in the center. The runway has white lines and "
    "yellow markings. The surrounding ground is dry, cracked earth. "
    "Realistic textures, daylight, high resolution photo."
)

# Negative prompt (what to avoid)
NEGATIVE_PROMPT = (
    "low quality, blurry, cartoon, 3d render, people, aircraft, vehicles, "
    "buildings, distorted, watermark, text"
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
    
    # Double check safety disable
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
    
    print("✅ Model Loaded. Generating image...")

    # 3. Generate
    with torch.autocast("cuda"):
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

    # 4. Save
    image.save(save_path)
    print(f"🎉 Image saved successfully to: {save_path}")

if __name__ == "__main__":
    main()