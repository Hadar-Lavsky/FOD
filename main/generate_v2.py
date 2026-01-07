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
OUTPUT_FILENAME = "runway_oil_puddle_aerial.png"

# --- NEW, SIMPLER PROMPT ---
# Focuses on a realistic aerial photo style.
PROMPT = (
    "Aerial photograph of an airport runway with a dark oil puddle on the asphalt surface. "
    "Realistic textures, high detail, daylight, sharp focus."
)

# Negative prompt (keeps it looking like a photo)
NEGATIVE_PROMPT = (
    "low quality, blurry, cartoon, 3d render, drawing, vehicles, people, "
    "watermark, text, distorted"
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
    print(f"Prompt: {PROMPT}")

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