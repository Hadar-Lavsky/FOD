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
OUTPUT_FILENAME = "runway_oil_puddle_high_aerial.png"

# --- NEW, FORCEFUL AERIAL PROMPT ---
# Using strong keywords to push the camera perspective high up.
PROMPT = (
    "A high-altitude, top-down drone photograph of an airport runway and surrounding taxiways."
    "A dark, iridescent oil puddle is visible on the grey asphalt runway surface. "
    "The runway appear small below. Wide angle view, daylight, sharp focus, realistic satellite imagery style."
)

# Negative prompt to prevent ground-level elements
NEGATIVE_PROMPT = (
    "eye-level view, ground view, close up, low angle, blurry, cartoon, 3d render, "
    "drawing, vehicles, people, buildings, large objects"
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
            guidance_scale=8.0, # Increased slightly to force the perspective
            height=512,
            width=512
        ).images[0]

    # 4. Save
    image.save(save_path)
    print(f"🎉 Image saved successfully to: {save_path}")

if __name__ == "__main__":
    main()