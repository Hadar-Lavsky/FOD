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
OUTPUT_FILENAME = "runway_oil_blob_topdown.png"

# --- NEW, SPECIFIC PROMPT ---
# - "Single, distinct... blob" -> Ensures one clear object for a bounding box.
# - "Wide concrete runway," "threshold stripes," "taxiway" -> Makes it clearly an airport.
PROMPT = (
    "Satellite photograph, straight down 90-degree nadir view of a wide grey concrete airport runway. "
    "A single, distinct, large, rounded black oil puddle blob is centered on the runway surface. "
    "The runway has prominent white threshold markings and a painted centerline. "
    "A dirt and grass shoulder is visible on the sides. "
    "Flat lighting, no shadows, map view, highly detailed texture."
)

# Negative prompt to ban city road features
NEGATIVE_PROMPT = (
    "angled view, horizon, sky, perspective, depth, tilt, isometric, "
    "3d render, cartoon, low resolution, blurry, "
    "city street, cars, sidewalks, intersections, road signs, buildings, trees"
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
            # High guidance to force all the specific details
            guidance_scale=9.5, 
            height=512,
            width=512
        ).images[0]

    # 4. Save
    image.save(save_path)
    print(f"🎉 Image saved successfully to: {save_path}")

if __name__ == "__main__":
    main()