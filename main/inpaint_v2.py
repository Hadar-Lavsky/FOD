import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionGLIGENPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels" # Use the labels folder where we randomized the classes (0,1,2,3)
OUTPUT_DIR = "./dataset/MFS/processed/"

# MAPPING CLASS ID -> GLIGEN PHRASE
# These phrases tell the model EXACTLY what to put in the box.
CLASS_PHRASES = {
    0: "a dark shiny oil spill",
    1: "a puddle of water",
    2: "white ice patch",
    3: "a cracked asphalt pothole"
}

# MAIN PROMPT (The "Vibe" of the whole image)
PROMPT = "top down aerial view of an airport runway, realistic asphalt texture, high quality, 4k, cctv footage style"
NEGATIVE_PROMPT = "cartoon, drawing, anime, low quality, distortion, blur, sand, grain, noise, 3d objects, buildings"

# SETTINGS
NUM_INFERENCE_STEPS = 50 # GLIGEN needs a bit more steps for detail
GUIDANCE_SCALE = 7.5

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_yolo_boxes(label_path, width, height):
    """
    Reads YOLO file and converts to [xmin, ymin, xmax, ymax] normalized 0-1 format.
    Returns: (list of boxes, list of phrases)
    """
    gligen_boxes = []
    gligen_phrases = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    
                    # Skip if class not in our mapping
                    if class_id not in CLASS_PHRASES:
                        continue

                    # YOLO format: class x_center y_center w h (normalized)
                    x_c, y_c, w_box, h_box = map(float, parts[1:])

                    # Convert to [xmin, ymin, xmax, ymax] for GLIGEN
                    xmin = x_c - (w_box / 2)
                    ymin = y_c - (h_box / 2)
                    xmax = x_c + (w_box / 2)
                    ymax = y_c + (h_box / 2)

                    # Clip to 0-1 to be safe
                    gligen_boxes.append([
                        max(0.0, xmin), 
                        max(0.0, ymin), 
                        min(1.0, xmax), 
                        min(1.0, ymax)
                    ])
                    
                    gligen_phrases.append(CLASS_PHRASES[class_id])
    
    return gligen_boxes, gligen_phrases

def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Load GLIGEN Model (Same as your friend's logic)
    print("⏳ Loading GLIGEN Model...")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-inpainting-text-box",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    print("✅ Model Loaded.")

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images. Starting processing...")

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")

        # Skip if no label file found
        if not os.path.exists(label_path):
            continue

        # Load Image
        original_image = Image.open(img_path).convert("RGB")
        w, h = original_image.size

        # Load Boxes & Phrases
        boxes, phrases = load_yolo_boxes(label_path, w, h)

        if not boxes:
            print(f"Skipping {img_file}: No valid boxes found in label file.")
            continue

        print(f"Generating {len(boxes)} objects on {img_file}...")

        # GLIGEN Generation
        # Note: We pass the original image as 'gligen_inpaint_image'
        # This tells the model: "Keep this image, but edit these boxes."
        with torch.autocast("cuda"):
            output = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                gligen_phrases=phrases,
                gligen_boxes=boxes,
                gligen_inpaint_image=original_image,
                gligen_scheduled_sampling_beta=1.0, # Controls how strictly it follows boxes (1.0 = strict)
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

        # Save Result
        save_path = os.path.join(OUTPUT_DIR, img_file)
        output.save(save_path)
        print(f"Saved: {save_path}")

    print("🎉 Done processing all images.")

if __name__ == "__main__":
    main()