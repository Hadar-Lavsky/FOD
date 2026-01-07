import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionGLIGENPipeline

# ==========================================
#              CONFIGURATION
# ==========================================

INPUT_IMAGES_DIR = "./dataset/MFS/raw_images/"
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels" # Use the labels folder where we randomized the classes (0,1,2,3)
OUTPUT_DIR = "./dataset/MFS/processed/"

CLASS_PROMPTS = {
    0: "close up photo of a dark shiny oil spill stain on asphalt, liquid texture, high detail",
    1: "close up photo of a water puddle on asphalt, reflection, liquid, high detail",
    2: "close up photo of an ice patch on asphalt, frozen texture, white frost",
    3: "close up photo of cracked asphalt, pothole, road damage, rough texture, concrete debris"
}

# MAPPING CLASS ID -> GLIGEN PHRASE
# These phrases tell the model EXACTLY what to put in the box.
CLASS_PHRASES = {
    0: "oil spill",
    1: "puddle",
    2: "ice",
    3: "pothole"
}

# MAIN PROMPT (The "Vibe" of the whole image)
NEGATIVE_PROMPT = "aerial view, horizon, sky, grass, buildings, cars, planes, lines, markings, cartoon, drawing, blur, low quality"

# SETTINGS
NUM_INFERENCE_STEPS = 50 # GLIGEN needs a bit more steps for detail
GUIDANCE_SCALE = 7.5

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_yolo_boxes(label_path, width, height):
    """
    Reads YOLO file. Returns list of dicts with box, phrase, and prompt.
    """
    defects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id not in CLASS_PROMPTS:
                        continue
                    
                    x_c, y_c, w_box, h_box = map(float, parts[1:])
                    xmin = x_c - (w_box / 2)
                    ymin = y_c - (h_box / 2)
                    xmax = x_c + (w_box / 2)
                    ymax = y_c + (h_box / 2)

                    defects.append({
                        "box": [max(0.0, xmin), max(0.0, ymin), min(1.0, xmax), min(1.0, ymax)],
                        "phrase": CLASS_PHRASES[class_id],
                        "prompt": CLASS_PROMPTS[class_id]
                    })
    return defects

def crop_and_process(pipe, original_image, defect_info):
    w, h = original_image.size
    box = defect_info["box"]
    
    # 1. Pixel Coordinates with Padding
    xmin, ymin, xmax, ymax = box
    px_xmin, px_ymin = int(xmin * w), int(ymin * h)
    px_xmax, px_ymax = int(xmax * w), int(ymax * h)
    
    # Add padding (context)
    padding = 64
    crop_x1 = max(0, px_xmin - padding)
    crop_y1 = max(0, px_ymin - padding)
    crop_x2 = min(w, px_xmax + padding)
    crop_y2 = min(h, px_ymax + padding)

    # 2. Crop & Resize for Model
    crop = original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    old_size = crop.size
    crop_resized = crop.resize((512, 512), Image.Resampling.LANCZOS)

    # 3. Generate
    # We use the SPECIFIC texture prompt here
    with torch.autocast("cuda"):
        generated = pipe(
            prompt=defect_info["prompt"], 
            negative_prompt=NEGATIVE_PROMPT,
            gligen_phrases=[defect_info["phrase"]],
            gligen_boxes=[[0.0, 0.0, 1.0, 1.0]], # Target the whole crop
            gligen_inpaint_image=crop_resized,
            gligen_scheduled_sampling_beta=0.3, # Low beta = blend with existing asphalt
            num_inference_steps=40,
            guidance_scale=7.5,
        ).images[0]

    # 4. Resize back
    generated = generated.resize(old_size, Image.Resampling.LANCZOS)

    # 5. Soft Blending Mask (To hide the square edges)
    # Create a white box with black borders
    mask = Image.new("L", old_size, 0)
    mask_w, mask_h = old_size
    # Draw inner white rectangle with some margin
    margin = 20
    if mask_w > margin*2 and mask_h > margin*2:
        inner_box = (margin, margin, mask_w - margin, mask_h - margin)
        ImageDraw = Image.new("L", old_size, 0) # Temp helper
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle(inner_box, fill=255)
        # Blur the mask to create a soft edge
        mask = mask.filter(ImageFilter.GaussianBlur(10))
    else:
        # If crop is too small, just use full opacity
        mask = Image.new("L", old_size, 255)

    # 6. Paste using the Soft Mask
    original_image.paste(generated, (crop_x1, crop_y1), mask=mask)
    return original_image

def main():
    ensure_dir(OUTPUT_DIR)

    print("⏳ Loading GLIGEN Model...")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-inpainting-text-box",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    ).to("cuda")
    # Redundant safety kill
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    print("✅ Model Loaded.")

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")
        img_path = os.path.join(INPUT_IMAGES_DIR, img_file)

        if not os.path.exists(label_path):
            continue

        # Load fresh image
        final_image = Image.open(img_path).convert("RGB")
        w, h = final_image.size
        
        # Load defects
        defects = load_yolo_boxes(label_path, w, h)
        if not defects: continue

        print(f"Processing {img_file} ({len(defects)} defects)...")

        for defect in defects:
            final_image = crop_and_process(pipe, final_image, defect)

        save_path = os.path.join(OUTPUT_DIR, img_file)
        final_image.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()