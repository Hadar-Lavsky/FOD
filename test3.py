import torch
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from ultralytics import SAM

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "dataset_sam_output"
# We use the Base model (sam_b.pt) for a balance of speed and high accuracy
SAM_MODEL_TYPE = "sam_b.pt" 

CLASSES = { 0: "Oil Spill", 1: "Foreign Object Debris" }

# --- LOAD MODELS ---
print(f"Loading Stable Diffusion on {DEVICE}...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

print(f"Loading SAM ({SAM_MODEL_TYPE})...")
# Ultralytics will auto-download this model on first run
sam_model = SAM(SAM_MODEL_TYPE)

def get_runway_mask_with_sam(image_path):
    """
    Uses SAM with a 'Point Prompt' at the bottom-center of the image.
    This assumes the drone/camera is looking at the runway.
    """
    # 1. Load image to get dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 2. Define a prompt point: Bottom Center (80% down, 50% across)
    # This tells SAM: "Segment the object located here"
    prompt_point = [[w/2, h * 0.8]]
    
    # 3. Predict
    # labels=[1] means the point is a "positive" click (include this area)
    results = sam_model(image_path, points=prompt_point, labels=[1], verbose=False)
    
    # 4. Extract Mask
    # SAM returns multiple masks (small, medium, large). We usually want the largest one.
    if results[0].masks is None:
        return None
        
    masks = results[0].masks.data.cpu().numpy() # Shape: (N, H, W)
    
    # Pick the mask with the largest area (usually the whole runway)
    best_mask = max(masks, key=lambda x: x.sum())
    
    # Convert to uint8 (0 or 255)
    return (best_mask * 255).astype(np.uint8)

def process_single_image(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    init_image = Image.open(image_path).convert("RGB")
    w_img, h_img = init_image.size

    # --- STEP 1: SEGMENTATION (The Fix) ---
    print(f"Segmenting {filename} using SAM...")
    runway_mask = get_runway_mask_with_sam(image_path)
    
    if runway_mask is None or runway_mask.sum() == 0:
        print("Error: SAM could not find the runway.")
        return

    # --- STEP 2: FIND VALID SPOT ---
    valid_y, valid_x = np.where(runway_mask > 0)
    
    # Erosion: Optional step to shrink the mask slightly so we don't spawn on the exact edge
    # This prevents debris from hanging half-off the runway
    # kernel = np.ones((20, 20), np.uint8)
    # eroded_mask = cv2.erode(runway_mask, kernel, iterations=1)
    # valid_y, valid_x = np.where(eroded_mask > 0)

    if len(valid_x) == 0:
        print("No valid runway area found after erosion.")
        return

    # Pick random center
    idx = random.randint(0, len(valid_x) - 1)
    center_x, center_y = valid_x[idx], valid_y[idx]

    # Box Size
    box_size = random.randint(int(min(w_img, h_img) * 0.05), int(min(w_img, h_img) * 0.12))
    
    # Top-Left calc
    x = int(center_x - box_size / 2)
    y = int(center_y - box_size / 2)
    x = max(0, min(x, w_img - box_size))
    y = max(0, min(y, h_img - box_size))

    # --- STEP 3: INPAINT ---
    inpaint_mask_arr = np.zeros((h_img, w_img), dtype=np.uint8)
    inpaint_mask_arr[y:y+box_size, x:x+box_size] = 255
    inpaint_mask = Image.fromarray(inpaint_mask_arr)

    class_id = random.choice([0, 1])
    # Tweak prompts slightly for overhead view
    prompts = [
        "dark shiny oil spill on asphalt, liquid texture, top down view, realistic, 4k",
        "cardboard box garbage lying on asphalt, casting a shadow, top down view, realistic"
    ]
    
    print(f"Generating {CLASSES[class_id]}...")
    result_image = pipe(
        prompt=prompts[class_id],
        image=init_image,
        mask_image=inpaint_mask,
        num_inference_steps=25,
        guidance_scale=8.0,
        strength=1.0
    ).images[0]

    # --- STEP 4: SAVE ---
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
    
    result_image.save(f"{OUTPUT_DIR}/images/{filename}_sam.jpg")
    
    # YOLO Format
    x_c, y_c = (x + box_size/2)/w_img, (y + box_size/2)/h_img
    w_n, h_n = box_size/w_img, box_size/h_img
    
    with open(f"{OUTPUT_DIR}/labels/{filename}_sam.txt", "w") as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. SAM Mask
    ax[0].imshow(runway_mask, cmap='gray')
    ax[0].plot(w_img/2, h_img*0.8, 'rx', markersize=12, markeredgewidth=3) # Show the click point
    ax[0].set_title("SAM Mask (Red X = Auto-Click)")
    
    # 2. Placement
    ax[1].imshow(inpaint_mask_arr, cmap='gray')
    ax[1].set_title(f"Target: {CLASSES[class_id]}")
    
    # 3. Result
    ax[2].imshow(result_image)
    rect = patches.Rectangle((x, y), box_size, box_size, linewidth=2, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)
    ax[2].set_title("Final Output")
    
    plt.show()

# Run
image_path = "runway_image.jpg" # Your uploaded file
process_single_image(image_path)