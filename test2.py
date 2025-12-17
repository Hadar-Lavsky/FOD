import torch
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "dataset_output"

# Define Classes
CLASSES = { 0: "Oil Spill", 1: "Foreign Object Debris" }

# --- LOAD MODELS ---
print(f"Loading models on {DEVICE}...")

# 1. SEGMENTATION: Using 'Cityscapes' (B4 size) which is superior for road/surface detection
# Class 0 in Cityscapes = Road
seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024").to(DEVICE)

# 2. INPAINTING: Standard SD Inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

def get_smart_runway_mask(image_pil):
    """
    1. Predicts 'Road' pixels.
    2. Fills small holes (morphology).
    3. Removes background roads (keeps only largest connected area).
    """
    # A. Model Prediction
    inputs = seg_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    # Resize mask to original image size
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image_pil.size[::-1], # (h, w)
        mode="bilinear", align_corners=False
    )
    pred_seg = logits.argmax(dim=1)[0].cpu().numpy()
    
    # B. Filter: Keep only Class 0 (Road)
    # Cityscapes: 0=Road, 1=Sidewalk, 2=Building...
    binary_mask = (pred_seg == 0).astype(np.uint8) * 255

    # C. Morphology: Close small holes in the detection
    kernel = np.ones((15, 15), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # D. Connected Components: Keep ONLY the largest blob (The Runway)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # If no road found, return empty
    if num_labels < 2: 
        return np.zeros_like(binary_mask)

    # Find the label with the largest area (ignoring label 0 which is background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a clean mask with ONLY the largest chunk
    final_mask = np.zeros_like(binary_mask)
    final_mask[labels == largest_label] = 255
    
    return final_mask

def process_single_image(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    
    # Load Image
    init_image = Image.open(image_path).convert("RGB")
    w_img, h_img = init_image.size
    
    print("Segmenting runway...")
    runway_mask = get_smart_runway_mask(init_image)
    
    # Check if we found a runway
    valid_y, valid_x = np.where(runway_mask == 255)
    if len(valid_x) == 0:
        print("Error: No runway detected.")
        return

    # --- PLACEMENT LOGIC ---
    # Pick random point on the runway
    idx = random.randint(0, len(valid_x) - 1)
    center_x, center_y = valid_x[idx], valid_y[idx]
    
    # Object Size (Adjust these for larger/smaller debris)
    box_size = random.randint(int(min(w_img, h_img) * 0.05), int(min(w_img, h_img) * 0.15))
    x = max(0, min(int(center_x - box_size/2), w_img - box_size))
    y = max(0, min(int(center_y - box_size/2), h_img - box_size))
    
    # Create Inpainting Mask (White box on black background)
    inpaint_mask_arr = np.zeros((h_img, w_img), dtype=np.uint8)
    inpaint_mask_arr[y:y+box_size, x:x+box_size] = 255
    inpaint_mask = Image.fromarray(inpaint_mask_arr)

    # --- GENERATION ---
    class_id = random.choice([0, 1])
    prompts = [
        "dark shiny oil spill on asphalt, highly detailed, realistic, 8k", # Class 0
        "rusty metal wrench tool lying on asphalt, casting a shadow, realistic" # Class 1
    ]
    
    print(f"Generating {CLASSES[class_id]}...")
    result_image = pipe(
        prompt=prompts[class_id],
        image=init_image,
        mask_image=inpaint_mask,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # --- SAVE RESULTS ---
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
    
    save_path_img = f"{OUTPUT_DIR}/images/{filename}_smart.jpg"
    save_path_txt = f"{OUTPUT_DIR}/labels/{filename}_smart.txt"
    
    result_image.save(save_path_img)
    
    # YOLO Label Normalization
    x_c, y_c = (x + box_size/2)/w_img, (y + box_size/2)/h_img
    w_n, h_n = box_size/w_img, box_size/h_img
    with open(save_path_txt, "w") as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- VISUALIZATION (Pyplot) ---
    print("Displaying Debug Plot...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. The Smart Mask (What the AI thinks is the runway)
    ax[0].imshow(runway_mask, cmap='gray')
    ax[0].set_title("Smart Mask (Largest Connected Component)")
    ax[0].axis('off')
    
    # 2. The Target Location (Where we are putting the object)
    # Overlay the box on the mask to see if it fits
    overlay = runway_mask.copy()
    overlay[y:y+box_size, x:x+box_size] = 128 # Grey box
    ax[1].imshow(overlay, cmap='gray')
    ax[1].set_title(f"Placement: {CLASSES[class_id]}")
    ax[1].axis('off')

    # 3. Final Result
    ax[2].imshow(result_image)
    rect = patches.Rectangle((x, y), box_size, box_size, linewidth=2, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)
    ax[2].text(x, y-10, CLASSES[class_id], color='red', fontweight='bold', fontsize=12)
    ax[2].set_title("Final Synthetic Image")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- RUN IT ---
# Replace with your actual image
image_path = "runway_image.jpg" 
if not os.path.exists(image_path):
    print("Please provide a valid image path.")
else:
    process_single_image(image_path)