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

# YOLO Class Mapping
CLASSES = {
    0: "Oil Spill",
    1: "Foreign Object Debris"
}

# Prompts for each class
PROMPTS = {
    0: [
        "dark and black oil spill on asphalt, iridescent, liquid texture, realistic, high contrast",
        "puddle of dirty oil leaking on concrete, realistic, 8k"
    ],
    1: [
        "rusty metal wrench lying on the ground, hard shadow, realistic, 8k",
        "crushed soda can garbage on asphalt, realistic",
        "metal bolt screw debris, macro photography, highly detailed"
    ]
}

# --- LOAD MODELS (Global to save time) ---
print(f"Loading models on {DEVICE}...")

# 1. Segmentation (Finds the road)
seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(DEVICE)

# 2. Inpainting (Generates the object)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

def get_driveway_mask(image_pil):
    """
    Returns a binary mask: 255 = Valid Driveway/Road, 0 = Invalid.
    ADE20K indices: 6=Road, 11=Sidewalk, 13=Earth, 52=Path
    """
    inputs = seg_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    # Resize output to match original image
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image_pil.size[::-1], 
        mode="bilinear", align_corners=False
    )
    pred_seg = logits.argmax(dim=1)[0]
    
    # Create mask for valid surfaces
    valid_classes = torch.tensor([6, 11, 52]).to(DEVICE)
    mask = torch.isin(pred_seg, valid_classes)
    
    return mask.cpu().numpy().astype(np.uint8) * 255

def process_single_image(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    
    # 1. Load Image
    init_image = Image.open(image_path).convert("RGB")
    w_img, h_img = init_image.size
    
    # 2. Segment Driveway
    valid_zone_mask = get_driveway_mask(init_image)
    
    # Find valid coordinates (y, x)
    valid_y, valid_x = np.where(valid_zone_mask == 255)
    
    if len(valid_x) == 0:
        print("Error: No driveway/road detected in this image.")
        return

    # 3. Choose Object Type & Location
    class_id = random.choice([0, 1]) # 0 = Oil, 1 = FOD
    prompt = random.choice(PROMPTS[class_id])
    
    # Pick random center from valid zone
    idx = random.randint(0, len(valid_x) - 1)
    center_x, center_y = valid_x[idx], valid_y[idx]
    
    # Random Box Size (5% to 15% of image dimension)
    box_size_min = int(min(w_img, h_img) * 0.05)
    box_size_max = int(min(w_img, h_img) * 0.15)
    
    box_w = random.randint(box_size_min, box_size_max)
    box_h = random.randint(box_size_min, box_size_max)
    
    # Calculate Top-Left (x,y)
    x = int(center_x - box_w / 2)
    y = int(center_y - box_h / 2)
    
    # Clamp to image boundaries
    x = max(0, min(x, w_img - box_w))
    y = max(0, min(y, h_img - box_h))
    
    # 4. Create Inpainting Mask
    inpaint_mask_arr = np.zeros((h_img, w_img), dtype=np.uint8)
    inpaint_mask_arr[y:y+box_h, x:x+box_w] = 255
    inpaint_mask = Image.fromarray(inpaint_mask_arr)
    
    # 5. Generate Object
    print(f"Generating '{CLASSES[class_id]}'...")
    result_image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=inpaint_mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=1.0 
    ).images[0]
    
    # 6. Save Data (YOLO Format)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
    
    result_image.save(f"{OUTPUT_DIR}/images/{filename}_aug.jpg")
    
    # Normalize coordinates for YOLO
    x_center = (x + box_w / 2) / w_img
    y_center = (y + box_h / 2) / h_img
    w_norm = box_w / w_img
    h_norm = box_h / h_img
    
    label_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
    
    with open(f"{OUTPUT_DIR}/labels/{filename}_aug.txt", "w") as f:
        f.write(label_str)

    # 7. Visualization (Pyplot)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot A: Segmentation Mask
    ax[0].imshow(valid_zone_mask, cmap='gray')
    ax[0].set_title("Valid Driveway Zone (White)")
    ax[0].axis('off')

    # Plot B: Inpaint Mask location
    ax[1].imshow(inpaint_mask_arr, cmap='gray')
    ax[1].set_title(f"Target Location for {CLASSES[class_id]}")
    ax[1].axis('off')

    # Plot C: Final Result with Bounding Box
    ax[2].imshow(result_image)
    rect = patches.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)
    ax[2].text(x, y - 5, CLASSES[class_id], color='red', fontweight='bold')
    ax[2].set_title("Result + YOLO Label")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Saved: {OUTPUT_DIR}/images/{filename}_aug.jpg")

# --- EXECUTION ---
# Replace this with your actual image path
image_path = "runway_image.jpg" 

# Create a dummy image for testing if file doesn't exist
if not os.path.exists(image_path):
    print("Test image not found, creating dummy image...")
    dummy = Image.new('RGB', (512, 512), color = (128, 128, 128)) # Grey background
    dummy.save("test_runway.jpg")
    image_path = "test_runway.jpg"
process_single_image(image_path)