"""
Semantic Segmentation Script for Runway Detection (FIXED)
"""

import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import random

def load_model(device='cpu'):
    # Using ADE20K model (Better for runways)
    model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    
    print(f"Loading ADE20K Model: {model_name}")
    
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    except Exception as e:
        print("\nERROR: Could not download the model.")
        raise e

    model.eval()
    
    if device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        model = model.to(device)
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = 'cpu'
    
    return processor, model, device


def create_debug_overlay(seg_map, original_image_cv, id2label):
    h, w, _ = original_image_cv.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_classes = np.unique(seg_map)
    
    np.random.seed(42) # Consistent colors
    
    print("\n--- DETAILED DEBUG LOG ---")
    for cls_id in unique_classes:
        label_name = id2label.get(cls_id, f"Unknown_ID_{cls_id}")
        color = np.random.randint(0, 255, size=3).tolist()
        color_mask[seg_map == cls_id] = color
        print(f"ID {cls_id}: '{label_name}' -> Colored RGB{color}")
    print("--------------------------\n")

    overlay = cv2.addWeighted(original_image_cv, 0.6, color_mask, 0.4, 0)
    return overlay

def get_runway_mask(image_path, device='cpu'):
    processor, model, device = load_model(device=device)
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Running segmentation...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=original_size[::-1], mode="bilinear", align_corners=False
    )
    
    seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # --- ADE20K TARGETS ---
    # 6: Road
    # 52: Runway
    TARGET_IDS = [6, 54] 
    
    # --- FIX START ---
    # We use np.isin() to check against multiple IDs at once
    mask_bool = np.isin(seg_map, TARGET_IDS)
    binary_mask = (mask_bool * 255).astype(np.uint8)
    # --- FIX END ---
    
    # Save debug overlay
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Get id2label mapping safely
    id2label = getattr(model.config, 'id2label', {})
    if not id2label:
        # Fallback: create a simple mapping
        id2label = {i: f"Class_{i}" for i in range(150)}
    overlay_img = create_debug_overlay(seg_map, open_cv_image, id2label)
    cv2.imwrite("debug_overlay_ade20k.jpg", overlay_img)

    print(f"Success! Detected {np.sum(binary_mask > 0)} runway pixels.")
    return binary_mask

if __name__ == "__main__":
    # Ensure this file exists or change the name
    try:
        mask = get_runway_mask("runway01.jpg")
        cv2.imwrite("final_mask.png", mask)
        
        if np.sum(mask) == 0:
            print("\nWARNING: Mask is empty.")
            print("Check debug_overlay_ade20k.jpg to see which ID the model is choosing.")
        else:
            print("Success! Mask saved as 'final_mask.png'.")
    except Exception as e:
        print(f"Error: {e}")