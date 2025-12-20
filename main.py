"""
Semantic Segmentation Script for Runway Detection (Debug Mode)
"""

import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import random

def load_model(device='cpu'):
    # START CHANGE: Switching to Cityscapes (Dashcam View)
    # This model understands roads that vanish into the distance (perspective).
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    # END CHANGE
    
    print(f"Loading Driver-View Model: {model_name}")
    
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
    """
    Creates a color-coded overlay to visualize what the model sees.
    """
    h, w, _ = original_image_cv.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(seg_map)
    
    # Generate random colors for each class found
    np.random.seed(42)
    
    print("\n--- DETAILED DEBUG LOG ---")
    for cls_id in unique_classes:
        # Get label name (handle cases where config might be missing)
        label_name = id2label.get(cls_id, f"Unknown_ID_{cls_id}")
        
        # Assign a random bright color
        color = np.random.randint(0, 255, size=3).tolist()
        
        # Color the mask where this class is detected
        color_mask[seg_map == cls_id] = color
        
        print(f"ID {cls_id}: '{label_name}' -> Colored RGB{color}")
    print("--------------------------\n")

    # Blend original image with the colored mask
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

    # --- CITYSCAPES CLASS MAPPING ---
    # 0: Road (This is your Runway!)
    # 1: Sidewalk
    # 2: Building
    # 8: Vegetation
    # 10: Sky
    # --------------------------------
    
    TARGET_CLASS_ID = 0  # In Cityscapes, Road is ALWAYS 0
    
    binary_mask = np.zeros_like(seg_map, dtype=np.uint8)
    binary_mask[seg_map == TARGET_CLASS_ID] = 255
    
    # Save the debug overlay again to verify
    # (Ensure you pasted the 'create_debug_overlay' function from before)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay_img = create_debug_overlay(seg_map, open_cv_image, model.config.id2label)
    cv2.imwrite("debug_overlay_cityscapes.jpg", overlay_img)

    print(f"Success! Detected {np.sum(binary_mask > 0)} runway pixels.")
    return binary_mask

if __name__ == "__main__":
    mask = get_runway_mask("runway01.jpg")
    cv2.imwrite("final_mask.png", mask)
    
    if np.sum(mask) == 0:
        print("\nWARNING: The final mask is empty (All Black).")
        print("Look at 'debug_overlay.jpg' to see what class ID the runway actually is,")
        print("then add that ID to the 'TARGET_CLASSES' list in the code.")
    else:
        print("Success! Mask saved.")