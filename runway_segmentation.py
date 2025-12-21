"""
Semantic Segmentation Script for Runway Detection
Uses Segformer-B0 model from Hugging Face Transformers to segment runway from aerial imagery.
Outputs a binary mask where runway pixels are white (255) and background is black (0).
"""

import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch


def load_model(device='cpu'):
    """Load the Segformer-B0 model and image processor from Hugging Face."""
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    print(f"Loading model: {model_name}")
    
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    
    # Move model to device
    if device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        model = model.to(device)
        print(f"Using device: {device}")
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        print(f"Using device: {device}")
    else:
        device = 'cpu'
        print(f"Using device: cpu")
    
    print("Model loaded successfully!")
    return processor, model, device


def get_runway_mask(image_path, device='cpu'):
    """
    Get binary mask for runway from an aerial image.
    
    Args:
        image_path: Path to input image file
        device: Device to run inference on ('cpu', 'cuda', or 'xpu')
    
    Returns:
        Binary mask as NumPy array (255 for runway, 0 for background)
    """
    # Load model
    processor, model, device = load_model(device=device)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    print(f"Loaded image: {image_path}")
    print(f"Original image size: {original_size}")
    
    # Process image (processor resizes to 512x512 for model)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    print("Running segmentation...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and upsample to original image size
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )
    
    # Get predicted class for each pixel
    seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # ADE20K class indices: Class 0 = road, Class 1 = sidewalk/pavement
    # Runways are typically classified as roads
    ROAD_CLASS_ID = 0
    SIDEWALK_CLASS_ID = 1
    

    # We want to treat Road (0), Sidewalk (1), and Traffic Signs (7) all as "Runway Surface"
    # to avoid holes where the painted numbers are.
    TARGET_CLASSES = [0, 1, 7] 
    
    binary_mask = np.zeros_like(seg_map, dtype=np.uint8)
    for class_id in TARGET_CLASSES:
        binary_mask[seg_map == class_id] = 255
    
    # FILL HOLES: Use morphological closing to fill in the small gaps (like the "car" misdetection)
    kernel = np.ones((7,7), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # Create binary mask: Runway = 255 (White), everything else = 0 (Black)
    # Get predicted class for each pixel
    seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # DEBUG: Print all classes found in the image to see what the model "sees"
    unique_classes = np.unique(seg_map)
    print(f"DEBUG: Unique classes detected in image: {unique_classes}")
    binary_mask = np.zeros_like(seg_map, dtype=np.uint8)
    binary_mask[seg_map == ROAD_CLASS_ID] = 255
    binary_mask[seg_map == SIDEWALK_CLASS_ID] = 100
    
    print(f"Detected classes: {np.unique(seg_map)}")
    print(f"Binary mask created: {binary_mask.shape}")
    
    return binary_mask


# Usage
if __name__ == "__main__":
    binary_mask = get_runway_mask("runway02.jpg")
    cv2.imwrite("runway_maskwww02.png", binary_mask)
    print("Mask saved to: runway_mask.png")
