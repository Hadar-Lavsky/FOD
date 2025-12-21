"""
maskMaker.py
Semantic Segmentation Class for Runway Detection
Refactored for Batch Processing
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import argparse

class MaskMaker:
    def __init__(self, device_name='auto'):
        """
        Initialize model and processor once to save resources.
        """
        self.device = self._setup_device(device_name)
        self.model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
        self.processor = None
        self.model = None
        
        # --- ADE20K TARGETS (Preserved from original code) ---
        # 6: Road, 54: Runway (based on code logic provided)
        self.TARGET_IDS = [6, 54] 
        
        self._load_model()

    def _setup_device(self, request):
        """
        Internal method to determine the best available hardware.
        """
        if request == 'auto':
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return 'xpu'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return request

    def _load_model(self):
        print(f"Loading ADE20K Model: {self.model_name} on {self.device}...")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print("\nERROR: Could not download or load the model.")
            raise e

    def get_mask(self, image_path):
        """
        Generates a binary mask for a single image.
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Skipping file (not an image or unreadable): {image_path}")
            return None

        original_size = image.size
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        
        seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        # Create Binary Mask based on TARGET_IDS
        mask_bool = np.isin(seg_map, self.TARGET_IDS)
        binary_mask = (mask_bool * 255).astype(np.uint8)
        
        return binary_mask

    def process_batch(self, input_folder, output_folder):
        """
        Main function to process all images in input_folder and save to output_folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")

        # specific extensions to look for
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
        total_files = len(files)
        
        print(f"Found {total_files} images in '{input_folder}'. Starting processing...")

        for i, filename in enumerate(files):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Generate mask
            mask = self.get_mask(input_path)
            
            if mask is not None:
                # Save the mask using OpenCV
                cv2.imwrite(output_path, mask)
                
                # Basic progress logging
                mask_pixels = np.sum(mask > 0)
                status = "Runway detected" if mask_pixels > 0 else "Empty mask"
                print(f"[{i+1}/{total_files}] Processed: {filename} -> {status}")
            else:
                print(f"[{i+1}/{total_files}] Failed: {filename}")

        print("\nBatch processing complete.")