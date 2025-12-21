"""
inpaintMaker.py
Class for Generating Synthetic Oil Spills with YOLO Labels
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

class InpaintMaker:
    def __init__(self, device_name='auto', model_id="runwayml/stable-diffusion-inpainting"):
        """
        Initializes the Stable Diffusion Inpainting pipeline.
        """
        self.device = self._setup_device(device_name)
        print(f"--- Loading Inpainting Model: {model_id} ---")
        
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in device_name else None
            ).to(self.device)
            
            # Disable safety checker to prevent false positives on industrial textures
            self.pipe.safety_checker = None
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise e

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

    def _get_yolo_labels(self, mask_pil, class_id=0):
        """
        Analyzes the mask to find bounding boxes for the YOLO label file.
        Returns a list of strings: "class_id x_c y_c w h"
        """
        # Convert PIL to Numpy for OpenCV processing
        mask_np = np.array(mask_pil.convert('L'))
        
        # Threshold to ensure binary (0 or 255)
        _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours (handles multiple spill spots in one mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = mask_np.shape
        yolo_lines = []

        for cnt in contours:
            # Filter out tiny noise pixels
            if cv2.contourArea(cnt) < 50: 
                continue

            # Get bounding box: x, y, w, h (top-left based)
            x, y, w, h = cv2.boundingRect(cnt)

            # Convert to YOLO format (Normalized Center-based)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Constraint checks to ensure 0-1 range
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            w_norm = min(max(w_norm, 0), 1)
            h_norm = min(max(h_norm, 0), 1)

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        return yolo_lines

    def process_batch(self, masks_folder, image_folder, output_image_folder, output_label_folder):
        """
        Main processing loop.
        """
        # Create output directories
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_label_folder, exist_ok=True)

        # Get list of masks
        mask_files = [f for f in os.listdir(masks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_files = len(mask_files)
        
        print(f"Found {total_files} masks. Starting generation...")

        for i, filename in enumerate(mask_files):
            mask_path = os.path.join(masks_folder, filename)
            img_path = os.path.join(image_folder, filename)

            # Check if corresponding original image exists
            if not os.path.exists(img_path):
                print(f"Skipping {filename}: Original image not found in {image_folder}")
                continue

            try:
                # 1. Load and Resize
                # SD v1.5 works best at 512x512. We resize both to match.
                original_img = Image.open(img_path).convert("RGB").resize((512, 512))
                mask_img = Image.open(mask_path).convert("L").resize((512, 512))

                # 2. Generate YOLO Labels *before* inpainting (based on the mask)
                labels = self._get_yolo_labels(mask_img)
                if not labels:
                    print(f"[{i+1}/{total_files}] Warning: Empty mask for {filename}. Skipping.")
                    continue

                # 3. Run Inpainting
                # Prompt optimized for industrial spills
                prompt = "A thick, opaque pitch-black industrial oil spill, wet reflective liquid, high contrast on gray asphalt"
                
                with torch.autocast(self.device):
                    result = self.pipe(
                        prompt=prompt,
                        image=original_img,
                        mask_image=mask_img,
                        guidance_scale=20.0, # High force to adhere to mask shape
                        strength=1.0,
                        num_inference_steps=30
                    ).images[0]

                # 4. Save Results
                # Save Image
                # We change extension to .jpg for standard YOLO datasets
                save_name = os.path.splitext(filename)[0]
                result.save(os.path.join(output_image_folder, f"{save_name}.jpg"))

                # Save Labels
                with open(os.path.join(output_label_folder, f"{save_name}.txt"), "w") as f:
                    f.write("\n".join(labels))

                print(f"[{i+1}/{total_files}] Generated: {save_name}")

            except Exception as e:
                print(f"ERROR processing {filename}: {e}")