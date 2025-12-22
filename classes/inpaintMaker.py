"""
inpaintMaker.py
Class for Generating Synthetic Oil Spills with YOLO Labels
Refactored for Single-Image Orchestration
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

class InpaintMaker:
    def __init__(self, device_name='auto', model_id="runwayml/stable-diffusion-inpainting"):
        self.device = self._setup_device(device_name)
        print(f"--- Loading Inpainting Model: {model_id} ---")
        
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in device_name else None
            ).to(self.device)
            self.pipe.safety_checker = None
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise e

    def _setup_device(self, request):
        if request == 'auto':
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return 'xpu'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return request

    def _get_yolo_labels(self, mask_pil, class_id=0):
        mask_np = np.array(mask_pil.convert('L'))
        _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = mask_np.shape
        yolo_lines = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue
            x, y, w, h = cv2.boundingRect(cnt)
            
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            # Clamp values
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            w_norm = min(max(w_norm, 0), 1)
            h_norm = min(max(h_norm, 0), 1)

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        return yolo_lines

    def process_single(self, mask_path, img_path, output_img_path, output_lbl_path):
        """
        Public method to process a single image/mask pair.
        """
        if not os.path.exists(img_path):
            print(f"Skipping: Original image missing at {img_path}")
            return False

        try:
            # 1. Load and Resize
            original_img = Image.open(img_path).convert("RGB").resize((512, 512))
            mask_img = Image.open(mask_path).convert("L").resize((512, 512))

            # 2. Generate YOLO Labels
            labels = self._get_yolo_labels(mask_img)
            if not labels:
                return False

            # 3. Run Inpainting
            prompt = "A thick, opaque pitch-black industrial oil spill, wet reflective liquid, high contrast on gray asphalt"
            
            # Use autocast only if on CUDA/XPU
            context = torch.autocast(self.device) if "cuda" in self.device else torch.no_grad()
            
            with context:
                result = self.pipe(
                    prompt=prompt,
                    image=original_img,
                    mask_image=mask_img,
                    guidance_scale=20.0,
                    strength=1.0,
                    num_inference_steps=30
                ).images[0]

            # 4. Save Results
            result.save(output_img_path)
            with open(output_lbl_path, "w") as f:
                f.write("\n".join(labels))

            return True

        except Exception as e:
            print(f"ERROR processing {os.path.basename(img_path)}: {e}")
            return False