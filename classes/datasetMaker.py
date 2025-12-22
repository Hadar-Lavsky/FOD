import os
import gc
import torch
import cv2
from classes.maskMaker import MaskMaker
from classes.inpaintMaker import InpaintMaker
import hyperParams as hp

class DatasetMaker:
    def __init__(self):
        self.raw_dir = hp.RAW_FOLDER
        self.mask_dir = hp.MASK_FOLDER
        self.inpaint_img_dir = hp.INPAINT_IMAGES_FOLDER
        self.inpaint_lbl_dir = hp.INPAINT_LABELS_FOLDER
        
        # Work Queues
        self.files_to_mask = []
        self.files_to_inpaint = []
        
        # Ensure directories exist
        for d in [self.mask_dir, self.inpaint_img_dir, self.inpaint_lbl_dir]:
            os.makedirs(d, exist_ok=True)

    def _get_filename_no_ext(self, filename):
        return os.path.splitext(filename)[0]

    # Scans folders and creates work queues
    def scan_and_plan(self):
        print("\n--- Scanning Dataset State ---")
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        raw_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith(valid_exts)]
        
        count_clean = 0
        total_count = 0
        
        for filename in raw_files:
            total_count += 1
            name = self._get_filename_no_ext(filename)
            
            # Define expected paths
            raw_path = os.path.join(self.raw_dir, filename)
            
            # To match specific filenames, we look for the file in the dirs
            mask_exists = any(f.startswith(name) for f in os.listdir(self.mask_dir))
            
            # For inpaint, we look for the name in the images folder
            inpaint_exists = any(f.startswith(name) for f in os.listdir(self.inpaint_img_dir))

            # Case: Invalid State (No mask, but Inpaint exists) -> Delete Inpaint, Redo All
            if not mask_exists and inpaint_exists:
                print(f"Correction needed: {filename} (Inpaint exists without Mask). Cleaning...")
                self._delete_inpaint_artifacts(name)
                self.files_to_mask.append(filename)
                self.files_to_inpaint.append(filename)
                count_clean += 1
                
            # Case: Fresh File (No Mask, No Inpaint)
            elif not mask_exists and not inpaint_exists:
                self.files_to_mask.append(filename)
                self.files_to_inpaint.append(filename)
                
            # Case: Partial State (Mask exists, No Inpaint)
            elif mask_exists and not inpaint_exists:
                self.files_to_inpaint.append(filename)
                
            # Case: Complete (Mask exists, Inpaint exists) -> Do nothing
            else:
                pass

        print(f"Scan Complete.")
        print(f"Total files: {total_count}")
        print(f"Files to clean/reset: {count_clean}")
        print(f"Files to mask:    {len(self.files_to_mask)}")
        print(f"Files to inpaint: {len(self.files_to_inpaint)}")

    # removes files
    def _delete_inpaint_artifacts(self, filename_no_ext):
        # Remove image
        for f in os.listdir(self.inpaint_img_dir):
            if f.startswith(filename_no_ext):
                os.remove(os.path.join(self.inpaint_img_dir, f))
        # Remove label
        for f in os.listdir(self.inpaint_lbl_dir):
            if f.startswith(filename_no_ext):
                os.remove(os.path.join(self.inpaint_lbl_dir, f))

    # main pipeline
    def run(self):
        self.scan_and_plan()

        if not self.files_to_mask and not self.files_to_inpaint:
            print("\nDataset is up to date. No actions needed.")
            return

        # masking
        if self.files_to_mask:
            print("\n>>> STAGE 1: Generating Masks")
            mask_engine = MaskMaker(device_name="auto")
            
            for i, filename in enumerate(self.files_to_mask):
                input_path = os.path.join(self.raw_dir, filename)
                output_path = os.path.join(self.mask_dir, filename) # Saving with same extension
                
                mask = mask_engine.get_mask(input_path)
                if mask is not None:
                    cv2.imwrite(output_path, mask)
                    print(f"[{i+1}/{len(self.files_to_mask)}] Masked: {filename}")
                else:
                    print(f"Failed to mask: {filename}")

            # clean up vram
            print("Cleaning up Mask Engine...")
            del mask_engine
            gc.collect()
            torch.cuda.empty_cache()

        # inpaint
        if self.files_to_inpaint:
            print("\n>>> STAGE 2: Generating Inpaint Data")
            inpaint_engine = InpaintMaker(device_name="auto")

            for i, filename in enumerate(self.files_to_inpaint):
                name = self._get_filename_no_ext(filename)
                mask_candidates = [f for f in os.listdir(self.mask_dir) if f.startswith(name)]
                if not mask_candidates:
                    print(f"Skipping {filename}: Mask not found (generation likely failed).")
                    continue
                
                mask_filename = mask_candidates[0]
                
                mask_path = os.path.join(self.mask_dir, mask_filename)
                raw_path = os.path.join(self.raw_dir, filename)
                
                # Force output to .jpg for YOLO standard
                out_img_path = os.path.join(self.inpaint_img_dir, f"{name}.jpg")
                out_lbl_path = os.path.join(self.inpaint_lbl_dir, f"{name}.txt")
                
                success = inpaint_engine.process_single(mask_path, raw_path, out_img_path, out_lbl_path)
                
                if success:
                    print(f"[{i+1}/{len(self.files_to_inpaint)}] Inpainted: {filename}")
            
            print("Cleaning up Inpaint Engine...")
            del inpaint_engine
            gc.collect()
            torch.cuda.empty_cache()

        print("\nAll pipeline tasks finished.")