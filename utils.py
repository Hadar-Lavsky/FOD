import os
import hyperParams as hp
import shutil
import random
import glob
from tqdm import tqdm

# checks and creates directories needed for the project, these directories are in .gitignore
def setup_directories():
    # Base path for the YOLO ready dataset

    folders_to_check = [
        # Existing folders
        hp.RAW_FOLDER,
        hp.MASK_FOLDER,
        hp.INPAINT_IMAGES_FOLDER,
        hp.INPAINT_LABELS_FOLDER,
        
        # Yolo folders
        os.path.join(hp.YOLO_BASE_FOLDER, "train", "images"),
        os.path.join(hp.YOLO_BASE_FOLDER, "train", "labels"),
        os.path.join(hp.YOLO_BASE_FOLDER, "val", "images"),
        os.path.join(hp.YOLO_BASE_FOLDER, "val", "labels"),
    ]

    print("Checking directories...")
    
    for folder_path in folders_to_check:
        try:
            # exist_ok=True creates the directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            # Only print if we actually created it or it's important (optional cleanup)
            # print(f"Verified: {folder_path}") 
        except OSError as e:
            print(f"Error creating {folder_path}: {e}")
            
    print("Directory structure verified.")
    

# Creates the final yolo dataset
# cleans the target folder
# splits data into train/val
# copies inpaints and their labels
# copies raws and creates empty labels for them
def split_and_copy_dataset():
    print("\n--- Starting Dataset Split & Copy ---")
    
    # 1. Clean and Recreate Directories
    if os.path.exists(hp.YOLO_BASE_FOLDER):
        print(f"Cleaning existing data in {hp.YOLO_BASE_FOLDER}...")
        shutil.rmtree(hp.YOLO_BASE_FOLDER)
    
    subdirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for sd in subdirs:
        os.makedirs(os.path.join(hp.YOLO_BASE_FOLDER, sd), exist_ok=True)

    # 2. Gather Files
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Positives: Images that have oil spills
    pos_files = [f for f in os.listdir(hp.INPAINT_IMAGES_FOLDER) if f.lower().endswith(valid_exts)]
    
    # Negatives: Raw images (clean runways)
    # We filter out raw images that were used for inpainting to avoid data leakage,
    # OR you can just use all of them as negatives if the oil covers the original texture significantly.
    # For safety, let's just grab all raw files.
    neg_files = [f for f in os.listdir(hp.RAW_FOLDER) if f.lower().endswith(valid_exts)]

    # 3. Shuffle & Split
    if hasattr(hp, 'SEED'): random.seed(hp.SEED)
    else: random.seed(42)
    
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    pos_split_idx = int(len(pos_files) * hp.DATASET_SPLIT_RATIO)
    neg_split_idx = int(len(neg_files) * hp.DATASET_SPLIT_RATIO)

    # 4. Processing Helper
    def process_batch(file_list, source_img_folder, split_type, is_positive):
        count = 0
        dest_img_root = os.path.join(hp.YOLO_BASE_FOLDER, split_type, 'images')
        dest_lbl_root = os.path.join(hp.YOLO_BASE_FOLDER, split_type, 'labels')

        for filename in file_list:
            name_no_ext = os.path.splitext(filename)[0]
            
            # Copy Image
            src_img_path = os.path.join(source_img_folder, filename)
            shutil.copy2(src_img_path, os.path.join(dest_img_root, filename))
            
            # Handle Labels
            dest_lbl_path = os.path.join(dest_lbl_root, f"{name_no_ext}.txt")
            
            if is_positive:
                # Copy existing label
                src_lbl_path = os.path.join(hp.INPAINT_LABELS_FOLDER, f"{name_no_ext}.txt")
                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_lbl_path, dest_lbl_path)
                else:
                    print(f"Warning: Label missing for positive sample {filename}")
            else:
                # Create EMPTY label for negative sample (Background image)
                open(dest_lbl_path, 'w').close()
            
            count += 1
        return count

    # 5. Execute
    print(f"Processing {len(pos_files)} Positive and {len(neg_files)} Negative samples...")
    
    # Train Set
    n_pos_train = process_batch(pos_files[:pos_split_idx], hp.INPAINT_IMAGES_FOLDER, 'train', True)
    n_neg_train = process_batch(neg_files[:neg_split_idx], hp.RAW_FOLDER, 'train', False)
    
    # Val Set
    n_pos_val = process_batch(pos_files[pos_split_idx:], hp.INPAINT_IMAGES_FOLDER, 'val', True)
    n_neg_val = process_batch(neg_files[neg_split_idx:], hp.RAW_FOLDER, 'val', False)

    print(f"\nDataset Ready: {hp.YOLO_BASE_FOLDER}")
    print(f"Train: {n_pos_train + n_neg_train} images")
    print(f"Val:   {n_pos_val + n_neg_val} images")