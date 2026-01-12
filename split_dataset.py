import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # pip install tqdm if you don't have it, or remove this

# --- CONFIGURATION ---
# Where your raw mixed files are currently
SOURCE_DIR = Path("dataset/MFS/processed") 

# Where we want the clean YOLO structure
DEST_DIR = Path("dataset/runway_dataset")

# Split ratio (0.8 = 80% train, 20% val)
TRAIN_RATIO = 0.8 

def setup_dirs():
    """Creates the standard YOLO directory structure."""
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            (DEST_DIR / dtype / split).mkdir(parents=True, exist_ok=True)

def split_data():
    if not SOURCE_DIR.exists():
        print(f"❌ Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print(f"📂 Scanning {SOURCE_DIR}...")
    
    # Get all unique file stems (filenames without extensions)
    # This ensures we handle .jpg, .png, etc. correctly
    files = os.listdir(SOURCE_DIR)
    stems = list(set([os.path.splitext(f)[0] for f in files if not f.startswith('.')]))
    
    # Filter only stems that have BOTH an image and a label
    # (Important for synthetic data where generation might fail occasionally)
    valid_stems = []
    supported_imgs = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for stem in stems:
        has_img = any((SOURCE_DIR / (stem + ext)).exists() for ext in supported_imgs)
        has_lbl = (SOURCE_DIR / (stem + ".txt")).exists()
        
        if has_img and has_lbl:
            valid_stems.append(stem)
    
    print(f"✅ Found {len(valid_stems)} valid image/label pairs.")
    
    # Shuffle and Split
    random.seed(42) # For reproducibility
    random.shuffle(valid_stems)
    
    split_idx = int(len(valid_stems) * TRAIN_RATIO)
    train_stems = valid_stems[:split_idx]
    val_stems = valid_stems[split_idx:]
    
    print(f"🚀 Splitting: {len(train_stems)} Train / {len(val_stems)} Val")
    
    # Helper to copy files
    def copy_files(stem_list, split_name):
        for stem in tqdm(stem_list, desc=f"Copying {split_name}"):
            # Find the image file (whatever extension it has)
            img_file = next(f for f in files if f.startswith(stem) and os.path.splitext(f)[1] in supported_imgs)
            lbl_file = stem + ".txt"
            
            # Copy Image
            shutil.copy2(SOURCE_DIR / img_file, DEST_DIR / 'images' / split_name / img_file)
            # Copy Label
            shutil.copy2(SOURCE_DIR / lbl_file, DEST_DIR / 'labels' / split_name / lbl_file)

    setup_dirs()
    copy_files(train_stems, 'train')
    copy_files(val_stems, 'val')
    
    print(f"\n✨ Dataset ready at: {DEST_DIR.resolve()}")
    print("Don't forget to update the 'path' in your YAML file!")

if __name__ == '__main__':
    split_data()