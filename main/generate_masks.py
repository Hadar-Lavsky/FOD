import os
import random
import cv2
import numpy as np
from PIL import Image

# ==========================================
#              HYPERPARAMETERS
# ==========================================

# INPUT CONFIGURATION
INPUT_IMAGES_DIR = "./dataset/MFS/raw_images"  # Folder with your runway images
INPUT_LABELS_DIR = "./dataset/MFS/raw_images_bounding_boxes"  # Files with your generic "Class 0" boxes

# OUTPUT CONFIGURATION
OUTPUT_MASKS_DIR = "./dataset/MFS/raw_images_masks"   # Will contain the generated mask files
OUTPUT_LABELS_DIR = "./dataset/MFS/raw_images_labels" # Will contain the new combined file with updated Class IDs

# CLASS CONFIGURATION
# Format: {Class_ID: 'suffix_name'}
# The script will generate one mask file per entry in this dictionary.
CLASS_MAPPING = {
    0: 'oil',
    1: 'water',
    2: 'ice',
    3: 'hole'
}

# MASK SETTINGS
MASK_FORMAT = ".jpg" # Output format for masks
MASK_QUALITY = 100   # JPEG quality (0-100), use 100 to prevent compression artifacts

# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# reads yolo file and returns list of boxes
# format of each box: [class, x_center, y_center, width, height]
def parse_yolo_file(file_path):
    boxes = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(x) for x in parts])
    return boxes

# saves list of boxes to yolo file
def save_yolo_file(boxes, output_path):
    with open(output_path, 'w') as f:
        for b in boxes:
            # Save as: class x y w h
            line = f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n"
            f.write(line)

# creates a mask for a specific class
def create_mask_for_class(image_size, boxes, target_class):
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    boxes_drawn = 0
    
    for box in boxes:
        # Check if this box belongs to the class we are currently drawing
        if int(box[0]) == target_class:
            # Convert YOLO -> Pixel
            bw = box[3] * w
            bh = box[4] * h
            bx = (box[1] * w) - (bw / 2)
            by = (box[2] * h) - (bh / 2)

            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            boxes_drawn += 1
            
    return mask, boxes_drawn

def main():
    ensure_dir(OUTPUT_MASKS_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images. Processing...")

    for img_file in image_files:
        # Setup paths
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(INPUT_LABELS_DIR, label_file)
        img_path = os.path.join(INPUT_IMAGES_DIR, img_file)

        if not os.path.exists(label_path):
            continue

        # 1. Get Image Size
        with Image.open(img_path) as img:
            w, h = img.size

        # 2. Load Original Boxes (Class 0 placeholders)
        original_boxes = parse_yolo_file(label_path)
        if not original_boxes:
            continue

        # 3. Randomize Classes
        # We create a NEW list of boxes with updated Class IDs
        updated_boxes = []
        possible_classes = list(CLASS_MAPPING.keys()) # [0, 1, 2, 3]

        for box in original_boxes:
            # Keep coords (indices 1-4), replace class (index 0)
            new_class = random.choice(possible_classes)
            new_box = [new_class] + box[1:]
            updated_boxes.append(new_box)

        # 4. Save the ONE master label file
        save_label_path = os.path.join(OUTPUT_LABELS_DIR, label_file)
        save_yolo_file(updated_boxes, save_label_path)

        # 5. Generate the 4 Mask Files (One per class type)
        for class_id, suffix in CLASS_MAPPING.items():
            
            mask_img, count = create_mask_for_class((w, h), updated_boxes, class_id)
            
            # Save path: e.g., output_masks/image01_mask_oil.jpg
            mask_filename = f"{base_name}_mask_{suffix}{MASK_FORMAT}"
            mask_save_path = os.path.join(OUTPUT_MASKS_DIR, mask_filename)
            
            # Write the mask image
            # Note: Even if count is 0 (black mask), we save it so the next script 
            # doesn't crash looking for missing files.
            cv2.imwrite(mask_save_path, mask_img, [cv2.IMWRITE_JPEG_QUALITY, MASK_QUALITY])

        print(f"Processed {img_file}: Saved labels and {len(CLASS_MAPPING)} masks.")

if __name__ == "__main__":
    main()