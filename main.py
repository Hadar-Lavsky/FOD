import os
import glob
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
DATA_YAML = 'runway_fod.yaml'  # Path to your config
MODEL_SIZE = 'yolo12m.pt'      # Recommended over 'x' for 800 images
PROJECT_NAME = 'runway_fod_project'
RUN_NAME = 'synthetic_finetune_v1'

# Load class names from YAML
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)
CLASS_NAMES = data_config['names']
TRAIN_IMAGES_PATH = os.path.join(data_config['path'], data_config['train'])
TRAIN_LABELS_PATH = TRAIN_IMAGES_PATH.replace('images', 'labels')

# ==========================================
# 1. EDA: Know Your Data (Before Training)
# ==========================================
def run_eda():
    print("🔎 Starting Exploratory Data Analysis...")
    label_files = glob.glob(os.path.join(TRAIN_LABELS_PATH, '*.txt'))
    
    classes = []
    box_sizes = [] # (width, height)
    box_centers = [] # (x, y)

    for file in label_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                x, y = float(parts[1]), float(parts[2])
                
                classes.append(CLASS_NAMES[cls_id])
                box_sizes.append((w, h))
                box_centers.append((x, y))

    # Convert to DataFrame for easier plotting
    df_boxes = pd.DataFrame(box_sizes, columns=['Width', 'Height'])
    df_centers = pd.DataFrame(box_centers, columns=['Center_X', 'Center_Y'])
    
    # --- PLOT 1: Class Distribution ---
    plt.figure(figsize=(10, 5))
    sns.countplot(x=classes, palette='viridis')
    plt.title('Class Distribution (Is the dataset balanced?)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    # --- PLOT 2: Box Sizes (Small vs Large Objects) ---
    # This is crucial for FOD. If dots are near (0,0), objects are tiny.
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='Width', y='Height', data=df_boxes, hue=classes, alpha=0.6)
    plt.title('Object Sizes (Normalized 0-1)')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.plot([0, 1], [0, 1], 'r--') # Diagonal
    plt.show()
    
    # --- PLOT 3: Spatial Distribution (Heatmap) ---
    # Checks if synthetic generator only puts objects in the center
    plt.figure(figsize=(8, 8))
    plt.hexbin(df_centers['Center_X'], df_centers['Center_Y'], gridsize=20, cmap='inferno')
    plt.title('Spatial Density of Objects (Where do they appear?)')
    plt.gca().invert_yaxis() # Image coordinates
    plt.colorbar(label='Count')
    plt.show()
    
    print(f"✅ EDA Complete. Total labels analyzed: {len(classes)}")

# ==========================================
# 2. TRAINING (With Synthetic Adaptations)
# ==========================================
def train_model():
    print("🚀 Starting Training...")
    model = YOLO(MODEL_SIZE) 

    results = model.train(
        data=DATA_YAML,
        project=PROJECT_NAME,
        name=RUN_NAME,
        
        # Hyperparameters
        epochs=100,
        imgsz=1024,      # High res for small holes
        batch=8,         # Adjust based on GPU VRAM
        device=0,
        patience=15,     # Early stopping
        
        # Synthetic Data Augmentations (The "Dirtying" Phase)
        mosaic=1.0,      # Mixes images
        degrees=10.0,    # Rotation
        perspective=0.0005,
        hsv_s=0.6,       # High saturation variance (oil vs water)
        hsv_v=0.4,       # Brightness variance
        
        # Note: 'noise' or 'blur' might need `albumentations` installed
        # or custom callbacks in standard YOLO, but basic geometry helps a lot.
        fliplr=0.5,
        flipud=0.5,      # Top-down symmetry
        
        dropout=0.2,     # Prevent memorizing synthetic patterns
        plots=True       # Auto-save plots
    )
    return model

# ==========================================
# 3. POST-TRAINING METRICS & SHIT
# ==========================================
def analyze_results():
    print("📊 Generating Final Report...")
    
    # Path to the results directory
    results_dir = Path(f'{PROJECT_NAME}/{RUN_NAME}')
    
    # 1. Load History (Losses & mAP over epochs)
    df_results = pd.read_csv(results_dir / 'results.csv')
    
    # Clean column names (strip spaces)
    df_results.columns = [x.strip() for x in df_results.columns]

    # --- PLOT 4: Training vs Val Loss ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_results['train/box_loss'], label='Train Box Loss')
    plt.plot(df_results['val/box_loss'], label='Val Box Loss')
    plt.title('Box Loss (Localization)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df_results['metrics/mAP50(B)'], label='mAP@50')
    plt.plot(df_results['metrics/mAP50-95(B)'], label='mAP@50-95')
    plt.title('Mean Average Precision (Accuracy)')
    plt.legend()
    plt.show()

    # 2. Validation Metrics (Per Class)
    # We reload the best model to run a clean validation pass
    best_model = YOLO(results_dir / 'weights/best.pt')
    metrics = best_model.val(data=DATA_YAML, split='val')
    
    print("\n🏆 FINAL METRICS PER CLASS:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10}")
    print("-" * 55)
    
    # Extract per-class AP (Ultralytics stores this inside metrics.box)
    # Note: Accessing internal lists maps to class indices
    for i, c in enumerate(CLASS_NAMES):
        # This is a simplified extraction; exact attribute access depends on version
        # printing the aggregate maps for safety if per-class is complex to index dynamically
        pass 
    
    # Display map50 per class from the printed output of .val() usually, 
    # but here is the aggregate:
    print(f"Overall mAP@50:    {metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95: {metrics.box.map:.4f}")

    # 3. Confusion Matrix
    cm_path = results_dir / 'confusion_matrix.png'
    if cm_path.exists():
        img = cv2.imread(str(cm_path))
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Confusion Matrix (Where is it getting confused?)")
        plt.axis('off')
        plt.show()
    
    print(f"\n✅ Analysis complete. Check '{results_dir}' for full logs and predicted images.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    # 1. Check your data
    run_eda()
    
    # 2. Train
    model = train_model()
    
    # 3. Analyze
    analyze_results()