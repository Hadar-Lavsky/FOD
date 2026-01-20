import os
import glob
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# --- IMPORT SETTINGS ---
import hyperParams as hp

# ==========================================
# 0. CONFIG FIXES (Speed & Stability)
# ==========================================
hp.MODEL_NAME = 'yolov8n.pt'   # Use v8 Nano for speed & auto-download
hp.YOLO_DATA_FILE = 'runway_fod.yaml'

# Path setup
BASE_DATA_PATH = os.path.join('dataset', 'runway_dataset')
TRAIN_LABELS_DIR = os.path.join(BASE_DATA_PATH, 'labels', 'train')

print(f"🔧 CONFIG: Using '{hp.YOLO_DATA_FILE}'")
print(f"🚀 MODEL:  {hp.MODEL_NAME}")

# ==========================================
# 1. EDA: Know Your Data
# ==========================================
def run_eda():
    print(f"🔎 Starting EDA...")
    label_files = glob.glob(os.path.join(TRAIN_LABELS_DIR, '*.txt'))
    
    if not label_files:
        print(f"❌ No labels found in {TRAIN_LABELS_DIR}. Skipping EDA.")
        return

    # Load class names
    try:
        with open(hp.YOLO_DATA_FILE, 'r') as f:
            d = yaml.safe_load(f)
            names = list(d.get('names', {}).values())
    except:
        names = ['0', '1', '2']

    classes_found = []
    box_sizes = []

    for file in label_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # Safe name lookup
                    c_name = names[cls_id] if cls_id < len(names) else str(cls_id)
                    classes_found.append(c_name)
                    # Width/Height
                    box_sizes.append((float(parts[3]), float(parts[4])))

    if classes_found:
        # Plot 1: Class Count
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.countplot(x=classes_found, hue=classes_found, palette='viridis', legend=False)
        plt.title('Class Distribution')

        # Plot 2: Box Sizes
        df = pd.DataFrame(box_sizes, columns=['W', 'H'])
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='W', y='H', data=df, hue=classes_found, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.title('Object Size (Normalized)')
        
        plt.tight_layout()
        plt.show()
        print(f"✅ EDA Complete. Analyzed {len(classes_found)} objects.")

# ==========================================
# 2. TRAINING
# ==========================================
def train_model():
    print(f"🚀 Starting Training...")
    model = YOLO(hp.MODEL_NAME) 

    # We name the project 'runway_project' and run 'train_run'
    # This makes finding results easy: runway_project/train_run
    results = model.train(
        data=hp.YOLO_DATA_FILE,
        project='runway_project',
        name='train_run',
        
        # Hyperparams
        epochs=hp.EPOCHS,
        imgsz=640,
        batch=16,
        
        # Speed & Stability
        device=0,
        workers=8,
        optimizer='AdamW',
        cache=True,
        amp=True,
        plots=True,      # CRITICAL: Generates the graphs we want
        patience=10,
        exist_ok=True    # Overwrite old run (saves disk space)
    )
    return model

# ==========================================
# 3. ADVANCED EVALUATION (New Graphs!)
# ==========================================
def display_image_grid(paths, title):
    """Helper to display a list of image paths in a grid"""
    if not paths: return
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16)
    for i, p in enumerate(paths[:3]): # Show max 3 images
        if os.path.exists(p):
            img = Image.open(p)
            plt.subplot(1, 3, i+1)
            plt.imshow(img)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_results():
    print("\n📊 STARTING ADVANCED ANALYSIS...")
    results_dir = Path('runway_project/train_run')
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # --- 1. LEARNING CURVES (Loss & mAP) ---
    csv_path = results_dir / 'results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [x.strip() for x in df.columns]
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: mAP (Accuracy)
        plt.subplot(1, 2, 1)
        plt.plot(df['metrics/mAP50(B)'], label='mAP@50', color='blue')
        plt.plot(df['metrics/mAP50-95(B)'], label='mAP@50-95', color='green')
        plt.title('Accuracy (Higher is better)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Box Loss (Precision of location)
        plt.subplot(1, 2, 2)
        plt.plot(df['train/box_loss'], label='Train Loss', color='orange')
        plt.plot(df['val/box_loss'], label='Val Loss', color='red')
        plt.title('Box Loss (Lower is better)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.show()

    # --- 2. CONFUSION MATRIX ---
    # Shows if "Oil" is being mistaken for "Crack"
    cm_path = results_dir / 'confusion_matrix_normalized.png'
    if cm_path.exists():
        print("🔹 Confusion Matrix (Normalized):")
        plt.figure(figsize=(8, 8))
        plt.imshow(Image.open(cm_path))
        plt.axis('off')
        plt.show()
    
    # --- 3. F1-CONFIDENCE CURVE ---
    # Helps you decide the confidence threshold for deployment
    f1_path = results_dir / 'F1_curve.png'
    if f1_path.exists():
        print("🔹 F1-Confidence Curve (Balance of Precision/Recall):")
        plt.figure(figsize=(8, 6))
        plt.imshow(Image.open(f1_path))
        plt.axis('off')
        plt.show()

    # --- 4. VISUAL PREDICTIONS (Ground Truth vs Prediction) ---
    print("🔹 Visual Validation (Truth vs Prediction):")
    
    # Validation Batch (Ground Truth)
    val_truth = sorted(list(results_dir.glob('val_batch*_labels.jpg')))
    display_image_grid(val_truth, "Ground Truth (Actual Labels)")

    # Validation Batch (Model Predictions)
    val_pred = sorted(list(results_dir.glob('val_batch*_pred.jpg')))
    display_image_grid(val_pred, "Model Predictions (What the AI saw)")

    print(f"✅ Full report saved to: {results_dir.absolute()}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    run_eda()
    try:
        model = train_model()
        analyze_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")