import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from classes.datasetMaker import DatasetMaker
import utils
from ultralytics import YOLO
import hyperParams as hp
import torch

def main():
    # verify/creates folders that are needed but are in the .gitignore
    utils.setup_directories()
    
    # creates masks then inpaints from images in the dataset/0_raw folder
    pipeline = DatasetMaker()
    pipeline.run()
    
    # creates final YOLO dataset for finetune training
    utils.split_and_copy_dataset()
    
    
    # Check Hardware
    if torch.cuda.is_available():
        device = 0
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("WARNING: CUDA not found. Training on CPU.")
        # AMD Note: If you are using DirectML or standard Windows torch, 
        # CPU is the default fallback.

    # Load YOLOv12 Model
    # If the file doesn't exist locally, Ultralytics will attempt to download it.
    print(f"Loading {hp.MODEL_NAME}...")
    try:
        model = YOLO(hp.MODEL_NAME)
    except Exception as e:
        print(f"Error loading {hp.MODEL_NAME}. Make sure you have the latest 'ultralytics' installed.")
        print("Try running: pip install -U ultralytics")
        raise e

    # train
    print(f"Starting training with {hp.MODEL_NAME}")
    results = model.train(
        data=hp.YOLO_DATA_FILE,
        epochs=hp.EPOCHS,          
        imgsz=hp.IMAGE_SIZE,          
        batch=hp.BATCH_SIZE,
        device=device,     
        name='fod_yolov12_finetune',
        patience=10,        
        
        # Augmentations
        degrees=hp.AUG_DEGREES,
        fliplr=hp.AUG_FLIP,
        mosaic=hp.AUG_MOSAIC,
    )

    print(f"\nTraining Complete!")
    print(f"Best model saved at: runs/detect/fod_yolov12_finetune/weights/best.pt")
    

if __name__ == "__main__":
    main()