RAW_FOLDER = "./dataset/0_raw"
MASK_FOLDER = "./dataset/1_mask"
INPAINT_IMAGES_FOLDER = "./dataset/2_inpaint/images"
INPAINT_LABELS_FOLDER = "./dataset/2_inpaint/labels"
YOLO_BASE_FOLDER = "./dataset/final"

YOLO_DATA_FILE = 'data.yaml'

MODEL_NAME = 'yolov12n.pt' 

DATASET_SPLIT_RATIO = 0.8       # use 80% train, 20% val

# Training
EPOCHS = 50
SEED = 42                       # random seed
IMAGE_SIZE = 640
BATCH_SIZE = 16

# Augmentations
AUG_DEGREES = 15.0
AUG_FLIP = 0.5
AUG_MOSAIC = 1.0