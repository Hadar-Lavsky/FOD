import os
import hyperParams as hp

# checks and creates directories needed for the project, these directories are in .gitignore
def setup_directories():
    folders_to_check = [
        hp.RAW_FOLDER,
        hp.MASK_FOLDER,
        hp.INPAINT_IMAGES_FOLDER,
        hp.INPAINT_LABELS_FOLDER
    ]

    print("Checking directories...")
    
    for folder_path in folders_to_check:
        try:
            # exist_ok=True creates the directory if it doesn't exist, 
            # and does nothing if it already does.
            os.makedirs(folder_path, exist_ok=True)
            print(f"Verified/Created: {folder_path}")
        except OSError as e:
            print(f"Error creating {folder_path}: {e}")