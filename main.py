from classes.maskMaker import MaskMaker
from classes.inpaintMaker import InpaintMaker
import hyperParams as hp

def main():
    mask_engine = MaskMaker(device_name="auto")
    inpaint_engine= InpaintMaker(device_name="auto")

    ## Create masks for images
    # mask_engine.process_batch(hp.RAW_FOLDER, hp.MASK_FOLDER)
    
    ## Inpaint oil spills on images based on masks
    # inpaint_engine.process_batch(masks_folder = hp.MASK_FOLDER, image_folder = hp.RAW_FOLDER, output_image_folder = hp.INPAINT_IMAGES_FOLDER, output_label_folder = hp.INPAINT_LABELS_FOLDER)

if __name__ == "__main__":
    main()