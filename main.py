import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from classes.datasetMaker import DatasetMaker
import utils

def main():
    # verify/creates folders that are needed but are in the .gitignore
    utils.setup_directories()
    
    # creates masks then inpaints from images in the dataset/0_raw folder
    pipeline = DatasetMaker()
    pipeline.run()
    

if __name__ == "__main__":
    main()