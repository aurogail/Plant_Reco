from tqdm import tqdm
import os
import numpy as np
from multiprocessing import Pool

from utils import *

if __name__ == "__main__":
    ## Parameters ##
    DELETE_LOW_QUALITY_IMAGES = params['clean']['low_quality']

    data_path = params['dataset_path']['data']
    plant_seedlings_path = params['dataset_path']['plant_seedlings']
    new_plant_diseases_path = params['dataset_path']['new_plant_diseases']
    plantvillage_path = params['dataset_path']['plantvillage']
    plantvillage_color_path = params['dataset_path']['plantvillage_color']
    plantvillage_gray_path = params['dataset_path']['plantvillage_gray']
    plantvillage_seg_path = params['dataset_path']['plantvillage_segmented']

    names = ["Plant Seedlings", "New Plant Diseases", "PlantVillage Color", "PlantVillage Grayscale", "PlantVillage Segmented"]
    paths = [plant_seedlings_path, new_plant_diseases_path, plantvillage_color_path, plantvillage_gray_path, plantvillage_seg_path]

    # Remove duplicates
    print("Removing duplicated images...")
    with Pool() as p:
        result = p.map(get_duplicates, paths)
    for name, duplicates in zip(names, result):
        for duplicate in duplicates:
            for path in duplicate[:-1]: # Keep only the image with the longest path
                os.remove(path)
        print(f"{name}: {len(duplicates)} duplicates removed")

    # Resize images
    print("Resizing images...")
    with Pool() as p:
        for name, path in zip(names, paths):
            img_paths = get_file_paths(path)
            print(f"Checking {name} images...")
            result = list(tqdm(p.imap(resize_image, img_paths), total=len(img_paths)))
            print(f"{sum(result)} images resized")

    # Remove low quality images
    def is_low_quality_seg(path):
        return is_low_quality(path=path,segmented=True)
    def is_low_quality_ps(path):
        return is_low_quality(path=path,sharpness_threshold=0)

    if DELETE_LOW_QUALITY_IMAGES:
        print("Removing low quality images...")
        with Pool() as p:
            for name, path in zip(names, paths):
                img_paths = get_file_paths(path)
                print(f"Checking {name} images...")
                if name == "PlantVillage Segmented":
                    result = list(tqdm(p.imap(is_low_quality_seg, img_paths), total=len(img_paths)))
                elif name == "Plant Seedlings":
                    result = list(tqdm(p.imap(is_low_quality_ps, img_paths), total=len(img_paths)))
                else:
                    result = list(tqdm(p.imap(is_low_quality, img_paths), total=len(img_paths)))
                low_quality_img_paths = np.array(img_paths)[result]
                for path in low_quality_img_paths:
                    os.remove(path)
                print(f"{name}: {len(low_quality_img_paths)} low quality images removed ({len(low_quality_img_paths)/len(img_paths)*100:.2f}%)")

    print("Done!")