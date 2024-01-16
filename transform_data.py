from utils import *

if __name__ == "__main__":
    ## PARAMETERS ##

    random_state = params['split']['random_state']
    train_size = params['split']['train_size']
    test_size = params['split']['test_size']

    plant_seedlings_path = params['dataset_path']['plant_seedlings']
    new_plant_diseases_path = params['dataset_path']['new_plant_diseases']
    plantvillage_color_path = params['dataset_path']['plantvillage_color']
    plantvillage_gray_path = params['dataset_path']['plantvillage_gray']
    plantvillage_seg_path = params['dataset_path']['plantvillage_segmented']

    # Split train/test/valid
    print("Splitting datasets into train/test/valid directories...")

    if params['split']['plant_seedlings'] and "train" not in os.listdir(plant_seedlings_path):
        print("Reorganize Plant Seedlings into train/test/valid directories...")
        split_dataset(plant_seedlings_path, train_size=train_size, test_size=test_size, random_state=random_state)
    if params['split']['new_plant_diseases'] and "train" not in os.listdir(new_plant_diseases_path):
        print("Reorganize New Plant Diseases into train/test/valid directories...")
        split_dataset(new_plant_diseases_path, train_size=train_size, test_size=test_size, random_state=random_state)
    if params['split']['plantvillage_color'] and "train" not in os.listdir(plantvillage_color_path):
        print("Reorganize PlantVillage Color into train/test/valid directories...")
        split_dataset(plantvillage_color_path, train_size=train_size, test_size=test_size, random_state=random_state)
    if params['split']['plantvillage_gray'] and "train" not in os.listdir(plantvillage_gray_path):
        print("Reorganize PlantVillage Grayscale into train/test/valid directories...")
        split_dataset(plantvillage_gray_path, train_size=train_size, test_size=test_size, random_state=random_state)
    if params['split']['plantvillage_segmented'] and "train" not in os.listdir(plantvillage_seg_path):
        print("Reorganize PlantVillage Segmented into train/test/valid directories...")
        split_dataset(plantvillage_seg_path, train_size=train_size, test_size=test_size, random_state=random_state)
