import kaggle
import shutil
import os
from utils import *

if __name__ == "__main__":

    ## PARAMETERS ##

    data_path = params['dataset_path']['data']
    plant_seedlings_path = params['dataset_path']['plant_seedlings']
    new_plant_diseases_path = params['dataset_path']['new_plant_diseases']
    plantvillage_path = params['dataset_path']['plantvillage']

    # Create the data directory if not exists
    os.makedirs(data_path)

    # Download the dataset from Kaggle
    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('vbookshelf/v2-plant-seedlings-dataset', path=plant_seedlings_path, unzip=True, quiet=False)
    kaggle.api.dataset_download_files('vipoooool/new-plant-diseases-dataset', path=new_plant_diseases_path, unzip=True, quiet=False)
    kaggle.api.dataset_download_files('abdallahalidev/plantvillage-dataset', path=plantvillage_path, unzip=True, quiet=False)


    # Rearranging the directories
    print("Rearranging", plant_seedlings_path, "...")
    # Rename the folder to remove non alphanumeric characters
    for dir in os.listdir(plant_seedlings_path):
        if dir.startswith("Shepherd"):
            os.rename(os.path.join(plant_seedlings_path, dir), os.path.join(plant_seedlings_path, 'Shepherds Purse'))
            break

    # Move nonsegmentedv2 to the data directory
    shutil.move(os.path.join(plant_seedlings_path, 'nonsegmentedv2'), data_path)
    for dir in os.listdir(os.path.join(data_path, 'nonsegmentedv2')):
        if dir.startswith("Shepherd"):
            os.rename(os.path.join(os.path.join(data_path, 'nonsegmentedv2'), dir),
                    os.path.join(os.path.join(data_path, 'nonsegmentedv2'), 'Shepherds Purse'))
            break
    diff1, diff2 = get_differences(os.path.join(data_path, 'nonsegmentedv2'), plant_seedlings_path)
    assert len(diff1) == 0 and len(diff2) == 0 # Check that there is no difference between the two directories

    # Remove the nonsegmentedv2 directory
    shutil.rmtree(os.path.join(data_path, 'nonsegmentedv2'))

    print("Rearranging", new_plant_diseases_path, "...")

    new_plant_diseases_train_path = os.path.join(new_plant_diseases_path, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)','train')
    new_plant_diseases_valid_path = os.path.join(new_plant_diseases_path, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)','valid')
    new_plant_diseases_test_path = os.path.join(new_plant_diseases_path, 'test','test')

    classes_train = sorted(os.listdir(new_plant_diseases_train_path))
    classes_valid = sorted(os.listdir(new_plant_diseases_valid_path))
    assert classes_train == classes_valid # Check that the classes are the same in train and valid

    # Assert that new plant diseases (augmented) dataset equals to New Plant Diseases Dataset(Augmented)
    New_Plant_Diseases_train_path = os.path.join(new_plant_diseases_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)','train')
    New_Plant_Diseases_valid_path = os.path.join(new_plant_diseases_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)','valid')

    assert sorted(os.listdir(New_Plant_Diseases_train_path)) == classes_train
    assert sorted(os.listdir(New_Plant_Diseases_valid_path)) == classes_train # Check that the classes are the same in train and valid

    diff1, diff2 = get_differences(New_Plant_Diseases_train_path, new_plant_diseases_train_path)
    assert len(diff1) == 0 and len(diff2) == 0
    diff1, diff2 = get_differences(New_Plant_Diseases_valid_path, new_plant_diseases_valid_path)
    assert len(diff1) == 0 and len(diff2) == 0 # Check that there is no difference between the two directories
    # => New Plant Diseases Dataset(Augmented) is the same as new plant diseases dataset(augmented)

    cpt_train = nb_files(new_plant_diseases_train_path)
    cpt_valid = nb_files(new_plant_diseases_valid_path)
    cpt_test = len(os.listdir(new_plant_diseases_test_path))

    # Move images from train and valid to the corresponding class folder
    for dir in classes_train:
        os.mkdir(os.path.join(new_plant_diseases_path, dir))
        for image in os.listdir(os.path.join(new_plant_diseases_train_path, dir)):
            shutil.move(os.path.join(new_plant_diseases_train_path, dir, image), os.path.join(new_plant_diseases_path, dir, image))
        for image in os.listdir(os.path.join(new_plant_diseases_valid_path, dir)):
            shutil.move(os.path.join(new_plant_diseases_valid_path, dir, image), os.path.join(new_plant_diseases_path, dir, image))

    # Move images from test to the corresponding class folder
    images_test = os.listdir(new_plant_diseases_test_path)
    for image in images_test:
        if image.startswith('AppleCedarRust'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Apple___Cedar_apple_rust'))
        elif image.startswith('AppleScab'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Apple___Apple_scab'))
        elif image.startswith('CornCommonRust'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Corn_(maize)___Common_rust_'))
        elif image.startswith('PotatoEarlyBlight'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Potato___Early_blight'))
        elif image.startswith('PotatoHealthy'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Potato___healthy'))
        elif image.startswith('TomatoEarlyBlight'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Tomato___Early_blight'))
        elif image.startswith('TomatoHealthy'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Tomato___healthy'))
        elif image.startswith('TomatoYellowCurlVirus'):
            shutil.move(os.path.join(new_plant_diseases_test_path, image), os.path.join(new_plant_diseases_path, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'))
        else:
            raise Exception("Unknown class for image", image)

    shutil.rmtree(os.path.join(new_plant_diseases_path, 'New Plant Diseases Dataset(Augmented)'))
    shutil.rmtree(os.path.join(new_plant_diseases_path, 'new plant diseases dataset(augmented)'))
    shutil.rmtree(os.path.join(new_plant_diseases_path, 'test'))

    assert cpt_train + cpt_valid + cpt_test == nb_files(new_plant_diseases_path) # Check that there is no image lost or overwrited

    print("Rearranging", plantvillage_path, "...")

    folder = os.path.join(plantvillage_path, os.listdir(plantvillage_path)[0])
    for dir in os.listdir(folder):
        shutil.move(os.path.join(folder, dir), plantvillage_path)
    os.rmdir(folder)

    print("Done !")