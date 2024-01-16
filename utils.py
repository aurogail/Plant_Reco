import os
import hashlib
import random
import cv2
import numpy as np
# import pandas as pd # Debug
import re
import shutil
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import toml

#######################
 ## FILE MANAGEMENT ##
#######################

def load_params():
    with open('params.toml') as f:
        params = toml.load(f)
    return params

params = load_params()

def nb_files(directory_path):
    """
    Counts the number of files in the given dataset path
    """
    return sum(len(files) for _, _, files in os.walk(directory_path))

def get_file_paths(path):
    """
    Returns a list of image paths from the given dataset path
    """
    return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]

def get_random_img_path(path):
    """
    Get a random image path from the given dataset path
    """
    return random.choice(get_file_paths(path))

def md5(path):
    """
    Returns the md5 hash of a file (string of hexadecimal digits)

    Each file has a unique md5 hash, so we can use this to check if two files are the same
    """
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_hash_table_from_path(paths):
    """
    Returns the hash table of a list of paths
    """
    hashes = dict()
    for path in paths:
        hash = md5(path)
        if hash not in hashes:
            hashes[hash] = [path]
        else:
            hashes[hash].append(path)
    return hashes


def get_hash_table(path):
    """
    Returns the hash table of a folder
    """
    paths = get_file_paths(path)
    return get_hash_table_from_path(paths)


def get_duplicates(path1, path2=None):
    """
    Returns intersection of two datasets

    If path2 is None, then returns duplicates of path1
    """

    # hashes : {hash: [path, path, ...]}
    # collisions : [[path1, path2, ...], [path1, path2, ...], ...]
    # A collision appears when two files have the same hash, in that case the value of hashes[hash] has a length > 1

    hashes1 = get_hash_table(path1)
    if path2 is None:
        collisions = [hashes1[hash] for hash in hashes1 if len(hashes1[hash]) > 1]
    else:
        hashes2 = get_hash_table(path2)
        collisions = [hashes1[hash] + hashes2[hash] for hash in hashes1 if hash in hashes2]
    return [sorted(collision, key=len) for collision in collisions]


def get_differences(path1, path2, duplicates=True):
    """
    Returns differences of two datasets

    diff1 : Images that are in path1 but not in path2
    diff2 : Images that are in path2 but not in path1
    duplicates : Return duplicates or not
    Structured: same as get_duplicates
    """
    def check_diff(hashes1, hashes2, duplicates):
        # Check differences
        diff = []
        for hash in hashes1:
            if hash not in hashes2:
                if duplicates:
                    for path in hashes1[hash]:
                        diff.append(path)
                else:
                    # Add only the longest path
                    diff.append(sorted(hashes1[hash], key=len)[-1])
        return diff

    hashes1 = get_hash_table(path1)
    hashes2 = get_hash_table(path2)

    # Check differences
    diff1 = check_diff(hashes1, hashes2, duplicates)
    diff2 = check_diff(hashes2, hashes1, duplicates)
    return diff1, diff2


def split_dataset(dataset_path, train_size=0.8, test_size=0.5, random_state=42):

    def create_class(class_path, paths):
        os.mkdir(os.path.join(class_path)) # Create the class directory
        for path in paths: # Populate the class directory
            # Replace the path to the dataset by the path to the train/test/valid directory
            newpath = os.path.join(class_path, os.path.basename(path)) # +1 to remove the slash
            shutil.move(path, newpath)
        return

    classes = sorted(os.listdir(dataset_path))

    dataset_train_path = os.path.join(dataset_path, 'train')
    dataset_test_path = os.path.join(dataset_path, 'test')
    os.mkdir(dataset_train_path)
    os.mkdir(dataset_test_path)
    if test_size != 1:
        dataset_valid_path = os.path.join(dataset_path, 'valid')
        os.mkdir(dataset_valid_path)

    for class_name in classes:
        # Get paths
        class_path = os.path.join(dataset_path, class_name)
        img_paths = get_file_paths(class_path)
        X = np.array(img_paths)
        y = np.array([class_name]*len(img_paths))
        # Split data
        X_train, X_temp, _, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state)
        if test_size != 1:
            X_valid, X_test, _, _ = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)
        else:
            X_test = X_temp

        # Populate Train/Test/Valid
        create_class(os.path.join(dataset_train_path, class_name), X_train)
        create_class(os.path.join(dataset_test_path, class_name), X_test)
        if test_size != 1:
            create_class(os.path.join(dataset_valid_path, class_name), X_valid)
        # Remove the old class directory
        assert len(os.listdir(class_path)) == 0
        shutil.rmtree(class_path)


def move_elements(path1, path2):
    """
    Move all elements from path1 to path2
    """
    for element in os.listdir(path1):
        oldpath = os.path.join(path1, element)
        newpath = os.path.join(path2, element)
        if os.path.exists(newpath) and os.path.isdir(newpath):
            move_elements(oldpath, newpath)
        else:
            shutil.move(os.path.join(path1, element), path2)


def unsplit_dataset(dataset_path):
    dirs = os.listdir(dataset_path)

    for dir in dirs:
        move_elements(os.path.join(dataset_path, dir), dataset_path)
        shutil.rmtree(os.path.join(dataset_path, dir))

##############################
 ##     Image analysis     ##
##############################

def exposure_score(path=None, img=None):
    """
    Returns an exposure score of a grayscale image
    Higher is the score, brighter is the image
    score in [0, 255]
    (Experimental)
    """
    assert path is not None or img is not None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.ravel()
    return int(np.mean(img[img!=0]))


# def exposure_score1(path=None, img=None):
#     """
#     Returns an exposure score of an HSV image
#     Higher is the score, brighter is the image
#     score in [0, 255]
#     Approximatively equals to the exposure_score
#     """
#     assert path is not None or img is not None
#     if img is None:
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
#     _, _, v = cv2.split(img)
#     return int(v[v>20].mean())


def pixel_ratio(path=None, img=None):
    """
    Returns the percentage of non black pixels of a grayscale image
    """
    assert path is not None or img is not None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.ravel()
    return np.count_nonzero(img) / img.shape[0]


# def plant_ratio1(path=None, img=None):
#     """
#     Returns the leaf ratio of an HSV segmented image
#     Pretty much equivalent to plant_ratio
#     """
#     assert path is not None or img is not None
#     if img is None:
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
#     _, _, v = cv2.split(img)
#     v = v.ravel()
#     return v[v>10].shape[0] / v.shape[0]


def sharpness_score(path=None, img=None):
    """
    Returns a sharpness score using the Laplacian variance
    Higher is the variance, sharper is the image

    threshold: min 300 for an average image sharpness
    """
    assert path is not None or img is not None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return int(cv2.Laplacian(img, cv2.CV_64F).var())


def contrast_score(path=None, img=None):
    """
    Returns a contrast score of a grayscale image
    using Michelson contrast.
    Higher is the score, better is the contrast
    Returns a score in [0, 1]
    """
    assert path is not None or img is not None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.ravel()
    min_val = np.min(img) # type : np.uint8
    max_val = np.max(img)
    if min_val == 0 and max_val == 0:
        return 0
    max_val = np.float64(max_val) # Convert to float64 to avoid overflow
    min_val = np.float64(min_val) # Convert to float64 to avoid overflow

    return (max_val - min_val) / (max_val + min_val)


def is_low_quality(path=None, img=None, segmented=False, sharpness_threshold=7, exposure_threshold=30, contrast_threshold=0.1, pixel_ratio_threshold=0.1):
    # Check if image is low quality
    assert path is not None or img is not None
    if img is None:
        img =cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if sharpness_score(img=img) < sharpness_threshold:
        return True
    if exposure_score(img=img) < exposure_threshold:
        return True
    if contrast_score(img=img) < contrast_threshold:
        return True
    if segmented and pixel_ratio(img=img) < pixel_ratio_threshold:
        return True
    return False


##############################
 ##       Image fix        ##
##############################


def resize_image(image_path, size=(256,256,3)):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    b = image.shape != size
    if b:
        image = cv2.resize(image, size[:2])
        cv2.imwrite(image_path, image)
    return b

###########################
 # Augmented & Segmented #
###########################

def get_augmented_suffix(path):
    """
    Returns the suffix of an augmented image

    Hypothesis: If a suffix is present more than 10 times, then it is an augmented image prefix
    """
    threshold = 10
    augmented_suffix = []
    suffixes = dict()
    directories = os.listdir(path)
    for directory in directories:
        images = os.listdir(os.path.join(path, directory))
        for image in images:
            suffix = image.split("_")[-1][:-4] # get last annotation and remove extension of 3 characters
            if suffix in suffixes:
                suffixes[suffix] += 1
            else:
                suffixes[suffix] = 1

    for suffix in suffixes:
        if suffixes[suffix] > threshold:
            augmented_suffix.append(suffix)
    # Debug
    # df = pd.DataFrame.from_dict(suffixes, orient='index', columns=['count'])
    # df = df.sort_values(by=['count'], ascending=False)
    # df = df[df['count'] > threshold]
    # print(df)
    return augmented_suffix

# Global variable #
if os.path.exists(params['dataset_path']['new_plant_diseases']):
    AUGMENTED_SUFFIX = get_augmented_suffix(params['dataset_path']['new_plant_diseases'])
else:
    AUGMENTED_SUFFIX = None

def is_augmented(path):
    """
    Returns True if the image is an augmented image
    """
    # Check if the global variable is initialized
    global AUGMENTED_SUFFIX
    if AUGMENTED_SUFFIX == None:
        AUGMENTED_SUFFIX = get_augmented_suffix(params['dataset_path']['new_plant_diseases'])

    suffix = path.split("_")[-1][:-4]
    return suffix in AUGMENTED_SUFFIX

def is_segmented(path=None, img=None):
    """
    Returns True if the image is segmented
    (Approximation)
    """
    assert path is not None or img is not None
    threshold_segmented = 0.1
    if img is None:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    pixels = img.shape[0] * img.shape[1]
    seuil_gray = 15
    black_px_threshold = int(threshold_segmented * pixels)
    black_px = np.count_nonzero(img.ravel() <= seuil_gray)
    return black_px > black_px_threshold