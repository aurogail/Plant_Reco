import os
from tqdm import tqdm
import pandas as pd
from functools import reduce
from multiprocessing import Pool
from utils import *

def generate_general_df(path, diseases=True):
    if diseases:
        data = {"path": [], "label": [], "species": [], "disease" : [], "sharpness": [], "exposure": [], "contrast":[], "duplicated":[], "augmented": [], "segmented": [], "set":[]}
    else:
        data = {"path": [], "label": [], "sharpness": [], "exposure": [], "contrast":[], "duplicated":[], "augmented": [], "segmented": [], "set":[]}

    sets = os.listdir(path)
    splitted = 'train' in sets
    if splitted:
        species = os.listdir(os.path.join(path, sets[0]))
    else:
        species = sets
        sets = ['']

    for dir in tqdm(species):
        for folder in sets:
            images = os.listdir(os.path.join(path, folder, dir))
            for image in images:
                img_path = os.path.join(path, folder, dir, image)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if diseases:
                    sp, ds = dir.split("___")
                    data["species"].append(sp)
                    data["disease"].append(ds)
                data["label"].append(dir)
                data["path"].append(img_path)
                data["sharpness"].append(sharpness_score(img=img))
                data["contrast"].append(contrast_score(img=img))
                data["exposure"].append(exposure_score(img=img))
                data["segmented"].append(int(is_segmented(img=img)))
                data["augmented"].append(int(is_augmented(img_path)))
                data["set"].append(folder)

    # Last element is considered as original
    duplicates = list(reduce(lambda x, y: x + y[:-1], get_duplicates(path1=path), []))
    data["duplicated"] = list(map(lambda x: int(x in duplicates), data["path"]))
    return pd.DataFrame.from_dict(data)

def generate_hist_df(dataset_key):
    diseases = dataset_key != "plant_seedlings"
    df = get_df('general_csv_path', dataset_key, lambda path: generate_general_df(path, diseases=diseases))[['path', 'label','set']]
    hists = np.zeros(shape=(len(df), 8 * 255), dtype=np.float32)
    for index, path in enumerate(df.path):
        img_bgr = cv2.imread(path)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        imgs = [img_bgr, img_hsv, img_hls, img_gray]
        color_spaces = [['b', 'g', 'r'], ['h', 's', 'v'], ['l'], ['w']]
        cpt = 0
        for i, color_space in enumerate(color_spaces):
            for j in range(len(color_space)):
                if color_space[0] == 'w':
                    non_zero_intensity = np.count_nonzero(imgs[i])
                else:
                    non_zero_intensity = np.count_nonzero(imgs[i][:,:,j])
                if non_zero_intensity == 0:
                    hists[index, 255*cpt] = 0
                else:
                    # Exclude intensity 0
                    hists[index, 255*cpt:255*(cpt+1)] = cv2.calcHist([imgs[i]], [j], None, [256], [0, 256])[1:].flatten() / non_zero_intensity
                cpt += 1

    columns = [f"{channel}_{i}" for color_space in color_spaces for channel in color_space for i in range(1,256)]
    df_hists = pd.DataFrame(hists, columns=columns)
    return pd.concat([df[["set", "label"]], df_hists], axis=1)

def generate_and_read_csv(df_path, dataset_path, generate_func):
    if not os.path.exists(df_path):
        print(f"Generating {df_path}...")
        os.makedirs(os.path.dirname(df_path), exist_ok=True)
        df = generate_func(dataset_path)
        df.to_csv(df_path, index=False)
    return pd.read_csv(df_path)

def get_df(csv_key, dataset_key, generate_func):
    return generate_and_read_csv(params[csv_key][dataset_key],
                                 params['dataset_path'][dataset_key], generate_func)

# V2 Plant Seedlings dataset
def get_general_df_plant_seedlings():
    return get_df('general_csv_path', 'plant_seedlings', lambda path: generate_general_df(path, diseases=False))

def get_hist_df_plant_seedlings():
    return generate_and_read_csv(params['hist_csv_path']['plant_seedlings'], 'plant_seedlings', generate_hist_df)

# PlantVillage dataset
def get_general_df_plantvillage_color():
    return get_df('general_csv_path', 'plantvillage_color', generate_general_df)

def get_hist_df_plantvillage_color():
    return generate_and_read_csv(params['hist_csv_path']['plantvillage_color'], 'plantvillage_color', generate_hist_df)

def get_general_df_plantvillage_segmented():
    return get_df('general_csv_path', 'plantvillage_segmented', generate_general_df)

def get_hist_df_plantvillage_segmented():
    return generate_and_read_csv(params['hist_csv_path']['plantvillage_segmented'], 'plantvillage_segmented', generate_hist_df)

def get_general_df_plantvillage_grayscale():
    return get_df('general_csv_path', 'plantvillage_gray', generate_general_df)

def get_hist_df_plantvillage_grayscale():
    return generate_and_read_csv(params['hist_csv_path']['plantvillage_gray'], 'plantvillage_gray', generate_hist_df)

# New Plant Diseases dataset
def get_general_df_new_plant_diseases():
    return get_df('general_csv_path', 'new_plant_diseases', generate_general_df)

def get_hist_df_new_plant_diseases():
    return generate_and_read_csv(params['hist_csv_path']['new_plant_diseases'], 'new_plant_diseases', generate_hist_df)

if __name__ == "__main__":
    # Generate general csv
    with Pool(5) as pool:
        res = list(map(pool.apply_async, [get_general_df_plant_seedlings, get_general_df_new_plant_diseases, get_general_df_plantvillage_color, get_general_df_plantvillage_segmented, get_general_df_plantvillage_grayscale]))
        for r in res:
            r.get()

    # Generate hists csv
    with Pool(5) as pool:
        res = list(map(pool.apply_async, [get_hist_df_plant_seedlings, get_hist_df_new_plant_diseases, get_hist_df_plantvillage_color, get_hist_df_plantvillage_segmented, get_hist_df_plantvillage_grayscale]))
        for r in res:
            r.get()