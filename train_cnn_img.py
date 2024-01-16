import os
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling, RandomCrop, RandomTranslation, RandomBrightness, RandomContrast
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import *
from generate_df import *
import argparse
from argparse import RawTextHelpFormatter

import resource
import platform
import sys


def memory_limit(percentage: float):
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percentage), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory(percentage=0.9):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator


def generate_predictions_and_evaluate(model, dataset, class_names):
    """
    Returns the confusion matrix and the classification report of the model on the dataset
    """
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    predictions = model.predict(dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([y for _, y in dataset], axis=0)
    confusion = confusion_matrix(true_labels, predicted_labels)
    class_report_dict = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict)
    return confusion, class_report_df

def generate_model(num, t, num_classes, seed):
    """
    Generate a model with the given number
    Example:
    Configuration 2230:
    - 3 times (Conv2D(power of 2, (3,3), 'relu') + MaxPool2D((2,2)))
    - Dropout(0.2)
    - 2 Dense(power of 2, 'relu')
    - (mandatory) 1 Dense(num_classes,'softmax')
    - 0 -> No augmentation
    """
    # Augmentations
    augment_num = num % 10
    augment_layer = []
    if augment_num != 0:
        augments = np.array([RandomFlip(mode="horizontal_and_vertical", seed=seed, input_shape=(256, 256, 3)),
                                RandomRotation(0.2, seed=seed),
                                RandomZoom(0.2, seed=seed),
                                RandomTranslation(0.2, 0.2, seed=seed),
                                RandomCrop(256, 256, seed=seed),
                                RandomBrightness(0.2, seed=seed),
                                RandomContrast(0.2, seed=seed)])
        if augment_num == 1:
            augment_layer  = list(augments[[0, 1]])
        elif augment_num == 2:
            augment_layer = list(augments[[5, 6]])
        elif augment_num == 3:
            augment_layer = list(augments[[0, 1, 5, 6]])
        else:
            raise ValueError("Augment number must be between 0 and 3")


    # Model layers
    layers = [Rescaling(1./255, input_shape=(256, 256, 3))]

    # Transfer learning
    if t != 0:
        if t == 1:
            base_model = tf.keras.applications.VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        elif t == 2:
            base_model = tf.keras.applications.EfficientNetV2L(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        elif t == 3:
            base_model = tf.keras.applications.Xception(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        elif t == 4:
            base_model = tf.keras.applications.InceptionResNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        layers.append(base_model)

    # Layers : Conv+MaxPool, Dropout, Dense
    layer_config = num // 10

    # Conv+MaxPool
    conv = layer_config % 10
    filters = 32
    for _ in range(conv):
        layers.append(Conv2D(filters, (3, 3), activation='relu'))
        layers.append(MaxPooling2D((2, 2)))
        filters *= 2

    # Dropout
    dropout = (layer_config // 10) % 10
    if dropout != 0:
        layers.append(Dropout(dropout/10))

    # Flatten
    layers.append(Flatten())

    # Dense
    dense = (layer_config // 100) % 10
    units = 2
    while units < num_classes:
        units *= 2

    layers += [Dense(units*(2**i), activation='relu') for i in range(1,dense+1)][::-1]

    layers.append(Dense(num_classes, activation='softmax'))
    model = tf.keras.Sequential(augment_layer + layers)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


@memory(percentage=0.9)
def main():
    ##################### ARGUMENTS ############################
    description = \
    """Script d'entrainement des modèles CNN sur des images:
    Choix du dataset:
        - 0 : Plant Seedlings
        - 1 : New Plant Diseases
        - 2 : PlantVillage Color
        - 3 : PlantVillage Segmented

    Choix du modèle:
        Les 3 premiers chiffres correspondent à la configuration des couches du modèle. Le chiffre des unités correspond à la configuration des augmentations faites par le modèle.

        - 3230 : 3 Conv+MaxPool, Dropout(0.2) + 3 Dense + output (= 1 Dense(num_classes,'softmax'))
        - 2231 : 3 Conv+MaxPool, Dropout(0.2) + 2 Dense + output (= 1 Dense(num_classes,'softmax')) + RandomFlip(mode="horizontal_and_vertical", seed=seed)
        - 3212 : 1 Conv+MaxPool, Dropout(0.2) + 3 Dense + output (= 1 Dense(num_classes,'softmax')) + RandomBrightness(0.2, seed=seed)
        - 3523 : 2 Conv+MaxPool, Dropout(0.5) + 3 Dense + output (= 1 Dense(num_classes,'softmax')) + RandomFlip(mode="horizontal_and_vertical", seed=seed) + RandomBrightness(0.2, seed=seed)

    Choix du transfert learning:
        - 0 : Pas de transfert learning
        - 1 : VGG16
        - 2 : EfficientNetV2L
        - 3 : Xception
        - 4 : InceptionResNetV2

    Choix du fine tuning: (uniquement si transfert learning)
        - Si --finetuning : fine tuning du modèle
        - Sinon : pas de fine tuning

    Choix de y:
        --mode=disease: Le modèle va chercher à savoir si la plante est malade (non compatible avec le dataset plant seedlings)
        --mode=species: Le modèle ne va chercher à reconnaitre que la plante

    Exemple d'utilisation:

        python main.py 0 3230
        python main.py 0 0031 1
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('i', type=int, help='indice du dataset à utiliser')
    parser.add_argument('m', type=int, help='Choix du modèle à entraîner')
    parser.add_argument('t', type=int, default=0, help='Choix du transfert learning à utiliser')
    parser.add_argument('--finetuning', action='store_true', help='Fine tuning du modèle')
    parser.add_argument('--mode', choices=['disease', 'species', 'normal'], default='normal', help='Spécifie le label à retrouver')

    args = parser.parse_args()
    i = args.i
    m = args.m
    t = args.t
    finetuning = args.finetuning and t != 0
    disease = args.mode == 'disease'
    species = args.mode == 'species'

    ####################### PATHS ##############################

    # Dataset paths
    data_path = params['dataset_path']['data']
    plant_seedlings_path = params['dataset_path']['plant_seedlings']
    new_plant_diseases_path = params['dataset_path']['new_plant_diseases']
    plantvillage_color_path = params['dataset_path']['plantvillage_color']
    plantvillage_seg_path = params['dataset_path']['plantvillage_segmented']

    # Models
    model_path = os.path.join(data_path, "model")
    if disease:
        img_model_path = os.path.join(model_path, "disease_img")
    elif species:
        img_model_path = os.path.join(model_path, "species_img")
    else:
        img_model_path = os.path.join(model_path, "img")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(img_model_path, exist_ok=True)

    ##################### PARAMS ############################

    dirs = [plant_seedlings_path, new_plant_diseases_path, plantvillage_color_path, plantvillage_seg_path]

    ds_names = ['ps', 'pv_color', 'npd', 'pv_seg']
    dataset_names = ['Plant Seedlings', 'New Plant Diseases', 'PlantVillage Color', 'PlantVillage Segmented']
    tl = ['', '-vgg16', '-efficientnetv2l', '-xception', '-InceptionResNetV2']
    # preprocess_input_func = [None, tf.keras.applications.vgg16.preprocess_input, tf.keras.applications.efficientnet.preprocess_input, tf.keras.applications.xception.preprocess_input, tf.keras.applications.inception_resnet_v2.preprocess_input]
    print("Dataset: {}".format(dataset_names[i]))

    IMG_SIZE = 256
    batch_size = 128
    epochs = 500
    seed = 123

    data_dir = dirs[i]
    ds_name = ds_names[i]
    path = os.path.join(img_model_path, ds_name, "{:04d}".format(m)+tl[t])

    os.makedirs(path, exist_ok=True)

    already_trained = os.path.isfile(os.path.join(path, 'report.csv')) \
    and os.path.isfile(os.path.join(path, 'mat_conf.npy')) \
    and os.path.isfile(os.path.join(path, 'history.csv')) \
    and os.path.isfile(os.path.join(path, 'cnn.keras'))

    finetuned = os.path.isfile(os.path.join(path+'-finetuning', 'report.csv')) \
    and os.path.isfile(os.path.join(path+'-finetuning', 'mat_conf.npy')) \
    and os.path.isfile(os.path.join(path+'-finetuning', 'history.csv')) \
    and os.path.isfile(os.path.join(path+'-finetuning', 'cnn.keras'))

    if already_trained and (finetuned or not finetuning):
        print("Task already done")
        return

    if data_dir == plant_seedlings_path and disease:
        print('Plant Seedlings dataset is incompatible with disease mode')
        return

    ###################### DATA ############################

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size)


    # Segment images if Plant Seedlings dataset
    if data_dir == plant_seedlings_path:
        class_names = train_ds.class_names
        num_classes = len(class_names)

        def segment(images, label):
            return tf.map_fn(segment_img, images), label

        def segment_img(image):
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            img = tf.image.rgb_to_hsv(image / 255.0) * 255.0
            img = tf.numpy_function(process_np, [img], tf.float32)
            return img

        def process_np(img_np):
            img_np = img_np.astype(np.uint8)
            lower_green = np.array([25, 30, 40])
            upper_green = np.array([80, 255, 255])
            mask = cv2.inRange(img_np, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(filter(lambda x: cv2.contourArea(x) > 30, contours))
            final_result = cv2.drawContours(np.zeros(img_np.shape[:2]), contours, -1, color=(255), thickness=cv2.FILLED).astype(np.uint8)

            img_np = cv2.bitwise_and(img_np, img_np, mask=final_result)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB).astype(np.float32)
            return img_np

        train_ds = train_ds.map(segment)
        val_ds = val_ds.map(segment)

    # Editing labels if disease = True or species = True
    elif disease:
        healthy_permutation = [int('healthy' in name) for name in train_ds.class_names]
        def update_healthy_labels(x,y):
            new_y = tf.gather(healthy_permutation, y)
            return x, new_y

        train_ds = train_ds.map(update_healthy_labels)
        val_ds = val_ds.map(update_healthy_labels)

        class_names = ['unhealthy', 'healthy']
        num_classes = 2

    elif species:
        new_classes = list(set(map(lambda x: x.split('___')[0], train_ds.class_names)))
        new_classes.sort()
        species_permutation = [new_classes.index(name.split('___')[0]) for name in train_ds.class_names] # permutation[old_index] = new_index

        def update_species_labels(x, y):
            new_y = tf.gather(species_permutation, y) # Does new_y = species_permutation[y]
            return x, new_y

        train_ds = train_ds.map(update_species_labels)
        val_ds = val_ds.map(update_species_labels)

        class_names = new_classes
        num_classes = len(new_classes)

    else:
        class_names = train_ds.class_names
        num_classes = len(class_names)

    # Preprocessing input if transfer learning
    # if t != 0:
    #     def preprocess_input(x, y):
    #         x = preprocess_input_func[t](x)
    #         return x, y

    #     train_ds = train_ds.map(preprocess_input)
    #     val_ds = val_ds.map(preprocess_input)

    #################### MODEL ##############################
    if not already_trained:
        # Callbacks
        earlystop = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            verbose = 1,
                            restore_best_weights = True)

        reduce_learning_rate = ReduceLROnPlateau(
                                            monitor="val_loss",
                                            patience=3,
                                            min_delta= 0.01,
                                            factor=0.1,
                                            cooldown = 4,
                                            verbose=1)

        # Model
        model = generate_model(m, t, num_classes, seed)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[earlystop, reduce_learning_rate])

        model.save(path + '/cnn.keras')
        #################### EVALUATION ##############################

        mat_conf, report = generate_predictions_and_evaluate(model, val_ds, class_names)

        # Print results
        # print("Classification Report:")
        # print(report)
        # print("Confusion Matrix:")
        # print(mat_conf)

        # Save results
        report.to_csv(path + '/report.csv')
        np.save(path + '/mat_conf.npy', mat_conf)
        pd.DataFrame(history.history).to_csv(path + '/history.csv', index=False)

    ##################### FINE TUNING ############################
    if finetuning and not finetuned:
        print("Fine tuning")
        if already_trained:
            model = tf.keras.models.load_model(path + '/cnn.keras')
        # Unfreeze the base model
        model.layers[1].trainable = True

        # Update path to fine-tuned model
        path = path + '-finetuned'
        os.makedirs(path, exist_ok=True)

        # Compile the model with a low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        # Callbacks
        earlystop = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            verbose = 1,
                            restore_best_weights = True)

        reduce_learning_rate = ReduceLROnPlateau(
                                            monitor="val_loss",
                                            patience=3,
                                            min_delta= 0.01,
                                            factor=0.1,
                                            cooldown = 4,
                                            verbose=1)

        # Fine-tune the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[earlystop, reduce_learning_rate])

        model.save(path + '/cnn.keras')
        #################### EVALUATION ##############################

        mat_conf, report = generate_predictions_and_evaluate(model, val_ds, class_names)

        # Print results
        # print("Classification Report:")
        # print(report)
        # print("Confusion Matrix:")
        # print(mat_conf)

        # Save results
        report.to_csv(path + '/report.csv')
        np.save(path + '/mat_conf.npy', mat_conf)
        pd.DataFrame(history.history).to_csv(path + '/history.csv', index=False)


if __name__ == '__main__':
    main()