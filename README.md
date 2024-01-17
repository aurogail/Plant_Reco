# Reconnaissance de plantes

## Présentation du contexte

Ce répertoire GitHub contient la contribution de code produit par **Jacques Colin** pour le projet "fil rouge" de la formation [Datascientest](https://datascientest.com/formation-data-scientist). Ce projet permet la reconnaissance de plantes et la détection de maladies d'une image de plante prise par un appareil photo.
Pour cela, trois datasets ont été utilisés : [New Plant Diseases](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), [Plantvillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) et [Plant Seedlings](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset). Ces datasets proviennent de [Kaggle](https://www.kaggle.com/).

* Le dataset **Plant seedlings** contient 5539 images de jeunes pousses de plantes appartenant à 12 classes.
* Le dataset **Plantvillage** contient 54305 images de plances appartenant à 38 classes.
* Le dataset **New Plant Diseases** contient 87900 images et est une augmentation des images présentes dans **Plantvillage**.

Sujet: `Reconnaissance de plantes.pdf`

## Mise en place de l'environnement

1. Installer les dépendances requises de requirements.txt dans un environnement virtuel Python en effectant les commandes suivantes:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Configurer sa clé API Kaggle (Générer sa clé API Kaggle et stocker dans .kaggle/kaggle.json)

3. Se placer dans le projet et exécuter la commande suivante

    ```bash
    python3 fetch_data.py && python3 clean_data.py && python3 transform_data.py && python3 generate_df.py
    ```

## Architecture du projet

### Data

Les données sont stockés dans le dossier `data`. Les trois datasets sont téléchargés via le fichier `fetch_data.py`, les datasets ne sont pas divisés en dossier train/test/valid.

Le dossier `csv` est généré grâce au fichier `generate_df.py`. Il contient des informations générales sur les datasets mais aussi sur les histogrammes d'intensité de chaque dataset.

Le dossier `model` est généré à partir de `train_cnn_img.py` et de `train_ml_hist.py`. Ces fichiers entrainent des modèles selon une configuration donnée.

* data
  * csv
    * general
    * hist
  * model
    * hist
    * pixel
  * new-plant-diseases-dataset
  * plantvillage-dataset
  * v2-plant-seedlings-dataset

### Scripts

Les étapes de récupération et de traitement des données sont automatisés afin d'avoir une reproductibilité des datasets. La génération des informations des datasets sert à centraliser les données des datasets et à éviter de recalculer les histogrammes d'intensité.

L'entrainement automatisé des modèles permet d'entrainer plusieurs modèles notamment la nuit sans avoir besoin d'intervenir pour exécuter chaque entrainement et permet la reproductibilité de l'entrainement des modèles.

#### Génération des ensembles de données

* fetch_data.py
* clean_data.py
* transform_data.py
* generate_df.py

#### Génération des modèles

* train_cnn_img.py
* train_ml_hist.py

### Visualisation des performances des modèles

Après avoir entrainé des modèles, le fichier `results.ipynb` permet de mettre en évidence et de comparer leurs performances. De plus, il est possible d'observer la Grad-CAM des modèles.

## Modèles utilisés

### Histogramme d'intensité

Les modèles de machine learning classiques ont été étudiés. En utilisant la **PCA** pour réduire le nombre de variables par conséquent le temps de calcul, ainsi que **GridSearchCV** afin de paramètrer au mieux les paramètres des modèles et en utilisant un **StratifiedKFold** pour limiter le biais du choix des ensembles d'entrainement et de test, le modèle SVM a été le plus performant et a pu atteindre 93% de précision.

### Images

Les modèles de Deep Learning notamment le CNN permettent d'atteindre 93% de précision et l'utilisation du transfert learning permet d'atteindre jusqu'à 98.7% de précision sur le dataset New Plant Diseases.

## Kaggle notebooks
#### Histogrammes
Creation des histogrammes qui serviront à la classification par des arbres de décisions avec lightGBM et par un réseau de neurones dense avec TensorFlow.
Les images segmentées du jeu de données plantvillage sont utilisées.

    * reco-plantes-lightgbm.ipynb

#### Extraction openimagesv7
Extraction d'images à partir du jeu de données openimagesv7. Ces images seront utilisées en tant qu'arrière-plan pour la modélisation avec le modèle Yolov8 de Ultralytics.

    * import-openimagesv7-for-background-images.ipynb

#### Yolov8
Augmentation des images et mdélisation avec le modèle Yolov8 pour la localisation et la classification. Les images segmentées du jeu de données plantvillage sont utilisées.

    * yolov8-with-image-augmentation.ipynb
