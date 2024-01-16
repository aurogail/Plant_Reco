import os
import pandas as pd
import argparse

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import joblib

# memory control
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

@memory(percentage=0.9)
def main(model, reduc_dim=False):
    data = pd.read_csv("data/csv/hist/lrgbwhsvlab_segmented_image_intensity_histograms.csv", index_col=0)
    X = data.iloc[:, :-2].values
    y = data.iloc[:, -1].values # encoded labels

    # Preprocessing data
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # PCA
    if reduc_dim:
        print("Applying dimension reduction (PCA)...")
        prev_size = X.shape[1]
        pca = PCA(n_components = 0.99)
        X = pca.fit_transform(X)
        print("Number of components:", prev_size, "->", X.shape[1])

    # Models
    param_grid = {
        'lr': {'C': [0.1, 1, 10, 100, 1000]},
        'rf': {'n_estimators': [100, 200, 300, 400, 500]},
        'svm': {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001]},
        'sgd': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'penalty': ['elasticnet'], 'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]},
        'knn': {'n_neighbors': [3, 5, 7, 9]},
        'xgb': {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6, 7]},
        'lgb': {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6, 7]},
        'cb' : {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6, 7]}
    }

    if model == 'lr':
        estimator = LogisticRegression()
        params = param_grid["lr"]
    elif model == 'rf':
        estimator = RandomForestClassifier()
        params = param_grid["rf"]
    elif model == 'svm':
        estimator = SVC()
        params = param_grid["svm"]
    elif model == 'sgd':
        estimator = SGDClassifier()
        params = param_grid["sgd"]
    elif model == 'knn':
        estimator = KNeighborsClassifier()
        params = param_grid["knn"]
    elif model == 'xgb':
        estimator = xgb.XGBClassifier()
        params = param_grid["xgb"]
    elif model == 'lgb':
        estimator = lgb.LGBMClassifier()
        params = param_grid["lgb"]
    elif model == 'cb':
        estimator = cb.CatBoostClassifier()
        params = param_grid["cb"]
    else:
        raise ValueError("Unknown estimator")

    # Fitting
    # GridSearchCV
    grid = GridSearchCV(estimator=estimator, param_grid=params, cv=folds, n_jobs=-1, verbose=3)
    grid.fit(X, y)
    print("Params:", grid.best_params_)
    print("Score:", grid.best_score_)

    # Saving model + results
    os.makedirs("data/model/hist", exist_ok=True)
    joblib.dump(grid.best_estimator_, "data/model/hist/" + model + "_hist.pkl")
    pd.DataFrame(grid.cv_results_).to_csv("data/model/hist/" + model + "_hist.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default="lr", help="Estimator to use")
    parser.add_argument("--reduc_dim", action="store_true", help="Apply dimension reduction")
    args = parser.parse_args()

    main(args.estimator, args.reduc_dim)