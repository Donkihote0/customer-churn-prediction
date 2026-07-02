from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

def cross_validate_model(
    model,
    X_train,
    y_train,
    cv=5,
):
    """
    Perform Cross Validation.
    """
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores.tolist(),
    }
    
def tune_random_forest(
    X_train,
    y_train,
):
    """
    Tune Random Forest using Grid Search.
    """
    param_grid = {
        "n_estimators":[100,200,300],
        "max_depth":[5,10,20,None],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1,2],
    }
    model = RandomForestClassifier(
        random_state=42,
    )
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(
        X_train,
        y_train,
    )
    return grid

def save_best_model(
    grid,
    save_path,
):
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        grid.best_estimator_,
        save_path,
    )
    
def save_best_params(
    grid,
    save_path,
):

    with open(
        save_path,
        "w",
    ) as f:

        json.dump(
            grid.best_params_,
            f,
            indent=4,
        )

def save_cv_results(
    grid,
    save_path,
):

    df = pd.DataFrame(
        grid.cv_results_
    )

    df.to_csv(
        save_path,
        index=False,
    )
    
def plot_top_results(
    grid,
    save_path,
):

    df = pd.DataFrame(
        grid.cv_results_
    )

    df = df.sort_values(
        "mean_test_score",
        ascending=False,
    ).head(10)

    plt.figure(figsize=(10,5))

    plt.bar(
        range(len(df)),
        df["mean_test_score"],
    )

    plt.xticks(
        range(len(df)),
        df.index,
        rotation=45,
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

