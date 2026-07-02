from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_models(random_state=42):
    """
    Return baseline models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),

        "Random Forest": RandomForestClassifier(
            random_state=random_state,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def evaluate_single_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train and evaluate one model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def compare_models(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Compare all baseline models.
    """
    models = get_models()
    results = []
    trained_models = {}
    for name, model in models.items():
        metrics = evaluate_single_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        metrics["Model"] = name
        trained_models[name] = model
        results.append(metrics)
    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ]
    return results_df, trained_models


def save_results(
    results_df,
    path,
):
    """
    Save comparison table.
    """
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_df.to_csv(
        path,
        index=False,
    )


def plot_results(
    results_df,
    save_path,
):
    """
    Plot ROC-AUC comparison.
    """
    plt.figure(figsize=(8,5))
    plt.bar(
        results_df["Model"],
        results_df["ROC-AUC"],
    )
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_best_model(
    results_df,
    trained_models,
    save_dir="../models",
):
    """
    Save best model according to ROC-AUC.
    """
    best = results_df.sort_values(
        "ROC-AUC",
        ascending=False,
    ).iloc[0]
    best_name = best["Model"]
    model = trained_models[best_name]
    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True,
    )
    filename = (
        best_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )
    joblib.dump(
        model,
        Path(save_dir) / filename,
    )
    joblib.dump(
        model,
        Path(save_dir) / "best_model.pkl",
    )
    return best