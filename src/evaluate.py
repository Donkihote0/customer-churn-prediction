from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)


def evaluate_model(
    y_true,
    y_pred,
    y_prob,
):
    """
    Calculate evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    return metrics


def save_metrics(
    metrics,
    save_path,
):
    """
    Save metrics as JSON.
    """

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_classification_report(
    y_true,
    y_pred,
    save_path,
):
    """
    Save classification report.
    """

    report = classification_report(
        y_true,
        y_pred
    )

    with open(save_path, "w") as f:
        f.write(report)


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path,
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig(save_path)

    plt.close()


def plot_roc_curve(
    model,
    X_test,
    y_test,
    save_path,
):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.savefig(save_path)

    plt.close()