"""
Explainability Module

Author: Dat Nguyen
Sprint 8

Features
--------
✓ Feature Importance
✓ Permutation Importance
✓ SHAP
✓ Business Insight
✓ Sparse Matrix Support
✓ OneHotEncoder Support
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from scipy.sparse import issparse
from sklearn.inspection import permutation_importance


# ============================================================
# Utilities
# ============================================================

def load_model(model_path: str):
    """Load trained model."""
    return joblib.load(model_path)


def ensure_dataframe(X, feature_names):
    """
    Convert sparse matrix or ndarray to DataFrame.
    """

    if isinstance(X, pd.DataFrame):
        return X

    if issparse(X):
        X = X.toarray()

    return pd.DataFrame(X, columns=feature_names)


# ============================================================
# Feature Importance
# ============================================================

def get_feature_importance(model, feature_names):
    """
    Return feature importance.

    Works with:
        - Random Forest
        - Decision Tree
        - Gradient Boosting
        - XGBoost

    Logistic Regression:
        use coef_
    """

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

    elif hasattr(model, "coef_"):

        importance = abs(model.coef_[0])

    else:

        raise ValueError(
            "This model doesn't support feature importance."
        )

    df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    return df.sort_values(
        "Importance",
        ascending=False
    )


# ============================================================
# Plot Importance
# ============================================================

def plot_feature_importance(
    importance_df,
    save_path,
    top_n=20,
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = importance_df.head(top_n)

    plt.figure(figsize=(10, 7))

    plt.barh(
        df["Feature"],
        df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.title("Top Feature Importance")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()


# ============================================================
# Permutation Importance
# ============================================================

def get_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    top_n=50,
    n_repeats=3,
):
    """
    Fast permutation importance.

    Strategy
    --------
    1. Convert sparse -> DataFrame
    2. Keep Top N features
    3. Run permutation
    """

    X_test = ensure_dataframe(
        X_test,
        feature_names,
    )

    base_importance = get_feature_importance(
        model,
        feature_names,
    )

    selected_features = base_importance.head(top_n)["Feature"]

    X_small = X_test[selected_features]

    result = permutation_importance(

        estimator=model,

        X=X_small,

        y=y_test,

        scoring="roc_auc",

        n_repeats=n_repeats,

        random_state=42,

        n_jobs=-1,

    )

    perm_df = pd.DataFrame({

        "Feature": selected_features,

        "Importance": result.importances_mean

    })

    return perm_df.sort_values(
        "Importance",
        ascending=False,
    )


# ============================================================
# SHAP
# ============================================================

def create_shap_summary(
    model,
    X,
    feature_names,
    save_path,
):
    """
    Create SHAP summary plot.
    """

    X = ensure_dataframe(
        X,
        feature_names,
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    plt.figure()

    shap.summary_plot(

        shap_values,

        X,

        show=False,

    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# Business Insight
# ============================================================

def generate_business_insight(
    importance_df,
    save_path,
    top_n=10,
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top = importance_df.head(top_n)

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("# Business Insights\n\n")

        for idx, row in enumerate(top.itertuples(), 1):

            feature = row.Feature

            score = row.Importance

            f.write(
                f"## {idx}. {feature}\n"
            )

            f.write(
                f"- Importance: {score:.4f}\n\n"
            )


# ============================================================
# Save CSV
# ============================================================

def save_importance(
    importance_df,
    save_path,
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        save_path,
        index=False,
    )