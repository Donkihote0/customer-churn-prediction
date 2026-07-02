"""
Prediction module.
"""

from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path="../models/best_model.pkl"):
    """
    Load trained model.
    """
    return joblib.load(model_path)


def load_preprocessor(preprocessor_path="../models/preprocessor.pkl"):
    """
    Load fitted preprocessor.
    """
    return joblib.load(preprocessor_path)

def predict_customer(
    customer_data,
    model,
    preprocessor,
):
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer_data : dict
        Customer information.

    Returns
    -------
    dict
    """

    df = pd.DataFrame([customer_data])

    X = preprocessor.transform(df)

    probability = model.predict_proba(X)[0][1]

    prediction = model.predict(X)[0]

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability), 4),
    }
    
def predict_from_dataframe(
    df,
    model,
    preprocessor,
):
    """
    Predict multiple customers.
    """

    X = preprocessor.transform(df)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = model.predict(X)

    result = df.copy()

    result["Prediction"] = [
        "Yes" if p == 1 else "No"
        for p in predictions
    ]

    result["Probability"] = probabilities

    return result

def predict_from_csv(
    input_csv,
    output_csv,
    model,
    preprocessor,
):
    """
    Predict from csv.
    """

    df = pd.read_csv(input_csv)

    result = predict_from_dataframe(
        df,
        model,
        preprocessor,
    )

    Path(output_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_csv,
        index=False,
    )

    return result


