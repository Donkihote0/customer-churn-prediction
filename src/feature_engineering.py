from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(X: pd.DataFrame,y: pd.Series,test_size: float = 0.2,random_state: int = 42,):
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target labels.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    return train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y,)


def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical columns.

    Returns
    -------
    tuple
        numerical_columns, categorical_columns
    """
    categorical_columns = (X.select_dtypes(include=["object", "category"]).columns.tolist())
    numerical_columns = (X.select_dtypes(exclude=["object", "category"]).columns.tolist())
    return numerical_columns, categorical_columns


def create_preprocessor(numerical_columns: list, categorical_columns: list,
):
    """
    Create preprocessing pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns,),
            ("cat", OneHotEncoder( handle_unknown="ignore"), categorical_columns,),
        ]
    )
    return preprocessor

def transform_data(preprocessor,X_train: pd.DataFrame,X_test: pd.DataFrame,):
    """
    Fit on training data and transform both
    training and testing datasets.
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return (X_train_processed,X_test_processed,)

def save_preprocessor(preprocessor,save_path: str,):
    """
    Save fitted preprocessor.
    """

    Path(save_path).parent.mkdir(parents=True,exist_ok=True,)
    joblib.dump(preprocessor,save_path,)

def load_preprocessor(path: str):
    """
    Load saved preprocessor.
    """
    return joblib.load(path)