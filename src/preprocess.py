import pandas as pd

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert incorrect data types.
    """

    df = df.copy()

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.

    Business rule:
    Customers with zero tenure have not been charged yet,
    therefore Total Charges should be 0.
    """

    df = df.copy()

    mask = (
        (df["Tenure Months"] == 0)
        &
        (df["Total Charges"].isna())
    )

    df.loc[mask, "Total Charges"] = 0

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows.
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)

    return df


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that do not contribute
    to model training.
    """

    df = df.copy()

    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Reason",
        "Churn Value",
        "Churn Score",
        "CLTV",
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True,
        errors="ignore"
    )

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate cleaned dataset.
    """

    if df["Total Charges"].isnull().sum() > 0:
        raise ValueError(
            "Total Charges still contains missing values."
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete preprocessing pipeline.
    """

    df = convert_dtypes(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = remove_unused_columns(df)

    validate_data(df)

    return df