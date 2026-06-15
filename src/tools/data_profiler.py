from typing import Dict, Any
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a dataframe and return basic profiling information.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    missing_ratio = df.isna().mean().to_dict()

    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_ratio": missing_ratio,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return profile