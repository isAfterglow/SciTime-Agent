from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def guess_target_col(df: pd.DataFrame) -> str | None:
    candidate_keywords = [
        "target",
        "label",
        "y",
        "soh",
        "rul",
        "capacity",
        "class",
    ]
    lowered = {col: col.lower() for col in df.columns}
    for col, col_lower in lowered.items():
        if any(keyword in col_lower for keyword in candidate_keywords):
            return col
    return df.columns[-1] if len(df.columns) > 0 else None


def _numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> Dict[str, Dict[str, float]]:
    if not numeric_cols:
        return {}
    summary = df[numeric_cols].agg(["mean", "std", "min", "max"]).T
    return {
        col: {
            "mean": float(summary.loc[col, "mean"]) if pd.notna(summary.loc[col, "mean"]) else None,
            "std": float(summary.loc[col, "std"]) if pd.notna(summary.loc[col, "std"]) else None,
            "min": float(summary.loc[col, "min"]) if pd.notna(summary.loc[col, "min"]) else None,
            "max": float(summary.loc[col, "max"]) if pd.notna(summary.loc[col, "max"]) else None,
        }
        for col in summary.index
    }


def profile_dataframe(df: pd.DataFrame, target_col: str | None = None) -> Dict[str, Any]:
    """
    Analyze a dataframe and return basic profiling information.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    missing_ratio = df.isna().mean().to_dict()

    resolved_target_col = target_col if target_col in df.columns else guess_target_col(df)

    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_ratio": missing_ratio,
        "duplicate_rows": int(df.duplicated().sum()),
        "target_col": resolved_target_col,
        "numeric_summary": _numeric_summary(df, numeric_cols),
    }

    return profile
