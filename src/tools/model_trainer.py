from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REGRESSION_MODELS = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
}


@dataclass
class TrainResult:
    task_type: str
    target_col: str
    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    metrics_df: pd.DataFrame
    predictions_df: pd.DataFrame


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target):
        unique_count = target.nunique(dropna=True)
        if unique_count <= 10 and set(target.dropna().unique()).issubset({0, 1}):
            return "classification"
        return "regression"
    return "classification"


def infer_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> tuple[List[str], List[str]]:
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return numeric_cols, categorical_cols


def get_supported_models(task_type: str) -> List[str]:
    if task_type == "regression":
        return list(REGRESSION_MODELS.keys())
    if task_type == "classification":
        return list(CLASSIFICATION_MODELS.keys())
    raise ValueError(f"Unsupported task_type: {task_type}")


def validate_target_for_task(df: pd.DataFrame, target_col: str, task_type: str):
    target = df[target_col].dropna()
    unique_count = target.nunique()
    if unique_count < 2:
        raise ValueError(f"Target column '{target_col}' has fewer than 2 unique non-null values and cannot be used for training.")
    if task_type == "regression" and not pd.api.types.is_numeric_dtype(target):
        raise ValueError(f"Target column '{target_col}' is not numeric, so it is not suitable for regression.")


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    average = "binary" if len(pd.Series(y_true).unique()) == 2 else "macro"
    f1 = f1_score(y_true, y_pred, average=average)
    roc_auc = float("nan")
    if y_score is not None and len(pd.Series(y_true).unique()) == 2:
        roc_auc = roc_auc_score(y_true, y_score)
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def build_model(model_name: str, task_type: str):
    if task_type == "regression":
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unsupported regression model: {model_name}")
        if model_name == "RandomForestRegressor":
            return REGRESSION_MODELS[model_name](random_state=42, n_estimators=200)
        return REGRESSION_MODELS[model_name]()

    if task_type == "classification":
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unsupported classification model: {model_name}")
        if model_name == "RandomForestClassifier":
            return CLASSIFICATION_MODELS[model_name](random_state=42, n_estimators=200)
        return CLASSIFICATION_MODELS[model_name](max_iter=2000)

    raise ValueError(f"Unsupported task_type: {task_type}")


def run_tabular_baselines(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str] | None = None,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    include_models: List[str] | None = None,
) -> TrainResult:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        feature_cols = list(feature_cols)

    clean_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    if clean_df.empty:
        raise ValueError("No rows left after dropping missing target values.")

    if task_type is None:
        task_type = infer_task_type(clean_df, target_col)

    validate_target_for_task(clean_df, target_col, task_type)

    numeric_cols, categorical_cols = infer_feature_types(clean_df, feature_cols)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if include_models is None:
        if task_type == "regression":
            include_models = ["LinearRegression", "RandomForestRegressor"]
        else:
            include_models = ["LogisticRegression", "RandomForestClassifier"]
    else:
        supported_models = set(get_supported_models(task_type))
        unsupported_models = [model for model in include_models if model not in supported_models]
        if unsupported_models:
            raise ValueError(
                f"Models {unsupported_models} are not supported for task_type='{task_type}'. "
                f"Supported models: {sorted(supported_models)}"
            )

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    stratify = None
    if task_type == "classification" and y.nunique() > 1:
        class_counts = y.value_counts()
        if class_counts.min() >= 2:
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if task_type == "classification" and y_train.nunique() < 2:
        raise ValueError(
            f"Training split for target '{target_col}' contains fewer than 2 classes. "
            "Choose a different target, increase dataset size, or adjust task type."
        )

    metrics_rows = []
    prediction_frames = []

    for model_name in include_models:
        model = build_model(model_name, task_type)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if task_type == "regression":
            metric = regression_metrics(y_test, y_pred)
            metrics_rows.append(
                {
                    "model": model_name,
                    "mae": metric["mae"],
                    "rmse": metric["rmse"],
                    "r2": metric["r2"],
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                }
            )
            pred_df = X_test.copy()
            pred_df[target_col] = y_test.values
            pred_df["y_true"] = y_test.values
            pred_df["y_pred"] = y_pred
            pred_df["model"] = model_name
            prediction_frames.append(pred_df.reset_index(drop=True))
            continue

        y_score = None
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        metric = classification_metrics(y_test, y_pred, y_score=y_score)
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": metric["accuracy"],
                "f1": metric["f1"],
                "roc_auc": metric["roc_auc"],
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
        )
        pred_df = X_test.copy()
        pred_df[target_col] = y_test.values
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred
        if y_score is not None:
            pred_df["y_score"] = y_score
        pred_df["model"] = model_name
        prediction_frames.append(pred_df.reset_index(drop=True))

    return TrainResult(
        task_type=task_type,
        target_col=target_col,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        metrics_df=pd.DataFrame(metrics_rows),
        predictions_df=pd.concat(prediction_frames, ignore_index=True),
    )
