from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.battery_metrics import regression_metrics


def make_onehot_encoder():
    """
    Compatible with different sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Define features for summary baseline.

    Important:
    - Do not use cell_id as feature, otherwise model may memorize cells.
    - Do not use target columns as features.
    """

    exclude_cols = {
        "cell_id",
        "target_soh_next",
        "target_capacity_next",
        "raw_capacity_col",
        "raw_soh_col",
        "raw_throughput_col",
        "raw_resistance_col",
    }

    candidate_numeric = [
        "temperature",
        "rpt_index",
        "charge_throughput",
        "energy_throughput",
        "capacity_c10",
        "soh",
        "resistance_0p1s",
        "ageing_cycles",
        "days_of_degradation",
        "delta_charge_throughput",
        "capacity_delta_prev",
        "soh_delta_prev",
        "resistance_delta_prev",
    ]

    candidate_categorical = [
        "soc_range",
    ]

    numeric_cols = [
        col for col in candidate_numeric
        if col in df.columns
        and col not in exclude_cols
        and df[col].notna().any()
    ]

    categorical_cols = [
        col for col in candidate_categorical
        if col in df.columns
        and col not in exclude_cols
        and df[col].notna().any()
    ]

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def build_models(preprocessor) -> Dict[str, Pipeline]:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }

    pipelines = {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }

    return pipelines


def lighten_color(color, amount: float = 0.45):
    """
    Blend a color toward white to create a lighter paired shade.
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def leave_one_cell_out_evaluation(
    df: pd.DataFrame,
    target_col: str = "target_soh_next",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-cell-out evaluation.

    Each round:
    - train on 5 cells
    - test on 1 unseen cell
    """

    df = df.copy()
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    numeric_cols, categorical_cols = get_feature_columns(df, target_col)

    print("Numeric features:")
    for c in numeric_cols:
        print(f"  - {c}")

    print("Categorical features:")
    for c in categorical_cols:
        print(f"  - {c}")

    feature_cols = numeric_cols + categorical_cols

    all_metrics = []
    all_predictions = []

    cells = sorted(df["cell_id"].unique())

    for test_cell in cells:
        train_df = df[df["cell_id"] != test_cell].copy()
        test_df = df[df["cell_id"] == test_cell].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        models = build_models(preprocessor)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metric = regression_metrics(y_test, y_pred)
            metric.update(
                {
                    "model": model_name,
                    "test_cell": test_cell,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }
            )

            all_metrics.append(metric)

            pred_df = pd.DataFrame(
                {
                    "model": model_name,
                    "test_cell": test_cell,
                    "cell_id": test_df["cell_id"].values,
                    "rpt_index": test_df["rpt_index"].values,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                }
            )

            all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, predictions_df


def plot_soh_degradation(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for cell, sub in df.groupby("cell_id"):
        sub = sub.sort_values("rpt_index")
        ax.plot(sub["rpt_index"], sub["soh"], marker="o", label=f"cell {cell}")

    ax.set_xlabel("RPT index")
    ax.set_ylabel("SOH")
    ax.set_title("SOH degradation by cell")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_predictions(predictions_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_df in predictions_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))
        cells = sorted(model_df["cell_id"].unique())
        cmap = plt.get_cmap("tab10")
        cell_colors = {
            cell: cmap(i % cmap.N)
            for i, cell in enumerate(cells)
        }

        for cell, sub in model_df.groupby("cell_id"):
            sub = sub.sort_values("rpt_index")
            true_color = cell_colors[cell]
            pred_color = lighten_color(true_color)
            ax.plot(
                sub["rpt_index"],
                sub["y_true"],
                marker="o",
                linestyle="-",
                color=true_color,
                label=f"cell {cell} true",
            )
            ax.plot(
                sub["rpt_index"],
                sub["y_pred"],
                marker="x",
                linestyle="--",
                color=pred_color,
                label=f"cell {cell} pred",
            )

        ax.set_xlabel("RPT index")
        ax.set_ylabel("Next SOH")
        ax.set_title(f"True vs predicted next SOH - {model_name}")
        ax.grid(True)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{model_name}.png", dpi=200)
        plt.close(fig)
