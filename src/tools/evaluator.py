from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def average_metrics(metrics_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["model"]
    return (
        metrics_df
        .groupby(group_cols)[["mae", "rmse", "mape", "r2"]]
        .mean()
        .reset_index()
    )


def save_metrics_bundle(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    metrics_filename: str = "metrics.csv",
    average_filename: str = "average_metrics.csv",
    group_cols: list[str] | None = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    avg_df = average_metrics(metrics_df, group_cols=group_cols)
    metrics_path = out_dir / metrics_filename
    average_path = out_dir / average_filename
    metrics_df.to_csv(metrics_path, index=False)
    avg_df.to_csv(average_path, index=False)
    return {
        "metrics": str(metrics_path),
        "average_metrics": str(average_path),
    }


def save_predictions(
    predictions_df: pd.DataFrame,
    out_dir: Path,
    filename: str = "predictions.csv",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    predictions_df.to_csv(path, index=False)
    return {"predictions": str(path)}


def plot_grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    x_values = df[x_col].drop_duplicates().tolist()
    hue_values = df[hue_col].drop_duplicates().tolist()
    width = 0.8 / max(len(hue_values), 1)
    x = list(range(len(x_values)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, hue in enumerate(hue_values):
        sub = df[df[hue_col] == hue].set_index(x_col).reindex(x_values)
        positions = [v - 0.4 + width / 2 + idx * width for v in x]
        ax.bar(positions, sub[y_col].values, width=width, label=hue)

    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
