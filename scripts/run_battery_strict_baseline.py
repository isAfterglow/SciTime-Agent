import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.battery_feature_sets import PROCESS_PLUS_RESISTANCE_FEATURES
from src.pipelines.battery_summary_baseline import leave_one_cell_out_evaluation


TASKS = {
    "next_step": {
        "dataset": "data/processed/battery/expt_2_2/summary_next_soh_dataset.csv",
        "target": "target_soh_next",
    },
    "horizon_3": {
        "dataset": "data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv",
        "target": "target_soh_h3",
    },
    "early_point_to_last": {
        "dataset": "data/processed/battery/expt_2_2/summary_early_point_to_last_dataset.csv",
        "target": "target_soh_last",
    },
    "early_trajectory_to_last": {
        "dataset": "data/processed/battery/expt_2_2/summary_early_trajectory_to_last_dataset.csv",
        "target": "target_soh_last",
    },
}


def lighten_color(color, amount: float = 0.45):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_metric_bars(avg_metrics_df: pd.DataFrame, metric: str, out_path: Path):
    models = avg_metrics_df["model"].drop_duplicates().tolist()
    width = 0.8 / max(len(models), 1)
    x = [0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, model in enumerate(models):
        sub = avg_metrics_df[avg_metrics_df["model"] == model]
        positions = [v - 0.4 + width / 2 + idx * width for v in x]
        ax.bar(positions, sub[metric].values, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(["process_plus_resistance"], rotation=0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Strict no-shortcut comparison by {metric.upper()}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_predictions(predictions_df: pd.DataFrame, out_dir: Path, task_name: str):
    for model_name, model_df in predictions_df.groupby("model"):
        if "rpt_index" not in model_df.columns:
            continue
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
        ax.set_ylabel("Target SOH")
        ax.set_title(f"{task_name} - {model_name}")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{task_name}_{model_name}.png", dpi=200)
        plt.close(fig)


def run_one_task(task_name: str, dataset_rel: str, target_col: str, models: list[str], out_root: Path):
    dataset_path = PROJECT_ROOT / dataset_rel
    out_dir = out_root / task_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    metrics_df, predictions_df = leave_one_cell_out_evaluation(
        df,
        target_col=target_col,
        feature_cols=PROCESS_PLUS_RESISTANCE_FEATURES,
        include_models=models,
    )

    avg_metrics_df = (
        metrics_df
        .groupby(["model"])[["mae", "rmse", "mape", "r2"]]
        .mean()
        .sort_values("rmse")
        .reset_index()
    )

    metrics_path = out_dir / "strict_metrics.csv"
    avg_metrics_path = out_dir / "strict_average_metrics.csv"
    predictions_path = out_dir / "strict_predictions.csv"
    manifest_path = out_dir / "strict_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    avg_metrics_df.to_csv(avg_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    manifest = {
        "task": task_name,
        "dataset": dataset_rel,
        "target": target_col,
        "feature_set": "process_plus_resistance",
        "feature_columns": PROCESS_PLUS_RESISTANCE_FEATURES,
        "models": models,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plot_metric_bars(avg_metrics_df, "rmse", fig_dir / "strict_rmse.png")
    plot_metric_bars(avg_metrics_df, "r2", fig_dir / "strict_r2.png")
    plot_predictions(predictions_df, fig_dir, task_name)

    print()
    print("=" * 80)
    print(f"Strict no-shortcut task: {task_name}")
    print("=" * 80)
    print(f"Dataset: {dataset_rel}")
    print(f"Target: {target_col}")
    print(f"Shape: {df.shape}")
    print("Rows per cell:")
    print(df.groupby('cell_id').size())
    print()
    print(avg_metrics_df.to_string(index=False))
    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved average metrics to: {avg_metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved figures to: {fig_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/battery/expt_2_2/strict_no_shortcut",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["LinearRegression", "GradientBoosting", "RandomForest", "Ridge"],
    )
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for task_name, cfg in TASKS.items():
        run_one_task(
            task_name=task_name,
            dataset_rel=cfg["dataset"],
            target_col=cfg["target"],
            models=args.models,
            out_root=out_root,
        )


if __name__ == "__main__":
    main()
