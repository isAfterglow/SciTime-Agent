import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.battery_feature_sets import FEATURE_SETS
from src.pipelines.battery_summary_baseline import (
    leave_one_cell_out_evaluation,
    persistence_evaluation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/battery/expt_2_2/summary_next_soh_dataset.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_soh_next",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/battery/expt_2_2/feature_ablation",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["LinearRegression", "GradientBoosting", "RandomForest", "Ridge"],
        help="Models to run for learned baselines.",
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    out_dir = PROJECT_ROOT / args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("=" * 80)
    print("Feature ablation dataset")
    print("=" * 80)
    print(df.head())
    print()
    print(f"Shape: {df.shape}")
    print("Rows per cell:")
    print(df.groupby("cell_id").size())

    all_metrics = []
    all_predictions = []

    persistence_metrics, persistence_predictions = persistence_evaluation(
        df,
        target_col=args.target,
        persistence_col="soh",
    )
    persistence_metrics["feature_set"] = "persistence"
    persistence_predictions["feature_set"] = "persistence"
    all_metrics.append(persistence_metrics)
    all_predictions.append(persistence_predictions)

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        print()
        print("=" * 80)
        print(f"Running feature set: {feature_set_name}")
        print("=" * 80)
        print("Features:")
        for col in feature_cols:
            print(f"  - {col}")

        metrics_df, predictions_df = leave_one_cell_out_evaluation(
            df,
            target_col=args.target,
            feature_cols=feature_cols,
            include_models=args.models,
        )
        metrics_df["feature_set"] = feature_set_name
        predictions_df["feature_set"] = feature_set_name
        all_metrics.append(metrics_df)
        all_predictions.append(predictions_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    avg_metrics_df = (
        metrics_df
        .groupby(["feature_set", "model"])[["mae", "rmse", "mape", "r2"]]
        .mean()
        .sort_values(["feature_set", "rmse"])
        .reset_index()
    )

    metrics_path = out_dir / "ablation_metrics.csv"
    avg_metrics_path = out_dir / "ablation_average_metrics.csv"
    predictions_path = out_dir / "ablation_predictions.csv"
    manifest_path = out_dir / "ablation_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    avg_metrics_df.to_csv(avg_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    manifest = {
        "target": args.target,
        "dataset": args.dataset,
        "models": args.models,
        "feature_sets": {
            "persistence": ["soh -> y_pred"],
            **FEATURE_SETS,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plot_metric_bars(avg_metrics_df, "rmse", fig_dir / "ablation_rmse.png")
    plot_metric_bars(avg_metrics_df, "r2", fig_dir / "ablation_r2.png")
    plot_ablation_predictions(predictions_df, fig_dir)

    print()
    print("=" * 80)
    print("Ablation metrics")
    print("=" * 80)
    print(metrics_df.sort_values(["feature_set", "model", "test_cell"]).to_string(index=False))

    print()
    print("=" * 80)
    print("Average metrics by feature set and model")
    print("=" * 80)
    print(avg_metrics_df.to_string(index=False))

    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved average metrics to: {avg_metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved figures to: {fig_dir}")


def plot_metric_bars(avg_metrics_df: pd.DataFrame, metric: str, out_path: Path):
    feature_sets = avg_metrics_df["feature_set"].drop_duplicates().tolist()
    models = avg_metrics_df["model"].drop_duplicates().tolist()
    width = 0.8 / max(len(models), 1)
    x = list(range(len(feature_sets)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, model in enumerate(models):
        sub = (
            avg_metrics_df[avg_metrics_df["model"] == model]
            .set_index("feature_set")
            .reindex(feature_sets)
        )
        positions = [v - 0.4 + width / 2 + idx * width for v in x]
        ax.bar(positions, sub[metric].values, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Feature ablation comparison by {metric.upper()}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def lighten_color(color, amount: float = 0.45):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_ablation_predictions(predictions_df: pd.DataFrame, out_dir: Path):
    for feature_set, feature_df in predictions_df.groupby("feature_set"):
        for model_name, model_df in feature_df.groupby("model"):
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
            ax.set_title(f"{feature_set} - {model_name}")
            ax.grid(True)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"pred_vs_true_{feature_set}_{model_name}.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
