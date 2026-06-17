import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.battery_feature_sets import PROCESS_PLUS_RESISTANCE_FEATURES
from src.pipelines.battery_summary_baseline import leave_one_cell_out_evaluation
from src.tools.evaluator import average_metrics, save_metrics_bundle, save_predictions
from src.tools.report_generator import generate_markdown_report


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


def plot_metric_bars(avg_metrics_df: pd.DataFrame, metric: str, out_path: Path, title: str):
    models = avg_metrics_df["model"].drop_duplicates().tolist()
    width = 0.8 / max(len(models), 1)
    x = [0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, model in enumerate(models):
        sub = avg_metrics_df[avg_metrics_df["model"] == model]
        positions = [v - 0.4 + width / 2 + idx * width for v in x]
        ax.bar(positions, sub[metric].values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([title], rotation=0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{title} by {metric.upper()}")
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
        cell_colors = {cell: cmap(i % cmap.N) for i, cell in enumerate(cells)}
        for cell, sub in model_df.groupby("cell_id"):
            sub = sub.sort_values("rpt_index")
            true_color = cell_colors[cell]
            pred_color = lighten_color(true_color)
            ax.plot(sub["rpt_index"], sub["y_true"], marker="o", linestyle="-", color=true_color, label=f"cell {cell} true")
            ax.plot(sub["rpt_index"], sub["y_pred"], marker="x", linestyle="--", color=pred_color, label=f"cell {cell} pred")
        ax.set_xlabel("RPT index")
        ax.set_ylabel("Target SOH")
        ax.set_title(f"{task_name} - {model_name}")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{task_name}_{model_name}.png", dpi=200)
        plt.close(fig)


def add_foldwise_pca_features(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    df = df.copy()
    grid_cols = [col for col in df.columns if col.startswith("cc0p1_grid_")]
    if not grid_cols:
        return df

    cells = sorted(df["cell_id"].unique())
    all_parts = []
    for test_cell in cells:
        train_mask = df["cell_id"] != test_cell
        test_mask = df["cell_id"] == test_cell

        train_grid = df.loc[train_mask, grid_cols].copy()
        test_grid = df.loc[test_mask, grid_cols].copy()

        # Median fill on train fold only.
        med = train_grid.median(axis=0)
        train_grid = train_grid.fillna(med)
        test_grid = test_grid.fillna(med)

        pca = PCA(n_components=min(n_components, len(grid_cols), len(train_grid)))
        train_pca = pca.fit_transform(train_grid)
        test_pca = pca.transform(test_grid)

        train_part = df.loc[train_mask].copy()
        test_part = df.loc[test_mask].copy()

        for i in range(train_pca.shape[1]):
            train_part[f"cc0p1_pca_{i+1}"] = train_pca[:, i]
            test_part[f"cc0p1_pca_{i+1}"] = test_pca[:, i]

        all_parts.append(train_part)
        all_parts.append(test_part)

    out = pd.concat(all_parts, ignore_index=True)
    out = out.drop(columns=grid_cols, errors="ignore")
    out = out.drop_duplicates(subset=["cell_id"] + (["rpt_index"] if "rpt_index" in out.columns else []), keep="last")
    return out.sort_values(["cell_id"] + (["rpt_index"] if "rpt_index" in out.columns else [])).reset_index(drop=True)


def run_variant(task_name: str, cfg: dict, models: list[str], out_root: Path, variant_name: str, include_handcrafted: bool):
    df = pd.read_csv(PROJECT_ROOT / cfg["dataset"])
    df = add_foldwise_pca_features(df, n_components=5)

    if "trajectory" in task_name:
        base_feature_cols = [
            "temperature",
            "soc_range",
            *[col for col in df.columns if col.startswith("charge_throughput_rpt")],
            *[col for col in df.columns if col.startswith("energy_throughput_rpt")],
            *[col for col in df.columns if col.startswith("resistance_0p1s_rpt")],
            *[col for col in df.columns if col.startswith("delta_charge_throughput_rpt")],
            *[col for col in df.columns if col.startswith("resistance_delta_prev_rpt")],
            *[col for col in df.columns if col.startswith("ageing_cycles_rpt")],
            *[col for col in df.columns if col.startswith("days_of_degradation_rpt")],
            *[col for col in df.columns if col.startswith("cc0p1_pca_")],
        ]
        if include_handcrafted:
            base_feature_cols += [col for col in df.columns if col.startswith("cc0p1_") and not col.startswith("cc0p1_grid_") and not col.startswith("cc0p1_pca_")]
    else:
        base_feature_cols = PROCESS_PLUS_RESISTANCE_FEATURES + [col for col in df.columns if col.startswith("cc0p1_pca_")]
        if include_handcrafted:
            base_feature_cols += [col for col in df.columns if col.startswith("cc0p1_") and not col.startswith("cc0p1_grid_") and not col.startswith("cc0p1_pca_")]

    out_dir = out_root / task_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, predictions_df = leave_one_cell_out_evaluation(
        df,
        target_col=cfg["target"],
        feature_cols=base_feature_cols,
        include_models=models,
    )
    avg_metrics_df = average_metrics(metrics_df, group_cols=["model"]).sort_values("rmse").reset_index(drop=True)
    metric_outputs = save_metrics_bundle(
        metrics_df,
        out_dir,
        metrics_filename="metrics.csv",
        average_filename="average_metrics.csv",
        group_cols=["model"],
    )
    prediction_outputs = save_predictions(
        predictions_df,
        out_dir,
        filename="predictions.csv",
    )
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "task": task_name,
            "dataset": cfg["dataset"],
            "target": cfg["target"],
            "variant": variant_name,
            "include_handcrafted": include_handcrafted,
            "models": models,
            "feature_columns": base_feature_cols,
        }, f, indent=2)

    plot_metric_bars(avg_metrics_df, "rmse", fig_dir / "rmse.png", variant_name)
    plot_metric_bars(avg_metrics_df, "r2", fig_dir / "r2.png", variant_name)
    plot_predictions(predictions_df, fig_dir, task_name)

    report_path = generate_markdown_report(
        title=f"{variant_name} - {task_name}",
        sections=[
            {
                "header": "Run Config",
                "kv": {
                    "task": task_name,
                    "dataset": cfg["dataset"],
                    "target": cfg["target"],
                    "variant": variant_name,
                    "include_handcrafted": include_handcrafted,
                    "models": ", ".join(models),
                },
            },
            {
                "header": "Dataset Shape",
                "kv": {
                    "n_rows": df.shape[0],
                    "n_cols": df.shape[1],
                },
            },
            {
                "header": "Average Metrics",
                "body": avg_metrics_df.to_string(index=False),
            },
            {
                "header": "Artifacts",
                "kv": {
                    **metric_outputs,
                    **prediction_outputs,
                    "manifest": str(out_dir / "manifest.json"),
                    "figures_dir": str(fig_dir),
                },
            },
        ],
        out_path=out_dir / "report.md",
    )

    print()
    print("=" * 80)
    print(f"{variant_name} task: {task_name}")
    print("=" * 80)
    print(avg_metrics_df.to_string(index=False))
    print(f"Saved metrics to: {metric_outputs['metrics']}")
    print(f"Saved average metrics to: {metric_outputs['average_metrics']}")
    print(f"Saved predictions to: {prediction_outputs['predictions']}")
    print(f"Saved figures to: {fig_dir}")
    print(f"Saved report to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=["LinearRegression", "GradientBoosting", "RandomForest", "Ridge"],
    )
    args = parser.parse_args()

    variant_roots = {
        "strict_with_cc0p1_pca": False,
        "strict_with_cc0p1_handcrafted_pca": True,
    }

    for root_name, include_handcrafted in variant_roots.items():
        out_root = PROJECT_ROOT / "reports" / "battery" / "expt_2_2" / root_name
        out_root.mkdir(parents=True, exist_ok=True)
        for task_name, cfg in TASKS.items():
            run_variant(task_name, cfg, args.models, out_root, root_name, include_handcrafted)


if __name__ == "__main__":
    main()
