import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.battery_summary_baseline import (
    leave_one_cell_out_evaluation,
    plot_soh_degradation,
    plot_predictions,
)
from src.tools.evaluator import average_metrics, save_metrics_bundle, save_predictions
from src.tools.report_generator import generate_markdown_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/battery/expt_2_2/summary_next_soh_dataset.csv",
    )
    parser.add_argument(
        "--rpt-table",
        type=str,
        default="data/processed/battery/expt_2_2/rpt_summary_table.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_soh_next",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/battery/expt_2_2/summary_baseline",
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    rpt_table_path = PROJECT_ROOT / args.rpt_table
    out_dir = PROJECT_ROOT / args.out_dir
    fig_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please run: python scripts/build_battery_summary_table.py"
        )

    df = pd.read_csv(dataset_path)
    rpt_df = pd.read_csv(rpt_table_path)

    print("=" * 80)
    print("Loaded supervised dataset")
    print("=" * 80)
    print(df.head())
    print()
    print(f"Shape: {df.shape}")
    print("Rows per cell:")
    print(df.groupby("cell_id").size())
    print()

    metrics_df, predictions_df = leave_one_cell_out_evaluation(
        df,
        target_col=args.target,
    )

    metric_outputs = save_metrics_bundle(
        metrics_df,
        out_dir,
        metrics_filename="summary_baseline_metrics.csv",
        average_filename="summary_baseline_average_metrics.csv",
        group_cols=["model"],
    )
    prediction_outputs = save_predictions(
        predictions_df,
        out_dir,
        filename="summary_baseline_predictions.csv",
    )

    print("=" * 80)
    print("Metrics")
    print("=" * 80)
    print(metrics_df.sort_values(["model", "test_cell"]).to_string(index=False))

    print()
    print("=" * 80)
    print("Average metrics by model")
    print("=" * 80)

    avg_metrics = average_metrics(metrics_df, group_cols=["model"]).sort_values("rmse").reset_index(drop=True)

    print(avg_metrics.to_string(index=False))

    plot_soh_degradation(
        rpt_df,
        fig_dir / "soh_degradation_by_cell.png",
    )

    plot_predictions(
        predictions_df,
        fig_dir,
    )

    report_path = generate_markdown_report(
        title="Battery Summary Baseline",
        sections=[
            {
                "header": "Run Config",
                "kv": {
                    "dataset": str(dataset_path),
                    "rpt_table": str(rpt_table_path),
                    "target": args.target,
                    "output_dir": str(out_dir),
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
                "body": avg_metrics.to_string(index=False),
            },
            {
                "header": "Artifacts",
                "kv": {
                    **metric_outputs,
                    **prediction_outputs,
                    "figures_dir": str(fig_dir),
                },
            },
        ],
        out_path=out_dir / "summary_baseline_report.md",
    )

    print()
    print(f"Saved metrics to: {metric_outputs['metrics']}")
    print(f"Saved average metrics to: {metric_outputs['average_metrics']}")
    print(f"Saved predictions to: {prediction_outputs['predictions']}")
    print(f"Saved figures to: {fig_dir}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
