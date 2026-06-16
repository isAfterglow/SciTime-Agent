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

    metrics_path = out_dir / "summary_baseline_metrics.csv"
    predictions_path = out_dir / "summary_baseline_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("=" * 80)
    print("Metrics")
    print("=" * 80)
    print(metrics_df.sort_values(["model", "test_cell"]).to_string(index=False))

    print()
    print("=" * 80)
    print("Average metrics by model")
    print("=" * 80)

    avg_metrics = (
        metrics_df
        .groupby("model")[["mae", "rmse", "mape", "r2"]]
        .mean()
        .sort_values("rmse")
        .reset_index()
    )

    avg_metrics_path = out_dir / "summary_baseline_average_metrics.csv"
    avg_metrics.to_csv(avg_metrics_path, index=False)

    print(avg_metrics.to_string(index=False))

    plot_soh_degradation(
        rpt_df,
        fig_dir / "soh_degradation_by_cell.png",
    )

    plot_predictions(
        predictions_df,
        fig_dir,
    )

    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved average metrics to: {avg_metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()