import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.battery_loader import BatteryExptLoader
from src.features.battery_summary_features import (
    build_rpt_summary_table,
    build_next_soh_dataset,
    build_horizon_soh_dataset,
    build_early_point_to_last_dataset,
    build_early_trajectory_to_last_dataset,
)
from src.features.battery_curve_features import build_cc_0p1c_feature_table
from src.features.battery_curve_pca import build_cc0p1_curve_vector_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/battery_expt_2_2.yaml",
    )
    args = parser.parse_args()

    loader = BatteryExptLoader(args.config)

    out_dir = PROJECT_ROOT / "data" / "processed" / "battery" / "expt_2_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rpt_df = build_rpt_summary_table(loader)
    cc0p1_df = build_cc_0p1c_feature_table(loader)
    cc0p1_vec_df = build_cc0p1_curve_vector_table(loader)
    rpt_df = rpt_df.merge(cc0p1_df, on=["cell_id", "rpt_index"], how="left")
    rpt_df = rpt_df.merge(cc0p1_vec_df, on=["cell_id", "rpt_index"], how="left")
    rpt_path = out_dir / "rpt_summary_table.csv"
    rpt_df.to_csv(rpt_path, index=False)

    next_df = build_next_soh_dataset(rpt_df)
    next_path = out_dir / "summary_next_soh_dataset.csv"
    next_df.to_csv(next_path, index=False)

    horizon3_df = build_horizon_soh_dataset(rpt_df, horizon=3)
    horizon3_path = out_dir / "summary_horizon_3_dataset.csv"
    horizon3_df.to_csv(horizon3_path, index=False)

    early_point_df = build_early_point_to_last_dataset(rpt_df, early_rpt_max=3)
    early_point_path = out_dir / "summary_early_point_to_last_dataset.csv"
    early_point_df.to_csv(early_point_path, index=False)

    early_traj_df = build_early_trajectory_to_last_dataset(rpt_df, early_rpt_max=3)
    early_traj_path = out_dir / "summary_early_trajectory_to_last_dataset.csv"
    early_traj_df.to_csv(early_traj_path, index=False)

    print("=" * 80)
    print("RPT summary table")
    print("=" * 80)
    print(rpt_df.head())
    print()
    print(f"Shape: {rpt_df.shape}")
    print(f"Saved to: {rpt_path}")
    print()
    print("Rows per cell:")
    print(rpt_df.groupby("cell_id").size())

    print()
    print("=" * 80)
    print("Next-SOH supervised dataset")
    print("=" * 80)
    print(next_df.head())
    print()
    print(f"Shape: {next_df.shape}")
    print(f"Saved to: {next_path}")
    print()
    print("Rows per cell:")
    print(next_df.groupby("cell_id").size())

    print()
    print("=" * 80)
    print("Horizon-3 supervised dataset")
    print("=" * 80)
    print(horizon3_df.head())
    print()
    print(f"Shape: {horizon3_df.shape}")
    print(f"Saved to: {horizon3_path}")
    print()
    print("Rows per cell:")
    print(horizon3_df.groupby("cell_id").size())

    print()
    print("=" * 80)
    print("Early-point-to-last supervised dataset")
    print("=" * 80)
    print(early_point_df.head())
    print()
    print(f"Shape: {early_point_df.shape}")
    print(f"Saved to: {early_point_path}")
    print()
    print("Rows per cell:")
    print(early_point_df.groupby("cell_id").size())

    print()
    print("=" * 80)
    print("Early-trajectory-to-last supervised dataset")
    print("=" * 80)
    print(early_traj_df.head())
    print()
    print(f"Shape: {early_traj_df.shape}")
    print(f"Saved to: {early_traj_path}")
    if "cell_id" in early_traj_df.columns:
        print()
        print("Rows per cell:")
        print(early_traj_df.groupby("cell_id").size())

    print()
    print("Columns:")
    for col in next_df.columns:
        print(f"- {col}")


if __name__ == "__main__":
    main()
