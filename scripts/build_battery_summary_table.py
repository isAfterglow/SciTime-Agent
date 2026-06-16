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
)


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
    rpt_path = out_dir / "rpt_summary_table.csv"
    rpt_df.to_csv(rpt_path, index=False)

    next_df = build_next_soh_dataset(rpt_df)
    next_path = out_dir / "summary_next_soh_dataset.csv"
    next_df.to_csv(next_path, index=False)

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
    print("Columns:")
    for col in next_df.columns:
        print(f"- {col}")


if __name__ == "__main__":
    main()