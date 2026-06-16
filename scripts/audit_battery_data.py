import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.battery_loader import BatteryExptLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/battery_expt_2_2.yaml",
        help="Path to battery dataset config file."
    )
    args = parser.parse_args()

    loader = BatteryExptLoader(args.config)

    print("=" * 80)
    print("Battery data audit")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Root dir: {loader.root_dir}")
    print(f"Experiment dir: {loader.experiment_dir}")
    print(f"Metadata path: {loader.metadata_path}")
    print()

    print("Metadata:")
    print(loader.metadata)
    print()

    rows = []
    for cell in loader.cells:
        print(f"Auditing cell {cell} ...")
        row = loader.audit_cell(cell)
        rows.append(row)

    df = pd.DataFrame(rows)

    display_cols = [
        "cell_id",
        "main_summary_exists",
        "set_summary_exists",
        "expected_rpt_count",
        "n_cc_0p1c",
        "n_cc_0p5c",
        "n_gitt_25p",
        "n_gitt_5p",
        "n_hybrid_0p5c",
        "n_hybrid_1c",
        "n_missing_files",
        "is_complete",
    ]

    print()
    print("=" * 80)
    print("Audit summary")
    print("=" * 80)
    print(df[display_cols].to_string(index=False))

    out_dir = PROJECT_ROOT / "reports" / "battery" / "expt_2_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "data_audit_report.csv"
    df.to_csv(out_csv, index=False)

    out_missing = out_dir / "missing_files.txt"
    with out_missing.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"\n===== Cell {row['cell_id']} =====\n")
            missing_files = row["missing_files"]
            if isinstance(missing_files, list):
                for item in missing_files:
                    f.write(str(item) + "\n")

    print()
    print(f"Saved audit report to: {out_csv}")
    print(f"Saved missing file list to: {out_missing}")

    if df["is_complete"].all():
        print("\nAll expected files are complete.")
    else:
        print("\nSome files are missing. Check missing_files.txt for details.")


if __name__ == "__main__":
    main()