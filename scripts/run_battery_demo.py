import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.battery_pipeline import BatteryPipeline, format_step_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/battery_expt_2_2.yaml",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=[
            "all",
            "audit",
            "build",
            "baseline",
            "ablation",
            "strict",
            "strict_cc0p1",
            "strict_cc0p1_pca",
            "profile",
        ],
    )
    args = parser.parse_args()

    pipeline = BatteryPipeline(config_path=args.config)

    if args.step == "all":
        results = pipeline.run_all()
        results.append(pipeline.generate_overview_report(results))
    elif args.step == "audit":
        results = [pipeline.audit_data()]
    elif args.step == "build":
        results = [pipeline.build_datasets()]
    elif args.step == "baseline":
        results = [pipeline.run_summary_baseline()]
    elif args.step == "ablation":
        results = [pipeline.run_feature_ablation()]
    elif args.step == "strict":
        results = [pipeline.run_strict_no_shortcut()]
    elif args.step == "strict_cc0p1":
        results = [pipeline.run_strict_with_cc0p1()]
    elif args.step == "strict_cc0p1_pca":
        results = [pipeline.run_strict_with_cc0p1_pca()]
    elif args.step == "profile":
        profile = pipeline.profile_summary_table()
        print("=" * 80)
        print("Battery summary profile")
        print("=" * 80)
        for key, value in profile.items():
            print(f"{key}: {value}")
        return
    else:
        raise ValueError(f"Unsupported step: {args.step}")

    print("=" * 80)
    print("Battery pipeline summary")
    print("=" * 80)
    print(format_step_results(results))


if __name__ == "__main__":
    main()
