import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.generic_pipeline import GenericPipeline, format_generic_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/demo/generic_regression_demo.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_value",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/generic/demo_run",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=["regression", "classification", None],
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "profile", "train"],
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
    )
    args = parser.parse_args()

    pipeline = GenericPipeline(
        dataset_path=args.dataset,
        target_col=args.target,
        out_dir=args.out_dir,
        task_type=args.task_type,
    )

    if args.step == "all":
        results = pipeline.run_all(include_models=args.models)
    elif args.step == "profile":
        results = [pipeline.profile_data()]
    elif args.step == "train":
        results = [pipeline.run_baselines(include_models=args.models)]
    else:
        raise ValueError(f"Unsupported step: {args.step}")

    print("=" * 80)
    print("Generic pipeline summary")
    print("=" * 80)
    print(format_generic_results(results))


if __name__ == "__main__":
    main()
