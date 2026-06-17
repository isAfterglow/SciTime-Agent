import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.battery_pipeline import format_step_results
from src.pipelines.generic_pipeline import format_generic_results
from src.pipelines.task_recommender import recommend_tasks_from_profile
from src.pipelines.task_registry import get_task_spec, list_registered_tasks, run_registered_task
from src.tools.data_profiler import profile_dataframe
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all registered tasks.")
    parser.add_argument("--recommend", action="store_true", help="Recommend tasks for a dataset.")
    parser.add_argument("--task", type=str, help="Registered task name.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None)
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--config", type=str, default="configs/battery_expt_2_2.yaml")
    parser.add_argument("--step", type=str, default="all")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_registered_tasks(), ensure_ascii=False, indent=2))
        return

    if args.recommend:
        if not args.dataset:
            raise ValueError("Please provide --dataset when using --recommend.")
        dataset_path = PROJECT_ROOT / args.dataset
        df = pd.read_csv(dataset_path)
        profile = profile_dataframe(df, target_col=args.target)
        recommendations = recommend_tasks_from_profile(profile)
        print(json.dumps(recommendations, ensure_ascii=False, indent=2))
        return

    if not args.task:
        raise ValueError("Please provide --task or use --list.")

    spec = get_task_spec(args.task)
    if args.task == "generic_tabular":
        results = run_registered_task(
            args.task,
            dataset_path=args.dataset,
            target_col=args.target,
            out_dir=args.out_dir or "reports/generic/registered_run",
            task_type=args.task_type,
            feature_cols=args.feature_cols,
            include_models=args.models,
        )
        print("=" * 80)
        print(spec.description)
        print("=" * 80)
        print(format_generic_results(results))
        return

    if args.task == "battery_summary":
        results = run_registered_task(
            args.task,
            config_path=args.config,
            step=args.step,
        )
        print("=" * 80)
        print(spec.description)
        print("=" * 80)
        if args.step == "profile":
            print(results[0])
        else:
            print(format_step_results(results))
        return

    raise ValueError(f"Unsupported task handler: {args.task}")


if __name__ == "__main__":
    main()
