from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.pipelines.battery_pipeline import BatteryPipeline
from src.pipelines.generic_pipeline import GenericPipeline
from src.pipelines.task_recommender import recommend_tasks_from_profile


@dataclass
class TaskSpec:
    task_name: str
    pipeline_type: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    tags: List[str]
    runner: Callable[..., Any]


def _run_generic_task(
    *,
    dataset_path: str,
    target_col: str,
    out_dir: str = "reports/generic/default_run",
    task_type: str | None = None,
    feature_cols: list[str] | None = None,
    include_models: list[str] | None = None,
):
    pipeline = GenericPipeline(
        dataset_path=dataset_path,
        target_col=target_col,
        out_dir=out_dir,
        task_type=task_type,
        feature_cols=feature_cols,
    )
    return pipeline.run_all(include_models=include_models)


def _run_battery_task(
    *,
    step: str = "all",
    config_path: str = "configs/battery_expt_2_2.yaml",
):
    pipeline = BatteryPipeline(config_path=config_path)
    if step == "all":
        results = pipeline.run_all()
        results.append(pipeline.generate_overview_report(results))
        return results
    if step == "audit":
        return [pipeline.audit_data()]
    if step == "build":
        return [pipeline.build_datasets()]
    if step == "baseline":
        return [pipeline.run_summary_baseline()]
    if step == "ablation":
        return [pipeline.run_feature_ablation()]
    if step == "strict":
        return [pipeline.run_strict_no_shortcut()]
    if step == "strict_cc0p1":
        return [pipeline.run_strict_with_cc0p1()]
    if step == "strict_cc0p1_pca":
        return [pipeline.run_strict_with_cc0p1_pca()]
    if step == "profile":
        return [pipeline.profile_summary_table()]
    raise ValueError(f"Unsupported battery step: {step}")


def _build_battery_step_runner(step: str) -> Callable[..., Any]:
    def _runner(*, config_path: str = "configs/battery_expt_2_2.yaml"):
        return _run_battery_task(step=step, config_path=config_path)

    return _runner


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "generic_tabular": TaskSpec(
        task_name="generic_tabular",
        pipeline_type="GenericPipeline",
        description="通用表格任务：上传 CSV 后做 profiling 和 baseline 训练。",
        required_params=["dataset_path", "target_col"],
        optional_params=["out_dir", "task_type", "feature_cols", "include_models"],
        tags=["tabular", "baseline", "generic"],
        runner=_run_generic_task,
    ),
    "battery_summary": TaskSpec(
        task_name="battery_summary",
        pipeline_type="BatteryPipeline",
        description="电池实验任务：支持 audit、build、baseline、ablation、strict 系列步骤。",
        required_params=[],
        optional_params=["config_path", "step"],
        tags=["battery", "summary", "aggregate"],
        runner=_run_battery_task,
    ),
    "battery_audit": TaskSpec(
        task_name="battery_audit",
        pipeline_type="BatteryPipeline",
        description="电池数据完整性审计。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "audit"],
        runner=_build_battery_step_runner("audit"),
    ),
    "battery_build": TaskSpec(
        task_name="battery_build",
        pipeline_type="BatteryPipeline",
        description="构建电池 summary 与监督数据集。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "build", "dataset"],
        runner=_build_battery_step_runner("build"),
    ),
    "battery_baseline": TaskSpec(
        task_name="battery_baseline",
        pipeline_type="BatteryPipeline",
        description="运行电池 summary baseline。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "baseline"],
        runner=_build_battery_step_runner("baseline"),
    ),
    "battery_feature_ablation": TaskSpec(
        task_name="battery_feature_ablation",
        pipeline_type="BatteryPipeline",
        description="运行电池特征消融实验。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "ablation", "feature"],
        runner=_build_battery_step_runner("ablation"),
    ),
    "battery_strict_no_shortcut": TaskSpec(
        task_name="battery_strict_no_shortcut",
        pipeline_type="BatteryPipeline",
        description="运行 strict no-shortcut 电池任务。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "strict", "no_shortcut"],
        runner=_build_battery_step_runner("strict"),
    ),
    "battery_strict_cc0p1": TaskSpec(
        task_name="battery_strict_cc0p1",
        pipeline_type="BatteryPipeline",
        description="运行 strict + 0.1C 曲线电池任务。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "strict", "cc0p1"],
        runner=_build_battery_step_runner("strict_cc0p1"),
    ),
    "battery_strict_cc0p1_pca": TaskSpec(
        task_name="battery_strict_cc0p1_pca",
        pipeline_type="BatteryPipeline",
        description="运行 strict + 0.1C PCA 电池任务。",
        required_params=[],
        optional_params=["config_path"],
        tags=["battery", "strict", "cc0p1", "pca"],
        runner=_build_battery_step_runner("strict_cc0p1_pca"),
    ),
}


def list_registered_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_name": spec.task_name,
            "pipeline_type": spec.pipeline_type,
            "description": spec.description,
            "required_params": spec.required_params,
            "optional_params": spec.optional_params,
            "tags": spec.tags,
        }
        for spec in TASK_REGISTRY.values()
    ]


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"Task not registered: {task_name}")
    return TASK_REGISTRY[task_name]


def run_registered_task(task_name: str, **kwargs):
    spec = get_task_spec(task_name)
    missing = [param for param in spec.required_params if param not in kwargs or kwargs[param] is None]
    if missing:
        raise ValueError(f"Missing required params for {task_name}: {missing}")
    return spec.runner(**kwargs)


def recommend_registered_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = recommend_tasks_from_profile(profile)
    enriched = []
    for item in recommendations:
        spec = get_task_spec(item["task_name"])
        enriched.append(
            {
                **item,
                "pipeline_type": spec.pipeline_type,
                "description": spec.description,
                "required_params": spec.required_params,
                "optional_params": spec.optional_params,
                "tags": spec.tags,
            }
        )
    return enriched
