from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.tools.data_profiler import profile_dataframe
from src.tools.model_trainer import run_tabular_baselines
from src.tools.report_generator import generate_markdown_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GenericStepResult:
    name: str
    outputs: Dict[str, str]


class GenericPipeline:
    """
    Generic tabular workflow for dataset profiling, baseline training,
    prediction export, and report generation.
    """

    def __init__(
        self,
        dataset_path: str,
        target_col: str,
        out_dir: str = "reports/generic/default_run",
        task_type: str | None = None,
        feature_cols: List[str] | None = None,
    ):
        self.dataset_path = PROJECT_ROOT / dataset_path
        self.target_col = target_col
        self.out_dir = PROJECT_ROOT / out_dir
        self.task_type = task_type
        self.feature_cols = feature_cols

    def load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if self.dataset_path.suffix.lower() == ".csv":
            return pd.read_csv(self.dataset_path)
        if self.dataset_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(self.dataset_path)
        raise ValueError(f"Unsupported dataset format: {self.dataset_path.suffix}")

    def profile_data(self) -> GenericStepResult:
        df = self.load_dataset()
        profile = profile_dataframe(df)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = generate_markdown_report(
            title="Generic Data Profile",
            sections=[
                {
                    "header": "Dataset",
                    "kv": {
                        "dataset_path": str(self.dataset_path),
                        "target_col": self.target_col,
                    },
                },
                {
                    "header": "Profile",
                    "kv": profile,
                },
            ],
            out_path=self.out_dir / "data_profile.md",
        )
        return GenericStepResult(
            name="profile_data",
            outputs={
                "profile_report": str(report_path),
            },
        )

    def run_baselines(
        self,
        include_models: List[str] | None = None,
    ) -> GenericStepResult:
        df = self.load_dataset()
        result = run_tabular_baselines(
            df=df,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            task_type=self.task_type,
            include_models=include_models,
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.out_dir / "metrics.csv"
        predictions_path = self.out_dir / "predictions.csv"
        manifest_path = self.out_dir / "manifest.json"

        result.metrics_df.to_csv(metrics_path, index=False)
        result.predictions_df.to_csv(predictions_path, index=False)
        manifest = {
            "dataset_path": str(self.dataset_path),
            "target_col": self.target_col,
            "task_type": result.task_type,
            "feature_cols": result.feature_cols,
            "numeric_cols": result.numeric_cols,
            "categorical_cols": result.categorical_cols,
            "models": include_models,
        }
        pd.Series(manifest).to_json(manifest_path, force_ascii=False, indent=2)

        report_path = generate_markdown_report(
            title="Generic Baseline Report",
            sections=[
                {
                    "header": "Run Config",
                    "kv": {
                        "dataset_path": str(self.dataset_path),
                        "target_col": self.target_col,
                        "task_type": result.task_type,
                        "n_features": len(result.feature_cols),
                    },
                },
                {
                    "header": "Feature Types",
                    "kv": {
                        "numeric_cols": ", ".join(result.numeric_cols),
                        "categorical_cols": ", ".join(result.categorical_cols),
                    },
                },
                {
                    "header": "Metrics",
                    "body": result.metrics_df.to_string(index=False),
                },
                {
                    "header": "Artifacts",
                    "kv": {
                        "metrics": str(metrics_path),
                        "predictions": str(predictions_path),
                        "manifest": str(manifest_path),
                    },
                },
            ],
            out_path=self.out_dir / "report.md",
        )

        return GenericStepResult(
            name="run_baselines",
            outputs={
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
                "manifest": str(manifest_path),
                "report": str(report_path),
            },
        )

    def run_all(self, include_models: List[str] | None = None) -> List[GenericStepResult]:
        return [
            self.profile_data(),
            self.run_baselines(include_models=include_models),
        ]


def format_generic_results(results: List[GenericStepResult]) -> str:
    lines = []
    for result in results:
        lines.append(f"[{result.name}]")
        for key, value in result.outputs.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).strip()
