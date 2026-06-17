from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import pandas as pd

from src.data.battery_loader import BatteryExptLoader
from src.tools.data_profiler import profile_dataframe
from src.tools.report_generator import generate_markdown_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StepResult:
    name: str
    outputs: Dict[str, str]


class BatteryPipeline:
    """
    Unified battery workflow entry for audit, dataset building, baselines,
    ablation, and strict no-shortcut experiments.
    """

    def __init__(self, config_path: str = "configs/battery_expt_2_2.yaml", python_env: str = "scitime-agent"):
        self.config_path = config_path
        self.python_env = python_env
        self.loader = BatteryExptLoader(self.config_path)
        self.dataset_name = self.loader.cfg.get("dataset_name", "battery_expt")

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / "battery" / "expt_2_2"

    @property
    def report_dir(self) -> Path:
        return PROJECT_ROOT / "reports" / "battery" / "expt_2_2"

    def profile_summary_table(self) -> Dict:
        rpt_path = self.processed_dir / "rpt_summary_table.csv"
        if not rpt_path.exists():
            raise FileNotFoundError(f"Missing summary table: {rpt_path}")
        df = pd.read_csv(rpt_path)
        return profile_dataframe(df)

    def audit_data(self) -> StepResult:
        loader = self.loader
        rows = [loader.audit_cell(cell) for cell in loader.cells]
        df = pd.DataFrame(rows)

        out_dir = self.report_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "data_audit_report.csv"
        out_missing = out_dir / "missing_files.txt"

        df.to_csv(out_csv, index=False)
        with out_missing.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f"\n===== Cell {row['cell_id']} =====\n")
                missing_files = row["missing_files"]
                if isinstance(missing_files, list):
                    for item in missing_files:
                        f.write(str(item) + "\n")

        return StepResult(
            name="audit_data",
            outputs={
                "report_csv": str(out_csv),
                "missing_files_txt": str(out_missing),
            },
        )

    def build_datasets(self) -> StepResult:
        self._run_script("scripts/build_battery_summary_table.py", "--config", self.config_path)
        outputs = {
            "rpt_summary_table": str(self.processed_dir / "rpt_summary_table.csv"),
            "next_step_dataset": str(self.processed_dir / "summary_next_soh_dataset.csv"),
            "horizon_3_dataset": str(self.processed_dir / "summary_horizon_3_dataset.csv"),
            "early_point_to_last_dataset": str(self.processed_dir / "summary_early_point_to_last_dataset.csv"),
            "early_trajectory_to_last_dataset": str(self.processed_dir / "summary_early_trajectory_to_last_dataset.csv"),
        }
        return StepResult(name="build_datasets", outputs=outputs)

    def run_summary_baseline(self) -> StepResult:
        self._run_script("scripts/run_battery_summary_baseline.py")
        out_dir = self.report_dir / "summary_baseline"
        return StepResult(
            name="run_summary_baseline",
            outputs={
                "metrics": str(out_dir / "summary_baseline_metrics.csv"),
                "average_metrics": str(out_dir / "summary_baseline_average_metrics.csv"),
                "predictions": str(out_dir / "summary_baseline_predictions.csv"),
                "figures_dir": str(out_dir / "figures"),
            },
        )

    def run_feature_ablation(self) -> StepResult:
        self._run_script("scripts/run_battery_feature_ablation.py")
        out_dir = self.report_dir / "feature_ablation"
        return StepResult(
            name="run_feature_ablation",
            outputs={
                "metrics": str(out_dir / "ablation_metrics.csv"),
                "average_metrics": str(out_dir / "ablation_average_metrics.csv"),
                "predictions": str(out_dir / "ablation_predictions.csv"),
                "manifest": str(out_dir / "ablation_manifest.json"),
                "figures_dir": str(out_dir / "figures"),
            },
        )

    def run_strict_no_shortcut(self) -> StepResult:
        self._run_script("scripts/run_battery_strict_baseline.py")
        out_dir = self.report_dir / "strict_no_shortcut"
        return StepResult(
            name="run_strict_no_shortcut",
            outputs={"root_dir": str(out_dir)},
        )

    def run_strict_with_cc0p1(self) -> StepResult:
        self._run_script("scripts/run_battery_strict_curve_baseline.py")
        out_dir = self.report_dir / "strict_with_cc0p1"
        return StepResult(
            name="run_strict_with_cc0p1",
            outputs={"root_dir": str(out_dir)},
        )

    def run_strict_with_cc0p1_pca(self) -> StepResult:
        self._run_script("scripts/run_battery_strict_curve_pca_baselines.py")
        return StepResult(
            name="run_strict_with_cc0p1_pca",
            outputs={
                "pca_root_dir": str(self.report_dir / "strict_with_cc0p1_pca"),
                "handcrafted_pca_root_dir": str(self.report_dir / "strict_with_cc0p1_handcrafted_pca"),
            },
        )

    def run_all(self) -> List[StepResult]:
        results = [
            self.audit_data(),
            self.build_datasets(),
            self.run_summary_baseline(),
            self.run_feature_ablation(),
            self.run_strict_no_shortcut(),
            self.run_strict_with_cc0p1(),
            self.run_strict_with_cc0p1_pca(),
        ]
        return results

    def generate_overview_report(self, results: List[StepResult] | None = None) -> StepResult:
        if results is None:
            results = []
        report_path = self.report_dir / "battery_pipeline_overview.md"
        sections = []
        sections.append(
            {
                "header": "Pipeline Config",
                "kv": {
                    "config_path": self.config_path,
                    "dataset_name": self.dataset_name,
                    "processed_dir": str(self.processed_dir),
                    "report_dir": str(self.report_dir),
                },
            }
        )
        try:
            profile = self.profile_summary_table()
            sections.append(
                {
                    "header": "Summary Table Profile",
                    "kv": {
                        "n_rows": profile["n_rows"],
                        "n_cols": profile["n_cols"],
                        "numeric_cols": len(profile["numeric_cols"]),
                        "categorical_cols": len(profile["categorical_cols"]),
                        "duplicate_rows": profile["duplicate_rows"],
                    },
                }
            )
        except Exception:
            pass

        for result in results:
            sections.append(
                {
                    "header": result.name,
                    "kv": result.outputs,
                }
            )

        generate_markdown_report(
            title="Battery Pipeline Overview",
            sections=sections,
            out_path=report_path,
        )
        return StepResult(
            name="generate_overview_report",
            outputs={"report": str(report_path)},
        )

    def _run_script(self, script_rel: str, *script_args: str):
        cmd = [
            "conda",
            "run",
            "-n",
            self.python_env,
            "python",
            script_rel,
            *script_args,
        ]
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
        )


def format_step_results(results: List[StepResult]) -> str:
    lines = []
    for result in results:
        lines.append(f"[{result.name}]")
        for key, value in result.outputs.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).strip()
