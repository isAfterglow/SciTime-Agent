from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BatteryExptLoader:
    """
    Loader for the battery degradation dataset.

    Current target:
    - experiment: expt 2,2
    - load metadata
    - load Performance Summary
    - load Summary per Set
    - locate Processed Timeseries Data files
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = load_yaml(self.config_path)

        self.root_dir = Path(self.cfg["root_dir"])
        self.metadata_path = self.root_dir / self.cfg["metadata_file"]

        self.experiment_key = self.cfg["experiment_key"]
        self.experiment_folder = self.cfg["experiment_folder"]
        self.experiment_dir = self.root_dir / self.experiment_folder

        # In the current battery dataset layout, summary and timeseries folders
        # live directly under root_dir rather than under experiment_folder.
        self.summary_dir = self.root_dir / self.cfg.get("summary_dir", "Summary Data")
        self.timeseries_dir = self.root_dir / self.cfg.get("timeseries_dir", "Processed Timeseries Data")

        self.cells = self.cfg["cells"]

        self.metadata = self.load_metadata()

    @property
    def expt_suffix(self) -> str:
        """
        Convert 'expt 2,2' -> '2,2', which is used in file names.
        """
        return self.experiment_key.replace("expt ", "")

    def load_metadata(self) -> pd.DataFrame:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        df = pd.read_excel(self.metadata_path, sheet_name=self.experiment_key)
        if "Cell" not in df.columns:
            raise ValueError(f"Metadata sheet {self.experiment_key} does not contain 'Cell' column.")

        df = df.set_index("Cell", drop=False)
        return df

    def cell_metadata(self, cell: str) -> pd.Series:
        if cell not in self.metadata.index:
            raise KeyError(f"Cell {cell} not found in metadata.")
        return self.metadata.loc[cell]

    def performance_summary_path(self, cell: str) -> Path:
        meta = self.cell_metadata(cell)
        temp = int(meta["Temp"])

        filename = f"Expt {self.expt_suffix} - cell {cell} ({temp}degC) - Processed Data.csv"
        return self.summary_dir / "Performance Summary" / filename

    def ageing_set_summary_path(self, cell: str) -> Path:
        filename = f"expt {self.expt_suffix} - cell {cell} - set_data.csv"
        return self.summary_dir / "Ageing Sets Summary" / "Summary per Set" / filename

    def load_performance_summary(self, cell: str) -> pd.DataFrame:
        path = self.performance_summary_path(cell)
        if not path.exists():
            raise FileNotFoundError(f"Missing performance summary: {path}")

        df = pd.read_csv(path, index_col=0)
        df["cell_id"] = cell
        return df

    def load_ageing_set_summary(self, cell: str) -> pd.DataFrame:
        path = self.ageing_set_summary_path(cell)
        if not path.exists():
            raise FileNotFoundError(f"Missing ageing set summary: {path}")

        df = pd.read_csv(path, index_col=0)
        df["cell_id"] = cell
        return df

    def timeseries_paths(self, cell: str, rpt: int) -> Dict[str, Path]:
        """
        Return expected timeseries file paths for one cell and one RPT.

        According to the official notebook:
        - Every RPT has 0.1C discharge curve.
        - Even RPT has 0.5C + GITT 25-pulse + GITT 5-pulse.
        - Odd RPT has Hybrid CC-Pulse 0.5C + Hybrid CC-Pulse 1C.
        """

        expt = self.expt_suffix

        paths = {
            "cc_0p1c": self.timeseries_dir
            / "0.1C Voltage Curves"
            / f"cell {cell}"
            / f"Expt {expt} - cell {cell} - RPT{rpt} - 0.1C discharge data.csv"
        }

        if rpt % 2 == 0:
            paths.update(
                {
                    "cc_0p5c": self.timeseries_dir
                    / "0.5C Voltage Curves"
                    / f"cell {cell}"
                    / f"Expt {expt} - cell {cell} - RPT{rpt} - 0.5C discharge data.csv",

                    "gitt_25p": self.timeseries_dir
                    / "GITT Voltage Curves"
                    / f"cell {cell}"
                    / f"Expt {expt} - cell {cell} - RPT{rpt} - 25-pulse GITT 0.5C discharge data.csv",

                    "gitt_5p": self.timeseries_dir
                    / "GITT Voltage Curves"
                    / f"cell {cell}"
                    / f"Expt {expt} - cell {cell} - RPT{rpt} - 5-pulse GITT 0.5C discharge data.csv",
                }
            )
        else:
            paths.update(
                {
                    "hybrid_0p5c": self.timeseries_dir
                    / "Hybrid CC-Pulse Voltage Curves"
                    / f"cell {cell}"
                    / f"Expt {expt} - cell {cell} - RPT{rpt} - Hybrid CC-Pulse 0.5C discharge data.csv",

                    "hybrid_1c": self.timeseries_dir
                    / "Hybrid CC-Pulse Voltage Curves"
                    / f"cell {cell}"
                    / f"Expt {expt} - cell {cell} - RPT{rpt} - Hybrid CC-Pulse 1C discharge data.csv",
                }
            )

        return paths

    def expected_rpt_count(self, cell: str) -> int:
        """
        Use Performance Summary row count as expected RPT count.
        """
        df = self.load_performance_summary(cell)
        return len(df)

    def audit_cell(self, cell: str) -> Dict[str, Any]:
        """
        Check whether summary files and timeseries files exist for a cell.
        """
        def finalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
            row["n_missing_files"] = len(row["missing_files"])
            row["is_complete"] = row["n_missing_files"] == 0
            return row

        row = {
            "cell_id": cell,
            "main_summary_exists": self.performance_summary_path(cell).exists(),
            "set_summary_exists": self.ageing_set_summary_path(cell).exists(),
            "expected_rpt_count": None,
            "n_cc_0p1c": 0,
            "n_cc_0p5c": 0,
            "n_gitt_25p": 0,
            "n_gitt_5p": 0,
            "n_hybrid_0p5c": 0,
            "n_hybrid_1c": 0,
            "missing_files": [],
        }

        if not row["main_summary_exists"]:
            row["missing_files"].append(str(self.performance_summary_path(cell)))
            return finalize_row(row)

        if not row["set_summary_exists"]:
            row["missing_files"].append(str(self.ageing_set_summary_path(cell)))

        try:
            n_rpt = self.expected_rpt_count(cell)
            row["expected_rpt_count"] = n_rpt
        except Exception as e:
            row["missing_files"].append(f"Cannot read RPT count: {e}")
            return finalize_row(row)

        for rpt in range(n_rpt):
            paths = self.timeseries_paths(cell, rpt)
            for key, path in paths.items():
                count_key = f"n_{key}"
                if path.exists():
                    row[count_key] += 1
                else:
                    row["missing_files"].append(str(path))

        return finalize_row(row)
