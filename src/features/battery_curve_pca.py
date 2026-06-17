from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


Q_GRID = np.linspace(0.0, 1.0, 100)


def curve_to_qgrid_vector(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    charge = pd.to_numeric(df["Charge (mA.h)"], errors="coerce").to_numpy()
    voltage = pd.to_numeric(df["Voltage (V)"], errors="coerce").to_numpy()

    mask = np.isfinite(charge) & np.isfinite(voltage)
    charge = charge[mask]
    voltage = voltage[mask]
    if len(charge) < 10:
        return np.full(len(Q_GRID), np.nan)

    q_end = np.nanmax(charge)
    if not np.isfinite(q_end) or q_end <= 0:
        return np.full(len(Q_GRID), np.nan)

    q_norm = charge / q_end
    order = np.argsort(q_norm)
    q_sorted = q_norm[order]
    v_sorted = voltage[order]

    # Deduplicate q grid points for interpolation stability.
    q_unique, unique_idx = np.unique(q_sorted, return_index=True)
    v_unique = v_sorted[unique_idx]
    if len(q_unique) < 2:
        return np.full(len(Q_GRID), np.nan)

    return np.interp(Q_GRID, q_unique, v_unique)


def build_cc0p1_curve_vector_table(loader) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for cell in loader.cells:
        n_rpt = loader.expected_rpt_count(cell)
        for rpt in range(n_rpt):
            path = loader.timeseries_paths(cell, rpt)["cc_0p1c"]
            vec = curve_to_qgrid_vector(path)
            row: Dict[str, float] = {
                "cell_id": cell,
                "rpt_index": rpt,
            }
            for i, val in enumerate(vec):
                row[f"cc0p1_grid_{i:03d}"] = float(val) if np.isfinite(val) else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)
