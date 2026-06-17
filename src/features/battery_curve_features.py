from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VOLTAGE_GRID = [3.6, 3.8, 4.0]
CHARGE_FRAC_GRID = [0.1, 0.5, 0.9]


def _interp_y_at_x(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if x_target < x_sorted[0] or x_target > x_sorted[-1]:
        return np.nan
    return float(np.interp(x_target, x_sorted, y_sorted))


def extract_cc_0p1c_curve_features(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)

    charge = pd.to_numeric(df["Charge (mA.h)"], errors="coerce").to_numpy()
    voltage = pd.to_numeric(df["Voltage (V)"], errors="coerce").to_numpy()
    current = pd.to_numeric(df["Current (mA)"], errors="coerce").to_numpy()
    temperature = pd.to_numeric(df["Temperature (degC)"], errors="coerce").to_numpy()

    mask = np.isfinite(charge) & np.isfinite(voltage)
    charge = charge[mask]
    voltage = voltage[mask]
    if len(charge) < 10:
        return {}

    q_end = float(np.nanmax(charge))
    q_norm = charge / q_end if q_end > 0 else np.full_like(charge, np.nan)

    features = {
        "cc0p1_q_end": q_end,
        "cc0p1_v_mean": float(np.nanmean(voltage)),
        "cc0p1_v_std": float(np.nanstd(voltage)),
        "cc0p1_current_mean": float(np.nanmean(current)) if len(current) else np.nan,
        "cc0p1_temp_mean": float(np.nanmean(temperature)) if len(temperature) else np.nan,
        "cc0p1_mid_slope": _segment_slope(charge, voltage, 0.2, 0.8),
        "cc0p1_tail_slope": _segment_slope(charge, voltage, 0.8, 1.0),
        "cc0p1_area_vdq": float(np.trapz(voltage, charge)),
    }

    for v in VOLTAGE_GRID:
        features[f"cc0p1_q_at_{str(v).replace('.', 'p')}v"] = _interp_y_at_x(
            voltage,
            charge,
            v,
        )

    for frac in CHARGE_FRAC_GRID:
        features[f"cc0p1_v_at_{int(frac * 100)}q"] = _interp_y_at_x(
            q_norm,
            voltage,
            frac,
        )

    return features


def _segment_slope(charge: np.ndarray, voltage: np.ndarray, q_start_frac: float, q_end_frac: float) -> float:
    q_max = np.nanmax(charge)
    q_start = q_max * q_start_frac
    q_end = q_max * q_end_frac
    mask = (charge >= q_start) & (charge <= q_end)
    if np.sum(mask) < 2:
        return np.nan
    q_seg = charge[mask]
    v_seg = voltage[mask]
    dq = q_seg[-1] - q_seg[0]
    if abs(dq) < 1e-12:
        return np.nan
    return float((v_seg[-1] - v_seg[0]) / dq)


def build_cc_0p1c_feature_table(loader) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    baseline_by_cell: Dict[str, Dict[str, float]] = {}

    for cell in loader.cells:
        n_rpt = loader.expected_rpt_count(cell)
        for rpt in range(n_rpt):
            path = loader.timeseries_paths(cell, rpt)["cc_0p1c"]
            feat = extract_cc_0p1c_curve_features(path)
            feat["cell_id"] = cell
            feat["rpt_index"] = rpt

            if rpt == 0:
                baseline_by_cell[cell] = feat.copy()

            base = baseline_by_cell.get(cell, {})
            feat["cc0p1_q_end_delta_from_rpt0"] = _safe_delta(
                feat.get("cc0p1_q_end"),
                base.get("cc0p1_q_end"),
            )
            feat["cc0p1_area_vdq_delta_from_rpt0"] = _safe_delta(
                feat.get("cc0p1_area_vdq"),
                base.get("cc0p1_area_vdq"),
            )
            feat["cc0p1_v_mean_delta_from_rpt0"] = _safe_delta(
                feat.get("cc0p1_v_mean"),
                base.get("cc0p1_v_mean"),
            )
            rows.append(feat)

    return pd.DataFrame(rows).sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)


def _safe_delta(x, x0):
    if x is None or x0 is None:
        return np.nan
    if pd.isna(x) or pd.isna(x0):
        return np.nan
    return float(x - x0)
