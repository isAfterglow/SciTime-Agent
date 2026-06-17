from typing import List, Optional
import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first existing column from candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_value(row: pd.Series, col: Optional[str], default=np.nan):
    if col is None:
        return default
    return row[col]


def normalize_soh_value(x):
    """
    Some datasets may store SoH as 0-1, others as 0-100.
    Convert to 0-1 if needed.
    """
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan

    if x > 2:
        return x / 100.0
    return x


def build_rpt_summary_table(loader) -> pd.DataFrame:
    """
    Build a standard RPT-level summary table.

    One row = one cell at one RPT point.
    """

    all_rows = []

    for cell in loader.cells:
        meta = loader.cell_metadata(cell)
        main_df = loader.load_performance_summary(cell)

        capacity_col = find_col(
            main_df,
            [
                "C/10 Capacity [mA h]",
                "C/10 Capacity [mAh]",
                "Capacity [mA h]",
                "Capacity [mAh]",
            ],
        )

        soh_col = find_col(main_df, ["SoH", "SOH", "State of Health"])
        throughput_col = find_col(
            main_df,
            [
                "Charge Throughput [A h]",
                "Charge Throughput [Ah]",
                "Throughput [A h]",
            ],
        )

        resistance_col = find_col(
            main_df,
            [
                "0.1s Resistance [Ohms]",
                "Resistance [Ohms]",
                "Internal Resistance [Ohms]",
            ],
        )

        ageing_cycles_col = find_col(
            main_df,
            [
                "Ageing Cycles",
                "Aging Cycles",
                "Cycles",
            ],
        )

        days_col = find_col(
            main_df,
            [
                "Days of degradation",
                "Days of Degradation",
                "Ageing Days",
                "Aging Days",
            ],
        )

        energy_col = find_col(
            main_df,
            [
                "Energy Throughput [W h]",
                "Energy Throughput [Wh]",
            ],
        )

        main_reset = main_df.reset_index(drop=False)

        initial_capacity = None
        if capacity_col is not None:
            initial_capacity = pd.to_numeric(main_df[capacity_col], errors="coerce").dropna()
            if len(initial_capacity) > 0:
                initial_capacity = float(initial_capacity.iloc[0])
            else:
                initial_capacity = np.nan
        else:
            initial_capacity = np.nan

        for rpt_index, row in main_reset.iterrows():
            capacity = get_value(row, capacity_col)
            capacity = pd.to_numeric(capacity, errors="coerce")

            soh = get_value(row, soh_col)
            soh = normalize_soh_value(soh)

            if pd.isna(soh) and initial_capacity and not pd.isna(capacity):
                soh = float(capacity) / float(initial_capacity)

            record = {
                "cell_id": cell,
                "temperature": meta.get("Temp", np.nan),
                "soc_range": meta.get("SoC range", np.nan),
                "rpt_index": rpt_index,

                "charge_throughput": pd.to_numeric(
                    get_value(row, throughput_col), errors="coerce"
                ),
                "energy_throughput": pd.to_numeric(
                    get_value(row, energy_col), errors="coerce"
                ),
                "capacity_c10": capacity,
                "soh": soh,
                "resistance_0p1s": pd.to_numeric(
                    get_value(row, resistance_col), errors="coerce"
                ),
                "ageing_cycles": pd.to_numeric(
                    get_value(row, ageing_cycles_col), errors="coerce"
                ),
                "days_of_degradation": pd.to_numeric(
                    get_value(row, days_col), errors="coerce"
                ),

                "raw_capacity_col": capacity_col,
                "raw_soh_col": soh_col,
                "raw_throughput_col": throughput_col,
                "raw_resistance_col": resistance_col,
            }

            all_rows.append(record)

    out_df = pd.DataFrame(all_rows)

    out_df = out_df.sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)

    return out_df


def build_next_soh_dataset(rpt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build supervised next-step prediction dataset.

    Feature at RPT_t -> target SOH at RPT_{t+1}
    """

    df = rpt_df.copy()
    df = df.sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)

    grouped = df.groupby("cell_id", group_keys=False)

    df["target_soh_next"] = grouped["soh"].shift(-1)
    df["target_capacity_next"] = grouped["capacity_c10"].shift(-1)

    df["delta_charge_throughput"] = grouped["charge_throughput"].diff()
    df["capacity_delta_prev"] = grouped["capacity_c10"].diff()
    df["soh_delta_prev"] = grouped["soh"].diff()
    df["resistance_delta_prev"] = grouped["resistance_0p1s"].diff()

    delta_cols = [
        "delta_charge_throughput",
        "capacity_delta_prev",
        "soh_delta_prev",
        "resistance_delta_prev",
    ]

    for col in delta_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Last RPT of each cell has no next target.
    df = df.dropna(subset=["target_soh_next"]).reset_index(drop=True)

    return df


def build_horizon_soh_dataset(
    rpt_df: pd.DataFrame,
    horizon: int = 3,
) -> pd.DataFrame:
    """
    Build a fixed-horizon supervised dataset.

    Feature at RPT_t -> target SOH at RPT_{t+horizon}
    """
    df = rpt_df.copy()
    df = df.sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)

    grouped = df.groupby("cell_id", group_keys=False)

    df[f"target_soh_h{horizon}"] = grouped["soh"].shift(-horizon)
    df[f"target_capacity_h{horizon}"] = grouped["capacity_c10"].shift(-horizon)

    df["delta_charge_throughput"] = grouped["charge_throughput"].diff()
    df["capacity_delta_prev"] = grouped["capacity_c10"].diff()
    df["soh_delta_prev"] = grouped["soh"].diff()
    df["resistance_delta_prev"] = grouped["resistance_0p1s"].diff()

    delta_cols = [
        "delta_charge_throughput",
        "capacity_delta_prev",
        "soh_delta_prev",
        "resistance_delta_prev",
    ]

    for col in delta_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = df.dropna(subset=[f"target_soh_h{horizon}"]).reset_index(drop=True)

    return df


def build_early_point_to_last_dataset(
    rpt_df: pd.DataFrame,
    early_rpt_max: int = 3,
) -> pd.DataFrame:
    """
    Build an early-point-to-last prognosis dataset.

    Keep only early RPT rows and assign each row the last available SOH/capacity
    from the same cell as the target.
    """
    df = rpt_df.copy()
    df = df.sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)

    grouped = df.groupby("cell_id", group_keys=False)

    df["target_soh_last"] = grouped["soh"].transform("last")
    df["target_capacity_last"] = grouped["capacity_c10"].transform("last")
    df["target_rpt_last"] = grouped["rpt_index"].transform("last")

    df["delta_charge_throughput"] = grouped["charge_throughput"].diff()
    df["capacity_delta_prev"] = grouped["capacity_c10"].diff()
    df["soh_delta_prev"] = grouped["soh"].diff()
    df["resistance_delta_prev"] = grouped["resistance_0p1s"].diff()

    delta_cols = [
        "delta_charge_throughput",
        "capacity_delta_prev",
        "soh_delta_prev",
        "resistance_delta_prev",
    ]

    for col in delta_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = df[df["rpt_index"] <= early_rpt_max].reset_index(drop=True)

    return df


def build_early_trajectory_to_last_dataset(
    rpt_df: pd.DataFrame,
    early_rpt_max: int = 3,
) -> pd.DataFrame:
    """
    Build an early-trajectory-to-last prognosis dataset.

    One row = one cell.
    Input = flattened early trajectory from RPT0..RPT_early_rpt_max.
    Target = last available SOH/capacity of that cell.
    """
    df = rpt_df.copy()
    df = df.sort_values(["cell_id", "rpt_index"]).reset_index(drop=True)
    grouped = df.groupby("cell_id", group_keys=False)
    df["delta_charge_throughput"] = grouped["charge_throughput"].diff().fillna(0)
    df["resistance_delta_prev"] = grouped["resistance_0p1s"].diff().fillna(0)

    rows = []
    base_cols = [
        "temperature",
        "soc_range",
    ]
    trajectory_cols = [
        "charge_throughput",
        "energy_throughput",
        "resistance_0p1s",
        "delta_charge_throughput",
        "resistance_delta_prev",
        "ageing_cycles",
        "days_of_degradation",
    ]

    for cell_id, sub in df.groupby("cell_id"):
        sub = sub.sort_values("rpt_index").reset_index(drop=True)
        early = sub[sub["rpt_index"] <= early_rpt_max].copy()
        if len(early) < early_rpt_max + 1:
            continue

        record = {
            "cell_id": cell_id,
            "target_soh_last": sub["soh"].iloc[-1],
            "target_capacity_last": sub["capacity_c10"].iloc[-1],
            "target_rpt_last": sub["rpt_index"].iloc[-1],
        }

        for col in base_cols:
            record[col] = early[col].iloc[0]

        for _, row in early.iterrows():
            rpt_idx = int(row["rpt_index"])
            for col in trajectory_cols:
                record[f"{col}_rpt{rpt_idx}"] = row[col]

        rows.append(record)

    return pd.DataFrame(rows)
