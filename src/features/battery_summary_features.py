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