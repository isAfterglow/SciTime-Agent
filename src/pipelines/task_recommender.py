from __future__ import annotations

from typing import Any, Dict, List


def recommend_tasks_from_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = profile.get("columns", [])
    n_cols = profile.get("n_cols", 0)
    column_set = set(columns)

    recommendations: List[Dict[str, Any]] = []

    battery_summary_markers = {
        "cell_id",
        "rpt_index",
        "capacity_c10",
        "soh",
        "charge_throughput",
        "energy_throughput",
    }
    battery_marker_hits = len(battery_summary_markers.intersection(column_set))
    has_curve_grid = any(col.startswith("cc0p1_grid_") for col in columns)

    if battery_marker_hits >= 4 or has_curve_grid:
        recommendations.append(
            {
                "task_name": "battery_summary",
                "score": 0.95 if has_curve_grid else 0.85,
                "reason": "检测到电池 RPT / SOH / throughput / 0.1C 曲线等典型字段，更适合走电池专用 pipeline。",
            }
        )

    generic_score = 0.7
    if n_cols >= 3:
        generic_score += 0.1
    recommendations.append(
        {
            "task_name": "generic_tabular",
            "score": min(generic_score, 0.9),
            "reason": "当前数据可视为普通表格任务，适合先走通用 profiling + baseline 流程。",
        }
    )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations
