from typing import Dict, List


PROCESS_ONLY_FEATURES = [
    "temperature",  # 温度: 电芯所处的实验温度条件, 反映外部老化环境
    "rpt_index",  # RPT序号: 第几个参考性能测试节点, 表示退化进程位置
    "charge_throughput",  # 充电吞吐量: 累积充电安时, 反映已历经的使用/老化负荷
    "energy_throughput",  # 能量吞吐量: 累积能量通量, 反映更综合的工作负荷
    "ageing_cycles",  # 老化循环数: 已经历的循环次数, 是退化进展的直接指标
    "days_of_degradation",  # 老化天数: 从开始老化到当前RPT经过的天数
    "delta_charge_throughput",  # 充电吞吐量增量: 相邻RPT之间新增的充电负荷
    "soc_range",  # SoC范围: 实验运行的荷电状态窗口, 反映工况设定
]

PROCESS_PLUS_RESISTANCE_FEATURES = [
    *PROCESS_ONLY_FEATURES,
    "resistance_0p1s",  # 0.1秒内阻: 当前RPT测得的瞬时内阻, 常与健康状态相关
    "resistance_delta_prev",  # 内阻增量: 相邻RPT之间内阻的变化量, 反映阻抗演化趋势
]

STATE_ONLY_FEATURES = [
    "capacity_c10",  # C/10容量: 当前RPT下的低倍率容量, 是健康状态的核心表征
    "soh",  # 健康状态SOH: 当前容量相对初始容量的归一化指标
    "capacity_delta_prev",  # 容量增量: 相邻RPT之间容量变化量, 反映近期衰减速度
    "soh_delta_prev",  # SOH增量: 相邻RPT之间SOH变化量, 反映近期健康度变化趋势
]

CAPACITY_ONLY_FEATURES = [
    "capacity_c10",  # C/10容量: 只保留当前容量主状态, 不使用SOH
]

CAPACITY_WITH_DELTA_FEATURES = [
    "capacity_c10",  # C/10容量: 当前容量水平
    "capacity_delta_prev",  # 容量增量: 近期容量变化速度
]

SOH_ONLY_FEATURES = [
    "soh",  # 健康状态SOH: 只保留当前SOH主状态, 不使用容量
]

SOH_WITH_DELTA_FEATURES = [
    "soh",  # 健康状态SOH: 当前健康度水平
    "soh_delta_prev",  # SOH增量: 近期SOH变化速度
]

STATE_AWARE_FULL_FEATURES = [
    *PROCESS_PLUS_RESISTANCE_FEATURES,
    *STATE_ONLY_FEATURES,
]


FEATURE_SETS: Dict[str, List[str]] = {
    "process_only": PROCESS_ONLY_FEATURES,
    "process_plus_resistance": PROCESS_PLUS_RESISTANCE_FEATURES,
    "capacity_only": CAPACITY_ONLY_FEATURES,
    "capacity_with_delta": CAPACITY_WITH_DELTA_FEATURES,
    "soh_only": SOH_ONLY_FEATURES,
    "soh_with_delta": SOH_WITH_DELTA_FEATURES,
    "state_only": STATE_ONLY_FEATURES,
    "state_aware_full": STATE_AWARE_FULL_FEATURES,
}
