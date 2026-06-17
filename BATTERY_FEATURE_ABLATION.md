# 电池特征消融与 Strict No-Shortcut 实验说明

## 1. 文档目的

本文档说明当前项目中与电池退化 summary 建模相关的两类实验：

1. 常规特征消融实验
2. strict no-shortcut 实验

重点包括：

- 任务定义
- 特征选择依据
- 运行方式
- 输出文件位置
- 当前结果的解释

## 2. 基础任务形式

当前项目不是在原始连续采样时间序列上直接做 forecasting，而是在 RPT 节点级别做预测。

可以理解为：

- 输入：某个 `cell` 在某个 `RPT` 节点上的 summary 特征
- 输出：该 `cell` 在后续某个 RPT 节点上的 SOH

因此，这更准确地属于：

- 基于 RPT 级表格特征的退化预测任务

## 3. RPT 是什么

`RPT` 是 `Reference Performance Test`，可理解为：

- 参考性能测试节点

电池在老化实验过程中会周期性进行标准化测量，每次 RPT 会记录：

- 容量
- SOH
- 内阻
- 吞吐量
- 循环数
- 老化时间

所以：

- `RPT0` 是较早期参考节点
- `RPT1`、`RPT2`、`RPT3` 表示后续参考节点

你现在建模不是在原始秒级/采样点级时间序列上预测，而是在这些离散的参考性能节点上预测。

## 4. 数据文件

当前构建出的关键中间数据表位于：

- `data/processed/battery/expt_2_2/rpt_summary_table.csv`
- `data/processed/battery/expt_2_2/summary_next_soh_dataset.csv`
- `data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv`
- `data/processed/battery/expt_2_2/summary_early_point_to_last_dataset.csv`
- `data/processed/battery/expt_2_2/summary_early_trajectory_to_last_dataset.csv`

含义如下：

- `rpt_summary_table.csv`
  - 每行表示一个 `cell` 在一个 `RPT` 节点上的 summary 特征

- `summary_next_soh_dataset.csv`
  - 用 `RPT_t` 的特征预测 `RPT_{t+1}` 的 SOH

- `summary_horizon_3_dataset.csv`
  - 用 `RPT_t` 的特征预测 `RPT_{t+3}` 的 SOH

- `summary_early_point_to_last_dataset.csv`
  - 保留早期每个单独 RPT 点作为输入样本，目标统一指向该 cell 的最后 SOH

- `summary_early_trajectory_to_last_dataset.csv`
  - 将 `[RPT0, RPT1, RPT2, RPT3]` 的早期轨迹展平成一个样本，目标是该 cell 的最后 SOH

## 5. 评估协议

所有实验统一采用：

- 划分方式：`leave-one-cell-out`
- 指标：
  - `MAE`
  - `RMSE`
  - `MAPE`
  - `R2`

含义：

- 每一轮训练用 5 个 cell
- 留 1 个未见过的 cell 做测试

目的是考察跨 cell 泛化能力，而不是同一个 cell 内插值拟合能力。

## 6. 常规特征消融实验

### 6.1 设计目的

常规消融实验主要回答：

1. process 特征有没有预测力
2. resistance 是否带来额外增益
3. 当前状态特征是否构成 shortcut
4. 完整特征组相比简单状态特征是否真的更强

### 6.2 特征组

#### `persistence`

规则：

- `y_pred = soh`

含义：

- 直接把当前 SOH 当作下一步 SOH 的预测值

#### `process_only`

特征：

- `temperature`
- `rpt_index`
- `charge_throughput`
- `energy_throughput`
- `ageing_cycles`
- `days_of_degradation`
- `delta_charge_throughput`
- `soc_range`

#### `process_plus_resistance`

特征：

- `process_only`
- `resistance_0p1s`
- `resistance_delta_prev`

#### `capacity_only`

特征：

- `capacity_c10`

#### `capacity_with_delta`

特征：

- `capacity_c10`
- `capacity_delta_prev`

#### `soh_only`

特征：

- `soh`

#### `soh_with_delta`

特征：

- `soh`
- `soh_delta_prev`

#### `state_only`

特征：

- `capacity_c10`
- `soh`
- `capacity_delta_prev`
- `soh_delta_prev`

#### `state_aware_full`

特征：

- `process_plus_resistance`
- `capacity_c10`
- `soh`
- `capacity_delta_prev`
- `soh_delta_prev`

### 6.3 代码位置

特征组定义：

- `src/features/battery_feature_sets.py`

运行脚本：

- `scripts/run_battery_feature_ablation.py`

### 6.4 结果位置

结果目录：

- `reports/battery/expt_2_2/feature_ablation/`

图目录：

- `reports/battery/expt_2_2/feature_ablation/figures/`

其中：

- 根目录保留总览图
  - `ablation_rmse.png`
  - `ablation_r2.png`

- 各消融组 true/pred 图分别按子目录分类

### 6.5 当前结论

当前结果说明：

1. `process_only` 比 `persistence` 强
   - 说明老化进程信息是有用的

2. `soh_only` 明显强于 `capacity_only`
   - 说明 shortcut 的主要来源是当前 `soh`

3. `soh_with_delta` 比 `soh_only` 提升不大
   - 说明最主要的信息来自当前 SOH 值本身

4. `state_aware_full` 只比 `state_only` 略好
   - 说明 process/resistance 有增益，但主导信号仍来自状态量

## 7. Strict No-Shortcut 实验

### 7.1 设计目的

strict no-shortcut 实验用于回答：

1. 如果禁止使用状态 shortcut，只保留 process + resistance，还剩多少预测能力
2. 预测跨度拉长后性能会下降多少
3. 只用早期信息预测后期时，这类特征是否还能支持有效预后

### 7.2 特征约束

strict no-shortcut 任务只允许使用：

- `temperature`
- `rpt_index`
- `charge_throughput`
- `energy_throughput`
- `ageing_cycles`
- `days_of_degradation`
- `delta_charge_throughput`
- `soc_range`
- `resistance_0p1s`
- `resistance_delta_prev`

也就是：

- 只允许 `process_plus_resistance`

禁止使用：

- `soh`
- `capacity_c10`
- `soh_delta_prev`
- `capacity_delta_prev`

注意：

- 这里去掉的是“状态 shortcut”
- 并不意味着完全去掉所有易预测信息
- `rpt_index`、`throughput`、`cycles`、`days` 仍然包含强烈的阶段信息

### 7.3 任务列表

#### Task-1: `next_step`

- 数据集：`summary_next_soh_dataset.csv`
- 目标：`target_soh_next`

含义：

- 用 `RPT_t` 的 process + resistance 特征预测 `RPT_{t+1}` 的 SOH

#### Task-2: `horizon_3`

- 数据集：`summary_horizon_3_dataset.csv`
- 目标：`target_soh_h3`

含义：

- 用 `RPT_t` 的 process + resistance 特征预测 `RPT_{t+3}` 的 SOH

#### Task-3: `early_point_to_last`

- 数据集：`summary_early_point_to_last_dataset.csv`
- 目标：`target_soh_last`

含义：

- 保留每个 cell 早期单点样本 `RPT0..RPT3`
- 每一行单独预测该 cell 的最后 SOH

形式上可理解为：

- `RPT0 -> last SOH`
- `RPT1 -> last SOH`
- `RPT2 -> last SOH`
- `RPT3 -> last SOH`

#### Task-4: `early_trajectory_to_last`

- 数据集：`summary_early_trajectory_to_last_dataset.csv`
- 目标：`target_soh_last`

含义：

- 把每个 cell 的早期轨迹 `[RPT0, RPT1, RPT2, RPT3]` 整体展平为一个样本
- 用这个完整早期轨迹预测最后 SOH

形式上可理解为：

- `[RPT0, RPT1, RPT2, RPT3] -> last SOH`

这比 `early_point_to_last` 更合理，因为：

- 输入不再是单个早期点
- 而是一个早期退化轨迹

### 7.4 代码位置

strict no-shortcut 脚本：

- `scripts/run_battery_strict_baseline.py`

### 7.5 如何运行

```bash
cd /home/ai4mater/003-Scitime-agent/scitime-agent
conda run -n scitime-agent python scripts/run_battery_strict_baseline.py
```

### 7.6 结果位置

结果目录：

- `reports/battery/expt_2_2/strict_no_shortcut/`

下分四个任务子目录：

- `next_step/`
- `horizon_3/`
- `early_point_to_last/`
- `early_trajectory_to_last/`

每个任务目录下包含：

- `strict_metrics.csv`
- `strict_average_metrics.csv`
- `strict_predictions.csv`
- `strict_manifest.json`
- `figures/`

### 7.7 当前结果解读

#### `next_step`

最佳结果大致为：

- `GradientBoosting`
- `RMSE ≈ 0.00695`
- `R2 ≈ 0.885`

说明：

- 即使去掉状态 shortcut，只靠 process + resistance，短期一步预测仍有一定能力

#### `horizon_3`

最佳结果大致为：

- `GradientBoosting`
- `RMSE ≈ 0.00795`
- `R2 ≈ 0.725`

说明：

- 当预测跨度从一步拉长到三步，性能明显下降

#### `early_point_to_last`

最佳结果大致为：

- `GradientBoosting`
- `RMSE ≈ 0.00805`
- `R2 = 0.0`

这里 `R2 = 0.0` 的原因不是模型一定完全失效，而是：

- 对某个测试 cell 来说
- `RPT0..RPT3` 这几行的目标都指向同一个最终 SOH
- 测试集目标在 cell 内是常数

这会导致：

- `R2` 失去正常解释意义

因此这个任务更应该关注：

- `MAE`
- `RMSE`
- `MAPE`

#### `early_trajectory_to_last`

最佳结果大致为：

- `RandomForest`
- `RMSE ≈ 0.00856`
- `R2 = NaN`

这里 `R2 = NaN` 的原因是：

- 每个测试 cell 只有 1 个样本
- 单样本无法定义正常的 `R2`

这说明：

- `early_trajectory_to_last` 在任务形式上更合理
- 但当前样本量极小，暂时更适合作为 proof-of-concept

## 8. 当前整体结论

综合常规消融与 strict no-shortcut，可以得到：

1. 当前 `soh` 是最强 shortcut 来源

2. `capacity_c10` 也有信息，但弱于 `soh`

3. process + resistance 在短期任务中仍有预测力

4. 一旦预测跨度拉长或任务改为早期到末期
   - 性能会明显下降

5. 当前项目最强的能力依然是：
   - 基于当前状态的一步预测

而不是：

- 严格意义上的强远期寿命预后

## 9. Strict + 0.1C 曲线特征实验

### 9.1 设计目的

这部分实验是在 strict no-shortcut 的基础上，额外加入 0.1C 放电曲线特征，回答：

1. 在不使用 `soh/capacity` 状态 shortcut 的前提下，0.1C 曲线是否能显著补充信息
2. 这些曲线特征对四类 strict 任务是否都有帮助

### 9.2 0.1C 曲线特征

当前提取的是一组低维、可解释的曲线统计特征，包括：

- `cc0p1_q_end`
- `cc0p1_v_mean`
- `cc0p1_v_std`
- `cc0p1_current_mean`
- `cc0p1_temp_mean`
- `cc0p1_mid_slope`
- `cc0p1_tail_slope`
- `cc0p1_area_vdq`
- `cc0p1_q_at_3p6v`
- `cc0p1_q_at_3p8v`
- `cc0p1_q_at_4p0v`
- `cc0p1_v_at_10q`
- `cc0p1_v_at_50q`
- `cc0p1_v_at_90q`
- `cc0p1_q_end_delta_from_rpt0`
- `cc0p1_area_vdq_delta_from_rpt0`
- `cc0p1_v_mean_delta_from_rpt0`

这些特征来源于：

- 0.1C 电压-容量曲线的位置
- 平均形状
- 平台斜率
- 曲线面积
- 相对 `RPT0` 的偏移

### 9.3 代码位置

曲线特征提取：

- `src/features/battery_curve_features.py`

运行脚本：

- `scripts/run_battery_strict_curve_baseline.py`

### 9.4 结果输出位置

输出目录：

- `reports/battery/expt_2_2/strict_with_cc0p1/`

和 `strict_no_shortcut/` 同级。

### 9.5 当前结果

和原始 strict no-shortcut 对比，0.1C 特征带来了明显提升：

#### `next_step`

原始 strict：

- 最好 `RMSE ≈ 0.00695`

加入 0.1C 后：

- 最好 `RMSE ≈ 0.00276`

说明：

- 0.1C 曲线特征对短期一步预测提升非常大

#### `horizon_3`

原始 strict：

- 最好 `RMSE ≈ 0.00795`

加入 0.1C 后：

- 最好 `RMSE ≈ 0.00429`

说明：

- 0.1C 曲线在更长跨度预测中同样有明显增益

#### `early_point_to_last`

原始 strict：

- 最好 `RMSE ≈ 0.00805`

加入 0.1C 后：

- 最好 `RMSE ≈ 0.00937`

说明：

- 对单点 early-to-last 任务，0.1C 特征没有带来提升
- 反而略有退化

#### `early_trajectory_to_last`

原始 strict：

- 最好 `RMSE ≈ 0.00856`

加入 0.1C 后：

- 最好 `RMSE ≈ 0.00726`

说明：

- 对轨迹到末期任务，0.1C 特征是有帮助的
- 但当前样本量只有 6，暂时只能做趋势性判断

### 9.6 当前判断

这轮结果说明：

1. 0.1C 曲线特征对 strict 任务是有价值的
2. 尤其对 `next_step` 和 `horizon_3` 提升非常明显
3. 对 `early_point_to_last` 这种“单点预测最终状态”的任务帮助不明显
4. 对 `early_trajectory_to_last` 这种“轨迹预测最终状态”的任务是正向帮助

因此：

- 如果后续继续做 strict 方向，0.1C 曲线特征值得保留
- 更适合和 trajectory 型任务结合

## 10. Strict + 0.1C PCA 与 Hand-crafted + PCA 实验

### 10.1 设计目的

这部分实验用于回答：

1. 只用 0.1C 曲线的 PCA 压缩表示，是否能替代手工 0.1C 特征
2. 0.1C 手工特征和 PCA 表征是否互补

### 10.2 PCA 做法

当前做法是：

1. 将每条 0.1C 曲线按归一化容量轴重采样到固定 100 点
2. 得到 `cc0p1_grid_000 ~ cc0p1_grid_099`
3. 在 leave-one-cell-out 的每个训练折内拟合 PCA
4. 取前 5 个主成分：
   - `cc0p1_pca_1`
   - `cc0p1_pca_2`
   - `cc0p1_pca_3`
   - `cc0p1_pca_4`
   - `cc0p1_pca_5`

这样可以避免测试 cell 信息泄漏到 PCA 拟合中。

### 10.3 两套实验

#### A. `strict_with_cc0p1_pca`

特征：

- `process_plus_resistance`
- `cc0p1_pca_1 ~ cc0p1_pca_5`

目录：

- `reports/battery/expt_2_2/strict_with_cc0p1_pca/`

#### B. `strict_with_cc0p1_handcrafted_pca`

特征：

- `process_plus_resistance`
- 手工 0.1C 特征
- `cc0p1_pca_1 ~ cc0p1_pca_5`

目录：

- `reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/`

### 10.4 当前结果

#### `strict_with_cc0p1_pca`

1. `next_step`

- 最好 `RMSE ≈ 0.00443` (`Ridge`)

2. `horizon_3`

- 最好 `RMSE ≈ 0.00638` (`GradientBoosting`)

3. `early_point_to_last`

- 最好 `RMSE ≈ 0.01079` (`RandomForest`)

4. `early_trajectory_to_last`

- 最好 `RMSE ≈ 0.00726` (`GradientBoosting`)

结论：

- PCA only 比原始 strict no-shortcut 有明显提升
- 但整体不如“手工 0.1C 特征”

#### `strict_with_cc0p1_handcrafted_pca`

1. `next_step`

- 最好 `RMSE ≈ 0.00286` (`GradientBoosting`)

2. `horizon_3`

- 最好 `RMSE ≈ 0.00418` (`LinearRegression`)

3. `early_point_to_last`

- 最好 `RMSE ≈ 0.00910` (`GradientBoosting`)

4. `early_trajectory_to_last`

- 最好 `RMSE ≈ 0.00726` (`GradientBoosting`)

结论：

- `hand-crafted + PCA` 整体强于 `PCA only`
- 在 `horizon_3` 上还优于单独手工 0.1C 特征
- 但在 `next_step` 上没有超过“手工 0.1C 特征”本身的最好结果

### 10.5 当前判断

综合三种 strict 曲线增强方式：

1. `strict_no_shortcut`
2. `strict_with_cc0p1`
3. `strict_with_cc0p1_pca`
4. `strict_with_cc0p1_handcrafted_pca`

可以得到：

1. PCA 是有意义的
   - 比完全不用 0.1C 曲线强

2. 但 PCA alone 不是最优方案
   - 它提供的是紧凑形状表征
   - 可解释性和效果都不如手工 0.1C 特征稳定

3. `hand-crafted + PCA` 有一定互补性
   - 特别是在 `horizon_3` 上表现最好

4. 对当前数据而言
   - 短期任务里，手工 0.1C 特征已经非常强
   - PCA 更像补充项，而不是替代项

## 11. 回答两个关键问题

### 9.1 提取 0.1C 曲线特征后加入到这四个任务中有没有意义？

有意义，而且很可能是下一步最值得做的增强方向之一。

原因：

1. 你当前 strict no-shortcut 只用了 summary 级 process/resistance 特征
   - 信息量有限

2. 0.1C 曲线本身包含更细粒度的状态和动力学信息，例如：
   - 电压平台形状
   - 不同容量区间的斜率
   - 极化变化
   - 充放电曲线弯曲程度

3. 对 strict 任务尤其有价值
   - 因为你去掉了 `soh/capacity` 这类显式 shortcut
   - 曲线特征可能提供更隐式但更真实的健康信息

建议：

- 优先先加到这四个 strict 任务中，而不是先加到 full baseline
- 这样更容易看清曲线特征是否真的弥补了 no-shortcut 场景的信息缺口

但要注意：

- 曲线特征也可能引入新的隐式 shortcut
- 例如某些特征和容量本质上几乎等价

因此做法上应当：

- 明确列出提取了哪些曲线特征
- 再单独做曲线特征消融

### 9.2 这个电池数据集可以做典型的时序预测吗？

可以做，但要先区分“哪种典型时序预测”。

#### 可以做的两类

1. **RPT 级序列预测**

把每个 cell 的：

- `RPT0, RPT1, ..., RPT12`

看作一条短序列，预测：

- 下一步 SOH
- 多步 SOH
- 最终 SOH

这类是可以做的。

但限制是：

- 每条序列很短
- 只有 6 个 cell
- 样本量非常小

所以：

- 可以做序列模型 baseline
- 但不适合一开始就上很重的深度时序模型

2. **0.1C / GITT / Hybrid 曲线级时序特征提取**

原始曲线本身是高频采样序列。

你可以做：

- 先从曲线提取时序统计特征
- 或对曲线做编码
- 再接回 RPT 级目标预测

这也是一种时序建模，只是它不是直接拿 RPT 节点序列当唯一输入。

#### 不太适合直接做的

如果你想做那种非常典型的深度时序预测：

- 大量序列
- 长时间窗口
- 直接训练 LSTM/Transformer

这批数据当前不太理想，原因是：

- cell 数量太少
- RPT 序列太短
- 跨 cell 的训练样本规模很有限

所以更现实的路线是：

1. 先做曲线特征工程
2. 再做 RPT 级 sequence baseline
3. 最后再考虑轻量时序模型

## 12. 后续建议

如果继续推进，建议优先做：

1. 从 0.1C 曲线提取一组可解释特征
   - 先接入 strict no-shortcut 四任务

2. 单独为曲线特征再做一次消融
   - process + resistance
   - process + resistance + 0.1C curve features

3. 如果再进一步
   - 尝试把 `[RPT0..RPT3]` 作为短序列输入，做一个轻量 sequence baseline
