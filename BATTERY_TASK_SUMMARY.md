# 电池任务汇总说明

## 1. 文档目的

本文档对当前电池退化建模相关任务做统一汇总，重点记录：

- 每个任务的特点
- 输入是什么
- 输出是什么
- 预测是如何进行的
- 预测图是如何得到的
- 做了哪些特征工程
- 当前已经得到的主要结论

本文档偏“总览”，用于帮助快速理解项目当前已经做了什么、每种任务在回答什么问题。

## 2. 数据背景

当前使用的数据来自电池老化实验 `expt_2_2`。

实验数据包含两大类：

1. Summary 数据
   - 例如容量、SOH、内阻、吞吐量、循环数、老化天数

2. Timeseries / 曲线数据
   - 例如 0.1C 放电曲线
   - 0.5C 曲线
   - GITT 曲线
   - Hybrid 曲线

当前主线工作主要围绕：

- Summary 表格特征
- 0.1C 曲线特征

进行建模。

## 3. RPT 是什么

`RPT` 是 `Reference Performance Test`，即参考性能测试节点。

电池在老化过程中不会每时每刻都做统一分析，而是：

- 老化一段时间
- 做一次标准化测量
- 得到该节点下的容量、SOH、内阻、曲线等

因此：

- `RPT0`
- `RPT1`
- `RPT2`
- ...

表示同一块电池在不同老化阶段上的离散参考测量节点。

所以当前任务并不是：

- 直接对连续原始采样时间序列做预测

而是：

- 对一系列 RPT 节点上的状态/特征做预测

## 4. 当前数据流

### 4.1 原始数据

主要来自：

- `data/raw/battery/expt_2_2/`

其中包括：

- `Summary Data`
- `Processed Timeseries Data`

### 4.2 中间处理结果

当前已经构建出以下处理后数据表：

- `data/processed/battery/expt_2_2/rpt_summary_table.csv`
- `data/processed/battery/expt_2_2/summary_next_soh_dataset.csv`
- `data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv`
- `data/processed/battery/expt_2_2/summary_early_point_to_last_dataset.csv`
- `data/processed/battery/expt_2_2/summary_early_trajectory_to_last_dataset.csv`

### 4.3 各数据表含义

#### `rpt_summary_table.csv`

一行代表：

- 某个 `cell`
- 某个 `rpt_index`

这是最基础的 RPT 级标准表。

#### `summary_next_soh_dataset.csv`

一行代表：

- 输入：`RPT_t` 的特征
- 目标：`RPT_{t+1}` 的 SOH

#### `summary_horizon_3_dataset.csv`

一行代表：

- 输入：`RPT_t` 的特征
- 目标：`RPT_{t+3}` 的 SOH

#### `summary_early_point_to_last_dataset.csv`

一行代表：

- 输入：某个早期单点，例如 `RPT0`、`RPT1`、`RPT2` 或 `RPT3`
- 目标：该 cell 的最后一个 SOH

#### `summary_early_trajectory_to_last_dataset.csv`

一行代表：

- 输入：一个 cell 的完整早期轨迹 `[RPT0, RPT1, RPT2, RPT3]`
- 目标：该 cell 的最后一个 SOH

## 5. 当前任务总览

目前已经做过的任务可以分成四类。

### 5.1 一步预测任务 `next_step`

任务定义：

- 输入：某个 cell 在 `RPT_t` 的特征
- 输出：同一个 cell 在 `RPT_{t+1}` 的 SOH

特点：

- 最容易
- 更接近局部平滑外推
- 当前状态和目标高度相关

### 5.2 固定跨度预测任务 `horizon_3`

任务定义：

- 输入：某个 cell 在 `RPT_t` 的特征
- 输出：同一个 cell 在 `RPT_{t+3}` 的 SOH

特点：

- 比一步预测更难
- 预测跨度更长
- 更适合检验特征是否具备更远的稳定预测力

### 5.3 早期单点到末期任务 `early_point_to_last`

任务定义：

- 输入：某个 cell 的早期单点，例如 `RPT0`、`RPT1`、`RPT2`、`RPT3`
- 输出：该 cell 的最后一个 SOH

特点：

- 属于“单个早期点预测最终状态”
- 比局部 one-step 预测更难
- 但因为同一个 cell 的多个早期点都指向同一个最终目标，所以测试集内部目标常常是常数

这会导致：

- `R2` 不好解释
- 更应看 `RMSE / MAE / MAPE`

### 5.4 早期轨迹到末期任务 `early_trajectory_to_last`

任务定义：

- 输入：一个 cell 的早期轨迹 `[RPT0, RPT1, RPT2, RPT3]`
- 输出：该 cell 的最后一个 SOH

特点：

- 比 `early_point_to_last` 更合理
- 因为输入不再是单个早期点，而是完整早期走势
- 更接近真实意义上的“预后”

但当前样本量很小：

- 每个 cell 只有 1 个 trajectory 样本
- 总共只有 6 个样本

所以：

- 结果可作为趋势参考
- 不能过度解读

## 6. 预测是怎么进行的

### 6.1 基本评估方式

当前所有任务统一采用：

- `leave-one-cell-out`

也就是：

- 用 5 个 cell 训练
- 留 1 个未见过的 cell 做测试
- 循环直到每个 cell 都被当过测试集

这样得到的结果更强调：

- 跨电芯泛化能力

而不是：

- 同一块电池内部的插值拟合能力

### 6.2 模型怎么训练

当前主要使用的是表格回归模型：

- `LinearRegression`
- `Ridge`
- `RandomForest`
- `GradientBoosting`

对于每个任务：

1. 读取对应的数据表
2. 选取当前实验允许使用的特征列
3. 进行 leave-one-cell-out 训练/测试
4. 汇总每一折、每个模型的指标
5. 保存预测值和图像

### 6.3 为什么有的任务能画预测曲线，有的不能

对于：

- `next_step`
- `horizon_3`
- `early_point_to_last`

每个 cell 在测试时有多个样本点，并且样本中保留了 `rpt_index`，所以可以画：

- 横轴：`rpt_index`
- 纵轴：真实目标与预测目标

因此能形成：

- `true vs pred` 曲线图

对于：

- `early_trajectory_to_last`

每个 cell 只有 1 个 trajectory 样本，所以：

- 没有一串 `rpt_index`
- 也就没有真正意义上的“预测曲线”

因此这个任务更适合：

- 散点图
- 柱状图
- 每个 cell 一个点的误差图

而不适合使用多点曲线图。

## 7. 预测图是怎么得到的

### 7.1 Summary 任务中的预测图

以 `next_step` 为例：

- 对于测试 cell 的每个样本，模型会输出一个预测值 `y_pred`
- 同时数据表中有真实目标 `y_true`
- 这些点按 `rpt_index` 排序后连接起来，就得到：
  - 深色实线：真实值
  - 同色浅色虚线：预测值

因此图的含义是：

- 模型对同一块电池在多个 RPT 节点上的目标预测，与真实退化轨迹之间的对比

### 7.2 为什么 `persistence` 也有图

`persistence` 不是训练模型，而是规则预测：

- `y_pred = 当前 soh`

然后同样把它和真实 `next SOH` 按 `rpt_index` 画出来，所以也能得到预测图。

它的作用是：

- 给最简单 baseline 提供一个可视化参照

## 8. 特征工程总览

当前特征工程主要有三大层次。

### 8.1 Summary 基础特征

来自 summary 表，主要包括：

- `temperature`
- `rpt_index`
- `charge_throughput`
- `energy_throughput`
- `capacity_c10`
- `soh`
- `resistance_0p1s`
- `ageing_cycles`
- `days_of_degradation`
- `delta_charge_throughput`
- `capacity_delta_prev`
- `soh_delta_prev`
- `resistance_delta_prev`
- `soc_range`

这类特征的作用是：

- 描述当前老化阶段
- 描述当前状态
- 描述近期变化趋势

### 8.2 手工 0.1C 曲线特征

当前已经提取了一组低维、可解释的 0.1C 曲线特征，主要包括：

- 曲线末端容量
- 平均电压
- 电压方差
- 中段斜率
- 尾段斜率
- 电压-容量面积
- 固定电压点对应容量
- 固定容量比例对应电压
- 相对 `RPT0` 的偏移量

其作用是：

- 在不显式使用 `soh/capacity` 的情况下
- 从曲线形状中提取健康相关信息

### 8.3 PCA 0.1C 曲线特征

除了手工特征，还做了 PCA 曲线表征：

1. 将 0.1C 曲线重采样到固定 100 个归一化容量网格点
2. 得到 `cc0p1_grid_000 ~ cc0p1_grid_099`
3. 在训练折内部做 PCA
4. 提取前 5 个主成分：
   - `cc0p1_pca_1`
   - `cc0p1_pca_2`
   - `cc0p1_pca_3`
   - `cc0p1_pca_4`
   - `cc0p1_pca_5`

它的作用是：

- 压缩整条曲线的整体形状信息
- 避免直接使用过高维的原始曲线向量

## 9. 当前实验系列

### 9.1 常规消融

主要比较：

- `persistence`
- `process_only`
- `process_plus_resistance`
- `capacity_only`
- `capacity_with_delta`
- `soh_only`
- `soh_with_delta`
- `state_only`
- `state_aware_full`

核心问题：

- 当前状态是不是 shortcut
- process / resistance 是否还有额外贡献

### 9.2 Strict no-shortcut

只允许用：

- `process_plus_resistance`

不允许用：

- `soh`
- `capacity_c10`
- `soh_delta_prev`
- `capacity_delta_prev`

核心问题：

- 禁掉显式状态 shortcut 后，模型还剩多少预测力

### 9.3 Strict + 手工 0.1C

在 strict 基础上加入：

- 手工 0.1C 曲线特征

核心问题：

- 0.1C 曲线能否补足 strict 情况下的信息缺口

### 9.4 Strict + PCA(0.1C)

在 strict 基础上加入：

- 0.1C PCA 特征

核心问题：

- 曲线整体形状的低维表示是否足够有用

### 9.5 Strict + 手工 0.1C + PCA

在 strict 基础上同时加入：

- 手工 0.1C 特征
- 0.1C PCA 特征

核心问题：

- 手工特征和 PCA 是否互补

## 10. 当前已得到的主要结论

### 10.1 关于 shortcut

1. 当前 `soh` 是最强 shortcut 来源
2. `capacity_c10` 也有信息，但弱于 `soh`
3. `state_aware_full` 只比 `state_only` 略强

说明：

- 当前状态量是最主要的预测信号

### 10.2 关于 strict no-shortcut

1. 去掉状态 shortcut 后，`next_step` 仍然能做
2. `horizon_3` 性能明显下降
3. `early_point_to_last` 和 `early_trajectory_to_last` 更难

说明：

- process + resistance 确实有信息
- 但不足以支撑非常强的长期预后

### 10.3 关于手工 0.1C 特征

1. 对 `next_step` 提升非常大
2. 对 `horizon_3` 也提升明显
3. 对 `early_point_to_last` 帮助不明显
4. 对 `early_trajectory_to_last` 有一定帮助

说明：

- 0.1C 曲线包含强健康信息
- 特别适合短期或中期任务

### 10.4 关于 PCA(0.1C)

1. PCA-only 比完全不用曲线强
2. 但总体不如手工 0.1C 特征稳定
3. 手工 + PCA 在 `horizon_3` 上有互补效果

说明：

- PCA 是有意义的
- 但当前数据下它更适合作为补充项，而不是替代项

## 11. 当前最适合怎么理解这个项目

当前项目最合理的定位不是：

- 一个直接做强时序预测的深度学习项目

而更像是：

- 一个面向电池退化实验数据的研究型分析 pipeline

它已经具备的能力包括：

1. 原始数据审计
2. RPT 级标准表构建
3. 多种任务定义
4. 多种 baseline 评估
5. 特征消融
6. strict no-shortcut 分析
7. 0.1C 曲线特征增强
8. PCA 曲线表征对照

## 12. 当前限制

当前还存在一些客观限制：

1. 电芯数量少
   - 只有 6 个 cell

2. RPT 序列短
   - 每个 cell 只有大约 13 个 RPT 节点

3. `early_trajectory_to_last` 样本极少
   - 每个 cell 只有 1 个 trajectory 样本

4. 当前结果更适合说明：
   - 任务定义
   - 特征贡献
   - shortcut 现象

而不适合直接宣称：

- 已经获得非常强的远期寿命预测能力

## 13. 后续建议

如果后面继续推进，建议优先考虑：

1. 做统一总对比表
   - 汇总 `strict_no_shortcut`
   - `strict_with_cc0p1`
   - `strict_with_cc0p1_pca`
   - `strict_with_cc0p1_handcrafted_pca`

2. 为 `early_trajectory_to_last` 设计更合适的图
   - 用散点或柱状图，而不是曲线图

3. 如果后续继续做时序方向
   - 优先做轻量 trajectory / sequence baseline
   - 不要急着上重型深度模型

4. 如果以后补充更多曲线
   - 可以继续探索 0.5C、GITT、Hybrid 的特征工程
