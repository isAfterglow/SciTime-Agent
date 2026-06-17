# SciTime-Agent

## 项目简介

SciTime-Agent 是一个面向科学时序预测的智能建模 Agent 平台，目标是支持电池寿命预测、材料参数到温度序列预测等任务。

系统将逐步实现：

- 数据上传与自动分析
- 电池 SOH / RUL 预测
- 材料参数到温度曲线预测
- Agent 自动选择建模流程
- Optuna 调参
- MLflow 实验追踪
- FastAPI 预测接口
- Docker 部署

## 当前版本

v0.1-basic-pipeline

当前功能：

- 项目基础结构
- Data Profiler：行列统计、字段类型、缺失率、重复值、目标列猜测、数值列统计
- 通用表格 baseline pipeline：支持回归与分类
- Streamlit 页面：CSV 上传、数据概览、目标列选择、baseline 训练
- FastAPI 接口：`/health`、`/profile`、`/train/basic`
- 电池场景专用 pipeline 与实验脚本

## 快速启动

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Streamlit：

```bash
streamlit run app/streamlit_app.py
```

启动 FastAPI：

```bash
uvicorn app.api:app --reload
```

前后端联调顺序：

```bash
# 终端 1：启动后端
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：启动前端
streamlit run app/streamlit_app.py
```

启动后，在 Streamlit 侧边栏确认 `API 地址` 为 `http://127.0.0.1:8000`，
点击“检查 API 状态”应返回正常，再上传 CSV 做画像和训练。

运行通用 demo：

```bash
python scripts/run_generic_demo.py --dataset data/demo/generic_regression_demo.csv --target target_value --task-type regression
python scripts/run_generic_demo.py --dataset data/demo/generic_classification_demo.csv --target fast_fail --task-type classification --step train
```

查看任务注册表：

```bash
python scripts/run_registered_task.py --list
```

对数据做任务推荐：

```bash
python scripts/run_registered_task.py \
  --recommend \
  --dataset data/demo/generic_regression_demo.csv \
  --target target_value
```

通过任务注册层运行 generic 任务：

```bash
python scripts/run_registered_task.py \
  --task generic_tabular \
  --dataset data/demo/generic_regression_demo.csv \
  --target target_value \
  --task-type regression \
  --models LinearRegression RandomForestRegressor
```

通过任务注册层运行 battery 任务：

```bash
python scripts/run_registered_task.py --task battery_summary --step audit
python scripts/run_registered_task.py --task battery_summary --step profile
```

## 当前系统通路

当前平台已经形成下面这条链路：

```text
Streamlit 前端
  -> FastAPI 接口层
    -> Task Registry 任务注册层
      -> GenericPipeline / BatteryPipeline
        -> tools/data_profiler.py / tools/model_trainer.py / battery scripts
```

具体来说：

- 用户在 `Streamlit` 上传 CSV
- 前端通过 HTTP 调 `/profile`
- 后端做数据画像，并基于画像给出任务推荐
- 前端展示字段统计、缺失率、推荐任务
- 用户选择目标列、任务类型、特征列、模型
- 前端通过 HTTP 调 `/train/basic`
- 后端执行通用 baseline 训练，返回指标和预测结果
- 如果后面走注册任务执行，则通过 `/tasks/run` 或 `run_registered_task.py` 统一进入对应 pipeline

运行电池 demo：

```bash
python scripts/run_battery_demo.py --step audit
python scripts/run_battery_demo.py --step profile
```

## 项目结构

```text
src/
  data/
  evaluation/
  features/
  pipelines/
  tools/
app/
scripts/
data/
reports/
```

## 当前功能说明

- `src/tools/data_profiler.py`
  提供表格数据概览能力。
- `src/tools/model_trainer.py`
  提供回归/分类 baseline 训练能力。
- `src/pipelines/generic_pipeline.py`
  提供通用表格任务 pipeline。
- `src/pipelines/battery_pipeline.py`
  提供电池任务专用 pipeline。
- `app/streamlit_app.py`
  提供最小可用前端页面。
- `app/api.py`
  提供最小可用后端接口。
