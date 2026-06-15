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
- Streamlit 初始页面
- FastAPI 健康检查接口
- Data Profiler 初版

## 快速启动

安装依赖：

```bash
pip install -r requirements.txt