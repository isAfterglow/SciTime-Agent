import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tools.data_profiler import profile_dataframe


st.set_page_config(
    page_title="SciTime-Agent",
    page_icon="🧪",
    layout="wide"
)

st.title("SciTime-Agent")
st.subheader("面向科学时序预测的智能建模 Agent 平台")

st.markdown(
    """
    当前版本：`v0.1-basic-pipeline`

    本阶段目标：
    - 上传 CSV 数据
    - 自动分析数据字段、缺失值、重复值
    - 为后续电池寿命预测和材料温度序列预测做准备
    """
)

uploaded_file = st.file_uploader("上传一个 CSV 文件", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    profile = profile_dataframe(df)

    st.success("数据上传成功")

    st.write("### 数据预览")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("行数", profile["n_rows"])
    col2.metric("列数", profile["n_cols"])
    col3.metric("重复行数", profile["duplicate_rows"])

    st.write("### 数值列")
    st.write(profile["numeric_cols"])

    st.write("### 类别列")
    st.write(profile["categorical_cols"])

    st.write("### 缺失率")
    missing_df = pd.DataFrame(
        {
            "column": list(profile["missing_ratio"].keys()),
            "missing_ratio": list(profile["missing_ratio"].values()),
        }
    )
    st.dataframe(missing_df)

else:
    st.info("请上传 CSV 文件开始。")