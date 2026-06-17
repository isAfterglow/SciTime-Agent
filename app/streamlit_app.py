from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tools.model_trainer import infer_feature_types


st.set_page_config(
    page_title="SciTime-Agent",
    page_icon="🧪",
    layout="wide",
)


def inject_page_style():
    st.markdown(
        """
        <style>
        :root {
            --ink: #1f2a37;
            --muted: #5a6877;
            --line: #d7dde5;
            --panel: #f7f4ee;
            --panel-strong: #efe6d8;
            --accent: #b6542d;
            --accent-soft: #f4d7c8;
            --green-soft: #dfeadf;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(182, 84, 45, 0.10), transparent 28%),
                linear-gradient(180deg, #fcfaf6 0%, #f6f1e8 100%);
        }
        .hero-shell {
            padding: 1.4rem 1.6rem;
            border: 1px solid var(--line);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(247,244,238,0.96));
            box-shadow: 0 10px 30px rgba(31, 42, 55, 0.06);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: var(--accent);
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            color: var(--ink);
            font-family: "Source Serif 4", Georgia, serif;
            font-size: 2.3rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 0.2rem 0 0.5rem 0;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
            margin: 0;
        }
        .section-title {
            color: var(--ink);
            font-family: "Source Serif 4", Georgia, serif;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0.2rem 0 0.7rem 0;
        }
        .section-index {
            display: inline-block;
            padding: 0.16rem 0.55rem;
            margin-right: 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .card {
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255,255,255,0.88);
            box-shadow: 0 6px 20px rgba(31, 42, 55, 0.04);
        }
        .mini-card {
            padding: 0.85rem 1rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--panel);
        }
        .mini-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .mini-value {
            color: var(--ink);
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }
        .hint {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .good-note {
            padding: 0.85rem 1rem;
            border: 1px solid #c8d8c8;
            border-radius: 14px;
            background: var(--green-soft);
            color: var(--ink);
        }
        .warn-note {
            padding: 0.85rem 1rem;
            border: 1px solid #ead5ae;
            border-radius: 14px;
            background: #fbf3df;
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f2e9dd 0%, #f8f4ec 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def api_get_health(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def api_profile_dataset(api_base_url: str, file_name: str, file_bytes: bytes, target_col: str | None = None) -> dict:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/profile",
        files={"file": (file_name, file_bytes, "text/csv")},
        data={"target_col": target_col or ""},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def api_train_basic(
    api_base_url: str,
    *,
    file_name: str,
    file_bytes: bytes,
    target_col: str,
    task_type: str | None,
    test_size: float,
    random_state: int,
    feature_cols: list[str],
    include_models: list[str],
) -> dict:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/train/basic",
        files={"file": (file_name, file_bytes, "text/csv")},
        data={
            "target_col": target_col,
            "task_type": task_type or "",
            "test_size": str(test_size),
            "random_state": str(random_state),
            "feature_cols": ",".join(feature_cols),
            "include_models": ",".join(include_models),
        },
        timeout=300,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"后端训练失败：{detail}")
    return response.json()


def api_list_tasks(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url.rstrip('/')}/tasks", timeout=20)
    response.raise_for_status()
    return response.json()


def api_run_registered_task(api_base_url: str, **payload_kwargs) -> dict:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/tasks/run",
        data=payload_kwargs,
        timeout=300,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"后端任务执行失败：{detail}")
    return response.json()


def detect_id_like_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    pattern = re.compile(r"(?:^id$|_id$|^idx$|index|sample|name|编号|序号)", re.IGNORECASE)
    candidates = []
    for col in df.columns:
        if col == target_col:
            continue
        if pattern.search(col):
            candidates.append(col)
    return candidates


def suggest_feature_columns(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    excluded = detect_id_like_columns(df, target_col)
    feature_cols = [col for col in df.columns if col != target_col and col not in excluded]
    return feature_cols, excluded


def sync_feature_selection_state(
    *,
    file_name: str,
    all_columns: list[str],
    target_col: str,
    suggested_features: list[str],
) -> list[str]:
    state_key = "selected_feature_cols"
    context_key = "selected_feature_cols_context"
    feature_options = [col for col in all_columns if col != target_col]
    current_context = {
        "file_name": file_name,
        "columns": tuple(all_columns),
        "target_col": target_col,
    }
    previous_context = st.session_state.get(context_key)

    existing_selection = st.session_state.get(state_key, [])
    sanitized_selection = [col for col in existing_selection if col in feature_options]

    if previous_context is None:
        st.session_state[state_key] = suggested_features
    elif previous_context["file_name"] != file_name or previous_context["columns"] != tuple(all_columns):
        st.session_state[state_key] = suggested_features
    elif previous_context["target_col"] != target_col:
        st.session_state[state_key] = sanitized_selection or suggested_features
    else:
        st.session_state[state_key] = sanitized_selection or suggested_features

    st.session_state[context_key] = current_context
    return st.session_state[state_key]


def sync_model_selection_state(task_type: str, model_options: list[str]) -> list[str]:
    state_key = "selected_models"
    context_key = "selected_models_context"
    previous_task_type = st.session_state.get(context_key)
    existing_selection = st.session_state.get(state_key, [])
    sanitized_selection = [model for model in existing_selection if model in model_options]

    if previous_task_type != task_type:
        st.session_state[state_key] = model_options
    else:
        st.session_state[state_key] = sanitized_selection or model_options

    st.session_state[context_key] = task_type
    return st.session_state[state_key]


def render_section_title(index: str, title: str):
    st.markdown(
        f"""
        <div class="section-title">
            <span class="section-index">{index}</span>{title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(profile: dict):
    col1, col2, col3 = st.columns(3)
    for col, label, value in [
        (col1, "Rows", profile["n_rows"]),
        (col2, "Columns", profile["n_cols"]),
        (col3, "Duplicate Rows", profile["duplicate_rows"]),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def format_ratio_df(profile: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": list(profile["missing_ratio"].keys()),
            "missing_ratio": list(profile["missing_ratio"].values()),
        }
    ).sort_values("missing_ratio", ascending=False)


def format_numeric_summary_df(profile: dict) -> pd.DataFrame:
    numeric_summary = profile.get("numeric_summary", {})
    if not numeric_summary:
        return pd.DataFrame(columns=["column", "mean", "std", "min", "max"])
    rows = [{"column": col, **stats} for col, stats in numeric_summary.items()]
    return pd.DataFrame(rows)


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            non_null = safe_df[col].dropna()
            if non_null.empty:
                safe_df[col] = safe_df[col].astype("string")
                continue
            mixed_string_like = any(isinstance(v, str) for v in non_null) and any(not isinstance(v, str) for v in non_null)
            if mixed_string_like:
                safe_df[col] = safe_df[col].map(lambda v: "" if pd.isna(v) else str(v)).astype("string")
    return safe_df


def infer_task_type_from_profile(profile: dict, target_col: str) -> str:
    if target_col in profile.get("numeric_cols", []):
        return "regression"
    return "classification"


def build_target_validation_message(df: pd.DataFrame, target_col: str, inferred_task_type: str) -> tuple[str, str]:
    series = df[target_col]
    non_null = series.dropna()
    unique_count = non_null.nunique()
    lower_name = target_col.lower()

    if unique_count < 2:
        return "error", f"目标列 `{target_col}` 只有 {unique_count} 个非空唯一值，不适合训练。"
    if lower_name.startswith("raw_"):
        return "warning", f"目标列 `{target_col}` 更像元数据/说明列，通常不建议作为训练目标。"
    if inferred_task_type == "regression" and target_col not in df.select_dtypes(include=['number']).columns:
        return "error", f"目标列 `{target_col}` 不是数值列，不适合回归任务。"
    if lower_name.startswith("target_"):
        return "success", f"目标列 `{target_col}` 看起来是监督学习标签列，适合当前训练流程。"
    return "info", f"目标列 `{target_col}` 可以尝试训练，但请确认它确实是你想预测的标签。"


def main():
    inject_page_style()

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">SciTime-Agent / Basic Pipeline</div>
            <div class="hero-title">科学数据建模工作台</div>
            <p class="hero-copy">
                这个页面对应当前已经实现的内容：上传 CSV、做数据画像、选择标签与特征、训练基础 baseline，
                并查看指标与预测结果。默认不会盲目使用全部列，而是优先排除疑似 ID 列后再给出建议特征集。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "01 上传数据",
            "02 数据画像",
            "03 任务配置",
            "04 特征配置",
            "05 模型训练",
            "06 结果展示",
        ]
    )

    with st.sidebar:
        st.markdown("## SciTime-Agent")
        st.markdown("`v0.1-basic-pipeline`")
        api_base_url = st.text_input("API 地址", value="http://127.0.0.1:8000")
        if st.button("检查 API 状态", width="stretch"):
            try:
                health = api_get_health(api_base_url)
                st.success(f"API 正常：{health.get('message', 'ok')}")
            except Exception as exc:
                st.error(f"API 不可用：{exc}")
        st.markdown(
            """
            **流程导航**

            1. 上传 CSV
            2. 查看数据画像
            3. 选择标签列与任务类型
            4. 调整特征列
            5. 训练 baseline
            6. 查看指标与预测结果
            """
        )

    with tabs[0]:
        render_section_title("01", "上传数据")
        st.markdown(
            '<div class="card"><p class="hint">支持普通表格型 CSV。后续如果要接更复杂的科学时序数据，可以把特征工程先整理成表格后再接入这里的 baseline 流程。</p></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"], label_visibility="visible")

    if uploaded_file is None:
        with tabs[1]:
            render_section_title("02", "数据画像")
            st.info("先在“01 上传数据”里选择 CSV 文件。")
        with tabs[2]:
            render_section_title("03", "任务配置")
            st.info("上传数据后，这里会显示目标列和任务类型配置。")
        with tabs[3]:
            render_section_title("04", "特征配置")
            st.info("上传数据后，这里会显示建议特征集和手动选择器。")
        with tabs[4]:
            render_section_title("05", "模型训练")
            st.info("上传数据并完成配置后，这里可以启动 baseline 训练。")
        with tabs[5]:
            render_section_title("06", "结果展示")
            st.info("训练完成后，这里会展示指标和预测结果。")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    df = load_uploaded_dataframe(file_bytes)
    try:
        profile_payload = api_profile_dataset(
            api_base_url,
            file_name=uploaded_file.name,
            file_bytes=file_bytes,
        )
        profile = profile_payload["profile"]
        recommended_tasks = profile_payload.get("recommended_tasks", [])
    except Exception as exc:
        st.error(f"无法从 FastAPI 获取数据画像：{exc}")
        st.stop()

    default_target = profile["target_col"] if profile["target_col"] in df.columns else df.columns[-1]

    with st.sidebar:
        st.markdown("### 当前数据")
        st.write(f"文件名：`{uploaded_file.name}`")
        st.write(f"行数：`{profile['n_rows']}`")
        st.write(f"列数：`{profile['n_cols']}`")
        st.write(f"猜测标签列：`{default_target}`")

    with tabs[1]:
        render_section_title("02", "数据画像")
        render_metric_cards(profile)
        st.markdown("")

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("数据预览")
            st.dataframe(make_arrow_safe(df.head(20)), width="stretch", height=360)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("字段类型")
            type_df = pd.DataFrame(
                {
                    "type": ["numeric", "categorical", "datetime"],
                    "count": [
                        len(profile["numeric_cols"]),
                        len(profile["categorical_cols"]),
                        len(profile["datetime_cols"]),
                    ],
                }
            )
            st.dataframe(make_arrow_safe(type_df), width="stretch", hide_index=True)
            st.write("缺失率")
            st.dataframe(make_arrow_safe(format_ratio_df(profile)), width="stretch", height=230)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("数值列统计")
        st.dataframe(make_arrow_safe(format_numeric_summary_df(profile)), width="stretch", height=260)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("任务推荐")
        if recommended_tasks:
            rec_df = pd.DataFrame(recommended_tasks)[["task_name", "pipeline_type", "score", "reason"]]
            st.dataframe(make_arrow_safe(rec_df), width="stretch", hide_index=True)
            recommended_task_names = [item["task_name"] for item in recommended_tasks]
            task_to_run = st.selectbox(
                "选择推荐任务",
                options=recommended_task_names,
                key="recommended_task_to_run",
            )
            if st.button("执行推荐任务", width="stretch"):
                try:
                    if task_to_run == "generic_tabular":
                        raise RuntimeError("`generic_tabular` 推荐任务请在“05 模型训练”页执行，因为它依赖当前上传文件与页面配置。")

                    run_payload = {"task_name": task_to_run}
                    if task_to_run.startswith("battery_") or task_to_run == "battery_summary":
                        run_payload.update(
                            {
                                "config_path": "configs/battery_expt_2_2.yaml",
                            }
                        )
                    task_run_payload = api_run_registered_task(api_base_url, **run_payload)
                    st.session_state["latest_task_run_payload"] = task_run_payload
                    st.success(f"任务 `{task_to_run}` 执行完成。")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("当前没有可用的任务推荐。")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        render_section_title("03", "任务配置")

        config_col1, config_col2 = st.columns(2)
        with config_col1:
            target_col = st.selectbox(
                "标签列",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(default_target),
                help="这是要预测的列，不会被放入输入特征。",
            )

        try:
            target_profile_payload = api_profile_dataset(
                api_base_url,
                file_name=uploaded_file.name,
                file_bytes=file_bytes,
                target_col=target_col,
            )
            target_profile = target_profile_payload["profile"]
        except Exception:
            target_profile = profile

        resolved_auto_task = infer_task_type_from_profile(target_profile, target_col)
        with config_col2:
            task_type_mode = st.selectbox(
                "任务类型",
                options=["auto", "regression", "classification"],
                index=0,
                help="`auto` 会根据标签列类型自动判断。",
            )

        effective_task_type = resolved_auto_task if task_type_mode == "auto" else task_type_mode
        st.markdown(
            f'<div class="good-note">当前生效任务类型：<strong>{effective_task_type}</strong>。自动判断结果为：<strong>{resolved_auto_task}</strong>。</div>',
            unsafe_allow_html=True,
        )
        validation_level, validation_message = build_target_validation_message(df, target_col, effective_task_type)
        if validation_level == "error":
            st.error(validation_message)
        elif validation_level == "warning":
            st.warning(validation_message)
        elif validation_level == "success":
            st.success(validation_message)
        else:
            st.info(validation_message)

        train_col1, train_col2 = st.columns(2)
        with train_col1:
            test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        with train_col2:
            random_state = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

        model_options = (
            ["LinearRegression", "RandomForestRegressor"]
            if effective_task_type == "regression"
            else ["LogisticRegression", "RandomForestClassifier"]
        )
        selected_models_default = sync_model_selection_state(effective_task_type, model_options)
        selected_models = st.multiselect(
            "模型列表",
            options=model_options,
            default=selected_models_default,
            key="selected_models",
            help="这里对应当前代码里已经实现的基础 baseline 模型。",
        )

    with tabs[3]:
        render_section_title("04", "特征配置")

        suggested_features, excluded_id_like = suggest_feature_columns(df, target_col)
        selected_feature_cols = sync_feature_selection_state(
            file_name=uploaded_file.name,
            all_columns=df.columns.tolist(),
            target_col=target_col,
            suggested_features=suggested_features,
        )
        feature_options = [col for col in df.columns if col != target_col]

        feature_left, feature_right = st.columns([1.25, 0.75])
        with feature_left:
            st.multiselect(
                "输入特征列",
                options=feature_options,
                key="selected_feature_cols",
                help="默认建议会排除标签列和疑似 ID 列。你也可以手动调整。",
            )
        with feature_right:
            st.markdown(
                f"""
                <div class="warn-note">
                    默认排除的疑似 ID 列：<strong>{", ".join(excluded_id_like) if excluded_id_like else "无"}</strong><br/>
                    建议特征数：<strong>{len(suggested_features)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("恢复建议特征", width="stretch"):
                st.session_state.selected_feature_cols = suggested_features
                st.rerun()

        selected_feature_cols = st.session_state.selected_feature_cols
        feature_numeric_cols, feature_categorical_cols = infer_feature_types(df, selected_feature_cols)

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("已选特征数", len(selected_feature_cols))
        summary_col2.metric("数值特征数", len(feature_numeric_cols))
        summary_col3.metric("类别特征数", len(feature_categorical_cols))

        with st.expander("查看已选特征列表", expanded=False):
            st.write(selected_feature_cols)

    with tabs[4]:
        render_section_title("05", "模型训练")
        st.markdown(
            '<div class="card"><p class="hint">训练逻辑对应当前项目里已经完成的基础 tabular baseline。这里不会做自动调参，也不会做复杂时序建模，只做第一周规划里的基础训练流程。</p></div>',
            unsafe_allow_html=True,
        )

        if not selected_feature_cols:
            st.error("当前没有选择任何特征列，无法训练。")
        elif not selected_models:
            st.error("当前没有选择任何模型，无法训练。")
        else:
            st.write("本次训练配置")
            config_df = pd.DataFrame(
                [
                    {"item": "target_col", "value": target_col},
                    {"item": "task_type", "value": effective_task_type},
                    {"item": "test_size", "value": test_size},
                    {"item": "random_state", "value": random_state},
                    {"item": "n_features", "value": len(selected_feature_cols)},
                    {"item": "models", "value": ", ".join(selected_models)},
                ]
            )
            st.dataframe(make_arrow_safe(config_df), width="stretch", hide_index=True)

            if st.button("开始训练 Baseline", type="primary", width="stretch"):
                try:
                    with st.spinner("正在训练 baseline ..."):
                        payload = api_train_basic(
                            api_base_url,
                            file_name=uploaded_file.name,
                            file_bytes=file_bytes,
                            target_col=target_col,
                            task_type=None if task_type_mode == "auto" else effective_task_type,
                            test_size=test_size,
                            random_state=random_state,
                            feature_cols=selected_feature_cols,
                            include_models=selected_models,
                        )
                    st.session_state["latest_train_result"] = {
                        "metrics_df": pd.DataFrame(payload["metrics"]),
                        "predictions_df": pd.DataFrame(payload["predictions"]),
                        "feature_cols": payload["feature_cols"],
                        "numeric_cols": payload["numeric_cols"],
                        "categorical_cols": payload["categorical_cols"],
                        "task_type": payload["task_type"],
                    }
                    st.session_state["latest_train_target_col"] = target_col
                    st.session_state["latest_train_task_type"] = payload["task_type"]
                    st.success("训练完成，请切到“06 结果展示”查看输出。")
                except Exception as exc:
                    st.error(str(exc))

    with tabs[5]:
        render_section_title("06", "结果展示")
        task_run_payload = st.session_state.get("latest_task_run_payload")
        if task_run_payload is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("注册任务执行结果")
            st.dataframe(
                make_arrow_safe(pd.DataFrame(task_run_payload.get("summary", []))),
                width="stretch",
                hide_index=True,
            )
            artifacts = task_run_payload.get("artifacts", {})
            if artifacts:
                artifact_df = pd.DataFrame(
                    [{"artifact": key, "path": value} for key, value in artifacts.items()]
                )
                st.write("Artifacts")
                st.dataframe(make_arrow_safe(artifact_df), width="stretch", hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        result = st.session_state.get("latest_train_result")
        if result is None:
            st.info("还没有训练结果。先到“05 模型训练”点击开始训练。")
        else:
            result_col1, result_col2 = st.columns([1.0, 1.0])
            with result_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("指标表")
                st.dataframe(make_arrow_safe(result["metrics_df"]), width="stretch", hide_index=True)
                st.download_button(
                    "下载指标 CSV",
                    data=result["metrics_df"].to_csv(index=False).encode("utf-8"),
                    file_name="metrics.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with result_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("训练摘要")
                summary_df = pd.DataFrame(
                    [
                        {"item": "task_type", "value": st.session_state.get("latest_train_task_type")},
                        {"item": "target_col", "value": st.session_state.get("latest_train_target_col")},
                        {"item": "n_features", "value": len(result["feature_cols"])},
                        {"item": "numeric_features", "value": len(result["numeric_cols"])},
                        {"item": "categorical_features", "value": len(result["categorical_cols"])},
                        {"item": "n_prediction_rows", "value": len(result["predictions_df"])},
                    ]
                )
                st.dataframe(make_arrow_safe(summary_df), width="stretch", hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("预测结果预览")
            st.dataframe(make_arrow_safe(result["predictions_df"].head(50)), width="stretch", height=320)
            st.download_button(
                "下载预测结果 CSV",
                data=result["predictions_df"].to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                width="stretch",
            )
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
