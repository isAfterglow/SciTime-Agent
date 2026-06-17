from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.task_registry import list_registered_tasks, recommend_registered_tasks, run_registered_task
from src.tools.data_profiler import profile_dataframe
from src.tools.model_trainer import run_tabular_baselines


app = FastAPI(
    title="SciTime-Agent API",
    description="Scientific time-series modeling agent backend",
    version="0.1.0",
)


def _read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return pd.read_csv(io.BytesIO(content))


def _normalize_task_results(results):
    normalized_results = []
    artifacts = {}
    summary = []
    for item in results:
        if isinstance(item, dict):
            normalized_item = {
                "name": "result",
                "outputs": item,
            }
        else:
            normalized_item = {
                "name": getattr(item, "name", "result"),
                "outputs": getattr(item, "outputs", {}),
            }
        normalized_results.append(normalized_item)
        if isinstance(normalized_item["outputs"], dict):
            for key, value in normalized_item["outputs"].items():
                if isinstance(value, str):
                    artifacts[f"{normalized_item['name']}.{key}"] = value
            summary.append(
                {
                    "step": normalized_item["name"],
                    "n_outputs": len(normalized_item["outputs"]),
                }
            )
    return normalized_results, artifacts, summary


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SciTime-Agent backend is running.",
    }


@app.get("/tasks")
def list_tasks():
    return {
        "status": "ok",
        "tasks": list_registered_tasks(),
    }


@app.post("/profile")
def profile_dataset(file: UploadFile = File(...), target_col: str | None = Form(default=None)):
    try:
        df = _read_uploaded_csv(file)
        profile = profile_dataframe(df, target_col=target_col)
        recommendations = recommend_registered_tasks(profile)
        return {
            "status": "ok",
            "file_name": file.filename,
            "profile": profile,
            "recommended_tasks": recommendations,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tasks/recommend")
def recommend_tasks(file: UploadFile = File(...), target_col: str | None = Form(default=None)):
    try:
        df = _read_uploaded_csv(file)
        profile = profile_dataframe(df, target_col=target_col)
        return {
            "status": "ok",
            "file_name": file.filename,
            "profile": profile,
            "recommended_tasks": recommend_registered_tasks(profile),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/basic")
def train_basic(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    task_type: str | None = Form(default=None),
    test_size: float = Form(default=0.2),
    random_state: int = Form(default=42),
    feature_cols: str | None = Form(default=None),
    include_models: str | None = Form(default=None),
):
    try:
        df = _read_uploaded_csv(file)

        parsed_feature_cols = None
        if feature_cols:
            parsed_feature_cols = [col for col in feature_cols.split(",") if col]

        parsed_models = None
        if include_models:
            parsed_models = [model for model in include_models.split(",") if model]

        result = run_tabular_baselines(
            df=df,
            target_col=target_col,
            feature_cols=parsed_feature_cols,
            task_type=task_type,
            test_size=test_size,
            random_state=random_state,
            include_models=parsed_models,
        )
        return {
            "status": "ok",
            "file_name": file.filename,
            "task_type": result.task_type,
            "target_col": result.target_col,
            "feature_cols": result.feature_cols,
            "numeric_cols": result.numeric_cols,
            "categorical_cols": result.categorical_cols,
            "metrics": result.metrics_df.to_dict(orient="records"),
            "predictions": result.predictions_df.to_dict(orient="records"),
            "n_predictions": int(len(result.predictions_df)),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tasks/run")
def run_task(
    task_name: str = Form(...),
    dataset_path: str | None = Form(default=None),
    target_col: str | None = Form(default=None),
    out_dir: str | None = Form(default=None),
    task_type: str | None = Form(default=None),
    feature_cols: str | None = Form(default=None),
    include_models: str | None = Form(default=None),
    config_path: str | None = Form(default=None),
    step: str | None = Form(default=None),
):
    try:
        kwargs = {}
        if dataset_path:
            kwargs["dataset_path"] = dataset_path
        if target_col:
            kwargs["target_col"] = target_col
        if out_dir:
            kwargs["out_dir"] = out_dir
        if task_type:
            kwargs["task_type"] = task_type
        if feature_cols:
            kwargs["feature_cols"] = [col for col in feature_cols.split(",") if col]
        if include_models:
            kwargs["include_models"] = [model for model in include_models.split(",") if model]
        if config_path:
            kwargs["config_path"] = config_path
        if step:
            kwargs["step"] = step

        results = run_registered_task(task_name, **kwargs)
        normalized_results, artifacts, summary = _normalize_task_results(results)
        return {
            "status": "ok",
            "task_name": task_name,
            "pipeline_type": next(
                (task["pipeline_type"] for task in list_registered_tasks() if task["task_name"] == task_name),
                "unknown",
            ),
            "artifacts": artifacts,
            "summary": summary,
            "errors": [],
            "results": normalized_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
