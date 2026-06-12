"""API FastAPI que serve o pipeline completo (preprocessor + modelo) registrado pelo agente.

Roda com: uvicorn serving.app:app --port 8000
O modelo é carregado de pipeline_artifacts/full_pipeline.joblib (gerado por register_model).
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ARTIFACTS = Path("pipeline_artifacts")

app = FastAPI(title="MLOps DeepAgent Model API", version="1.0")

_pipeline = None
_state: dict[str, Any] = {}


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


@app.on_event("startup")
def load_model() -> None:
    global _pipeline, _state
    state_file = ARTIFACTS / "state.json"
    if state_file.exists():
        _state = json.loads(state_file.read_text())
    pipeline_path = ARTIFACTS / "full_pipeline.joblib"
    if pipeline_path.exists():
        _pipeline = joblib.load(pipeline_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "registered_model_name": _state.get("registered_model_name"),
        "version": _state.get("registered_model_version"),
        "alias": _state.get("registered_model_alias"),
        "best_estimator": _state.get("best_estimator"),
        "task": _state.get("task_type"),
        "test_metric": _state.get("cv_metric_value"),
        "expected_columns": _state.get("raw_feature_columns"),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    if not req.records:
        raise HTTPException(status_code=422, detail="Lista 'records' vazia.")

    df = pd.DataFrame(req.records)
    expected = _state.get("raw_feature_columns") or []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Colunas faltando: {missing}")
    df = df[expected] if expected else df

    try:
        preds = _pipeline.predict(df)
        out: dict[str, Any] = {"predictions": preds.tolist()}
        if hasattr(_pipeline, "predict_proba"):
            out["probabilities"] = _pipeline.predict_proba(df).tolist()
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}") from e
