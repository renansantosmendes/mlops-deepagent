"""Tools de deploy: sobe a API FastAPI (uvicorn) servindo o modelo registrado,
verifica saúde e derruba o serviço."""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from .state import load_state, save_state

logger = logging.getLogger(__name__)

SERVING_APP = "serving.app:app"


def deploy_model_api(port: int = 8000) -> str:
    """Faz o deploy do modelo registrado em uma API FastAPI local (uvicorn em background).

    Endpoints: GET /health, GET /model-info, POST /predict.
    Requer modelo registrado (full_pipeline.joblib presente).

    Args:
        port: Porta da API (padrão 8000).
    """
    logger.info("🚀 Iniciando deploy da API...")
    
    state = load_state()
    if "full_pipeline_path" not in state:
        logger.error("❌ Erro: nenhum modelo registrado")
        return "ERRO: nenhum modelo registrado. Rode register_model primeiro."

    logger.info(f"🌐 Porta: {port}")
    logger.info("🔧 Iniciando servidor uvicorn...")
    log = open("api_server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", SERVING_APP, "--host", "0.0.0.0", "--port", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
    )
    logger.info("⏳ Aguardando inicialização do servidor...")
    time.sleep(4)
    if proc.poll() is not None:
        logger.error("❌ Servidor caiu ao iniciar")
        return f"ERRO: servidor caiu ao iniciar. Veja api_server.log:\n{Path('api_server.log').read_text()[-1500:]}"
    
    logger.info(f"✅ API em execução! PID: {proc.pid}")

    save_state({"api_pid": proc.pid, "api_port": port, "api_url": f"http://localhost:{port}"})
    return (
        f"API no ar em http://localhost:{port} (PID {proc.pid}). "
        f"Endpoints: GET /health, GET /model-info, POST /predict. "
        f"Use check_api_health para validar com uma predição real."
    )


def check_api_health() -> str:
    """Verifica a API deployada: chama /health, /model-info e faz um POST /predict
    de smoke test com uma linha real do dataset."""
    import pandas as pd
    import requests

    state = load_state()
    url = state.get("api_url")
    if not url:
        return "ERRO: nenhuma API deployada."

    try:
        health = requests.get(f"{url}/health", timeout=5).json()
        info = requests.get(f"{url}/model-info", timeout=5).json()
        raw = pd.read_parquet(state["raw_data_path"])
        sample = raw.drop(columns=[state["target_column"]]).head(2)
        pred = requests.post(
            f"{url}/predict",
            json={"records": json.loads(sample.to_json(orient="records"))},
            timeout=15,
        ).json()
    except Exception as e:  # noqa: BLE001
        return f"ERRO ao chamar a API: {e}"

    return (
        f"health: {health}\nmodel-info: {info}\n"
        f"smoke test /predict (2 linhas): {json.dumps(pred)[:800]}\nDeploy validado com sucesso."
    )


def stop_api() -> str:
    """Derruba o servidor da API FastAPI que estiver rodando."""
    import os
    import signal

    state = load_state()
    pid = state.get("api_pid")
    if not pid:
        return "Nenhuma API rodando."
    try:
        os.kill(pid, signal.SIGTERM)
        return f"API (PID {pid}) finalizada."
    except ProcessLookupError:
        return "Processo já não existia."
