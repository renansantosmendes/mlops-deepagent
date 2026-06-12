"""Estado compartilhado do pipeline (artefatos em disco + state.json).

Cada tool lê/escreve aqui para que o deep agent consiga retomar o pipeline
de qualquer etapa e para que as etapas se comuniquem entre si.
"""

import json
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path("pipeline_artifacts")
STATE_FILE = ARTIFACTS_DIR / "state.json"


def _ensure_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, Any]:
    _ensure_dir()
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(updates: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    state.update(updates)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    return state


def artifact_path(name: str) -> str:
    _ensure_dir()
    return str(ARTIFACTS_DIR / name)
