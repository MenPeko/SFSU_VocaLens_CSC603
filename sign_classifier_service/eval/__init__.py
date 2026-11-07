"""Held-out evaluation for the core-sign classifier. See docs/SIGN_EVAL_PLAN.md."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..classifier import APP_ROOT

EVAL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EVAL_DIR / "config.json"
GLOSS_MAP_PATH = EVAL_DIR / "gloss_map.json"
TEST_SESSIONS_PATH = EVAL_DIR / "test_sessions.json"
EXPERIMENTS_DIR = EVAL_DIR / "experiments"

SESSIONS_DIR = APP_ROOT / "data" / "sign-samples" / "sessions"
SIGN_EVAL_DATA_DIR = APP_ROOT / "data" / "sign-eval"
ASL_CITIZEN_DIR = APP_ROOT / "data" / "asl-citizen"
LANDMARKS_DIR = APP_ROOT / "data" / "landmarks"
REPORTS_DIR = APP_ROOT / "data" / "sign-reports" / "eval"
MODELS_DIR = APP_ROOT / "models" / "sign"
RESULTS_DOC_PATH = APP_ROOT / "docs" / "SIGN_RESULTS.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
