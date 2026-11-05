#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
SAMPLES_DIR="${SIGN_SAMPLES_DIR:-$ROOT_DIR/data/sign-samples}"
REPORTS_DIR="${SIGN_REPORTS_DIR:-$ROOT_DIR/data/sign-reports}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN"
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found in PATH: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" -m sign_classifier_service.report_results \
  --samples-dir "$SAMPLES_DIR" \
  --output-dir "$REPORTS_DIR" \
  "$@"
