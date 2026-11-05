#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

bash "$ROOT_DIR/scripts/dev-stop.sh" || true
bash "$ROOT_DIR/scripts/dev-start.sh"
