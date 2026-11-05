#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_PID_FILE="$ROOT_DIR/.devserver.pid"
SERVER_LOG_FILE="$ROOT_DIR/.devserver.log"
TUNNEL_PID_FILE="$ROOT_DIR/.ngrok.pid"
TUNNEL_LOG_FILE="$ROOT_DIR/.ngrok.log"
CLASSIFIER_PID_FILE="$ROOT_DIR/.sign-classifier.pid"
CLASSIFIER_LOG_FILE="$ROOT_DIR/.sign-classifier.log"

PORT="${PORT:-3333}"
SIGN_CLASSIFIER_PORT="${SIGN_CLASSIFIER_PORT:-8001}"
PYTHON_BIN="${PYTHON_BIN:-}"
DEMO_REQUIRE_STABLE_NGROK="${DEMO_REQUIRE_STABLE_NGROK:-false}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
  PORT="${PORT:-3333}"
  DEMO_REQUIRE_STABLE_NGROK="${DEMO_REQUIRE_STABLE_NGROK:-false}"
fi

if [[ -z "${NGROK_DOMAIN:-}" ]]; then
  echo "========================================="
  echo "  WARNING: NGROK_DOMAIN is not set"
  echo "========================================="
  echo ""
  echo "A random ngrok URL will be used for this run."
  echo "If you restart the server or tunnel, the URL may change and"
  echo "written-text photo uploads can start timing out until you:"
  echo "  1. update the Mentra app publicUrl"
  echo "  2. relaunch the app session on the glasses"
  echo ""
  echo "For demos, set NGROK_DOMAIN to a stable domain."
  echo ""

  if [[ "$DEMO_REQUIRE_STABLE_NGROK" == "true" ]]; then
    echo "DEMO_REQUIRE_STABLE_NGROK=true is set, so refusing to start without NGROK_DOMAIN."
    exit 1
  fi
fi

# ============================================
# Step 0: Run tests before starting
# ============================================

echo "Running tests..."
cd "$ROOT_DIR"
if ! bun test; then
  echo ""
  echo "Tests failed. Fix the errors above before starting."
  exit 1
fi
echo "All tests passed."
echo ""

# ============================================
# Step 0.5: Start sign classifier service
# ============================================

if [[ -f "$CLASSIFIER_PID_FILE" ]]; then
  EXISTING_CLASSIFIER_PID="$(cat "$CLASSIFIER_PID_FILE")"
  if kill -0 "$EXISTING_CLASSIFIER_PID" >/dev/null 2>&1; then
    echo "Sign classifier already running with PID $EXISTING_CLASSIFIER_PID"
  else
    rm -f "$CLASSIFIER_PID_FILE"
  fi
fi

if [[ ! -f "$CLASSIFIER_PID_FILE" ]]; then
  if command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    cd "$ROOT_DIR"
    nohup "$PYTHON_BIN" -m uvicorn sign_classifier_service.app:app --host 127.0.0.1 --port "$SIGN_CLASSIFIER_PORT" >"$CLASSIFIER_LOG_FILE" 2>&1 &
    CLASSIFIER_PID=$!
    echo "$CLASSIFIER_PID" >"$CLASSIFIER_PID_FILE"
    sleep 2

    if kill -0 "$CLASSIFIER_PID" >/dev/null 2>&1; then
      echo "Started sign classifier (PID $CLASSIFIER_PID)"
      echo "Log file: $CLASSIFIER_LOG_FILE"
    else
      echo "Sign classifier failed to start. Gemini fallback will still work."
      tail -n 40 "$CLASSIFIER_LOG_FILE" || true
      rm -f "$CLASSIFIER_PID_FILE"
    fi
  else
    echo "FastAPI/uvicorn not installed in $PYTHON_BIN. Gemini fallback will still work."
    echo "Install Python deps into the app virtualenv with:"
    echo "  .venv/bin/pip install -r sign_classifier_service/requirements.txt"
  fi
fi

echo ""

# ============================================
# Step 1: Start dev server
# ============================================

if [[ -f "$SERVER_PID_FILE" ]]; then
  EXISTING_PID="$(cat "$SERVER_PID_FILE")"
  if kill -0 "$EXISTING_PID" >/dev/null 2>&1; then
    echo "Server already running with PID $EXISTING_PID"
  else
    rm -f "$SERVER_PID_FILE"
  fi
fi

if [[ ! -f "$SERVER_PID_FILE" ]]; then
  cd "$ROOT_DIR"
  nohup bun run dev >"$SERVER_LOG_FILE" 2>&1 &
  NEW_PID=$!
  echo "$NEW_PID" >"$SERVER_PID_FILE"

  sleep 2
  if kill -0 "$NEW_PID" >/dev/null 2>&1; then
    echo "Started dev server (PID $NEW_PID)"
    echo "Log file: $SERVER_LOG_FILE"
  else
    echo "Failed to start dev server"
    tail -n 40 "$SERVER_LOG_FILE" || true
    exit 1
  fi
else
  echo "Dev server already running"
fi

# ============================================
# Step 2: Start ngrok tunnel
# ============================================

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok is not installed. Install via: brew install ngrok"
  exit 1
fi

if [[ -f "$TUNNEL_PID_FILE" ]]; then
  EXISTING_TUNNEL_PID="$(cat "$TUNNEL_PID_FILE")"
  if kill -0 "$EXISTING_TUNNEL_PID" >/dev/null 2>&1; then
    echo "ngrok tunnel already running with PID $EXISTING_TUNNEL_PID"
  else
    rm -f "$TUNNEL_PID_FILE"
  fi
fi

if [[ ! -f "$TUNNEL_PID_FILE" ]]; then
  cd "$ROOT_DIR"

  # Use stable domain if NGROK_DOMAIN is set, otherwise use random
  if [[ -n "${NGROK_DOMAIN:-}" ]]; then
    nohup ngrok http --url="$NGROK_DOMAIN" "${PORT}" >"$TUNNEL_LOG_FILE" 2>&1 &
  else
    nohup ngrok http "${PORT}" >"$TUNNEL_LOG_FILE" 2>&1 &
  fi
  TUNNEL_PID=$!
  echo "$TUNNEL_PID" >"$TUNNEL_PID_FILE"
  sleep 3

  if kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
    echo "Started ngrok tunnel (PID $TUNNEL_PID)"
    echo "Log file: $TUNNEL_LOG_FILE"
  else
    echo "Failed to start ngrok tunnel"
    tail -n 40 "$TUNNEL_LOG_FILE" || true
    exit 1
  fi
fi

# Extract tunnel URL from ngrok's local API
TUNNEL_URL=""
for attempt in 1 2 3; do
  TUNNEL_URL="$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | grep -o '"public_url":"[^"]*"' \
    | head -1 \
    | sed 's/"public_url":"//;s/"//' || true)"
  if [[ -n "$TUNNEL_URL" ]]; then
    break
  fi
  sleep 2
done

echo ""
if [[ -n "$TUNNEL_URL" ]]; then
  echo "========================================="
  echo "  Tunnel URL: $TUNNEL_URL"
  echo "========================================="
  echo ""
  if [[ -n "${NGROK_DOMAIN:-}" ]]; then
    echo "  Set this as your app's publicUrl in the"
    echo "  Mentra developer console."
    echo ""
  else
    echo "  Random ngrok URL detected."
    echo "  Set this as your app's publicUrl in the"
    echo "  Mentra developer console and relaunch the"
    echo "  app on the glasses if this URL changes."
    echo ""
  fi
else
  echo "Could not detect tunnel URL. Check http://127.0.0.1:4040"
fi
