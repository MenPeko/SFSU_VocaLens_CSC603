#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_PID_FILE="$ROOT_DIR/.devserver.pid"
TUNNEL_PID_FILE="$ROOT_DIR/.ngrok.pid"
CLASSIFIER_PID_FILE="$ROOT_DIR/.sign-classifier.pid"

PORT="${PORT:-3333}"
BUN_PORT="${BUN_PORT:-$((PORT + 1))}"
SIGN_CLASSIFIER_PORT="${SIGN_CLASSIFIER_PORT:-8001}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
  PORT="${PORT:-3333}"
fi

stopped_any=false

# ============================================
# Stop dev server (parent + children)
# ============================================

if [[ -f "$SERVER_PID_FILE" ]]; then
  PID="$(cat "$SERVER_PID_FILE")"

  if kill -0 "$PID" >/dev/null 2>&1; then
    # Kill the entire process group to catch child processes (bun --hot)
    kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done

    if kill -0 "$PID" >/dev/null 2>&1; then
      kill -9 -- -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null || true
    fi

    echo "Stopped dev server (PID $PID)"
    stopped_any=true
  else
    echo "Dev server process $PID not running"
  fi

  rm -f "$SERVER_PID_FILE"
fi

# ============================================
# Kill any orphan bun processes on the app port
# ============================================

ORPHAN_PIDS="$(lsof -ti :"${PORT}" -ti :"${BUN_PORT}" 2>/dev/null || true)"
if [[ -n "$ORPHAN_PIDS" ]]; then
  echo "$ORPHAN_PIDS" | while read -r OPID; do
    if [[ -n "$OPID" ]] && kill -0 "$OPID" 2>/dev/null; then
      kill "$OPID" 2>/dev/null || true
      sleep 0.2
      kill -9 "$OPID" 2>/dev/null || true
      echo "Killed orphan process on port ${PORT}/${BUN_PORT} (PID $OPID)"
      stopped_any=true
    fi
  done
fi

# ============================================
# Stop ngrok tunnel
# ============================================

if [[ -f "$TUNNEL_PID_FILE" ]]; then
  TUNNEL_PID="$(cat "$TUNNEL_PID_FILE")"

  if kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
    kill "$TUNNEL_PID" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done

    if kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
      kill -9 "$TUNNEL_PID" >/dev/null 2>&1 || true
    fi

    echo "Stopped ngrok tunnel (PID $TUNNEL_PID)"
    stopped_any=true
  else
    echo "ngrok tunnel process $TUNNEL_PID not running"
  fi

  rm -f "$TUNNEL_PID_FILE"
fi

# ============================================
# Stop sign classifier
# ============================================

if [[ -f "$CLASSIFIER_PID_FILE" ]]; then
  CLASSIFIER_PID="$(cat "$CLASSIFIER_PID_FILE")"

  if kill -0 "$CLASSIFIER_PID" >/dev/null 2>&1; then
    kill "$CLASSIFIER_PID" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "$CLASSIFIER_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done

    if kill -0 "$CLASSIFIER_PID" >/dev/null 2>&1; then
      kill -9 "$CLASSIFIER_PID" >/dev/null 2>&1 || true
    fi

    echo "Stopped sign classifier (PID $CLASSIFIER_PID)"
    stopped_any=true
  else
    echo "Sign classifier process $CLASSIFIER_PID not running"
  fi

  rm -f "$CLASSIFIER_PID_FILE"
fi

CLASSIFIER_ORPHANS="$(lsof -ti :"${SIGN_CLASSIFIER_PORT}" 2>/dev/null || true)"
if [[ -n "$CLASSIFIER_ORPHANS" ]]; then
  echo "$CLASSIFIER_ORPHANS" | while read -r OPID; do
    if kill -0 "$OPID" 2>/dev/null; then
      kill "$OPID" 2>/dev/null || true
      echo "Killed orphan sign classifier on port ${SIGN_CLASSIFIER_PORT} (PID $OPID)"
      stopped_any=true
    fi
  done
fi

if [[ "$stopped_any" = false ]]; then
  echo "No running dev server or tunnel found."
fi
