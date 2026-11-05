#!/bin/bash
# Start / stop / tail the TouchDesigner vision pipeline as a detached process.
#
# Usage:
#   scripts/td-vision.sh start       # launch (viewer at http://localhost:7500)
#   scripts/td-vision.sh stop        # terminate
#   scripts/td-vision.sh status      # show pid + uptime + ports
#   scripts/td-vision.sh log         # tail the log
#   scripts/td-vision.sh restart

set -e

SESSION=tdvision
LOG=/tmp/td-vision.log
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv-td"
VIEWER_PORT=${VIEWER_PORT:-7500}
OSC_PORT=${OSC_PORT:-7400}
EXTRA_FLAGS=${EXTRA_FLAGS:-"--disable-face"}

if [ ! -x "$VENV/bin/python" ]; then
    echo "error: $VENV not found — run: python3 -m venv .venv-td && .venv-td/bin/pip install -r touchdesigner/requirements.txt" >&2
    exit 1
fi

cmd=${1:-status}

start() {
    if screen -ls | grep -q "\.$SESSION\b"; then
        echo "already running (session $SESSION)"
        exit 0
    fi
    : > "$LOG"
    screen -dmS "$SESSION" bash -c "
        cd '$REPO_ROOT'
        export MPLCONFIGDIR=/tmp/mpl
        exec '$VENV/bin/python' -u -m touchdesigner.main \
            --viewer $VIEWER_PORT --osc-port $OSC_PORT $EXTRA_FLAGS \
            >> '$LOG' 2>&1
    "
    sleep 2
    status
    echo ""
    echo "viewer: http://localhost:$VIEWER_PORT/"
    echo "osc:    127.0.0.1:$OSC_PORT"
    echo "log:    $LOG"
}

stop() {
    screen -S "$SESSION" -X quit 2>/dev/null || true
    pkill -f "touchdesigner.main" 2>/dev/null || true
    sleep 0.3
    echo "stopped"
}

status() {
    if screen -ls | grep -q "\.$SESSION\b"; then
        echo "running"
        ps -ao pid,etime,%cpu,%mem,command | grep -E "touchdesigner.main" | grep -v grep || true
    else
        echo "not running"
    fi
}

case "$cmd" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    log)     tail -f "$LOG" ;;
    *) echo "usage: $0 {start|stop|restart|status|log}" >&2; exit 2 ;;
esac
