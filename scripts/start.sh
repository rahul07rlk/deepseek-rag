#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate

echo "[1/2] Starting file watcher..."
python -m src.watcher &
WATCHER_PID=$!
trap 'kill $WATCHER_PID 2>/dev/null || true' EXIT

echo "[2/2] Starting proxy server..."
echo "  Proxy URL : http://localhost:8000"
echo "  Point your IDE to: http://localhost:8000/v1"
python -m src.proxy_server
