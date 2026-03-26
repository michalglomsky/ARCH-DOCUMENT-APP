#!/bin/bash
# ============================================================
# ARCH Document Extractor — startup script
#
# Starts:
#   1. VLM server   → http://127.0.0.1:8081  (loads Qwen2.5-VL)
#   2. App server   → http://localhost:8000   (browser UI)
#
# Usage:
#   ./start.sh                          # zero-shot model
#   ./start.sh --lora-adapter path/     # with LoRA adapter
#   ./start.sh --port-app 8001          # custom app port
#   ./start.sh --max-pages 4            # limit pages per inference
# ============================================================

set -e

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/finetune_qwen_vl_pytorch/.venv311/bin/python3"
VLM_SCRIPT="$SCRIPT_DIR/finetune_qwen_vl_qa/scripts/serve_vlm_qa.py"
APP_SCRIPT="$SCRIPT_DIR/app/server.py"
LOG_DIR="$SCRIPT_DIR/logs"

# ---- Defaults ----
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER=""
PORT_VLM=8081
PORT_APP=8000
MAX_PAGES=6

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --lora-adapter) LORA_ADAPTER="$2"; shift 2 ;;
    --model)        MODEL="$2";        shift 2 ;;
    --port-vlm)     PORT_VLM="$2";     shift 2 ;;
    --port-app)     PORT_APP="$2";     shift 2 ;;
    --max-pages)    MAX_PAGES="$2";    shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---- Check Python ----
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: venv Python not found at $VENV_PYTHON"
  echo "       Make sure the venv exists: finetune_qwen_vl_pytorch/.venv311/"
  exit 1
fi

# ---- Ensure required packages ----
echo "Checking required packages…"
"$VENV_PYTHON" -m pip install --quiet uvicorn fastapi httpx openpyxl 2>/dev/null || true

# ---- Logging ----
mkdir -p "$LOG_DIR"
VLM_LOG="$LOG_DIR/vlm_server.log"
APP_LOG="$LOG_DIR/app_server.log"

# ---- Cleanup on exit ----
VLM_PID=""
APP_PID=""

cleanup() {
  echo ""
  echo "Shutting down…"
  [[ -n "$APP_PID" ]] && kill "$APP_PID" 2>/dev/null && echo "  App server stopped."
  [[ -n "$VLM_PID" ]] && kill "$VLM_PID" 2>/dev/null && echo "  VLM server stopped."
  exit 0
}
trap cleanup INT TERM

# ---- Start VLM server ----
VLM_ARGS="--model $MODEL --port $PORT_VLM --max-pages $MAX_PAGES"
[[ -n "$LORA_ADAPTER" ]] && VLM_ARGS="$VLM_ARGS --lora-adapter $LORA_ADAPTER"

echo ""
echo "Starting VLM server on port $PORT_VLM…"
echo "  Model:   $MODEL"
[[ -n "$LORA_ADAPTER" ]] && echo "  Adapter: $LORA_ADAPTER"
echo "  Log:     $VLM_LOG"
echo ""

"$VENV_PYTHON" "$VLM_SCRIPT" $VLM_ARGS >"$VLM_LOG" 2>&1 &
VLM_PID=$!

# ---- Wait for VLM to be ready ----
echo -n "Waiting for VLM server"
TRIES=0
MAX_TRIES=120  # 2 min
while true; do
  sleep 2
  TRIES=$((TRIES + 1))

  # Check the process is still alive
  if ! kill -0 "$VLM_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: VLM server process died. Check log: $VLM_LOG"
    tail -20 "$VLM_LOG"
    exit 1
  fi

  # Check if it's responding
  if curl -sf "http://127.0.0.1:$PORT_VLM/health" >/dev/null 2>&1; then
    echo " ✓"
    break
  fi

  echo -n "."

  if [[ $TRIES -ge $MAX_TRIES ]]; then
    echo ""
    echo "ERROR: VLM server did not start within $((MAX_TRIES * 2))s. Check log: $VLM_LOG"
    tail -20 "$VLM_LOG"
    exit 1
  fi
done

# ---- Start App server ----
echo ""
echo "Starting App server on port $PORT_APP…"
echo "  Log: $APP_LOG"
echo ""

"$VENV_PYTHON" "$APP_SCRIPT" --port "$PORT_APP" >"$APP_LOG" 2>&1 &
APP_PID=$!

sleep 2
if ! kill -0 "$APP_PID" 2>/dev/null; then
  echo "ERROR: App server process died. Check log: $APP_LOG"
  tail -20 "$APP_LOG"
  cleanup
fi

# ---- Ready ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ARCH Document Extractor is running"
echo ""
echo "  Browser UI  →  http://localhost:$PORT_APP"
echo "  VLM API     →  http://127.0.0.1:$PORT_VLM"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Keep script alive — wait for Ctrl+C
wait
