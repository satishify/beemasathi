#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/venv/bin/python"
STREAMLIT="$ROOT_DIR/venv/bin/streamlit"
RUN_DIR="${TMPDIR:-/tmp}/beemasathi-${UID}"
APP_PID_FILE="$RUN_DIR/streamlit.pid"
OLLAMA_PID_FILE="$RUN_DIR/ollama.pid"
APP_LOG="$RUN_DIR/streamlit.log"
OLLAMA_LOG="$RUN_DIR/ollama.log"
APP_URL="http://127.0.0.1:8501"

mkdir -p "$RUN_DIR"

if [[ ! -x "$PYTHON" || ! -x "$STREAMLIT" ]]; then
  echo "Virtual environment is missing. Create venv and install requirements first."
  exit 1
fi

cd "$ROOT_DIR"
MODEL="$("$PYTHON" -c 'from config import LLM_LOCAL; print(LLM_LOCAL)')"
API_MODE="$("$PYTHON" -c 'from config import LLM_API_BASE, LLM_API_KEY; print(1 if LLM_API_BASE and LLM_API_KEY else 0)')"
API_MODEL="$("$PYTHON" -c 'from config import LLM_API_MODEL; print(LLM_API_MODEL)')"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-30}"
  for ((i = 0; i < attempts; i++)); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

start() {
  if [[ "$API_MODE" != "1" ]]; then
    command -v ollama >/dev/null 2>&1 || {
      echo "Ollama is not installed or is not on PATH."
      exit 1
    }

    if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      nohup ollama serve >"$OLLAMA_LOG" 2>&1 &
      echo $! >"$OLLAMA_PID_FILE"
      if ! wait_for_url "http://127.0.0.1:11434/api/tags"; then
        echo "Ollama failed to start. See $OLLAMA_LOG"
        exit 1
      fi
    fi

    if ! ollama show "$MODEL" >/dev/null 2>&1; then
      echo "Model $MODEL is not installed. Run: ollama pull $MODEL"
      exit 1
    fi

    # Load the configured model into memory and keep it warm.
    curl -fsS "http://127.0.0.1:11434/api/generate" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":\"10m\"}" \
      >/dev/null
  fi

  if is_running "$APP_PID_FILE"; then
    echo "BeemaSathi is already running at $APP_URL"
    return
  fi

  BEEMASATHI_DEBUG="${BEEMASATHI_DEBUG:-0}" \
    nohup "$STREAMLIT" run "$ROOT_DIR/ui/app.py" \
      --server.address=127.0.0.1 \
      --server.port=8501 \
      --server.headless=true \
      >"$APP_LOG" 2>&1 &
  echo $! >"$APP_PID_FILE"

  if wait_for_url "$APP_URL" 20; then
    echo "BeemaSathi started: $APP_URL"
    if [[ "$API_MODE" == "1" ]]; then
      echo "Model backend: hosted API ($API_MODEL)"
    else
      echo "Model loaded: $MODEL"
    fi
  else
    echo "Streamlit failed to start. See $APP_LOG"
    exit 1
  fi
}

stop() {
  if is_running "$APP_PID_FILE"; then
    kill "$(cat "$APP_PID_FILE")"
    echo "BeemaSathi stopped."
  else
    echo "BeemaSathi is not running."
  fi
  rm -f "$APP_PID_FILE"

  if [[ "$API_MODE" != "1" ]] && curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    ollama stop "$MODEL" >/dev/null 2>&1 || true
    echo "Model unloaded: $MODEL"
  fi

  # Stop Ollama only when this script started its server.
  if is_running "$OLLAMA_PID_FILE"; then
    kill "$(cat "$OLLAMA_PID_FILE")"
    echo "Ollama server stopped."
  fi
  rm -f "$OLLAMA_PID_FILE"
}

status() {
  is_running "$APP_PID_FILE" && echo "App: running at $APP_URL" || echo "App: stopped"
  if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    echo "Ollama: running"
    ollama ps
  else
    echo "Ollama: stopped"
  fi
}

index() {
  "$PYTHON" "$ROOT_DIR/rag/sync_knowledge.py" "$@"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  index)
    shift
    index "$@"
    ;;
  *) echo "Usage: $0 {start|stop|restart|status|index}"; exit 1 ;;
esac
