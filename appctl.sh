#!/usr/bin/env bash
# Compatibility wrapper. Start, stop, restart and status are owned by the
# cross-platform launcher; this script keeps the old command names working.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

case "${1:-}" in
  index)
    shift
    if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
      exec "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/rag/sync_knowledge.py" "$@"
    fi
    echo "Virtual environment is missing. Run ./beemasathi.sh once, then retry." >&2
    exit 1
    ;;
  start|stop|restart|status)
    PYTHON_BIN="$ROOT_DIR/venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
      PYTHON_BIN="$(command -v python3 || command -v python || true)"
    fi
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "Python 3.11 or newer is required." >&2
      exit 1
    fi
    exec "$PYTHON_BIN" "$ROOT_DIR/launcher.py" "$1"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|index}"
    exit 1
    ;;
esac
