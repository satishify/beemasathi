#!/usr/bin/env bash
# BeemaSathi launcher for macOS and Linux.
# The real logic lives in launcher.py so Windows shares the same behaviour.
#
# Usage:
#   ./beemasathi.sh            interactive menu
#   ./beemasathi.sh start      start in the background
#   ./beemasathi.sh stop       stop the app
#   ./beemasathi.sh status     show backend and app state
#   ./beemasathi.sh config     edit the hosted model settings in .env
#   ./beemasathi.sh check      report the environment, install nothing
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
  [[ -n "$candidate" ]] || continue
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" "$ROOT/launcher.py" "$@"
  fi
done

echo "Python 3.11 or newer is required. Install it, then run this script again." >&2
exit 1
