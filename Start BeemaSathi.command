#!/bin/bash
# Double-click entry point for Finder: runs the launcher in Terminal.
cd "$(dirname "$0")" || exit 1
exec ./beemasathi.sh
