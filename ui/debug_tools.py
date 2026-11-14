"""Local runtime diagnostics for the opt-in Streamlit debug UI."""

import platform
import subprocess
from datetime import datetime
from typing import Any, Dict

import requests


OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"


def format_bytes(value: int | float | None) -> str:
    """Format a byte count for compact display."""
    if value is None:
        return "Unavailable"

    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "Unavailable"


def _system_memory_bytes() -> int | None:
    """Return physical system memory on macOS without an extra dependency."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _ollama_process_rss_bytes() -> int | None:
    """Return combined resident RAM for local processes containing 'ollama'."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "rss=,command="],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    total_kb = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or "ollama" not in parts[1].lower():
            continue
        try:
            total_kb += int(parts[0])
        except ValueError:
            continue
    return total_kb * 1024


def get_ollama_runtime(configured_model: str) -> Dict[str, Any]:
    """Collect local Ollama model and memory diagnostics."""
    runtime: Dict[str, Any] = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configured_model": configured_model,
        "service": "unavailable",
        "active_models": [],
        "ollama_process_rss_bytes": _ollama_process_rss_bytes(),
        "system_memory_bytes": _system_memory_bytes(),
        "platform": f"{platform.system()} {platform.machine()}",
        "error": None,
    }

    try:
        response = requests.get(OLLAMA_PS_URL, timeout=1.5)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        runtime["error"] = str(exc)
        return runtime

    runtime["service"] = "connected"
    runtime["active_models"] = [
        {
            "name": model.get("name") or model.get("model") or "Unknown",
            "loaded_size_bytes": model.get("size"),
            "vram_bytes": model.get("size_vram"),
            "cpu_allocation_bytes": max(
                0,
                (model.get("size") or 0) - (model.get("size_vram") or 0),
            ),
            "expires_at": model.get("expires_at"),
            "details": model.get("details") or {},
        }
        for model in payload.get("models", [])
    ]
    return runtime
