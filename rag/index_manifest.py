"""Index manifest: the model and chunk contract an index was built with.

A vector index is only searchable by the model that built it and the chunk
text that produced it. The manifest records both so a later change is visible
instead of silently returning wrong neighbours.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(
    path: Path,
    *,
    model: str,
    dim: Optional[int],
    chunks: int,
    params: Dict,
    extra: Optional[Dict] = None,
) -> Dict:
    payload = {
        "generated_at": now(),
        "model": model,
        "dim": dim,
        "chunks": chunks,
        "params": params,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_manifest(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_manifest(
    path: Path, *, model: str, dim: Optional[int] = None
) -> List[str]:
    """Return problems; an absent manifest is reported, never raised."""
    payload = read_manifest(path)
    if not payload:
        return ["index manifest is missing; the index cannot be traced to a model"]
    problems = []
    if payload.get("model") != model:
        problems.append(
            f"index was built with {payload.get('model')!r}, "
            f"queries use {model!r}"
        )
    if dim is not None and payload.get("dim") not in (None, dim):
        problems.append(
            f"index dimension {payload.get('dim')} does not match query dimension {dim}"
        )
    return problems
