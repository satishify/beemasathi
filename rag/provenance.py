"""Decide which stored chunks may be quoted.

A chunk is allowed only when its source file is one of:
  * a file recorded in the ingest ledger (data/chunks/ingest_index.json), or
  * an official regulation saved under data/irdai/.

Orphan chunks from demo fixtures or earlier prototype runs have no ledger
entry and no file on disk, so retrieval drops them even though they are still
present in the vector store.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LEDGER_PATH = BASE_DIR / "data" / "chunks" / "ingest_index.json"
REGULATION_DIR = BASE_DIR / "data" / "irdai"


@lru_cache(maxsize=1)
def allowed_source_files() -> FrozenSet[str]:
    allowed = set()
    try:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ledger = {}
    for path in (ledger.get("files") or {}):
        allowed.add(Path(path).name)
    if REGULATION_DIR.is_dir():
        allowed.update(path.name for path in REGULATION_DIR.glob("*.pdf"))
    return frozenset(allowed)


def is_allowed(metadata: Optional[Dict]) -> bool:
    metadata = metadata or {}
    name = metadata.get("source_file")
    if not name:
        return False
    return Path(name).name in allowed_source_files()
