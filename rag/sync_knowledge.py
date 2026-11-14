"""Index new or changed policy files from the data folders."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import KNOWLEDGE_SOURCES
from rag.chunker import parse_policy_to_chunks
from rag.embedder import PolicyVectorStore
from rag.ingest_pdf import ingest_pdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = (
    PROJECT_ROOT / "data" / "raw_pdfs",
    PROJECT_ROOT / "data" / "raw_policies",
)
INDEX_PATH = PROJECT_ROOT / "data" / "chunks" / "ingest_index.json"
DOC_TYPES = {"policy_wording", "brochure", "regulation"}
SKIP_NAMES = {"manifest.json", "ingest_index.json"}
INSURER_ALIASES = {
    "bajaj": "bajaj_allianz",
    "bajaj_allianz": "bajaj_allianz",
    "bajaj_alliaz": "bajaj_allianz",
    "baja_allianz": "bajaj_allianz",
    "allianz": "bajaj_allianz",
    "zkgi": "zurich_kotak",
    "zurich": "zurich_kotak",
    "zurich_kotak": "zurich_kotak",
    "kotak": "zurich_kotak",
    "niva": "max_bupa",
    "niva_bupa": "max_bupa",
    "max": "max_bupa",
    "irda": "irdai",
    "irdai": "irdai",
}
POLICY_NOISE = (
    "brochure",
    "print",
    "policy_wording",
    "wording",
    "pw",
    "revised",
    "v13",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def match_insurer(value: str) -> Optional[str]:
    slug = _slug(value)
    if not slug:
        return None
    if slug in KNOWLEDGE_SOURCES:
        return slug
    if slug in INSURER_ALIASES:
        return INSURER_ALIASES[slug]
    compact = slug.replace("_", "")
    for key, info in KNOWLEDGE_SOURCES.items():
        if compact == key.replace("_", "") or _slug(info["name"]) == slug:
            return key
        parts = [part for part in key.split("_") if len(part) > 2]
        if parts and all(part in slug for part in parts):
            return key
    for alias, key in INSURER_ALIASES.items():
        if alias in slug:
            return key
    return None


def _policy_from_stem(stem: str, insurer: str) -> str:
    policy = _slug(stem)
    drop_tokens = {insurer, *INSURER_ALIASES.keys()}
    for token in sorted(drop_tokens, key=len, reverse=True):
        token = _slug(token)
        if token and policy.startswith(f"{token}_"):
            policy = policy[len(token) + 1 :]
        policy = policy.replace(f"_{token}_", "_")
    for noise in POLICY_NOISE:
        policy = re.sub(rf"(^|_){noise}(_|$)", "_", policy)
    policy = re.sub(r"_+[0-9a-f]{8,}(_|$)", "_", policy)
    return _slug(policy) or "unknown"


def _load_index() -> Dict:
    if not INDEX_PATH.exists():
        return {"files": {}}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(index: Dict) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _sidecar_meta(path: Path) -> Dict:
    sidecar = path.with_name(path.stem + ".meta.json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def infer_metadata(path: Path) -> Optional[Dict]:
    """Read sidecar metadata, then infer insurer/policy from folders and filename."""
    meta = {
        "insurer": None,
        "policy_name": None,
        "document_type": "policy_wording",
        "line_of_business": None,
    }
    sidecar = _sidecar_meta(path)
    meta.update(
        {
            key: sidecar[key]
            for key in ("insurer", "policy_name", "document_type", "line_of_business")
            if sidecar.get(key)
        }
    )

    for part in path.parts:
        matched = match_insurer(part)
        if matched and not meta["insurer"]:
            meta["insurer"] = matched

    stem = path.stem
    lowered = stem.lower()
    meta["insurer"] = meta["insurer"] or match_insurer(stem)
    if "brochure" in lowered:
        meta["document_type"] = "brochure"
    elif any(token in lowered for token in ("wording", "_pw", "-pw", "policy wording")):
        meta["document_type"] = "policy_wording"
    if any(token in lowered for token in ("regulation", "guideline")) or match_insurer(stem) == "irdai":
        if "brochure" not in lowered and "wording" not in lowered:
            meta["document_type"] = "regulation"
            meta["insurer"] = meta["insurer"] or "irdai"

    tokens = [token for token in re.split(r"__+", stem) if token]
    if tokens and tokens[0] in {"health", "motor", "travel", "home", "commercial"}:
        meta["line_of_business"] = meta["line_of_business"] or tokens[0]
    if len(tokens) >= 3 and tokens[-1] in DOC_TYPES:
        meta["document_type"] = tokens[-1]
        meta["insurer"] = meta["insurer"] or match_insurer(tokens[0])
        meta["policy_name"] = meta["policy_name"] or _policy_from_stem(
            "_".join(tokens[1:-1]), meta["insurer"] or ""
        )
    elif len(tokens) >= 2 and match_insurer(tokens[0]):
        meta["insurer"] = meta["insurer"] or match_insurer(tokens[0])
        meta["policy_name"] = meta["policy_name"] or _policy_from_stem(
            "_".join(tokens[1:]), meta["insurer"] or ""
        )

    if not meta["policy_name"] and meta["insurer"]:
        meta["policy_name"] = _policy_from_stem(stem, meta["insurer"])

    if meta["document_type"] not in DOC_TYPES:
        meta["document_type"] = "policy_wording"
    if not meta["insurer"]:
        return None
    return meta


def discover_files() -> List[Path]:
    found = []
    for folder in SCAN_DIRS:
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.name in SKIP_NAMES or path.name.endswith(".meta.json"):
                continue
            if path.suffix.lower() in {".pdf", ".txt"}:
                found.append(path)
    return sorted(found)


def ingest_text(
    path: Path,
    insurer: str,
    policy_name: str,
    document_type: str,
    store: PolicyVectorStore = None,
    extra_metadata: Dict = None,
) -> Dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = parse_policy_to_chunks(insurer, policy_name, text)
    if not chunks:
        raise ValueError("No usable chunks were produced from the text file.")
    extra = {key: value for key, value in (extra_metadata or {}).items() if value}
    for chunk in chunks:
        chunk["metadata"].update(
            {
                "source": f"{insurer}_{policy_name}_{document_type}",
                "source_file": path.name,
                "document_type": document_type,
                "page": chunk["metadata"].get("page", "unknown"),
                **extra,
            }
        )
    store = store or PolicyVectorStore()
    store.replace_source_chunks(chunks)
    return {"chunks": len(chunks), "database_chunks": store.get_stats()["total_chunks"]}


def sync_knowledge(force: bool = False, dry_run: bool = False) -> Dict:
    index = _load_index()
    summary = {"indexed": [], "skipped": [], "unchanged": [], "errors": []}
    store = None if dry_run else PolicyVectorStore()
    files = discover_files()
    total = len(files)

    for position, path in enumerate(files, 1):
        rel = str(path.relative_to(PROJECT_ROOT))
        digest = _sha256(path)
        previous = index["files"].get(rel, {})
        if not force and previous.get("sha256") == digest:
            summary["unchanged"].append(rel)
            continue

        meta = infer_metadata(path)
        if not meta:
            summary["skipped"].append(
                {
                    "file": rel,
                    "reason": (
                        "Could not infer insurer. Put the file in "
                        "data/raw_policies/{insurer}/ or name it "
                        "{insurer}__{policy}__policy_wording.pdf, or add a "
                        f"{path.stem}.meta.json sidecar."
                    ),
                }
            )
            continue

        if dry_run:
            summary["indexed"].append(
                {"file": rel, "action": "would_index", **meta}
            )
            continue

        print(
            f"[{position}/{total}] {meta['insurer']}  {path.name}",
            flush=True,
        )
        try:
            extra = {
                key: meta[key]
                for key in ("line_of_business",)
                if meta.get(key)
            }
            if path.suffix.lower() == ".pdf":
                result = ingest_pdf(
                    pdf_path=path,
                    insurer=meta["insurer"],
                    policy_name=meta["policy_name"],
                    document_type=meta["document_type"],
                    store=store,
                    extra_metadata=extra,
                )
                chunk_count = result["chunks"]
            else:
                result = ingest_text(
                    path=path,
                    insurer=meta["insurer"],
                    policy_name=meta["policy_name"],
                    document_type=meta["document_type"],
                    store=store,
                    extra_metadata=extra,
                )
                chunk_count = result["chunks"]
        except Exception as exc:
            summary["errors"].append({"file": rel, "error": str(exc)})
            continue

        index["files"][rel] = {
            **meta,
            "sha256": digest,
            "chunks": chunk_count,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        summary["indexed"].append(
            {"file": rel, "chunks": chunk_count, **meta}
        )
        if not dry_run:
            _save_index(index)

    if not dry_run:
        _save_index(index)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk and embed new or changed PDFs/TXT files from "
            "data/raw_pdfs and data/raw_policies."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index every discovered file, even if it has not changed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without writing to ChromaDB.",
    )
    args = parser.parse_args()
    summary = sync_knowledge(force=args.force, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    if summary["errors"] or summary["skipped"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
