"""Extract, chunk, and index a local insurance PDF."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import infer_chunk_type
from rag.embedder import PolicyVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

SECTION_HINTS = (
    "what's in it for you",
    "what’s in it for you",
    "coverage details",
    "optional covers",
    "waiting period",
    "what's not covered",
    "what’s not covered",
    "exclusions",
    "for cashless claims",
    "how to make a claim",
    "table of benefits",
    "terms of cancellation",
)

NOISE_LINES = {
    "back to contents",
    "coverage details",
    "optional covers",
    "other base covers",
    "exclusions & cancellation terms",
    "how to make a claim?",
    "waiting period",
    "table of benefits",
}

NAVIGATION_FRAGMENTS = (
    "coverage details",
    "optional covers",
    "base covers",
    "exclusions",
    "how to make",
    "waiting",
    "table of",
)


def _clean_page_text(text: str) -> str:
    """Remove extraction artifacts and repeated brochure navigation labels."""
    text = text.replace("\x00", " ")
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line.lower() in NOISE_LINES:
            continue
        if sum(
            fragment in line.lower()
            for fragment in NAVIGATION_FRAGMENTS
        ) >= 2:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract non-empty page text with one-based page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            text = _clean_page_text(page.extract_text() or "")
            if text:
                pages.append((page_number, text))

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or image-only."
        )
    return pages


def _section_for_page(text: str, page_number: int) -> str:
    lowered = text.lower()
    for hint in SECTION_HINTS:
        if hint in lowered:
            return f"{hint.title()} (page {page_number})"
    return f"Page {page_number}"


def _split_page(text: str, max_chars: int = 1600) -> List[str]:
    """Split page text into bounded, overlapping line groups."""
    lines = text.splitlines()
    chunks = []
    current = []
    current_size = 0

    for line in lines:
        projected_size = current_size + len(line) + 1
        if current and projected_size > max_chars:
            chunks.append("\n".join(current))
            overlap = current[-2:]
            current = overlap + [line]
            current_size = sum(len(item) + 1 for item in current)
        else:
            current.append(line)
            current_size = projected_size

    if current:
        chunks.append("\n".join(current))
    return [chunk for chunk in chunks if len(chunk) >= 80]


def build_pdf_chunks(
    pages: List[Tuple[int, str]],
    insurer: str,
    policy_name: str,
    source_file: str,
    document_type: str,
) -> List[Dict]:
    """Create searchable chunks while preserving page-level citations."""
    source = f"{insurer}_{policy_name}_{document_type}"
    chunks = []
    for page_number, page_text in pages:
        section = _section_for_page(page_text, page_number)
        for page_chunk in _split_page(page_text):
            chunks.append(
                {
                    "text": page_chunk,
                    "metadata": {
                        "insurer": insurer,
                        "policy_name": policy_name,
                        "section": section,
                        "page": page_number,
                        "chunk_type": infer_chunk_type(page_chunk),
                        "source": source,
                        "source_file": source_file,
                        "document_type": document_type,
                    },
                }
            )
    return chunks


def ingest_pdf(
    pdf_path: Path,
    insurer: str,
    policy_name: str,
    document_type: str = "policy_wording",
    store: PolicyVectorStore = None,
    extra_metadata: Dict = None,
) -> Dict:
    """Run extraction, chunk persistence, and ChromaDB replacement."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_pdf_pages(pdf_path)
    chunks = build_pdf_chunks(
        pages=pages,
        insurer=insurer,
        policy_name=policy_name,
        source_file=pdf_path.name,
        document_type=document_type,
    )
    if not chunks:
        raise ValueError("Text was extracted, but no usable chunks were produced.")

    extra = {key: value for key, value in (extra_metadata or {}).items() if value}
    if extra:
        for chunk in chunks:
            chunk["metadata"].update(extra)

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"{insurer}_{policy_name}_{document_type}"
    parsed_path = PARSED_DIR / f"{output_stem}.txt"
    chunks_path = CHUNKS_DIR / f"{output_stem}.json"

    parsed_path.write_text(
        "\n\n".join(
            f"--- Page {page_number} ---\n{text}"
            for page_number, text in pages
        ),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    store = store or PolicyVectorStore()
    store.replace_source_chunks(chunks)
    return {
        "pdf": str(pdf_path),
        "pages": len(pages),
        "chunks": len(chunks),
        "parsed_path": str(parsed_path),
        "chunks_path": str(chunks_path),
        "database_chunks": store.get_stats()["total_chunks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and index a local insurance PDF."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--insurer", required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument(
        "--document-type",
        default="policy_wording",
        choices=["policy_wording", "brochure", "regulation"],
    )
    args = parser.parse_args()

    result = ingest_pdf(
        pdf_path=args.pdf,
        insurer=args.insurer,
        policy_name=args.policy_name,
        document_type=args.document_type,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
