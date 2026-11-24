"""Parse policy text (from HTML or PDF) into semantic chunks."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from rag.query_guard import scan_document

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 2
CHARS_PER_TOKEN = 4
TABLE_PIPE_LINE = re.compile(r"^\s*[^|\n]*\|[^|\n]*")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def stable_chunk_id(source: str, section: str, page, text: str) -> str:
    """Content-and-location hash: equal chunks collapse, different ones never do."""
    digest = hashlib.sha1()
    digest.update(f"{source}|{section}|{page}|{text}".encode("utf-8"))
    return digest.hexdigest()[:20]


def looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    piped = sum(1 for line in lines if TABLE_PIPE_LINE.match(line))
    return piped >= max(2, len(lines) // 2)


def render_table(table: List[List]) -> str:
    """Render an extracted table as pipe-separated rows the model can read."""
    rows = []
    for row in table or []:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def enrich_chunks(chunks: List[Dict], source: str) -> List[Dict]:
    """Attach schema-v2 metadata. Text is never rewritten."""
    for chunk in chunks:
        meta = chunk["metadata"]
        page = meta.get("page", "unknown")
        meta.setdefault("page_start", page)
        meta.setdefault("page_end", page)
        meta["section_path"] = meta.get("section", "")
        meta.setdefault(
            "element_type",
            "table" if looks_like_table(chunk["text"]) else "text",
        )
        meta["schema_version"] = SCHEMA_VERSION
        meta["token_estimate"] = estimate_tokens(chunk["text"])
        flags = scan_document(chunk["text"])
        meta["injection_flags"] = json.dumps(flags) if flags else ""
        meta["chunk_id"] = stable_chunk_id(
            source, meta.get("section", ""), page, chunk["text"]
        )
    return chunks


def _split_bounded(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Cut long text at line/space boundaries, carrying a small overlap."""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    total = len(text)
    while start < total:
        end = min(total, start + max_chars)
        if end < total:
            floor = start + max_chars // 2
            cut = text.rfind("\n", floor, end)
            if cut == -1:
                cut = text.rfind(" ", floor, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= total:
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def parse_policy_to_chunks(
    insurer: str,
    policy_name: str,
    text: str,
    output_path: Path = None
) -> List[Dict]:
    """
    Split policy text into semantic chunks.
    Works with both HTML-extracted text and PDF-extracted text.
    """
    chunks = []
    
    # Clean the text first
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
    
    # Strategy 1: Split by numbered sections (common in policy docs)
    # Pattern: "1. Room Rent" or "Section 1:" or "1) Definitions"
    section_pattern = r'(?:\n|^)(?:\d+\.|\d+\)|Section\s+\d+[.:]?)\s*([A-Z][^\n]{0,100})'
    sections = re.split(section_pattern, text)
    
    # If no numbered sections found, try header-based splitting
    if len(sections) < 3:
        # Look for ALL CAPS lines or Title Case lines as headers
        sections = re.split(r'\n([A-Z][A-Z\s]{3,50}[A-Z]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\n', text)
    
    # If still no good split, use paragraph chunks
    if len(sections) < 3:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        sections = [''] + paragraphs  # Pad to match pattern
    
    # Process sections
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip() if i < len(sections) else "General"
        section_body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        
        if len(section_body) < 30:
            continue
        
        # Further split large sections by sub-clauses, then bound each piece
        sub_sections = re.split(r'\n(?=(?:\d+\.|[•\-]\s|\([a-z]\)\s))', section_body)
        
        for sub in sub_sections:
            sub = sub.strip()
            if len(sub) < 30:
                continue

            for piece in _split_bounded(sub, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                if len(piece) < 30:
                    continue

                # Detect chunk type
                chunk_type = classify_chunk_type(piece, section_title)

                chunk = {
                    "text": piece,
                    "metadata": {
                        "insurer": insurer,
                        "policy_name": policy_name,
                        "section": section_title[:100],
                        "chunk_type": chunk_type,
                        "source": f"{insurer}_{policy_name}",
                    }
                }
                chunks.append(chunk)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
        print(f"Created {len(chunks)} chunks → {output_path}")
    
    return chunks


def classify_chunk_type(text: str, section_title: str) -> str:
    """Classify what type of insurance clause this chunk contains."""
    combined = (text + " " + section_title).lower()
    
    patterns = {
        "room_rent": ["room rent", "rent cap", "1% of sum", "room category", "ac room", "single room", "shared room"],
        "waiting_period": ["pre-existing", "ped", "waiting period", "cooling off", "48 months", "36 months"],
        "co_pay": ["co-pay", "copay", "co payment", "deductible", "20% of claim"],
        "exclusion": ["exclusion", "not covered", "will not pay", "not payable", "we shall not"],
        "cashless": ["cashless", "network hospital", "tpa", "ppn", "pre-authorization"],
        "claim_process": ["claim", "intimation", "rejection", "claim form", "documents required", "notification"],
        "coverage": ["sum insured", "coverage amount", "policy limit", "maximum liability"],
        "renewal": ["renewal", "lapse", "grace period", "portability", "migration"],
        "premium": ["premium", "loading", "risk loading", "premium payment"],
        "maternity": ["maternity", "new born", "delivery", "pregnancy"],
        "day_care": ["day care", "daycare", "24 hours", "less than 24 hours"],
    }
    
    for ctype, keywords in patterns.items():
        if any(k in combined for k in keywords):
            return ctype
    
    return "general"


def infer_chunk_type(text: str) -> str:
    """Backward-compatible classifier used by page-aware PDF ingestion."""
    return classify_chunk_type(text, "")


def process_all_raw_policies():
    """Find all .txt and .pdf files in data/raw_policies/ and chunk them."""
    raw_dir = BASE_DIR / "data" / "raw_policies"
    
    all_chunks = []
    
    # Process .txt files (from HTML scraping)
    for txt_file in raw_dir.rglob("*.txt"):
        insurer = txt_file.parent.name
        policy_name = txt_file.stem
        
        print(f"Processing TXT: {insurer}/{policy_name}")
        text = txt_file.read_text(encoding="utf-8")
        chunks = parse_policy_to_chunks(insurer, policy_name, text)
        all_chunks.extend(enrich_chunks(chunks, f"{insurer}_{policy_name}"))
    
    # Process .pdf files (convert to text first)
    try:
        import pdfplumber
        for pdf_file in raw_dir.rglob("*.pdf"):
            insurer = pdf_file.parent.name
            policy_name = pdf_file.stem
            
            print(f"Processing PDF: {insurer}/{policy_name}")
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            
            chunks = parse_policy_to_chunks(insurer, policy_name, text)
            all_chunks.extend(enrich_chunks(chunks, f"{insurer}_{policy_name}"))
    except ImportError:
        print("pdfplumber not installed. Run: pip install pdfplumber")
    
    # Save unified chunks
    if all_chunks:
        unified_path = CHUNKS_DIR / "all_policies.json"
        unified_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False))
        print(f"\n✅ Unified {len(all_chunks)} chunks from all sources → {unified_path}")
    
    return all_chunks


if __name__ == "__main__":
    # Option 1: Process a single text
    sample = """
1. Room Rent Cap
Room rent is capped at 1% of sum insured per day. 
If you choose a room costing more than this limit, 
the insurer will proportionately reduce all other charges.

2. Pre-existing Disease Waiting Period
A waiting period of 36 months applies to diabetes, hypertension, 
and any condition treated in the 48 months before policy inception.

3. Co-Payment Clause
A co-payment of 20% applies to all claims for insured persons above 60 years.
"""
    # parse_policy_to_chunks("star_health", "comprehensive", sample, CHUNKS_DIR / "test.json")
    
    # Option 2: Process all downloaded files
    process_all_raw_policies()