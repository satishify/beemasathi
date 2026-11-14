"""Parse policy text (from HTML or PDF) into semantic chunks."""

import json
import re
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


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
        
        # Further split large sections by sub-clauses
        sub_sections = re.split(r'\n(?=(?:\d+\.|[•\-]\s|\([a-z]\)\s))', section_body)
        
        for sub in sub_sections:
            sub = sub.strip()
            if len(sub) < 30:
                continue
            
            # Detect chunk type
            chunk_type = classify_chunk_type(sub, section_title)
            
            chunk = {
                "text": sub,
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
        all_chunks.extend(chunks)
    
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
            all_chunks.extend(chunks)
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