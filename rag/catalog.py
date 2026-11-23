"""Classify catalog entries by line of business.

Catalog entries carry no line label, but policy names and filenames are
descriptive enough to separate health from motor. This matched the
line_of_business field on 913 of 914 indexed entries, so listing and scope
answers can filter by line without re-reading chunk metadata.
"""

import re
from typing import Dict, List, Optional

MOTOR_RE = re.compile(
    r"\b(motor|vehicle|vehicles|car|cars|two wheelers?|tw|bike|bikes|"
    r"gcv|pcv|pvt|misc|own damage|od cover|liability|dealer|commercial|"
    r"battery protect|charging equipment|standalone od|package policy)\b"
)
HEALTH_RE = re.compile(
    r"\b(health|mediclaim|hospital|care|arogya|sanjeevani|critical illness|"
    r"personal accident|pa|wellness|top up|topup|medicare)\b"
)
TWO_WHEELER_RE = re.compile(
    r"\b(two wheelers?|2 wheelers?|tw|bikes?|scooters?|motorcycles?)\b"
)
COMMERCIAL_RE = re.compile(r"\b(commercial|gcv|pcv)\b")
CAR_RE = re.compile(r"\b(private car|cars?|pvt|four wheelers?|4 wheelers?)\b")

# Insurers that sell only health cover, so a generic policy name is health.
HEALTH_ONLY_INSURERS = frozenset(
    {"star_health", "care_health", "max_bupa", "aditya_birla"}
)


def _blob(entry: Dict) -> str:
    text = f"{entry.get('policy_name', '')} {entry.get('source_file', '')}".lower()
    return re.sub(r"[^a-z0-9]+", " ", text)


def entry_line(entry: Dict) -> Optional[str]:
    blob = _blob(entry)
    if MOTOR_RE.search(blob):
        return "motor"
    if HEALTH_RE.search(blob):
        return "health"
    if entry.get("insurer") in HEALTH_ONLY_INSURERS:
        return "health"
    return None


def entry_vehicle(entry: Dict) -> Optional[str]:
    """Vehicle class for a motor entry; None for health or unclear names."""
    blob = _blob(entry)
    if TWO_WHEELER_RE.search(blob):
        return "two_wheeler"
    if COMMERCIAL_RE.search(blob):
        return "commercial"
    if CAR_RE.search(blob):
        return "car"
    return None


def wording_entries(
    catalog: List[Dict],
    line: Optional[str] = None,
    vehicle: Optional[str] = None,
    insurers: Optional[List[str]] = None,
) -> List[Dict]:
    entries = [
        entry
        for entry in catalog
        if entry.get("document_type") == "policy_wording"
        and (not insurers or entry.get("insurer") in insurers)
    ]
    if line:
        matching = [entry for entry in entries if entry_line(entry) == line]
        entries = matching or entries
    if vehicle:
        matching = [entry for entry in entries if entry_vehicle(entry) == vehicle]
        entries = matching or entries
    return entries


def line_counts(catalog: List[Dict]) -> Dict[str, int]:
    """Policy-wording counts per line, omitting entries with no clear line."""
    counts: Dict[str, int] = {}
    for entry in catalog:
        if entry.get("document_type") != "policy_wording":
            continue
        line = entry_line(entry)
        if line:
            counts[line] = counts.get(line, 0) + 1
    return counts
