"""Evidence checks that run after generation."""

import re
from typing import Dict, List

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|per\s?cent|percent|months?|years?|days?|lakhs?|crores?)(?!\w)"
    r"|\b[A-Z]{2,}[A-Z0-9]{4,}\b"
)

FAMILY_LIMIT_RE = re.compile(
    r"\bfamily of\s+(\d+)\b"
    r"|\bup to\s+(\d+)\s+(?:family\s+)?members?\b"
    r"|\b(\d+)\s+(?:family\s+)?members?\b",
    re.I,
)


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-]+", " ", (text or "").lower())


def _canonical_fact(token: str) -> str:
    value = _normalize(token)
    match = re.fullmatch(r"(\d+(?:\.\d+)?) (years?|months?|days?)", value)
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("year"):
            number *= 12
            unit = "months"
        if unit.startswith("month"):
            unit = "months"
        if unit.startswith("day"):
            unit = "days"
        whole = int(number) if number == int(number) else number
        return f"{whole} {unit}"
    match = re.fullmatch(r"(\d+(?:\.\d+)?) ?(%|per cent|percent)", value)
    if match:
        return f"{match.group(1)}%"
    return value


def _looks_like_uin(token: str) -> bool:
    return sum(char.isdigit() for char in token) >= 2


def verify_answer(answer: str, chunks: List[Dict]) -> Dict:
    """Flag numbers and UINs in the answer that are absent from the clauses."""
    context = _normalize(" ".join(chunk.get("text") or "" for chunk in chunks))
    allowed = {_canonical_fact(match.group(0)) for match in NUMBER_RE.finditer(context)}
    unsupported = []
    for match in NUMBER_RE.finditer(answer or ""):
        token = match.group(0)
        if re.fullmatch(r"[A-Z0-9]+", token) and not _looks_like_uin(token):
            continue
        if _canonical_fact(token) not in allowed and _normalize(token) not in context:
            unsupported.append(token)
    unsupported = list(dict.fromkeys(unsupported))
    return {"verified": not unsupported, "unsupported": unsupported}


def _family_limit_supported(number: str, context: str) -> bool:
    num = re.escape(number)
    patterns = (
        rf"\bfamily of\s+{num}\b",
        rf"\bup to\s+{num}\b",
        rf"\b{num}\s+(?:family\s+)?members?\b",
    )
    return any(re.search(pattern, context) for pattern in patterns)


def strip_unbacked_family_limits(answer: str, chunks: List[Dict]) -> str:
    """Drop family-size claims that the retrieved clauses do not support.

    Models echo "family of 4" from the question as if the plan confirmed it.
    When the wording does not set a limit, say so instead.
    """
    context = _normalize(" ".join(chunk.get("text") or "" for chunk in chunks))
    sentences = re.split(r"(?<=[.!?])\s+", answer or "")
    changed = False
    for index, sentence in enumerate(sentences):
        for match in FAMILY_LIMIT_RE.finditer(sentence):
            number = next(group for group in match.groups() if group)
            if not _family_limit_supported(number, context):
                sentences[index] = "The wording does not state a family-size limit."
                changed = True
                break
    return " ".join(sentences) if changed else answer


def resolve_authority(chunks: List[Dict], prefer_brochure: bool = False) -> List[Dict]:
    """Drop brochure clauses when binding wording for the same policy is present."""
    if prefer_brochure:
        return chunks
    wording_policies = {
        (chunk.get("metadata") or {}).get("policy_name")
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("document_type") == "policy_wording"
    }
    kept = [
        chunk
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("document_type")
        in {"policy_wording", "endorsement"}
        or (chunk.get("metadata") or {}).get("policy_name") not in wording_policies
    ]
    return kept or chunks
