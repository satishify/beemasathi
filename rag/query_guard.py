"""Typed query-time guard that runs before routing, retrieval, and the model.

Three deterministic layers, cheapest first:

1. the configured safety and security patterns, matched on the raw question;
2. the same patterns matched again after normalisation (NFKC, zero-width
   removal, homoglyph folding, collapsed spaced letters, common leetspeak),
   which catches obfuscated instructions the raw patterns miss;
3. narrow intent patterns that need more context than a keyword: advice on
   concealing a condition, and requests for another customer's data.

The guard never answers, never rewrites the question, and never calls a model.
It returns a typed decision so the orchestrator can refuse with the canonical
text and record why, keeping a refused question away from retrieval entirely.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import (
    BANNED_PATTERNS,
    SAFETY_REFUSAL,
    SECURITY_PATTERNS,
    SECURITY_REFUSAL,
)

_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2028\u2029\u2060\ufeff\u00ad]")
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")
_HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
        "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u0455": "s",
        "\u0501": "d", "\u0261": "g", "\u217c": "l", "\u03b1": "a", "\u03bf": "o",
        "\u03c1": "p", "\u03c4": "t", "\u03ba": "k", "\u03bd": "v",
    }
)
_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

# Advice-seeking concealment. Deliberately framed as "can I omit / how can I
# hide" so a legitimate "what happens if I forget to disclose?" is not refused.
FRAUD_DISCLOSURE_PATTERNS = (
    r"\b(?:can|could|should|may)\s+i\s+(?:just\s+)?"
    r"(?:omit|skip|leave\s+out|hide|conceal|"
    r"not\s+(?:declare|disclose|mention|report))\b"
    r"[^.?!\n]{0,60}\b(?:condition|disease|diabetes|history|ped|"
    r"pre[\s-]?existing|treatment|medication|surgery|hospitali[sz]ation)\b",
    r"\bhow\s+(?:can|do|to)\s+(?:i\s+)?"
    r"(?:hide|conceal|omit|leave\s+out|"
    r"avoid\s+(?:declaring|disclosing|mentioning|reporting))\b"
    r"[^.?!\n]{0,60}\b(?:condition|disease|diabetes|history|ped|"
    r"pre[\s-]?existing|treatment|medication)\b",
)

PII_PATTERNS = (
    r"\b(?:another|other)\s+(?:customer|customer's|patient|policyholder|insured)"
    r"(?:'s)?\s+(?:data|details|policy|claim|records?|number|phone|aadhaar|pan)\b",
    r"\bsomeone\s+else'?s\s+(?:policy|claim|data|details|number|records?)\b",
)

# Scanned in documents, not in the user's question. A scraped PDF is not
# trusted input: these phrases mark a chunk that tries to steer the model.
DOCUMENT_INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+"
    r"(?:instructions|rules|prompts?)",
    r"(?:reveal|print|show|repeat)\s+(?:your|the)\s+"
    r"(?:system\s+prompt|instructions|rules)",
    r"you\s+are\s+now\s+(?:an?\s+)?(?:admin|insurer|system|developer)",
    r"approve\s+(?:every|all)\s+claims?",
    r"<\s*/?\s*(?:system|assistant|developer)\s*>",
    r"\bbegin\s+(?:system|instructions?)\b",
)

ALLOW = "allow"
REFUSE = "refuse"


@dataclass(frozen=True)
class GuardDecision:
    action: str = ALLOW
    category: Optional[str] = None
    reason: str = ""
    matched: Optional[str] = None
    obfuscated: bool = False
    risk_score: float = 0.0
    normalized_query: str = ""

    @property
    def plan_intent(self) -> str:
        return "safety" if self.category == "safety" else "security"

    def as_dict(self) -> Dict:
        return {
            "action": self.action,
            "category": self.category,
            "reason": self.reason,
            "matched": self.matched,
            "obfuscated": self.obfuscated,
            "risk_score": self.risk_score,
        }

    def refusal_result(self, model: str) -> Dict:
        safe_category = self.category == "safety"
        return {
            "answer": SAFETY_REFUSAL if safe_category else SECURITY_REFUSAL,
            "safe": False,
            "sources": [],
            "rejection_risk": "high" if safe_category else None,
            "model_metrics": {
                "backend": "guard",
                "model": model,
                "reason": self.reason or "blocked by the query guard",
            },
        }


def _join_spaced(match: "re.Match[str]") -> str:
    """Join one spaced-out word, keeping word gaps the user left between runs."""
    parts = [part for part in re.split(r"\s{2,}", match.group(0)) if part]
    return " ".join(re.sub(r"\s+", "", part) for part in parts)


def normalize_for_scan(text: str) -> str:
    """Fold obfuscation that the raw patterns would otherwise miss.

    Case is folded here only for matching; the original question is what the
    model receives, so wording and names are never changed.
    """
    value = unicodedata.normalize("NFKC", text or "")
    value = _ZERO_WIDTH_RE.sub("", value).lower()
    value = value.translate(_HOMOGLYPHS)
    value = _SPACED_LETTERS_RE.sub(_join_spaced, value)
    value = value.translate(_LEET)
    return re.sub(r"\s+", " ", value).strip()


def _first_match(
    layers: Sequence[tuple], text: str
) -> tuple:
    for category, patterns, label in layers:
        for index, pattern in enumerate(patterns):
            if re.search(pattern, text, re.IGNORECASE):
                return category, label, index
    return None, None, None


def scan_query(question: str) -> GuardDecision:
    """Return a refusal decision or an allow decision with zero side effects."""
    normalized = normalize_for_scan(question)

    layers = (
        ("safety", tuple(BANNED_PATTERNS), "banned-pattern"),
        ("safety", FRAUD_DISCLOSURE_PATTERNS, "concealment-advice"),
        ("security", tuple(SECURITY_PATTERNS), "security-pattern"),
        ("security", PII_PATTERNS, "third-party-data"),
    )

    category, label, index = _first_match(layers, question)
    obfuscated = False
    if category is None and normalized and normalized != (question or "").strip().lower():
        category, label, index = _first_match(layers, normalized)
        obfuscated = category is not None

    if category is None:
        return GuardDecision(action=ALLOW, normalized_query=normalized)

    reason = f"Blocked by query guard: {label} #{index}."
    return GuardDecision(
        action=REFUSE,
        category=category,
        reason=reason,
        matched=label,
        obfuscated=obfuscated,
        risk_score=0.9 if obfuscated else 1.0,
        normalized_query=normalized,
    )


def scan_document(text: str) -> List[str]:
    """Return the document-injection phrases found in supplied text.

    Used when chunks are ingested or updated so a poisoned passage can be
    flagged and dropped instead of being treated as evidence.
    """
    found: List[str] = []
    for pattern in DOCUMENT_INJECTION_PATTERNS:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            found.append(match.group(0)[:80])
    return found


def document_flags_json(text: str) -> str:
    """Metadata-safe representation: Chroma does not store lists."""
    flags = scan_document(text)
    return json.dumps(flags) if flags else ""
