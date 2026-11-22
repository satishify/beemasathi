"""Classify a question into a typed plan before any retrieval happens."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import BANNED_PATTERNS, SECURITY_PATTERNS
from rag.policy_match import match_policy, normalize

INSURER_ALIASES: Dict[str, List[str]] = {
    "star_health": ["star health", "star"],
    "hdfc_ergo": ["hdfc ergo", "hdfc"],
    "icici_lombard": ["icici lombard", "icici"],
    "max_bupa": ["niva bupa", "max bupa", "bupa"],
    "care_health": ["care health", "care insurance"],
    "zurich_kotak": ["zurich kotak", "kotak"],
    "acko": ["acko"],
    "digit": ["go digit", "digit", "digital"],
    "aditya_birla": ["aditya birla"],
    "bajaj_allianz": ["bajaj allianz", "bajaj general", "bajaj"],
    "tata_aig": ["tata aig", "tata"],
    "cholamandalam": ["cholamandalam", "chola ms", "chola"],
}

CLAIM_RE = re.compile(
    r"\b(claim\s+(?:be\s+)?reject|rejection\s+risk|reject(?:ed|ion)?|"
    r"will\s+my\s+claim|should\s+i\s+file|claim\s+settle|pre-?auth|"
    r"cashless\s+approval)\b",
    re.I,
)
COMPARE_RE = re.compile(r"\b(compare|compared|versus|vs\.?|difference\s+between|better\s+than)\b", re.I)
GAP_RE = re.compile(
    r"\b(which\s+insurers?\s+do\s+you\s+(?:not|don'?t)\s+have|"
    r"what\s+(?:insurers?|policies?)\s+(?:do\s+you\s+)?(?:not|don'?t)\s+have|"
    r"missing|not\s+loaded|no\s+policy\s+loaded|knowledge\s+gap)\b",
    re.I,
)
KNOWN_RE = re.compile(
    r"\b(which\s+insurers?\s+do\s+you\s+have|what\s+policies?\s+do\s+you\s+have|"
    r"what\s+(?:do\s+you\s+know|can\s+you\s+help)|known\s+policies)\b",
    re.I,
)
BROCHURE_RE = re.compile(r"\b(brochure|prospectus|cis|summary\s+document)\b", re.I)
RECOMMEND_RE = re.compile(
    r"\b(?:suggest|recommend)\b.*\b(?:plan|policy|insurance|cover)\b|"
    r"\b(?:best|good|suitable)\b.*\b(?:family\s+|health\s+|insurance\s+)*(?:plan|policy)\b|"
    r"\bwhich\b.*\b(?:plan|policy)\b.*\b(?:should|buy|choose|for\s+my)\b",
    re.I,
)


@dataclass
class TaskStep:
    action: str
    target: Optional[str] = None


@dataclass
class TaskPlan:
    question: str
    intent: str
    insurers: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)
    steps: List[TaskStep] = field(default_factory=list)
    prefer_brochure: bool = False

    def as_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "insurers": self.insurers,
            "policies": self.policies,
            "prefer_brochure": self.prefer_brochure,
            "steps": [{"action": step.action, "target": step.target} for step in self.steps],
        }


def _normalize(text: str) -> str:
    return normalize(text)


class IntentRouter:
    def __init__(self, catalog: Optional[List[Dict]] = None):
        self.catalog = catalog or []

    def _catalog_policies(self) -> List[tuple]:
        return sorted(
            {
                (entry.get("insurer"), entry.get("policy_name"))
                for entry in self.catalog
                if entry.get("policy_name")
            }
        )

    def find_insurers(self, question: str) -> List[str]:
        text = f" {_normalize(question)} "
        found = []
        for key, aliases in INSURER_ALIASES.items():
            if any(f" {alias} " in text for alias in aliases):
                found.append(key)
        return found

    def find_policies(self, question: str, insurers: List[str]) -> List[str]:
        matched = match_policy(
            _normalize(question).split(),
            self._catalog_policies(),
            insurers[0] if len(insurers) == 1 else None,
        )
        return [matched] if matched else []

    def route(
        self,
        question: str,
        insurer: Optional[str] = None,
        policy_name: Optional[str] = None,
    ) -> TaskPlan:
        insurers = [insurer] if insurer else self.find_insurers(question)
        policies = [policy_name] if policy_name else self.find_policies(question, insurers)

        if any(re.search(pattern, question, re.I) for pattern in BANNED_PATTERNS):
            plan = TaskPlan(question, "safety", insurers, policies)
            return plan

        if any(re.search(pattern, question, re.I) for pattern in SECURITY_PATTERNS):
            return TaskPlan(question, "security", insurers, policies)

        if GAP_RE.search(question) or KNOWN_RE.search(question):
            return TaskPlan(
                question,
                "scope",
                insurers,
                policies,
                steps=[TaskStep("catalog_scope")],
            )

        if CLAIM_RE.search(question):
            return TaskPlan(
                question,
                "claim_risk",
                insurers,
                policies,
                steps=[TaskStep("predict_claim"), TaskStep("explain")],
            )

        if COMPARE_RE.search(question) or len(insurers) >= 2:
            steps = [TaskStep("retrieve", target=key) for key in insurers]
            steps.append(TaskStep("compare"))
            if not steps[:-1]:
                steps.insert(0, TaskStep("retrieve"))
            return TaskPlan(question, "comparison", insurers, policies, steps)

        if RECOMMEND_RE.search(question):
            return TaskPlan(
                question,
                "recommend",
                insurers,
                policies,
                steps=[TaskStep("catalog_wordings"), TaskStep("explain_limits")],
            )

        prefer_brochure = bool(BROCHURE_RE.search(question))
        return TaskPlan(
            question,
            "coverage",
            insurers,
            policies,
            steps=[
                TaskStep("retrieve"),
                TaskStep("resolve_authority"),
                TaskStep("generate"),
                TaskStep("verify"),
            ],
            prefer_brochure=prefer_brochure,
        )
