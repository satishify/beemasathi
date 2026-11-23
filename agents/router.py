"""Classify a question into a typed plan before any retrieval happens."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import BANNED_PATTERNS, SECURITY_PATTERNS
from rag.policy_match import PRODUCT_CLASS_NAMES, match_policy, normalize

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
TWO_WHEELER_RE = re.compile(
    r"\b(two[\s-]?wheelers?|2[\s-]?wheelers?|bikes?|scooters?|motorcycles?)\b", re.I
)
CAR_RE = re.compile(r"\b(cars?|four[\s-]?wheelers?|4[\s-]?wheelers?)\b", re.I)
MOTOR_RE = re.compile(r"\b(motor|vehicle|vehicles|auto|gcv|pcv)\b", re.I)
HEALTH_LINE_RE = re.compile(
    r"\b(health|mediclaim|hospitalisation|hospitalization|family\s+floater)\b", re.I
)
# A line question that asks what a clause says is coverage, not a catalog list.
CLAUSE_ASK_RE = re.compile(
    r"\b(cover(?:s|ed|age)?|exclud(?:e|es|ed|ing|ion|ions)|waiting|pre[\s-]?existing|"
    r"room\s+rent|co[\s-]?pay|copay|sub[\s-]?limit|deductible|claim(?:s|ed|ing)?|"
    r"renew(?:al)?|premium|network|cashless|day\s+care|maternity|ambulance|"
    r"no\s+claim\s+bonus|ncb|depreciation|add[\s-]?ons?|clause|section|"
    r"benefits?|terms|conditions)\b",
    re.I,
)
LIST_RE = re.compile(
    r"\b(list|show|name|display|enumerate|available|availability|"
    r"do\s+you\s+(?:have|hold|carry)|is\s+there|are\s+there|how\s+many|options?)\b",
    re.I,
)
COUNT_RE = re.compile(
    r"(?<![\w-])(\d{1,2})\s+(?:motor|health|car|policy|policies|plans?|insurance|options?)",
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
    line: Optional[str] = None
    vehicle: Optional[str] = None
    limit: int = 5

    def as_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "insurers": self.insurers,
            "policies": self.policies,
            "prefer_brochure": self.prefer_brochure,
            "line": self.line,
            "vehicle": self.vehicle,
            "limit": self.limit,
            "steps": [{"action": step.action, "target": step.target} for step in self.steps],
        }


def _normalize(text: str) -> str:
    return normalize(text)


def _specific_policies(policies: List[str]) -> List[str]:
    """Drop class-only names like "car" or "two wheeler"."""
    return [name for name in policies if _normalize(name) not in PRODUCT_CLASS_NAMES]


def detect_line(question: str):
    """Return (line, vehicle) named by the question, if any."""
    text = f" {_normalize(question)} "
    if TWO_WHEELER_RE.search(text):
        return "motor", "two_wheeler"
    if CAR_RE.search(text):
        return "motor", "car"
    if re.search(r"\b(commercial\s+vehicles?|gcv|pcv)\b", text):
        return "motor", "commercial"
    if MOTOR_RE.search(text):
        return "motor", None
    if HEALTH_LINE_RE.search(text):
        return "health", None
    return None, None


def _listing_limit(question: str) -> int:
    found = COUNT_RE.search(question)
    return min(max(int(found.group(1)), 1), 10) if found else 5


def _is_listing_question(
    question: str, line: Optional[str], insurers: List[str], policies: List[str]
) -> bool:
    if policies or len(insurers) >= 2:
        return False
    if CLAUSE_ASK_RE.search(question):
        return False
    if LIST_RE.search(question):
        return True
    if line and re.search(r"\b(what|which)\b", question, re.I):
        return True
    return bool(COMPARE_RE.search(question) and not insurers)


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
        policies = (
            [policy_name]
            if policy_name
            else _specific_policies(self.find_policies(question, insurers))
        )

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

        line, vehicle = detect_line(question)
        named = _specific_policies(policies)
        if _is_listing_question(question, line, insurers, named):
            return TaskPlan(
                question,
                "catalog_list",
                insurers,
                named,
                steps=[TaskStep("catalog_line", target=line)],
                line=line,
                vehicle=vehicle,
                limit=_listing_limit(question),
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
