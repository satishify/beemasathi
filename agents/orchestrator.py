"""Plan, retrieve, verify, and answer with the existing RAG components."""

from pathlib import Path
import re
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.claim_predictor import ClaimPredictor
from agents.critic import resolve_authority, strip_unbacked_family_limits, verify_answer
from agents.router import COMPARE_RE, IntentRouter, TaskPlan, TaskStep
from config import INSURERS
from observability.trace import Trace
from rag.catalog import entry_line, entry_vehicle
from rag.generator import (
    BimaSaathiGenerator,
    asked_aspects,
    is_chat_meta_question,
    multi_part_question,
    table_requested,
)
from rag.query_guard import scan_query
from rag.retriever import PolicyRetriever

# A short "give me in table format" message carries no subject of its own; it
# asks for the previous exchange in a new shape. Anything naming coverage keeps
# its own retrieval.
FOLLOWUP_COVERAGE_RE = re.compile(
    r"\b(?:what|which|when|where|why|how|does|do|is|are|can|should|cover|covers|"
    r"coverage|waiting|exclusion|excluded|claim|policy|policies|plan|plans|insurer|"
    r"premium|sum insured|network|cashless|room rent|copay|co-?pay|ped|pre-existing)\b",
    re.I,
)

OFF_TOPIC_ANSWER = (
    "I can only answer questions about the Indian health and motor insurance "
    "policies loaded in BeemaSathi. I can't help with general-knowledge topics.\n\n"
    "Ask me about a plan, a clause, a claim, or a comparison, and I'll quote the "
    "wording I hold."
)

# Vocabulary that ties a question to insurance. Everything here comes from the
# policy domain, so a question with none of these words (and no plan, insurer,
# or chat follow-up) is answered with the off-topic refusal instead of letting
# the model fall back on general knowledge.
DOMAIN_TERMS = frozenset(
    """
    insurance insure insured insurer insurers policy policies policyholder proposer
    plan plans cover covers covered coverage claim claims claimant claimable premium
    premiums deductible excess waiting exclusion exclusions exclude sublimit sub limit
    copay co pay co payment room rent cashless reimbursement hospital hospitalisation
    hospitalization network sum insured irdai tpa pre existing ped maternity renewal
    renew ncb bonus addon add ons rider benefit benefits clause section terms
    conditions wording endorsement underwriting disease surgery treatment
    opd ipd daycare ambulance pharmacy diagnosis ailment diagnostic icu itcu transplant
    donor dialysis chemotherapy cataract knee hip nursing expenses medical mediclaim
    health motor vehicle vehicles car cars bike bikes two wheeler scooty scooter
    accident damage liability third party od own idv depreciation garage garaging
    theft burglary fire flood engine kilometre kms consumable portability migration
    grace indemnity floater
    """.split()
)

# A short follow-up can lean on the previous turns instead of repeating the
# domain words ("why?", "and for Niva Bupa?").
FOLLOWUP_LEAD_RE = re.compile(
    r"^(?:why|how|what\s+about|and|also|more|explain|elaborate|then|ok(?:ay)?|"
    r"which\s+one|really|so)\b",
    re.I,
)

COMPARE_HELP_ANSWER = (
    "I need two plans, two insurers, or a product family to compare. Name them "
    "and I'll quote each wording, for example: \"Compare Health Plus for Niva "
    "Bupa and Digit\" or \"Compare room rent caps for HDFC Ergo and Tata AIG\"."
)

# "compare each", "compare all of them", "compare the above" point at the
# plans the previous answer listed; on their own they name no plan.
FOCUS_REFERENCE_RE = re.compile(
    r"\b(?:each|all|both|them|these|those|above|every\s+one|every)\b", re.I
)
# "the above", "these", "those" point at the last answer even when a word in
# the follow-up happened to match some unrelated plan.
FOCUS_STRONG_RE = re.compile(
    r"\b(?:above|these|those|them|the\s+list)\b", re.I
)
FOCUS_ASPECTS = (
    "Compare these plans on room rent cap, co-pay, waiting period and exclusions."
)
FOCUS_ASPECTS_DIFFERENCES = (
    "What differs between these plans on room rent cap, co-pay, waiting period "
    "and exclusions?"
)
FOCUS_RETRIEVAL_QUERY = (
    "room rent cap co-pay waiting period exclusions sub-limit"
)
# Each plan needs its own clauses in the prompt; past this many the table gets
# wider than the model can fill without dropping a column, and the clauses no
# longer fit beside the answer.
FOCUS_PLAN_CAP = 4


class BeemaSathiOrchestrator:
    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        generator: Optional[BimaSaathiGenerator] = None,
        predictor: Optional[ClaimPredictor] = None,
    ):
        self.retriever = retriever or PolicyRetriever()
        self.generator = generator or BimaSaathiGenerator()
        self.predictor = predictor or ClaimPredictor()
        self.router = IntentRouter(self.retriever.store.get_catalog())

    def route(
        self, question: str, insurer: Optional[str] = None, policy_name: Optional[str] = None
    ) -> TaskPlan:
        return self.router.route(question, insurer=insurer, policy_name=policy_name)

    @staticmethod
    def _is_format_only(question: str) -> bool:
        return (
            table_requested(question)
            and len(question.split()) <= 8
            and not FOLLOWUP_COVERAGE_RE.search(question)
        )

    def _resolve_followup_question(
        self, question: str, history: Optional[List[Dict]]
    ) -> str:
        """Answer a format-only follow-up against the question it refers to."""
        if not history or not self._is_format_only(question):
            return question
        for message in reversed(history):
            if message.get("role") != "user":
                continue
            previous = str(message.get("content") or "").strip()
            if previous and not self._is_format_only(previous):
                return previous
        return question

    @staticmethod
    def _wants_focus_comparison(question: str) -> bool:
        """True for "compare each / all of them / the above" style follow-ups."""
        text = question or ""
        return bool(COMPARE_RE.search(text)) and bool(FOCUS_REFERENCE_RE.search(text))

    @staticmethod
    def _apply_focus_plans(plan: TaskPlan, focus: Optional[List[Dict]]):
        """Point a plan-less comparison at the plans the last answer listed.

        Returns the plan and how many plans the listing held (0 when the
        follow-up does not resolve to any plan). At most FOCUS_PLAN_CAP plans
        are put in one table; the caller notes how many were left out.
        """
        buckets: Dict[str, List[str]] = {}
        listed = 0
        selected = 0
        for item in focus or []:
            insurer = (item or {}).get("insurer")
            policy = (item or {}).get("policy")
            if not insurer or not policy:
                continue
            if policy in (buckets.get(insurer) or []):
                continue
            listed += 1
            if selected >= FOCUS_PLAN_CAP:
                continue
            buckets.setdefault(insurer, []).append(policy)
            selected += 1
        if not selected:
            return plan, 0
        insurers = list(buckets)
        return (
            replace(
                plan,
                intent="comparison",
                insurers=insurers,
                policies=[],
                topic_plans=buckets,
                lead_topic=None,
                steps=[TaskStep("retrieve", target=key) for key in insurers]
                + [TaskStep("compare")],
            ),
            listed,
        )

    def _in_domain(
        self,
        question: str,
        insurer: Optional[str],
        policy_name: Optional[str],
        history: Optional[List[Dict]],
    ) -> bool:
        """True when the question belongs to the loaded insurance corpus.

        A selection in the UI, an insurer or plan name, an insurance word, a
        chat-meta question, or a short follow-up is enough. Everything else is
        refused before retrieval so the model cannot answer from general
        knowledge.
        """
        if insurer or policy_name:
            return True
        if self.router.find_insurers(question):
            return True
        tokens = set(re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).split())
        if tokens & DOMAIN_TERMS:
            return True
        if is_chat_meta_question(question):
            return True
        if history and FOLLOWUP_LEAD_RE.match((question or "").strip()):
            return True
        return False

    def _compare_help(self, plan: TaskPlan) -> Dict:
        """Offer real families to compare instead of guessing a pairing."""
        if len(plan.topic_plans) == 1:
            insurer, plans = next(iter(plan.topic_plans.items()))
            answer = (
                f"I hold **{self._policy_label(plans[0])}** for "
                f"{self._insurer_label(insurer)}. Name a second plan or insurer "
                "and I'll compare their clauses side by side."
            )
        else:
            families = self.router.topic_families(3)
            if families:
                lines = []
                for family in families:
                    names = [self._insurer_label(key) for key in family["plans"]]
                    if len(names) == 2:
                        joined = f"{names[0]} and {names[1]}"
                    else:
                        joined = ", ".join(names[:-1]) + " and " + names[-1]
                    lines.append(f"- **{family['label'].title()}** — {joined}")
                answer = (
                    "I can compare any product family held for two or more "
                    "insurers. For example:\n\n"
                    + "\n".join(lines)
                    + "\n\nName the family, or two plans, and I'll line up "
                    "their constraints clause by clause."
                )
            else:
                answer = COMPARE_HELP_ANSWER
        result = {
            "answer": answer,
            "safe": True,
            "sources": [],
            "rejection_risk": None,
            "model_metrics": {
                "backend": "guard",
                "model": self.generator.model,
                "reason": "comparison-needs-two-plans",
            },
        }
        return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

    def answer(
        self,
        question: str,
        insurer: Optional[str] = None,
        policy_name: Optional[str] = None,
        strategy: str = "baseline",
        claim: Optional[Dict] = None,
        catalog: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        followup_plans: Optional[List[Dict]] = None,
    ) -> Dict:
        trace = Trace(question, insurer=insurer, policy_name=policy_name)
        started = time.perf_counter()
        try:
            result = self._answer(
                question, insurer, policy_name, strategy, claim, catalog, history,
                followup_plans,
            )
        except Exception as exc:
            trace.record_error(exc)
            trace.emit()
            raise
        trace.record_result(result, (time.perf_counter() - started) * 1000)
        trace.emit()
        return result

    def _answer(
        self,
        question: str,
        insurer: Optional[str] = None,
        policy_name: Optional[str] = None,
        strategy: str = "baseline",
        claim: Optional[Dict] = None,
        catalog: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        followup_plans: Optional[List[Dict]] = None,
    ) -> Dict:
        guard = scan_query(question)
        if guard.action == "refuse":
            plan = replace(
                self.route(question, insurer, policy_name), intent=guard.plan_intent
            )
            result = guard.refusal_result(self.generator.model)
            result["guard"] = guard.as_dict()
            return self._wrap(
                result, plan, {"verified": True, "unsupported": []}, []
            )

        catalog = catalog if catalog is not None else self.retriever.store.get_catalog()
        table = table_requested(question)
        lookup_question = self._resolve_followup_question(question, history)
        plan = self.route(lookup_question, insurer, policy_name)

        # "compare each" after a list names no plan by itself; the plans it
        # points at are the ones the previous answer listed. Only override the
        # route when it could not assemble a comparison on its own.
        routed_plans = sum(len(plans) for plans in plan.topic_plans.values()) or len(
            plan.insurers
        )
        focus_count = 0
        if (
            followup_plans
            and self._wants_focus_comparison(lookup_question)
            and (
                FOCUS_STRONG_RE.search(lookup_question)
                or (routed_plans < 2 and len(plan.insurers) < 2)
            )
        ):
            plan, focus_count = self._apply_focus_plans(plan, followup_plans)

        if plan.intent in {"safety", "security", "scope"}:
            result = self.generator.generate(
                [], question, catalog=catalog, strategy=strategy, history=history
            )
            return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

        if not focus_count and not self._in_domain(
            lookup_question, insurer, policy_name, history
        ):
            result = {
                "answer": OFF_TOPIC_ANSWER,
                "safe": True,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "guard",
                    "model": self.generator.model,
                    "reason": "off-topic",
                },
            }
            return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

        if plan.intent == "recommend":
            return self._recommend(plan, catalog)

        if plan.intent == "catalog_list":
            return self._catalog_list(plan, catalog)

        if plan.intent == "claim_risk" and claim:
            return self._claim_answer(question, plan, claim)

        if plan.intent == "comparison":
            plan_count = sum(len(plans) for plans in plan.topic_plans.values()) or len(
                plan.insurers
            )
            if plan_count < 2:
                return self._compare_help(plan)
            return self._compare(
                lookup_question, plan, plan.insurers, catalog, history,
                focus=focus_count,
            )

        chunks = self._retrieve_for_question(
            lookup_question,
            insurer or (plan.insurers[0] if plan.insurers else None),
            policy_name or (plan.policies[0] if plan.policies else None),
            prefer_brochure=plan.prefer_brochure,
        )
        result = self.generator.generate(
            chunks, lookup_question, catalog=catalog, strategy=strategy,
            history=history, table=table,
        )
        verification = self._verify(
            result, chunks, lookup_question, catalog, strategy, history, table
        )
        result["answer"] = strip_unbacked_family_limits(result.get("answer", ""), chunks)
        return self._wrap(result, plan, verification, chunks)

    def _retrieve_for_question(
        self,
        question: str,
        insurer: Optional[str],
        policy_name: Optional[str],
        n_results: int = 8,
        prefer_brochure: bool = False,
    ) -> List[Dict]:
        """Retrieve once per asked aspect when the question packs several.

        A single embedding for "waiting period and co-pay" collapses onto the
        first aspect, so the second never reaches the prompt. Each aspect is
        retrieved on its own and put first, then the whole-question chunks fill
        the rest.
        """
        chunks = self.retriever.retrieve(
            query=question,
            insurer=insurer,
            policy_name=policy_name,
            n_results=n_results,
        )
        aspects = asked_aspects(question) if multi_part_question(question) else []
        if len(aspects) >= 2:
            seen = {
                chunk.get("metadata", {}).get("chunk_id")
                for chunk in chunks
                if chunk.get("metadata", {}).get("chunk_id")
            }
            per_aspect = []
            for aspect in aspects:
                for chunk in self.retriever.retrieve(
                    query=aspect,
                    insurer=insurer,
                    policy_name=policy_name,
                    n_results=3,
                ):
                    key = chunk.get("metadata", {}).get("chunk_id")
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    per_aspect.append(chunk)
            chunks = per_aspect + chunks
        return resolve_authority(chunks, prefer_brochure=prefer_brochure)

    def _recommend(self, plan: TaskPlan, catalog: List[Dict]) -> Dict:
        """List plans with loaded wording; never rank or invent family coverage."""
        skip = (
            "motor", "vehicle", "auto", "two wheeler", "battery", "commercial",
            "liability", "gcv", "personal accident", "home", "travel", "package",
            "group", "loan", "credit", "micro", "rural", "janata", "shramik",
            "aap ke liye", "aapke liye", "apnar jonyo", "aponar babe",
            "mee kosam", "khyed kyi", "i tan", "adomgidamak", "antodaya",
            "pm jay", "gramin", "educational", "institution", "school",
            "college", "healthy glow",
        )
        entries = [
            entry
            for entry in catalog
            if entry.get("document_type") == "policy_wording"
            and (not plan.insurers or entry.get("insurer") in plan.insurers)
        ]
        entries.sort(key=lambda entry: -(entry.get("chunks") or 0))
        picked, seen = [], set()
        per_insurer: Dict[str, int] = {}
        for entry in entries:
            blob = (
                f"{entry.get('policy_name', '')} {entry.get('source_file', '')}"
                .lower()
                .replace("_", " ")
            )
            if any(hint in blob for hint in skip):
                continue
            label = entry["policy_name"].replace("_", " ").title()
            key = (entry["insurer"], label.lower())
            if key in seen or per_insurer.get(entry["insurer"], 0) >= 2:
                continue
            seen.add(key)
            per_insurer[entry["insurer"]] = per_insurer.get(entry["insurer"], 0) + 1
            picked.append((entry["insurer"], label))
            if len(picked) == 8:
                break

        lines = [
            'I can\'t say which plan is "good" or "best" for your family. I hold '
            "policy wordings, not premiums, network lists, or the insurer's "
            "underwriting rules, so any ranking would be a guess."
        ]
        if picked:
            lines.append("")
            lines.append("Plans whose full wording is loaded here:")
            for insurer, label in picked:
                lines.append(
                    f"- {INSURERS.get(insurer, {}).get('name', insurer)}: {label}"
                )
            lines.append("")
            lines.append("For a family, compare these clauses first:")
            lines.append("- Who counts as family, and whether the sum insured is shared")
            lines.append("- Room-rent cap and co-pay")
            lines.append("- Initial, specified-disease, and pre-existing waiting periods")
            lines.append(
                "Name a plan above and I'll pull those clauses exactly."
            )
        else:
            if plan.insurers:
                names = ", ".join(
                    INSURERS.get(key, {}).get("name", key) for key in plan.insurers
                )
                lines.append(
                    f"I don't have a full health policy wording loaded for {names}, "
                    "so I can't list its plans yet."
                )
            else:
                lines.append(
                    "No health policy wording is loaded yet, so I can't even list "
                    "plans reliably."
                )
        result = {
            "answer": "\n".join(lines),
            "safe": True,
            "sources": [],
            "rejection_risk": None,
            "model_metrics": {"backend": "catalog", "model": "policy_catalog"},
        }
        return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

    @staticmethod
    def _policy_label(name: str) -> str:
        text = re.sub(r"\.(pdf|txt)$", "", str(name or "unknown"), flags=re.I)
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"^(?:of|for)\s+(?:covers?\s+)?", "", text, flags=re.I)
        text = re.sub(r"^csc\s+", "", text, flags=re.I)
        text = re.sub(
            r"\b(covers|wef|cum|print|document|prospectus|prospectuse|website)\b\s*$",
            "",
            text,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\b(?:healthplus|finalpolicy|loansure|prospectuse)\b", "", text, flags=re.I
        )
        text = cleaned if cleaned.strip() else text
        text = re.sub(r"(?<=[a-z])\d+$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text.title() or "Unknown"

    @staticmethod
    def _insurer_label(insurer: str, line: Optional[str] = None) -> str:
        if line == "motor" and insurer == "digit":
            return "Go Digit"
        return (
            INSURERS.get(insurer, {}).get("name")
            or str(insurer).replace("_", " ").title()
        )

    @staticmethod
    def _line_label(plan: TaskPlan) -> str:
        if plan.line == "motor":
            return {
                "two_wheeler": "two-wheeler motor",
                "car": "car / private car",
                "commercial": "commercial vehicle",
            }.get(plan.vehicle, "motor")
        if plan.line == "health":
            return "health"
        return "policy"

    def _catalog_list(self, plan: TaskPlan, catalog: List[Dict]) -> Dict:
        """List loaded wordings for a line; never rank or invent coverage."""
        entries = [
            entry
            for entry in catalog
            if entry.get("document_type") == "policy_wording"
            and (not plan.insurers or entry.get("insurer") in plan.insurers)
        ]
        if plan.line:
            matching = [entry for entry in entries if entry_line(entry) == plan.line]
            if matching:
                entries = matching
        if plan.vehicle:
            matching = [entry for entry in entries if entry_vehicle(entry) == plan.vehicle]
            if matching:
                entries = matching

        total = len(entries)
        buckets: Dict[str, List[str]] = {}
        raw_names: Dict[tuple, str] = {}
        seen = set()
        for entry in sorted(entries, key=lambda item: -(item.get("chunks") or 0)):
            insurer = entry.get("insurer", "unknown")
            label = self._policy_label(entry.get("policy_name", ""))
            key = (insurer, label.lower())
            if key in seen:
                continue
            seen.add(key)
            raw_names[key] = entry.get("policy_name") or label
            buckets.setdefault(insurer, []).append(label)

        insurers_present = sorted(
            buckets, key=lambda key: list(INSURERS).index(key) if key in INSURERS else 99
        )
        picked: List[tuple] = []
        while len(picked) < plan.limit:
            added = False
            for insurer in insurers_present:
                if buckets[insurer]:
                    picked.append((insurer, buckets[insurer].pop(0)))
                    added = True
                    if len(picked) >= plan.limit:
                        break
            if not added:
                break

        wants_ranking = bool(
            re.search(
                r"\b(best|good|rank|recommend|suggest|which|compare)\b",
                plan.question,
                re.I,
            )
        )
        line_label = self._line_label(plan)
        names = [
            self._insurer_label(key, plan.line) for key in insurers_present
        ]

        lines = []
        if wants_ranking:
            lines.append(
                'I can\'t say which plan is "good" or "best": I hold policy wordings, '
                "not premiums, network lists, or the insurer's underwriting rules."
            )
            lines.append("")
        if picked:
            noun = "wording" if total == 1 else "wordings"
            lines.append(
                f"I have {total} {line_label} policy {noun} loaded"
                + (f", from {', '.join(names)}" if names else "")
                + (". Here it is:" if len(picked) == 1 else f". Here are {len(picked)}:")
            )
            for insurer, label in picked:
                lines.append(
                    f"- {self._insurer_label(insurer, plan.line)}: {label}"
                )
            lines.append("")
            if plan.line == "motor":
                lines.append(
                    "Ask me about any plan above and I'll quote its clauses exactly: "
                    "own damage, third-party liability, zero-depreciation and other "
                    "add-ons, exclusions, and no-claim bonus."
                )
            elif plan.line == "health":
                lines.append(
                    "Ask me about any plan above and I'll quote its clauses exactly: "
                    "room-rent cap, co-pay, waiting periods, exclusions, and day care."
                )
            else:
                lines.append(
                    "Ask me about any plan above and I'll quote its clauses exactly."
                )
        elif plan.insurers:
            who = ", ".join(
                self._insurer_label(key, plan.line) for key in plan.insurers
            )
            lines.append(
                f"I don't have a {line_label} policy wording loaded for {who}, "
                "so I can't list its plans yet."
            )
        else:
            lines.append(f"No {line_label} policy wording is loaded yet.")

        result = {
            "answer": "\n".join(lines),
            "safe": True,
            "sources": [],
            "rejection_risk": None,
            "model_metrics": {"backend": "catalog", "model": "policy_catalog"},
            # The listed plans, so "compare each" can point back at them.
            "focus_plans": [
                {
                    "insurer": insurer,
                    "policy": raw_names.get((insurer, label.lower()), label),
                }
                for insurer, label in picked
            ],
        }
        return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

    @staticmethod
    def _clean_plan_label(label: str) -> bool:
        """Reject filename artifacts so examples read like plan names."""
        lowered = label.lower()
        if any(
            marker in lowered
            for marker in (
                "finalpolicy", "t and c", "variant", "healthplus", "loansure",
                "document", "niva bupa", "max bupa", "prospectus", "print", "cum",
            )
        ):
            return False
        return not re.search(r"\beff\b", lowered)

    def _loaded_plan_labels(
        self, catalog: List[Dict], insurer: str, limit: int = 3
    ) -> List[str]:
        entries = sorted(
            (
                entry
                for entry in catalog or []
                if entry.get("insurer") == insurer and entry.get("policy_name")
            ),
            key=lambda entry: -(entry.get("chunks") or 0),
        )
        # Prefer binding wordings: the answer promises to quote clauses.
        wording = [entry for entry in entries if entry.get("document_type") == "policy_wording"]
        labels: List[str] = []
        seen = set()
        for entry in wording or entries:
            label = self._policy_label(entry.get("policy_name", ""))
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
        if not labels:
            return []
        clean = [name for name in labels if self._clean_plan_label(name)]
        return (clean or labels)[:limit]

    def _missing_plan_note(
        self, label: str, insurer: str, question: str, catalog: List[Dict]
    ) -> str:
        """Name the missing plan without claiming the whole insurer is absent.

        The insurer dropdown lists every indexed insurer, so "I don't have the
        policy named for Niva Bupa" reads as a contradiction when only one
        plan of theirs is missing.
        """
        phrase = self.router.named_plan_phrase(question, insurer)
        asked = f"the {phrase} wording" if phrase else "the plan you named"
        held = self._loaded_plan_labels(catalog, insurer)
        if not held:
            return (
                f"**{label}**\nI don't have a {label} policy wording loaded, "
                "so I can't compare its clauses."
            )
        if len(held) == 1:
            inventory = held[0]
        else:
            inventory = ", ".join(held[:-1]) + f" and {held[-1]}"
        return (
            f"**{label}**\nI don't have {asked} loaded for {label}. "
            f"What I do hold for {label} includes {inventory}. "
            "Pick one and I'll quote its clauses exactly."
        )

    @staticmethod
    def _compare_lead(plan: TaskPlan, insurers: List[str]) -> str:
        """Say which product family the sections compare, when it was inferred."""
        if not plan.lead_topic or len(insurers) < 2:
            return ""
        names = [BeemaSathiOrchestrator._insurer_label(key) for key in insurers]
        if len(names) == 2:
            joined = f"{names[0]} and {names[1]}"
        else:
            joined = ", ".join(names[:-1]) + " and " + names[-1]
        return f"Comparing the {plan.lead_topic.title()} plans I hold for {joined}:"

    def _compare(
        self,
        question: str,
        plan: TaskPlan,
        insurers: List[str],
        catalog: List[Dict],
        history: Optional[List[Dict]] = None,
        focus: int = 0,
    ) -> Dict:
        notes = []
        groups = []
        used_chunks = []
        # Resolve the plan each insurer was asked about, separately. A single
        # global match can return a plan from the wrong insurer, and insurer-wide
        # retrieval would let one product's clause be attributed to another.
        named_by_insurer = self.router.find_policies_by_insurer(question)
        question_names_plan = any(named_by_insurer.values()) or any(
            plan.topic_plans.values()
        )
        resolved = []
        for key in insurers:
            label = self._insurer_label(key)
            topics = plan.topic_plans.get(key) or []
            # Two plans of one insurer come from topic_plans; a single plan can
            # come from either resolver.
            named = topics if len(topics) >= 2 else (named_by_insurer.get(key) or topics)
            if question_names_plan and not named:
                notes.append(self._missing_plan_note(label, key, question, catalog))
                continue
            for policy in (named or [None])[:FOCUS_PLAN_CAP]:
                resolved.append((key, label, policy))

        retrieval_question = question
        compare_question = question
        if focus and not asked_aspects(question):
            # "compare each" names no aspect on its own; settle the same four
            # constraints the listing offered. A wide table keeps only the rows
            # that differ so it fits the output budget.
            compare_question = (
                FOCUS_ASPECTS_DIFFERENCES if len(resolved) > 3 else FOCUS_ASPECTS
            )
            retrieval_question = FOCUS_RETRIEVAL_QUERY
        # Each plan needs a few clauses of its own; keep the whole prompt small
        # enough that the table still has room to finish.
        per_plan = max(2, 12 // max(len(resolved), 1))

        for key, label, policy in resolved:
            plan_label = label
            if policy:
                named_label = re.sub(
                    r"\s+polic(?:y|ies)(?:\s+(?:revision|wording|document|terms|cis))?$",
                    "",
                    self._policy_label(policy),
                    flags=re.I,
                )
                plan_label = (
                    named_label
                    if named_label.lower().startswith(label.lower())
                    else f"{label} {named_label}"
                )
            chunks = self._retrieve_for_question(
                retrieval_question,
                key,
                policy,
                n_results=per_plan,
                prefer_brochure=plan.prefer_brochure,
            )
            if not chunks:
                notes.append(self._missing_plan_note(label, key, question, catalog))
                continue
            groups.append((plan_label, chunks))
            used_chunks.extend(chunks)

        if len(groups) < 2:
            if notes and groups:
                notes.append(
                    f"I hold only the {groups[0][0]} wording here, so I can't line "
                    "the two plans up side by side."
                )
            merged = {
                "answer": "\n\n".join(notes) or COMPARE_HELP_ANSWER,
                "safe": True,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {"backend": "guard", "model": self.generator.model},
            }
            return self._wrap(merged, plan, {"verified": True, "unsupported": []}, used_chunks)

        result = self.generator.compare_plans(compare_question, groups, history=history)
        answer = result.get("answer", "")
        lead = self._compare_lead(plan, insurers)
        if lead:
            answer = lead + "\n\n" + answer
        if focus and len(groups) < focus:
            notes.append(
                f"I compared {len(groups)} of those {focus} plans. Name the "
                "others and I'll line them up too."
            )
        if notes and result.get("safe", True):
            answer += "\n\n" + "\n\n".join(notes)

        merged = {
            **result,
            "answer": answer,
            "sources": self._dedupe_sources(result.get("sources") or []),
        }
        verification = verify_answer(merged["answer"], used_chunks)
        merged["answer"] = strip_unbacked_family_limits(merged["answer"], used_chunks)
        return self._wrap(merged, plan, verification, used_chunks)

    def _claim_answer(self, question: str, plan: TaskPlan, claim: Dict) -> Dict:
        prediction = self.predictor.predict(
            insurer=claim.get("insurer") or "unknown",
            policy_name=claim.get("policy_name") or "unknown",
            hospital_name=claim.get("hospital_name"),
            is_network_hospital=claim.get("is_network_hospital"),
            procedure=claim.get("procedure"),
            room_rent_per_day=claim.get("room_rent_per_day"),
            sum_insured=claim.get("sum_insured"),
            has_ped=claim.get("has_ped"),
            months_since_policy=claim.get("months_since_policy"),
            intimation_hours=claim.get("intimation_hours"),
        )
        lines = [
            f"Rejection risk: {prediction['rejection_risk_percent']}% "
            f"({prediction['risk_level']})."
        ]
        summary = re.sub(
            r"^rejection risk:\s*[^.]*\.\s*", "", prediction["summary"], flags=re.I
        )
        if summary:
            lines.append(summary)
        for check in prediction["failed_checks"]:
            lines.append(f"- {check['message']} Action: {check['action']}")
        result = {
            "answer": "\n".join(lines),
            "safe": True,
            "sources": [],
            "rejection_risk": prediction["risk_level"],
            "prediction": prediction,
            "model_metrics": {"backend": "rules", "model": "claim_predictor"},
        }
        return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

    def _verify(
        self,
        result: Dict,
        chunks: List[Dict],
        question: str,
        catalog: List[Dict],
        strategy: str,
        history: Optional[List[Dict]] = None,
        table: bool = False,
    ) -> Dict:
        verification = verify_answer(result.get("answer", ""), chunks)
        if verification["verified"] or not result.get("safe", True):
            return verification
        retry = self.generator.generate(
            chunks, question, catalog=catalog,
            strategy="baseline" if table else "structured",
            history=history, table=table,
        )
        retry_check = verify_answer(retry.get("answer", ""), chunks)
        if retry_check["unsupported"]:
            # The retry is discarded, so only cite figures the customer can see.
            visible = verification.get("unsupported") or []
            if visible:
                result["answer"] = (
                    result.get("answer", "")
                    + "\n\nNote: I could not verify "
                    + ", ".join(visible)
                    + " in the loaded wording. Please confirm with the insurer."
                )
            return {**verification, "retried": True}
        result.update(
            {
                "answer": retry.get("answer"),
                "safe": retry.get("safe"),
                "model_metrics": retry.get("model_metrics"),
            }
        )
        return {**retry_check, "retried": True}

    @staticmethod
    def _dedupe_sources(sources: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for source in sources:
            key = (source.get("source"), source.get("page"), source.get("policy_name"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(source)
        return unique

    @staticmethod
    def _wrap(
        result: Dict, plan: TaskPlan, verification: Dict, chunks: List[Dict]
    ) -> Dict:
        return {
            **result,
            "plan": plan.as_dict(),
            "verification": verification,
            "retrieved": len(chunks),
        }
