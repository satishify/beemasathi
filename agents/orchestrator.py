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
from agents.router import IntentRouter, TaskPlan
from config import INSURERS
from observability.trace import Trace
from rag.catalog import entry_line, entry_vehicle
from rag.generator import BimaSaathiGenerator
from rag.query_guard import scan_query
from rag.retriever import PolicyRetriever


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

    def answer(
        self,
        question: str,
        insurer: Optional[str] = None,
        policy_name: Optional[str] = None,
        strategy: str = "baseline",
        claim: Optional[Dict] = None,
        catalog: Optional[List[Dict]] = None,
    ) -> Dict:
        trace = Trace(question, insurer=insurer, policy_name=policy_name)
        started = time.perf_counter()
        try:
            result = self._answer(
                question, insurer, policy_name, strategy, claim, catalog
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
        plan = self.route(question, insurer, policy_name)

        if plan.intent in {"safety", "security", "scope"}:
            result = self.generator.generate([], question, catalog=catalog, strategy=strategy)
            return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

        if plan.intent == "recommend":
            return self._recommend(plan, catalog)

        if plan.intent == "catalog_list":
            return self._catalog_list(plan, catalog)

        if plan.intent == "claim_risk" and claim:
            return self._claim_answer(question, plan, claim)

        if plan.intent == "comparison" and len(plan.insurers) >= 2:
            return self._compare(question, plan, plan.insurers, strategy, catalog)

        chunks = self.retriever.retrieve(
            query=question,
            insurer=insurer or (plan.insurers[0] if plan.insurers else None),
            policy_name=policy_name or (plan.policies[0] if plan.policies else None),
            n_results=8,
        )
        chunks = resolve_authority(chunks, prefer_brochure=plan.prefer_brochure)
        result = self.generator.generate(chunks, question, catalog=catalog, strategy=strategy)
        verification = self._verify(result, chunks, question, catalog, strategy)
        result["answer"] = strip_unbacked_family_limits(result.get("answer", ""), chunks)
        return self._wrap(result, plan, verification, chunks)

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
        text = re.sub(r"(?<=[a-z])\d+$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text.title() or "Unknown"

    @staticmethod
    def _insurer_label(insurer: str, line: Optional[str] = None) -> str:
        if line == "motor" and insurer == "digit":
            return "Go Digit"
        return INSURERS.get(insurer, {}).get("name", insurer)

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
        seen = set()
        for entry in sorted(entries, key=lambda item: -(item.get("chunks") or 0)):
            insurer = entry.get("insurer", "unknown")
            label = self._policy_label(entry.get("policy_name", ""))
            key = (insurer, label.lower())
            if key in seen:
                continue
            seen.add(key)
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
        }
        return self._wrap(result, plan, {"verified": True, "unsupported": []}, [])

    def _compare(
        self,
        question: str,
        plan: TaskPlan,
        insurers: List[str],
        strategy: str,
        catalog: List[Dict],
    ) -> Dict:
        parts = []
        sources = []
        used_chunks = []
        prompt_tokens = 0
        completion_tokens = 0
        safe = True
        # Resolve the plan each insurer was asked about, separately. A single
        # global match can return a plan from the wrong insurer, and insurer-wide
        # retrieval would let one product's clause be attributed to another.
        named_by_insurer = self.router.find_policies_by_insurer(question)
        question_names_plan = any(named_by_insurer.values())
        for key in insurers:
            label = INSURERS.get(key, {}).get("name", key)
            named = named_by_insurer.get(key) or []
            if question_names_plan and not named:
                parts.append(
                    f"**{label}**\nI don't have the policy named for {label} "
                    "loaded, so I can't compare its clauses."
                )
                continue
            chunks = self.retriever.retrieve(
                query=question,
                insurer=key,
                policy_name=named[0] if named else None,
                n_results=6,
            )
            chunks = resolve_authority(chunks, prefer_brochure=plan.prefer_brochure)
            # Each call sees only one insurer's clauses, so scope the question
            # to that insurer. Otherwise the model reads the absent insurer as
            # a missing document and contradicts the section that covers it.
            scoped = (
                f"{question}\n\nAnswer only for {label}. Another section covers "
                "every other insurer in the question, so do not comment on them "
                "or on whether their documents are loaded."
            )
            result = self.generator.generate(
                chunks, scoped, catalog=catalog, strategy=strategy
            )
            parts.append(f"**{label}**\n{result.get('answer', '')}")
            sources.extend(result.get("sources") or [])
            used_chunks.extend(chunks)
            metrics = result.get("model_metrics") or {}
            prompt_tokens += metrics.get("prompt_tokens") or 0
            completion_tokens += metrics.get("completion_tokens") or 0
            safe = safe and bool(result.get("safe", True))

        merged = {
            "answer": "\n\n".join(parts),
            "safe": safe,
            "sources": self._dedupe_sources(sources),
            "rejection_risk": None,
            "model_metrics": {
                "model": self.generator.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "calls": len(insurers),
            },
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
    ) -> Dict:
        verification = verify_answer(result.get("answer", ""), chunks)
        if verification["verified"] or not result.get("safe", True):
            return verification
        retry = self.generator.generate(
            chunks, question, catalog=catalog, strategy="structured"
        )
        retry_check = verify_answer(retry.get("answer", ""), chunks)
        if retry_check["unsupported"]:
            result["answer"] = (
                result.get("answer", "")
                + "\n\nNote: I could not verify "
                + ", ".join(retry_check["unsupported"])
                + " in the loaded wording. Please confirm with the insurer."
            )
            return {**retry_check, "retried": True}
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
