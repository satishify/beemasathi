"""Generate responses using local LLM with safety checks."""

import re
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BANNED_PATTERNS,
    CONSTITUTIONAL_PROMPT,
    INSURERS,
    KNOWLEDGE_SOURCES,
    LLM_LOCAL,
)

# Try to import ollama; if not available, use subprocess fallback
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


MISSING_KNOWLEDGE_ANSWER = (
    "I don't have that policy wording in BeemaSathi's knowledge base yet, "
    "so I can't confirm this accurately. I don't want to guess and give you "
    "the wrong coverage picture.\n\n"
    "Once the official document is added, I can walk through the exact clause. "
    "Until then, please check the insurer's official policy wording or network list."
)

LEAK_PATTERNS = [
    r"excerpts?\s+you\s+(?:provided|shared|gave)",
    r"documents?\s+you\s+(?:provided|shared|uploaded|gave)",
    r"in the (?:provided )?(?:excerpts?|context|retrieved (?:documents?|chunks?))",
    r"policy wording is not included in the excerpts",
    r"not included in the excerpts",
    r"page number where you found",
    r"provide the specific policy document or page number",
]


class BimaSaathiGenerator:
    def __init__(self, model: str = None):
        self.model = model or LLM_LOCAL

    def _source_name(self, insurer_key: str) -> str:
        source = KNOWLEDGE_SOURCES.get(insurer_key, {})
        return source.get("name", insurer_key.replace("_", " ").title())

    def _mentioned_insurers(self, question: str) -> List[str]:
        text = question.lower()
        found = []
        for key, info in KNOWLEDGE_SOURCES.items():
            labels = {
                key.replace("_", " "),
                info["name"].lower(),
            }
            if any(label and label in text for label in labels):
                found.append(key)
        aliases = {
            "hdfc": "hdfc_ergo",
            "hdfc ergo": "hdfc_ergo",
            "niva": "max_bupa",
            "niva bupa": "max_bupa",
            "max bupa": "max_bupa",
            "birla": "aditya_birla",
            "aditya birla": "aditya_birla",
            "kotak": "zurich_kotak",
            "zurich": "zurich_kotak",
            "irdai": "irdai",
            "irda": "irdai",
            "tata aig": "tata_aig",
            "tata-aig": "tata_aig",
            "bajaj": "bajaj_allianz",
            "bajaj allianz": "bajaj_allianz",
            "chola": "cholamandalam",
            "cholamandalam": "cholamandalam",
            "digit": "digit",
            "go digit": "digit",
            "acko": "acko",
            "star health": "star_health",
            "care health": "care_health",
            "icici lombard": "icici_lombard",
        }
        for alias, key in aliases.items():
            if alias in text and key not in found:
                found.append(key)
        return found

    def _select_chunks(
        self, retrieved_chunks: List[Dict], question: str
    ) -> Tuple[List[Dict], str]:
        asked = self._mentioned_insurers(question)
        matching = retrieved_chunks
        if asked:
            matching = [
                chunk
                for chunk in retrieved_chunks
                if chunk.get("metadata", {}).get("insurer") in asked
            ]
            if not matching:
                names = " or ".join(self._source_name(key) for key in asked)
                return [], (
                    f"I don't have {names} policy wording in BeemaSathi's knowledge "
                    "base yet, so I can't confirm this accurately. I don't want to "
                    "guess from another insurer's document.\n\n"
                    "Once the official wording is added, I can check the exact clause. "
                    "Until then, please verify this on the insurer's official website."
                )

        hint = self._policy_hint(question)
        if hint:
            hinted = [
                chunk
                for chunk in matching
                if hint in str(chunk.get("metadata", {}).get("policy_name", "")).lower().replace("_", " ")
                or hint in chunk.get("text", "").lower()[:400]
            ]
            if hinted:
                matching = hinted
            elif asked:
                names = " or ".join(self._source_name(key) for key in asked)
                return [], (
                    f"I don't have the {hint.title()} wording for {names} in "
                    "BeemaSathi's knowledge base yet, so I can't confirm this "
                    "accurately. I don't want to guess from another plan.\n\n"
                    "Once that official wording is added, I can check the exact clause."
                )

        relevant = [
            chunk
            for chunk in matching
            if chunk.get("distance") is None or chunk.get("distance", 0) <= 0.72
        ]
        if not relevant:
            return [], MISSING_KNOWLEDGE_ANSWER
        return relevant[:6], ""

    def _policy_hint(self, question: str) -> str:
        q = question.lower()
        for phrase in (
            "critical illness",
            "health guard",
            "everyday health",
            "liability only",
            "family optima",
            "medicare",
            "arogya sanjeevani",
        ):
            if phrase in q:
                return phrase
        return ""

    def _is_scope_question(self, question: str) -> bool:
        q = question.lower()
        if re.search(r"\b(don'?t|do not|not)\s+(know|have|cover|support)\b", q):
            return True
        return bool(
            re.search(
                r"\b(what (?:all )?(?:policies|insurers)|which policies|"
                r"what do you (?:know|have)|what can you (?:help|answer))\b",
                q,
            )
        )

    def _knowledge_gap_result(self, answer: str) -> Dict:
        return {
            "answer": answer,
            "safe": True,
            "sources": [],
            "rejection_risk": None,
            "model_metrics": {
                "backend": "not-called",
                "model": self.model,
                "reason": "Asked policy details were not in the knowledge base.",
            },
        }

    def _sanitize_answer(self, text: str) -> str:
        cleaned = text
        for pattern in LEAK_PATTERNS:
            cleaned = re.sub(pattern, "BeemaSathi's knowledge base", cleaned, flags=re.I)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if any(re.search(pattern, text, re.I) for pattern in LEAK_PATTERNS):
            return (
                "I don't have enough of that policy wording in BeemaSathi's knowledge "
                "base to confirm this accurately, so I won't guess.\n\n"
                "Please check the insurer's official policy wording, or add that "
                "document to BeemaSathi and I can walk through the exact clause."
            )
        return cleaned

    def _build_messages(
        self, context: str, question: str, scope: str
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are BeemaSathi, a senior Indian health-insurance desk colleague. "
                    "Answer the customer's actual question, including not/don't/except. "
                    "Use the known-policy list for scope questions and matching clauses "
                    "for coverage questions. If facts are missing, say you cannot confirm it. "
                    "Never mention excerpts or invent insurers, policies, or coverage."
                ),
            },
            {
                "role": "user",
                "content": CONSTITUTIONAL_PROMPT.format(
                    context=context, question=question, scope=scope
                ),
            },
        ]

    def _build_prompt(self, context: str, question: str, scope: str) -> str:
        return CONSTITUTIONAL_PROMPT.format(
            context=context, question=question, scope=scope
        )

    def _safety_check(self, text: str) -> tuple:
        """Check for banned patterns. Returns (is_safe, reason)."""
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Blocked: Response contained disallowed content matching pattern"
        return True, ""

    def _call_ollama(
        self, prompt: str, messages: List[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Call the local model and retain non-sensitive runtime metrics."""
        chat_messages = messages or [{"role": "user", "content": prompt}]
        if HAS_OLLAMA:
            response = ollama.chat(
                model=self.model,
                messages=chat_messages,
                options={"temperature": 0.4, "num_ctx": 8192}
            )
            get_value = (
                response.get
                if isinstance(response, dict)
                else lambda key, default=None: getattr(response, key, default)
            )
            message = get_value("message", {})
            content = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            metrics = {
                "backend": "ollama-python",
                "model": get_value("model", self.model),
                "done_reason": get_value("done_reason"),
                "total_duration_ns": get_value("total_duration"),
                "load_duration_ns": get_value("load_duration"),
                "prompt_eval_count": get_value("prompt_eval_count"),
                "prompt_eval_duration_ns": get_value("prompt_eval_duration"),
                "eval_count": get_value("eval_count"),
                "eval_duration_ns": get_value("eval_duration"),
            }
            return content, metrics
        else:
            # Fallback: use ollama CLI via subprocess
            import subprocess
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True, text=True, timeout=120
            )
            return result.stdout.strip(), {
                "backend": "ollama-cli",
                "model": self.model,
            }

    def _pretty_policy_name(self, raw: str) -> str:
        raw = str(raw or "unknown")
        text = re.sub(r"[_-]+", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if re.search(r"(_1| \(1\))$", raw):
            text = re.sub(r"\s+\d+$", "", text)
        words = []
        seen = set()
        for word in text.split():
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
        if len(words) > 5:
            words = words[:4]
        return " ".join(words).title() or "Unknown"

    def _catalog_facts(self, catalog: List[Dict]) -> Tuple[Dict[str, List[str]], bool]:
        grouped = {}
        regulations = False
        for item in catalog or []:
            insurer_key = item.get("insurer", "unknown")
            if insurer_key == "irdai" or item.get("document_type") == "regulation":
                regulations = True
                continue
            policy = self._pretty_policy_name(item.get("policy_name", "unknown"))
            grouped.setdefault(insurer_key, {})
            grouped[insurer_key][policy.lower()] = policy
        return (
            {
                key: sorted(names.values())
                for key, names in grouped.items()
            },
            regulations,
        )

    def _missing_insurers(self, grouped: Dict[str, List[str]]) -> List[str]:
        return [key for key in INSURERS if key not in grouped]

    def _catalog_fact_text(
        self,
        grouped: Dict[str, List[str]],
        regulations: bool,
        missing: List[str] = None,
    ) -> str:
        if not grouped and not regulations:
            known = "No policies are loaded."
        else:
            lines = []
            for insurer_key in sorted(grouped, key=self._source_name):
                names = grouped[insurer_key]
                sample = ", ".join(names[:6])
                extra = f" (+{len(names) - 6} more)" if len(names) > 6 else ""
                lines.append(
                    f"{self._source_name(insurer_key)}: {len(names)} policies. "
                    f"Examples: {sample}{extra}"
                )
            if regulations:
                lines.append("IRDAI regulations: available for legal background only")
            known = "\n".join(lines)
        missing = missing if missing is not None else self._missing_insurers(grouped)
        unknown = (
            ", ".join(self._source_name(key) for key in missing)
            or "none of the configured insurers"
        )
        return (
            f"Known policies:\n{known}\n\n"
            f"Insurers with no policy loaded: {unknown}"
        )

    def _clause_context(self, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "None. No matching policy clause was found."
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk["metadata"]
            source_details = (
                f"{meta.get('document_type', 'policy_wording')}"
                f" | page {meta.get('page', 'unknown')}"
            )
            context_parts.append(
                f"[{i}] {meta['insurer']} | {meta['policy_name']} | "
                f"{meta['section']} | {source_details}\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(context_parts)

    def generate(
        self,
        retrieved_chunks: List[Dict],
        question: str,
        catalog: List[Dict] = None,
    ) -> Dict:
        """Full RAG pipeline: prompt → generate → safety check."""
        user_safe, user_reason = self._safety_check(question)
        if not user_safe:
            return {
                "answer": (
                    "I cannot help with hiding medical history, backdating, "
                    "or fabricating documents. That can void the policy and is illegal.\n\n"
                    "If you want, I can explain what the loaded wording says about "
                    "disclosure and pre-existing conditions."
                ),
                "safe": False,
                "sources": [],
                "rejection_risk": "high",
                "model_metrics": {
                    "backend": "not-called",
                    "model": self.model,
                    "reason": user_reason,
                },
            }

        grouped, regulations = self._catalog_facts(catalog)
        scope = self._catalog_fact_text(grouped, regulations)
        if self._is_scope_question(question):
            retrieved_chunks = []

        gap_answer = ""
        if retrieved_chunks:
            retrieved_chunks, gap_answer = self._select_chunks(
                retrieved_chunks, question
            )
        elif not self._is_scope_question(question):
            gap_answer = MISSING_KNOWLEDGE_ANSWER

        if gap_answer and not self._is_scope_question(question):
            return self._knowledge_gap_result(gap_answer)

        context = self._clause_context(retrieved_chunks)
        prompt = self._build_prompt(context, question, scope)
        messages = self._build_messages(context, question, scope)

        # Call Ollama
        try:
            answer, model_metrics = self._call_ollama(prompt, messages=messages)
        except Exception as e:
            return {
                "answer": f"Error calling local model: {e}. Is Ollama running? Run: ollama serve",
                "safe": False,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "ollama",
                    "model": self.model,
                    "error": str(e),
                },
            }

        answer = self._sanitize_answer(answer)

        # Safety check
        is_safe, reason = self._safety_check(answer)
        if not is_safe:
            answer = (
                "I cannot provide that guidance as it may violate insurance regulations "
                "or ethical standards. I'm here to help you understand your policy "
                "and navigate claims fairly.\n\n"
                f"System note: {reason}"
            )
        elif any(
            chunk["metadata"].get("document_type") in {"brochure", "cis", "prospectus"}
            for chunk in retrieved_chunks
        ):
            answer += (
                "\n\n**Source note:** This answer uses a product brochure, which is "
                "only a coverage summary. Verify the full policy wording before "
                "making a claim or purchase decision."
            )
        elif any(
            chunk["metadata"].get("document_type") == "regulation"
            for chunk in retrieved_chunks
        ):
            answer += (
                "\n\n**Regulatory source note:** This answer summarizes an IRDAI "
                "regulation. Check the applicable policy wording and current official "
                "regulatory publication before acting on a specific claim."
            )

        return {
            "answer": answer,
            "safe": is_safe,
            "sources": [c["metadata"] for c in retrieved_chunks],
            "rejection_risk": None,
            "model_metrics": model_metrics,
        }


if __name__ == "__main__":
    gen = BimaSaathiGenerator()

    dummy_chunks = [
        {
            "text": "Room rent is capped at 1% of sum insured per day. Exceeding this reduces claim proportionately.",
            "metadata": {"insurer": "star_health", "policy_name": "comprehensive", "section": "Room Rent Cap", "chunk_type": "room_rent"}
        },
        {
            "text": "Pre-existing disease waiting period is 36 months for diabetes and hypertension.",
            "metadata": {"insurer": "star_health", "policy_name": "comprehensive", "section": "PED Clause", "chunk_type": "waiting_period"}
        }
    ]

    result = gen.generate(dummy_chunks, "What is the room rent limit in Star Health Comprehensive?")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print("Sources:", result["sources"])
