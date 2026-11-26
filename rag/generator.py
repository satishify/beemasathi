"""Generate responses using local LLM with safety checks."""

import re
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BANNED_PATTERNS,
    CONSTITUTIONAL_PROMPT,
    INSURERS,
    KNOWLEDGE_SOURCES,
    LLM_API_BASE,
    LLM_API_CA,
    LLM_API_KEY,
    LLM_API_MAX_TOKENS,
    LLM_API_CONTEXT,
    LLM_API_MODEL,
    LLM_LOCAL,
    SAFETY_REFUSAL,
    SECURITY_PATTERNS,
    SECURITY_REFUSAL,
)
from rag.catalog import line_counts
from rag.policy_match import insurer_phrases

# The hosted group key allows one in-flight request, so serialize API calls.
_API_LOCK = threading.Lock()
_API_RETRY_DELAYS = (2, 4, 8)
_ROOT_DIR = Path(__file__).resolve().parent.parent


def _hosted_ca_verify() -> Any:
    """Resolve a relative CA path against the project root."""
    if not LLM_API_CA:
        return True
    path = Path(LLM_API_CA)
    return str(path if path.is_absolute() else _ROOT_DIR / path)


def _local_backend_ready() -> bool:
    """True when an Ollama server is listening, so a fallback can answer."""
    try:
        import httpx

        httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


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

# Models sometimes copy the clause header from the prompt into the answer, e.g.
# "Deciding Clause: [2] digit | digit_two_wheeler... | Page 5 | policy_wording |
# page 5 "Policy may be..." Answer: ...". Customers should see only plain prose.
CLAUSE_HEADER_RE = re.compile(
    r"\[\d+\][^|\n]*(?:\|[^|\n]*)*\|\s*page\s*\d+",
    re.I,
)
ECHO_LABEL_RE = re.compile(
    r"^[ \t>*\-•]*(?:\*\*)?"
    r"(?:deciding clause|relevant policy clause|policy clause|direct answer|"
    r"final answer|short answer|answer|conclusion)"
    r"\s*(?:\*\*\s*:\s*|:\s*(?:\*\*)?)\s*",
    re.I,
)
ECHO_QUOTE_RE = re.compile(
    r'^[ \t>*\-•]*(?:\*\*)?["“][^"”\n]{40,}["”](?:\*\*)?\s*',
)

# Questions about the chat itself ("what was my first question?") have no
# matching policy clause, so they must not hit the gap refusal.
CHAT_META_RE = re.compile(
    r"\b(?:what\s+(?:did|was|were)\s+(?:i|my|we)|"
    r"(?:my|our)\s+(?:first|last|previous|earlier|initial)\s+(?:question|message)|"
    r"(?:previous|earlier|last)\s+(?:question|answer|message)|"
    r"summari[sz]e\s+(?:our|the|this)\s+(?:chat|conversation)|"
    r"what\s+have\s+(?:i|we)\s+(?:asked|discussed))\b",
    re.I,
)

HISTORY_MAX_MESSAGES = 8
HISTORY_MAX_CHARS = 2400
HISTORY_MESSAGE_CHARS = 500

# A formatting request changes presentation only. Without an explicit output
# contract the model drops it and repeats plain prose.
TABLE_REQUEST_RE = re.compile(r"\b(?:table|tabular|grid)\b|\brows?\s+and\s+columns?\b", re.I)
TABLE_FORMAT_INSTRUCTION = (
    "\n\nFORMAT: Answer with a GitHub-flavoured Markdown table only. "
    "Start with a header row and a | --- | separator row, then one row per fact. "
    "Do not add prose before or after the table."
)

# A comparison asked as "what is different" needs a decompose -> extract ->
# differentiate answer, not two separate plan monologues that each talk about
# the question's wording instead of the plans' constraints.
COMPARE_STEP_DIFFERENTIATE = (
    "STEP 3 - DIFFERENTIATE. Keep only areas where the plans actually differ. "
    "Drop any area where they match or where neither plan states a rule. If one "
    'plan states a limit the other does not, that is a difference and the missing '
    'side is "Not stated".'
)
COMPARE_STEP_ALIGN = (
    "STEP 3 - ALIGN. The question asks about a specific constraint, so keep one "
    "row for each constraint asked about and show every plan's own rule. Keep "
    "rows where the plans agree; drop a row only when no plan states a rule for it."
)
COMPARE_NOTE_DIFFERENCES = (
    "Never write that the plans are the same or similar. If no row shows a "
    "difference, say so in one sentence instead of inventing one."
)
COMPARE_NOTE_ALIGN = (
    "Do not invent a difference and do not claim more than the table shows."
)

COMPARE_TEMPLATE = """CUSTOMER QUESTION:
{question}

PLANS BEING COMPARED: {labels}

STEP 1 - UNDERSTAND. Break the question into the constraint areas it asks about. When the question is vague ("what constraints differ"), use these areas in order: room rent and ICU limits, co-pay, waiting periods (initial, pre-existing, specific diseases), sub-limits and disease caps, exclusions, hospital network and cashless rules, no-claim bonus and sum-insured refill, renewal and loading rules. Keep at most six areas.

STEP 2 - EXTRACT. For each plan, note what that plan's own clauses say for each area, with its section number or page.

{step3}

STEP 4 - ANSWER. Start with one short line naming the plans. Then a GitHub-flavoured Markdown table:

{table_header}
{table_sep}
| ... | ... | ... |

Each cell must state the actual rule from that plan - the limit, waiting period, co-pay or exclusion exactly as written - followed by its section or page in brackets, for example "1% of sum insured (Sec. 4.2, p. 42)". A cell holding only a page number is not an answer. If a plan does not state the rule, write "Not stated". Then at most three bullets, and each bullet must point to one row of the table. {mode_note}

RULES:
- Compare plan against plan. Never discuss "two companies", the question's phrasing, or that you are an AI.
- Copy the header row and separator row from STEP 4 exactly, with the leading and trailing pipe.
- One row per constraint. Never pair two unrelated rules in the same row: both plans must address the same constraint, or one addresses it and the other says "Not stated" for that same constraint.
- Never output a row where every plan cell says "Not stated".
- Quote numbers exactly as written in the clauses.
- Never state a constraint that is not in the clauses below.
- If the clauses do not settle a row, leave the row out.
- Close with: "This is informational only. For legal advice, consult a lawyer."

RECENT CONVERSATION (may be empty):
{history}

PLAN CLAUSES:
{context}
"""

# Questions that name several aspects at once ("room rent, co-pay and waiting
# period") need each part answered separately, in a table, instead of one
# blended paragraph that covers only the easiest part.
ASPECT_PATTERNS = (
    ("waiting period", r"\bwaiting\s*(?:period|time)\b|\bpre-?existing\b|\bspecific\s+disease"),
    ("co-pay", r"\bco-?pay(?:ment)?\b|\bdeductible\b"),
    ("room rent and ICU", r"\broom\s*rent\b|\bicu\b|intensive\s+care"),
    ("sub-limits", r"\bsub-?limits?\b"),
    ("exclusions", r"\bexclu(?:de|des|ded|ding|sion|sions)\b"),
    ("network and cashless", r"\bcashless\b|\bnetwork\b"),
    ("no-claim bonus and refill", r"\bno[- ]claim\b|\bncb\b|\brefill\b|\brestore\b"),
    ("renewal and grace", r"\brenew(?:al|als|able)?\b|\bgrace\s+period\b|\bloading\b"),
    ("ambulance", r"\bambulance\b"),
    ("day care", r"\bday[- ]care\b"),
    ("maternity", r"\bmaternity\b"),
    ("domiciliary", r"\bdomiciliary\b"),
    ("organ donor", r"\borgan\s+donor\b"),
)

DIFFERENCE_ASK_RE = re.compile(
    r"\b(?:what|which)\b[^.?!]{0,40}\b(?:differ|differs|different|difference)\b|"
    r"\bdifferent\s+(?:constraints?|terms|conditions|limits|coverage|benefits)\b|"
    r"\bsame\s+(?:purpose|benefit|cover(?:age)?)\b[^.?!\n]{0,40}\bdifferent\b",
    re.I,
)

def multi_aspect_instruction(question: str) -> str:
    """Table contract that names the aspects, so none is silently dropped."""
    aspects = asked_aspects(question)
    if aspects:
        order = (
            "The aspects asked, in order: "
            + ", ".join(aspects)
            + ". Write one row for each of them, in that order."
        )
    else:
        order = (
            "Write one row for each separate part of the question, in the order "
            "asked."
        )
    return (
        "\n\nFORMAT: The question asks about more than one aspect. Answer with a "
        "GitHub-flavoured Markdown table only: a header row '| Aspect | What the "
        "wording says | Where |', a | --- | separator row, then "
        + order
        + " Put several sub-rules of one aspect in that one row. Quote the exact "
        "numbers and the section or page. An aspect the clauses do not settle must "
        "read 'Not stated in the loaded wording'. Do not add prose before or after "
        "the table."
    )


def asked_aspects(question: str) -> List[str]:
    """Insurance aspects the question names, in the order they appear."""
    text = question or ""
    hits = []
    for name, pattern in ASPECT_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            hits.append((match.start(), name))
    return [name for _, name in sorted(hits)]


def compare_mode(question: str) -> str:
    """Differences-only for "what differs"; side-by-side for a named clause."""
    return "differences" if DIFFERENCE_ASK_RE.search(question or "") else "side_by_side"


def multi_part_question(question: str) -> bool:
    """True when one answer must settle two or more separate aspects."""
    text = question or ""
    if text.count("?") >= 2:
        return True
    spans = []
    for name, pattern in ASPECT_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            spans.append((match.start(), match.end(), name))
    spans.sort()
    for (_, end, first), (start, _, second) in zip(spans, spans[1:]):
        if first == second or start - end > 60:
            continue
        if re.search(r"[,;]|\band\b|\balso\b|\bplus\b|as well as", text[end:start], re.I):
            return True
    return False


COMPARE_SYSTEM = (
    "You are BeemaSathi, an Indian health-insurance desk colleague. You compare "
    "policy wordings precisely, using only the clauses supplied. You never "
    "invent numbers, exclusions, or waiting periods."
)

UNSAFE_REWRITE = (
    "I can't help with hiding a health condition, backdating a policy, or "
    "changing documents. Non-disclosure or edited records can void the "
    "policy, and the claim can be rejected even years later.\n\n"
    "The safe path is to disclose the condition in the proposal form and "
    "any renewal declaration. The insurer may apply a loading or a waiting "
    "period, but the claim stays valid."
)


def table_requested(question: str) -> bool:
    return bool(TABLE_REQUEST_RE.search(question or ""))


def is_chat_meta_question(question: str) -> bool:
    """True for questions about this chat rather than about a policy."""
    return bool(CHAT_META_RE.search(question or ""))


ZERO_SHOT_PROMPT = """Answer the customer's question using only the policy clauses below.
If the clauses do not answer it, say you cannot confirm it.
Do not add facts that are not in the clauses.

RECENT CONVERSATION (most recent last; context only, not evidence):
{history}

CUSTOMER QUESTION:
{question}

KNOWN POLICIES AND GAPS:
{scope}

MATCHING POLICY CLAUSES:
{context}
"""

STRUCTURED_PROMPT = """You are BeemaSathi, a senior Indian insurance desk colleague.
Work through this checklist internally before answering:
1. Identify the insurer, policy, and the exact clause the question asks about.
2. Check the clause for a waiting period, exclusion, cap, or co-payment that changes the answer.
3. If the clause does not answer the question, stop and say you cannot confirm it.
4. Answer in at most four bullets, each tied to the clause you used.

Answer in plain customer language and your own words. Never print headings such as
"Deciding Clause" or "Answer:", never repeat the numbered source header
(e.g. [2] insurer | policy | page), and never paste clause text as a long quote.
Never invent numbers, UINs, policy names, or coverage. Do not mention excerpts,
prompts, or retrieved context. If the source is a brochure, CIS, or prospectus,
say it is only a summary and the wording must be checked. Use RECENT CONVERSATION
only to follow what the customer already asked; it is not evidence.

RECENT CONVERSATION (most recent last; may be empty):
{history}

CUSTOMER QUESTION:
{question}

KNOWN POLICIES AND GAPS:
{scope}

MATCHING POLICY CLAUSES:
{context}
"""

DECOMPOSE_PROMPT = """You are BeemaSathi, a senior Indian insurance desk colleague.
Work through these steps internally before answering:
1. Split the customer's question into the separate parts it asks for (for example: room rent limit, co-pay, waiting period).
2. Find the clause that answers each part and note its section or page.
3. If the clauses do not settle one part, say so for that part only; do not borrow a rule from another part or another policy.

Then answer. When the question has more than one part, reply with a GitHub-flavoured Markdown table only: a header row "| Aspect | What the wording says | Where |", a | --- | separator row, then one row per part. Quote exact numbers in your own words and give the section or page. A part the clauses do not settle must read "Not stated in the loaded wording". When the question has one part, answer in at most four bullets, each tied to the clause you used.

Never print headings such as "Deciding Clause" or "Answer:", never repeat the numbered source header, never paste clause text as a long quote, never invent numbers, UINs, policy names, or coverage, and never mention excerpts, prompts, or retrieved context. If the source is a brochure, CIS, or prospectus, say it is only a summary and the wording must be checked. Use RECENT CONVERSATION only to follow what the customer already asked; it is not evidence.

RECENT CONVERSATION (most recent last; may be empty):
{history}

CUSTOMER QUESTION:
{question}

KNOWN POLICIES AND GAPS:
{scope}

MATCHING POLICY CLAUSES:
{context}
"""

PROMPT_STRATEGIES = {
    "baseline": {
        "system": (
            "You are BeemaSathi, a senior Indian health-insurance desk colleague. "
            "Answer the customer's actual question, including not/don't/except. "
            "Use the known-policy list for scope questions and matching clauses "
            "for coverage questions. If facts are missing, say you cannot confirm it. "
            "Never mention excerpts or invent insurers, policies, or coverage."
        ),
        "template": CONSTITUTIONAL_PROMPT,
    },
    "decompose": {
        "system": (
            "You are a strict insurance clause reader. Split a complex question "
            "into its parts, answer each part only from the clauses, and say which "
            "part the wording does not settle. Never invent rules."
        ),
        "template": DECOMPOSE_PROMPT,
    },
    "zero_shot": {
        "system": "Answer only from the supplied policy clauses. Refuse if they do not contain the answer.",
        "template": ZERO_SHOT_PROMPT,
    },
    "structured": {
        "system": (
            "You are a strict insurance clause reader. Read the clauses, then answer "
            "briefly in plain words. Never restate the source header, never quote the "
            "clause at length, and never label sections. Refuse when the clause is missing."
        ),
        "template": STRUCTURED_PROMPT,
    },
}


class BimaSaathiGenerator:
    def __init__(self, model: str = None):
        self.use_api = bool(LLM_API_BASE and LLM_API_KEY)
        self.model = model or (LLM_API_MODEL if self.use_api else LLM_LOCAL)

    def _source_name(self, insurer_key: str) -> str:
        source = KNOWLEDGE_SOURCES.get(insurer_key, {})
        return source.get("name", insurer_key.replace("_", " ").title())

    def _mentioned_insurers(self, question: str) -> List[str]:
        text = question.lower()
        found = []

        def says(alias: str) -> bool:
            return bool(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text
                )
            )

        for key, info in KNOWLEDGE_SOURCES.items():
            phrases = insurer_phrases(key, info.get("name", ""))
            if any(says(phrase) for phrase in phrases):
                found.append(key)
        return found

    def _select_chunks(
        self, retrieved_chunks: List[Dict], question: str, limit: int = 6
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
        return relevant[:limit], ""

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
        if re.search(
            r"\b(is there|are there|do you (?:have|hold)|is .* available)\b.*"
            r"\b(policy|policies|plan|plans|wording|wordings)\b",
            q,
        ):
            return True
        if re.search(
            r"\b(how many|list|show|name)\b.*\b(policies|plans|wordings)\b",
            q,
        ):
            return True
        return bool(
            re.search(
                r"\b(what (?:all )?(?:policies|insurers)|which (?:policies|insurers)|"
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

    def _strip_clause_echoes(self, text: str) -> str:
        """Remove copied clause headers, meta labels, and long verbatim quotes."""
        lines = []
        for raw in text.splitlines():
            line = raw
            previous = None
            while previous != line:
                previous = line
                line = CLAUSE_HEADER_RE.sub(" ", line)
                line = ECHO_LABEL_RE.sub("", line)
                line = ECHO_QUOTE_RE.sub("", line)
            if line != raw:
                line = re.sub(r"[ \t]{2,}", " ", line).strip()
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        return cleaned or text.strip()

    @staticmethod
    def _drop_repeated_sentences(text: str) -> str:
        """Collapse generation loops that repeat the same sentence."""
        seen = set()
        output = []
        for line in text.splitlines():
            if not line.strip():
                output.append("")
                continue
            kept = []
            for part in re.split(r"(?<=[.!?])\s+", line):
                key = re.sub(r"\s+", " ", part).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                kept.append(part)
            if kept:
                output.append(" ".join(kept))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()

    def _sanitize_answer(self, text: str) -> str:
        cleaned = self._strip_clause_echoes(text)
        for pattern in LEAK_PATTERNS:
            cleaned = re.sub(pattern, "BeemaSathi's knowledge base", cleaned, flags=re.I)
        # Keep line structure: Markdown lists and tables depend on newlines.
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = self._drop_repeated_sentences(cleaned)
        if any(re.search(pattern, text, re.I) for pattern in LEAK_PATTERNS):
            return (
                "I don't have enough of that policy wording in BeemaSathi's knowledge "
                "base to confirm this accurately, so I won't guess.\n\n"
                "Please check the insurer's official policy wording, or add that "
                "document to BeemaSathi and I can walk through the exact clause."
            )
        return cleaned

    def _history_text(self, history: List[Dict]) -> str:
        """Render the recent turns inside a fixed character budget."""
        if not history:
            return "None. This is the first question of the chat."
        lines = []
        for msg in list(history)[-HISTORY_MAX_MESSAGES:]:
            role = "Customer" if msg.get("role") == "user" else "BeemaSathi"
            text = re.sub(r"\s+", " ", str(msg.get("content") or "")).strip()
            if not text:
                continue
            if len(text) > HISTORY_MESSAGE_CHARS:
                text = text[:HISTORY_MESSAGE_CHARS].rstrip() + " …"
            lines.append(f"{role}: {text}")
        while lines and len("\n".join(lines)) > HISTORY_MAX_CHARS:
            lines.pop(0)
        return "\n".join(lines) or "None. This is the first question of the chat."

    def _is_chat_meta_question(self, question: str) -> bool:
        return is_chat_meta_question(question)

    def _build_messages(
        self,
        context: str,
        question: str,
        scope: str,
        strategy: str = "baseline",
        history: str = "None. This is the first question of the chat.",
    ) -> List[Dict[str, str]]:
        config = PROMPT_STRATEGIES.get(strategy, PROMPT_STRATEGIES["baseline"])
        return [
            {"role": "system", "content": config["system"]},
            {
                "role": "user",
                "content": config["template"].format(
                    context=context, question=question, scope=scope, history=history
                ),
            },
        ]

    def _build_prompt(
        self,
        context: str,
        question: str,
        scope: str,
        strategy: str = "baseline",
        history: str = "None. This is the first question of the chat.",
    ) -> str:
        config = PROMPT_STRATEGIES.get(strategy, PROMPT_STRATEGIES["baseline"])
        return config["template"].format(
            context=context, question=question, scope=scope, history=history
        )

    def _safety_check(self, text: str) -> tuple:
        """Check for banned patterns. Returns (is_safe, reason)."""
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Blocked: Response contained disallowed content matching pattern"
        return True, ""

    def _security_check(self, text: str) -> tuple:
        """Detect prompt injection, secret or PII extraction, and role-play bypass."""
        for pattern in SECURITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, f"Blocked: security pattern {pattern}"
        return False, ""

    def _call_model(
        self,
        prompt: str,
        messages: List[Dict[str, str]] = None,
        max_tokens: int = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Answer with the configured backend and retain runtime metrics."""
        chat_messages = messages or [{"role": "user", "content": prompt}]
        if not self.use_api:
            return self._call_local(prompt, chat_messages)
        try:
            return self._call_api(chat_messages, max_tokens=max_tokens)
        except Exception as hosted_error:
            # A hosted outage should not take the app down when the machine
            # already has Ollama; retry the same prompt locally and say so in
            # the metrics so a run is never silently served by another model.
            if not _local_backend_ready():
                raise
            answer, metrics = self._call_local(
                prompt, chat_messages, model=LLM_LOCAL
            )
            return answer, {
                **metrics,
                "fallback": "hosted-unavailable",
                "hosted_error": str(hosted_error)[:200],
            }

    def _call_api(
        self, messages: List[Dict[str, str]], max_tokens: int = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Call the hosted OpenAI-compatible endpoint with bounded retries."""
        import httpx

        # The endpoint counts prompt + completion against one window (8192).
        estimated_prompt_tokens = sum(len(m["content"]) for m in messages) // 3
        budget = LLM_API_CONTEXT - estimated_prompt_tokens - 64
        cap = max_tokens or LLM_API_MAX_TOKENS
        max_tokens = max(256, min(cap, budget))

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
        data = None
        with _API_LOCK:
            with httpx.Client(
                verify=_hosted_ca_verify(),
                trust_env=False,
                timeout=httpx.Timeout(180.0, connect=10.0),
            ) as client:
                for delay in _API_RETRY_DELAYS + (None,):
                    response = client.post(
                        f"{LLM_API_BASE}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    if response.status_code in (429, 503) and delay is not None:
                        time.sleep(delay)
                        continue
                    if response.status_code == 401:
                        raise RuntimeError(
                            "The model API rejected the key (401). Check "
                            "BEEMASATHI_LLM_KEY and BEEMASATHI_LLM_BASE for your group."
                        )
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"Model API error {response.status_code}: {response.text[:300]}"
                        )
                    data = response.json()
                    break

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or message.get("reasoning_content") or ""
        content = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", content, flags=re.S | re.I
        ).strip()
        usage = data.get("usage") or {}
        metrics = {
            "backend": "openai-compatible",
            "model": data.get("model", self.model),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        return content, metrics

    def _call_local(
        self,
        prompt: str,
        chat_messages: List[Dict[str, str]],
        model: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call the local Ollama model and retain non-sensitive runtime metrics."""
        model_name = model or self.model
        try:
            import ollama
        except ImportError:
            ollama = None

        if ollama is not None:
            response = ollama.chat(
                model=model_name,
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
                "model": get_value("model", model_name),
                "done_reason": get_value("done_reason"),
                "total_duration_ns": get_value("total_duration"),
                "load_duration_ns": get_value("load_duration"),
                "prompt_eval_count": get_value("prompt_eval_count"),
                "prompt_eval_duration_ns": get_value("prompt_eval_duration"),
                "eval_count": get_value("eval_count"),
                "eval_duration_ns": get_value("eval_duration"),
            }
            return content, metrics

        # Fallback: use ollama CLI via subprocess
        import subprocess
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip(), {
            "backend": "ollama-cli",
            "model": model_name,
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

    def _line_fact_text(self, catalog: List[Dict]) -> str:
        counts = line_counts(catalog or [])
        if not counts:
            return "Loaded wordings by line: none."
        labels = {"health": "health", "motor": "motor (car and two-wheeler)"}
        parts = [
            f"{labels.get(line, line)}: {count} policy wordings"
            for line, count in sorted(counts.items())
            if count
        ]
        return (
            "Loaded wordings by line: "
            + ", ".join(parts)
            + ". I can list the plans for any line on request."
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

    @staticmethod
    def _source_note(chunks: List[Dict]) -> str:
        doc_types = {chunk["metadata"].get("document_type") for chunk in chunks}
        if doc_types & {"brochure", "cis", "prospectus"}:
            return (
                "**Source note:** This answer uses a product brochure, which is "
                "only a coverage summary. Verify the full policy wording before "
                "making a claim or purchase decision."
            )
        if "regulation" in doc_types:
            return (
                "**Regulatory source note:** This answer summarizes an IRDAI "
                "regulation. Check the applicable policy wording and current official "
                "regulatory publication before acting on a specific claim."
            )
        return ""

    @staticmethod
    def _tidy_compare_answer(
        answer: str, header: str, separator: str, differences_only: bool = True
    ) -> str:
        """Repair the table the contract promised: header, empty rows, bullets."""
        lines = answer.split("\n")
        sep_index = None
        for i, line in enumerate(lines):
            if "---" in line and set(line.strip()) <= set("|-: "):
                sep_index = i
                if i == 0 or "constraint" not in lines[i - 1].lower():
                    if i == 0:
                        lines.insert(0, header)
                        sep_index = 1
                    else:
                        lines[i - 1] = header
                lines[sep_index] = separator
                break
        if sep_index is None:
            return answer

        # The contract is a table, so any prose the model put above it is noise.
        if any(
            line.strip() and not line.strip().startswith("|")
            for line in lines[: sep_index - 1]
        ):
            lines = lines[sep_index - 1 :]
            sep_index = 1

        empty_cells = {"", "not stated", "not specified", "not mentioned", "not applicable"}

        def bare_rule(cell: str) -> str:
            text = re.sub(r"\([^)]*(?:sec\.?|p\.|page|code)[^)]*\)", "", cell, flags=re.I)
            text = re.sub(r"[^a-z0-9% ]+", " ", text.lower())
            return re.sub(r"\s+", " ", text).strip()

        kept = []
        bullets = 0
        data_rows = 0
        kept_rows = 0
        truncated = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|"):
                if not stripped.endswith("|"):
                    truncated = True
                    continue  # truncated row
                if i > sep_index:
                    data_rows += 1
                    cells = [cell.strip(" *_`").lower() for cell in stripped.strip("|").split("|")]
                    if all(cell in empty_cells for cell in cells[1:]):
                        continue
                    rules = {bare_rule(cell) for cell in cells[1:]}
                    if differences_only and len(rules) == 1 and rules != {""}:
                        continue
                    kept_rows += 1
                kept.append(line)
                continue
            if stripped.startswith(("- ", "* ")):
                if differences_only and re.search(
                    r"\b(?:same|identical|no difference)\b", stripped, flags=re.I
                ):
                    continue
                bullets += 1
                if bullets > 3:
                    continue
            kept.append(line)

        if data_rows and kept_rows == 0 and not truncated:
            labels = [cell.strip() for cell in header.strip().strip("|").split("|")][1:]
            if len(labels) == 2:
                names = f"{labels[0]} and {labels[1]}"
            elif labels:
                names = ", ".join(labels[:-1]) + " and " + labels[-1]
            else:
                return answer
            if differences_only:
                return (
                    f"I compared {names} clause by clause. I found no constraint "
                    "where they state different rules in the wording I hold.\n\nThis "
                    "is informational only. For legal advice, consult a lawyer."
                )
            return (
                f"The wording I hold for {names} does not state a rule for what you "
                "asked, so I can't line the plans up on it.\n\nThis is informational "
                "only. For legal advice, consult a lawyer."
            )

        while kept:
            last = kept[-1].strip()
            if last.startswith(("- ", "* ")) and not last.endswith((".", ")", "!", "?")):
                kept.pop()
                continue
            break
        return "\n".join(kept)

    def compare_plans(
        self,
        question: str,
        groups: List[tuple],
        history: List[Dict] = None,
    ) -> Dict:
        """Compare two or more plans in one call and keep only the differences.

        ``groups`` is ``[(label, chunks), ...]``. Each chunk carries that plan's
        own clauses, so the model extracts constraints plan by plan instead of
        writing two separate monologues.
        """
        user_safe, user_reason = self._safety_check(question)
        if not user_safe:
            return {
                "answer": SAFETY_REFUSAL,
                "safe": False,
                "sources": [],
                "rejection_risk": "high",
                "model_metrics": {
                    "backend": "not-called",
                    "model": self.model,
                    "reason": user_reason,
                },
            }

        security_hit, security_reason = self._security_check(question)
        if security_hit:
            return {
                "answer": SECURITY_REFUSAL,
                "safe": False,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "not-called",
                    "model": self.model,
                    "reason": security_reason,
                },
            }

        usable = []
        for label, chunks in groups:
            # Each group is already scoped to one plan by the orchestrator, so
            # the question's own plan name must not filter another plan's
            # clauses away; only the relevance cut applies here.
            chunks, _gap = self._select_chunks(chunks, "")
            if chunks:
                usable.append((label, chunks))
        if len(usable) < 2:
            return self._knowledge_gap_result(MISSING_KNOWLEDGE_ANSWER)

        labels = [label for label, _ in usable]
        used_chunks = []
        context_parts = []
        for label, chunks in usable:
            context_parts.append(f"=== {label} ===")
            for i, chunk in enumerate(chunks, 1):
                meta = chunk["metadata"]
                context_parts.append(
                    f"[{label} #{i}] {meta.get('insurer')} | "
                    f"{meta.get('policy_name')} | {meta.get('section')} | "
                    f"{meta.get('document_type', 'policy_wording')} | "
                    f"page {meta.get('page', 'unknown')}\n"
                    f"{chunk['text']}\n"
                )
                used_chunks.append(chunk)

        table_header = "| Constraint | " + " | ".join(labels) + " |"
        table_sep = "| --- |" + " --- |" * len(labels)
        differences_only = compare_mode(question) == "differences"
        prompt = COMPARE_TEMPLATE.format(
            question=question,
            labels=", ".join(labels),
            step3=COMPARE_STEP_DIFFERENTIATE if differences_only else COMPARE_STEP_ALIGN,
            mode_note=COMPARE_NOTE_DIFFERENCES if differences_only else COMPARE_NOTE_ALIGN,
            table_header=table_header,
            table_sep=table_sep,
            history=self._history_text(history),
            context="\n".join(context_parts),
        )
        messages = [
            {"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            answer, model_metrics = self._call_model(
                prompt, messages=messages, max_tokens=LLM_API_MAX_TOKENS
            )
        except Exception as e:
            hint = (
                "Check BEEMASATHI_LLM_* in .env."
                if self.use_api
                else "Is Ollama running? Run: ollama serve"
            )
            return {
                "answer": f"Error calling the model: {e}. {hint}",
                "safe": False,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "api" if self.use_api else "ollama",
                    "model": self.model,
                    "error": str(e),
                },
            }

        answer = self._sanitize_answer(answer)
        answer = self._tidy_compare_answer(
            answer, table_header, table_sep, differences_only=differences_only
        )

        is_safe, _reason = self._safety_check(answer)
        if not is_safe:
            answer = UNSAFE_REWRITE
        else:
            if "informational only" not in answer.lower():
                answer = (
                    answer.rstrip()
                    + "\n\nThis is informational only. For legal advice, consult "
                    "a lawyer."
                )
            note = self._source_note(used_chunks)
            if note:
                answer += "\n\n" + note

        return {
            "answer": answer,
            "safe": is_safe,
            "sources": [c["metadata"] for c in used_chunks],
            "rejection_risk": None,
            "model_metrics": model_metrics,
        }

    def generate(
        self,
        retrieved_chunks: List[Dict],
        question: str,
        catalog: List[Dict] = None,
        strategy: str = "baseline",
        history: List[Dict] = None,
        table: bool = False,
    ) -> Dict:
        """Full RAG pipeline: prompt → generate → safety check."""
        user_safe, user_reason = self._safety_check(question)
        if not user_safe:
            return {
                "answer": SAFETY_REFUSAL,
                "safe": False,
                "sources": [],
                "rejection_risk": "high",
                "model_metrics": {
                    "backend": "not-called",
                    "model": self.model,
                    "reason": user_reason,
                },
            }

        security_hit, security_reason = self._security_check(question)
        if security_hit:
            return {
                "answer": SECURITY_REFUSAL,
                "safe": False,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "not-called",
                    "model": self.model,
                    "reason": security_reason,
                },
            }

        grouped, regulations = self._catalog_facts(catalog)
        scope = self._catalog_fact_text(grouped, regulations)
        scope += "\n\n" + self._line_fact_text(catalog)
        chat_meta = self._is_chat_meta_question(question)
        if self._is_scope_question(question):
            retrieved_chunks = []

        multi = (
            multi_part_question(question)
            and not chat_meta
            and not self._is_scope_question(question)
        )
        gap_answer = ""
        if retrieved_chunks:
            retrieved_chunks, gap_answer = self._select_chunks(
                retrieved_chunks, question, limit=10 if multi else 6
            )
        elif not self._is_scope_question(question) and not chat_meta:
            gap_answer = MISSING_KNOWLEDGE_ANSWER

        if gap_answer and not self._is_scope_question(question) and not chat_meta:
            return self._knowledge_gap_result(gap_answer)

        context = self._clause_context(retrieved_chunks)
        # A multi-part question is self-contained; a previous table in the chat
        # can prime the model to copy its "Not stated" cells, so history is left
        # out of that prompt.
        history_text = "" if multi else self._history_text(history)
        if multi:
            model_question = question + multi_aspect_instruction(question)
            if strategy == "baseline":
                strategy = "decompose"
        else:
            model_question = question + (TABLE_FORMAT_INSTRUCTION if table else "")
        prompt = self._build_prompt(
            context, model_question, scope, strategy, history_text
        )
        messages = self._build_messages(
            context, model_question, scope, strategy, history_text
        )

        # Call the configured model backend
        try:
            answer, model_metrics = self._call_model(prompt, messages=messages)
        except Exception as e:
            hint = (
                "Check BEEMASATHI_LLM_* in .env."
                if self.use_api
                else "Is Ollama running? Run: ollama serve"
            )
            return {
                "answer": f"Error calling the model: {e}. {hint}",
                "safe": False,
                "sources": [],
                "rejection_risk": None,
                "model_metrics": {
                    "backend": "api" if self.use_api else "ollama",
                    "model": self.model,
                    "error": str(e),
                },
            }

        answer = self._sanitize_answer(answer)

        # Safety check
        is_safe, reason = self._safety_check(answer)
        if not is_safe:
            answer = UNSAFE_REWRITE
        else:
            note = self._source_note(retrieved_chunks)
            if note:
                answer += "\n\n" + note

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
