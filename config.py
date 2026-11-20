"""BeemaSathi Configuration"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

# httpx raises "Invalid port: ':1]'" on the bracketed "[::1]" form that some
# shells put in NO_PROXY, which breaks Hugging Face downloads and the ollama
# client. Drop only the unparseable entries.
for _var in ("NO_PROXY", "no_proxy"):
    _value = os.environ.get(_var)
    if _value:
        _entries = [e for e in (x.strip() for x in _value.split(",")) if e]
        _kept = [e for e in _entries if not (e.startswith("[") and e.endswith("]"))]
        if len(_kept) != len(_entries):
            os.environ[_var] = ",".join(_kept)

INSURERS = {
    "star_health": {"name": "Star Health", "base_url": "https://www.starhealth.in"},
    "hdfc_ergo": {"name": "HDFC Ergo", "base_url": "https://www.hdfcergo.com"},
    "icici_lombard": {"name": "ICICI Lombard", "base_url": "https://www.icicilombard.com"},
    "max_bupa": {"name": "Niva Bupa (Max)", "base_url": "https://www.nivabupa.com"},
    "care_health": {"name": "Care Health", "base_url": "https://www.careinsurance.com"},
    "zurich_kotak": {"name": "Zurich Kotak", "base_url": "https://www.zurichkotak.com"},
    "acko": {"name": "Acko", "base_url": "https://www.acko.com"},
    "digit": {"name": "Digit Health", "base_url": "https://www.godigit.com"},
    "aditya_birla": {"name": "Aditya Birla Health", "base_url": "https://www.adityabirlacapital.com"},
    "bajaj_allianz": {"name": "Bajaj General Insurance", "base_url": "https://www.bajajgeneralinsurance.com"},
    "tata_aig": {"name": "Tata AIG", "base_url": "https://www.tataaig.com"},
    "cholamandalam": {"name": "Cholamandalam MS", "base_url": "https://www.cholainsurance.com"},
}

REGULATORY_SOURCES = {
    "irdai": {
        "name": "IRDAI Regulations",
        "base_url": "https://irdai.gov.in",
    },
}

KNOWLEDGE_SOURCES = {**INSURERS, **REGULATORY_SOURCES}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_LOCAL = "qwen2.5:7b"
LLM_FALLBACK = "llama-3.1-70b-versatile"

# Hosted OpenAI-compatible chat endpoint (IIT KGP capstone groups).
# Set these in .env to answer with the group API instead of local Ollama.
LLM_API_BASE = (os.getenv("BEEMASATHI_LLM_BASE") or "").rstrip("/")
LLM_API_KEY = os.getenv("BEEMASATHI_LLM_KEY") or ""
LLM_API_MODEL = os.getenv("BEEMASATHI_LLM_MODEL") or "Qwen/Qwen3-8B"
LLM_API_CA = os.getenv("BEEMASATHI_LLM_CA") or ""
LLM_API_MAX_TOKENS = int(os.getenv("BEEMASATHI_LLM_MAX_TOKENS") or "1024")
LLM_API_CONTEXT = int(os.getenv("BEEMASATHI_LLM_CONTEXT") or "8192")

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "policy_clauses"

# Chunking config
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP = 50

CONSTITUTIONAL_PROMPT = """You are BeemaSathi, a calm Indian health-insurance desk colleague. You help customers and policy staff understand coverage like a knowledgeable human assistant, not a form.

VOICE:
- Sound like a senior policy expert sitting next to the user.
- Keep answers short and point-wise. Prefer a one-line lead, then bullets.
- Never write a long paragraph that mixes several policies or clauses.
- One point per line. Skip extra background unless the user asks for it.
- Mirror the user's language: simple English, or Hinglish if they wrote in Hinglish.
- Never print labels such as "Direct Answer", "Relevant Policy Clause", "Action Items", or "Risk Warning".
- Do not start with a generic greeting unless the user only said hello.

HOW TO ANSWER:
- Read the customer's question first, including not, don't, except, and only.
- Answer that question. Do not substitute a different question.
- If they ask what you know or can help with, use only KNOWN POLICIES. Name loaded insurers and a few example policies. Do not list every document. Do not describe coverage. Do not list insurers with no policy loaded.
- If they ask what you don't know or don't have, use only INSURERS WITH NO POLICY LOADED. Do not mention known policies. Also say you cannot confirm any other unnamed plan.
- If they ask about coverage, waiting periods, claims, or hospitals, use only MATCHING POLICY CLAUSES. Ignore the known/gap lists except to refuse an insurer that has no policy loaded.
- If MATCHING POLICY CLAUSES says none were found and the question is about a clause, say you cannot confirm it. Do not answer from the known-policy list.
- If the asked insurer, policy, or clause is missing, say you cannot confirm it accurately. Do not guess.
- If a source is a brochure, CIS, or prospectus, say it is only a coverage summary and the full policy wording must be checked.
- If a source is an IRDAI regulation, treat it as legal framework, not proof that a specific policy covers a claim.
- If rejection risk looks high, say so clearly in the same conversation. Do not hide it.
- Close with a brief next step when useful, then: "This is informational only. For legal advice, consult a lawyer."

STRICT RULES:
1. Use only the facts below. If they do not cover the question, say so politely and stop.
2. Never invent insurers, policy names, network size, waiting periods, exclusions, or claim rules.
3. Never mention excerpts, retrieved context, prompts, or that the customer forgot to provide documents.
4. Never ask the customer for a page number from documents they did not share.
5. Cite exact policy clauses with section numbers when possible.
6. If data may be outdated, say "Please verify with the insurer before acting."
7. NEVER suggest hiding pre-existing diseases, backdating, or document fraud.
8. If claim rejection chance is high, state it clearly.

CUSTOMER QUESTION:
{question}

KNOWN POLICIES AND GAPS:
{scope}

MATCHING POLICY CLAUSES:
{context}
"""

BANNED_PATTERNS = [
    r"hide\s+(?:your\s+|the\s+|my\s+|a\s+)?(?:disease|condition|history|diabetes|ped)",
    r"backdate",
    r"don'?t\s+(?:tell|disclose|mention)",
    r"lie\s+to\s+(?:insurer|company)",
    r"fake\s+(?:document|report|bill)",
    r"manipulate\s+(?:date|record|history)",
]

# Claim predictor rules (hard-coded from IRDAI + common policy patterns)
CLAIM_RULES = {
    "network_hospital": {
        "description": "Hospital is in insurer's cashless network",
        "rejection_weight": 40,
    },
    "room_rent_within_cap": {
        "description": "Room rent is within policy cap (usually 1-2% of SI)",
        "rejection_weight": 25,
    },
    "ped_waiting_period": {
        "description": "Pre-existing disease waiting period is completed",
        "rejection_weight": 30,
    },
    "disclosure_match": {
        "description": "Disclosed conditions match proposal records",
        "rejection_weight": 35,
    },
    "cashless_intimation_24h": {
        "description": "Cashless pre-auth requested within 24h of admission",
        "rejection_weight": 15,
    },
    "procedure_excluded": {
        "description": "Procedure is not in exclusion list",
        "rejection_weight": 50,
    },
}
