# BeemaSathi — Project Guide

**An evidence-grounded copilot for Indian health and motor insurance.**
Executive Post Graduate Programme in Generative AI and Agentic AI, IIT Kharagpur — Track B (API-side service with a Streamlit client).

BeemaSathi answers insurance questions only from official insurer and IRDAI documents. It cites the clause and page behind every coverage statement, separates binding policy wordings from marketing brochures, and refuses when a product is not loaded instead of guessing.

This file is the single written reference for the project. The HTML pages under `docs/` present the same material in a browsable form:

| Page | What it covers |
| --- | --- |
| [`index.html`](index.html) | Overview and quick start |
| [`architecture.html`](architecture.html) | Shipped architecture and stage-by-stage pipeline |
| [`data.html`](data.html) | Corpus, governance and extraction limits |
| [`evaluation.html`](evaluation.html) | Gold set, metrics, results and red team |
| [`plan.html`](plan.html) | Four-week plan, gates, risks and deliverables |

Measured figures below are from the working prototype on the local corpus, last evaluated 26 November 2026.

---

## 1. The problem

Indian retail insurance is sold as a short brochure and governed by a long policy wording. The facts a customer needs — waiting periods, room-rent caps, sub-limits, co-pay, pre-existing disease conditions, exclusions, cashless rules — sit in 30–100 page PDFs that change by insurer, product, UIN and revision year. The brochure summarises the product; it is not the contract.

This has measurable consequences:

- A room-rent choice above the cap proportionally reduces every other claim head, not just the room charge.
- Treatment inside a pre-existing disease waiting period is rejected even when the hospitalisation itself is covered.
- Delayed intimation or a non-network hospital turns a cashless claim into reimbursement with a different document set.
- A customer reading a brochure can reasonably conclude a benefit exists that the wording excludes.

General-purpose chatbots make this worse. They answer confidently from generic insurance knowledge, cannot distinguish a brochure from a binding wording, and cannot say "this product is not in my knowledge base". The user receives fluent, unsourced advice about a contract the model has never read.

**The product promise:** every sentence that states coverage must be traceable to a clause, or the system says it does not know.

---

## 2. What the system does

| Capability | Behaviour |
| --- | --- |
| Retrieves from primary sources | Hybrid dense + BM25 retrieval over official insurer and IRDAI documents, filtered by insurer, product and document type |
| Ranks authority, not prose | A policy wording outranks a CIS, prospectus or brochure; summary sources are labelled as summaries in the answer |
| Refuses when evidence is missing | An unloaded insurer or product produces a named gap and a refusal, never an answer borrowed from another insurer |
| Shows its sources | Every answer carries insurer, product, document type, section, page and source file |
| Keeps the recent conversation | The last eight chat messages travel with each question as context — 500 characters each, 2,400 in total — inside the 8,192-token window; chat-history questions are answered from that block, which is never treated as evidence |
| Compares per insurer | A two-insurer question retrieves each plan separately, including two plans of one insurer, and never attributes one insurer's clause to another |
| Answers multi-part questions in a table | A question that packs two or more aspects is split into those aspects, each aspect is retrieved on its own, and the answer is a table with one row per aspect, naming the aspects it was asked about |
| Checks claim-rejection risk | Rule-based assessment of network status, room-rent cap, PED waiting period and intimation timing |
| Blocks unsafe advice | Hiding a pre-existing disease, backdating admission or fabricating bills is refused before the model is called |

---

## 3. Architecture

Retrieval runs first and the agent layer runs on top. Hybrid search finds the clause, an authority resolver drops marketing summaries when binding wording exists, and a critic checks every number against the retrieved text.

### 3.1 Shipped components

| Layer | Implementation |
| --- | --- |
| Collection | Official-domain crawler, robots fail-closed, SHA-256 manifest, sidecar metadata per document |
| Index | ChromaDB, cosine distance, 48,572 chunks from 923 readable source files; pure-Python Okapi BM25 over the same chunks |
| Embeddings | BGE-small-en-v1.5 (384 dimensions) through a hosted OpenAI-compatible `/embeddings` endpoint, or the local sentence-transformers model. Index and query vectors always come from the same model |
| Retrieval | Dense + BM25 + typed clause probes fused with reciprocal-rank fusion (k=60), authority weights, policy-name detection |
| Agent layer | Intent router with typed plans, per-insurer fan-out for comparisons, authority resolver, number/UIN critic with one structured retry |
| Generation | Qwen3-8B via an OpenAI-compatible API, or `qwen2.5:7b` via local Ollama; baseline, zero-shot and structured prompts |
| Guardrails | Pre-model query guard, refusal rules, leak sanitiser, banned fraud patterns, post-model number verification |
| Observability | Per-request JSONL traces: hashed question, redacted preview, plan, sources, tokens, latency |
| Claim risk | Rule engine, 4 of 6 configured rules implemented |
| Interfaces | Streamlit client; FastAPI service with `/ask`, `/plan`, `/predict`, `/insurers`, `/health` and a token-bucket rate limit |
| Delivery | Dockerfile, compose for API + UI, GitHub Actions CI (compile all modules, run unit tests) |

### 3.2 One request, end to end

1. **Guard** — `rag/query_guard.py` normalises Unicode (zero-width, homoglyph, spaced-letter, leetspeak) and refuses prompt/secret/PII extraction, fraud advice and instruction override without calling the model.
2. **Route** — `agents/router.py` turns the question into a typed plan: intent, insurers, products, line of business, vehicle class, list limit and ordered steps.
3. **Branch** — `agents/orchestrator.py` runs exactly one bounded path for that intent: refusal, catalogue listing, recommendation, claim risk, comparison fan-out, or the coverage path.
4. **Retrieve** — `rag/retriever.py` fuses dense (Chroma) and lexical (BM25) candidates, applies typed clause probes, prefers the requested line of business, and reranks by document authority plus policy-name agreement. A question that asks about two or more aspects is retrieved once per aspect, so the second aspect is not crowded out by the first.
5. **Resolve authority** — `agents/critic.py` drops brochure/CIS/prospectus chunks when binding wording for the same policy is present (unless the user explicitly asked about the brochure).
6. **Generate** — `rag/generator.py` builds a prompt from numbered clauses, the indexed catalogue and the recent conversation, then calls the hosted or local model. When the question has several parts, the prompt names each asked aspect and requires a table with one row per aspect.
7. **Verify** — every number, unit and UIN in the answer must appear in the retrieved clauses; one structured retry is allowed, then the answer carries an explicit "could not verify" note. Unsupported family-size claims are stripped.
8. **Trace** — the request is recorded as JSONL with a hashed question and redacted preview; no prompt text, clause text or credentials are stored.

### 3.3 The agent layer

The `agents/` package contains four modules. None of them is an LLM-to-LLM agent: there is exactly one model call per answer, including a comparison, and everything else is deterministic Python. Routing, scoping and fact-checking are the parts that must be reproducible.

| File | Responsibility | Model calls |
| --- | --- | --- |
| `router.py` | Classifies the question into a `TaskPlan` before retrieval: intent, insurers, products, line, vehicle class, list limit, ordered steps | 0 |
| `orchestrator.py` | Executes the plan for that intent, retrieves each plan for a comparison, applies verification and tracing | 1 |
| `critic.py` | Post-generation evidence checks: number/UIN verification, family-limit stripping, authority resolution | 0 (can trigger 1 retry) |
| `claim_predictor.py` | Scores claim-rejection risk from form inputs using weighted rules | 0 |

**Router.** Intents are `safety`, `security`, `scope`, `catalog_list`, `claim_risk`, `comparison`, `recommend` and `coverage`. Insurer names are resolved through an alias table; product names are matched per insurer so that a plan named for one insurer cannot be pulled into another insurer's section. Safety and security questions never reach retrieval.

**Orchestrator.** Every intent gets a small, purpose-built branch instead of one generic prompt: a catalogue answer is computed from the index (no model), a recommendation explicitly refuses to rank plans, a claim question uses the rule engine, a comparison retrieves each named plan separately — including two plans of the same insurer — and then makes one structured call that extracts constraints per plan, keeps only the rows where they differ, and answers with a table, and a coverage question runs the retrieve → authority → generate → verify path, splitting a multi-part question into one retrieval per aspect and answering in a table. The orchestrator is the only entry point used by the UI and the API.

**Critic.** Three independent checks: `resolve_authority()` removes marketing summaries when a wording exists for the same policy; `verify_answer()` canonicalises units ("3 years" = "36 months") and requires every number and UIN in the answer to appear in the retrieved clauses; `strip_unbacked_family_limits()` removes a family size the model echoed from the question when the wording never set one.

**Claim predictor.** Four weighted checks — out-of-network hospital, room rent above 1% of sum insured, PED within the 36-month wait, cashless intimation past 24 hours — summed into a 0–100 score with Low/Medium/High/Very High bands. It also returns a document checklist and one action per failed check. It is a deterministic heuristic, not a reading of the specific policy wording, and the UI presents it as a separate tool rather than as a coverage answer.

### 3.4 Build-time pipeline

| Stage | Technique | Module |
| --- | --- | --- |
| Discover | Bounded crawl of insurer-owned hosts, robots fail-closed, newest revision wins, files saved for review and never auto-ingested | `scrapers/hybrid_scraper.py` |
| Extract | Page-level PDF text with null bytes, repeated headings and navigation labels stripped; one-based page numbers kept | `rag/ingest_pdf.py` |
| Chunk | Bounded 1,600-character windows inside one page, two trailing lines of overlap, fragments under 80 characters dropped | `rag/ingest_pdf.py` |
| Label | Keyword rules assign one of 11 clause types (room rent, waiting period, co-pay, exclusion, cashless, claim process, coverage, renewal, premium, maternity, day care) | `rag/chunker.py` |
| Embed | Hosted `/embeddings` calls, 16 texts per request, inputs capped at 512 tokens with a proportional re-cut on HTTP 422, retried on capacity errors, vectors L2-normalised | `rag/embeddings.py` |
| Store | Cosine HNSW collection; re-ingesting a source replaces that file's chunks instead of duplicating or colliding with another document | `rag/embedder.py` |
| Lexical index | Okapi BM25 (k1=1.5, b=0.75) rebuilt from `data/chunks/*.json`, with insurer, product and document type indexed as a header | `rag/bm25.py` |
| Ledger | SHA-256 change detection, dry-run planning, sidecar metadata, filename and folder inference | `rag/sync_knowledge.py` |

---

## 4. Data

### 4.1 Sources

| Source | Content | Access |
| --- | --- | --- |
| Insurer document hubs | Policy wordings, CIS, prospectus, brochures, endorsements | Official-domain crawler; 8 insurers represented |
| Manual acquisition | Star Health, Care Health, ICICI Lombard, Acko, Zurich Kotak | Browser download with human review where crawling is blocked |
| IRDAI | Regulations, circulars, master circulars | Manual; `robots.txt` disallows crawling |
| User-uploaded policy | A customer's own wording and schedule | Local only, never shared |

Only insurer-owned or regulator domains are accepted. Third-party policy mirrors are rejected even when they are easier to crawl.

### 4.2 What is indexed today

| Measure | Value |
| --- | --- |
| Source files indexed | 923 |
| Clause chunks in Chroma | 48,572 |
| PDFs collected | 924 (1.1 GB) |
| Policy wordings | 337 |
| Brochures, CIS and prospectus | 558 |
| Endorsements | 26 |
| IRDAI regulations | 2 |
| Insurers | 8 plus IRDAI |

Composition matters. Summaries are the majority of the corpus, so default similarity search tends to surface them. Document type is therefore treated as a hard authority signal for coverage questions, not a tiebreaker.

### 4.3 Governance

- A document without provenance cannot enter the collection. Each chunk carries `source_rel`, `source_file`, `chunk_id`, `page_start`/`page_end`, `section_path`, `element_type`, `document_type`, `line_of_business`, `token_estimate` and `injection_flags` under schema v2.
- A chunk is quoted only when its source file is in the ingest ledger or is an official IRDAI regulation on disk. Demo fixtures and orphan chunks are dropped before ranking.
- Each sync records SHA-256, chunk count and indexing time in the ingest ledger; re-ingesting a source replaces that file's chunks rather than duplicating another document's.
- Revision is resolved by date and revision marker; duplicate manual and scraped copies are resolved per file, not per product key.

Target record shape:

```json
{
  "document_id": "sha256:9f2c…",
  "insurer": "tata_aig",
  "product_key": "TATHLIP21292V022021",
  "line_of_business": "health",
  "document_type": "policy_wording",
  "revision_date": "2024-03-01",
  "source_url": "https://www.tataaig.com/…",
  "sha256": "9f2c…",
  "pages": 26,
  "ocr_derived": false,
  "review": { "status": "approved", "reviewer": "member-1" }
}
```

### 4.4 Data layout and exclusions

The PDFs are third-party material with no redistribution licence, so they are **not** in this repository. What is committed is the pipeline output needed to inspect and re-score the corpus without re-downloading it.

| Path | Committed | Content |
| --- | --- | --- |
| `data/parsed/` | Yes | Per-document extracted text, one file per PDF |
| `data/chunks/` | Yes | Chunk JSON per document, including page and section metadata |
| `data/chunks/ingest_index.json` | Yes | Ingest ledger: source key, file, document type, chunk count |
| `eval/reports/` | Yes | Dated evaluation reports and the latest run |
| `data/raw_policies/` | No | Insurer PDFs, ~1.1 GB, no licence |
| `data/irdai/` | No | IRDAI regulation PDFs; download from irdai.gov.in |
| `data/chroma_db/` | No | Vector store, ~1 GB, derived |
| `venv/`, `logs/` | No | Local environment and runtime output |

Every downloaded document has a sidecar `.meta.json` recording insurer, product, document type, source URL and SHA-256. The manifest is the authority for where a clause came from. Questions about a specific clause should be resolved against the original PDF and its UIN, not against extracted text in this repository.

### 4.5 Known extraction limits

- Page-sliced chunking cuts some clauses at page boundaries.
- Table-aware extraction exists behind `BEEMASATHI_TABLE_EXTRACT=1` and 39 table chunks are indexed, but many tables still arrive as flat text, so waiting-period and sub-limit data can be incomplete.
- Six image-only PDFs fail extraction on every run; OCR fallback is not implemented yet.
- 57% of chunks carry only a page label, so section-level citation is often approximate.

---

## 5. Retrieval and generation

**Hybrid retrieval.** Dense search over BGE-small embeddings and a pure-Python Okapi BM25 index run over the same chunks. Candidate lists — dense, lexical, and typed clause probes — are fused with reciprocal-rank fusion at k=60, deduplicated, filtered to the requested line of business when one is named, and then reranked.

**Authority weights.** Wording 1.0, endorsement 0.9, CIS 0.6, prospectus 0.5, regulation 0.4, brochure 0.3. The resolver additionally drops summaries entirely when a binding wording for the same policy is in the result set.

**Policy matching.** Product names are normalised and matched per insurer. The router also splits a comparison question at insurer mentions so a plan named next to one insurer is not attributed to another.

**Generation.** The prompt is built from numbered clauses, a compact catalogue of what is loaded, and a `RECENT CONVERSATION` block: the last eight messages of the Streamlit session, cut to 500 characters each and 2,400 characters in total, about 800 of the 8,192-token window. The block resolves follow-ups and questions about the chat itself, but it is labelled context, not evidence — coverage facts still come only from the clauses. Three prompt strategies are available: `baseline`, `zero_shot` and `structured`. If no clause passes the relevance threshold, the generator returns a knowledge-gap answer instead of calling the model with weak context, except for chat-history questions such as "what was my first question?", which skip that refusal and answer from the history block.

**Refusal.** A missing insurer, product or clause produces a named gap and a suggestion to check the insurer's official document. A summary-only product is answered with an explicit note that the source is a brochure, not the contract.

---

## 6. Guardrails and robustness

**Pre-model guard.** `rag/query_guard.py` runs before retrieval. It normalises Unicode and refuses:

- system-prompt, API-key, secret and PII extraction;
- fraud advice (hiding a condition, backdating, editing bills);
- role-play that tries to grant claim approval or coaching;
- instruction override and document-borne injection;
- attempts to make the system claim fabricated day-one cover.

**In-prompt and post-model checks.** The system prompt fixes the assistant's role, requires clause-level answers, and forbids generic insurance knowledge. After generation, a leak sanitiser removes model/system text and clause-echo artefacts — the `[i] insurer | policy | page` scaffold, "Deciding Clause"/"Answer:" labels and long verbatim quotes — so the customer sees plain prose, and `verify_answer()` blocks unsupported numbers and UINs.

**Uploaded documents.** User uploads are treated as untrusted text, delimited in the prompt, and scanned for injection instructions. They are stored locally and never shared.

**Red team.** Eighteen probes run today: five safety prompts, four knowledge-gap probes (Star Health, Acko, Care Health, ICICI Lombard), and a nine-case guardrail set covering prompt extraction, secret extraction, PII extraction, role-play approval, role-play fraud coaching, document-borne injection, index poisoning and instruction override. All nine security probes are refused by a deterministic pattern check before retrieval or the model, so no document text can steer them.

**Traces.** Each request writes a JSONL record with a hashed question, redacted preview, plan, sources, tokens and latency. Prompts, clause text and credentials are never stored. `venv/bin/python observability/dashboard.py` renders the log to `docs/trace_dashboard.html` (requests, p50/p95 latency, intent mix, refusals, verification pass rate, most-cited plans, recent requests).

### 6.1 Model card

**Intended use.** An informational desk assistant for Indian retail health and motor insurance. A customer asks about a named plan; BeemaSathi quotes the clause, section and page from the loaded wording, or says the product is not indexed. It is built for reading and comparison, not for selling or adjudicating.

**Who uses it.** Insurance desk staff, brokers and policyholders comparing what a wording actually says. Every answer is framed as informational only.

**Out of scope.** No price ranking, no recommendation of one insurer over another, no claim filing, no legal opinion, no advice on hiding facts, backdating cover or editing documents. Questions outside Indian health and motor insurance are refused before retrieval.

**Models and components.** Retrieval is deterministic: BGE-small-en-v1.5 embeddings, Chroma plus Okapi BM25, reciprocal-rank fusion, authority-aware reranking. Generation is one call to Qwen3-8B (OpenAI-compatible endpoint) or `qwen2.5:7b` via local Ollama; there is no chain of model calls. Routing, scope checks, claim-risk scoring and number/UIN verification are deterministic Python.

**Data.** Official insurer and IRDAI documents only, with a SHA-256 manifest and per-document metadata. The repository does not redistribute insurer PDFs; the index is rebuilt locally. Traces store a question hash, a redacted preview and request metadata, never prompts, clause text or credentials.

**Failure modes.**

| Failure | What the user sees | Mitigation |
| --- | --- | --- |
| Product not indexed | A named knowledge-gap refusal, never a borrowed answer | Catalogue check before retrieval |
| Brochure mistaken for wording | A summary warning or a wording-first answer | Authority resolver drops summaries when wording exists |
| Wrong plan attributed across insurers | Plans are matched per insurer mention | Per-insurer segmentation in the router |
| Flat-table extraction (waiting periods, sub-limits) | Coarse or "not stated" cells | Table-aware chunking where enabled; limitation documented |
| Unsupported number or UIN | Answer replaced or flagged unverified | `verify_answer()` with one structured retry |
| Prompt injection in an uploaded PDF | Refusal before retrieval or generation | Delimited untrusted text, injection scan, red-team probes |
| Model API unavailable | Explicit error with the check to run | Local Ollama fallback when it is reachable |

**Refusals.** The pre-model guard refuses secret/PII/system-prompt extraction, fraud advice, role-play that grants claim approval, instruction override, document-borne injection and fabricated day-one cover. Refusals are deterministic and never reach retrieval or the model.

**Adversarial tests.** Nine security probes — prompt, secret and PII extraction; role-play claim approval and fraud coaching; document-borne injection; index poisoning; instruction override; fabricated day-one cover — are refused 9/9 before retrieval. Four knowledge-gap probes confirm that unloaded insurers (Star Health, Acko, Care Health, ICICI Lombard) produce a clean refusal. The suite runs in CI.

**Limitations and monitoring.** The claim-risk score is a heuristic, not an insurer decision. Tables in scanned PDFs can extract imperfectly. Answers are only as current as the loaded revisions; UIN and revision dates are carried in metadata for audit. The trace dashboard is the monitoring surface: refusal rate, verification pass rate and latency are visible per day.

---

## 7. Evaluation

Accuracy alone rewards a model that never says "I don't know". This evaluation measures retrieval, grounding and refusal quality separately, on 100 gold cases whose expected facts are checked against indexed chunks and source policy PDFs.

### 7.1 Where the baseline stands

| Metric | Result | Target |
| --- | --- | --- |
| Gold cases | 100 | — |
| Answer pass rate (rules + judge) | 83/100 | ≥ 0.90 |
| Grounded pass rate | 83/100 | ≥ 0.85 |
| Document recall@8 | 0.975 (79/81) | ≥ 0.85 |
| Supporting-clause recall@8 | 0.912 (73/80) | ≥ 0.70 |
| Mean rank of the evidence chunk | 2.2 | ≤ 3 |
| Mean answer latency | 2.5 s | p95 ≤ 20 s |
| Prompt / completion tokens (100-case run) | 309k / 8.1k | — |
| Red-team refusals | 9/9 | 9/9 |

Hybrid retrieval finds the expected policy in 79 of 81 labelled cases and the exact supporting clause in 73 of 80. The latest frozen run (`eval/reports/eval_20261126T173042Z.json`) passes 83 cases; the remaining 17 are 11 answer errors, 4 evidence misses and 2 document misses, and all 83 passing answers had the expected document in context, so no answer was right by luck.

### 7.2 Failure taxonomy

The 17 failures split into 2 document misses, 4 supporting-clause misses and 11 answer errors. The recurring shape is picking the wrong sub-clause or the wrong number (24 months where the wording says 36, INR 10,000 where it says Rs 2,500). Fifteen of the 17 trip a hard rule while the LLM judge accepts 14 of them. In some the rule is right and the judge is too lenient; in others the rule string is too narrow — one case bans "60 days" even though 60 days is the correct post-hospitalisation window. Read 83/100 as conservative, not as a rule-versus-judge tie.

### 7.3 Gold set composition

| Category | Count | What it tests |
| --- | --- | --- |
| Waiting periods (PED, specified, initial) | 18 | Clause extraction across seven product lines |
| UIN | 13 | Exact identifier from the wording, no invented numbers |
| Co-payment | 10 | 5% vs 10% confusion, add-on and dental exclusion |
| Grace period, room rent, deductible | 9 | Small-print numbers that are easy to misread |
| Other core facts | 3 | Tata AIG PED exclusion, Digit liability-only, Bajaj 24-month wait |
| Gap and refusal cases | 4 | Star Health, Acko, Care Health, ICICI Lombard |
| Safety and fraud prompts | 5 | Hiding PED or medication, backdating, edited documents |
| Brochure, scope, leakage | 3 | Brochure honesty, catalog gaps, no excerpt leakage |
| Customer-style health and motor questions | 26 | HDFC Arogya, Digit Group Complete Protect, Digit private car, Bajaj two-wheeler benefits and claim conditions |
| Red-team injections | 9 | Prompt, secret and PII extraction, role-play, document injection, index poisoning |

Eighteen waiting-period, twelve UIN and ten co-pay cases were mined from indexed policy wording and machine-checked: the expected document must exist, and the exact gold fact must appear in a chunk from that document. The rest are hand-written. This is a single-annotator set; double annotation is still planned.

### 7.4 Systems compared and ablations

- Single-prompt LLM with no retrieval — not measured.
- Dense-only RAG — earlier 74-case baseline: document recall@8 of 0.60, clause recall 0.28.
- Hybrid RAG with BM25 plus dense — document recall@8 of 0.975, clause recall 0.912.
- Hybrid RAG with authority-aware reranking — the shipped retriever.
- Agentic routing with the critic loop — the shipped orchestrator; answer pass rate 83/100.
- Prompt strategies (baseline, zero-shot, structured) compared on identical retrieved context with `eval/compare_prompts.py`: 64/74, 60/74 and 63/74 rules-only passes; mean latency 2.36 s, 1.06 s and 2.31 s; 187k, 150k and 157k prompt tokens.

Planned ablations: authority-aware reranking on/off (brochure contamination), critic on/off (unsupported-claim rate), per-insurer fan-out on/off (comparison quality), calibrated distance threshold on/off (weak-evidence answers).

### 7.5 How cases are judged

Every case combines hard rule checks with an LLM judge, and the rules override the judge on refusal and safety. The judge sees only the gold notes and retrieved clauses; it is not allowed to apply extra insurance knowledge. The judge currently runs on the same model as the generator, so hard rules catch what a generous judge would miss. Human audit of a verdict sample is still planned.

---

## 8. Engineering review and what changed

An internal review on 14 November 2026 read every source file and probed the live index. It found four corpus-integrity defects that made coverage answers untrustworthy, plus a set of correctness and product gaps. The defects and their remediation:

| # | Finding | Resolution |
| --- | --- | --- |
| P0-1 | Hand-written demo clauses for Star Health were retrievable in production | Quoting now requires a ledgered source file or an official regulation; orphan chunks are dropped before ranking, so the demo source cannot be retrieved or catalogued |
| P0-2 | Vector keys were `insurer_policy_doctype`, so documents sharing a key overwrote each other | Replacement is scoped to a single source file (`source_rel`/`source_file`); the corpus was rebuilt with content-addressed chunk records |
| P0-3 | The catalogue came from an ingest ledger that had drifted from Chroma | The catalogue is now derived from Chroma, with the ledger merged only for sources indexed under another key |
| P0-4 | Document-type mapping collapsed CIS and prospectus into "brochure" and typed endorsements as wordings | Types are carried into metadata and ranked separately (wording, endorsement, CIS, prospectus, brochure, regulation) |

Additional fixes: per-insurer retrieval for comparisons followed by one differences-only table; per-insurer product matching; brochure demotion in the authority resolver; an alias-based insurer detector with word boundaries; scope questions separated from coverage questions; the claim predictor connected to the chat flow and its output cleaned up; hosted embedding support with the same BGE model; a pre-model query guard; schema v2 chunk metadata; and a 100-case gold set with retrieval metrics and cost.

Still open, and stated honestly:

- The relevance threshold is still permissive (0.72) relative to genuine match distances of 0.14–0.30.
- Retrieval is not conversation-aware. The model receives the last eight turns, but follow-ups still retrieve on their bare text, so a pronoun-only question can miss the right clause.
- Two configured claim rules (`disclosure_match`, `procedure_excluded`) are not implemented, and the score is a generic heuristic rather than a policy-specific one.
- OCR is not implemented; six image-only PDFs fail extraction.
- Table extraction is partial (39 table chunks).
- The judge is the same model as the generator, on a single-annotator gold set.
- The planned ablations and human audit have not run.

---

## 9. Running locally

### 9.1 Prerequisites

- Python 3.11+
- One model backend: a hosted OpenAI-compatible endpoint, or Ollama with `qwen2.5:7b`
- The corpus and index (`data/raw_policies/`, `data/chroma_db/`) are not committed; rebuild them with the steps in §9.3

### 9.2 Setup

`./beemasathi.sh` performs the whole setup: it checks Python, creates the venv,
installs what is missing, and offers a menu to save the hosted endpoint in
`.env` and to start or stop the app. The menu is implemented in `launcher.py`
so it behaves the same everywhere: `beemasathi.sh` is the macOS/Linux entry
point, `Start BeemaSathi.command` runs it from Finder, and
`Start BeemaSathi.bat` is the double-click entry point on Windows. On Windows
the launcher uses `venv\Scripts\python.exe`; on macOS/Linux it uses
`venv/bin/python`. The equivalent manual steps:

```bash
git clone https://github.com/satishify/beemasathi.git
cd beemasathi
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in the hosted base URL and key
```

```ini
BEEMASATHI_LLM_BASE=https://<host>/v1
BEEMASATHI_LLM_KEY=<key>
BEEMASATHI_LLM_MODEL=Qwen/Qwen3-8B
BEEMASATHI_LLM_CA=/absolute/path/to/ca.crt

BEEMASATHI_EMBED_BASE=      # hosted /embeddings endpoint
BEEMASATHI_EMBED_KEY=<key>
BEEMASATHI_EMBED_MODEL=bge-small-en-v1.5
BEEMASATHI_EMBED_CA=/absolute/path/to/ca.crt
```

Without `.env`, the app falls back to local Ollama (`ollama pull qwen2.5:7b`) and to the local `BAAI/bge-small-en-v1.5` model. With `.env` configured, the hosted endpoint is tried first; if it is unreachable, rejects the key, or fails mid-request, generation retries on local Ollama when its server is running, and the metrics record `fallback: hosted-unavailable`. Index vectors and query vectors must come from the same model, so do not mix hosted and local embedding backends against one Chroma collection. Relative CA paths in `.env` (for example `certs/vm06-ca.crt`) are resolved against the project root, so they work from any working directory and on Windows.

### 9.3 Corpus and index

```bash
./venv/bin/python scrapers/hybrid_scraper.py --dry-run   # preview discovery
./venv/bin/python scrapers/hybrid_scraper.py             # download official PDFs
./appctl.sh index                                        # parse, chunk, embed
./beemasathi.sh start                                    # http://127.0.0.1:8501
```

Some insurer sites block automated collection; the scraper prints a manual fallback guide (`--manual-guide`). A rebuild reproduces the pipeline, not byte-identical contents. On Windows, replace `./venv/bin/python` with `venv\Scripts\python.exe` and run the index step with `python launcher.py index`.

### 9.4 Run

```bash
./beemasathi.sh start       # launcher: background Streamlit on :8501
streamlit run ui/app.py     # manual alternative, Streamlit UI
python api/main.py          # FastAPI backend on :8000
docker compose up --build   # API + UI, mounts ./data read-only
```

On Windows, start with `Start BeemaSathi.bat start` or `py launcher.py start`. API endpoints: `POST /ask`, `POST /plan`, `POST /predict`, `GET /insurers`, `GET /health`. Requests are limited by a per-client token bucket.

Debug diagnostics are opt-in: `BEEMASATHI_DEBUG=1 streamlit run ui/app.py`.

The sidebar's "Insurer or regulator" list offers `Any` plus only insurers whose wordings are indexed. Configured insurers without documents — today Star Health, ICICI Lombard, Care Health and Acko — are named in a muted "Not indexed yet — coming soon" caption below the dropdown and cannot be selected, so the UI matches what the corpus can answer.

### 9.5 Evaluation

```bash
python eval/judge.py                # retrieval + answer + judge, writes eval/reports/
python eval/retrieval_eval.py       # retrieval only
python eval/compare_prompts.py      # baseline vs zero_shot vs structured
python eval/mine_gold_cases.py      # reproducible miner for new fact cases
```

---

## 10. Repository layout

```text
BeemaSathi/
├── agents/            # router, orchestrator, critic, claim predictor
├── api/               # FastAPI service and rate limiter
├── ui/                # Streamlit client
├── .streamlit/        # theme: palette, rounded surfaces, Inter and Plus Jakarta typography
├── assets/            # brand mark used by the sidebar logo
├── rag/               # crawler support, PDF extraction, chunking, embeddings,
│                      # Chroma store, BM25, retriever, generator, query guard
├── observability/     # redacted JSONL traces
├── eval/              # gold set, judge, miners, prompt comparison, reports
├── tests/             # unit tests without heavy model dependencies
├── scrapers/          # insurer-specific discovery and download
├── docs/              # this guide and the HTML project pages
├── data/              # pipeline output (PDFs and vector store excluded)
├── config.py          # central configuration
├── launcher.py        # cross-platform launcher: check, config, start and stop
├── beemasathi.sh      # macOS/Linux entry point for launcher.py
├── Start BeemaSathi.bat   # Windows entry point for launcher.py
├── Start BeemaSathi.command  # macOS Finder entry point
├── appctl.sh          # compatibility wrapper: start/stop/status/index
├── Dockerfile         # API image
└── docker-compose.yml # API + UI
```

---

## 11. Project plan

**Four weeks, six members.** Week 1 is deliberately unglamorous: until documents have unique identities, the catalogue matches the vector store, and a frozen question set exists, every later number measures the wrong thing.

| Week | Workstream | Gate |
| --- | --- | --- |
| 1 | Corpus integrity and evaluation freeze: content-addressed document IDs and reindex, demo data purged, document-type mapping fixed, catalogue reconciled, IRDAI scan root added, frozen 100-question set, baselines measured | G1 — zero chunks without provenance, zero collisions, catalogue equals vector store, 100 questions versioned |
| 2 | Hybrid retrieval and evidence quality: BM25 plus dense, authority-aware reranking, calibrated threshold, clause- and table-aware chunking, UIN and effective date in metadata, citation contract | G2 — recall@5 ≥ 0.80 and wording share ≥ 0.85 |
| 3 | Agentic layer: router, retrieval, authority resolver, clause analyst, claim-risk and critic agents; typed tool schemas; retries and budgets; trace store; agentic versus RAG-only comparison | G3 — task completion ≥ 95%, tool success ≥ 98%, critic catches injected instructions |
| 4 | Evaluation, hardening and delivery: full evaluation and ablations, failure taxonomy, red-team report, observability dashboard, API and UI integration, deployment pack, report and demo | G4 — targets met or documented with a remediation plan |

**Team roles:** domain specification and gold-set annotation; prompting, critic design, safety and red team; data acquisition and hybrid retrieval; orchestration, tracing and claim risk; evaluation, metrics and ablations; API, UI and packaging.

**Risk and mitigation:**

| Risk | Mitigation |
| --- | --- |
| Insurer sites block automated access or change URLs | Official-domain crawler plus documented manual acquisition; no third-party mirrors |
| Policy revisions silently supersede indexed documents | Revision date and UIN in metadata; scheduled re-crawl with diff alerts |
| Scanned or table-heavy PDFs extract poorly | OCR fallback with an `ocr_derived` flag; table-aware extraction; per-document quality metric |
| Brochure text contaminates coverage answers | Authority-aware reranking, wording-first policy, brochure share measured in evaluation |
| Hallucinated coverage or UIN | Refusal-first prompt, number/UIN critic, citation-required answers |
| LLM judge is generous or biased | Rules override the judge; human audit of a sample; disagreement reported |
| Prompt injection through uploaded PDFs | Delimited untrusted content, injection scan, critic check, red-team probes |
| Scope creep into legal or claim adjudication | Explicit non-goals, informational-only framing, no claim filing |

**Deliverables:** working FastAPI + Streamlit application; reproducible repository with tests and a one-command start; documented knowledge base with manifest and hashes; trace dashboard; evaluation report with frozen questions and failure taxonomy; responsible-AI and injection report; technical report and a ten-minute demonstration.

**Demo success criteria:** a coverage question returns a clause-level citation with insurer, product, document type, page and source; an insurer with no loaded wording returns a clean refusal; a brochure-only product is answered with a summary warning; a two-insurer comparison returns evidence from both; a claim scenario returns per-rule findings or explicitly labelled defaults; a poisoned upload cannot change behaviour; the trace reproduces every step; the evaluation report shows the frozen results.

---

## 12. Distribution and submission
The repository does not ship the 1.1 GB corpus or the ~1 GB vector store. What is committed is the code, tests, documentation, gold set and dated evaluation reports; the index is rebuilt locally with `./appctl.sh index` (§9.3). Insurer PDFs are third-party material with no redistribution licence and are never published.

For this submission only the technical report PDF is uploaded. It links the public repository, the trace dashboard and the ten-minute demo. A local run reproduces the evaluation in §7 by re-running `eval/judge.py` against a rebuilt index.

**Non-goals.** BeemaSathi is informational. It does not sell insurance, rank products by price, adjudicate claims, or provide legal advice. It does not scrape third-party mirrors, redistribute insurer PDFs, or claim to know products that are not indexed. Claim-risk output is a heuristic and must not be treated as an insurer decision.

---

*BeemaSathi · informational only, not legal or financial advice.*
