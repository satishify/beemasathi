# BeemaSathi — Architecture Review and Loose Ends

**Review date:** 14 November 2026
**Scope:** Full repository (RAG pipeline, scraper, ingestion, UI, API, claim predictor, eval), plus a live inspection of the indexed corpus and retrieval behaviour.
**Method:** Static read of every source file, plus runtime probes against the existing `data/chroma_db` (45,844 chunks) and `data/chunks/ingest_index.json` (923 files). Items marked **[verified]** were reproduced during this review; **[code]** items are from reading the code. No files were modified except this document.

---

## TL;DR

BeemaSathi is a well-intentioned and unusually careful prototype — official-domain-only crawling, robots fail-closed, brochure disclaimers, refusal-first prompting, an LLM-as-judge eval. The *engineering instincts* are good. The *system as it stands cannot be trusted with a coverage question*, for four reasons:

1. **A fabricated policy is live in the production index.** Six seed chunks for `star_health / comprehensive` ("room rent capped at 1% of SI", "20% co-pay") are still in Chroma, while the same system tells the model that Star Health has no policy loaded. Asking "What is the room rent cap in Star Health?" retrieves the fake policy at distance 0.265 and will answer from it. **[verified]**
2. **The corpus in Chroma and the corpus the system describes are not the same corpus.** Vector IDs are keyed on `insurer_policy_doctype`; 923 files collapse into 768 keys, and 123 groups overwrite each other during `replace_source_chunks` — about 3,043 of 48,611 chunks were deleted by later ingests. In the other direction, 276 chunks under 6 keys exist only in Chroma: the real IRDAI regulations, plus older ingests of three products, plus the seed policy. The catalogue, the sidebar count, and the prompt's "known policies" list are built from the ledger, so they list documents that are no longer in Chroma and hide the ones that are. **[verified]**
3. **The retrieval layer does not enforce the product's core promise** (answer only from the actual policy wording). There is no stable document identity, `policy_name` filtering is exact-match and silently falls back to unfiltered search, and brochures outrank wordings on many coverage questions (5 of 6 chunks for a PED-exclusion question, 4 of 6 for cashless and ambulance questions). **[verified]**
4. **The two halves of the product are not connected.** The claim predictor is a separate hard-coded screen; chat never computes or reports rejection risk (`rejection_risk` is always `None`), even though the prompt and README advertise it.

| Area | Grade | One-line reason |
| --- | --- | --- |
| Crawl / collection | B | Genuinely careful (official hosts, robots, hashes), but non-incremental and overwrite-prone |
| Ingestion / chunking | D | No stable doc identity, page-sliced text, tables and section structure lost, 6 PDFs fail every run |
| Vector store / catalogue | D+ | Two sources of truth disagree; seed data left in production |
| Retrieval | C- | Dense-only (README says hybrid), brochures not demoted, filters silently dropped |
| Answer grounding / safety | C | Good prompt and refusal rules; defeated by bad metadata and loose threshold |
| Claim predictor | D | Hard-coded health assumptions, 2 of 6 rules unimplemented, disconnected from RAG |
| UI | B- | Clean and honest in tone; upload is a placeholder, no multi-turn, no policy dropdown |
| API | C | Works, but unauthenticated on `0.0.0.0`, unvalidated inputs, no contract tests |
| Evaluation | C | 11 cases, same model judges itself, no retrieval or predictor tests, not a gate |
| Ops / reproducibility | D | Unpinned deps, unused deps, no git/CI/tests, 1.9 GB data in tree, docs stale |

---

## 1. What the product is trying to be

From `README.md` and the UI copy: a calm Indian insurance desk colleague that

- answers coverage questions **only** from official insurer/IRDAI documents,
- names what it does *not* have instead of guessing,
- flags claim-rejection risk before a claim is filed,
- warns when a source is a brochure/CIS rather than binding wording,
- keeps everything local (Ollama + BGE + Chroma) for privacy.

That is a defensible, differentiated product. The review below is measured against *that* promise, not against a generic chatbot.

---

## 2. Architecture as built

### 2.1 System context

```text
                    ┌──────────────────────────────────────────────┐
                    │  Official insurer / IRDAI websites           │
                    └───────────────┬──────────────────────────────┘
                                    │  (robots.gov, official hosts only)
                                    ▼
                       scrapers/hybrid_scraper.py
                       hubs + sitemaps → classify → newest revision
                                    │
                                    ▼
        data/raw_policies/{insurer}/*.pdf  +  *.meta.json (sidecar)
        data/raw_policies/manifest.json, discovery_report.json
                                    │
                                    ▼
                       rag/sync_knowledge.py  (./appctl.sh index)
                       infer metadata → ingest_pdf / ingest_text
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   data/parsed/*.txt                           data/chunks/*.json
   data/chunks/ingest_index.json (ledger)      (per-document chunks)
                                    │
                                    ▼
                  PolicyVectorStore  (ChromaDB, BGE-small-en-v1.5)
                  collection "policy_clauses", cosine, 45,844 chunks
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     PolicyRetriever          BimaSaathiGenerator      ClaimPredictor
     (dense + heuristics)     (prompt + guardrails)    (hard-coded rules)
              │                     │                     │
              └──────────┬──────────┴──────────┬──────────┘
                         ▼                     ▼
                 ui/app.py (Streamlit)    api/main.py (FastAPI)
                         │                     │
                         └──────► eval/judge.py (offline, LLM judge)
```

### 2.2 Components and responsibilities

| Component | File | Responsibility | Key coupling |
| --- | --- | --- | --- |
| Scraper | `scrapers/hybrid_scraper.py` (1003 lines) | 16-insurer registry, hub/sitemap discovery, classification, revision selection, validated download, sidecar + manifest | Writes folder/file names that become the *only* identity for downstream metadata |
| Metadata inference | `rag/sync_knowledge.py` | SHA-256 ledger, filename/folder/sidecar parsing → insurer, policy, doc type, line | Reimplements insurer aliases separately from `config.py` and the scraper registry |
| PDF ingestion | `rag/ingest_pdf.py` | pdfplumber per-page extraction, 1600-char slices, page-level metadata, chunk persistence, Chroma replacement | `source` key is `insurer_policy_doctype` |
| Text ingestion (legacy) | `rag/chunker.py` | Regex section splitting for `.txt`, writes `all_policies.json` | Different metadata schema, `source` excludes doc type |
| Vector store | `rag/embedder.py` | Chroma client, sentence-transformers embedding function, upsert/replace/search, catalogue | Catalogue reads the **ledger**, not Chroma |
| Retriever | `rag/retriever.py` | Query → chunk-type inference → dense search (pools of 16–24) → dedupe → heuristic rerank | Filters are exact-match; rerank is distance + small bonuses |
| Generator | `rag/generator.py` | Guardrails, insurer/policy detection, context assembly, prompt, Ollama call, leak sanitisation, source notes | Makes scope claims from the full catalogue; assumes metadata homogeneity |
| Claim predictor | `agents/claim_predictor.py` | 4 implemented rules, weighted score, documents checklist | Not called from chat; not policy-aware |
| UI | `ui/app.py` | 3 modes (Ask / Predict / Upload), sources, optional debug panel | Upload is a placeholder; filters are free text |
| API | `api/main.py` | `/ask`, `/predict`, `/insurers`, `/health` | No auth, no validation, no streaming |
| Eval | `eval/judge.py` + 11 gold cases | Rule checks + LLM judge | Same model as generator; no retrieval metrics |

### 2.3 Data stores (there is no SQL database)

| Store | Path | Contents today |
| --- | --- | --- |
| Raw documents | `data/raw_policies/` (1.1 GB), `data/raw_pdfs/` | 924 + 1 PDFs; 8 duplicate hashes; a manual `Bajaj-Allianz/` folder plus the scraper's `bajaj_allianz/` |
| Orphan documents | `data/irdai/` | 2 IRDAI regulations that **no code scans** (`SCAN_DIRS` = `raw_pdfs`, `raw_policies` only) |
| Sidecars | `*.meta.json` | insurer, policy, doc type, line, `source_doc_type`, `source_url`, `uin` |
| Parsed text | `data/parsed/` | 773 `.txt` files (collide on the same triplet as chunks) |
| Chunk files | `data/chunks/` | 771 JSON files (again colliding), plus `ingest_index.json` |
| Ingest ledger | `data/chunks/ingest_index.json` | 923 rows, 48,611 claimed chunks, SHA-256, timestamps |
| Vector index | `data/chroma_db/` (622 MB) | 45,844 chunks, 774 distinct `(insurer, policy, doc_type)` keys |
| Scrape manifest | `data/raw_policies/manifest.json` | 920 sources, 6 runs, review-required, never auto-ingested |
| Eval | `eval/reports/latest.json` | 11/11 passed at 2026-09-19T08:38Z |

### 2.4 Ask path (what actually happens)

1. UI sidebar picks an insurer (display name) and an optional free-text policy name (`ui/app.py:37-42`).
2. `PolicyRetriever.retrieve` infers a chunk type from keywords, runs a dense query with `insurer`/`policy_name` filters, optionally a second typed query, merges, dedupes, reranks, returns 8 (`rag/retriever.py:27-59`).
3. `BimaSaathiGenerator.generate` re-detects insurers in the question, filters chunks by insurer, applies a policy-phrase filter, drops chunks with distance > 0.72, builds a scope block from the **ledger catalogue**, formats clauses, calls Ollama (`qwen2.5:7b`, temp 0.4, ctx 8192), sanitises leaks, checks banned patterns, appends a source note, returns sources (`rag/generator.py:364-462`).
4. UI renders the answer and a page/section citation list; no UIN, no source URL, no document date.

### 2.5 Ingestion path

`SHA-256 change detection → infer metadata (sidecar > folder > filename) → pdfplumber per page → clean lines → 1600-char slices with 2-line overlap → classify chunk_type by keywords → write parsed/chunk JSON → delete by `source` in Chroma → embed and upsert`.

That single `source` string is the pivot on which data integrity turns.

---

## 3. Measured state (facts, not impressions)

| Metric | Value | Source |
| --- | --- | --- |
| Ledger files / claimed chunks | 923 / 48,611 | `ingest_index.json` |
| Actual Chroma chunks | 45,844 | Chroma count |
| Documents whose chunks were overwritten | ~123 files, 3,043 chunks | recomputed from ledger groups |
| Distinct catalogue keys | 768 (ledger) vs 774 (Chroma) | both |
| Chroma-only entries (not in ledger) | 6 keys / 276 chunks: IRDAI regs (94), Aditya Birla Super Health Plus (83), Niva RISE (71), Zurich Health 360 (22), seed policy (6) | Chroma |
| Doc-type mix (ledger) | 559 brochure, 362 policy_wording, 2 "regulation" | ledger |
| Both ledger "regulation" entries are misclassified | HDFC "mosquito…irdai changes" prospectus; Cholamandalam IRDAI circular | ledger |
| Chunks whose `section` is just "Page N" | 26,418 / 45,788 (58%) | chunk JSONs |
| Chunks with no `document_type` (seed data) | 6, all `star_health/comprehensive` | Chroma |
| PDFs that fail extraction every run (image-only) | 6 | `logs/index.log` |
| Corpora in tree | 1.9 GB `data/`, 622 MB Chroma | `du` |
| Eval | 11/11 pass, 2026-09-19 | `eval/reports/latest.json` |

Two live probes summarise the quality problem better than any table:

```text
Q: What is the room rent cap in Star Health?
   retrieved: star_health/comprehensive  d=0.265  (seed data, no page, no document_type)
   selected:  6 chunks, no gap refusal
   scope:     "Insurers with no policy loaded: Star Health, ICICI Lombard, ..."
   -> the same context both answers from a fabricated Star Health policy
      and declares that Star Health is not loaded.

Q: policy_name filter = "Comprehensive" (as typed in the UI)
   -> 0 exact matches -> fallback to unfiltered search -> returns
      bajaj_allianz/star_package, hdfc_ergo/my_sampoorna_suraksha, ...
```

---

## 4. Problems

### P0 — trust-breaking (fix before any demo to a real user)

#### P0-1 A fabricated policy is retrievable in production **[verified]**
`star_health_comprehensive_0..5` are hand-written demo clauses (the same sample shape used in `rag/chunker.py`). They have `source=star_health_comprehensive`, no `document_type`, no `page`, no `source_file`. They are the top hit for Star Health queries (distance 0.265) and pass `_select_chunks` whenever the user does not name a policy. The gold case `gap_star_health` only passes because the phrase "Family Optima" triggers a policy-hint mismatch — a lucky path, not a guarantee.
**Impact:** fabricated coverage advice for an insurer the product explicitly says it cannot cover. **Fix:** delete the seed source, add a startup/reindex guard that rejects chunks without `source_file` + `document_type`, and namespace any future demo data (`demo:` collection) so it can never mix with production. Add a regression test that queries five un-loaded insurers and asserts refusal.

#### P0-2 Vector `source` is not a document identity; ingestion silently destroys documents **[verified]**
`source = f"{insurer}_{policy_name}_{document_type}"` (`rag/ingest_pdf.py:131`) and `replace_source_chunks` deletes *all* chunks sharing it (`rag/embedder.py:41-52`). Three mechanisms create collisions:
- `write_sidecar` maps `cis`, `prospectus`, and `brochure` all to `"brochure"` (`scrapers/hybrid_scraper.py:659-660`), so a policy's brochure, CIS, and prospectus are one key.
- Duplicate manual and scraped files (`Bajaj-Allianz/` vs `bajaj_allianz/`; `tata_aig_medicare_protect.pdf` vs `health__tata_aig_medi_care_protect__policy_wording.pdf`) share a key.
- Coarse filename inference produces identical policy names for different products.

Result: 923 files → 768 keys; 123 groups overwrite each other; 3,043 chunks vanish. Which file survives depends on alphabetical scan order, so answers are not reproducible across machines.
**Impact:** a large, invisible share of the corpus is unreachable; the catalogue and the LLM's "known policies" block claim documents that are not there. **Fix:** key on content (`source_file` + SHA-256, or `insurer/policy/doc_type/sha8`), migrate and reindex, and make replacement a per-file operation. Keep `source_doc_type` distinct (see P0-4).

#### P0-3 The catalogue and Chroma have drifted in both directions **[verified]**
`get_catalog()` prefers `data/chunks/ingest_index.json` (`rag/embedder.py:94-113,155-160`). The ledger records chunk counts at ingest time, before later files delete them, and is never pruned. It also no longer contains 6 sources that are still in Chroma:

| Chroma-only key | Chunks | Consequence |
| --- | --- | --- |
| `irdai / maintenance_information_2025 / regulation` | 74 | Real regulation, retrievable, absent from the catalogue and the "known policies" block |
| `irdai / regulatory_sandbox_2025 / regulation` | 20 | Same |
| `aditya_birla / super_health_plus / policy_wording` | 83 | Older ingest of a product now listed under two other names |
| `max_bupa / rise / policy_wording` | 71 | Older ingest; ledger has `niva_bupa_rise` |
| `zurich_kotak / health_360 / brochure` | 22 | Older ingest; ledger has `health_360_1` (suffix from `(1).pdf`) |
| `star_health / comprehensive / (no type)` | 6 | Seed policy (P0-1) |

The two real IRDAI regulations therefore exist **only** as unmanaged leftovers in this one local database. A clean reindex on another machine cannot reproduce them, because their PDFs live in `data/irdai/`, which no scan root includes.
**Impact:** scope claims, sidebar counts, and citations are wrong in both directions; the corpus is not reproducible. **Fix:** derive the catalogue from Chroma (or reconcile on every run and fail loudly), make `data/irdai` an explicit source, and delete or re-ingest the legacy duplicates so each product has one canonical source.

#### P0-4 Document-type mapping is lossy and occasionally wrong **[verified]**
- Endorsements/riders are written as `policy_wording` (`hybrid_scraper.py:660`), so an add-on is ranked as the primary contract.
- CIS and prospectus become `brochure`; the distinction is lost.
- Filename inference marks a document as `regulation` if the filename contains "irdai" or "guideline" (`rag/sync_knowledge.py:149-152`). Both ledger regulations are misclassified: an HDFC product ("…irdai changes…") and a Cholamandalam IRDAI circular. Meanwhile the two genuine IRDAI regulations are outside every scan root (`data/irdai/`), so their chunks exist in Chroma only as unreproducible leftovers (P0-3).
**Impact:** primary sources are demoted, secondary sources promoted, and the regulatory disclaimer is attached to ordinary product documents. **Fix:** carry `source_doc_type` into Chroma; restrict `regulation` to `insurer == irdai` or an explicit sidecar flag; add `data/irdai` to the scan roots.

---

### P1 — answer correctness and product promise

#### P1-1 `policy_name` filtering is exact-match, then silently dropped **[verified]**
The UI passes raw text (`ui/app.py:506`); Chroma compares for equality; on zero results the retriever falls back to an unfiltered search (`rag/retriever.py:55-56`). Typing "Comprehensive" returns Bajaj/HDFC/Niva documents.
**Impact:** wrong-insurer evidence presented as if filtered; the user has no signal. **Fix:** build a policy dropdown from the catalogue per insurer, normalise slugs, and never silently widen a filter (return empty + an explicit "not loaded" answer).

#### P1-2 Comparison and multi-insurer questions are structurally unsupported **[code]**
Retrieval is one query with one optional insurer filter; the generator refuses when chunks do not match the insurer named in the question. The UI simultaneously suggests "Compare two health plans on waiting periods".
**Impact:** the advertised comparison use case produces a one-sided or refusing answer. **Fix:** extract all insurers in the query, fan out retrieval per insurer with a per-insurer quota, then synthesise; or remove the suggestion until the compare mode exists.

#### P1-3 Insurer detection is substring-based **[verified]**
`_mentioned_insurers` uses `alias in text` (`rag/generator.py:63-87`). "Is digital cashless available in Star Health?" yields `["star_health", "digit"]`, which can inject a "Digit policy not loaded" refusal into a Star Health answer.
**Fix:** word-boundary matching, longest-alias-first, explicit alias table, unit tests including "digital", "critical", "birla", "max".

#### P1-4 Scope detection swallows real coverage questions **[verified]**
`_is_scope_question` matches `not (know|have|cover|support)` (`rag/generator.py:157-167`). "Does this plan not cover room rent?" is classified as a scope question, retrieval is discarded (`:392`), and the user gets a gap answer. Conversely "What don't you have?" is *not* detected.
**Fix:** scope detection should require a scope *object* ("which insurers/policies do you have") and must never trigger when the question contains coverage nouns; cover both phrasings in tests.

#### P1-5 Brochures outrank policy wordings **[verified]**
`DOC_TYPE_RANK` is only a tiebreaker (`rag/retriever.py:95,102`); distance dominates. Probes: PED exclusions for Cholamandalam → 5 of 6 chunks were brochures; cashless claim process for Bajaj → 4 of 6 brochures; ambulance cover for Digit → 4 of 6 brochures. The corpus is 61% brochures (559/923), so this is systematic.
**Impact:** marketing summaries become the factual basis for coverage answers; a footnote at the end does not repair it. **Fix:** for coverage/exclusion/waiting intents, hard-prefer `policy_wording` (filter, or a large penalty, not a tiebreak); allow brochures only when no wording exists, and say so up front. Add eval cases that assert the source type.

#### P1-6 The relevance threshold is too permissive **[code]**
`distance <= 0.72` (`rag/generator.py:136`) while genuine matches in this index score 0.14–0.30. 0.72 admits weakly related clauses and keeps the refusal rate artificially low.
**Fix:** calibrate per embedding model (start ~0.45–0.50), make it config, and validate with a labelled negative set.

#### P1-7 No conversation state in retrieval or generation **[code]**
History is rendered (`ui/app.py:433-439`) but never sent to the model and never used to rewrite the query. Follow-ups ("what about the cap?", "and for maternity?") retrieve against the bare follow-up text.
**Fix:** pass the last 2–3 turns into a query-rewrite step (or a chat prompt) and preserve/merge filters; add multi-turn eval cases.

#### P1-8 The claim predictor is disconnected, non-policy-aware, and partly unimplemented **[code]**
- `rejection_risk` is always `None` in RAG answers (`rag/generator.py:460,174,418`) even though the system prompt tells the model to surface high risk and the README advertises it.
- `ClaimPredictor` uses fixed assumptions: 1% room-rent cap (`agents/claim_predictor.py:54`), 36-month PED wait (`:68`), 24-hour intimation (`:80`), generic documents (`:112-125`).
- Two configured rules (`disclosure_match`, `procedure_excluded`) never fire; `insurer`, `policy_name`, and `hospital_name` are effectively cosmetic.
- The UI offers the predictor for motor insurers and asks for room rent and sum insured.
**Impact:** an authoritative-looking percentage that is not tied to the loaded policy and contradicts the "answers only from wording" promise. **Fix:** either (a) derive cap/waiting/PED/intimation from retrieved clauses and show the clause beside each failed check, or (b) label the tool explicitly as generic IRDAI heuristics; gate fields by line of business; implement the missing rules or remove them from config.

---

### P2 — ingestion quality, operations, reproducibility

#### P2-1 Chunking loses exactly the structure insurance questions depend on **[verified/code]**
- Page-sliced 1600-character windows with a 2-line overlap (`rag/ingest_pdf.py:100-120`) cut clauses mid-sentence at page boundaries.
- No table extraction, although waiting-period and sub-limit data is tabular; `pdfplumber.extract_tables` is unused.
- Section detection is a dozen fixed English phrases (`:22-35`); 58% of chunks are labelled only "Page N", so "cite the section" in the prompt cannot be honoured.
- No OCR: 6 image-only PDFs fail on every run.
- `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP` in `config.py:35-36` are never read.
**Fix:** clause-aware chunking with heading/numbering detection, table extraction to structured text, OCR fallback (mark as OCR-derived), and real overlap. This is the single highest-leverage quality investment.

#### P2-2 Three ingestion paths, three metadata schemas **[verified]**
`chunker.py` (legacy) writes `source` without document type and `page="unknown"`; `ingest_pdf` writes typed sources and int pages; `ingest_text` adds `source_file`/`document_type` again. The seed chunks match none of them. `ingest_pdf`'s `--document-type` accepts only three values, while the scraper classifies eight.
**Fix:** one pipeline, one schema, validated at the vector-store boundary; delete the legacy path or make it call the same builder; add a schema version to metadata.

#### P2-3 No deletion, no tombstones, noisy failure semantics **[code]**
Removing a PDF from disk leaves its chunks in Chroma and its row in the ledger. Failed files are retried every run because they are never recorded as failures. `sync_knowledge` exits 1 if *any* file is skipped or errors (`rag/sync_knowledge.py:328-329`), so `./appctl.sh index` reports failure on a corpus that is 99% fine.
**Fix:** mark tombstones/deletions, persist failure status with a reason, and separate "fatal" from "partial" exit codes (`--strict` for CI).

#### P2-4 Scraper is non-incremental and has concurrency/overwrite hazards **[code]**
- No `ETag`/`If-Modified-Since`: a re-run re-downloads all ~924 documents.
- One `requests.Session` shared across 4 threads (not thread-safe).
- `_throttle` sleeps while holding a global lock, serialising all hosts, not just the slow one.
- `safe_filename` truncates at 120 chars, so long product names can collide and silently overwrite.
- 281 `skipped_robots` are counted but not attributably reported per URL in the discovery report.
**Fix:** per-host sessions or a connection pool with per-host locks, conditional GETs, collision-safe filenames with short digests, and a per-URL skip table.

#### P2-5 The catalogue has no product identity, and the noise reaches the LLM **[verified]**
Filename inference plus `_pretty_policy_name` (4-word truncation, `rag/generator.py:276-292`) fragments one product into many entries. Niva Bupa alone yields `health_premia`, `health_premia_br`, `health_premia_gold_br`, `health_premia_platinum_br`; `smart_health`, `smart_health_finalpolicy`, `smart_health_healthplus`, `smart_health_prsopectus` (typo intact); `everyday_health` four times; `health_companion_variant` variants; plus "Aap Ke Liye Madhya", "Arogya Sanjeevani Policy Go" beside "…Go Digit", and "Auto Secure Commercial Vehivle". Aditya Birla appears as `activ_one` and `health_insurance_activ_one`; Zurich's one brochure becomes `health_360_1`. The scope block sent to the LLM contains all of it, along with 157–161 "policies" for Bajaj.
**Impact:** the product looks unreliable in exactly the answer where it should look authoritative, and the model is primed to confuse product variants. **Fix:** a reviewed product registry keyed by UIN; group documents under a product (wording, CIS, brochure, endorsement) rather than treating each filename as a product; hide low-confidence names from the prompt; show counts only for answerable products.

#### P2-6 Reproducibility and hygiene **[verified]**
- `requirements.txt` is unpinned and contains unused `langchain`, `langchain-community`, `unsloth`; `uvicorn` is missing although `api/main.py` runs it.
- No git repository, no `.gitignore`, no CI, no tests directory, no linter/formatter config.
- 1.9 GB of PDFs and 622 MB of Chroma live in the working tree; a stray `__pycache__/` sits at the root.
**Fix:** pin and prune dependencies, move corpus/index out of the repo (or Git LFS + a data README), add `pyproject.toml`, pytest, ruff, and a CI job that runs rules-only eval on a small fixture corpus.

#### P2-7 Import-time environment fragility **[verified]**
`from rag.generator import ...` constructs an `httpx` client inside the `ollama` package at import time. With this machine's `NO_PROXY` containing `[::1]`, httpx raises `InvalidURL: Invalid port: ':1]'` and the app/eval cannot start. Workaround used during review: `env -u NO_PROXY -u no_proxy`.
**Fix:** construct the Ollama client lazily with an explicit host, pin compatible `httpx`/`ollama` versions, and document the environment (or ship a container).

#### P2-8 Documentation drift **[verified]**
`README.md` says `mistral:7b` (config: `qwen2.5:7b`), "hybrid search" (dense + heuristic rerank, no BM25), "semantic chunking" (regex/page windows), 50 MB download cap (code: 60 MB), and lists roadmap items that are partially built. `PROGRESS.md` still says "the dummy policy can be chunked, indexed…" while the corpus has 923 files. `PROGRESS.html` duplicates the markdown and will drift further.
**Fix:** one source of truth; generate the architecture/status doc from the code and the ledger.

---

### P3 — security, privacy, evaluation depth

#### P3-1 API exposure **[code]**
`uvicorn.run(app, host="0.0.0.0", port=8000)` (`api/main.py:111`), no auth, no rate limit, no CORS policy, no request IDs or structured logs. `/predict` accepts medical attributes; `/ask` will burn local GPU/CPU on any caller.
**Fix:** bind `127.0.0.1` by default, add an optional API key, input validation (`insurer` in registry, ranges), and request logging.

#### P3-2 Prompt injection path is open by design **[code]**
Retrieved clause text and the user question are interpolated directly into `CONSTITUTIONAL_PROMPT` with no delimiters or "treat as data" instruction. The roadmap's "upload your own policy PDF" turns this into a user-controlled injection channel.
**Fix:** delimit and label untrusted text, instruct the model to ignore instructions inside clauses, strip instruction-like lines from extracted text, and add a canary test with a malicious PDF.

#### P3-3 Guardrail implementation leaks and is brittle **[code]**
A blocked answer appends `System note: {reason}` to the user-visible text (`rag/generator.py:435`). `BANNED_PATTERNS` is an English regex list, so Hinglish/Hindi variants pass. `_sanitize_answer` can replace a phrase mid-answer and then discard the whole answer.
**Fix:** never surface internal reasons, add Hinglish/Hindi patterns and eval cases, and return a clean fixed refusal.

#### P3-4 Evaluation is a demo, not a gate **[verified]**
11 cases, judged by the same `qwen2.5:7b` that generated the answer, with loose JSON regex parsing. No retrieval metrics, no claim-predictor tests, no API tests, no scraper tests, and no calibration of the distance threshold. The `gap_star_health` case passes while fabricated Star Health data is live (P0-1). Reports are dated 2026-09-19 and are not run in any pipeline.
**Fix:** add a retrieval gold set (per-insurer questions with expected `source_file`/page), groundedness checks, a refusal suite, deterministic predictor fixtures, API contract tests, a different judge model or human spot-checks, and a CI threshold.

#### P3-5 Citations are not auditable **[code]**
Sources show insurer, policy, section, page — not UIN, source URL, document date, or hash. `sync_knowledge` drops `source_url` and `uin` even though the sidecar carries them. `page` is `"unknown"` for text-sourced chunks and the UI renders "Page unknown".
**Fix:** persist `source_url`, `uin`, `retrieved_at`, `sha256`, `document_date`; show them in the citation; never emit "Page unknown" as a citation.

#### P3-6 Data licensing and privacy **[code]**
927 insurer PDFs are redistributed inside the project with no license, attribution, or retention note, and no consent story for medical attributes sent to `/predict`. This is a product/legal loose end to resolve before any shared deployment.

---

## 5. Root causes (why these bugs keep appearing)

1. **Document identity was never modelled.** Identity is a convenience string (`insurer_policy_doctype`) used simultaneously as a display name, a filter, a deletion key, and a catalogue key. Every P0 follows from that one decision.
2. **Two sources of truth and three ingestion paths.** The ledger, the chunk files, and Chroma each claim to describe the corpus; they disagree, and no reconciliation exists.
3. **String heuristics stand in for domain modelling.** Insurer aliases exist in three places, scope detection and policy hints are hard-coded phrase lists, doc types are guessed from filenames, and claim rules are hard-coded constants. This is fast to write and impossible to test exhaustively.
4. **Retrieval is optimised for a demo, not for evidence hierarchy.** Cosine distance alone decides what counts as truth; the distinction between binding wording, a summary, and a regulation is a tiebreak and a footnote.
5. **No automated gate.** No unit tests, no CI, and an eval that the same model grades itself. Regressions in the corpus (P0-1, P0-2) were invisible for months while the eval reported 11/11.

---

## 6. Loose ends — promised or implied but not finished

| Item | Where it is implied | Actual state |
| --- | --- | --- |
| Upload a policy PDF and scan it | `README.md` roadmap, `ui/app.py:384-398` | Placeholder that tells the user to copy the file and run a CLI command |
| Hospital cashless-network validator | README roadmap | No network data; `is_network_hospital` is self-reported |
| Ombudsman complaint drafter | README roadmap | Not started |
| Hinglish/Hindi support | README roadmap, prompt asks the model to mirror Hinglish | English-only embedding model; no multilingual eval cases |
| Hybrid (lexical + dense) retrieval | README project structure | Dense + heuristic bonuses only |
| Configurable chunk size/overlap | `config.py:35-36` | Never read by any ingester |
| LLM fallback | `config.py:29` (`llama-3.1-70b-versatile`) | Never referenced by the generator |
| Disclosure-match and procedure-exclusion risk rules | `config.py:105-116`, `PROGRESS.md` | Never implemented |
| IRDAI regulations in the knowledge base | `PROGRESS.md` | 94 real chunks sit in Chroma as ledger orphans; their PDFs are in the unscanned `data/irdai/`; the ledger's two `regulation` rows are misclassified product documents |
| "Cite exact clauses with section numbers" | `config.py:67` | 58% of chunks have only "Page N" as their section |
| Chat-level claim risk | Prompt + README | `rejection_risk` is always `None` |
| Tests, CI, `.gitignore` | `PROGRESS.md` planned work | None exist; project is not a git repository |
| Freshness / revision tracking | Product need | No "as of" date in citations; no scheduled re-crawl or diff alert |

---

## 7. Recommended sequence

### Phase 0 — restore trust (1–2 working days)
1. Delete the seed Star Health source; add a reindex guard requiring `source_file`, `document_type`, `sha256` in every chunk.
2. Switch `source` to a content-addressed id; wipe and reindex from scratch; verify `Chroma count == expected count`, no duplicate source ids, and every ledger row retrievable.
3. Add `data/irdai` as a scan root (or move those PDFs into `raw_policies/irdai/`), reindex them properly, and delete the three legacy Chroma-only product ingests (`super_health_plus`, `rise`, `health_360`) so each product has one canonical source.
4. Build the catalogue from Chroma only; add a reconcile command that fails on drift.
5. Fix doc-type mapping (endorsement ≠ wording; CIS/prospectus preserved; regulation only for IRDAI) and reindex.
6. Re-run the gold set plus new adversarial cases (fake-policy query, un-loaded insurer, filter typo, brochure-only question).

### Phase 1 — correct retrieval behaviour (week 1)
6. Policy dropdown per insurer; no silent filter fallback.
7. Per-insurer fan-out for comparison questions, or remove the compare suggestion.
8. Word-boundary insurer matching; scope-detection rewrite; regression tests for both.
9. Wording-priority retrieval and a calibrated distance threshold.
10. Claim predictor: line-of-business gating, clause-linked or explicitly-labelled heuristics, implement or delete unused rules.

### Phase 2 — evidence quality (weeks 2–3)
11. Clause-aware, table-aware, OCR-fallback extraction; real overlap; section numbering.
12. Provenance in every citation (UIN, URL, date, page); remove "Page unknown".
13. Retrieval gold set (40+ questions, expected document/page), groundedness checks, refusal suite, multi-turn eval.
14. Multi-turn query rewriting.

### Phase 3 — product hardening (week 4)
15. Real upload flow with the same identity/provenance guarantees.
16. API auth, localhost bind, input validation, logging.
17. Pinned dependencies, pytest, ruff, CI running the rules-only eval on a fixture corpus; Dockerfile.
18. Corpus freshness job (scheduled crawl, revision diff, re-index) and a retention/licensing note.
19. Then the promised features (network validator, ombudsman drafter, Hinglish).
---

## 8. Acceptance gates for the next milestone ("trustworthy prototype")

A release should not be demoed to a real policyholder unless all of these hold:

1. Zero chunks without `source_file`, `document_type`, `sha256`; zero seed/demo chunks in the production collection.
2. One source id per document; `Chroma count == ledger count`; a reconcile command returns clean.
3. Every insurer in the scope block is answerable from Chroma, and every unloaded insurer refuses in 100% of a 20-question negative set.
4. No answer is produced with a top distance above the calibrated threshold.
5. For coverage/exclusion questions, when a policy wording exists, at least 80% of returned chunks are wordings.
6. Comparison questions return evidence from every insurer named.
7. Every citation carries document name, page, doc type, UIN (where available), and source URL.
8. Claim risk is either clause-linked with a visible source, or visibly labelled generic.
9. `./appctl.sh index` is idempotent: second run changes nothing; deleted files disappear from retrieval.
10. CI runs unit tests + rules-only eval on a fixture corpus without network or Ollama.

---

## 9. Open questions for the owner

1. **Depth vs breadth:** is v1 meant to be *correct for 20 policies* or *present for 900 documents*? The current corpus suggests breadth, but the product promise requires depth.
2. **Who signs off a document?** The manifest says `review_required: true` but nothing records who reviewed a PDF, its revision date, or whether it is current.
3. **Hosted or local-only?** This decides whether API auth, rate limiting, and medical-data handling are blockers.
4. **Is the corpus redistributable?** If not, the data and index must leave the repo and be built on first run.
5. **Target user:** policyholder or desk agent? The claim predictor and citation UX differ substantially between the two.

---

## Appendix A — reproduction commands

```bash
# Catalogue vs vector-store drift
./venv/bin/python - <<'PY'
import json, chromadb
c = chromadb.PersistentClient(path="./data/chroma_db").get_collection("policy_clauses")
print("chroma:", c.count())
idx = json.load(open("data/chunks/ingest_index.json"))["files"]
print("ledger files:", len(idx), "ledger chunks:",
      sum(int(v.get("chunks") or 0) for v in idx.values()))
PY

# Seed-policy contamination
./venv/bin/python - <<'PY'
import chromadb
c = chromadb.PersistentClient(path="./data/chroma_db").get_collection("policy_clauses")
print(c.get(where={"insurer": "star_health"}, include=["metadatas"])["metadatas"])
PY

# Retrieval without the broken NO_PROXY env var
env -u NO_PROXY -u no_proxy HF_HUB_OFFLINE=1 ./venv/bin/python - <<'PY'
from rag.retriever import PolicyRetriever
r = PolicyRetriever()
for q, kw in [
    ("What is the room rent cap in Star Health?", {"insurer": "star_health"}),
    ("What is the room rent cap?", {"policy_name": "Comprehensive"}),
]:
    print(q, kw)
    for row in r.retrieve(q, n_results=4, **kw):
        m = row["metadata"]
        print("  ", m.get("insurer"), m.get("policy_name"),
              m.get("document_type"), round(row.get("distance") or 0, 3))
PY

# Eval (requires Ollama running)
./venv/bin/python eval/judge.py --rules-only
```

## Appendix B — file map used in this review

```text
config.py                      models, insurers, prompt, claim weights
rag/chunker.py                 legacy regex chunker (txt)
rag/ingest_pdf.py              PDF extraction + chunking + replacement
rag/embedder.py                Chroma store, search, catalogue
rag/retriever.py               dense retrieval + heuristic rerank
rag/generator.py               guardrails, scope, prompt, Ollama
rag/sync_knowledge.py          ledger, metadata inference, orchestration
agents/claim_predictor.py      rule engine
ui/app.py / ui/debug_tools.py  Streamlit UI and diagnostics
api/main.py                    FastAPI surface
scrapers/hybrid_scraper.py     registry, discovery, download, sidecar
eval/judge.py, gold_cases.json eval harness
appctl.sh                      start/stop/index wrapper
```
