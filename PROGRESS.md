# BeemaSathi Progress Tracker

Last updated: 23 August 2026, 10:04 AM IST

## Status Legend

- `[x]` Completed
- `[~]` In progress
- `[ ]` Planned
- `[!]` Blocked

## Progress Log

### 22 August 2026

#### 10:58 PM–11:01 PM — Project discovery

- [x] Reviewed the project structure and architecture.
- [x] Identified the Streamlit UI, FastAPI API, RAG pipeline, and claim predictor.
- [x] Identified startup defects, missing ingestion, and testing gaps.
- [x] Created a visual project architecture overview.

#### 11:01 PM–11:06 PM — Local LLM setup

- [x] Downloaded `mistral:7b` through Ollama.
- [x] Verified that `mistral:7b` is installed.
- [x] Confirmed that the Ollama service is running locally.

#### 11:06 PM–11:10 PM — Python environment

- [x] Created the `venv` virtual environment with Python 3.12.
- [x] Installed packages from `requirements.txt`.
- [x] Verified that there are no broken Python dependencies.

#### 11:10 PM–11:17 PM — First working demo

- [x] Fixed undefined project paths in `rag/chunker.py` and `rag/embedder.py`.
- [x] Fixed the missing `Path` import in `rag/generator.py`.
- [x] Generated six chunks from the built-in dummy policy.
- [x] Downloaded the BGE embedding model from its official Hugging Face source.
- [x] Indexed the dummy policy chunks into ChromaDB.
- [x] Launched the Streamlit application at `http://localhost:8501`.

#### 11:47 PM–11:51 PM — Debug observability

- [x] Added an environment-protected Streamlit debug mode.
- [x] Added live retrieval, generation, and total request timings.
- [x] Added configured and actively loaded Ollama model diagnostics.
- [x] Added Ollama process RAM, model size, and reported VRAM diagnostics.
- [x] Added retrieved chunk distances and Ollama token metrics.
- [x] Documented how to start the application in debug mode.

### 23 August 2026

#### 12:02 AM–12:08 AM — First real insurance document

- [x] Added Zurich Kotak as a supported insurer.
- [x] Extracted all 14 pages from the Health 360 brochure.
- [x] Created and indexed 22 page-aware brochure chunks.
- [x] Added a reusable PDF ingestion command with source replacement.
- [x] Fixed multi-filter compatibility with the installed ChromaDB version.
- [x] Added insurer, policy, document type, page, and source-file metadata.
- [x] Added brochure limitations to the prompt, generated answer, and UI.
- [x] Verified room-rent, waiting-period, and emergency-claim retrieval.
- [x] Verified end-to-end answer generation with `mistral:7b`.

#### 8:50 AM–8:57 AM — Safe hybrid scraper trial

- [x] Restored compatibility between the modified chunker, embedder, PDF ingestion, and UI.
- [x] Added official-domain enforcement and fail-closed `robots.txt` checks.
- [x] Added PDF signature validation, a 50 MB limit, SHA-256 hashes, and atomic downloads.
- [x] Added dry-run, insurer, mode, limit, and explicit third-party CLI controls.
- [x] Added persistent review manifests that are not erased by empty runs.
- [x] Confirmed ICICI Lombard and Star Health could not be scraped without bypassing their denied robots checks.
- [x] Downloaded and reviewed Niva Bupa RISE policy wording from its official domain.
- [x] Indexed 34 pages into 71 page-aware chunks.
- [x] Verified scoped RISE retrieval; the knowledge base now contains 99 chunks.
- [!] IRDAI official endpoints returned HTTP 504 during the trial.

#### 9:26 AM–9:32 AM — Top-10 insurer collection run

- [x] Ran the official-only collector across all ten configured insurers.
- [x] Collected four validated official policy wordings for Niva Bupa, Tata AIG, and Aditya Birla.
- [x] Reviewed and indexed Tata AIG MediCare Protect (26 pages, 67 chunks).
- [x] Reviewed and indexed Aditya Birla Activ One (56 pages, 204 chunks).
- [x] Reviewed and indexed Aditya Birla Super Health Plus (20 pages, 83 chunks).
- [x] Added Aditya Birla, Tata AIG, Acko, Digit, and Bajaj to the UI insurer registry.
- [x] Fixed HTML filename collisions, redirect validation, and manifest hash deduplication.
- [x] Verified scoped retrieval for all three newly indexed policies.
- [x] Increased the knowledge base from 99 to 453 chunks.
- [!] Star Health, Care Health, and ICICI Lombard denied verifiable robots access.
- [!] Acko, Digit, HDFC Optima, Niva product pages, and Bajaj configured URLs returned 404.
- [ ] Find updated official landing/document URLs for failed sources without using third-party mirrors.

#### 10:02 AM–10:04 AM — Local IRDAI regulations

- [x] Registered IRDAI as a regulatory knowledge source separate from insurers.
- [x] Added `regulation` document-type support and regulatory answer safeguards.
- [x] Indexed the 2025 Maintenance and Sharing of Information regulation (28 pages, 74 chunks).
- [x] Indexed the 2025 Regulatory Sandbox regulation (8 pages, 20 chunks).
- [x] Added IRDAI filtering and regulatory notices to the Q&A interface.
- [x] Verified scoped retrieval for both regulations.
- [x] Increased the knowledge base from 453 to 547 chunks.

## Planned Work

The dates and time blocks below are a proposed working schedule and can be adjusted.

### 23 August 2026 — Stabilize the prototype

#### 10:00 AM–12:00 PM

- [ ] Test all three Streamlit modes.
- [ ] Test the FastAPI endpoints.
- [ ] Record and fix clean-start runtime errors.
- [ ] Add `uvicorn` to `requirements.txt` if it is still missing.

#### 2:00 PM–4:00 PM

- [ ] Add a `.gitignore` for `venv`, model caches, ChromaDB data, and Python artifacts.
- [ ] Remove or document unused dependencies.
- [ ] Add clear local setup and troubleshooting instructions to `README.md`.

### 24 August 2026 — Policy document ingestion

#### 10:00 AM–1:00 PM

- [ ] Implement PDF text extraction using `pdfplumber`.
- [ ] Save parsed policy text under `data/parsed/`.
- [ ] Add validation for encrypted, empty, or invalid PDFs.

#### 2:00 PM–5:00 PM

- [ ] Connect parsed documents to the chunking pipeline.
- [ ] Make chunk size and overlap configuration effective.
- [ ] Add stable document and chunk identifiers.
- [ ] Prevent duplicate documents in ChromaDB.

### 25 August 2026 — Upload and indexing workflow

#### 10:00 AM–1:00 PM

- [ ] Replace the Streamlit upload placeholder with functional PDF upload.
- [ ] Add insurer and policy metadata fields.
- [ ] Parse, chunk, and index an uploaded policy.

#### 2:00 PM–5:00 PM

- [ ] Display upload and indexing progress.
- [ ] Show actionable errors to the user.
- [ ] Confirm that uploaded policies become searchable immediately.

### 26 August 2026 — Improve policy retrieval

#### 10:00 AM–1:00 PM

- [ ] Evaluate retrieval using representative insurance questions.
- [ ] Add lexical or hybrid retrieval if it improves results.
- [ ] Add result re-ranking.

#### 2:00 PM–5:00 PM

- [ ] Improve insurer and policy filters.
- [ ] Include source document, section, and page information in citations.
- [ ] Add a minimum relevance threshold for weak matches.

### 27 August 2026 — Claim predictor correctness

#### 10:00 AM–1:00 PM

- [ ] Implement the configured disclosure-match rule.
- [ ] Implement the configured procedure-exclusion rule.
- [ ] Move fixed assumptions into policy-specific configuration.

#### 2:00 PM–5:00 PM

- [ ] Explain how every failed rule affects the score.
- [ ] Add stronger informational and policy-verification disclaimers.
- [ ] Validate the predictor against sample claim scenarios.

### 28 August 2026 — Tests and code quality

#### 10:00 AM–1:00 PM

- [ ] Add unit tests for chunk classification.
- [ ] Add unit tests for claim predictor rules and score limits.
- [ ] Add tests for generator safety patterns.

#### 2:00 PM–5:00 PM

- [ ] Add integration tests for indexing and retrieval.
- [ ] Add API endpoint tests.
- [ ] Configure formatting and linting.

### 29 August 2026 — User experience and safety

#### 10:00 AM–1:00 PM

- [ ] Improve loading, empty, and error states.
- [ ] Improve policy source presentation.
- [ ] Test English, Hindi, and Hinglish questions.

#### 2:00 PM–5:00 PM

- [ ] Review responses for unsupported insurance or legal claims.
- [ ] Ensure answers distinguish policy text from general guidance.
- [ ] Add clear instructions for verifying information with the insurer.

### 30 August 2026 — MVP review

#### 10:00 AM–1:00 PM

- [ ] Run the complete setup on a clean environment.
- [ ] Test upload, indexing, Q&A, claim prediction, and API flows.
- [ ] Record known limitations and unresolved defects.

#### 2:00 PM–4:00 PM

- [ ] Update project documentation.
- [ ] Prepare a short demo flow.
- [ ] Define the next milestone based on MVP feedback.

## Current Milestone

**Milestone:** Local end-to-end prototype

**Current state:** The dummy policy can be chunked, indexed, retrieved, and used by the locally running application.

**Next task:** Validate every Streamlit mode and FastAPI endpoint before implementing PDF ingestion.

## Daily Update Template

Copy this section for each work session:

```markdown
### DD Month YYYY

#### HH:MM AM/PM–HH:MM AM/PM — Task name

- [x] Completed:
- [~] In progress:
- [!] Blocked:
- [ ] Next:

Notes:
- Decisions made:
- Files changed:
- Test results:
```
