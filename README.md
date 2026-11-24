# 🛡️ BeemaSathi

Your AI companion for Indian health insurance — understand policies, predict claim rejection, and never get surprised by hidden clauses.

**Project site:** https://satishify.github.io/beemasathi/ — overview, architecture, data governance, evaluation design and the four-week plan.

## Quick Start

### 0. Choose a model backend

Answers come from either the local Ollama model or the hosted group API. The
hosted endpoint is used when `.env` contains a base URL and key:

```bash
cp .env.example .env    # then fill in your group values
```

```ini
BEEMASATHI_LLM_BASE=https://103.42.50.41:1032/v1   # :1031–:1034, by group
BEEMASATHI_LLM_KEY=your-group-key
BEEMASATHI_LLM_MODEL=Qwen/Qwen3-8B
BEEMASATHI_LLM_CA=/absolute/path/to/vm06-ca.crt
```

With those set, `appctl.sh` skips Ollama entirely and does not load a local
model. Without them, it falls back to the local Ollama setup below.

### 1. Install Ollama + Mistral 7B
```bash
brew install ollama
ollama pull mistral:7b
```

### 2. Setup Python Environment
```bash
cd /Users/satjo/Desktop/satish/SelfLearniningPrototypes/BeemaSathi
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Collect Policy Documents

The scraper discovers health and motor policy documents on insurer-owned domains.
It crawls each insurer's download hubs (and their sitemaps, so it survives URL
changes), classifies every PDF by line of business and document type, keeps only
the newest revision of each product, then downloads and validates it.

Always start with a dry run to see what would be collected:

```bash
python scrapers/hybrid_scraper.py --dry-run
```

Then collect for real. Use `--limit` on the first run, since a full pass finds
several thousand documents:

```bash
# health policy wordings only, capped at 50 files
python scrapers/hybrid_scraper.py --line health --doc-type policy_wording --limit 50

# motor documents from one insurer
python scrapers/hybrid_scraper.py --insurer tata_aig --line motor

# everything the registry can reach (large download)
python scrapers/hybrid_scraper.py
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--list-insurers` | Show the registry, supported lines, and known blockers |
| `--insurer KEY` | Restrict to one insurer; repeat for several |
| `--line health\|motor\|all` | Line of business; default is health and motor |
| `--doc-type policy_wording\|cis\|prospectus\|brochure` | Document types to keep |
| `--limit N` / `--per-insurer-limit N` | Cap downloads |
| `--keep-all` | Keep older revisions instead of only the newest |
| `--include-forms` | Also collect claim and proposal forms |
| `--dry-run` | Discover and report only |

Every run writes two files: `data/raw_policies/discovery_report.json` (what was
found, what was skipped, which insurers are blocked) and
`data/raw_policies/manifest.json` (what was downloaded, with SHA-256 hashes).
Each PDF also gets a `.meta.json` sidecar so `./appctl.sh index` knows the
insurer, policy, and document type without any manual tagging.

The scraper only fetches insurer-owned hostnames, respects `robots.txt`
(failing closed when it cannot be read), rate-limits each host, validates PDF
signatures and size, and never auto-ingests anything.

Some insurers cannot be scraped and need manual downloads: ICICI Lombard, Care
Health, and Star Health block automated clients, Zurich Kotak renders its
download list in JavaScript, and IRDAI disallows all crawling in `robots.txt`.
Run `python scrapers/hybrid_scraper.py --manual-guide` for the fallback steps.

### 4. Parse & Index
Dropping a file into `data/` does not update the chat. After adding or replacing a PDF/TXT, tell the system to chunk and embed it:

```bash
./appctl.sh index
```

That scans `data/raw_pdfs/` and `data/raw_policies/`, indexes new or changed files, and skips unchanged ones. Use `./appctl.sh index --force` to re-index everything.

Preferred names so insurer and policy are inferred:

```text
data/raw_policies/hdfc_ergo/optima_secure.pdf
data/raw_pdfs/star_health__comprehensive__policy_wording.pdf
```

Or add a sidecar next to the file, for example `optima_secure.meta.json`:

```json
{"insurer": "hdfc_ergo", "policy_name": "optima_secure", "document_type": "policy_wording"}
```

To index one known PDF by hand:

```bash
python rag/ingest_pdf.py "data/raw_pdfs/policy.pdf" \
  --insurer star_health \
  --policy-name comprehensive \
  --document-type policy_wording
```

Use `--document-type brochure` for marketing brochures. Brochures remain clearly
identified in retrieval metadata and UI citations because they do not replace the
full policy wording. Re-running the command replaces that source instead of
creating duplicate chunks.

### 5. Run the App
```bash
# Streamlit UI
streamlit run ui/app.py

# OR FastAPI backend
python api/main.py
```

Both front ends call the same agent layer: an intent router builds a typed plan
(coverage, comparison, claim risk, scope, safety, security), hybrid retrieval
fuses dense + BM25 + document-type authority, a critic verifies every number
against the retrieved clauses, and each request writes a redacted trace.

### Run with Docker

The vector store is mounted at runtime instead of baked into the image, because
third-party policy PDFs and `data/chroma_db/` are not committed.

```bash
docker compose up --build
# API     -> http://127.0.0.1:8000
# UI      -> http://127.0.0.1:8501
```

`docker compose` reads `.env` for the model backend, mounts `./data` read-only,
and stores traces in `./logs`.

### Debug Mode

Debug diagnostics are disabled by default. Enable them only for local development:

```bash
BEEMASATHI_DEBUG=1 streamlit run ui/app.py
```

An isolated right-side runtime panel then shows:

- Configured and actively loaded Ollama model
- Retrieval and generation timings
- Retrieved chunk count and similarity distances
- Ollama token and inference metrics
- Loaded model size, reported VRAM allocation, and Ollama process RAM

On Apple Silicon, CPU and GPU use unified memory, so the displayed RAM and VRAM
figures can overlap. Debug details do not include the full prompt or retrieved text,
and no developer information is added to the chat history.

## Evaluation and Safety

`eval/gold_cases.json` holds 100 cases: 74 existing tests plus 26 customer-style
questions grounded in the corresponding raw policy PDFs and indexed chunks.
The additions cover health benefits, claim timing, optional covers, motor damage,
No Claim Bonus, IDV and theft/repair conditions. The set also includes brochure,
safety, knowledge-gap and red-team cases.

```bash
python eval/judge.py                # retrieval + answer + judge, writes eval/reports/
python eval/retrieval_eval.py       # retrieval only, fast
python eval/compare_prompts.py      # baseline vs zero_shot vs structured
python eval/mine_gold_cases.py      # reproducible miner for new fact cases
```

The full 100-case judged run on the hosted Qwen3-8B backend (23 Nov 2026) scores
**85/100 pass**, doc recall@8 = 0.975 (79/81 labelled), supporting-clause recall@8
= 0.912 (73/80, re-measured 24 Nov 2026 after the corpus rebuild), and zero
answers that were right without the expected source.
Twelve of the 15 failures trip a hard rule while the judge accepts the answer, so
the count is conservative. Hybrid retrieval raised document recall from 0.60 to
0.975, and all nine red-team prompts are refused before the model is called.

Guardrails run before the model: banned fraud advice and the security patterns in
`config.py` return a fixed refusal without an LLM call, so prompt injection and
secret extraction cannot reach the backend. Traces (`logs/traces/*.jsonl`) hash
the question, keep a redacted preview, and never store the prompt, retrieved text,
or credentials.

CI (`.github/workflows/ci.yml`) compiles every module and runs the unit tests:
router, critic, rate limiter, redaction, security guardrail, and gold-set schema.

## Project Structure
```
BeemaSathi/
├── data/
│   ├── raw_pdfs/          # Downloaded policy PDFs
│   ├── parsed/            # Parsed markdown
│   ├── chunks/            # JSON chunks
│   └── chroma_db/         # Vector database
├── rag/
│   ├── chunker.py         # Semantic chunking
│   ├── ingest_pdf.py      # PDF extraction, chunking, and indexing
│   ├── embedder.py        # ChromaDB + BGE embeddings
│   ├── bm25.py            # Pure-Python Okapi BM25 index
│   ├── retriever.py       # Dense + BM25 + authority fusion
│   └── generator.py       # Hosted/local LLM, prompt strategies, safety
├── agents/
│   ├── router.py          # Intent → typed task plan
│   ├── orchestrator.py    # Plan, fan-out, verify, answer
│   ├── critic.py          # Number verification, authority resolution
│   └── claim_predictor.py # Rule-based rejection predictor
├── observability/
│   └── trace.py           # Redacted JSONL traces and summary
├── eval/                  # Gold set, judge, miners, prompt comparison
├── tests/                 # Unit tests (no heavy model deps)
├── scrapers/              # Insurer-specific scrapers
├── api/                   # FastAPI backend with token-bucket rate limit
├── ui/                    # Streamlit frontend
├── Dockerfile             # API image
├── docker-compose.yml     # API + UI
└── config.py              # Central configuration
```

## Roadmap
- [x] RAG pipeline with local LLM
- [x] Claim rejection predictor
- [x] Safety/alignment guardrails
- [x] Hybrid retrieval with evidence verification
- [x] Agentic routing, redacted traces, rate limiting
- [x] Docker, CI, and a red-team evaluation suite
- [ ] Hospital cashless network validator
- [ ] Policy PDF upload + "Gotcha Scanner"
- [ ] Ombudsman complaint drafter
- [ ] Multilingual support (Hinglish, Hindi)

## Disclaimer
This is an informational tool. It does not provide legal or financial advice. Always verify with your insurer before making decisions.
