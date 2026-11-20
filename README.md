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
│   ├── retriever.py       # Hybrid search
│   └── generator.py       # Ollama + safety
├── agents/
│   └── claim_predictor.py # Rule-based rejection predictor
├── scrapers/              # Insurer-specific scrapers
├── api/                   # FastAPI backend
├── ui/                    # Streamlit frontend
└── config.py              # Central configuration
```

## Roadmap
- [x] RAG pipeline with local LLM
- [x] Claim rejection predictor
- [x] Safety/alignment guardrails
- [ ] Hospital cashless network validator
- [ ] Policy PDF upload + "Gotcha Scanner"
- [ ] Ombudsman complaint drafter
- [ ] Multilingual support (Hinglish, Hindi)

## Disclaimer
This is an informational tool. It does not provide legal or financial advice. Always verify with your insurer before making decisions.
