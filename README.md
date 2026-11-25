# 🛡️ BeemaSathi

An evidence-grounded copilot for Indian health and motor insurance. It answers only from official insurer and IRDAI documents, cites the clause and page behind every coverage statement, separates binding wordings from marketing brochures, and refuses when a product is not loaded.

**Project site:** https://satishify.github.io/beemasathi/
**Complete project guide:** [docs/PROJECT.md](docs/PROJECT.md) — architecture, agent layer, data governance, evaluation, plan and operations.

## What it does

- **Hybrid retrieval** over 48,572 clause chunks from 923 official source files, fused with BM25 and reranked by document authority.
- **Grounded answers** with insurer, product, document type, section and page; a critic verifies every number and UIN against the retrieved clauses.
- **Honest gaps** — an unloaded insurer or product returns a named refusal, never an answer borrowed from another insurer.
- **Claim-risk check** from network status, room-rent cap, PED waiting period and intimation timing.
- **Guardrails** for fraud advice, prompt/secret/PII extraction, role-play and document-borne injection, refused before the model is called.

## Quick start

One model backend is required: a hosted OpenAI-compatible endpoint, or local Ollama.

```bash
git clone https://github.com/satishify/beemasathi.git
cd beemasathi
./beemasathi.sh
```

The launcher checks Python, creates the venv and installs whatever is missing,
then offers a menu to configure the hosted endpoint, start and stop the app.
Choose `4` to paste the hosted base URL and key; they are saved to `.env`,
which is gitignored, and the connection is tested on save. Press `1` to start
at http://127.0.0.1:8501.

The same launcher runs on Windows: double-click `Start BeemaSathi.bat`. On
macOS, double-click `Start BeemaSathi.command`. Both start the menu without a
terminal command. The logic lives in `launcher.py`, which uses the Windows
`venv\Scripts` layout and the macOS/Linux `venv/bin` layout automatically.

The hosted endpoint is tried first. If it is unreachable, rejects the key, or
fails mid-request, the app answers with local Ollama (`qwen2.5:7b`) when it is
installed, and records the fallback in the request metrics. Leave `.env` blank
to run on Ollama only. Loading the index locally also needs
`BAAI/bge-small-en-v1.5`, which `sentence-transformers` fetches on first use.

The same tasks are available without the menu:

```bash
./beemasathi.sh check     # report what is installed, install nothing
./beemasathi.sh config    # edit the hosted URL, key and model
./beemasathi.sh start     # or stop, restart, status
```

On Windows, run the same commands through the batch file:
`Start BeemaSathi.bat start`, `Start BeemaSathi.bat config`, and so on.

Without `.env` the app uses Ollama (`ollama pull qwen2.5:7b`). The corpus and vector store are not committed; build them with:

```bash
./venv/bin/python scrapers/hybrid_scraper.py
./appctl.sh index
```

The API runs separately with `python api/main.py`, or both services with `docker compose up --build`. Full setup, configuration, scraper flags and debugging notes are in [docs/PROJECT.md](docs/PROJECT.md#9-running-locally).

## Evaluation

100 frozen gold cases. Last measured on the hosted Qwen3-8B backend:

- 85/100 passing, 0 ungrounded passes
- document recall@8 0.975, supporting-clause recall@8 0.912
- 2.8 s mean answer latency, 296k prompt tokens for the full run
- 9/9 red-team refusals

```bash
python eval/judge.py            # retrieval + answer + judge
python eval/retrieval_eval.py   # retrieval only
python eval/compare_prompts.py  # prompt strategy comparison
```

Method, failure taxonomy and limits: [docs/evaluation.html](docs/evaluation.html).

## Documentation

| Document | Content |
| --- | --- |
| [docs/PROJECT.md](docs/PROJECT.md) | Complete project guide |
| [docs/architecture.html](docs/architecture.html) | Shipped architecture and stage-by-stage pipeline |
| [docs/data.html](docs/data.html) | Corpus composition, governance and extraction limits |
| [docs/evaluation.html](docs/evaluation.html) | Gold set, metrics, results and red team |
| [docs/plan.html](docs/plan.html) | Four-week plan, gates, risks and deliverables |
| [eval/README.md](eval/README.md) | How cases are scored |

## Structure

```text
launcher.py      cross-platform launcher: environment check, hosted config, start and stop
beemasathi.sh    macOS/Linux entry point (runs launcher.py)
Start BeemaSathi.bat   Windows entry point (runs launcher.py)
appctl.sh        compatibility wrapper: start/stop/restart/status/index
agents/          intent router, orchestrator, critic, claim predictor
api/             FastAPI service with token-bucket rate limit
ui/              Streamlit client
.streamlit/      theme: palette, rounded surfaces, Inter and Plus Jakarta typography
assets/          brand mark used by the sidebar logo
rag/             PDF extraction, chunking, embeddings, Chroma, BM25, retriever, generator, guard
observability/   redacted JSONL traces
eval/            gold set, judge, miners, prompt comparison, reports
tests/           unit tests without heavy model dependencies
scrapers/        insurer-specific discovery and download
docs/            project guide and HTML pages
data/            pipeline output (PDFs and vector store excluded)
config.py        central configuration
```

## Disclaimer

This is an informational tool. It does not provide legal or financial advice, sell insurance, or adjudicate claims. Always verify with your insurer before making decisions.
