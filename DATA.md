# Data layout

BeemaSathi reads official Indian insurer and IRDAI documents. The PDFs are
third-party material with no redistribution licence, so they are **not** in this
repository. What is committed is the pipeline output needed to inspect and
re-score the corpus without re-downloading it.

## What is committed

| Path | Content |
| --- | --- |
| `data/parsed/` | Per-document extracted text, one file per PDF |
| `data/chunks/` | Chunk JSON per document, including page and section metadata |
| `data/chunks/ingest_index.json` | Ingest ledger: source key, file, document type, chunk count |
| `eval/reports/` | Dated evaluation reports and the latest run |

## What is not committed

| Path | Why | How to rebuild |
| --- | --- | --- |
| `data/raw_policies/` | Insurer PDFs, ~1.1 GB, no redistribution licence | `./appctl.sh scrape` |
| `data/irdai/` | IRDAI regulation PDFs | Download from irdai.gov.in |
| `data/chroma_db/` | Vector store, ~620 MB, derived | `./appctl.sh index` |
| `venv/` | Local virtualenv | `python3 -m venv venv && ./venv/bin/pip install -r requirements.txt` |
| `logs/` | Runtime output | Generated at run time |

## Rebuilding from scratch

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./appctl.sh scrape     # fetch official documents into data/raw_policies/
./appctl.sh index      # parse, chunk and embed into data/chroma_db/
./appctl.sh run        # Streamlit UI
```

Some insurer sites block automated collection (see `scrapers/hybrid_scraper.py
--manual-guide` for the manual fallback). A rebuild therefore reproduces the
pipeline, not byte-identical contents.

## Provenance

Every downloaded document has a sidecar `.meta.json` recording insurer, product,
document type, source URL and SHA-256. The manifest is the authority for where a
clause came from. Questions about a specific clause should be resolved against
the original PDF and its UIN, not against the extracted text in this repository.
