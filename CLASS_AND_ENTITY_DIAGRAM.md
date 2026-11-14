# BeemaSathi — class and entity diagrams

**Visual view:** open [`CLASS_AND_ENTITY_DIAGRAM.html`](CLASS_AND_ENTITY_DIAGRAM.html) in a browser.

Mermaid source (open with a Mermaid preview / VS Code Mermaid extension):

- [`class_diagram.mmd`](class_diagram.mmd)
- [`entity_diagram.mmd`](entity_diagram.mmd)
- [`pipeline_diagram.mmd`](pipeline_diagram.mmd)

They reflect the current Python modules, not an ORM.

---

## 1. Class diagram

```mermaid
classDiagram
    direction TB

    class Insurer {
        +str name
        +set hosts
        +list hubs
        +list sitemaps
        +set lines
        +str note
    }

    class Fetcher {
        +Session session
        +bool respect_robots
        +robots_allows(url)
        +get(url)
    }

    class PolicyVectorStore {
        +PersistentClient client
        +Collection collection
        +add_chunks(chunks)
        +replace_source_chunks(chunks)
        +search(query, insurer, policy_name, chunk_type)
        +get_catalog()
        +get_stats()
    }

    class PolicyRetriever {
        +PolicyVectorStore store
        +retrieve(query, insurer, policy_name, n_results)
        -_rerank(results, query)
        -_infer_line(query)
        -_infer_chunk_type(query)
    }

    class BimaSaathiGenerator {
        +str model
        +generate(chunks, question, catalog)
        -_select_chunks(chunks, question)
        -_safety_check(text)
        -_call_ollama(prompt, messages)
        -_catalog_fact_text(grouped, regulations)
    }

    class ClaimPredictor {
        +dict rules
        +predict(...) dict
        -_get_documents(procedure)
        -_build_summary(score, failed)
    }

    class QueryRequest {
        +str question
        +str insurer
        +str policy_name
    }

    class QueryResponse {
        +str answer
        +bool safe
        +list sources
    }

    class ClaimPredictRequest {
        +str insurer
        +str policy_name
        +bool has_ped
        +float room_rent_per_day
        +float sum_insured
    }

    class StreamlitApp {
        <<ui/app.py>>
        +retriever
        +generator
        +predictor
        +session_state messages
    }

    class FastAPIApp {
        <<api/main.py>>
        +POST /ask
        +POST /predict
        +GET /health
    }

    class SyncKnowledge {
        <<rag/sync_knowledge.py>>
        +discover_files()
        +infer_metadata(path)
        +ingest_pdf(path, store)
        +ingest_text(path, store)
    }

    class HybridScraper {
        <<scrapers/hybrid_scraper.py>>
        +run(insurer_keys, lines)
        +discover_documents(fetcher, key)
        +download_document(fetcher, doc)
    }

    class EvalJudge {
        <<eval/judge.py>>
        +run_eval(case_ids)
        +_rule_score(case, result)
        +_call_judge(generator, case, result)
    }

    HybridScraper --> Insurer : registry
    HybridScraper --> Fetcher : uses
    SyncKnowledge --> PolicyVectorStore : writes
    PolicyRetriever --> PolicyVectorStore : reads
    PolicyRetriever --> BimaSaathiGenerator : chunks
    StreamlitApp --> PolicyRetriever
    StreamlitApp --> BimaSaathiGenerator
    StreamlitApp --> ClaimPredictor
    FastAPIApp --> PolicyRetriever
    FastAPIApp --> BimaSaathiGenerator
    FastAPIApp --> ClaimPredictor
    FastAPIApp --> QueryRequest
    FastAPIApp --> QueryResponse
    FastAPIApp --> ClaimPredictRequest
    EvalJudge --> PolicyRetriever
    EvalJudge --> BimaSaathiGenerator
    HybridScraper ..> SyncKnowledge : PDFs then index
```

---

## 2. Entity-relationship diagram

Logical stores: files under `data/`, ChromaDB, and JSON sidecars / manifests.
There is no SQL database.

```mermaid
erDiagram
    KNOWLEDGE_SOURCE ||--o{ POLICY_DOCUMENT : publishes
    KNOWLEDGE_SOURCE ||--o{ CATALOG_ENTRY : summarized_as
    POLICY_DOCUMENT ||--o| DOCUMENT_SIDECAR : described_by
    POLICY_DOCUMENT ||--o{ POLICY_CHUNK : split_into
    POLICY_DOCUMENT ||--o| INGEST_INDEX_ROW : tracked_by
    POLICY_CHUNK }o--|| CATALOG_ENTRY : rolls_up_to
    POLICY_CHUNK }o--o{ GENERATED_ANSWER : retrieved_for
    QUERY ||--o| GENERATED_ANSWER : produces
    QUERY }o--o| KNOWLEDGE_SOURCE : filters
    CLAIM_CASE ||--o| CLAIM_ASSESSMENT : scored_as
    CLAIM_CASE }o--o| KNOWLEDGE_SOURCE : about
    GOLD_CASE ||--o| EVAL_RESULT : judged_as
    GOLD_CASE }o--o| POLICY_DOCUMENT : validated_against
    SCRAPE_RUN ||--o{ DISCOVERY_CANDIDATE : finds
    DISCOVERY_CANDIDATE }o--o| POLICY_DOCUMENT : may_download

    KNOWLEDGE_SOURCE {
        string key PK "star_health, tata_aig, irdai"
        string name
        string base_url
        string kind "insurer | regulator"
    }

    POLICY_DOCUMENT {
        string path PK "data/raw_policies/{insurer}/{file}"
        string insurer FK
        string policy_name
        string document_type "policy_wording | brochure | regulation"
        string line_of_business "health | motor | ..."
        string source_url
        string sha256
        int size_bytes
    }

    DOCUMENT_SIDECAR {
        string path PK "{pdf}.meta.json"
        string insurer
        string policy_name
        string document_type
        string line_of_business
        string source_url
        string uin
    }

    POLICY_CHUNK {
        string id PK "source_i in Chroma"
        string source "insurer_policy_doctype"
        string insurer
        string policy_name
        string section
        string chunk_type
        int page
        string document_type
        string line_of_business
        string text
        float distance "query-time only"
    }

    CATALOG_ENTRY {
        string insurer
        string policy_name
        string document_type
        string source_file
        int chunks
    }

    INGEST_INDEX_ROW {
        string file PK "relative path"
        string sha256
        string insurer
        string policy_name
        string document_type
        int chunks
        datetime indexed_at
    }

    QUERY {
        string question
        string insurer
        string policy_name
        string mode "ask | predict | eval"
    }

    GENERATED_ANSWER {
        string answer
        bool safe
        json sources
        json model_metrics
        string rejection_risk
    }

    CLAIM_CASE {
        string insurer
        string policy_name
        string hospital_name
        bool is_network_hospital
        bool has_ped
        int months_since_policy
        float room_rent_per_day
        float sum_insured
    }

    CLAIM_ASSESSMENT {
        int risk_score "0-100"
        list failed_checks
        list passed_checks
        string summary
    }

    GOLD_CASE {
        string id PK
        string category
        string question
        string gold_notes
        list must_include_any
        bool expect_refuse
        bool expect_block
    }

    EVAL_RESULT {
        string case_id FK
        bool passed
        json rules
        json judge_scores
        string answer
    }

    SCRAPE_RUN {
        datetime finished_at
        list selected_insurers
        string mode
        int downloaded
    }

    DISCOVERY_CANDIDATE {
        string url
        string insurer
        string line
        string doc_type
        string policy
        string uin
        bool stale_marker
    }
```

---

## 3. How the pieces connect

```
official insurer site
        │
        ▼
hybrid_scraper ──► data/raw_policies/{insurer}/*.pdf + .meta.json
        │
        ▼
sync_knowledge / ingest_pdf ──► data/chunks/*.json
        │                         data/chunks/ingest_index.json
        ▼
PolicyVectorStore (ChromaDB) ──► POLICY_CHUNK + CATALOG_ENTRY
        │
        ├── PolicyRetriever.retrieve()
        │         │
        │         ▼
        └── BimaSaathiGenerator.generate() ──► Streamlit / FastAPI / eval judge
                                                      │
                                                      └── ClaimPredictor.predict()
```

| Store | Path | Entity |
| --- | --- | --- |
| Raw PDFs / HTML text | `data/raw_policies/`, `data/raw_pdfs/` | POLICY_DOCUMENT |
| Sidecar | `*.meta.json` | DOCUMENT_SIDECAR |
| Scrape manifest | `data/raw_policies/manifest.json` | SCRAPE_RUN, DISCOVERY_CANDIDATE |
| Ingest ledger | `data/chunks/ingest_index.json` | INGEST_INDEX_ROW |
| Vector index | `data/chroma_db/` | POLICY_CHUNK |
| Gold set | `eval/gold_cases.json` | GOLD_CASE |
| Eval report | `eval/reports/latest.json` | EVAL_RESULT |
