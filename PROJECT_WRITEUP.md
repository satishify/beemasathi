# BeemaSathi
## An Evidence-Grounded Multi-Agent Copilot for Indian Health and Motor Insurance

**Programme:** Executive Post Graduate Programme in Generative AI and Agentic AI, IIT Kharagpur
**Selected Track:** Track B — API-Side (FastAPI service; Streamlit client for demonstration)
**Team Size:** 6 Members
**Duration:** 4 Weeks

---

### 1. Problem Statement

Indian retail insurance is sold as a short brochure and governed by a long policy wording. The information a customer actually needs — waiting periods, room-rent caps, sub-limits, co-pay, pre-existing disease conditions, exclusions, cashless rules — is buried in 30–100 page PDFs that differ by insurer, product, UIN, and revision year. Brochures and comparison sites summarise the product but are not the contract.

This gap has measurable consequences:

- A room-rent choice above the policy cap proportionately reduces every other claim head, not just the room charge.
- A pre-existing condition treated inside the waiting period is rejected even when the hospitalisation itself is covered.
- A delayed intimation or a non-network hospital changes a cashless claim into a reimbursement claim with a different document set.
- A customer reading a brochure reasonably concludes a benefit exists that the wording excludes.

General-purpose chatbots make this worse: they answer confidently from generic insurance knowledge, cannot tell a brochure from a binding wording, and cannot say "this product is not in my knowledge base". The user receives fluent, unsourced advice about a contract the model has never read.

BeemaSathi addresses this with an evidence-grounded copilot that retrieves from official insurer and IRDAI documents, separates binding wordings from marketing summaries, cites the exact clause and page, refuses when the product is not loaded, and estimates claim-rejection risk only from evidence it can show.

### 2. Project Objectives

- **Curate a governed knowledge base** of 800–1,200 normalized documents from official insurer and IRDAI sources, covering at least 10 insurers and at least 75 products with a complete policy wording. Each product is keyed by UIN, and each document by content hash, document type and revision.
- **Implement hybrid retrieval** combining BM25 and dense vector search, with authority-aware reranking so that a policy wording always outranks a CIS, prospectus or brochure for coverage questions.
- **Build a multi-agent workflow** for intent routing, retrieval, clause analysis, claim-risk assessment, verification and response composition, with persistent state, typed tool outputs, retries and tracing.
- **Generate clause-level citations** containing insurer, product, UIN, document type, section, page, source URL and retrieval date, and flag every answer that relies on a non-binding summary.
- **Refuse correctly** when an insurer, product, or clause is absent, when a document is stale, or when the question requires legal or financial advice.
- **Evaluate rigorously** on a frozen 80-question set created before the agentic layer, reporting retrieval recall, factual accuracy, citation correctness, refusal quality, claim-risk agreement, hallucination rate, latency and cost, against four baselines and three ablations.

### 3. Proposed Architecture

#### 3.1 Current baseline (already built)

A working local prototype exists and provides the substrate for the four-week build:

| Asset | Current state (at the time of writing) |
| --- | --- |
| Corpus | 923 indexed files, 45,844 chunks, 927 PDFs (1.9 GB), 8 insurers plus IRDAI |
| Document mix | 362 policy wordings, 559 brochures/CIS/prospectus, plus IRDAI regulations |
| Collection | Official-domain-only crawler, robots fail-closed, SHA-256 manifest, sidecar metadata |
| Retrieval | Dense (BGE-small-en-v1.5) + keyword chunk-type routing + heuristic rerank |
| Generation | `qwen2.5:7b` via Ollama, constitutional prompt, refusal rules, leak sanitisation |
| Interfaces | Streamlit client, FastAPI (`/ask`, `/predict`, `/insurers`, `/health`) |
| Evaluation | 11 gold cases with rule checks and an LLM judge (11/11 at last run) |
| Claim risk | Rule-based predictor, 4 of 6 configured rules implemented |

An internal review of this baseline identified four corpus-integrity defects that must be closed in Week 1 before any evaluation number is meaningful: residual demo data in the production collection, vector IDs that collide across documents, a catalogue built from an ingest ledger that has drifted from the vector store, and lossy document-type mapping (endorsements labelled as wordings; CIS and prospectus collapsed into brochure). The detailed defect list, evidence and remediation plan are maintained in `PROJECT_REVIEW.md`. Closing these is Gate G1 of the plan in Section 6.

#### 3.2 Target architecture

**Input:** a free-text question, a comparison request, a claim scenario, or an optional user policy PDF.

```
                       ┌──────────────────────────────────────────┐
   question / claim ──►│  Intake & Router Agent                   │
                       │  intent, insurers, products, line, risk  │
                       └───────────────┬──────────────────────────┘
                                       │  typed task plan
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
   │ Policy Retrieval   │  │ Product & Authority │  │ Claim-Risk Agent     │
   │ Agent              │  │ Resolver            │  │ (health / motor)     │
   │ BM25 + dense,      │  │ UIN, revision,      │  │ rule engine grounded │
   │ filters, fan-out,  │  │ wording > CIS >     │  │ in retrieved clauses │
   │ rerank             │  │ prospectus > brochure│ │ + missing-input list │
   └─────────┬──────────┘  └─────────┬──────────┘  └──────────┬───────────┘
             │                       │                        │
             └───────────────┬───────┴────────────────────────┘
                             ▼
                 ┌──────────────────────────┐
                 │ Clause Analyst Agent     │
                 │ clause extraction, table │
                 │ parsing, section/page,   │
                 │ effective date, conflicts│
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Critic / Verifier Agent  │
                 │ entailment, citation     │
                 │ check, refusal check,    │
                 │ safety, injection scan   │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Response Composer        │
                 │ short answer + bullets + │
                 │ evidence + disclaimer    │
                 └──────────────────────────┘

   Side systems: persistent run state, tool registry (Pydantic schemas),
   retry/budget policy, trace store, freshness monitor, eval harness.
```

**Agents and tools**

1. **Intake & Router Agent** — classifies intent (coverage, exclusion, waiting period, sub-limit, claim process, comparison, claim risk, catalogue/gap, out of scope), extracts insurer and product mentions with word-boundary aliases, detects line of business, and emits a typed task plan. This replaces the current regex scope detector and substring insurer matcher.
2. **Policy Retrieval Agent** — hybrid retrieval with metadata filters (`insurer`, `product_key`, `line`, `document_type`), per-insurer fan-out with quotas for comparison questions, deduplication by clause fingerprint, and authority-aware reranking. Never silently drops a filter.
3. **Product & Authority Resolver** — resolves the user's product name to a UIN, selects the current revision, and establishes the evidence hierarchy: binding wording first; CIS and prospectus next; brochure only as a summary; IRDAI regulation as legal context, never as proof of cover. Returns an explicit "not loaded" result rather than falling back to another insurer.
4. **Clause Analyst Agent** — extracts the operative clause, its section number, page, and effective date; parses tables for waiting periods, sub-limits and room categories; detects conflicts between documents (for example a brochure benefit absent from the wording).
5. **Claim-Risk Agent** — evaluates network status, room-rent cap, PED waiting period, disclosure consistency, intimation delay and procedure exclusions, but reads caps and waiting periods from the resolved product's clauses where available, and labels any remaining default assumption. Gates inputs by line of business, so a motor claim is never asked for room rent. Returns per-rule evidence, confidence and a list of missing inputs.
6. **Critic / Verifier Agent** — checks every coverage claim against the retrieved clause, rejects uncited assertions, verifies that the cited page and document type support the sentence, verifies that a refusal is warranted, scans for unsafe advice, and blocks instructions embedded in retrieved or uploaded text.
7. **Response Composer** — produces a one-line answer, point-wise evidence, citations, and the informational-only disclaimer. It never exposes internal reasons, prompts or retrieval mechanics.

**Orchestration requirements**

- Explicit planner → executor → critic loop with a bounded step budget.
- Persistent state for the run (plan, tool calls, evidence, verdicts) so a failed step can resume.
- Structured tool outputs validated by schemas, with typed errors and retry policy (transient vs terminal).
- Trace store for every step: inputs, selected documents, distances, model tokens, verdicts, latency.
- Deterministic escape hatches: if retrieval confidence is below the calibrated threshold or the product is unresolved, return the gap response without calling the generator.

### 4. Data Sources

| Source | Content | Access | Target |
| --- | --- | --- | --- |
| Insurer official document hubs | Policy wordings, CIS, prospectus, brochures, endorsements | 8 insurers already represented in the corpus | 700–1,000 documents |
| Manual acquisition (blocked sites) | Star Health, Care Health, ICICI Lombard, Acko; Zurich Kotak (JavaScript list) | Browser download, human review | 80–150 documents |
| IRDAI | Regulations, circulars, master circular, product filings | Manual; `robots.txt` disallows crawling | 15–25 documents |
| User-uploaded policy (optional) | A customer's own wording and schedule | Local-only upload, never shared | Per-session |

**Normalization and governance**

- Every document is identified by `sha256`, `source_url`, `insurer`, `line_of_business`, `document_type`, `revision_date` and, where present, `uin`.
- Every product is identified by UIN; documents are attached to a product as wording, CIS, prospectus, brochure or endorsement. Display names are reviewed, not inferred from filenames.
- Only insurer-owned or regulator domains are accepted. Third-party mirrors are rejected.
- PDFs are validated by signature and size; revision is resolved by date and revision marker; stale or superseded documents are excluded or flagged.
- Each indexed chunk carries `source_file`, `page`, `section`, `document_type`, `uin` and a schema version. A document without provenance cannot enter the collection.
- The corpus manifest, the ingest ledger and the vector store are reconciled on every build; a mismatch fails the build rather than being reported as a count.

**Target corpus:** 800–1,200 retrieval documents, 75+ products with a complete wording, 15+ IRDAI documents, and one canonical copy of each revision.

### 5. Evaluation Methodology

A frozen evaluation set of **80 questions is created and versioned before the agentic layer is implemented**, so the agents cannot be tuned to the test.

**Question stratification**

| Category | Count | What it tests |
| --- | --- | --- |
| Coverage and exclusion facts | 20 | Clause retrieval and factual accuracy across at least 6 insurers |
| Waiting periods, PED, sub-limits, room rent, co-pay | 15 | Table and clause extraction on the hardest real questions |
| Claim process and cashless rules | 10 | Procedural accuracy and document lists |
| Two-insurer comparisons | 10 | Per-insurer retrieval and balanced synthesis |
| Gap and refusal cases | 10 | Unloaded insurers, missing products, out-of-scope questions |
| Safety and fraud prompts | 5 | Hiding PED, backdating, fabricated bills |
| Prompt-injection and leakage | 5 | Instructions inside PDFs, excerpt/page-number leakage |
| Claim-risk scenarios | 5 | Health and motor fixtures with adjudicated outcomes |

**Metrics**

| Metric | Definition | Target |
| --- | --- | --- |
| Retrieval Recall@5 | Fraction of questions whose gold clause appears in the top 5 chunks | ≥ 0.85 |
| MRR | Mean reciprocal rank of the first gold chunk | ≥ 0.70 |
| Document-type precision | Share of coverage answers whose evidence is a binding wording when one exists | ≥ 0.90 |
| Answer factual accuracy | Rubric score against gold notes (2/1/0) | ≥ 90% of cases at 2 |
| Citation correctness | Cited page/section/doc type actually supports the sentence | ≥ 95% |
| Unsupported-claim rate | Coverage claims with no entailment in retrieved evidence | ≤ 3% |
| Refusal precision / recall | Correct refusals vs all refusals / vs all gap cases | ≥ 0.95 each |
| Claim-risk agreement | Risk band matches adjudicated fixture | ≥ 80% |
| Task completion | Runs producing a valid, cited answer or a valid refusal | ≥ 95% |
| Tool-call success | Successful typed tool calls / attempted calls | ≥ 98% |
| Injection resistance | Attacks blocked or safely ignored | 100% |
| Latency | p50 and p95 end-to-end, local inference | p95 ≤ 20 s RAG-only, ≤ 45 s agentic |
| Cost | Prompt and completion tokens, local GPU-seconds per answer | Reported, no regression vs baseline |

**Systems compared**

1. Single-prompt LLM, no retrieval.
2. Dense-only RAG (current baseline).
3. Hybrid RAG (BM25 + dense).
4. Hybrid RAG + reranking.
5. Full multi-agent system with critic.

**Ablations**

- Authority-aware reranking on/off (measures brochure contamination).
- Critic agent on/off (measures unsupported-claim rate).
- Per-insurer fan-out on/off (measures comparison quality).
- Calibrated distance threshold on/off (measures weak-evidence answers).

**Annotation protocol**

Two annotators label the 80 questions against the cited official PDFs; disagreements are adjudicated by a third; inter-annotator agreement is reported (Cohen's kappa). The LLM judge is a different model from the generator, is prompted with gold notes and retrieved clauses only, and 20% of its verdicts are human-audited. Rule checks always override the judge for refusal and safety cases.

**Failure taxonomy**

Every failed case is labelled as one of: retrieval miss, wrong document type, stale revision, product-resolution error, clause-extraction error, unsupported claim, citation error, incorrect refusal, over-answer, safety failure, injection failure, risk miscalibration, latency/budget failure. The distribution, not just the pass rate, is a Week 4 deliverable.

**Red-team suite**

15 attacks across three channels: user prompts (fraud, prompt extraction, role-play), poisoned PDFs (instructions in clause text, invisible text), and retrieval manipulation (questions crafted to surface a brochure or another insurer). Success means refusal, safe redirection, or a correct answer with no instruction following.

### 6. Four-Week Execution Plan

| Week | Workstream | Deliverable | Gate |
| --- | --- | --- | --- |
| 1 | Corpus integrity and evaluation freeze | Content-addressed document IDs and reindex; demo data purged; document-type mapping fixed; catalogue reconciled to the vector store; IRDAI scan root and current regulations ingested; frozen 80-question set with gold notes; baselines 1–2 measured | G1: zero chunks without provenance, zero collisions, catalogue equals vector store; 80 questions versioned |
| 2 | Hybrid retrieval and evidence quality | BM25 + dense index; authority-aware reranking; calibrated threshold; clause- and table-aware chunking with OCR fallback; UIN, URL, effective date and section in metadata; citation contract; Recall@5 and MRR reported | G2: Recall@5 ≥ 0.80 and wording share ≥ 0.85 on the frozen set (interim; final targets 0.85 and 0.90) |
| 3 | Agentic layer | Router, retrieval, authority resolver, clause analyst, claim-risk and critic agents; typed tool schemas; persistent state, retries, budgets; trace store; agentic vs RAG-only comparison | G3: task completion ≥ 95%, tool success ≥ 98%, critic catches injected instructions |
| 4 | Evaluation, hardening, delivery | Full evaluation and ablations; failure taxonomy; red-team report; observability dashboard; API and UI integration; deployment pack; 12-page report and 10-minute demo | G4: all Section 5 targets met or documented with a remediation plan |

### 7. Team Responsibilities

| Member | Responsibility |
| --- | --- |
| 1 | Domain specification, insurance SME liaison, 80-question gold set, annotation and adjudication |
| 2 | Prompting strategy, critic design, safety guardrails, red-team and responsible-AI report |
| 3 | Data acquisition and normalization, hybrid index, chunking, reranking, retrieval evaluation |
| 4 | Agent orchestration, tool schemas, persistent state, retries, tracing, claim-risk integration |
| 5 | Evaluation harness, metrics, ablations, observability dashboard, cost and latency analysis |
| 6 | FastAPI service, Streamlit client, citation UX, packaging, deployment, demo production |

### 8. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Insurer sites block automated access or change URLs | Corpus gaps | Official-domain crawler plus documented manual acquisition; provenance review; no third-party mirrors |
| Policy revisions silently supersede indexed documents | Wrong answers | Revision date and UIN in metadata; freshness monitor; scheduled re-crawl and diff alerts |
| Scanned or table-heavy PDFs extract poorly | Missing sub-limits and waiting periods | OCR fallback with an `ocr_derived` flag; table-aware extraction; extraction-quality metric per document |
| Brochure text contaminates coverage answers | Overstated benefits | Authority-aware reranking; wording-first policy; brochure share measured in evaluation |
| Hallucinated coverage or UIN | Direct financial harm | Refusal-first prompt, critic entailment check, citation-required answers, UIN verification against metadata |
| LLM judge is generous or biased | Inflated results | Different judge model, rule checks override, 20% human audit, disagreement reporting |
| Prompt injection through uploaded PDFs | Instruction hijacking | Delimited untrusted content, injection scan, critic check, 15-attack red-team suite |
| Local model latency on commodity hardware | Poor demo experience | Small context budget, cached embeddings, streaming, p95 target, optional hosted fallback kept out of the default path |
| Scope creep into legal or claim adjudication | Regulatory and ethical risk | Explicit non-goals, informational-only framing, human verification, no claim filing |

### 9. Responsible AI and Compliance

- **Positioning.** BeemaSathi is an informational reading aid, not legal, financial, or claim advice. Every answer carries that framing and points to the insurer's official wording.
- **Evidence first.** A coverage answer without a supporting clause is suppressed. A product that is not loaded produces a refusal, not an inference from another insurer.
- **Authority transparency.** Brochures, CIS and prospectus are labelled as summaries; IRDAI regulations are labelled as legal context, not proof of cover.
- **Human in the loop.** Claim-risk output is advisory, shows the rule and the clause behind each finding, lists missing inputs, and never recommends concealment or document manipulation.
- **Privacy.** Local inference by default; no chat or medical attributes persisted; uploaded policies stay on the machine; logs exclude policy text and personal data.
- **Language and accessibility.** The current embedding model is English-only; Hinglish and Hindi support is scoped as a post-MVP item with its own evaluation set rather than asserted in the prompt.
- **Auditability.** Every answer maps to a document, UIN, page, section, source URL, revision and retrieval date, so a reviewer can reproduce the evidence.

### 10. Expected Deliverables

- Working BeemaSathi application with a FastAPI service and a Streamlit demonstration client.
- Reproducible source repository with pinned dependencies, tests, and a one-command local start.
- Documented, versioned knowledge base with a manifest, SHA-256 hashes, UIN mapping, revision dates and review sign-off.
- Agent execution and tracing dashboard showing plans, tool calls, evidence, verdicts, tokens and latency.
- Evaluation report covering 80 frozen questions, five systems, four ablations, all Section 5 metrics, and a failure taxonomy.
- Responsible-AI and prompt-injection testing report with the red-team suite and results.
- 12-page technical report and a 10-minute demonstration.

**Expected outcome:** an evaluation-driven, evidence-grounded insurance copilot that demonstrates hybrid RAG, authority-aware retrieval, multi-agent orchestration with a critic, clause-level citations, honest refusal, and measurable claim-risk reasoning on real Indian policy documents — and that can state, with evidence, why it is right or why it does not know.

---

## Appendix A — Success criteria for the final demonstration

1. A coverage question returns a clause-level citation containing insurer, product, UIN, document type, page, section and source URL.
2. A question about an insurer with no loaded wording returns a clean refusal with the gap named.
3. A brochure-only product is answered with an explicit summary warning; where a wording exists, wording evidence is used.
4. A two-insurer comparison returns evidence from both insurers.
5. A claim scenario returns per-rule risk findings linked to clauses, or explicitly labelled defaults, and never gives unsafe advice.
6. A poisoned uploaded PDF cannot change system behaviour.
7. The trace dashboard reproduces every step of each demo answer.
8. The evaluation report shows the frozen 80-question results, baselines, ablations and failure distribution.

## Appendix B — Corpus record schema (target)

```json
{
  "document_id": "sha256:9f2c…",
  "insurer": "tata_aig",
  "product_key": "TATHLIP21292V022021",
  "product_name": "Critical Illness",
  "line_of_business": "health",
  "document_type": "policy_wording",
  "revision_date": "2024-03-01",
  "source_url": "https://www.tataaig.com/…",
  "retrieved_at": "2026-11-14T09:10:00Z",
  "sha256": "9f2c…",
  "pages": 26,
  "ocr_derived": false,
  "review": {"status": "approved", "reviewer": "member-1", "reviewed_at": "2026-11-15"}
}
```

## Appendix C — Baseline defect register

The measured baseline review (`PROJECT_REVIEW.md`) records 26 findings, including four trust-breaking defects (residual demo policy in the production index, colliding vector IDs, catalogue/vector-store drift, lossy document-type mapping), retrieval problems (exact-match policy filters that silently widen, brochure dominance, permissive distance threshold, substring insurer matching), and a disconnected claim predictor. These are the Week 1 gate, not hidden debt; the evaluation in Section 5 is only meaningful after they are closed.
