# 🏛️ Fintech RegTech Compliance Copilot

A RAG-based compliance assistant that audits architecture and payment-flow descriptions against **PCI DSS v4.0.1**, returning a risk classification, a grounded assessment, and citations to the specific requirements that apply.

Describe a system in plain English — *"we log full card numbers including CVV to an unencrypted debug file"* — and get back a cited, requirement-level compliance assessment.

> Built as a hands-on study in AI product development: an agentic-coded RAG system where the interesting engineering is in retrieval quality and evaluation, not the UI. Not validated for production compliance use.

**Product Management Case Study:** This repository serves as an end-to-end blueprint in AI Product Development, focusing heavily on minimizing time-to-value (TTV), implementing rigorous LLMOps evaluation harnesses, and mitigating hallucination risks within highly regulated domains.

---

## 💼 Business Context & Product Strategy
In fast-moving fintech organizations, product feature velocity is frequently bottlenecked by manual compliance and security reviews, which can take weeks per sprint. This application acts as a defensive guardrail for product managers and architects, allowing them to instantly discover compliance violations *before* writing a single line of code.

### Core Product Guardrails:
1. **Zero-Hallucination Mandate:** The model is strictly prohibited from utilising its parametric pre-trained data to answer queries. If an explicit rule is missing from the underlying vector database, the application drops a programmatic fallback warning.
2. **Asymmetrical Risk Optimization:** In compliance, saying a violation is "Compliant" can result in regulatory shutdowns. The system's prompt architecture is explicitly weighted to favour conservative warnings over false compliance confirmations.

---

## What it does
- Accepts a natural-language architecture / payment-flow / feature description.
- Decomposes it into focused compliance sub-questions, retrieves the relevant PCI DSS v4.0.1 requirement passages, and generates an assessment **grounded in and cited to** the source standard.
- Returns: **risk classification** (Compliant / Warning / Non-Compliant), **assessment**, **citations** to specific requirements, and a **grounding-confidence** indicator.

## 🧱 Technical System Architecture
The application leverages a decoupled, API-first design lifecycle to support scalability and omnichannel access (e.g., integrating into Streamlit dashboards, internal CLIs, or CI/CD build pipelines).

## Architecture
```
Streamlit UI  ──HTTP──▶  FastAPI (auth)  ──▶  RAG engine
                                              ├─ query decomposition (Claude)
                                              ├─ embeddings (OpenAI text-embedding-3-small)
                                              ├─ retrieval (ChromaDB, cosine)
                                              ├─ merge (top-k per sub-query)
                                              └─ assessment + citations (Claude)
```

**Key design choices**
- **Query decomposition / multi-query retrieval** — multi-issue scenarios are split into focused single-concept sub-queries, retrieved separately, then merged (top-3 per sub-query). This is the core fix that makes retrieval accurate on real, messy inputs.
- **Decoupled frontend/backend over HTTP** — the UI consumes the API like any other client; auth and error handling are exercised, not bypassed.
- **Managed services** (model API + hosted vector store) — a deliberate build-vs-buy choice so effort goes into retrieval quality and evaluation rather than infrastructure.
- **Structure-aware chunking** — the PCI DSS PDF is split on requirement-number boundaries so chunks map to requirements; citations trace to specific sections.

## Stack
FastAPI · Streamlit · ChromaDB (cosine) · OpenAI embeddings (`text-embedding-3-small`) · Anthropic Claude (decomposition + assessment) · agentic-coded with Kiro.

## Getting started
```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure environment (copy and fill in)
cp .env.example .env
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   AUTH_TOKEN=...            # bearer token for the API
#   BACKEND_URL=http://127.0.0.1:8000   # used by the Streamlit frontend

# 3. Ingest the corpus (PCI DSS v4.0.1 PDF in data/raw/)
python3 -m src.pipeline     # builds the Chroma index

# 4. Run the backend
python3 -m uvicorn src.main:app --reload

# 5. Run the frontend (separate terminal)
streamlit run src/app.py
```
> Secrets live in `.env` (gitignored). A model ID is configurable — Anthropic retires older models on a schedule, so pin a current one (see `GET /v1/models`).

## API
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/compliance/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{"query": "Is storing credit card CVV codes after authorization permitted?"}'
```
Returns: `assessment`, `risk_classification`, `citations`, `grounding_confidence`, `retrieved_chunk_ids`.

## Evaluation
A lightweight harness (`evals/`) scores the pipeline against 10 hand-labelled scenarios (8 non-compliant, 2 compliant), measuring **retrieval recall** (did it surface the expected requirements?) and **classification accuracy**. Recall is scored **hierarchy-aware** — PCI requirements nest (3.5 → 3.5.1), so retrieving a child of an expected requirement counts as a hit; both strict and hierarchy-aware recall are reported.

**Latest results:** classification **10/10** (including both compliant cases — no false positives); retrieval recall **~95%** (hierarchy-aware). Known gaps: can miss secondary/implementation sub-requirements (e.g. retrieved the two MFA-required controls but missed the MFA-configuration one); latency is high and variable (see below).

```bash
python3 -m evals.run_eval    # runs all scenarios, prints a results table, saves a timestamped CSV
```

## Known limitations
- **Latency & token usage** — decomposition means ~7 model/embedding calls per query plus a large assessment context (~30s typical, occasional outliers). Optimisation (parallel retrieval, context trimming, caching) is deferred future work.
- **Secondary requirements** — surfaces headline controls reliably; can miss supporting implementation sub-requirements.
- **Date-conditional requirements** — PCI DSS v4 has "best practice until [date]" requirements; the date logic isn't robustly handled.
- **Not production-validated** — built to production-*style* standards on a 10-scenario eval; not a substitute for a qualified assessor.

## Notes
This project's most interesting work was debugging retrieval — diagnosing a query-side topic-dominance failure via score-distribution analysis and fixing it with query decomposition. See the accompanying write-up for the full journey.
