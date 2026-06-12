# Implementation Plan: PCI DSS RAG Compliance

## Overview

This plan implements a Python-based RAG application that parses PCI DSS v4.0/v4.0.1 documents, builds semantic vector embeddings using ChromaDB, and exposes a FastAPI endpoint for compliance queries with strict citations, anti-hallucination grounding, and programmatic risk classification. Implementation proceeds bottom-up: data models → parser → embeddings → vector store → RAG engine → API layer → pipeline orchestration → wiring.

## Tasks

- [x] 1. Set up project structure, dependencies, and core data models
  - [x] 1.1 Create project directory structure and configuration files
    - Create `src/` package with subpackages: `parsers/`, `embeddings/`, `vectorstore/`, `rag/`, `api/`, `pipeline/`
    - Create `tests/` directory with subdirectories: `unit/`, `property/`, `integration/`
    - Create `pyproject.toml` with dependencies: fastapi, uvicorn, pydantic, pymupdf, chromadb, openai, anthropic, python-dotenv
    - Create dev dependencies: pytest, hypothesis, pytest-asyncio, httpx
    - Create `.env.example` with required environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `API_AUTH_TOKEN`
    - _Requirements: 1.1, 2.1, 3.1, 4.1_

  - [x] 1.2 Implement core Pydantic data models
    - Create `src/models.py` with all domain models: `Chunk`, `PageContent`, `ParsedDocument`, `EmbeddingResult`, `RetrievedChunk`
    - Implement enums: `RiskClassification`, `GroundingConfidence`, `PipelineStatus`
    - Implement API models: `ComplianceQueryRequest`, `ComplianceResponseSchema`
    - Implement pipeline models: `IngestionCheckpoint`, `IngestionReport`
    - Add Pydantic field validators for query length (1-5000 chars) and enum constraints
    - _Requirements: 4.2, 6.1, 7.4, 8.1_

- [ ] 2. Implement Document Parser
  - [x] 2.1 Implement PDF text extraction with PyMuPDF
    - Create `src/parsers/pdf_parser.py` with `PDFParser` class
    - Implement `extract_pages()` method using `fitz.open()` and `page.get_text("dict")` for structured text extraction
    - Detect section headings and PCI DSS requirement numbers via regex patterns (e.g., `Requirement \d+\.\d+`)
    - Implement error handling for corrupted or unsupported files (log and skip without terminating)
    - _Requirements: 1.1, 1.4_

  - [x] 2.2 Implement chunk segmentation logic
    - Implement `segment_into_chunks()` method with configurable `max_chunk_tokens` parameter (default 512)
    - Preserve semantic boundaries at section/requirement level
    - Extend chunk boundaries to include complete sentences when a boundary falls mid-sentence
    - Annotate each chunk with metadata: source_file, requirement_number, section_heading, page_number, chunk_index
    - Preserve verbatim text without summarization, including special characters, tables, and list structures
    - _Requirements: 1.2, 1.3, 8.1, 8.2, 8.3, 8.4_

  - [ ] 2.3 Write property tests for document chunking (Property 1)
    - **Property 1: Document Chunking Round-Trip**
    - Generate random multi-section document text, parse into chunks, concatenate in order, and verify equivalence to original
    - **Validates: Requirements 8.1, 8.2, 8.3**

  - [ ] 2.4 Write property tests for chunk size and sentence boundaries (Property 2)
    - **Property 2: Chunk Size and Sentence Boundary Invariants**
    - Generate random multi-sentence text with various max_chunk_tokens values, verify all chunks respect token limit and end at sentence boundaries
    - **Validates: Requirements 1.2, 8.4**

  - [ ] 2.5 Write property tests for chunk metadata completeness (Property 3)
    - **Property 3: Chunk Metadata Completeness**
    - Generate random parsed documents and verify every chunk has non-null source_file, page_number, and chunk_index
    - **Validates: Requirements 1.3**

  - [ ] 2.6 Write unit tests for PDF parser
    - Test extraction with a small PCI DSS PDF fixture
    - Test error handling for corrupted files
    - Test empty PDF produces zero chunks
    - Test requirement number regex detection
    - _Requirements: 1.1, 1.4_

- [x] 3. Checkpoint - Ensure parser tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Embedding Service
  - [x] 4.1 Implement embedding generation with OpenAI API
    - Create `src/embeddings/embedding_service.py` with `EmbeddingService` class
    - Implement `embed_chunks()` for batch processing with configurable batch_size (default 100)
    - Implement `embed_query()` for single query embedding
    - Implement `_retry_with_backoff()` with exponential backoff (3 retries) for API failures
    - Use `text-embedding-3-small` model (1536 dimensions)
    - _Requirements: 2.1, 2.3, 2.4_

  - [ ]* 4.2 Write unit tests for embedding service
    - Test batch embedding with mocked OpenAI client
    - Test retry logic with simulated rate limits (429) and timeouts
    - Test single query embedding
    - _Requirements: 2.1, 2.3, 2.4_

- [x] 5. Implement Vector Store
  - [x] 5.1 Implement ChromaDB vector store with retrieval logic
    - Create `src/vectorstore/chroma_store.py` with `ChromaVectorStore` class
    - Initialize ChromaDB `PersistentClient` with cosine similarity metric at `./data/vectordb`
    - Implement `add_embeddings()` to store embeddings with full chunk metadata
    - Implement `query()` with top-k retrieval (configurable 1-20, default 5), minimum similarity threshold (default 0.7), and optional requirement group filtering
    - Implement `mark_superseded()` to handle document re-ingestion
    - Return empty result set when no chunks exceed similarity threshold
    - _Requirements: 2.2, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 5.2 Write property tests for embedding store-retrieve round-trip (Property 4)
    - **Property 4: Embedding Store-Retrieve Round-Trip**
    - Use in-memory ChromaDB, store random chunks with embeddings, retrieve by ID, verify text and metadata preserved
    - **Validates: Requirements 2.2, 3.3**

  - [ ]* 5.3 Write property tests for retrieval ordering (Property 5)
    - **Property 5: Retrieval Ordering by Similarity**
    - Generate random query embeddings, retrieve results, verify strictly non-increasing cosine similarity scores
    - **Validates: Requirements 3.1**

  - [ ]* 5.4 Write property tests for top-k result size (Property 6)
    - **Property 6: Top-K Result Set Size Constraint**
    - Generate collections of various sizes, query with random top_k in [1,20], verify result count ≤ min(top_k, N_matching)
    - **Validates: Requirements 3.2**

  - [ ]* 5.5 Write property tests for requirement group filtering (Property 7)
    - **Property 7: Requirement Group Filter Correctness**
    - Store chunks with various requirement numbers, apply group filter, verify all returned chunks fall within specified range
    - **Validates: Requirements 3.5**

  - [ ]* 5.6 Write unit tests for vector store
    - Test empty collection returns empty results
    - Test mark_superseded removes old embeddings
    - Test metadata filtering works correctly
    - _Requirements: 3.1, 3.4, 3.5_

- [x] 6. Checkpoint - Ensure embedding and vector store tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement RAG Engine
  - [x] 7.1 Implement RAG engine with Claude 3.5 Sonnet integration
    - Create `src/rag/engine.py` with `RAGEngine` class
    - Implement `process_query()` orchestrating: embed query → retrieve chunks → generate response → classify risk
    - Implement `_build_prompt()` with anti-hallucination constraints (restrict generation to retrieved content only)
    - Implement `_generate_response()` using Anthropic API with tool use for structured JSON output
    - Handle empty retrieval: return "Clause not found in source documentation" with 🟡 Warning
    - Handle low grounding confidence: prepend disclaimer, override to 🟡 Warning minimum
    - Handle conflicting chunks: flag conflict in assessment, assign 🟡 Warning
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.5_

  - [ ] 7.2 Write property tests for response schema conformance (Property 8)
    - **Property 8: Response Schema Conformance**
    - Mock Claude responses, generate random valid queries with non-empty retrieval, verify response contains all required fields with valid enum values
    - **Validates: Requirements 4.2, 5.5, 6.1**

  - [ ] 7.3 Write property tests for citation presence (Property 10)
    - **Property 10: Citation Presence in Assessments**
    - Mock Claude to produce valid responses, verify citations array is non-empty for all non-empty retrievals
    - **Validates: Requirements 5.2**

  - [ ] 7.4 Write property tests for citation format (Property 11)
    - **Property 11: Citation Format Consistency**
    - Generate random citation strings from mock responses, verify each matches pattern "Requirement X.Y[.Z] under PCI DSS v4.0.1, Section [section_name]"
    - **Validates: Requirements 5.3**

  - [ ]* 7.5 Write property tests for low confidence safety (Property 12)
    - **Property 12: Low Confidence Safety Override**
    - Generate responses with grounding_confidence="Low", verify risk_classification is never 🟢 Compliant and assessment starts with disclaimer
    - **Validates: Requirements 5.6, 6.5**

  - [ ]* 7.6 Write property tests for retrieved chunk ID auditability (Property 13)
    - **Property 13: Retrieved Chunk ID Auditability**
    - Generate successful responses, verify retrieved_chunk_ids is non-empty and all IDs exist in vector store
    - **Validates: Requirements 7.4**

  - [ ]* 7.7 Write unit tests for RAG engine
    - Test empty retrieval fallback behavior
    - Test conflicting chunks detection and flagging
    - Test Claude API failure retry and 503 propagation
    - Test low confidence disclaimer prepending
    - _Requirements: 5.4, 6.5, 7.3, 7.5_

- [x] 8. Checkpoint - Ensure RAG engine tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Compliance API
  - [x] 9.1 Implement FastAPI application with authentication and query endpoint
    - Create `src/api/app.py` with FastAPI application setup and CORS configuration
    - Create `src/api/routes.py` with POST `/api/v1/compliance/query` endpoint
    - Implement API key / bearer token authentication via FastAPI dependency injection (`verify_api_key`)
    - Add request validation (non-empty query, max 5000 chars) returning 400 on violation
    - Add 30-second timeout on RAG processing
    - Return 503 with `Retry-After` header when RAG engine unavailable
    - Return structured `ComplianceResponseSchema` JSON on success
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 9.2 Write property tests for input validation boundary (Property 9)
    - **Property 9: Input Validation Boundary**
    - Generate empty/whitespace-only strings and strings exceeding 5000 chars, verify 400 status code with descriptive error
    - **Validates: Requirements 4.4**

  - [ ]* 9.3 Write unit tests for API layer
    - Test authentication rejection (missing/invalid token returns 401)
    - Test valid query returns correct response schema
    - Test 503 response when RAG engine unavailable
    - Test timeout behavior (504)
    - Use httpx AsyncClient for async endpoint testing
    - _Requirements: 4.3, 4.4, 4.5_

- [ ] 10. Implement Pipeline Orchestrator
  - [ ] 10.1 Implement ingestion pipeline with observability and resumability
    - Create `src/pipeline/orchestrator.py` with `PipelineOrchestrator` class
    - Implement `run_ingestion()` with progress logging (total documents, per-document completion)
    - Implement `resume_ingestion()` using `IngestionCheckpoint` to skip already-processed documents
    - Implement `get_status()` returning pipeline status (idle/running/error) and last ingestion timestamp
    - Handle individual document failures: continue processing, collect in failure summary
    - Log total duration, chunk count, and embedding count upon completion
    - Store checkpoint state in `data/pipeline_state.json`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ] 10.2 Implement health check endpoint
    - Add GET `/api/v1/health` endpoint returning pipeline status and last successful ingestion timestamp
    - _Requirements: 9.3_

  - [ ]* 10.3 Write unit tests for pipeline orchestrator
    - Test resumption skips already-processed documents
    - Test individual failure doesn't terminate pipeline
    - Test progress logging emits correct messages
    - Test completion report contains expected metrics
    - _Requirements: 9.1, 9.2, 9.4, 9.5_

- [ ] 11. Integration wiring and final assembly
  - [ ] 11.1 Wire all components together with dependency injection
    - Create `src/config.py` for application settings (env vars, defaults)
    - Create `src/dependencies.py` for FastAPI dependency injection (instantiate services, wire store → embedding → engine)
    - Create `src/main.py` as application entry point with uvicorn startup
    - Create CLI entry point for running ingestion pipeline (`python -m src.pipeline.run`)
    - Ensure all imports resolve and application starts successfully
    - _Requirements: 4.1, 9.1, 9.3_

  - [ ]* 11.2 Write integration tests for ingestion pipeline
    - Test full pipeline: parse small test PDF → chunk → embed (mocked) → store in in-memory ChromaDB
    - Test document re-ingestion marks previous embeddings as superseded
    - _Requirements: 1.1, 2.4, 2.5_

  - [ ]* 11.3 Write integration tests for query flow
    - Test end-to-end query: API request → auth → embed query → retrieve → generate (mocked Claude) → structured response
    - Test concurrent query handling
    - _Requirements: 4.1, 4.2, 5.1, 7.4_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each major layer
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples, edge cases, and error conditions using pytest
- External APIs (OpenAI, Anthropic) should be mocked in tests; use real ChromaDB in-memory for vector store tests
- The `data/raw/` directory contains the PCI DSS source PDFs for ingestion
- Excel parsing is explicitly deferred to Phase 2 per requirements scope note

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "4.1"] },
    { "id": 3, "tasks": ["2.2", "4.2"] },
    { "id": 4, "tasks": ["5.1", "2.3", "2.4", "2.5", "2.6"] },
    { "id": 5, "tasks": ["7.1", "5.2", "5.3", "5.4", "5.5", "5.6"] },
    { "id": 6, "tasks": ["9.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7"] },
    { "id": 7, "tasks": ["10.1", "9.2", "9.3"] },
    { "id": 8, "tasks": ["10.2", "10.3"] },
    { "id": 9, "tasks": ["11.1"] },
    { "id": 10, "tasks": ["11.2", "11.3"] }
  ]
}
```
