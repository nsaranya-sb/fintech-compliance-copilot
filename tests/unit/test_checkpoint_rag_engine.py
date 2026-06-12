"""Checkpoint tests for RAG Engine (Task 8).

Verifies:
1. Imports work correctly: from src.rag import RAGEngine, from src.rag.engine import RAGEngine
2. RAGEngine can be instantiated with mock dependencies
3. Empty retrieval path returns "Clause not found in source documentation" with Warning
4. _build_prompt() includes anti-hallucination constraints
5. _detect_conflicts() logic works correctly
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    Chunk,
    ComplianceQueryRequest,
    ComplianceResponseSchema,
    GroundingConfidence,
    RetrievedChunk,
    RiskClassification,
)


class TestRAGEngineImports:
    """Verify all RAG engine imports work correctly."""

    def test_import_from_package(self):
        """RAGEngine is importable from src.rag."""
        from src.rag import RAGEngine

        assert RAGEngine is not None

    def test_import_from_module(self):
        """RAGEngine is importable from src.rag.engine."""
        from src.rag.engine import RAGEngine

        assert RAGEngine is not None

    def test_import_constants(self):
        """Engine constants are importable."""
        from src.rag.engine import EMPTY_RETRIEVAL_MESSAGE, LOW_CONFIDENCE_DISCLAIMER

        assert "Clause not found" in EMPTY_RETRIEVAL_MESSAGE
        assert "LOW SOURCE CONFIDENCE" in LOW_CONFIDENCE_DISCLAIMER


class TestRAGEngineInstantiation:
    """Verify RAGEngine can be instantiated with mock dependencies."""

    def test_instantiation_with_mocks(self):
        """RAGEngine can be created with mock vector store and embedding service."""
        from src.rag.engine import RAGEngine

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_service=mock_embedding_service,
            )

        assert engine is not None
        assert engine._vector_store is mock_vector_store
        assert engine._embedding_service is mock_embedding_service
        assert engine._model == "claude-sonnet-4-20250514"

    def test_instantiation_custom_model(self):
        """RAGEngine accepts a custom model name."""
        from src.rag.engine import RAGEngine

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_service=mock_embedding_service,
                model="claude-3-haiku-20240307",
            )

        assert engine._model == "claude-3-haiku-20240307"


class TestEmptyRetrievalPath:
    """Test the empty retrieval fallback behavior."""

    @pytest.fixture
    def engine(self):
        """Create a RAGEngine with mocked dependencies for empty retrieval."""
        from src.rag.engine import RAGEngine

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()

        # Mock embed_query to return a fake embedding vector
        mock_embedding_service.embed_query.return_value = [0.1] * 1536

        # Mock vector store to return empty results (no relevant chunks)
        mock_vector_store.query.return_value = []

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_service=mock_embedding_service,
            )

        return engine

    def test_empty_retrieval_returns_clause_not_found(self, engine):
        """When vector store returns empty results, assessment is 'Clause not found'."""
        request = ComplianceQueryRequest(query="What about XYZ compliance?")
        response = engine.process_query(request)

        assert response.assessment == "Clause not found in source documentation"

    def test_empty_retrieval_returns_warning_classification(self, engine):
        """Empty retrieval assigns Warning risk classification."""
        request = ComplianceQueryRequest(query="Unknown compliance topic")
        response = engine.process_query(request)

        assert response.risk_classification == RiskClassification.WARNING

    def test_empty_retrieval_returns_low_confidence(self, engine):
        """Empty retrieval assigns Low grounding confidence."""
        request = ComplianceQueryRequest(query="Something not in corpus")
        response = engine.process_query(request)

        assert response.grounding_confidence == GroundingConfidence.LOW

    def test_empty_retrieval_returns_empty_citations(self, engine):
        """Empty retrieval returns no citations."""
        request = ComplianceQueryRequest(query="Any query")
        response = engine.process_query(request)

        assert response.citations == []

    def test_empty_retrieval_returns_empty_chunk_ids(self, engine):
        """Empty retrieval returns no retrieved chunk IDs."""
        request = ComplianceQueryRequest(query="Any query")
        response = engine.process_query(request)

        assert response.retrieved_chunk_ids == []

    def test_response_is_valid_schema(self, engine):
        """Empty retrieval response conforms to ComplianceResponseSchema."""
        request = ComplianceQueryRequest(query="Any query")
        response = engine.process_query(request)

        assert isinstance(response, ComplianceResponseSchema)


class TestBuildPrompt:
    """Verify _build_prompt() produces prompts with anti-hallucination constraints."""

    @pytest.fixture
    def engine(self):
        """Create a RAGEngine instance for prompt building tests."""
        from src.rag.engine import RAGEngine

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_service=mock_embedding_service,
            )

        return engine

    @pytest.fixture
    def sample_chunks(self):
        """Create sample RetrievedChunk objects for prompt building."""
        return [
            RetrievedChunk(
                chunk=Chunk(
                    id="chunk_001",
                    text="PANs must be masked when displayed per Requirement 3.3.",
                    source_file="pci-dss-v4.pdf",
                    requirement_number="3.3",
                    section_heading="Protect Stored Account Data",
                    page_number=45,
                    chunk_index=0,
                ),
                similarity_score=0.92,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="chunk_002",
                    text="Multi-factor authentication is required for admin access.",
                    source_file="pci-dss-v4.pdf",
                    requirement_number="8.2",
                    section_heading="Identify Users",
                    page_number=120,
                    chunk_index=1,
                ),
                similarity_score=0.85,
            ),
        ]

    def test_prompt_contains_anti_hallucination_instruction(self, engine, sample_chunks):
        """Prompt includes explicit anti-hallucination constraints."""
        prompt = engine._build_prompt(
            query="Is PAN masking required?",
            context=None,
            chunks=sample_chunks,
        )

        assert "ANTI-HALLUCINATION" in prompt
        assert "EXCLUSIVELY" in prompt or "exclusively" in prompt
        assert "Do NOT introduce external knowledge" in prompt

    def test_prompt_contains_source_chunk_content(self, engine, sample_chunks):
        """Prompt includes the actual text from retrieved chunks."""
        prompt = engine._build_prompt(
            query="Is PAN masking required?",
            context=None,
            chunks=sample_chunks,
        )

        assert "PANs must be masked" in prompt
        assert "Multi-factor authentication" in prompt

    def test_prompt_contains_chunk_metadata(self, engine, sample_chunks):
        """Prompt includes chunk metadata (requirement numbers, sources)."""
        prompt = engine._build_prompt(
            query="Is PAN masking required?",
            context=None,
            chunks=sample_chunks,
        )

        assert "Requirement 3.3" in prompt
        assert "Requirement 8.2" in prompt
        assert "pci-dss-v4.pdf" in prompt

    def test_prompt_contains_user_query(self, engine, sample_chunks):
        """Prompt includes the user's original query."""
        prompt = engine._build_prompt(
            query="Is PAN masking required?",
            context=None,
            chunks=sample_chunks,
        )

        assert "Is PAN masking required?" in prompt

    def test_prompt_includes_context_when_provided(self, engine, sample_chunks):
        """Prompt includes optional system context when provided."""
        prompt = engine._build_prompt(
            query="Is our setup compliant?",
            context="We use AWS KMS for encryption at rest.",
            chunks=sample_chunks,
        )

        assert "AWS KMS for encryption at rest" in prompt

    def test_prompt_excludes_context_section_when_none(self, engine, sample_chunks):
        """Prompt does not include context section when context is None."""
        prompt = engine._build_prompt(
            query="Is PAN masking required?",
            context=None,
            chunks=sample_chunks,
        )

        assert "SYSTEM CONTEXT" not in prompt

    def test_prompt_contains_citation_format_instructions(self, engine, sample_chunks):
        """Prompt includes citation format requirements."""
        prompt = engine._build_prompt(
            query="Any query",
            context=None,
            chunks=sample_chunks,
        )

        assert "CITATION FORMAT" in prompt
        assert "Requirement X.Y" in prompt


class TestDetectConflicts:
    """Verify _detect_conflicts() logic works correctly."""

    @pytest.fixture
    def engine(self):
        """Create a RAGEngine instance for conflict detection tests."""
        from src.rag.engine import RAGEngine

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_service=mock_embedding_service,
            )

        return engine

    def test_no_conflict_with_single_chunk(self, engine):
        """Single chunk cannot conflict."""
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Some text",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Section A",
                    page_number=1,
                    chunk_index=0,
                ),
                similarity_score=0.9,
            ),
        ]

        assert engine._detect_conflicts(chunks) is False

    def test_no_conflict_same_section(self, engine):
        """Multiple chunks from same requirement and section do not conflict."""
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Text 1",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Same Section",
                    page_number=1,
                    chunk_index=0,
                ),
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="c2",
                    text="Text 2",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Same Section",
                    page_number=2,
                    chunk_index=1,
                ),
                similarity_score=0.85,
            ),
        ]

        assert engine._detect_conflicts(chunks) is False

    def test_conflict_detected_same_req_different_sections(self, engine):
        """Chunks referencing same requirement from different sections are conflicts."""
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Guidance text A",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Section A",
                    page_number=1,
                    chunk_index=0,
                ),
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="c2",
                    text="Guidance text B",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Section B",
                    page_number=5,
                    chunk_index=1,
                ),
                similarity_score=0.85,
            ),
        ]

        assert engine._detect_conflicts(chunks) is True

    def test_no_conflict_different_requirements(self, engine):
        """Chunks from different requirements do not conflict."""
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Req 3.3 content",
                    source_file="file.pdf",
                    requirement_number="3.3",
                    section_heading="Section A",
                    page_number=1,
                    chunk_index=0,
                ),
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="c2",
                    text="Req 8.2 content",
                    source_file="file.pdf",
                    requirement_number="8.2",
                    section_heading="Section B",
                    page_number=5,
                    chunk_index=1,
                ),
                similarity_score=0.85,
            ),
        ]

        assert engine._detect_conflicts(chunks) is False

    def test_no_conflict_empty_list(self, engine):
        """Empty chunk list does not conflict."""
        assert engine._detect_conflicts([]) is False

    def test_no_conflict_chunks_without_requirement_numbers(self, engine):
        """Chunks without requirement numbers cannot conflict."""
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="General text",
                    source_file="file.pdf",
                    requirement_number=None,
                    section_heading="Intro",
                    page_number=1,
                    chunk_index=0,
                ),
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="c2",
                    text="More general text",
                    source_file="file.pdf",
                    requirement_number=None,
                    section_heading="Overview",
                    page_number=2,
                    chunk_index=1,
                ),
                similarity_score=0.85,
            ),
        ]

        assert engine._detect_conflicts(chunks) is False
