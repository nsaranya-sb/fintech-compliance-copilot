"""Checkpoint tests for Embedding Service and Vector Store (Task 6).

Verifies:
1. Imports work correctly
2. ChromaVectorStore can be instantiated with a temp directory
3. Smoke test: store fake chunks with embeddings, query, verify ordering
4. mark_superseded() removes documents
5. Requirement group filtering works
"""

import tempfile

import pytest

from src.embeddings import EmbeddingService
from src.vectorstore import ChromaVectorStore
from src.models import Chunk, EmbeddingResult, RetrievedChunk


class TestImports:
    """Verify all imports work correctly."""

    def test_embedding_service_import(self):
        """EmbeddingService is importable from src.embeddings."""
        assert EmbeddingService is not None

    def test_chroma_vector_store_import(self):
        """ChromaVectorStore is importable from src.vectorstore."""
        assert ChromaVectorStore is not None

    def test_models_import(self):
        """Core models are importable from src.models."""
        assert Chunk is not None
        assert EmbeddingResult is not None
        assert RetrievedChunk is not None


class TestChromaVectorStoreInstantiation:
    """Verify ChromaVectorStore can be instantiated with temp directory."""

    def test_instantiation_with_temp_dir(self, tmp_path):
        """ChromaVectorStore can be created using a temporary directory."""
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="test_collection",
        )
        assert store is not None

    def test_instantiation_creates_collection(self, tmp_path):
        """ChromaVectorStore creates the ChromaDB collection on init."""
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="test_pci_dss",
        )
        # The internal collection should exist
        assert store._collection is not None


class TestSmokeStoreAndQuery:
    """Smoke test: store fake chunks, query, verify results ordered by similarity."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh ChromaVectorStore in a temp directory."""
        return ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="smoke_test",
        )

    @pytest.fixture
    def fake_chunks(self):
        """Create a few fake Chunk objects with distinct content."""
        return [
            Chunk(
                id="chunk_001",
                text="PCI DSS Requirement 3.3 states that PANs must be masked when displayed.",
                source_file="pci-dss-v4.pdf",
                requirement_number="3.3",
                section_heading="Protect Stored Account Data",
                page_number=45,
                chunk_index=0,
            ),
            Chunk(
                id="chunk_002",
                text="Requirement 8.2 mandates multi-factor authentication for admin access.",
                source_file="pci-dss-v4.pdf",
                requirement_number="8.2",
                section_heading="Identify Users and Authenticate Access",
                page_number=120,
                chunk_index=1,
            ),
            Chunk(
                id="chunk_003",
                text="Encryption of cardholder data during transmission over open networks is required by Requirement 4.1.",
                source_file="pci-dss-v4.pdf",
                requirement_number="4.1",
                section_heading="Protect Cardholder Data",
                page_number=60,
                chunk_index=2,
            ),
        ]

    @pytest.fixture
    def fake_embeddings(self, fake_chunks):
        """Create fake embedding vectors (1536-dim) for each chunk.

        We use simple patterns to control similarity:
        - chunk_001 embedding: mostly 1.0s
        - chunk_002 embedding: mix of 1.0s and 0.0s
        - chunk_003 embedding: mostly 0.5s
        """
        dim = 1536
        return [
            EmbeddingResult(
                chunk=fake_chunks[0],
                embedding=[1.0] * dim,
            ),
            EmbeddingResult(
                chunk=fake_chunks[1],
                embedding=[0.0] * (dim // 2) + [1.0] * (dim // 2),
            ),
            EmbeddingResult(
                chunk=fake_chunks[2],
                embedding=[0.5] * dim,
            ),
        ]

    def test_add_and_query_returns_results(self, store, fake_embeddings):
        """Storing embeddings and querying returns non-empty results."""
        store.add_embeddings(fake_embeddings)

        # Query with vector close to chunk_001 (all 1.0s)
        query_embedding = [1.0] * 1536
        results = store.query(query_embedding=query_embedding, top_k=5, min_similarity=0.0)

        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_query_results_ordered_by_similarity(self, store, fake_embeddings):
        """Results are returned in descending similarity order."""
        store.add_embeddings(fake_embeddings)

        query_embedding = [1.0] * 1536
        results = store.query(query_embedding=query_embedding, top_k=5, min_similarity=0.0)

        # Verify descending similarity scores
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_most_similar_chunk_is_first(self, store, fake_embeddings):
        """The chunk most similar to the query should rank first."""
        store.add_embeddings(fake_embeddings)

        # Query vector of all 1.0s is most similar to chunk_001 (also all 1.0s)
        query_embedding = [1.0] * 1536
        results = store.query(query_embedding=query_embedding, top_k=5, min_similarity=0.0)

        assert results[0].chunk.id == "chunk_001"

    def test_min_similarity_threshold_filters_results(self, store, fake_embeddings):
        """Results below min_similarity are excluded."""
        store.add_embeddings(fake_embeddings)

        # Use a very high threshold to filter out most results
        query_embedding = [1.0] * 1536
        results_strict = store.query(
            query_embedding=query_embedding, top_k=5, min_similarity=0.99
        )
        results_lax = store.query(
            query_embedding=query_embedding, top_k=5, min_similarity=0.0
        )

        # Strict should return fewer (or equal) results than lax
        assert len(results_strict) <= len(results_lax)

    def test_empty_collection_returns_empty(self, store):
        """Querying an empty collection returns empty list."""
        query_embedding = [1.0] * 1536
        results = store.query(query_embedding=query_embedding, top_k=5, min_similarity=0.0)
        assert results == []


class TestMarkSuperseded:
    """Test that mark_superseded() removes documents."""

    @pytest.fixture
    def store(self, tmp_path):
        return ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="supersede_test",
        )

    def test_mark_superseded_removes_documents(self, store):
        """mark_superseded removes all chunks from the specified source file."""
        dim = 1536
        chunks = [
            Chunk(
                id="chunk_a1",
                text="First chunk from file A.",
                source_file="file_a.pdf",
                requirement_number="1.1",
                section_heading="Section A",
                page_number=1,
                chunk_index=0,
            ),
            Chunk(
                id="chunk_a2",
                text="Second chunk from file A.",
                source_file="file_a.pdf",
                requirement_number="1.2",
                section_heading="Section A",
                page_number=2,
                chunk_index=1,
            ),
            Chunk(
                id="chunk_b1",
                text="First chunk from file B.",
                source_file="file_b.pdf",
                requirement_number="2.1",
                section_heading="Section B",
                page_number=1,
                chunk_index=0,
            ),
        ]
        embeddings = [
            EmbeddingResult(chunk=c, embedding=[0.5] * dim) for c in chunks
        ]

        store.add_embeddings(embeddings)

        # Verify all 3 chunks stored
        query = [0.5] * dim
        results = store.query(query_embedding=query, top_k=10, min_similarity=0.0)
        assert len(results) == 3

        # Mark file_a.pdf as superseded
        store.mark_superseded("file_a.pdf")

        # Only file_b.pdf chunks should remain
        results = store.query(query_embedding=query, top_k=10, min_similarity=0.0)
        assert len(results) == 1
        assert results[0].chunk.source_file == "file_b.pdf"

    def test_mark_superseded_nonexistent_file_is_noop(self, store):
        """mark_superseded with a file that doesn't exist does nothing."""
        # Should not raise
        store.mark_superseded("nonexistent.pdf")


class TestRequirementGroupFiltering:
    """Test that requirement group filtering works correctly."""

    @pytest.fixture
    def store_with_data(self, tmp_path):
        """Store with chunks spanning requirement groups 1-12."""
        store = ChromaVectorStore(
            persist_directory=str(tmp_path / "vectordb"),
            collection_name="filter_test",
        )
        dim = 1536
        chunks = [
            Chunk(
                id=f"chunk_req_{i}",
                text=f"Content for requirement {i}.1 compliance.",
                source_file="pci-dss-v4.pdf",
                requirement_number=f"{i}.1",
                section_heading=f"Requirement {i}",
                page_number=i * 10,
                chunk_index=i,
            )
            for i in range(1, 13)  # Requirements 1.1 through 12.1
        ]
        embeddings = [
            EmbeddingResult(chunk=c, embedding=[0.5] * dim) for c in chunks
        ]
        store.add_embeddings(embeddings)
        return store

    def test_single_requirement_group_filter(self, store_with_data):
        """Filtering by single group returns only chunks in that group."""
        query = [0.5] * 1536
        results = store_with_data.query(
            query_embedding=query,
            top_k=20,
            min_similarity=0.0,
            requirement_filter="3",
        )

        assert len(results) == 1
        assert results[0].chunk.requirement_number == "3.1"

    def test_range_requirement_group_filter(self, store_with_data):
        """Filtering by range returns only chunks within that range."""
        query = [0.5] * 1536
        results = store_with_data.query(
            query_embedding=query,
            top_k=20,
            min_similarity=0.0,
            requirement_filter="1-3",
        )

        # Should get requirements 1.1, 2.1, 3.1
        assert len(results) == 3
        req_numbers = {r.chunk.requirement_number for r in results}
        assert req_numbers == {"1.1", "2.1", "3.1"}

    def test_no_filter_returns_all(self, store_with_data):
        """No requirement filter returns all matching chunks."""
        query = [0.5] * 1536
        results = store_with_data.query(
            query_embedding=query,
            top_k=20,
            min_similarity=0.0,
            requirement_filter=None,
        )

        assert len(results) == 12

    def test_filter_with_no_matches_returns_empty(self, store_with_data):
        """Filter for a group with no matching chunks returns empty."""
        query = [0.5] * 1536
        results = store_with_data.query(
            query_embedding=query,
            top_k=20,
            min_similarity=0.0,
            requirement_filter="99",
        )

        assert results == []
