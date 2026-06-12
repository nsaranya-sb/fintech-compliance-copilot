"""ChromaDB vector store for PCI DSS compliance chunk storage and retrieval.

Provides persistent vector storage with cosine similarity search,
metadata filtering by requirement group, and document supersession support.
"""

import logging

import chromadb

from src.models import Chunk, EmbeddingResult, RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Persistent vector store using ChromaDB with cosine similarity retrieval."""

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "pci_dss",
    ):
        """Initialize ChromaDB with persistent storage.

        Args:
            persist_directory: Path for ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection.
        """
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(self, embeddings: list[EmbeddingResult]) -> None:
        """Store embeddings with metadata in ChromaDB.

        Args:
            embeddings: List of EmbeddingResult objects containing chunks and their vectors.
        """
        if not embeddings:
            return

        ids: list[str] = []
        documents: list[str] = []
        vectors: list[list[float]] = []
        metadatas: list[dict] = []

        for result in embeddings:
            chunk = result.chunk
            ids.append(chunk.id)
            documents.append(chunk.text)
            vectors.append(result.embedding)
            metadatas.append({
                "source_file": chunk.source_file,
                "requirement_number": chunk.requirement_number or "",
                "section_heading": chunk.section_heading or "",
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "req_major": self._extract_major_requirement(chunk.requirement_number),
            })

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
        requirement_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks by cosine similarity with optional filtering.

        Args:
            query_embedding: Query vector for similarity search.
            top_k: Maximum number of results to return (1-20).
            min_similarity: Minimum similarity threshold (0.0-1.0). Results below
                this threshold are excluded.
            requirement_filter: Optional requirement group filter (e.g., "1-3" means
                requirements 1.x through 3.x).

        Returns:
            List of RetrievedChunk objects ordered by descending similarity score.
            Returns empty list when no chunks exceed the similarity threshold.
        """
        # Clamp top_k to valid range
        top_k = max(1, min(20, top_k))

        # Build where filter for requirement group
        where_filter = self._build_requirement_filter(requirement_filter)

        # Query ChromaDB
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        # Process results - ChromaDB returns distances; for cosine, similarity = 1 - distance
        retrieved_chunks: list[RetrievedChunk] = []

        if not results["ids"] or not results["ids"][0]:
            return retrieved_chunks

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, doc_id in enumerate(ids):
            similarity = 1.0 - distances[i]

            # Skip results below threshold
            if similarity < min_similarity:
                continue

            metadata = metadatas[i]
            chunk = Chunk(
                id=doc_id,
                text=documents[i],
                source_file=metadata["source_file"],
                requirement_number=metadata["requirement_number"] or None,
                section_heading=metadata["section_heading"] or None,
                page_number=metadata["page_number"],
                chunk_index=metadata["chunk_index"],
            )
            retrieved_chunks.append(
                RetrievedChunk(chunk=chunk, similarity_score=similarity)
            )

        return retrieved_chunks

    def mark_superseded(self, source_file: str) -> None:
        """Delete all documents from the collection matching the given source_file.

        Used during document re-ingestion to remove stale embeddings before
        storing updated versions.

        Args:
            source_file: The source file name whose embeddings should be removed.
        """
        # Get all IDs matching this source file
        results = self._collection.get(
            where={"source_file": {"$eq": source_file}},
            include=[],
        )

        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    @staticmethod
    def _extract_major_requirement(requirement_number: str | None) -> int:
        """Extract the major requirement number as an integer.

        For example, '3.3' -> 3, '12.1.2' -> 12, None -> 0.

        Args:
            requirement_number: The full requirement number string, or None.

        Returns:
            The major (first) component as an integer, or 0 if not parseable.
        """
        if not requirement_number:
            return 0
        try:
            major_str = requirement_number.split(".")[0]
            return int(major_str)
        except (ValueError, IndexError):
            return 0

    def _build_requirement_filter(self, requirement_filter: str | None) -> dict | None:
        """Build a ChromaDB where filter for requirement group ranges.

        Uses the numeric `req_major` metadata field for efficient integer-based
        range filtering. Supports range format "1-3" (requirements 1.x through
        3.x) and single group format "3" (requirements 3.x only).

        Args:
            requirement_filter: A range string like "1-3" meaning requirements
                1.x through 3.x, or a single number like "3", or None for no filter.

        Returns:
            A ChromaDB-compatible where clause dict, or None if no filter.
        """
        if not requirement_filter:
            return None

        # Parse range like "1-3" into start and end group numbers
        parts = requirement_filter.split("-")
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                return None

            # Use numeric range on req_major field
            return {
                "$and": [
                    {"req_major": {"$gte": start}},
                    {"req_major": {"$lte": end}},
                ]
            }

        # Single requirement group number
        try:
            group_num = int(requirement_filter.strip())
            return {"req_major": {"$eq": group_num}}
        except ValueError:
            # Treat as exact match on requirement_number if not a number
            return {"requirement_number": {"$eq": requirement_filter}}
