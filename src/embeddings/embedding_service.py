"""Embedding generation service using OpenAI API.

Provides batch embedding for document chunks during ingestion and single-query
embedding for compliance queries during retrieval. Implements exponential backoff
retry logic for API resilience.
"""

import logging
import os
import time
from typing import Any, Callable

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from src.models import Chunk, EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating dense vector embeddings via OpenAI API.

    Uses text-embedding-3-small (1536 dimensions) for both document chunk
    ingestion and query embedding. Supports configurable batch sizes and
    automatic retry with exponential backoff on API failures.
    """

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        """Initialize with embedding model configuration.

        Args:
            model: OpenAI embedding model name. Defaults to text-embedding-3-small.
            batch_size: Number of chunks to embed per API call. Defaults to 100.
        """
        self.model = model
        self.batch_size = batch_size
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddingResult]:
        """Batch-embed document chunks with retry logic.

        Processes chunks in batches of configurable size, calling the OpenAI
        embeddings API for each batch. Failed batches are retried with
        exponential backoff.

        Args:
            chunks: List of document chunks to embed.

        Returns:
            List of EmbeddingResult objects pairing each chunk with its vector.
        """
        results: list[EmbeddingResult] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [chunk.text for chunk in batch]

            logger.info(
                "Embedding batch %d/%d (%d chunks)",
                (i // self.batch_size) + 1,
                (len(chunks) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

            embeddings = self._retry_with_backoff(
                lambda t=texts: self._call_embedding_api(t)
            )

            for chunk, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(chunk=chunk, embedding=embedding))

        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single compliance query for retrieval.

        Args:
            query: The compliance query text to embed.

        Returns:
            A list of floats representing the query embedding vector (1536 dimensions).
        """
        embeddings = self._retry_with_backoff(
            lambda: self._call_embedding_api([query])
        )
        return embeddings[0]

    def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]

    def _retry_with_backoff(
        self, func: Callable[[], Any], max_retries: int = 3
    ) -> Any:
        """Retry failed API calls with exponential backoff.

        Retries on rate limit errors, timeouts, and transient API errors.
        Uses exponential backoff with base delay of 1 second (1s, 2s, 4s).

        Args:
            func: Callable that performs the API operation.
            max_retries: Maximum number of retry attempts. Defaults to 3.

        Returns:
            The result of the successful function call.

        Raises:
            The last exception encountered after all retries are exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                return func()
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Embedding API call failed (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding API call failed after %d attempts: %s",
                        max_retries,
                        str(e),
                    )

        raise last_exception  # type: ignore[misc]
