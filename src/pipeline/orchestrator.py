"""Pipeline orchestrator for PCI DSS document ingestion.

Coordinates parsing, embedding, and storage of regulatory documents with
progress tracking, checkpoint-based resumability, and per-document failure
isolation.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.embeddings.embedding_service import EmbeddingService
from src.models import IngestionCheckpoint, IngestionReport, PipelineStatus
from src.parsers.pdf_parser import PDFParser
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("data/pipeline_state.json")


class PipelineOrchestrator:
    """Coordinates document ingestion with observability and resumability.

    Manages the full ingestion pipeline (parse → chunk → embed → store) with:
    - Progress logging per document
    - Checkpoint-based resume after interruption
    - Per-document failure isolation (one failure doesn't stop the pipeline)
    - Completion metrics (duration, chunk count, embedding count)
    """

    def __init__(
        self,
        parser: PDFParser,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    ):
        """Initialize with all pipeline components.

        Args:
            parser: PDF document parser for text extraction and chunking.
            embedding_service: Service for generating vector embeddings.
            vector_store: ChromaDB store for persisting embeddings.
            checkpoint_path: Path for storing pipeline state JSON.
        """
        self._parser = parser
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._checkpoint_path = checkpoint_path
        self._status = PipelineStatus.IDLE
        self._last_ingestion_timestamp: str | None = None

    def run_ingestion(self, documents_dir: Path) -> IngestionReport:
        """Run full ingestion pipeline with progress tracking.

        Lists all PDF files in the directory, processes each through the
        parse → chunk → embed → store pipeline, and logs progress throughout.
        Individual document failures are caught and collected without
        terminating the pipeline.

        Args:
            documents_dir: Directory containing PDF files to ingest.

        Returns:
            IngestionReport summarizing the pipeline run.
        """
        start_time = time.time()
        self._status = PipelineStatus.RUNNING

        # Discover PDF files
        pdf_files = sorted(documents_dir.glob("*.pdf"))
        total_documents = len(pdf_files)
        logger.info(
            "Starting ingestion pipeline: %d documents found in %s",
            total_documents,
            documents_dir,
        )

        processed_files: list[str] = []
        failed_files: list[str] = []
        failures: list[dict] = []
        total_chunks = 0
        total_embeddings = 0

        for idx, pdf_path in enumerate(pdf_files, start=1):
            file_name = pdf_path.name
            logger.info(
                "Processing document %d/%d: %s", idx, total_documents, file_name
            )

            try:
                # Parse → Chunk
                parsed_doc = self._parser.parse_document(pdf_path)
                chunks = parsed_doc.chunks
                doc_chunk_count = len(chunks)

                # Embed
                embeddings = self._embedding_service.embed_chunks(chunks)
                doc_embedding_count = len(embeddings)

                # Store
                self._vector_store.add_embeddings(embeddings)

                total_chunks += doc_chunk_count
                total_embeddings += doc_embedding_count
                processed_files.append(file_name)

                logger.info(
                    "Completed document %d/%d: %s (%d chunks, %d embeddings)",
                    idx,
                    total_documents,
                    file_name,
                    doc_chunk_count,
                    doc_embedding_count,
                )

            except Exception as e:
                logger.error(
                    "Failed to process document %d/%d: %s - %s",
                    idx,
                    total_documents,
                    file_name,
                    str(e),
                )
                failed_files.append(file_name)
                failures.append({"file": file_name, "error": str(e)})

            # Save checkpoint after each document
            self._save_checkpoint(
                IngestionCheckpoint(
                    last_processed_file=file_name,
                    processed_files=processed_files.copy(),
                    failed_files=failed_files.copy(),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        duration = time.time() - start_time
        self._status = PipelineStatus.IDLE if not failed_files else PipelineStatus.IDLE
        self._last_ingestion_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Ingestion pipeline complete: duration=%.2fs, chunks=%d, embeddings=%d, "
            "successful=%d, failed=%d",
            duration,
            total_chunks,
            total_embeddings,
            len(processed_files),
            len(failed_files),
        )

        return IngestionReport(
            total_documents=total_documents,
            successful=len(processed_files),
            failed=len(failed_files),
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            duration_seconds=round(duration, 2),
            failures=failures,
        )

    def resume_ingestion(self) -> IngestionReport:
        """Resume ingestion from last checkpoint after interruption.

        Loads checkpoint state from disk, determines which documents were
        already processed, and continues with remaining documents.

        Returns:
            IngestionReport for the resumed run.

        Raises:
            FileNotFoundError: If no checkpoint file exists.
            ValueError: If checkpoint has no documents directory context.
        """
        checkpoint = self._load_checkpoint()

        if not checkpoint:
            raise FileNotFoundError(
                f"No checkpoint found at {self._checkpoint_path}. "
                "Run a full ingestion first."
            )

        logger.info(
            "Resuming ingestion from checkpoint: %d files already processed, "
            "%d files previously failed",
            len(checkpoint.processed_files),
            len(checkpoint.failed_files),
        )

        start_time = time.time()
        self._status = PipelineStatus.RUNNING

        # Determine the documents directory from the last processed file
        # We need to find the parent directory - use the checkpoint's processed files
        # to locate the source directory
        already_processed = set(checkpoint.processed_files)
        already_failed = set(checkpoint.failed_files)
        skip_files = already_processed | already_failed

        # Try to find documents directory from checkpoint context
        # Look for PDF files in common locations
        documents_dir = self._find_documents_dir(checkpoint)
        if documents_dir is None:
            self._status = PipelineStatus.ERROR
            raise ValueError(
                "Cannot determine documents directory from checkpoint. "
                "Please run a full ingestion with an explicit directory."
            )

        pdf_files = sorted(documents_dir.glob("*.pdf"))
        remaining_files = [f for f in pdf_files if f.name not in skip_files]
        total_remaining = len(remaining_files)

        logger.info(
            "Found %d remaining documents to process (skipping %d already processed)",
            total_remaining,
            len(skip_files),
        )

        processed_files = list(checkpoint.processed_files)
        failed_files = list(checkpoint.failed_files)
        failures: list[dict] = []
        total_chunks = 0
        total_embeddings = 0

        for idx, pdf_path in enumerate(remaining_files, start=1):
            file_name = pdf_path.name
            logger.info(
                "Processing document %d/%d (resumed): %s",
                idx,
                total_remaining,
                file_name,
            )

            try:
                parsed_doc = self._parser.parse_document(pdf_path)
                chunks = parsed_doc.chunks
                doc_chunk_count = len(chunks)

                embeddings = self._embedding_service.embed_chunks(chunks)
                doc_embedding_count = len(embeddings)

                self._vector_store.add_embeddings(embeddings)

                total_chunks += doc_chunk_count
                total_embeddings += doc_embedding_count
                processed_files.append(file_name)

                logger.info(
                    "Completed document %d/%d (resumed): %s (%d chunks, %d embeddings)",
                    idx,
                    total_remaining,
                    file_name,
                    doc_chunk_count,
                    doc_embedding_count,
                )

            except Exception as e:
                logger.error(
                    "Failed to process document %d/%d (resumed): %s - %s",
                    idx,
                    total_remaining,
                    file_name,
                    str(e),
                )
                failed_files.append(file_name)
                failures.append({"file": file_name, "error": str(e)})

            # Update checkpoint after each document
            self._save_checkpoint(
                IngestionCheckpoint(
                    last_processed_file=file_name,
                    processed_files=processed_files.copy(),
                    failed_files=failed_files.copy(),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        duration = time.time() - start_time
        self._status = PipelineStatus.IDLE
        self._last_ingestion_timestamp = datetime.now(timezone.utc).isoformat()

        total_documents = len(pdf_files)

        logger.info(
            "Resumed ingestion complete: duration=%.2fs, chunks=%d, embeddings=%d, "
            "successful=%d, failed=%d",
            duration,
            total_chunks,
            total_embeddings,
            len(remaining_files) - len(failures),
            len(failures),
        )

        return IngestionReport(
            total_documents=total_documents,
            successful=len(remaining_files) - len(failures),
            failed=len(failures),
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            duration_seconds=round(duration, 2),
            failures=failures,
        )

    def get_status(self) -> dict:
        """Return current pipeline status for health check.

        Returns:
            Dictionary containing:
                - status: Current PipelineStatus (idle/running/error)
                - last_ingestion_timestamp: ISO timestamp of last successful run, or None
        """
        return {
            "status": self._status.value,
            "last_ingestion_timestamp": self._last_ingestion_timestamp,
        }

    def _save_checkpoint(self, checkpoint: IngestionCheckpoint) -> None:
        """Save checkpoint state to disk as JSON.

        Args:
            checkpoint: The checkpoint state to persist.
        """
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(
            json.dumps(checkpoint.model_dump(), indent=2), encoding="utf-8"
        )
        logger.debug("Checkpoint saved to %s", self._checkpoint_path)

    def _load_checkpoint(self) -> IngestionCheckpoint | None:
        """Load checkpoint state from disk.

        Returns:
            The loaded IngestionCheckpoint, or None if no checkpoint file exists.
        """
        if not self._checkpoint_path.exists():
            return None

        data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        return IngestionCheckpoint(**data)

    def _find_documents_dir(self, checkpoint: IngestionCheckpoint) -> Path | None:
        """Attempt to locate the documents directory from checkpoint state.

        Uses the default data/raw/ directory as the standard location for
        PCI DSS source documents.

        Args:
            checkpoint: The loaded checkpoint with file history.

        Returns:
            Path to the documents directory, or None if not found.
        """
        # Default location for PCI DSS documents
        default_dir = Path("data/raw")
        if default_dir.exists():
            return default_dir

        return None
