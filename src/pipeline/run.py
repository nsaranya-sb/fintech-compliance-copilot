"""CLI entry point for running the PCI DSS document ingestion pipeline.

Usage:
    python -m src.pipeline.run [--documents-dir PATH]

Loads environment variables from .env, instantiates all pipeline services
from configuration, runs the ingestion pipeline, and prints the report.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Run the ingestion pipeline from the command line."""
    # Load .env file — override=True ensures .env takes precedence over shell env vars
    load_dotenv(override=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import after dotenv so env vars are available
    from src.config import get_settings
    from src.embeddings.embedding_service import EmbeddingService
    from src.parsers.pdf_parser import PDFParser
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.vectorstore.chroma_store import ChromaVectorStore

    settings = get_settings()

    # Export API keys to environment for service constructors
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.ANTHROPIC_API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser(
        description="Run the PCI DSS document ingestion pipeline."
    )
    arg_parser.add_argument(
        "--documents-dir",
        type=str,
        default=settings.DOCUMENTS_DIR,
        help=f"Path to directory containing PDF documents (default: {settings.DOCUMENTS_DIR})",
    )
    args = arg_parser.parse_args()

    documents_dir = Path(args.documents_dir)
    if not documents_dir.exists():
        print(f"Error: Documents directory not found: {documents_dir}", file=sys.stderr)
        sys.exit(1)

    # Instantiate services
    parser = PDFParser(max_chunk_tokens=settings.MAX_CHUNK_TOKENS)
    embedding_service = EmbeddingService(
        model=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        api_key=settings.OPENAI_API_KEY,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.VECTORDB_PATH,
        collection_name=settings.COLLECTION_NAME,
    )
    orchestrator = PipelineOrchestrator(
        parser=parser,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Run ingestion
    print(f"Starting ingestion from: {documents_dir}")
    report = orchestrator.run_ingestion(documents_dir)

    # Print report
    print("\n" + "=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    print(f"  Total documents:  {report.total_documents}")
    print(f"  Successful:       {report.successful}")
    print(f"  Failed:           {report.failed}")
    print(f"  Total chunks:     {report.total_chunks}")
    print(f"  Total embeddings: {report.total_embeddings}")
    print(f"  Duration:         {report.duration_seconds:.2f}s")

    if report.failures:
        print("\n  Failures:")
        for failure in report.failures:
            print(f"    - {failure['file']}: {failure['error']}")

    print("=" * 60)

    # Exit with error code if there were failures
    if report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
