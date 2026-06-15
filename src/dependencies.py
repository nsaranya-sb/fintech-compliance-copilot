"""Dependency injection wiring for the PCI DSS RAG Compliance system.

Instantiates all services and wires them together during application startup.
Injects the RAG engine and pipeline orchestrator into the API routes module.
"""

import logging
import os

from src.api.routes import set_pipeline_orchestrator, set_rag_engine
from src.config import Settings, get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.parsers.pdf_parser import PDFParser
from src.pipeline.orchestrator import PipelineOrchestrator
from src.rag.engine import RAGEngine
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def wire_dependencies(settings: Settings | None = None) -> None:
    """Instantiate all services and wire them into the API layer.

    Creates the full dependency graph:
        PDFParser
        EmbeddingService
        ChromaVectorStore
        RAGEngine(vector_store, embedding_service)
        PipelineOrchestrator(parser, embedding_service, vector_store)

    Then injects the RAG engine and pipeline orchestrator into the routes
    module via set_rag_engine() and set_pipeline_orchestrator().

    Args:
        settings: Optional Settings instance. If None, loads from environment.
    """
    if settings is None:
        settings = get_settings()

    # Export API keys to environment so service constructors can pick them up
    # Use direct assignment to override any stale shell-level env vars
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.ANTHROPIC_API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
    if settings.API_AUTH_TOKEN:
        os.environ["API_AUTH_TOKEN"] = settings.API_AUTH_TOKEN

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

    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_service=embedding_service,
        api_key=settings.ANTHROPIC_API_KEY,
        use_query_decomposition=settings.USE_QUERY_DECOMPOSITION,
    )

    pipeline_orchestrator = PipelineOrchestrator(
        parser=parser,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Wire into routes
    set_rag_engine(rag_engine)
    set_pipeline_orchestrator(pipeline_orchestrator)

    logger.info("All dependencies wired successfully")
