"""Core Pydantic data models for the PCI DSS RAG Compliance system.

Defines domain models, enums, API request/response schemas, and pipeline state models
used throughout the ingestion and query pipelines.
"""

from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---


class RiskClassification(str, Enum):
    """Tri-state risk classification for compliance assessments."""

    COMPLIANT = "🟢 Compliant"
    WARNING = "🟡 Warning"
    NON_COMPLIANT = "🔴 Non-Compliant"


class GroundingConfidence(str, Enum):
    """Confidence level indicating how well the response is grounded in source material."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class PipelineStatus(str, Enum):
    """Current status of the ingestion pipeline."""

    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"


# --- Core Domain Models ---


class Chunk(BaseModel):
    """A semantically coherent segment of a PCI DSS document."""

    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Verbatim text content")
    source_file: str = Field(description="Source PDF filename")
    requirement_number: str | None = Field(
        default=None, description="PCI DSS requirement number e.g. '3.3'"
    )
    section_heading: str | None = Field(
        default=None, description="Section heading text"
    )
    page_number: int = Field(description="Source page number")
    chunk_index: int = Field(description="Position within the document for ordering")


class PageContent(BaseModel):
    """Raw extracted content from a single PDF page."""

    page_number: int
    text: str
    headings: list[str]


class ParsedDocument(BaseModel):
    """Result of parsing a full PDF document."""

    source_file: str
    total_pages: int
    chunks: list[Chunk]


class EmbeddingResult(BaseModel):
    """A chunk paired with its vector embedding."""

    chunk: Chunk
    embedding: list[float]


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with similarity score."""

    chunk: Chunk
    similarity_score: float


# --- API Request/Response Models ---


class ComplianceQueryRequest(BaseModel):
    """Incoming compliance query from the API."""

    query: str = Field(
        min_length=1,
        max_length=5000,
        description="Compliance question text",
    )
    context: str | None = Field(
        default=None,
        description="Optional system architecture or payment flow context",
    )


class ComplianceResponseSchema(BaseModel):
    """Structured compliance assessment response."""

    assessment: str = Field(
        description="Citation-backed compliance assessment text"
    )
    risk_classification: RiskClassification = Field(
        description="Tri-state risk classification"
    )
    citations: list[str] = Field(
        description="Exact PCI DSS clause references"
    )
    grounding_confidence: GroundingConfidence = Field(
        description="Source backing confidence level"
    )
    retrieved_chunk_ids: list[str] = Field(
        description="IDs of retrieved chunks for auditability"
    )


# --- Pipeline State Models ---


class IngestionCheckpoint(BaseModel):
    """Tracks ingestion progress for resumability."""

    last_processed_file: str | None = None
    processed_files: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)
    timestamp: str


class IngestionReport(BaseModel):
    """Summary of an ingestion pipeline run."""

    total_documents: int
    successful: int
    failed: int
    total_chunks: int
    total_embeddings: int
    duration_seconds: float
    failures: list[dict] = Field(default_factory=list)
