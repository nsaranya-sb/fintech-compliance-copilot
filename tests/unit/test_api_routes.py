"""Unit tests for the API routes - Task 9.1 and 10.2 validation.

Tests authentication, request validation, 503 handling, timeout behavior,
and health check endpoint.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.routes import (
    get_rag_engine,
    set_pipeline_orchestrator,
    set_rag_engine,
)
from src.models import (
    ComplianceResponseSchema,
    GroundingConfidence,
    RiskClassification,
)
from src.pipeline.orchestrator import PipelineOrchestrator
from src.rag.engine import RAGEngine


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance for each test."""
    return create_app()


@pytest.fixture
def mock_rag_engine():
    """Create a mock RAG engine that returns a valid response."""
    engine = MagicMock(spec=RAGEngine)
    engine.process_query.return_value = ComplianceResponseSchema(
        assessment="PCI DSS Requirement 3.3 mandates masking of PAN.",
        risk_classification=RiskClassification.COMPLIANT,
        citations=["Requirement 3.3 under PCI DSS v4.0.1, Section Data Protection"],
        grounding_confidence=GroundingConfidence.HIGH,
        retrieved_chunk_ids=["chunk-001", "chunk-002"],
    )
    return engine


@pytest.fixture
def auth_headers():
    """Return valid authentication headers."""
    return {"Authorization": "Bearer test-token-123"}


@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment for all tests."""
    with patch.dict(os.environ, {"API_AUTH_TOKEN": "test-token-123"}):
        yield
    # Clean up RAG engine and orchestrator state
    set_rag_engine(None)
    set_pipeline_orchestrator(None)


@pytest.mark.asyncio
async def test_auth_missing_token(app):
    """Test that missing auth token returns 401."""
    set_rag_engine(MagicMock(spec=RAGEngine))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_invalid_token(app):
    """Test that an invalid auth token returns 401."""
    set_rag_engine(MagicMock(spec=RAGEngine))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_auth_valid_bearer_token(app, mock_rag_engine, auth_headers):
    """Test that a valid bearer token passes authentication."""
    set_rag_engine(mock_rag_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers=auth_headers,
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_valid_api_key_header(app, mock_rag_engine):
    """Test that a valid X-API-Key header passes authentication."""
    set_rag_engine(mock_rag_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers={"X-API-Key": "test-token-123"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_request_validation_empty_query(app, mock_rag_engine, auth_headers):
    """Test that an empty query returns 422 (Pydantic validation)."""
    set_rag_engine(mock_rag_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": ""},
            headers=auth_headers,
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_validation_query_too_long(app, mock_rag_engine, auth_headers):
    """Test that a query exceeding 5000 chars returns 422."""
    set_rag_engine(mock_rag_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "x" * 5001},
            headers=auth_headers,
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_successful_response_schema(app, mock_rag_engine, auth_headers):
    """Test that a valid request returns ComplianceResponseSchema JSON."""
    set_rag_engine(mock_rag_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers=auth_headers,
        )
    assert response.status_code == 200
    data = response.json()
    assert "assessment" in data
    assert "risk_classification" in data
    assert "citations" in data
    assert "grounding_confidence" in data
    assert "retrieved_chunk_ids" in data


@pytest.mark.asyncio
async def test_503_when_rag_engine_unavailable(app, auth_headers):
    """Test that 503 with Retry-After is returned when RAG engine is None."""
    set_rag_engine(None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers=auth_headers,
        )
    assert response.status_code == 503
    assert response.headers.get("retry-after") == "30"
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_503_when_rag_engine_raises_exception(app, auth_headers):
    """Test that 503 is returned when RAG engine raises an exception."""
    engine = MagicMock(spec=RAGEngine)
    engine.process_query.side_effect = RuntimeError("Connection refused")
    set_rag_engine(engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/compliance/query",
            json={"query": "Is PAN masking required?"},
            headers=auth_headers,
        )
    assert response.status_code == 503
    assert response.headers.get("retry-after") == "30"



# --- Health check endpoint tests (Task 10.2) ---


@pytest.mark.asyncio
async def test_health_check_no_orchestrator(app):
    """Test health check returns idle with null timestamp when no orchestrator is set."""
    set_pipeline_orchestrator(None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["last_ingestion_timestamp"] is None


@pytest.mark.asyncio
async def test_health_check_no_auth_required(app):
    """Test health check does NOT require authentication."""
    set_pipeline_orchestrator(None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth headers provided
        response = await client.get("/api/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check_with_idle_orchestrator(app):
    """Test health check returns idle status from orchestrator."""
    orchestrator = MagicMock(spec=PipelineOrchestrator)
    orchestrator.get_status.return_value = {
        "status": "idle",
        "last_ingestion_timestamp": "2024-01-15T10:30:00+00:00",
    }
    set_pipeline_orchestrator(orchestrator)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["last_ingestion_timestamp"] == "2024-01-15T10:30:00+00:00"


@pytest.mark.asyncio
async def test_health_check_with_running_orchestrator(app):
    """Test health check returns running status from orchestrator."""
    orchestrator = MagicMock(spec=PipelineOrchestrator)
    orchestrator.get_status.return_value = {
        "status": "running",
        "last_ingestion_timestamp": None,
    }
    set_pipeline_orchestrator(orchestrator)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["last_ingestion_timestamp"] is None


@pytest.mark.asyncio
async def test_health_check_with_error_orchestrator(app):
    """Test health check returns error status from orchestrator."""
    orchestrator = MagicMock(spec=PipelineOrchestrator)
    orchestrator.get_status.return_value = {
        "status": "error",
        "last_ingestion_timestamp": "2024-01-14T08:00:00+00:00",
    }
    set_pipeline_orchestrator(orchestrator)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["last_ingestion_timestamp"] == "2024-01-14T08:00:00+00:00"
