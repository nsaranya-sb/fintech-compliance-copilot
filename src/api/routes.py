"""API routes for the PCI DSS RAG Compliance system.

Implements the POST /api/v1/compliance/query endpoint with:
- API key / bearer token authentication
- Request validation (1-5000 characters)
- 30-second processing timeout
- 503 with Retry-After when RAG engine unavailable
- Structured ComplianceResponseSchema JSON responses
"""

import asyncio
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.models import ComplianceQueryRequest, ComplianceResponseSchema
from src.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Security schemes
_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    api_key: str | None = Depends(_api_key_header),
) -> str:
    """Validate API key or bearer token from request headers.

    Accepts either:
    - Authorization: Bearer <token>
    - X-API-Key: <key>

    Compares against the API_AUTH_TOKEN environment variable.

    Returns:
        The validated token string.

    Raises:
        HTTPException: 401 if token is missing or invalid.
    """
    expected_token = os.environ.get("API_AUTH_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=401,
            detail={"error": "Authentication required"},
        )

    # Check Bearer token first, then X-API-Key header
    provided_token: str | None = None
    if bearer and bearer.credentials:
        provided_token = bearer.credentials
    elif api_key:
        provided_token = api_key

    if not provided_token or provided_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail={"error": "Authentication required"},
        )

    return provided_token


# --- RAG Engine dependency ---

_rag_engine: RAGEngine | None = None


def set_rag_engine(engine: RAGEngine | None) -> None:
    """Set the RAG engine instance for dependency injection.

    This allows the engine to be wired in during application startup
    (Task 11.1) or overridden in tests.

    Args:
        engine: The RAGEngine instance, or None to mark as unavailable.
    """
    global _rag_engine
    _rag_engine = engine


def get_rag_engine() -> RAGEngine:
    """FastAPI dependency that provides the RAG engine.

    Returns:
        The configured RAGEngine instance.

    Raises:
        HTTPException: 503 if the RAG engine is not available.
    """
    if _rag_engine is None:
        raise HTTPException(status_code=503)

    return _rag_engine


# --- Query endpoint ---

# Timeout for RAG processing in seconds
RAG_PROCESSING_TIMEOUT = 30


@router.post("/api/v1/compliance/query", response_model=ComplianceResponseSchema)
async def query_compliance(
    request: ComplianceQueryRequest,
    api_key: Annotated[str, Depends(verify_api_key)] = ...,
    rag_engine: Annotated[RAGEngine, Depends(get_rag_engine)] = ...,
) -> ComplianceResponseSchema:
    """Process a compliance query and return a cited assessment.

    Authenticates the request, validates the query, processes it through
    the RAG engine with a 30-second timeout, and returns a structured
    compliance response with citations and risk classification.

    Args:
        request: The compliance query request with query text and optional context.
        api_key: The validated authentication token (injected via dependency).
        rag_engine: The RAG engine instance (injected via dependency).

    Returns:
        ComplianceResponseSchema with assessment, risk classification,
        citations, grounding confidence, and retrieved chunk IDs.
    """
    try:
        # Run RAG processing with a 30-second timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(rag_engine.process_query, request),
            timeout=RAG_PROCESSING_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={"error": "Query processing timed out"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RAG engine processing failed: %s", str(e))
        raise HTTPException(status_code=503)

    return response
