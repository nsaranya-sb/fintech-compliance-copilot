"""FastAPI application setup with CORS configuration.

Creates and configures the main FastAPI application instance for the
PCI DSS RAG Compliance system.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application with CORS middleware and routes.
    """
    application = FastAPI(
        title="PCI DSS RAG Compliance API",
        description=(
            "Retrieval-Augmented Generation API for PCI DSS v4.0/v4.0.1 "
            "compliance queries with strict citations and risk classification."
        ),
        version="0.1.0",
    )

    # CORS configuration - allow all origins (development mode)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    application.include_router(router)

    # Custom exception handler for 503 to add Retry-After header
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with custom response formatting."""
        if exc.status_code == 503:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "retry_after": 30,
                },
                headers={"Retry-After": "30"},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)},
        )

    return application


app = create_app()
