"""FastAPI application and route definitions."""

from src.api.app import app, create_app
from src.api.routes import get_rag_engine, set_rag_engine, verify_api_key

__all__ = ["app", "create_app", "get_rag_engine", "set_rag_engine", "verify_api_key"]
