"""Application configuration loaded from environment variables.

Uses pydantic-settings for typed configuration with defaults and
environment variable overrides. Loads from .env file if present.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    A .env file in the project root is loaded automatically if present.
    """

    # API keys (required for production use)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    API_AUTH_TOKEN: str = ""

    # Vector store configuration
    VECTORDB_PATH: str = "./data/vectordb"
    COLLECTION_NAME: str = "pci_dss"

    # Embedding configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100

    # Parser configuration
    MAX_CHUNK_TOKENS: int = 200

    # Retrieval configuration
    USE_QUERY_DECOMPOSITION: bool = True

    # Documents directory for ingestion
    DOCUMENTS_DIR: str = "./data/raw"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings populated from environment variables and .env file.
    """
    return Settings()
