"""Application entry point for the PCI DSS RAG Compliance API.

Loads environment variables, wires all dependencies, and starts the
FastAPI application via uvicorn.
"""

import logging

from dotenv import load_dotenv

# Load .env file before anything else — override=True ensures .env takes
# precedence over any stale shell-level environment variables
load_dotenv(override=True)

from src.api.app import app  # noqa: E402
from src.dependencies import wire_dependencies  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Enable verbose retrieval diagnostics for the RAG engine
logging.getLogger("src.rag.engine").setLevel(logging.DEBUG)

# Wire all dependencies on module load so the app is ready to serve
wire_dependencies()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
