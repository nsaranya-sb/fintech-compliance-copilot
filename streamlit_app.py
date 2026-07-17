"""
Fintech RegTech Compliance Copilot — Streamlit Cloud Entrypoint (Single-Process)

This frontend interfaces directly with the Python RAG engine to audit system 
architectures against PCI DSS v4.0.1, bypassing HTTP API calls. Sessions are capped 
at 3 queries to manage resource usage.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from src.config import get_settings
from src.models import ComplianceQueryRequest
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_store import ChromaVectorStore
from src.rag.engine import RAGEngine

# ---------------------------------------------------------------------------
# Caching the RAG Engine Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def get_rag_engine():
    """Initialize and cache the RAG engine dependencies and client."""
    settings = get_settings()
    
    # Resolve API keys from settings or environment
    openai_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    anthropic_key = settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    
    embedding_service = EmbeddingService(
        model=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        api_key=openai_key,
    )
    
    vector_store = ChromaVectorStore(
        persist_directory=settings.VECTORDB_PATH,
        collection_name=settings.COLLECTION_NAME,
    )
    
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_service=embedding_service,
        api_key=anthropic_key,
        use_query_decomposition=settings.USE_QUERY_DECOMPOSITION,
    )
    
    return rag_engine

# ---------------------------------------------------------------------------
# Page configuration and Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fintech RegTech Compliance Copilot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .stTextArea textarea { font-family: 'SF Mono', 'Fira Code', monospace; }
        /* Premium custom typography and styling */
        h1, h2, h3 { font-family: 'Outfit', 'Inter', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏛️ Fintech RegTech Compliance Copilot")
st.caption("Instant, citation-backed PCI DSS v4.0.1 architecture auditing (Direct Engine Access).")
st.divider()

# ---------------------------------------------------------------------------
# Session State & Query Capping
# ---------------------------------------------------------------------------

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

remaining_queries = max(0, 3 - st.session_state.query_count)

# Display Session Status Badge Card
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(29, 78, 216, 0.15) 0%, rgba(30, 64, 175, 0.03) 100%);
                padding: 14px 20px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 1.05rem; color: #60a5fa;">🔐 Demo Session Status</span>
            <span style="background: rgba(96, 165, 250, 0.2); color: #93c5fd; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold;">
                {remaining_queries} / 3 Queries Left
            </span>
        </div>
        <p style="font-size: 0.85rem; margin-top: 6px; margin-bottom: 0; color: #a1a1aa;">
            To manage demo resource consumption, each session is limited to 3 queries. Ask targeted architectural questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------

DEFAULT_QUERY = """\
Our payments microservice temporarily logs full credit card numbers \
(including CVV/CVC) to an application debug log file stored on an \
unencrypted EBS volume. The logs rotate every 24 hours and are \
accessible by all engineers via SSH. We plan to tokenize in Phase 2.\
"""

query = st.text_area(
    "Describe your architecture, payment flow, or product feature:",
    value=DEFAULT_QUERY,
    height=160,
    placeholder="Paste a system design, data-flow description, or compliance question…",
)

button_disabled = remaining_queries <= 0

if button_disabled:
    st.warning("⚠️ You have reached the query limit of 3 for this session. Please reload or restart the application to reset.")

run_audit = st.button(
    "🔍 Run Compliance Audit",
    type="primary",
    use_container_width=True,
    disabled=button_disabled
)

# ---------------------------------------------------------------------------
# Audit Execution
# ---------------------------------------------------------------------------

if run_audit:
    if not query.strip():
        st.error("Please enter a compliance query or architecture description.")
        st.stop()
        
    settings = get_settings()
    openai_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    anthropic_key = settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key or not anthropic_key:
        st.error(
            "🔑 API keys are missing. Please configure `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` "
            "as environment variables or Streamlit Secrets before running the audit."
        )
        st.stop()

    with st.spinner("Auditing architecture against PCI DSS mandates… (using direct RAG engine)"):
        try:
            # Get the cached RAG Engine
            engine = get_rag_engine()
            
            # Execute RAG query (synchronous invocation)
            request_data = ComplianceQueryRequest(query=query.strip())
            response = engine.process_query(request_data)
            
            # Update session state and refresh
            st.session_state.query_count += 1
            st.session_state.last_result = response
            st.rerun()
            
        except Exception as exc:
            st.error(f"⚠️ RAG engine processing failed: {exc}")

# ---------------------------------------------------------------------------
# Rendering Results
# ---------------------------------------------------------------------------

if st.session_state.last_result is not None:
    res = st.session_state.last_result
    
    st.divider()
    
    # Layout Status Indicators
    col_risk, col_confidence = st.columns([3, 1])
    
    with col_risk:
        # Safely extract the string value and name of the enum to handle Pydantic/Python serialization differences
        risk_val = getattr(res.risk_classification, "value", str(res.risk_classification))
        risk_name = getattr(res.risk_classification, "name", "")
        
        # Check both lowercased value and name for status matching
        if "non_compliant" in risk_name.lower() or "non-compliant" in risk_val.lower():
            bg_color = "rgba(239, 68, 68, 0.12)"
            border_color = "rgba(239, 68, 68, 0.3)"
            text_color = "#fca5a5"
        elif "warning" in risk_name.lower() or "warning" in risk_val.lower():
            bg_color = "rgba(245, 158, 11, 0.12)"
            border_color = "rgba(245, 158, 11, 0.3)"
            text_color = "#fde047"
        else:
            bg_color = "rgba(16, 185, 129, 0.12)"
            border_color = "rgba(16, 185, 129, 0.3)"
            text_color = "#6ee7b7"
            
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; border: 1px solid {border_color}; 
                        padding: 12px 18px; border-radius: 8px; font-weight: bold; 
                        color: {text_color}; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
                Risk Status: {risk_val}
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_confidence:
        confidence_str = str(res.grounding_confidence)
        if "High" in confidence_str:
            conf_color = "#10b981"
        elif "Medium" in confidence_str:
            conf_color = "#f59e0b"
        else:
            conf_color = "#ef4444"
            
        st.markdown(
            f"""
            <div style="text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); 
                        padding: 12px; border-radius: 8px; background-color: rgba(255, 255, 255, 0.02);">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #a1a1aa; letter-spacing: 0.05em;">Grounding Confidence</div>
                <div style="font-weight: bold; color: {conf_color}; font-size: 1.1rem;">{confidence_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Assessment text
    st.subheader("📋 Audit Assessment")
    st.markdown(res.assessment)
    
    # Citations
    if res.citations:
        st.subheader("📌 Cited PCI DSS Requirements")
        for i, cite in enumerate(res.citations, 1):
            st.markdown(f"**{i}.** {cite}")
            
    # Source metadata/chunk diagnostics
    with st.expander("📚 Source Grounding Context"):
        if res.retrieved_chunk_ids:
            st.markdown("**Retrieved Document Chunk IDs:**")
            for chunk_id in res.retrieved_chunk_ids:
                st.code(chunk_id, language=None)
        else:
            st.info("No grounding context was returned.")
