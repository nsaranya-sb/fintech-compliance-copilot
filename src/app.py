"""
Fintech RegTech Compliance Copilot — Streamlit Frontend

A professional interface for running PCI DSS v4.0.1 compliance audits
against system architectures via the RAG-powered FastAPI backend.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{BACKEND_URL}/api/v1/compliance/query"
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
REQUEST_TIMEOUT_SECONDS = 120

DEFAULT_QUERY = """\
Our payments microservice temporarily logs full credit card numbers \
(including CVV/CVC) to an application debug log file stored on an \
unencrypted EBS volume. The logs rotate every 24 hours and are \
accessible by all engineers via SSH. We plan to tokenize in Phase 2.\
"""

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fintech RegTech Compliance Copilot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .stTextArea textarea { font-family: 'SF Mono', 'Fira Code', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏛️ Fintech RegTech Compliance Copilot")
st.caption("Instant, citation-backed PCI DSS v4.0.1 architecture auditing.")

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

query = st.text_area(
    "Describe your architecture, payment flow, or product feature:",
    value=DEFAULT_QUERY,
    height=180,
    placeholder="Paste a system design, data-flow description, or compliance question…",
)

run_audit = st.button("🔍 Run Compliance Audit", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Audit execution
# ---------------------------------------------------------------------------

if run_audit:
    if not query.strip():
        st.error("Please enter a compliance query or architecture description.")
        st.stop()

    if not AUTH_TOKEN:
        st.error(
            "Authentication token not configured. "
            "Set the `AUTH_TOKEN` environment variable before running the app."
        )
        st.stop()

    with st.spinner("Auditing architecture against PCI DSS mandates…"):
        try:
            response = requests.post(
                QUERY_ENDPOINT,
                json={"query": query.strip()},
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Could not reach the compliance backend. "
                f"Ensure the server is running at `{BACKEND_URL}`."
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error(
                "⏱️ The request timed out. The backend may be under heavy load. "
                "Please try again shortly."
            )
            st.stop()
        except requests.exceptions.RequestException as exc:
            st.error(f"⚠️ Unexpected network error: {exc}")
            st.stop()

    # -----------------------------------------------------------------------
    # Handle non-200 responses
    # -----------------------------------------------------------------------

    if response.status_code == 401:
        st.error("🔒 Authentication failed. Check your `AUTH_TOKEN` value.")
        st.stop()
    elif response.status_code == 400:
        detail = response.json().get("detail", "Invalid request.")
        st.warning(f"Bad request: {detail}")
        st.stop()
    elif response.status_code == 503:
        st.warning(
            "🔄 The compliance engine is temporarily unavailable. "
            "Please retry in a few moments."
        )
        st.stop()
    elif response.status_code != 200:
        st.error(
            f"Server returned HTTP {response.status_code}. "
            "Please contact the platform team if this persists."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Parse and render the response
    # -----------------------------------------------------------------------

    data = response.json()

    assessment: str = data.get("assessment", "")
    risk_classification: str = data.get("risk_classification", "")
    citations: list[str] = data.get("citations", [])
    grounding_confidence: str = data.get("grounding_confidence", "")
    retrieved_chunk_ids: list[str] = data.get("retrieved_chunk_ids", [])

    st.divider()

    # --- Risk banner ---
    col_risk, col_confidence = st.columns([4, 1])

    with col_risk:
        if "Non-Compliant" in risk_classification:
            st.error(f"**{risk_classification}**")
        elif "Warning" in risk_classification:
            st.warning(f"**{risk_classification}**")
        else:
            st.success(f"**{risk_classification}**")

    with col_confidence:
        confidence_colors = {"High": "green", "Medium": "orange", "Low": "red"}
        badge_color = confidence_colors.get(grounding_confidence, "gray")
        st.markdown(
            f'<span style="background:{badge_color};color:white;'
            f'padding:4px 10px;border-radius:12px;font-size:0.85em;">'
            f"Confidence: {grounding_confidence}</span>",
            unsafe_allow_html=True,
        )

    # --- Assessment text ---
    st.subheader("Assessment")
    st.markdown(assessment)

    # --- Citations ---
    if citations:
        st.subheader("📌 Cited PCI DSS Sources")
        for i, cite in enumerate(citations, 1):
            st.markdown(f"{i}. {cite}")

    # --- Retrieval context ---
    with st.expander("📚 Source Citations & Grounding Context"):
        if retrieved_chunk_ids:
            st.markdown("**Retrieved Chunk IDs used for this assessment:**")
            for chunk_id in retrieved_chunk_ids:
                st.code(chunk_id, language=None)
        else:
            st.info("No retrieval chunks were returned for this query.")
