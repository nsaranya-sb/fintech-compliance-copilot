"""RAG Engine for PCI DSS compliance query processing.

Orchestrates the full retrieval-augmented generation pipeline:
embed query → retrieve chunks → generate response → classify risk.

Uses Claude 3.5 Sonnet via Anthropic API with structured JSON output (tool use)
to produce citation-backed compliance assessments with anti-hallucination constraints.
"""

import logging
import os

from anthropic import Anthropic

from src.embeddings.embedding_service import EmbeddingService
from src.models import (
    ComplianceQueryRequest,
    ComplianceResponseSchema,
    GroundingConfidence,
    RetrievedChunk,
    RiskClassification,
)
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Disclaimer prepended when grounding confidence is Low
LOW_CONFIDENCE_DISCLAIMER = (
    "⚠️ LOW SOURCE CONFIDENCE: The following assessment has limited backing "
    "in the available PCI DSS documentation. Verify with authoritative sources "
    "before making compliance decisions.\n\n"
)

# Fallback message when no chunks are retrieved
EMPTY_RETRIEVAL_MESSAGE = "Clause not found in source documentation"


class RAGEngine:
    """Orchestrates retrieval-augmented generation for PCI DSS compliance queries.

    Implements the full RAG pipeline: embed query → retrieve relevant chunks
    from vector store → generate citation-backed response via Claude 3.5 Sonnet
    → classify risk. Enforces anti-hallucination grounding by constraining
    generation to retrieved source content only.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        """Initialize with vector store and embedding service dependencies.

        Args:
            vector_store: ChromaDB vector store for chunk retrieval.
            embedding_service: Service for generating query embeddings.
            model: Anthropic model to use for generation. Defaults to Claude 3.5 Sonnet.
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._model = model
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=resolved_key)

    def process_query(self, request: ComplianceQueryRequest) -> ComplianceResponseSchema:
        """Full RAG pipeline: embed query → retrieve → generate → classify.

        Orchestrates the complete compliance query pipeline:
        1. Embed the query using the embedding service
        2. Retrieve top-k similar chunks from the vector store
        3. Handle empty retrieval with fallback response
        4. Build constrained prompt with anti-hallucination instructions
        5. Generate response via Claude with structured output
        6. Apply safety overrides (low confidence, conflicts)

        Args:
            request: The compliance query request containing query text
                and optional system context.

        Returns:
            A ComplianceResponseSchema with assessment, risk classification,
            citations, grounding confidence, and retrieved chunk IDs.
        """
        # Step 1: Embed the query
        query_embedding = self._embedding_service.embed_query(request.query)

        # Step 2: Retrieve relevant chunks
        chunks = self._vector_store.query(query_embedding=query_embedding)

        # Debug log: top retrieved chunks before any filtering or LLM generation
        logger.debug(
            "Retrieved %d chunks for query: %.100s", len(chunks), request.query
        )
        for i, rc in enumerate(chunks[:10]):
            logger.debug(
                "  [%d] score=%.4f | id=%s | req=%s | section=%s",
                i + 1,
                rc.similarity_score,
                rc.chunk.id,
                rc.chunk.requirement_number or "N/A",
                rc.chunk.section_heading or "N/A",
            )

        # Step 3: Handle empty retrieval
        if not chunks:
            logger.info("No relevant chunks found for query: %s", request.query[:100])
            return ComplianceResponseSchema(
                assessment=EMPTY_RETRIEVAL_MESSAGE,
                risk_classification=RiskClassification.WARNING,
                citations=[],
                grounding_confidence=GroundingConfidence.LOW,
                retrieved_chunk_ids=[],
            )

        # Step 4: Build prompt with anti-hallucination constraints
        prompt = self._build_prompt(
            query=request.query,
            context=request.context,
            chunks=chunks,
        )

        # Step 5: Generate response via Claude
        response = self._generate_response(prompt=prompt, chunks=chunks)

        return response

    def _build_prompt(
        self, query: str, context: str | None, chunks: list[RetrievedChunk]
    ) -> str:
        """Construct Claude prompt with anti-hallucination constraints.

        Builds a structured prompt that:
        - Provides the retrieved source chunks as the ONLY knowledge base
        - Explicitly instructs Claude to restrict generation to chunk content
        - Includes citation format requirements
        - Provides the user's query and optional system context

        Args:
            query: The compliance question text.
            context: Optional system architecture or payment flow context.
            chunks: Retrieved chunks from the vector store.

        Returns:
            The fully constructed prompt string for Claude.
        """
        # Format source chunks
        chunks_text = ""
        for i, retrieved in enumerate(chunks, 1):
            chunk = retrieved.chunk
            req_info = f"Requirement {chunk.requirement_number}" if chunk.requirement_number else "General"
            section_info = chunk.section_heading or "N/A"
            chunks_text += (
                f"--- Source Chunk {i} ---\n"
                f"ID: {chunk.id}\n"
                f"Requirement: {req_info}\n"
                f"Section: {section_info}\n"
                f"Source: {chunk.source_file}, Page {chunk.page_number}\n"
                f"Similarity Score: {retrieved.similarity_score:.3f}\n"
                f"Content:\n{chunk.text}\n\n"
            )

        # Build the full prompt
        prompt = (
            "You are a PCI DSS v4.0.1 compliance assessment expert. Your role is to "
            "provide accurate, citation-backed compliance assessments based EXCLUSIVELY "
            "on the provided source chunks.\n\n"
            "## CRITICAL ANTI-HALLUCINATION CONSTRAINTS\n\n"
            "You MUST only use information from the provided source chunks. "
            "Do NOT introduce external knowledge.\n"
            "You MUST NOT infer, assume, or generate any regulatory interpretation "
            "beyond what is explicitly stated in the source chunks.\n"
            "Every factual claim in your assessment MUST be directly traceable to "
            "at least one source chunk.\n"
            "If the source chunks do not contain sufficient information to fully "
            "answer the query, state what is covered and what is not.\n\n"
            "## CITATION FORMAT\n\n"
            "Format all citations as: "
            '"Requirement X.Y[.Z] under PCI DSS v4.0.1, Section [section_name]"\n'
            "where X, Y, Z are the requirement numbers and section_name is the "
            "section heading from the source chunk.\n\n"
            "## CONFLICT DETECTION\n\n"
            "If source chunks provide conflicting or contradictory guidance on the "
            "same topic, you MUST:\n"
            "1. Flag the conflict explicitly in your assessment\n"
            "2. Reference all conflicting sources\n"
            "3. Set risk_classification to '🟡 Warning'\n\n"
            "## SOURCE CHUNKS\n\n"
            f"{chunks_text}"
            "## USER QUERY\n\n"
            f"{query}\n\n"
        )

        if context:
            prompt += f"## SYSTEM CONTEXT\n\n{context}\n\n"

        prompt += (
            "## RESPONSE INSTRUCTIONS\n\n"
            "Provide your compliance assessment using the structured output tool. "
            "Ensure:\n"
            "- assessment: A detailed, citation-backed compliance evaluation\n"
            "- risk_classification: Exactly one of '🟢 Compliant', '🟡 Warning', "
            "or '🔴 Non-Compliant'\n"
            "- citations: Array of citation strings in the required format\n"
            "- grounding_confidence: 'High', 'Medium', or 'Low' based on how well "
            "the source chunks support your assessment\n"
        )

        return prompt

    def _generate_response(
        self, prompt: str, chunks: list[RetrievedChunk]
    ) -> ComplianceResponseSchema:
        """Call Claude 3.5 Sonnet with structured output via tool use.

        Uses Anthropic's tool use feature to enforce structured JSON output
        matching the ComplianceResponseSchema. Handles API failures with a
        single retry, then propagates the error.

        Args:
            prompt: The fully constructed prompt with source chunks and constraints.
            chunks: The retrieved chunks (used for chunk ID extraction).

        Returns:
            A ComplianceResponseSchema populated from Claude's structured output.

        Raises:
            Exception: If the Anthropic API fails after retry.
        """
        # Define the tool schema for structured output
        compliance_tool = {
            "name": "provide_compliance_assessment",
            "description": (
                "Provide a structured PCI DSS compliance assessment based on "
                "the retrieved source chunks. All fields are required."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "assessment": {
                        "type": "string",
                        "description": (
                            "Citation-backed compliance assessment text. Must reference "
                            "specific PCI DSS requirements from the source chunks."
                        ),
                    },
                    "risk_classification": {
                        "type": "string",
                        "enum": [
                            "🟢 Compliant",
                            "🟡 Warning",
                            "🔴 Non-Compliant",
                        ],
                        "description": "Tri-state risk classification.",
                    },
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of citations in format: "
                            "'Requirement X.Y[.Z] under PCI DSS v4.0.1, Section [section_name]'"
                        ),
                    },
                    "grounding_confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"],
                        "description": (
                            "How well the source chunks support the assessment. "
                            "'High' = strong direct support, 'Medium' = partial support, "
                            "'Low' = weak or indirect support."
                        ),
                    },
                },
                "required": [
                    "assessment",
                    "risk_classification",
                    "citations",
                    "grounding_confidence",
                ],
            },
        }

        # Call Claude with tool use
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                tools=[compliance_tool],
                tool_choice={"type": "tool", "name": "provide_compliance_assessment"},
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            # Retry once on failure
            logger.warning("Claude API call failed, retrying: %s", str(e))
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    tools=[compliance_tool],
                    tool_choice={
                        "type": "tool",
                        "name": "provide_compliance_assessment",
                    },
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as retry_error:
                logger.error(
                    "Claude API call failed after retry: %s", str(retry_error)
                )
                raise

        # Extract structured output from tool use response
        tool_result = self._extract_tool_result(message)

        # Collect retrieved chunk IDs
        retrieved_chunk_ids = [rc.chunk.id for rc in chunks]

        # Parse grounding confidence
        grounding_confidence = GroundingConfidence(tool_result["grounding_confidence"])

        # Parse risk classification
        risk_classification = RiskClassification(tool_result["risk_classification"])

        # Get assessment text
        assessment = tool_result["assessment"]

        # Get citations
        citations = tool_result.get("citations", [])

        # Detect conflicting chunks and apply conflict handling
        if self._detect_conflicts(chunks):
            if "conflict" not in assessment.lower():
                assessment = (
                    "⚠️ CONFLICTING SOURCES DETECTED: The retrieved source chunks "
                    "provide conflicting guidance on this topic. "
                ) + assessment
            risk_classification = RiskClassification.WARNING

        # Apply low confidence safety override (Requirement 6.5)
        if grounding_confidence == GroundingConfidence.LOW:
            # Never assign Compliant when confidence is Low
            if risk_classification == RiskClassification.COMPLIANT:
                risk_classification = RiskClassification.WARNING
            # Prepend disclaimer
            assessment = LOW_CONFIDENCE_DISCLAIMER + assessment

        return ComplianceResponseSchema(
            assessment=assessment,
            risk_classification=risk_classification,
            citations=citations,
            grounding_confidence=grounding_confidence,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    def _extract_tool_result(self, message) -> dict:
        """Extract the tool use result from Claude's response.

        Parses the message content blocks to find the tool_use block
        and returns its input as a dictionary.

        Args:
            message: The Anthropic API message response.

        Returns:
            Dictionary containing the structured tool output fields.

        Raises:
            ValueError: If no tool use block is found in the response.
        """
        for block in message.content:
            if block.type == "tool_use":
                return block.input

        raise ValueError(
            "No tool_use block found in Claude response. "
            "Expected structured output via tool use."
        )

    def _detect_conflicts(self, chunks: list[RetrievedChunk]) -> bool:
        """Detect potential conflicts between retrieved chunks.

        Checks if chunks reference the same requirement number but come from
        different sections or source files, which may indicate conflicting guidance.

        A simple heuristic: if multiple chunks reference the same requirement
        number but have different section headings, flag as potential conflict.

        Args:
            chunks: List of retrieved chunks to check for conflicts.

        Returns:
            True if potential conflicts are detected, False otherwise.
        """
        if len(chunks) < 2:
            return False

        # Group chunks by requirement number
        req_sections: dict[str, set[str]] = {}
        for rc in chunks:
            req_num = rc.chunk.requirement_number
            if req_num:
                section = rc.chunk.section_heading or ""
                if req_num not in req_sections:
                    req_sections[req_num] = set()
                req_sections[req_num].add(section)

        # If any requirement has chunks from multiple different sections,
        # it might indicate conflicting guidance
        for req_num, sections in req_sections.items():
            # Filter out empty sections and check if there are truly different sections
            non_empty_sections = {s for s in sections if s}
            if len(non_empty_sections) > 1:
                logger.info(
                    "Potential conflict detected for Requirement %s across sections: %s",
                    req_num,
                    non_empty_sections,
                )
                return True

        return False
