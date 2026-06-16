"""RAG Engine for PCI DSS compliance query processing.

Orchestrates the full retrieval-augmented generation pipeline:
embed query → retrieve chunks → generate response → classify risk.

Uses Claude 3.5 Sonnet via Anthropic API with structured JSON output (tool use)
to produce citation-backed compliance assessments with anti-hallucination constraints.
"""

import json
import logging
import os
import re

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

# Maximum number of sub-queries from decomposition
MAX_SUB_QUERIES = 5

# Maximum merged chunks to pass to the LLM
MAX_MERGED_CHUNKS = 12

# Similarity threshold for merged results
SIMILARITY_THRESHOLD = 0.4


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
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        use_query_decomposition: bool = True,
    ):
        """Initialize with vector store and embedding service dependencies.

        Args:
            vector_store: ChromaDB vector store for chunk retrieval.
            embedding_service: Service for generating query embeddings.
            model: Anthropic model to use for generation. Defaults to Claude 3.5 Sonnet.
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            use_query_decomposition: If True, uses multi-query decomposition for
                retrieval. If False, uses legacy single-query path.
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._model = model
        self._use_query_decomposition = use_query_decomposition
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=resolved_key)

    def process_query(self, request: ComplianceQueryRequest) -> ComplianceResponseSchema:
        """Full RAG pipeline: decompose → embed → retrieve → merge → generate → classify.

        Orchestrates the complete compliance query pipeline:
        1. Decompose query into focused sub-queries (or use single query if disabled)
        2. For each sub-query: preprocess, embed, retrieve top-5
        3. Merge all retrieved chunks (deduplicate, keep highest score, cap at 12)
        4. Handle empty retrieval with fallback response
        5. Build constrained prompt with anti-hallucination instructions
        6. Generate response via Claude with structured output
        7. Apply safety overrides (low confidence, conflicts)

        Args:
            request: The compliance query request containing query text
                and optional system context.

        Returns:
            A ComplianceResponseSchema with assessment, risk classification,
            citations, grounding confidence, and retrieved chunk IDs.
        """
        if self._use_query_decomposition:
            chunks = self._retrieve_with_decomposition(request.query)
        else:
            chunks = self._retrieve_single_query(request.query)

        # Handle empty retrieval
        if not chunks:
            logger.info("No relevant chunks found for query: %s", request.query[:100])
            return ComplianceResponseSchema(
                assessment=EMPTY_RETRIEVAL_MESSAGE,
                risk_classification=RiskClassification.WARNING,
                citations=[],
                grounding_confidence=GroundingConfidence.LOW,
                retrieved_chunk_ids=[],
            )

        # Build prompt with anti-hallucination constraints
        prompt = self._build_prompt(
            query=request.query,
            context=request.context,
            chunks=chunks,
        )

        # Generate response via Claude
        response = self._generate_response(prompt=prompt, chunks=chunks)

        return response

    # -------------------------------------------------------------------------
    # Retrieval paths
    # -------------------------------------------------------------------------

    def _retrieve_single_query(self, query: str) -> list[RetrievedChunk]:
        """Legacy single-query retrieval path.

        Preprocesses the query, embeds it, and retrieves top-k chunks
        with threshold filtering.

        Args:
            query: Raw user query text.

        Returns:
            List of RetrievedChunk objects passing the similarity threshold.
        """
        cleaned_query = self._preprocess_query(query)
        logger.debug("Single-query path | cleaned: %s", cleaned_query[:200])

        query_embedding = self._embedding_service.embed_query(cleaned_query)

        # Diagnostic: raw top-10 unfiltered
        raw_top_10 = self._vector_store.query(
            query_embedding=query_embedding, top_k=10, min_similarity=0.0
        )
        logger.debug("RAW top-10 (unfiltered) for single query:")
        for i, rc in enumerate(raw_top_10):
            logger.debug(
                "  [%d] score=%.4f | id=%s | req=%s | section=%s",
                i + 1,
                rc.similarity_score,
                rc.chunk.id,
                rc.chunk.requirement_number or "N/A",
                rc.chunk.section_heading or "N/A",
            )

        # Filtered retrieval
        chunks = self._vector_store.query(
            query_embedding=query_embedding, top_k=10, min_similarity=SIMILARITY_THRESHOLD
        )
        # Drop front-matter/appendix/scope chunks with no requirement number
        chunks = [rc for rc in chunks if rc.chunk.requirement_number]
        return chunks

    def _retrieve_with_decomposition(self, query: str) -> list[RetrievedChunk]:
        """Multi-query decomposition retrieval path.

        Decomposes the scenario into focused sub-queries, retrieves top-3 from
        each sub-query to guarantee representation of each compliance issue,
        then deduplicates by chunk ID (keeping highest score and tracking which
        sub-query contributed it).

        Args:
            query: Raw user query/scenario text.

        Returns:
            Merged, deduplicated list of RetrievedChunk objects sorted by
            descending similarity score.
        """
        # Number of top chunks to take from each sub-query
        TOP_PER_SUBQUERY = 3

        # Step 1: Decompose into sub-queries
        sub_queries = self._decompose_query(query)
        logger.debug(
            "Query decomposed into %d sub-queries: %s",
            len(sub_queries),
            sub_queries,
        )

        # Step 2: For each sub-query, preprocess → embed → retrieve top-5,
        # then take top-3 per sub-query for the merge pool
        # Track best score per chunk ID and which sub-query contributed it
        chunk_map: dict[str, dict] = {}  # chunk_id -> {chunk, score, source_query}

        for sq_idx, sub_query in enumerate(sub_queries):
            cleaned = self._preprocess_query(sub_query)
            logger.debug("  Sub-query %d: '%s' → cleaned: '%s'", sq_idx + 1, sub_query[:80], cleaned[:80])

            embedding = self._embedding_service.embed_query(cleaned)
            results = self._vector_store.query(
                query_embedding=embedding, top_k=5, min_similarity=0.0
            )

            logger.debug("  Sub-query %d retrieved %d chunks:", sq_idx + 1, len(results))
            for i, rc in enumerate(results):
                logger.debug(
                    "    [%d] score=%.4f | id=%s | req=%s",
                    i + 1,
                    rc.similarity_score,
                    rc.chunk.id,
                    rc.chunk.requirement_number or "N/A",
                )

            # Take only top-3 from this sub-query for the merge pool
            # Skip chunks without a requirement number (front-matter/appendix/scope)
            top_for_subquery = [rc for rc in results if rc.chunk.requirement_number][:TOP_PER_SUBQUERY]
            for rc in top_for_subquery:
                chunk_id = rc.chunk.id
                if chunk_id not in chunk_map or rc.similarity_score > chunk_map[chunk_id]["score"]:
                    chunk_map[chunk_id] = {
                        "chunk": rc,
                        "score": rc.similarity_score,
                        "source_query": sub_query,
                    }

        # Step 3: Apply threshold, sort by score descending
        filtered = [
            entry
            for entry in chunk_map.values()
            if entry["score"] >= SIMILARITY_THRESHOLD
        ]
        filtered.sort(key=lambda e: e["score"], reverse=True)

        merged = [entry["chunk"] for entry in filtered]

        logger.debug(
            "Merged results: %d unique chunks (top-%d per sub-query, threshold %.2f)",
            len(merged),
            TOP_PER_SUBQUERY,
            SIMILARITY_THRESHOLD,
        )
        for i, entry in enumerate(filtered):
            rc = entry["chunk"]
            logger.debug(
                "  Final [%d] score=%.4f | id=%s | req=%s | contributed_by='%s'",
                i + 1,
                rc.similarity_score,
                rc.chunk.id,
                rc.chunk.requirement_number or "N/A",
                entry["source_query"][:60],
            )

        return merged

    # -------------------------------------------------------------------------
    # Query decomposition
    # -------------------------------------------------------------------------

    def _decompose_query(self, scenario: str) -> list[str]:
        """Decompose a multi-issue compliance scenario into focused sub-queries.

        Calls a fast LLM to split the scenario into up to MAX_SUB_QUERIES
        self-contained compliance questions. Falls back to [scenario] on
        any failure so the request never breaks.

        Args:
            scenario: The full user scenario/query text.

        Returns:
            List of focused compliance questions (1 to MAX_SUB_QUERIES items).
        """
        decomposition_prompt = (
            "You are a PCI DSS compliance analyst. Decompose the following scenario "
            "into focused retrieval queries for searching a PCI DSS regulatory corpus.\n\n"
            "RULES — each sub-query MUST be:\n"
            "- Under 12 words.\n"
            "- About exactly ONE compliance concept (storage, encryption, access, etc.).\n"
            "- Phrased as a direct permissibility or requirement question.\n"
            "- Free of incidental context words from the scenario (e.g., don't repeat "
            "'logging', 'microservice', 'EBS volume' — extract the underlying regulatory issue).\n"
            "- Maximum 5 sub-queries.\n"
            "- If the scenario covers only one concern, return one item.\n\n"
            "EXAMPLE:\n"
            "Scenario: \"We log full card numbers including CVV to unencrypted debug logs "
            "accessible by all engineers.\"\n"
            "Output:\n"
            '["Is storing the card verification code after authorization permitted?", '
            '"Must stored cardholder data be encrypted at rest?", '
            '"Who may access systems storing cardholder data?"]\n\n'
            "Output ONLY the JSON array, no explanation.\n\n"
            f"Scenario:\n{scenario}"
        )

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": decomposition_prompt},
                ],
            )

            raw_text = message.content[0].text.strip()
            sub_queries = self._parse_json_array(raw_text)

            if not sub_queries:
                logger.warning("Decomposition returned empty list, falling back to original query")
                return [scenario]

            # Cap at max
            return sub_queries[:MAX_SUB_QUERIES]

        except Exception as e:
            logger.warning("Query decomposition failed, falling back to original: %s", str(e))
            return [scenario]

    @staticmethod
    def _parse_json_array(raw_text: str) -> list[str]:
        """Defensively parse a JSON array from LLM output.

        Strips markdown code fences and attempts JSON parsing. Returns an
        empty list on any failure.

        Args:
            raw_text: Raw LLM response text, potentially wrapped in code fences.

        Returns:
            Parsed list of strings, or empty list on failure.
        """
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
            logger.warning("Decomposition JSON is not a list of strings: %s", type(parsed))
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse decomposition JSON: %s | raw: %s", e, cleaned[:200])
            return []

    # -------------------------------------------------------------------------
    # Prompt building & generation
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
        """Detect genuine conflicts between retrieved chunks from DIFFERENT requirements.

        Chunks from the same requirement number (even with different section headings
        like "Defined Approach Requirements" vs "Testing Procedures") are complementary
        parts of one requirement and are never treated as conflicting.

        A conflict is only flagged when chunks from different requirement numbers
        address the same topic but provide opposing guidance — detected when multiple
        distinct major requirement groups (e.g., Req 3 vs Req 7) are present and
        their source files differ, suggesting potentially contradictory regulatory
        directions on the same subject.

        Args:
            chunks: List of retrieved chunks to check for conflicts.

        Returns:
            True if potential conflicts are detected, False otherwise.
        """
        if len(chunks) < 2:
            return False

        # Group chunks by their requirement number
        req_groups: dict[str, list[RetrievedChunk]] = {}
        for rc in chunks:
            req_num = rc.chunk.requirement_number
            if req_num:
                if req_num not in req_groups:
                    req_groups[req_num] = []
                req_groups[req_num].append(rc)

        # Only consider conflicts BETWEEN different requirement numbers.
        # Multiple sections within the same requirement are complementary, not conflicting.
        # A genuine conflict: different requirement numbers from different source files
        # addressing the same query (which means the retrieval pulled opposing guidance).
        unique_requirements = list(req_groups.keys())
        if len(unique_requirements) < 2:
            return False

        # Check if different requirements come from different source documents,
        # which could indicate genuinely conflicting guidance across standard versions
        source_files_per_req: dict[str, set[str]] = {}
        for req_num, rcs in req_groups.items():
            source_files_per_req[req_num] = {rc.chunk.source_file for rc in rcs}

        # Conflict: same source file has different requirements that were retrieved
        # This is normal (a query can span multiple requirements) — NOT a conflict.
        # Real conflict: different source files giving different requirement numbers
        # on the same topic (e.g., v4.0 says X, v4.0.1 says Y)
        all_sources = set()
        for sources in source_files_per_req.values():
            all_sources.update(sources)

        if len(all_sources) > 1:
            # Multiple source documents with different requirements — potential conflict
            logger.info(
                "Potential conflict: chunks from %d different requirements across %d source files: %s",
                len(unique_requirements),
                len(all_sources),
                {req: list(srcs) for req, srcs in source_files_per_req.items()},
            )
            return True

        return False

    @staticmethod
    def _preprocess_query(query: str) -> str:
        """Strip conversational filler from query to improve embedding quality.

        Removes common non-consequential phrases that dilute semantic signal
        (e.g., "is this permitted", "can you tell me", "please check if")
        while preserving the substantive technical content.

        Args:
            query: Raw user query text.

        Returns:
            Cleaned query with filler phrases removed. Falls back to the
            original query if cleaning would produce an empty string.
        """
        # Phrases that add no semantic value for retrieval
        filler_patterns = [
            r"\bis\s+this\s+(permitted|allowed|compliant|okay|ok)\b",
            r"\bcan\s+you\s+(tell\s+me|check|verify|confirm|assess)\b",
            r"\bplease\s+(check|verify|confirm|tell\s+me|assess|evaluate)\b",
            r"\bI\s+(want\s+to|need\s+to|would\s+like\s+to)\s+know\b",
            r"\bdo\s+we\s+(need|have)\s+to\b",
            r"\bis\s+it\s+(possible|necessary|required|okay|ok)\s+to\b",
            r"\bwhat\s+are\s+the\s+requirements\s+for\b",
            r"\bhow\s+do\s+I\b",
            r"\bwould\s+this\s+be\b",
            r"\bdoes\s+this\s+(comply|violate|meet)\b",
        ]

        cleaned = query
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Collapse multiple spaces and strip
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Don't return empty string — fall back to original
        return cleaned if cleaned else query
