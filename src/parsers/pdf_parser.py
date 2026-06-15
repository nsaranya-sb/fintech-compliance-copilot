"""PDF document parser for PCI DSS regulatory documents.

Uses PyMuPDF (fitz) for structured text extraction with section heading
detection and PCI DSS requirement number identification.
"""

import logging
import re
from pathlib import Path

import fitz

from src.models import Chunk, PageContent, ParsedDocument

logger = logging.getLogger(__name__)

# Regex patterns for PCI DSS document structure
REQUIREMENT_PATTERN = re.compile(
    r"Requirement\s+(\d+(?:\.\d+)*(?:\.\d+)*)", re.IGNORECASE
)
# Matches lines starting with a requirement number directly, e.g. "3.3 Sensitive..."
# Must be at least X.Y format (two numeric components) to avoid matching page numbers
INLINE_REQUIREMENT_PATTERN = re.compile(
    r"^(\d+\.\d+(?:\.\d+)*)\s+[A-Z]"
)
SECTION_HEADING_PATTERN = re.compile(
    r"^(?:Requirement\s+\d+|Section\s+\d+|Appendix\s+[A-Z]|"
    r"Overview|Purpose|Scope|Definitions?|Guidance|"
    r"Testing\s+Procedures?|Customized\s+Approach)",
    re.IGNORECASE | re.MULTILINE,
)


class PDFParser:
    """Parses PCI DSS PDF documents into structured content with metadata.

    Extracts text page-by-page using PyMuPDF, detects section headings and
    PCI DSS requirement numbers, and produces structured PageContent and
    ParsedDocument objects.
    """

    def __init__(self, max_chunk_tokens: int = 200):
        """Initialize parser with configurable chunk size.

        Args:
            max_chunk_tokens: Maximum number of tokens per chunk. Smaller chunks
                (~200 tokens) produce sharper embeddings for regulatory text,
                reducing score clustering around 0.5.
        """
        self.max_chunk_tokens = max_chunk_tokens

    def parse_document(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF file and return structured content.

        Extracts pages, then produces chunks from the extracted content.
        Chunking logic (segment_into_chunks) is a placeholder pending Task 2.2.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with source file info, total pages, and chunks.

        Raises:
            No exceptions are raised to the caller; errors are logged and
            a ParsedDocument with zero chunks is returned.
        """
        pages = self.extract_pages(file_path)

        if not pages:
            return ParsedDocument(
                source_file=file_path.name,
                total_pages=0,
                chunks=[],
            )

        chunks = self.segment_into_chunks(pages, file_path.name)

        return ParsedDocument(
            source_file=file_path.name,
            total_pages=len(pages),
            chunks=chunks,
        )

    def extract_pages(self, file_path: Path) -> list[PageContent]:
        """Extract raw text content page by page with metadata.

        Uses PyMuPDF fitz.open() and page.get_text("dict") for structured
        text extraction. Detects section headings and PCI DSS requirement
        numbers via regex patterns.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of PageContent objects, one per page with text and detected
            headings. Returns an empty list if the file cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("PDF file not found: %s", file_path)
            return []

        if not file_path.suffix.lower() == ".pdf":
            logger.error(
                "Unsupported file format '%s': %s", file_path.suffix, file_path
            )
            return []

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            logger.error(
                "Failed to open PDF file '%s': %s", file_path, exc
            )
            return []

        pages: list[PageContent] = []

        try:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_dict = page.get_text("dict")

                    text_parts: list[str] = []
                    headings: list[str] = []

                    for block in page_dict.get("blocks", []):
                        # Skip image blocks (type 1)
                        if block.get("type") != 0:
                            continue

                        for line in block.get("lines", []):
                            line_text = ""
                            max_font_size = 0.0
                            is_bold = False

                            for span in line.get("spans", []):
                                span_text = span.get("text", "")
                                line_text += span_text
                                font_size = span.get("size", 0)
                                if font_size > max_font_size:
                                    max_font_size = font_size
                                # Detect bold via font name containing "Bold"
                                font_name = span.get("font", "")
                                if "bold" in font_name.lower():
                                    is_bold = True

                            line_text = line_text.strip()
                            if not line_text:
                                continue

                            text_parts.append(line_text)

                            # Detect headings: larger font, bold text, or
                            # matching known section heading patterns
                            if self._is_heading(
                                line_text, max_font_size, is_bold
                            ):
                                headings.append(line_text)

                    full_text = "\n".join(text_parts)

                    # Also detect requirement numbers mentioned in the text
                    req_matches = REQUIREMENT_PATTERN.findall(full_text)
                    for req_num in req_matches:
                        heading = f"Requirement {req_num}"
                        if heading not in headings:
                            headings.append(heading)

                    pages.append(
                        PageContent(
                            page_number=page_num + 1,  # 1-indexed
                            text=full_text,
                            headings=headings,
                        )
                    )

                except Exception as exc:
                    logger.warning(
                        "Error extracting page %d from '%s': %s",
                        page_num + 1,
                        file_path,
                        exc,
                    )
                    # Skip this page but continue processing
                    continue

        finally:
            doc.close()

        return pages

    def segment_into_chunks(
        self, pages: list[PageContent], source_file: str
    ) -> list[Chunk]:
        """Segment extracted pages into semantically coherent chunks.

        Preserves semantic boundaries at section/requirement level, extends
        chunk boundaries to include complete sentences, and annotates each
        chunk with metadata.

        The concatenation of all chunk texts joined with "\\n" reproduces the
        original full text from all pages joined with "\\n" (round-trip property).

        Args:
            pages: List of PageContent objects extracted from a PDF.
            source_file: Name of the source PDF file.

        Returns:
            List of Chunk objects with metadata annotations.
        """
        if not pages:
            return []

        # Build a flat list of segments, each being a section boundary unit.
        # A segment is a contiguous block of text belonging to one semantic section.
        segments = self._build_segments(pages)

        # Now chunk each segment respecting max_chunk_tokens and sentence boundaries
        chunks: list[Chunk] = []
        chunk_index = 0

        for segment in segments:
            segment_chunks = self._chunk_segment(
                text=segment["text"],
                source_file=source_file,
                requirement_number=segment["requirement_number"],
                section_heading=segment["section_heading"],
                page_number=segment["page_number"],
                start_chunk_index=chunk_index,
            )
            chunks.extend(segment_chunks)
            chunk_index += len(segment_chunks)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-count approximation.

        Uses len(text.split()) as a simple token estimation approach.
        """
        return len(text.split())

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences at sentence boundaries.

        Splits on `. `, `? `, `! ` followed by an uppercase letter or end-of-text.
        Preserves the sentence-ending punctuation with the sentence.
        """
        if not text:
            return []

        # Split on sentence boundaries: period/question/exclamation followed by
        # space and uppercase letter, or end of text
        sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])"
        )

        sentences = sentence_pattern.split(text)
        return [s for s in sentences if s]

    def _build_segments(
        self, pages: list[PageContent]
    ) -> list[dict]:
        """Build semantic segments from pages based on section/requirement boundaries.

        Each segment represents a contiguous block of text that belongs to a
        single semantic section. A new segment starts when a REQUIREMENT_PATTERN
        or SECTION_HEADING_PATTERN is detected at the start of a line.

        Delegates to _build_segments_from_full_text to guarantee the round-trip
        property: "\n".join(segment texts) == "\n".join(page.text for page in pages).

        Returns a list of dicts with keys: text, requirement_number,
        section_heading, page_number.
        """
        return self._build_segments_from_full_text(pages)

    def _build_segments_from_full_text(
        self, pages: list[PageContent]
    ) -> list[dict]:
        """Build segments from the full joined text to guarantee round-trip property.

        The full text is defined as "\\n".join(page.text for page in pages).
        We split this into segments at section/requirement boundaries (line-level),
        tracking which page each segment starts on.
        """
        if not pages:
            return []

        # Build full text and a mapping from line index to page number
        all_lines: list[str] = []
        line_to_page: list[int] = []

        for page in pages:
            page_lines = page.text.split("\n")
            for line in page_lines:
                all_lines.append(line)
                line_to_page.append(page.page_number)

        # Now segment by section/requirement boundaries
        segments: list[dict] = []
        current_start: int = 0
        current_requirement: str | None = None
        current_heading: str | None = None

        # Extract initial metadata from the first page
        if pages[0].headings:
            req_num = self._extract_requirement_number(pages[0].headings)
            current_requirement = req_num
            current_heading = pages[0].headings[0]

        for i, line in enumerate(all_lines):
            is_new_section = False
            detected_req_num = None

            req_match = REQUIREMENT_PATTERN.match(line)
            if req_match:
                is_new_section = True
                detected_req_num = req_match.group(1)
            else:
                inline_match = INLINE_REQUIREMENT_PATTERN.match(line)
                if inline_match:
                    is_new_section = True
                    detected_req_num = inline_match.group(1)
                elif SECTION_HEADING_PATTERN.match(line):
                    is_new_section = True

            if is_new_section and i > current_start:
                # Flush previous segment
                segment_text = "\n".join(all_lines[current_start:i])
                segments.append({
                    "text": segment_text,
                    "requirement_number": current_requirement,
                    "section_heading": current_heading,
                    "page_number": line_to_page[current_start],
                })
                current_start = i

                # Update metadata for new section
                if detected_req_num:
                    current_requirement = detected_req_num
                    current_heading = line.strip()
                else:
                    current_heading = line.strip()

        # Flush last segment
        if current_start < len(all_lines):
            segment_text = "\n".join(all_lines[current_start:])
            segments.append({
                "text": segment_text,
                "requirement_number": current_requirement,
                "section_heading": current_heading,
                "page_number": line_to_page[current_start],
            })

        return segments

    def _chunk_segment(
        self,
        text: str,
        source_file: str,
        requirement_number: str | None,
        section_heading: str | None,
        page_number: int,
        start_chunk_index: int,
    ) -> list[Chunk]:
        """Split a segment into chunks respecting max_chunk_tokens and sentence boundaries.

        For segments within the token limit, produces a single chunk.
        For larger segments, splits at sentence boundaries, extending chunk
        boundaries to include complete sentences.

        The concatenation of all chunk texts joined with "\\n" equals the
        original segment text.

        Args:
            text: The segment text to chunk.
            source_file: Source PDF filename.
            requirement_number: PCI DSS requirement number for this segment.
            section_heading: Section heading for this segment.
            page_number: Page number where this segment starts.
            start_chunk_index: Starting chunk_index for numbering.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        # If the segment fits within the token limit, return as single chunk
        if self._estimate_tokens(text) <= self.max_chunk_tokens:
            return [
                Chunk(
                    id=f"{source_file}::chunk{start_chunk_index}",
                    text=text,
                    source_file=source_file,
                    requirement_number=requirement_number,
                    section_heading=section_heading,
                    page_number=page_number,
                    chunk_index=start_chunk_index,
                )
            ]

        # Split into sentences for boundary-respecting chunking
        sentences = self._split_sentences(text)

        # If we can't split into sentences (e.g., no sentence boundaries),
        # return the whole text as one chunk
        if len(sentences) <= 1:
            return [
                Chunk(
                    id=f"{source_file}::chunk{start_chunk_index}",
                    text=text,
                    source_file=source_file,
                    requirement_number=requirement_number,
                    section_heading=section_heading,
                    page_number=page_number,
                    chunk_index=start_chunk_index,
                )
            ]

        # Group sentences into chunks respecting the token limit
        chunks: list[Chunk] = []
        current_chunk_sentences: list[str] = []
        current_tokens = 0
        chunk_idx = start_chunk_index

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If adding this sentence would exceed the limit and we have content
            if (
                current_tokens + sentence_tokens > self.max_chunk_tokens
                and current_chunk_sentences
            ):
                # Flush current chunk
                # We need to reconstruct the text that when joined matches original
                chunk_text = self._rejoin_sentences(current_chunk_sentences)
                chunks.append(
                    Chunk(
                        id=f"{source_file}::chunk{chunk_idx}",
                        text=chunk_text,
                        source_file=source_file,
                        requirement_number=requirement_number,
                        section_heading=section_heading,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                current_chunk_sentences = []
                current_tokens = 0

            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining sentences
        if current_chunk_sentences:
            chunk_text = self._rejoin_sentences(current_chunk_sentences)
            chunks.append(
                Chunk(
                    id=f"{source_file}::chunk{chunk_idx}",
                    text=chunk_text,
                    source_file=source_file,
                    requirement_number=requirement_number,
                    section_heading=section_heading,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                )
            )

        # Verify round-trip: the chunks joined with "\n" must equal the original text.
        # Since sentence splitting may lose the exact separator, we need a different
        # approach that preserves the original text exactly.
        # Let's verify and if it doesn't match, fall back to line-based splitting.
        reconstructed = "\n".join(c.text for c in chunks)
        if reconstructed != text:
            # Fall back to line-based splitting which preserves text exactly
            chunks = self._chunk_segment_by_lines(
                text=text,
                source_file=source_file,
                requirement_number=requirement_number,
                section_heading=section_heading,
                page_number=page_number,
                start_chunk_index=start_chunk_index,
            )

        return chunks

    def _rejoin_sentences(self, sentences: list[str]) -> str:
        """Rejoin sentences with the separator that was removed during splitting.

        Since we split on the pattern `(?<=[.!?])\\s+(?=[A-Z])`, the separator
        between sentences is whitespace (typically a single space or newline).
        We rejoin with a single space to approximate the original.
        """
        return " ".join(sentences)

    def _chunk_segment_by_lines(
        self,
        text: str,
        source_file: str,
        requirement_number: str | None,
        section_heading: str | None,
        page_number: int,
        start_chunk_index: int,
    ) -> list[Chunk]:
        """Fall-back chunking that splits by lines to guarantee round-trip property.

        Splits on newline boundaries, grouping lines into chunks that respect
        max_chunk_tokens. Extends boundaries to include complete sentences
        when possible.

        The concatenation of chunk texts joined with "\\n" equals the original text.
        """
        lines = text.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        chunk_idx = start_chunk_index

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            # If adding this line would exceed the limit and we have content
            if (
                current_tokens + line_tokens > self.max_chunk_tokens
                and current_lines
            ):
                # Check if we're mid-sentence and should extend
                # A line is mid-sentence if it doesn't end with sentence-ending punctuation
                # and the next logical break hasn't been reached
                if self._is_mid_sentence(current_lines[-1]) and line_tokens + current_tokens <= self.max_chunk_tokens * 1.1:
                    # Extend to include this line (allow slight overflow for sentence completion)
                    current_lines.append(line)
                    current_tokens += line_tokens
                    continue

                # Flush current chunk
                chunk_text = "\n".join(current_lines)
                chunks.append(
                    Chunk(
                        id=f"{source_file}::chunk{chunk_idx}",
                        text=chunk_text,
                        source_file=source_file,
                        requirement_number=requirement_number,
                        section_heading=section_heading,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                current_lines = []
                current_tokens = 0

            current_lines.append(line)
            current_tokens += line_tokens

        # Flush remaining
        if current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(
                Chunk(
                    id=f"{source_file}::chunk{chunk_idx}",
                    text=chunk_text,
                    source_file=source_file,
                    requirement_number=requirement_number,
                    section_heading=section_heading,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                )
            )

        return chunks

    def _is_mid_sentence(self, line: str) -> bool:
        """Check if a line appears to end mid-sentence.

        A line is considered mid-sentence if it doesn't end with
        sentence-ending punctuation (.!?:) and is not empty.
        """
        stripped = line.rstrip()
        if not stripped:
            return False
        return stripped[-1] not in ".!?:"

    def _is_heading(
        self, text: str, font_size: float, is_bold: bool
    ) -> bool:
        """Determine if a line of text is likely a section heading.

        Uses a combination of font size, bold formatting, and pattern
        matching against known PCI DSS heading patterns.
        """
        # Match known section heading patterns
        if SECTION_HEADING_PATTERN.match(text):
            return True

        # Lines with larger font are often headings (threshold heuristic)
        if font_size >= 12.0 and is_bold:
            return True

        # Requirement number patterns are headings
        if REQUIREMENT_PATTERN.match(text):
            return True

        return False

    def _pages_to_placeholder_chunks(
        self, pages: list[PageContent], source_file: str
    ) -> list[Chunk]:
        """Convert pages to one-chunk-per-page as a placeholder.

        This is a temporary implementation until segment_into_chunks is
        completed in Task 2.2. It ensures parse_document returns valid
        chunks for downstream use.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in pages:
            if not page.text.strip():
                continue

            # Determine requirement number from page headings
            requirement_number = self._extract_requirement_number(
                page.headings
            )
            section_heading = page.headings[0] if page.headings else None

            chunks.append(
                Chunk(
                    id=f"{source_file}::page{page.page_number}::chunk{chunk_index}",
                    text=page.text,
                    source_file=source_file,
                    requirement_number=requirement_number,
                    section_heading=section_heading,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        return chunks

    def _extract_requirement_number(
        self, headings: list[str]
    ) -> str | None:
        """Extract the first PCI DSS requirement number from headings.

        Args:
            headings: List of detected heading strings.

        Returns:
            The requirement number string (e.g. '3.3') or None.
        """
        for heading in headings:
            match = REQUIREMENT_PATTERN.search(heading)
            if match:
                return match.group(1)
        return None
