"""OpsPilot AI - Retrieval-Augmented Generation (RAG) & Document Ingestion Service.

Provides secure document validation, path sanitization, text extraction (PDF, TXT, MD),
paragraph/sentence-aware text chunking, and atomic persistence with compensating
filesystem cleanup.

Kept completely independent of Streamlit UI components and external cloud SDKs.
"""

import hashlib
import io
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pypdf

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_LIMIT,
    KNOWLEDGE_BASE_DIR,
    MAX_DOCUMENT_SIZE_BYTES,
    MAX_RETRIEVAL_LIMIT,
    MIN_SEARCH_KEYWORD_LENGTH,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from app.database import db
from app.services import incident_service
from app.utils.helpers import clean_text


# ==============================================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# ==============================================================================

class IngestionError(Exception):
    """Base exception for all document ingestion and extraction failures."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file extension is not supported."""
    pass


class FileSizeLimitError(IngestionError):
    """Raised when an uploaded file exceeds the maximum allowed size (10 MB)."""
    pass


class EmptyFileError(IngestionError):
    """Raised when an uploaded file contains zero bytes."""
    pass


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical SHA-256 content hash already exists."""
    pass


class InvalidFilenameError(IngestionError):
    """Raised when an uploaded filename is empty, malformed, or attempts path traversal."""
    pass


class TextExtractionError(IngestionError):
    """Raised when text extraction or decoding fails."""
    pass


class EmptyExtractedContentError(TextExtractionError):
    """Raised when an extracted document contains no readable text (e.g. blank or scanned PDF)."""
    pass


class PersistenceError(IngestionError):
    """Raised when filesystem write or database persistence fails."""
    pass


# ==============================================================================
# 2. VALIDATION & FILENAME SANITIZATION
# ==============================================================================

def compute_content_hash(file_bytes: bytes) -> str:
    """Compute the SHA-256 hex digest of the raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_file(file_name: str, file_bytes: bytes) -> str:
    """Validate file name, bytes, extension, and size limits.

    Returns:
        The normalized lowercase file extension (e.g. '.pdf').

    Raises:
        InvalidFilenameError: If file_name is missing or empty.
        EmptyFileError: If file_bytes is empty.
        FileSizeLimitError: If file_bytes exceeds MAX_DOCUMENT_SIZE_BYTES.
        UnsupportedFileTypeError: If file extension is not in SUPPORTED_DOCUMENT_EXTENSIONS.
    """
    if not file_name or not str(file_name).strip():
        raise InvalidFilenameError("File name cannot be empty.")

    raw_clean = os.path.basename(str(file_name).replace("\\", "/")).strip()
    if not raw_clean or not raw_clean.strip('.'):
        raise InvalidFilenameError(f"Invalid filename: '{file_name}'.")

    p = Path(raw_clean)
    if not p.stem.strip('.'):
        raise InvalidFilenameError(f"Filename '{file_name}' lacks a valid base name.")

    if file_bytes is None or len(file_bytes) == 0:
        raise EmptyFileError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        max_mb = MAX_DOCUMENT_SIZE_BYTES / (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        raise FileSizeLimitError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    ext = p.suffix.lower()
    if not ext:
        raise InvalidFilenameError(f"Filename '{file_name}' lacks a file extension.")
    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}."
        )

    return ext


def sanitize_filename(file_name: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and filesystem errors.

    Strips directory paths, null bytes, and dangerous characters.
    """
    if not file_name or not str(file_name).strip():
        raise InvalidFilenameError("File name cannot be empty.")

    # Strip directory components (handles Windows and POSIX path separators)
    clean_name = os.path.basename(file_name.replace("\\", "/"))

    # Remove null bytes and control characters
    clean_name = re.sub(r'[\x00-\x1f\x7f]', '', clean_name)

    # Remove filesystem-illegal characters: < > : " / \ | ? *
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)

    # Strip leading/trailing dots and whitespace
    clean_name = clean_name.strip('. ')

    if not clean_name:
        raise InvalidFilenameError("Sanitized filename resulted in an empty string.")

    # Ensure it retains a valid extension
    suffix = Path(clean_name).suffix.lower()
    if not suffix or suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise InvalidFilenameError(f"Sanitized filename '{clean_name}' does not have a supported extension.")

    return clean_name


def resolve_unique_destination(
    storage_dir: Path,
    safe_filename: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Tuple[str, Path]:
    """Resolve a safe destination path inside storage_dir, avoiding overwrites.

    Ensures the path stays strictly within storage_dir (traversal defense).
    If a file or database record with the same name exists, generates a
    deterministic collision-safe suffix (`{stem}_{counter}{suffix}`).

    Returns:
        Tuple of (unique_document_name, unique_destination_path).
    """
    target_dir = storage_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dest_path = (target_dir / safe_filename).resolve()

    # Directory traversal assertion
    try:
        dest_path.relative_to(target_dir)
    except ValueError:
        raise InvalidFilenameError(f"Filename '{safe_filename}' attempts path traversal.")

    candidate_name = safe_filename
    counter = 1
    stem = Path(safe_filename).stem
    suffix = Path(safe_filename).suffix

    # Loop until candidate_name does not exist on disk AND does not exist in SQLite
    while True:
        candidate_path = (target_dir / candidate_name).resolve()
        file_exists = candidate_path.exists()

        doc_exists_in_db = False
        try:
            conn = db.get_connection(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM knowledge_documents WHERE document_name = ? LIMIT 1;",
                    (candidate_name,)
                )
                doc_exists_in_db = cursor.fetchone() is not None
            finally:
                conn.close()
        except sqlite3.Error:
            doc_exists_in_db = False

        if not file_exists and not doc_exists_in_db:
            return candidate_name, candidate_path

        candidate_name = f"{stem}_{counter}{suffix}"
        counter += 1


# ==============================================================================
# 3. TEXT EXTRACTION & CLEANING
# ==============================================================================

def clean_extracted_text(text: str) -> str:
    """Clean extracted text conservatively while preserving code, logs, and layout.

    - Removes invalid ASCII control characters (preserving tab and newline).
    - Normalizes line endings (\r\n and \r to \n).
    - Collapses excessive blank lines (3+ to 2).
    - Trims trailing whitespace on each line.
    """
    if not text:
        return ""

    # Remove non-printable control characters except \t, \n
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize line breaks
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')

    # Strip trailing whitespace on lines
    cleaned = "\n".join(line.rstrip() for line in cleaned.split('\n'))

    # Collapse 3 or more consecutive newlines into 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from PDF page-by-page using pypdf.

    Returns:
        List of tuples: (page_number, page_text) with 1-based page numbers.

    Raises:
        EmptyExtractedContentError: If the PDF contains no extractable text.
        TextExtractionError: On PDF parsing or decompression failure.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_output = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_extracted_text(raw_text)
            if cleaned:
                pages_output.append((page_num, cleaned))

        if not pages_output:
            raise EmptyExtractedContentError(
                "PDF contains no extractable text (document is blank or scanned image without OCR)."
            )

        return pages_output
    except EmptyExtractedContentError:
        raise
    except Exception as e:
        raise TextExtractionError(f"Failed to extract text from PDF: {e}") from e


def extract_text_from_text(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Decode text from TXT or Markdown files handling UTF-8 with BOM and safe fallbacks.

    Returns:
        List of tuples: [(1, cleaned_text)].

    Raises:
        EmptyExtractedContentError: If decoded content is empty.
        TextExtractionError: If content cannot be safely decoded as text.
    """
    decoded = None
    # 1. Try UTF-8 with BOM handling
    try:
        decoded = file_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass

    # 2. Try Windows-1252 / ISO-8859-1 fallback
    if decoded is None:
        try:
            candidate = file_bytes.decode('cp1252')
            # Check for excessive null bytes or unprintable characters indicating binary file
            if '\x00' in candidate:
                raise TextExtractionError("File contains binary data and cannot be safely decoded as text.")
            decoded = candidate
        except (UnicodeDecodeError, TextExtractionError):
            pass

    if decoded is None:
        raise TextExtractionError("Unable to decode text file. File encoding must be valid UTF-8 or CP1252.")

    cleaned = clean_extracted_text(decoded)
    if not cleaned:
        raise EmptyExtractedContentError("Document contains no readable text content.")

    return [(1, cleaned)]


def extract_document_text(file_name: str, file_bytes: bytes) -> Tuple[str, List[Tuple[int, str]]]:
    """Route document extraction based on extension.

    Returns:
        Tuple of (document_type, list_of_pages) where list_of_pages contains (page_num, page_text).
    """
    ext = Path(file_name).suffix.lower()

    if ext == ".pdf":
        doc_type = "PDF"
        pages = extract_text_from_pdf(file_bytes)
    elif ext == ".md":
        doc_type = "Markdown"
        pages = extract_text_from_text(file_bytes)
    elif ext == ".txt":
        doc_type = "TXT"
        pages = extract_text_from_text(file_bytes)
    else:
        raise UnsupportedFileTypeError(f"Unsupported format '{ext}'.")

    return doc_type, pages


# ==============================================================================
# 4. CHUNKING ENGINE
# ==============================================================================

def _split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split a single text string into overlapping chunks preferring paragraph and sentence boundaries.

    Guarantees:
    - Never loops infinitely (strictly advances start index).
    - Never exceeds chunk_size characters.
    - Preserves all source text without silent drops.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end == text_len:
            chunk = text[start:target_end].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Search backwards from target_end for natural boundary
        # Candidate break points: paragraph break (\n\n), line break (\n), sentence end, space
        search_window = text[start:target_end]
        min_break_pos = max(1, chunk_overlap + 1) if chunk_overlap < chunk_size else 1

        break_pos = None
        is_paragraph_break = False

        # 1. Paragraph break (\n\n)
        p_idx = search_window.rfind('\n\n')
        if p_idx >= min_break_pos:
            break_pos = p_idx + 2
            is_paragraph_break = True

        # 2. Line break (\n)
        if break_pos is None:
            n_idx = search_window.rfind('\n')
            if n_idx >= min_break_pos:
                break_pos = n_idx + 1

        # 3. Sentence boundary (. / ! / ?)
        if break_pos is None:
            sentence_matches = list(re.finditer(r'[.!?]\s+', search_window))
            if sentence_matches:
                last_m = sentence_matches[-1]
                if last_m.end() >= min_break_pos:
                    break_pos = last_m.end()

        # 4. Word boundary (space)
        if break_pos is None:
            s_idx = search_window.rfind(' ')
            if s_idx >= min_break_pos:
                break_pos = s_idx + 1

        # 5. Hard break fallback
        if break_pos is None:
            break_pos = target_end - start

        actual_end = start + break_pos
        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next start point with overlap
        if is_paragraph_break:
            # Paragraph breaks are natural semantic boundaries; next chunk starts at next paragraph
            next_start = actual_end
        else:
            raw_start = actual_end - chunk_overlap
            if raw_start < actual_end:
                # Snap forward to next whitespace so we don't start with a severed word fragment
                space_idx = text.find(' ', raw_start, actual_end)
                if space_idx != -1 and space_idx > start:
                    next_start = space_idx + 1
                else:
                    next_start = raw_start
            else:
                next_start = actual_end

        if next_start <= start:
            next_start = actual_end

        start = next_start

    return chunks


def chunk_document_content(
    pages: List[Tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Chunk document text page-by-page while preserving page numbers and assigning sequential indexes.

    Returns:
        List of chunk dictionaries:
        [{"page_number": int, "chunk_index": int, "content": str}, ...]
    """
    safe_size = max(100, int(chunk_size))
    safe_overlap = max(0, min(int(chunk_overlap), safe_size // 2))

    all_chunks = []
    global_index = 0

    for page_num, page_text in pages:
        page_chunks = _split_text_into_chunks(page_text, safe_size, safe_overlap)
        for chunk_text in page_chunks:
            all_chunks.append({
                "page_number": page_num,
                "chunk_index": global_index,
                "content": chunk_text,
            })
            global_index += 1

    return all_chunks


# ==============================================================================
# 5. PUBLIC INGESTION API
# ==============================================================================

def ingest_document(
    file_name: str,
    file_bytes: bytes,
    storage_dir: Optional[Union[Path, str]] = None,
    db_path: Optional[Union[Path, str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """Ingest, extract, chunk, and persist a knowledge document atomically.

    Guarantees:
    1. Validation of format, size, and content before writing.
    2. SHA-256 deduplication before disk or database writes.
    3. Path sanitization and directory traversal defense.
    4. Collision-safe unique naming without overwriting existing files.
    5. Atomic database persistence of document metadata and chunks.
    6. Compensating filesystem rollback: unlinks new file if database persistence fails.
    7. Sets 'Indexed' status only after successful commit.

    Returns:
        Structured document record dictionary:
        {
            "document_id": int,
            "document_name": str,
            "file_path": str,
            "file_size_bytes": int,
            "content_hash": str,
            "document_type": str,
            "chunk_count": int,
            "status": "Indexed",
            "uploaded_at": str,
        }

    Raises:
        InvalidFilenameError: If filename is invalid or attempts traversal.
        EmptyFileError: If file is 0 bytes.
        FileSizeLimitError: If file exceeds 10 MB limit.
        UnsupportedFileTypeError: If format is not .pdf, .txt, .md.
        DuplicateDocumentError: If identical content hash already exists.
        EmptyExtractedContentError: If document contains no readable text.
        TextExtractionError: If text extraction fails.
        PersistenceError: If filesystem write or database commit fails.
    """
    # 1. Validate file extension, size, and emptiness
    validate_file(file_name, file_bytes)

    # 2. Compute SHA-256 content hash and check for duplicates
    content_hash = compute_content_hash(file_bytes)
    existing_doc = db.get_knowledge_document_by_hash(content_hash, db_path=db_path)
    if existing_doc is not None:
        raise DuplicateDocumentError(
            f"A document with identical content already exists: '{existing_doc['document_name']}' "
            f"(Document ID: {existing_doc['id']})."
        )

    # 3. Extract and clean text based on format
    safe_filename = sanitize_filename(file_name)
    doc_type, pages = extract_document_text(safe_filename, file_bytes)

    # 4. Chunk document content
    chunks = chunk_document_content(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise EmptyExtractedContentError(f"Document '{safe_filename}' produced 0 chunks after text extraction.")

    # 5. Resolve storage directory and collision-safe destination path
    target_storage_dir = Path(storage_dir) if storage_dir is not None else KNOWLEDGE_BASE_DIR
    target_storage_dir.mkdir(parents=True, exist_ok=True)

    unique_doc_name, dest_file_path = resolve_unique_destination(
        target_storage_dir,
        safe_filename,
        db_path=db_path,
    )

    # 6. Write file to disk with compensating cleanup on database failure
    written_file_path = None
    try:
        with open(dest_file_path, "wb") as f:
            f.write(file_bytes)
        written_file_path = dest_file_path

        # 7. Atomically persist document and chunks in SQLite
        doc_record = db.create_knowledge_document_with_chunks(
            document_name=unique_doc_name,
            file_path=str(dest_file_path),
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            document_type=doc_type,
            chunks=chunks,
            status="Indexed",
            db_path=db_path,
        )

        return doc_record

    except Exception as err:
        # Compensating filesystem cleanup for ordinary persistence failures
        if written_file_path is not None and written_file_path.exists():
            try:
                written_file_path.unlink()
            except OSError:
                pass

        if isinstance(err, IngestionError):
            raise
        raise PersistenceError(f"Failed to persist document '{unique_doc_name}': {err}") from err


# ==============================================================================
# 6. KNOWLEDGE RETRIEVAL & QUERY FORMULATION ENGINE
# ==============================================================================

# Standard English stop words and conversational ticket noise to filter during keyword extraction
STOP_WORDS = {
    # Articles, prepositions, conjunctions, pronouns, auxiliary verbs
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
    # Conversational ticket filler and noise
    "please", "help", "issue", "problem", "incident", "error", "facing", "seeing", "team", "hi",
    "hello", "urgent", "urgently", "thanks", "regards", "noticed", "occurred", "happening", "getting",
    "reported", "check", "need", "investigate",
}


def format_source_citation(
    document_name: str,
    page_number: int,
    chunk_index: int,
    compact: bool = False,
) -> str:
    """Format uniform, human-readable source citations for knowledge chunks.

    Args:
        document_name: Name of the source document (e.g. 'PostgreSQL_Runbook.pdf').
        page_number: 1-based page number within the document.
        chunk_index: 0-based sequential chunk index within the document.
        compact: If True, returns a compact bracketed citation (e.g. '[Doc, p.1 #2]');
                 otherwise returns a descriptive citation (e.g. 'Runbook: Doc (Page 1, Chunk 2)').

    Returns:
        Formatted citation string.
    """
    clean_name = str(document_name).strip() if document_name else "Unknown Document"
    try:
        page_num = int(page_number) if not isinstance(page_number, bool) and int(page_number) >= 1 else 1
    except (TypeError, ValueError):
        page_num = 1

    try:
        chunk_idx = int(chunk_index) if not isinstance(chunk_index, bool) and int(chunk_index) >= 0 else 0
    except (TypeError, ValueError):
        chunk_idx = 0

    if compact:
        return f"[{clean_name}, p.{page_num} #{chunk_idx}]"
    return f"Runbook: {clean_name} (Page {page_num}, Chunk {chunk_idx})"


def extract_search_keywords(
    title: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
) -> List[str]:
    """Extract high-signal technical keywords from incident attributes for FTS retrieval.

    Prioritizes:
    1. Incident title terms (highest signal).
    2. Incident category domain taxonomy terms.
    3. Technical signals from description (HTTP status codes, exceptions, acronyms, identifiers).
    4. General non-stopword tokens.

    Deduplicates tokens case-insensitively while preserving priority order and capping
    at 15 keywords to optimize SQLite FTS5 matching.
    """
    candidate_tokens: List[str] = []

    # 1. Title tokens (highest signal)
    if title and isinstance(title, str):
        title_tokens = re.findall(r'\b\w+\b', title)
        candidate_tokens.extend(title_tokens)

    # 2. Category taxonomy tokens
    if category and isinstance(category, str):
        cat_tokens = re.findall(r'\b\w+\b', category)
        candidate_tokens.extend(cat_tokens)

    # 3. Description technical terms
    if description and isinstance(description, str):
        # Extract high-signal patterns first:
        # - HTTP status codes (e.g. 500, 502, 504)
        status_codes = re.findall(r'\b[1-5][0-9]{2}\b', description)
        candidate_tokens.extend(status_codes)

        # - CamelCase / PascalCase identifiers (e.g. NullPointerException)
        camel_cases = re.findall(r'\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b', description)
        candidate_tokens.extend(camel_cases)

        # - Uppercase acronyms (e.g. OOM, K8S, BGP, CPU, RAM)
        acronyms = re.findall(r'\b[A-Z]{2,10}\b', description)
        candidate_tokens.extend(acronyms)

        # - Technical identifiers with underscores (e.g. max_connections, pg_stat_activity)
        underscore_ids = re.findall(r'\b[a-zA-Z0-9]+_[a-zA-Z0-9_]+\b', description)
        candidate_tokens.extend(underscore_ids)

        # - General word tokens from description
        desc_tokens = re.findall(r'\b\w+\b', description)
        candidate_tokens.extend(desc_tokens)

    # Filter stopwords, short tokens, pure small integers, and deduplicate preserving order
    keywords: List[str] = []
    seen_lower = set()

    for token in candidate_tokens:
        t_clean = token.strip('_')
        t_lower = t_clean.lower()

        # Length check
        if len(t_clean) < MIN_SEARCH_KEYWORD_LENGTH:
            continue

        # Filter pure 1-digit or 2-digit numbers (preserve 3+ digit numbers like error 500, port 8080)
        if t_lower.isdigit() and len(t_clean) < 3:
            continue

        # Stop word filtering
        if t_lower in STOP_WORDS:
            continue

        # Deduplication
        if t_lower not in seen_lower:
            seen_lower.add(t_lower)
            keywords.append(t_clean)

    # Cap at top 15 terms to fit within FTS5's 20-word matching window
    return keywords[:15]


def build_incident_search_query(
    title: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Build a sanitized, high-signal full-text search query string for an incident."""
    keywords = extract_search_keywords(title=title, description=description, category=category)
    return " ".join(keywords)


def retrieve_knowledge_chunks(
    query: str,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top relevant knowledge chunks using SQLite FTS5 BM25 ranking.

    Reuses db.search_knowledge_chunks_fts() directly without duplicating SQL logic.
    Enriches results with 1-based ranks and standardized source citations.

    Args:
        query: Search query text.
        limit: Maximum number of chunks to return (clamped between 1 and MAX_RETRIEVAL_LIMIT).
               Explicitly prevents boolean types from being treated as integers.
        db_path: Optional path to SQLite database.

    Returns:
        List of attributed knowledge chunk evidence packets.
    """
    # Safely validate limit: prevent bool from acting as int, clamp between 1 and MAX_RETRIEVAL_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        safe_limit = DEFAULT_RETRIEVAL_LIMIT
    else:
        try:
            safe_limit = max(1, min(int(limit), MAX_RETRIEVAL_LIMIT))
        except (TypeError, ValueError, OverflowError):
            safe_limit = DEFAULT_RETRIEVAL_LIMIT

    # Handle empty, whitespace-only, or non-string queries
    if not query or not isinstance(query, str) or not query.strip():
        return []

    # Execute FTS5 BM25 search via existing database layer
    raw_results = db.search_knowledge_chunks_fts(
        query=query.strip(),
        limit=safe_limit,
        db_path=db_path,
    )

    # Assemble structured evidence packets with complete source attribution
    packets = []
    for idx, r in enumerate(raw_results):
        doc_name = r.get("document_name") or "Unknown Document"
        page_num = r.get("page_number", 1)
        chunk_idx = r.get("chunk_index", 0)
        bm25_val = float(r.get("bm25_score", 0.0))
        rel_val = float(r.get("relevance_score", round(abs(bm25_val), 4)))

        packet = {
            "chunk_id": r.get("chunk_id"),
            "document_id": r.get("document_id"),
            "document_name": doc_name,
            "page_number": page_num,
            "chunk_index": chunk_idx,
            "content": r.get("content", ""),
            "bm25_score": bm25_val,
            "relevance_score": rel_val,
            "rank": idx + 1,
            "citation": format_source_citation(doc_name, page_num, chunk_idx, compact=True),
            "citation_full": format_source_citation(doc_name, page_num, chunk_idx, compact=False),
        }
        packets.append(packet)

    return packets


def retrieve_knowledge_for_incident(
    incident: Union[int, Dict[str, Any]],
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Retrieve top relevant knowledge chunks for an incident by ID or incident dictionary.

    Extracts high-signal keywords from title, description, and category, executes FTS5
    retrieval, and returns a comprehensive evidence bundle with full source provenance.

    Args:
        incident: Either an integer incident_id or an incident dictionary.
        limit: Maximum number of chunks to retrieve.
        db_path: Optional path to SQLite database.

    Returns:
        Structured retrieval response dictionary:
        {
            "incident_id": Optional[int],
            "incident_title": str,
            "search_query": str,
            "keywords_used": List[str],
            "total_chunks_retrieved": int,
            "chunks": List[Dict[str, Any]],
        }
    """
    incident_data: Optional[Dict[str, Any]] = None
    target_id: Optional[int] = None

    # 1. Resolve incident data
    if isinstance(incident, int) and not isinstance(incident, bool):
        target_id = incident
        try:
            incident_data = incident_service.get_incident_by_id(incident, db_path=db_path)
        except (ValueError, TypeError):
            incident_data = None
    elif isinstance(incident, dict):
        incident_data = incident
        raw_id = incident.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
            target_id = raw_id
    else:
        incident_data = None

    # Handle missing or invalid incident gracefully
    if not incident_data:
        return {
            "incident_id": target_id,
            "incident_title": "",
            "search_query": "",
            "keywords_used": [],
            "total_chunks_retrieved": 0,
            "chunks": [],
        }

    # 2. Extract fields with safe fallbacks
    title = str(incident_data.get("title") or "").strip()
    description = str(incident_data.get("description") or "").strip()
    category = str(incident_data.get("category") or "").strip()

    # 3. Formulate search query
    keywords = extract_search_keywords(title=title, description=description, category=category)
    search_query = " ".join(keywords)

    # 4. Execute retrieval
    chunks = retrieve_knowledge_chunks(query=search_query, limit=limit, db_path=db_path)

    return {
        "incident_id": target_id,
        "incident_title": title,
        "search_query": search_query,
        "keywords_used": keywords,
        "total_chunks_retrieved": len(chunks),
        "chunks": chunks,
    }


# ==============================================================================
# 7. DOCUMENT MANAGEMENT & DELETION SERVICE APIS
# ==============================================================================

def list_documents(
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all indexed knowledge documents as standard dictionaries, ordered newest first."""
    rows = db.get_all_knowledge_documents(db_path=db_path)
    return [dict(r) for r in rows]


def get_document_by_id(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single document record by its integer ID."""
    if not isinstance(document_id, int) or document_id <= 0:
        return None
    row = db.get_knowledge_document_by_id(document_id, db_path=db_path)
    return dict(row) if row is not None else None


def get_document_chunks(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all chunks belonging to a document ordered by page and chunk index."""
    if not isinstance(document_id, int) or document_id <= 0:
        return []
    rows = db.get_chunks_for_document(document_id, db_path=db_path)
    return [dict(r) for r in rows]


def delete_document(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Safely delete a document, its physical storage file, chunks, and FTS5 index entries.

    Guarantees:
    1. Validates document existence.
    2. Retrieves document record to obtain file_path before deletion.
    3. Deletes database record via db.delete_knowledge_document(), triggering
       foreign-key cascade chunk deletion and FTS5 synchronization triggers.
    4. Safely removes the physical file from disk if present. If disk unlinking
       encounters an OS error, it is safely handled without leaving an inconsistent database state.
    5. Returns True if successfully deleted, False if document was not found or invalid ID.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return False

    doc = db.get_knowledge_document_by_id(document_id, db_path=db_path)
    if doc is None:
        return False

    file_path_str = doc["file_path"] if "file_path" in doc.keys() else None

    # Delete from database; triggers CASCADE chunk deletion and FTS cleanup
    db_deleted = db.delete_knowledge_document(document_id, db_path=db_path)
    if not db_deleted:
        return False

    # Clean up physical file on disk if it exists
    if file_path_str:
        try:
            p = Path(file_path_str)
            if p.exists() and p.is_file():
                p.unlink()
        except OSError:
            pass

    return True
