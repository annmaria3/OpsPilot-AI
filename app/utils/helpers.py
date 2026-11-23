"""OpsPilot AI - Reusable Utility Helpers.

Provides helper functions for text sanitization, timestamp formatting,
incident identifier generation, and safe conversion of SQLite database
records into standard Python dictionaries.

Kept completely independent of Streamlit UI components and external C-libraries.
"""

import html
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def clean_text(text: Any) -> str:
    """Normalize and clean input strings.

    Strips leading/trailing whitespace, converts None to an empty string,
    decodes space entities (&#x20;, &#32;), removes unnecessary markdown
    escapes before digits, and collapses consecutive whitespace characters.
    """
    if text is None:
        return ""
    val = str(text).replace("&amp;#x20;", " ").replace("&#x20;", " ").replace("&#32;", " ")
    val = re.sub(r'(\d+)\\\.', r'\1.', val)
    val = re.sub(r'\\([*_{}\[\]()#+\-!~`>|])', r'\1', val)
    val = re.sub(r'\\+\s*(\r?\n|$)', r' ', val)
    # Strip and collapse internal whitespace
    return " ".join(val.split())


def format_incident_id(incident_id: int, prefix: str = "INC", padding: int = 4) -> str:
    """Format an integer database ID into a standardized enterprise identifier.

    Example:
        format_incident_id(1) -> "INC-0001"
        format_incident_id(42) -> "INC-0042"
    """
    if not isinstance(incident_id, int) or incident_id < 0:
        raise ValueError(f"Invalid incident_id for formatting: {incident_id}")
    return f"{prefix}-{incident_id:0{padding}d}"


def parse_incident_id(formatted_id: Union[str, int], prefix: str = "INC") -> Optional[int]:
    """Parse an enterprise identifier or string integer back into an integer ID.

    Example:
        parse_incident_id("INC-0042") -> 42
        parse_incident_id("42") -> 42
        parse_incident_id("invalid") -> None
    """
    if isinstance(formatted_id, int):
        return formatted_id if formatted_id > 0 else None

    clean_str = clean_text(formatted_id)
    if not clean_str:
        return None

    # Handle formatted "INC-0042"
    upper_str = clean_str.upper()
    prefix_tag = f"{prefix.upper()}-"
    if upper_str.startswith(prefix_tag):
        number_part = upper_str[len(prefix_tag):]
        if number_part.isdigit():
            return int(number_part)

    # Handle raw numeric string "42"
    if clean_str.isdigit():
        return int(clean_str)

    return None


def format_timestamp(
    ts: Optional[Union[str, datetime]],
    output_format: str = "%b %d, %Y • %I:%M %p"
) -> str:
    """Format an ISO or SQLite timestamp string into a readable date string.

    Example:
        format_timestamp("2026-11-21 10:30:00") -> "Nov 21, 2026 • 10:30 AM"
    """
    if not ts:
        return "N/A"

    if isinstance(ts, datetime):
        return ts.strftime(output_format)

    str_val = clean_text(str(ts))
    # Common SQLite / ISO date formats
    known_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
    ]

    for fmt in known_formats:
        try:
            dt = datetime.strptime(str_val, fmt)
            return dt.strftime(output_format)
        except ValueError:
            continue

    # Fallback to returning original string if format is unparseable
    return str_val


def get_utc_now() -> str:
    """Return the current timestamp formatted as a standard SQLite timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Safely convert a sqlite3.Row object into a standard Python dictionary.

    Returns None if the input row is None.
    """
    if row is None:
        return None
    return dict(row)


def rows_to_dict_list(rows: Optional[List[sqlite3.Row]]) -> List[Dict[str, Any]]:
    """Safely convert a list of sqlite3.Row objects into a list of dictionaries.

    Returns an empty list if rows is None or empty.
    """
    if not rows:
        return []
    return [dict(r) for r in rows if r is not None]


def truncate_text(text: Optional[str], max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum character length with an ellipsis suffix."""
    cleaned = clean_text(text)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + suffix


def safe_html_text(val: Any, preserve_newlines: bool = False) -> str:
    """Safely escapes text for HTML interpolation to prevent XSS / HTML injection.

    Optionally preserves user newlines by safely converting them to <br>.
    Collapses multiple consecutive newlines to avoid excessive blank lines (<br><br>).
    Removes extraneous markdown backslash escapes before digits and punctuation (e.g. '2\\.' -> '2.')
    while strictly preserving legitimate backslashes (e.g. file paths 'C:\\path' or regex '\\d+').
    Strips trailing CommonMark hard line break backslashes at line endings (e.g. 'line\\').
    Decodes literal space entities (&#x20;, &#32;) so text renders cleanly in HTML.
    """
    if val is None:
        return ""
    text = str(val)
    # Decode literal space entities
    text = text.replace("&amp;#x20;", " ").replace("&#x20;", " ").replace("&#32;", " ")
    # Strip unnecessary markdown escape backslashes before digits followed by dot (e.g. 2\. -> 2.)
    text = re.sub(r'(\d+)\\\.', r'\1.', text)
    # Strip unnecessary markdown escape backslashes before markdown punctuation symbols
    text = re.sub(r'\\([*_{}\[\]()#+\-!~`>|])', r'\1', text)
    # Strip trailing backslashes before newlines or end of string
    text = re.sub(r'\\+\s*(\r?\n|$)', r'\1', text)

    if preserve_newlines:
        # Normalize newlines and collapse multiple blank lines to a single newline
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        collapsed = re.sub(r'\n\s*\n+', '\n', normalized)
        clean_lines = [
            re.sub(r'\\+\s*$', '', line.strip()).strip()
            for line in collapsed.splitlines()
            if line.strip()
        ]
        if not clean_lines:
            return ""
        escaped_lines = [html.escape(line) for line in clean_lines]
        return "<br>".join(escaped_lines)

    clean_val = re.sub(r'\\+\s*$', '', text.strip()).strip()
    return html.escape(clean_val)


def format_remediation_html(text: Any) -> str:
    """Format operator or AI remediation procedures into clean, compact HTML steps.

    Security & Formatting Guarantees:
    1. Strictly neutralizes XSS payloads via html.escape().
    2. Removes unnecessary markdown escape backslashes before numerals and punctuation
       (e.g., '1\\.' -> '1.') while strictly preserving legitimate backslashes (e.g. file paths
       like 'C:\\path' or regex patterns like '\\d+').
    3. Strips literal CommonMark hard-line-break backslashes at line endings (e.g. 'step\\')
       so they are cleanly represented by HTML line breaks instead of raw backslash characters.
    4. Normalizes consecutive newlines so steps appear compact without excessive blank lines
       (no double <br><br>).
    5. Decodes literal space entities (&#x20;, &#32;).
    6. Renders multi-step numbered lines as cleanly formatted, readable step rows.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    # Decode literal space entities
    cleaned = raw.replace("&amp;#x20;", " ").replace("&#x20;", " ").replace("&#32;", " ")
    # Normalize line endings
    normalized = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Strip markdown escape backslashes before numbers and punctuation
    normalized = re.sub(r'(\d+)\\\.', r'\1.', normalized)
    normalized = re.sub(r'\\([*_{}\[\]()#+\-!~`>|])', r'\1', normalized)
    # Collapse multiple consecutive empty lines to a single newline
    normalized = re.sub(r'\n\s*\n+', '\n', normalized)

    # Clean each line: strip whitespace and trailing line-ending backslashes
    clean_lines = []
    for line in normalized.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # Strip trailing backslashes at line end (CommonMark hard line break artifacts)
        line_clean = re.sub(r'\\+\s*$', '', line_clean).strip()
        if not line_clean:
            continue
        clean_lines.append(html.escape(line_clean))

    if not clean_lines:
        return ""

    # Join with a single <br> for compact, readable stacking
    return "<br>".join(clean_lines)


def format_fallback_guidance(
    text: Any,
    citations: Optional[List[Any]] = None,
    total_docs: int = 0
) -> str:
    """Format offline diagnostic fallback guidance truthfully based on knowledge base state.

    If 0 citations or 0 documents are available, ensures operators are not falsely instructed
    to 'review retrieved enterprise runbook chunks below'. Instead, clearly and truthfully states
    that no runbook chunks were retrieved from the Knowledge Base (0 indexed) and directs operators
    to author an operator remediation procedure in the Governance Console.
    """
    if text is None:
        return ""
    val = str(text)
    has_citations = bool(citations and len(citations) > 0)
    if not has_citations and (total_docs == 0 or not citations):
        false_reference = "Please review the retrieved enterprise runbook chunks below to proceed with manual investigation."
        truthful_replacement = "No enterprise runbooks or documentation chunks were retrieved from the Knowledge Base (0 indexed). Author an operator remediation procedure in the Governance Console based on verified operational procedures."
        if false_reference in val:
            val = val.replace(false_reference, truthful_replacement)
        elif "retrieved enterprise runbook chunks below" in val:
            val = re.sub(
                r'Please review the retrieved enterprise runbook chunks below to proceed with manual investigation\.?',
                truthful_replacement,
                val,
                flags=re.IGNORECASE
            )
            val = re.sub(
                r'review the retrieved enterprise runbook chunks below',
                "no enterprise runbooks were retrieved from the Knowledge Base (0 indexed)",
                val,
                flags=re.IGNORECASE
            )
    return val


def clean_html_block(html_str: Any) -> str:
    """Strip leading and trailing whitespace from each line in an HTML block.

    Prevents CommonMark from misinterpreting indented lines (4+ spaces)
    as preformatted code blocks (<pre><code>) inside Streamlit markdown.
    Preserves inner content and tag structure while ensuring every line
    begins at column 0. Cleans up literal space entities and normalizes
    inline label spacing to eliminate trailing space entity leaks.
    """
    if html_str is None:
        return ""
    text = str(html_str)
    # Decode space entities and normalize bullet entities
    text = text.replace("&amp;#x20;", " ").replace("&#x20;", " ").replace("&#32;", " ")
    text = text.replace("&bull;", "•")
    text = text.strip()
    if not text:
        return ""

    # Normalize inline label whitespace before child tags (prevents rehype entity leaks)
    text = re.sub(
        r'<span>([^<:]+):\s*<strong([^>]*)>',
        r'<span style="margin-right: 4px;">\1:</span><strong\2>',
        text
    )

    clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(clean_lines)


def format_file_size(size_bytes: Optional[Union[int, float]]) -> str:
    """Format raw byte counts into human-readable strings.

    Example:
        format_file_size(512) -> "512 B"
        format_file_size(24576) -> "24.0 KB"
        format_file_size(1572864) -> "1.5 MB"
    """
    if size_bytes is None:
        return "0 B"
    try:
        bytes_val = float(size_bytes)
        if bytes_val < 0:
            return "0 B"
    except (TypeError, ValueError):
        return "0 B"

    if bytes_val < 1024:
        return f"{int(bytes_val)} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"
