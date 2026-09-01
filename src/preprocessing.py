"""
Text preprocessing utilities for ticket clustering.
Cleans and normalizes ticket text for better clustering results.
Each cleaning step can be individually toggled via a settings dict.
"""

import re

from src.config import DEFAULTS

# Default preprocessing settings (mirrors config.py DEFAULTS["preprocessing"])
_DEFAULT_SETTINGS = DEFAULTS["preprocessing"]

# Pre-compiled regex patterns for performance
_RE_EMAIL = re.compile(r'\S+@\S+')
_RE_URL = re.compile(r'http\S+|www\.\S+')
_RE_TICKET_ID = re.compile(r'(inc|sr|req|chg|prb|task|tkt|case)[-_]?\d+', re.IGNORECASE)
_RE_DATE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
_RE_TIME = re.compile(r'\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?', re.IGNORECASE)
_RE_PHONE = re.compile(r'\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}')
_RE_IP = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
_RE_WIN_PATH = re.compile(r'[a-zA-Z]:\\[\w\\]+')
_RE_UNIX_PATH = re.compile(r'/[\w/]+')
_RE_SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\s]')
_RE_WHITESPACE = re.compile(r'\s+')

# ServiceNow journal metadata (work notes / additional comments). Every entry is
# prefixed "<timestamp> - <author> (Work notes)"; left in, clustering resolution
# text groups by author name and boilerplate instead of by the actual fix.
_RE_JOURNAL_HEADER = re.compile(
    r'\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}\s*-\s*[^\n(]*\((?:work notes|additional comments)\)',
    re.IGNORECASE,
)
_RE_JOURNAL_MARKER = re.compile(r'\((?:work notes|additional comments)\)', re.IGNORECASE)
_RE_CR_ARTIFACT = re.compile(r'_x000d_|\r', re.IGNORECASE)


def clean_resolution_text(text):
    """Strip ServiceNow journal metadata from resolution / work-notes text.

    Removes "<timestamp> - <author> (Work notes)" entry headers, stray
    "(Work notes)"/"(Additional comments)" markers and _x000D_ carriage-return
    artifacts so the actual resolution content drives clustering. The result
    still needs normal preprocessing (``preprocess_documents``) applied on top.
    """
    if not isinstance(text, str):
        return ""
    text = _RE_CR_ARTIFACT.sub(' ', text)
    text = _RE_JOURNAL_HEADER.sub(' ', text)
    text = _RE_JOURNAL_MARKER.sub(' ', text)
    return _RE_WHITESPACE.sub(' ', text).strip()


def clean_ticket_text(text, settings=None):
    """Clean and normalize ticket text for better clustering.

    Args:
        text: Raw ticket text.
        settings: Dict of preprocessing toggles. Keys match
                  config DEFAULTS["preprocessing"]. ``None`` uses defaults.
    """
    if not isinstance(text, str):
        return ""

    s = settings or _DEFAULT_SETTINGS

    # Convert to lowercase
    text = text.lower()

    if s.get("remove_emails", True):
        text = _RE_EMAIL.sub('', text)

    if s.get("remove_urls", True):
        text = _RE_URL.sub('', text)

    if s.get("remove_ticket_ids", True):
        text = _RE_TICKET_ID.sub('', text)

    if s.get("remove_timestamps", True):
        text = _RE_DATE.sub('', text)
        text = _RE_TIME.sub('', text)

    if s.get("remove_phone_numbers", True):
        text = _RE_PHONE.sub('', text)

    if s.get("remove_ip_addresses", True):
        text = _RE_IP.sub('', text)

    if s.get("remove_file_paths", True):
        text = _RE_WIN_PATH.sub('', text)
        text = _RE_UNIX_PATH.sub('', text)

    if s.get("remove_special_chars", True):
        text = _RE_SPECIAL_CHARS.sub(' ', text)

    # Apply user-defined custom regex patterns
    for pattern in s.get("custom_regex_patterns", []):
        try:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        except re.error:
            pass  # skip invalid patterns

    # Remove extra whitespace
    text = _RE_WHITESPACE.sub(' ', text).strip()

    # Remove very short tokens (configurable)
    min_len = s.get("min_token_length", 3)
    tokens = [t for t in text.split() if len(t) >= min_len]

    return ' '.join(tokens)


def remove_boilerplate(text, custom_patterns=None):
    """Remove common boilerplate phrases from tickets"""
    if not isinstance(text, str):
        return ""

    default_patterns = [
        # Greetings
        r'dear\s+(sir|madam|team|support|all|it)',
        r'hi\s+(team|all|there|support)',
        r'hello\s+(team|all|there|support)',
        r'good\s+(morning|afternoon|evening|day)',
        # Sign-offs
        r'thanks?\s*(and)?\s*regards?',
        r'best\s+regards?',
        r'kind\s+regards?',
        r'sincerely',
        r'yours\s+(truly|faithfully)',
        r'cheers',
        # Common filler
        r'please\s+(help|assist|advise|look\s+into)',
        r'kindly\s+(help|assist|advise|look\s+into)',
        r'urgent[ly]?\s*(please)?',
        r'asap',
        r'as\s+soon\s+as\s+possible',
        r'at\s+your\s+earliest\s+convenience',
        # Auto-generated
        r'this\s+is\s+an?\s+automated?\s+message',
        r'do\s+not\s+reply\s+to\s+this\s+email',
        r'sent\s+from\s+my\s+(iphone|android|mobile)',
    ]

    patterns = default_patterns + (custom_patterns or [])

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def preprocess_document(text, remove_boilerplate_text=None, clean_text=True,
                        settings=None):
    """
    Full preprocessing pipeline for a single document.

    Args:
        text: The input text
        remove_boilerplate_text: Whether to remove boilerplate phrases. Defaults to
            None, meaning "take it from settings['remove_boilerplate']" (itself
            defaulting to True). An explicit True/False still wins, for direct callers.
        clean_text: Whether to clean/normalize the text
        settings: Optional dict of preprocessing toggles (passed to
                  ``clean_ticket_text``).

    Returns:
        Preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    result = text

    # Previously this only consulted the parameter, which no caller ever passed — so the
    # Settings page's "Remove Boilerplate" checkbox was gathered into the config and then
    # ignored, stripping boilerplate whether it was ticked or not.
    if remove_boilerplate_text is None:
        remove_boilerplate_text = (settings or {}).get("remove_boilerplate", True)

    if remove_boilerplate_text:
        custom_bp = (settings or {}).get("custom_boilerplate_patterns", [])
        result = remove_boilerplate(result, custom_patterns=custom_bp or None)

    if clean_text:
        result = clean_ticket_text(result, settings=settings)

    # If preprocessing removed too much, return original (cleaned minimally)
    if len(result) < 10:
        result = re.sub(r'\s+', ' ', text).strip()

    return result


def preprocess_documents(docs, callback=None, settings=None):
    """
    Preprocess a list of documents.

    Args:
        docs: List of document strings
        callback: Optional callback function for progress updates
        settings: Optional dict of preprocessing toggles

    Returns:
        List of preprocessed documents
    """
    processed = []
    total = len(docs)

    for i, doc in enumerate(docs):
        processed.append(preprocess_document(doc, settings=settings))

        if callback and i % 100 == 0:
            progress = int((i / total) * 100)
            callback(f"Preprocessing... {progress}%")

    return processed
