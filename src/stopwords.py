"""
Domain-specific stopwords for IT ticket clustering.
Combines standard English stopwords with IT support terminology.
Supports user-defined custom stopwords loaded from config.
"""

import os


# IT Support domain-specific stopwords
IT_STOPWORDS = [
    # Generic request terms
    "please", "help", "need", "want", "issue", "problem", "error",
    "unable", "cannot", "cant", "doesnt", "dont", "wont", "isnt",
    "work", "working", "worked", "works",
    
    # Ticket system noise
    "ticket", "request", "incident", "submitted", "created", "updated",
    "assigned", "resolved", "closed", "pending", "status", "priority",
    "severity", "category", "subcategory", "description", "summary",
    
    # User references
    "user", "users", "customer", "client", "employee", "staff",
    "requester", "caller", "reported", "reporter",
    
    # Time related
    "today", "yesterday", "tomorrow", "morning", "afternoon", "evening",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "month", "year", "date", "time", "day", "days", "hours", "hour",
    
    # Common verbs
    "trying", "tried", "try", "getting", "get", "got", "received", "receive",
    "sent", "send", "called", "call", "asking", "ask", "asked",
    "using", "use", "used", "open", "opened", "opening", "close", "closing",
    "start", "started", "starting", "stop", "stopped", "stopping",
    
    # Filler words
    "also", "still", "already", "again", "always", "sometimes", "never",
    "however", "therefore", "regarding", "concerning", "related", "attached",
    "following", "below", "above", "mentioned", "noted", "stated",
    
    # Pronouns and articles (extra)
    "ive", "youve", "theyve", "weve", "its", "theyre", "youre", "were",
    "hes", "shes", "thats", "whats", "whos", "wheres", "whens", "hows",
    
    # Common adjectives
    "new", "old", "good", "bad", "same", "different", "available",
    "current", "previous", "next", "last", "first",
    
    # Quantities
    "one", "two", "three", "many", "multiple", "several", "few", "some",
    "all", "any", "every", "each", "both", "none",
    
    # # IT generic terms (too common to be meaningful)
    # "system", "systems", "computer", "computers", "laptop", "laptops",
    # "machine", "device", "devices", "screen", "button", "click", "clicking",
]

# Path for user-managed custom stopwords (one word per line)
_CUSTOM_STOPWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "custom_stopwords.txt"
)


def _load_custom_stopwords_file():
    """Load custom stopwords from the text file on disk."""
    if not os.path.exists(_CUSTOM_STOPWORDS_PATH):
        return []
    try:
        with open(_CUSTOM_STOPWORDS_PATH, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except OSError:
        return []


def save_custom_stopwords(words):
    """Persist a list of custom stopwords to the text file."""
    os.makedirs(os.path.dirname(_CUSTOM_STOPWORDS_PATH), exist_ok=True)
    with open(_CUSTOM_STOPWORDS_PATH, "w", encoding="utf-8") as f:
        for w in sorted(set(w.strip().lower() for w in words if w.strip())):
            f.write(w + "\n")


def get_stopwords(settings=None):
    """
    Returns combined English + IT domain + custom stopwords as a list.

    Args:
        settings: Optional dict with keys ``use_english_stopwords``,
                  ``use_it_stopwords``, ``custom_stopwords``. This is the
                  ``config["stopwords"]`` *section*, not the whole config — passing the
                  whole config finds none of these keys and silently uses every default.
                  Falls back to sensible defaults when ``None``.
    """
    s = settings or {}
    combined = []

    if s.get("use_english_stopwords", True):
        # Imported lazily: pulling in sklearn is slow (multi-second on first
        # import), so callers that only need the file helpers — e.g. the
        # "Manage Custom Stopwords" dialog — don't pay that cost. Both callers
        # (clustering and the Word Cloud Studio) already have sklearn loaded.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        combined.extend(ENGLISH_STOP_WORDS)

    if s.get("use_it_stopwords", True):
        combined.extend(IT_STOPWORDS)

    # Add custom stopwords from config dict
    combined.extend(s.get("custom_stopwords", []))

    # Add custom stopwords from file
    combined.extend(_load_custom_stopwords_file())

    return list(set(combined))


