"""
Logging infrastructure for the ticket clustering tool.
Provides file-based and console logging with configurable levels.
"""

import logging
import logging.handlers
import os
from datetime import datetime

_logger = None
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# A fresh timestamped log file is created per launch; keep only the newest
# MAX_LOG_FILES so the logs/ directory cannot grow without bound.
MAX_LOG_FILES = 15

# Per-file byte ceiling. The count cap alone bounded the number of files but not
# their size, so a long DEBUG-level run over a large ticket set could write an
# arbitrarily large file (15 unbounded files is still unbounded).
MAX_LOG_BYTES = 10 * 1024 * 1024      # 10 MB
LOG_BACKUP_COUNT = 1                  # one .1 rollover per launch file


def _is_log_file(name):
    """True for a log this app wrote, including RotatingFileHandler rollovers.

    Rollovers get a numeric suffix appended *after* the extension
    ("clustering_20260813.log.1"), so a plain endswith(".log") misses them.
    """
    return name.endswith(".log") or ".log." in name


def _prune_old_logs(keep=MAX_LOG_FILES, reserve=1):
    """Delete all but the newest `keep` log files.

    `reserve` accounts for the file this launch is about to create, so the directory
    settles at `keep` files rather than `keep + 1` (pruning runs before the new
    handler opens its file).

    Covers RotatingFileHandler's rollover files too. Those are named
    `clustering_<ts>.log.1`, which does NOT end in ".log" — so an `endswith(".log")`
    filter skipped them entirely and each one (up to MAX_LOG_BYTES, i.e. 10 MB) stayed
    on disk forever, which is exactly what the size cap was added to prevent.
    """
    try:
        logs = sorted(f for f in os.listdir(_LOG_DIR) if _is_log_file(f))
    except OSError:
        return
    if keep <= 0:
        stale = logs
    else:
        limit = max(0, keep - reserve)
        stale = logs[:-limit] if limit else logs
    for name in stale:
        try:
            os.remove(os.path.join(_LOG_DIR, name))
        except OSError:
            pass


def get_logger(name="clustering"):
    """Get or create the application logger."""
    global _logger
    if _logger is not None:
        return _logger

    os.makedirs(_LOG_DIR, exist_ok=True)
    _prune_old_logs()

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if _logger.handlers:
        return _logger

    # File handler - detailed, size-capped so one long run can't fill the disk.
    log_file = os.path.join(_LOG_DIR, f"clustering_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler - info and above. Reconfigure the console stream to UTF-8
    # (errors="replace") so non-ASCII marking chars (·, —, →) in log messages
    # can't raise UnicodeEncodeError on a default cp1252 Windows console.
    import sys
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    _logger.addHandler(fh)
    _logger.addHandler(ch)

    _logger.info(f"Log file: {log_file}")
    return _logger


def get_log_directory():
    """Return the log directory path."""
    return _LOG_DIR


def get_latest_log_file():
    """Return path to the most recent *active* log file, or None.

    Deliberately still ".log" only, unlike _prune_old_logs: a ".log.1" rollover holds the
    OLDER half of a launch's output, so offering it as "the latest log" would hand the
    user the wrong file. Pruning has to see them; this must not.
    """
    if not os.path.exists(_LOG_DIR):
        return None
    logs = sorted(
        [f for f in os.listdir(_LOG_DIR) if f.endswith(".log")],
        reverse=True,
    )
    return os.path.join(_LOG_DIR, logs[0]) if logs else None
