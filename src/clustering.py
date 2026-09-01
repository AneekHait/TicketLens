import hashlib
import os
import re
import time
from collections import Counter

# Lazy imports for heavy ML libraries
_np = None
_BERTopic = None
_SentenceTransformer = None
_UMAP = None
_HDBSCAN = None
_KMeans = None
_CountVectorizer = None
_ClassTfidfTransformer = None

def _lazy_import_ml():
    """Lazy import heavy ML libraries only when needed"""
    global _np, _BERTopic, _SentenceTransformer, _UMAP, _HDBSCAN, _KMeans, _CountVectorizer, _ClassTfidfTransformer
    
    if _np is None:
        import numpy
        _np = numpy
    
    if _BERTopic is None:
        from bertopic import BERTopic
        _BERTopic = BERTopic
    
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    
    if _UMAP is None:
        from umap import UMAP
        _UMAP = UMAP
    
    if _HDBSCAN is None:
        from hdbscan import HDBSCAN
        _HDBSCAN = HDBSCAN
    
    if _KMeans is None:
        from sklearn.cluster import KMeans
        _KMeans = KMeans
    
    if _CountVectorizer is None:
        from sklearn.feature_extraction.text import CountVectorizer
        _CountVectorizer = CountVectorizer
    
    if _ClassTfidfTransformer is None:
        from bertopic.vectorizers import ClassTfidfTransformer
        _ClassTfidfTransformer = ClassTfidfTransformer

from src.preprocessing import preprocess_documents
from src.stopwords import get_stopwords
from src.logger import get_logger
from src.config import EMBEDDING_MODELS, embedding_dir_complete

logger = get_logger()


class ClusteringCancelled(Exception):
    """Raised inside TicketClusterer.run() when the caller's should_stop() returns True."""


# UMAP must drop its random seed to run multi-core, which loses bit-for-bit
# reproducibility. In "auto" mode we only parallelize once the dataset is large
# enough that the speed-up is worth giving that up.
UMAP_PARALLEL_THRESHOLD = 10000


# Seed for the training subsample drawn on very large datasets. Fixed so that
# "reproducible" mode really is reproducible: this previously used the global numpy
# RNG, so the subsample (and therefore the fitted UMAP) differed on every run even
# when the user had explicitly asked for the reproducible single-core mode.
SAMPLE_SEED = 42


def _sample_indices(n, size):
    """Choose `size` distinct indices out of `n`, reproducibly."""
    return _np.random.default_rng(SAMPLE_SEED).choice(n, size, replace=False)


def _umap_parallel(mode, n_samples):
    """Resolve the UMAP parallelism choice → (random_state, n_jobs, label).
    mode: 'auto' (parallel only on large data) | 'parallel' | 'reproducible'.

    The label is surfaced in the run log, so it states the reproducibility
    consequence outright — "auto" silently becomes non-reproducible above
    UMAP_PARALLEL_THRESHOLD rows, which is the common case for real datasets.
    """
    parallel = mode == "parallel" or (mode == "auto" and n_samples >= UMAP_PARALLEL_THRESHOLD)
    if parallel:
        return None, -1, "parallel (all cores) — results are NOT bit-for-bit reproducible"
    return 42, 1, "single-core (reproducible)"


# Above this many rows, transform in chunks rather than in one call. A single
# transform of a very large array is what the batching exists to avoid.
UMAP_TRANSFORM_BATCH_THRESHOLD = 100000
UMAP_TRANSFORM_BATCH_SIZE = 50000


def _umap_transform_all(umap_model, embeddings, log=None, progress_from=None,
                        progress_to=None):
    """Project every row of `embeddings` with a fitted `umap_model`.

    Shared by run() and suggest_min_cluster_size() so the two produce the *same*
    projection. This is not merely a tidy-up: UMAP's ``transform`` runs its own
    optimisation per call, so a single full-array transform and a batched one give
    different arrays. When the sweep's projection is handed back to run() for reuse,
    any difference here would silently change the clustering — and the sweep used to
    do in one call exactly what run() batches "to avoid memory issues".
    """
    n = len(embeddings)
    if n <= UMAP_TRANSFORM_BATCH_THRESHOLD:
        return umap_model.transform(embeddings)

    parts = []
    for i in range(0, n, UMAP_TRANSFORM_BATCH_SIZE):
        end = min(i + UMAP_TRANSFORM_BATCH_SIZE, n)
        parts.append(umap_model.transform(embeddings[i:end]))
        if log:
            if progress_from is not None and progress_to is not None:
                span = progress_to - progress_from
                log(f"  → Transforming: {end:,}/{n:,} docs...",
                    progress_from + span * end / n)
            else:
                log(f"  → Transforming: {end:,}/{n:,} docs...")
    # Imported locally rather than via the module-level `_np` handle: that one is only
    # populated by _lazy_import_ml(), so relying on it would make this helper silently
    # depend on the caller having triggered the lazy import first.
    import numpy as np
    return np.vstack(parts)


# ----- Embedding cache helpers -----
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE_DIR = os.path.join(_REPO_ROOT, "cache")


def resolve_cache_dir(directory):
    """Resolve the configured cache.directory to an absolute path.

    Relative paths (the default, "cache") are anchored to the repo root, so the
    default resolves to the original <repo>/cache location and existing cache
    files still hit. Absolute paths are used as-is.
    """
    if not directory:
        return _CACHE_DIR
    return directory if os.path.isabs(directory) else os.path.join(_REPO_ROOT, directory)


def _embedding_cache_key(docs, model_name, backend="pytorch", precision=None):
    """Compute a deterministic hash for a list of docs + model name + backend.

    The backend/precision are folded in so PyTorch and OpenVINO-INT8 embeddings
    (which differ numerically) don't collide in the cache. The pytorch path is
    kept byte-identical to the original so existing cache files still hit.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for d in docs:
        h.update(d.encode("utf-8"))
    if backend and backend != "pytorch":
        h.update(f"|{backend}|{precision or ''}".encode("utf-8"))
    return h.hexdigest()


def _load_cached_embeddings(cache_key, cache_dir=_CACHE_DIR):
    """Return cached numpy array or None."""
    _lazy_import_ml()
    path = os.path.join(cache_dir, f"{cache_key}.npy")
    if os.path.exists(path):
        return _np.load(path)
    return None


def _save_embeddings_cache(cache_key, embeddings, cache_dir=_CACHE_DIR):
    """Persist embeddings to disk."""
    _lazy_import_ml()
    os.makedirs(cache_dir, exist_ok=True)
    _np.save(os.path.join(cache_dir, f"{cache_key}.npy"), embeddings)


def _candidate_sizes(n, current=None):
    """Sensible Min Cluster Size candidates for n docs (always includes `current`)."""
    base = [2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120]
    hi = max(2, n // 3)
    sizes = {c for c in base if 2 <= c <= hi}
    if current and 2 <= int(current) <= max(2, n):
        sizes.add(int(current))
    return sorted(sizes) or [2]

# EMBEDDING_MODELS (label -> HF path) now lives in src/config.py (imported above)
# so the GUI can read it without importing this sklearn-pulling module.
# NOTE: its keys MUST match MODEL_SPEED_ESTIMATES keys below and the GUI dropdown.

# Estimated docs/second for each model (CPU baseline)
# These are rough estimates and will be refined during processing
MODEL_SPEED_ESTIMATES = {
    "all-MiniLM-L6-v2 (Basic · Fastest)": 80,
    "gte-small (Good · Fast)": 60,
    "bge-small-en-v1.5 (Good · Fast)": 50,
    "bge-base-en-v1.5 (High · Moderate — Recommended)": 40,
    "gte-base-en-v1.5 (High · Moderate · 8k ctx)": 32,
    "embeddinggemma-300m (Top · Slow)": 25,
    "harrier-270m (Top · Slow · 32k ctx)": 22,
    "bge-large-en-v1.5 (High · Slow)": 20,
    "gte-large-en-v1.5 (High · Slow · 8k ctx)": 18,
    "qwen3-embedding-0.6b (Top · Slowest)": 10,
    "harrier-0.6b (Top · Slowest · 32k ctx)": 9,
}

# Baked into the instruction-tuned prefixes below. One concise sentence — these
# models are trained on short task descriptions, so verbose instructions hurt.
DEFAULT_CLUSTER_INSTRUCTION = "Identify the topic or category of the IT support ticket"

# Prefixes prepended to each doc before embedding — fixed by each model's training
# (not user-tunable). EmbeddingGemma is prompt-based (fixed clustering prompt);
# Qwen3/Harrier are instruction-tuned. Note Qwen3 has NO space after "Query:" while
# Harrier REQUIRES one — the prompt is prepended as prefix + doc, so that trailing
# space must live in the literal.
MODEL_PREFIXES = {
    "intfloat/e5-large-v2": "query: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
    "unsloth/embeddinggemma-300m": "task: clustering | query: ",
    "Qwen/Qwen3-Embedding-0.6B":     f"Instruct: {DEFAULT_CLUSTER_INSTRUCTION}\nQuery:",
    "microsoft/harrier-oss-v1-270m": f"Instruct: {DEFAULT_CLUSTER_INSTRUCTION}\nQuery: ",
    "microsoft/harrier-oss-v1-0.6b": f"Instruct: {DEFAULT_CLUSTER_INSTRUCTION}\nQuery: ",
}


def format_time(seconds):
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def estimate_processing_time(num_docs, model_name):
    """
    Estimate total processing time based on document count and model.
    Returns (total_seconds, breakdown_dict)
    """
    # Get embedding speed estimate
    docs_per_sec = MODEL_SPEED_ESTIMATES.get(model_name, 40)
    
    # Embedding time (biggest factor)
    embedding_time = num_docs / docs_per_sec
    
    # Other steps (rough estimates based on document count)
    preprocessing_time = num_docs * 0.001  # ~1ms per doc
    clustering_time = min(num_docs * 0.005, 3600)  # UMAP/HDBSCAN scales non-linearly
    labeling_time = 60 + (num_docs * 0.0005)  # LLM labeling
    
    total = embedding_time + preprocessing_time + clustering_time + labeling_time
    
    breakdown = {
        'embedding': embedding_time,
        'preprocessing': preprocessing_time,
        'clustering': clustering_time,
        'labeling': labeling_time,
        'total': total,
        'docs_per_sec': docs_per_sec
    }
    
    return total, breakdown


# Predefined keyword -> macro-category templates.
# Categories are the recurring macro-categories observed across the client
# deliverables taxonomy (see dist/canonical_taxonomy.*). Each subcategory the
# clusterer produces is filed under one of these so output stays consistent
# with how the analyst teams already organise tickets.
CATEGORY_TEMPLATES = {
    # --- Access & Authorization ---
    "password": "Access & Authorization",
    "login": "Access & Authorization",
    "account": "Access & Authorization",
    "permission": "Access & Authorization",
    "unlock": "Access & Authorization",
    "access": "Access & Authorization",
    "joiner": "Access & Authorization",
    "leaver": "Access & Authorization",
    "mover": "Access & Authorization",
    "onboarding": "Access & Authorization",
    "offboarding": "Access & Authorization",
    "ad account": "Access & Authorization",
    "active directory": "Access & Authorization",
    "credentials": "Access & Authorization",
    "authentication": "Access & Authorization",
    "mfa": "Access & Authorization",
    "2fa": "Access & Authorization",
    "sso": "Access & Authorization",
    "single sign": "Access & Authorization",
    # --- Email & Collaboration ---
    "email": "Email & Collaboration",
    "outlook": "Email & Collaboration",
    "mailbox": "Email & Collaboration",
    "calendar": "Email & Collaboration",
    "shared mailbox": "Email & Collaboration",
    "distribution list": "Email & Collaboration",
    "mail flow": "Email & Collaboration",
    "smtp": "Email & Collaboration",
    "exchange": "Email & Collaboration",
    "inbox": "Email & Collaboration",
    "teams": "Email & Collaboration",
    "sharepoint": "Email & Collaboration",
    "onedrive": "Email & Collaboration",
    "collaboration": "Email & Collaboration",
    "meeting": "Email & Collaboration",
    "webex": "Email & Collaboration",
    "zoom": "Email & Collaboration",
    "ms teams": "Email & Collaboration",
    "microsoft teams": "Email & Collaboration",
    # --- Network & Connectivity ---
    "vpn": "Network & Connectivity",
    "network": "Network & Connectivity",
    "wifi": "Network & Connectivity",
    "internet": "Network & Connectivity",
    "connectivity": "Network & Connectivity",
    "dns": "Network & Connectivity",
    "proxy": "Network & Connectivity",
    "firewall": "Network & Connectivity",
    "lan": "Network & Connectivity",
    "wan": "Network & Connectivity",
    "wireless": "Network & Connectivity",
    "vdi": "Network & Connectivity",
    "citrix": "Network & Connectivity",
    "remote desktop": "Network & Connectivity",
    "rdp": "Network & Connectivity",
    "virtual desktop": "Network & Connectivity",
    "horizon": "Network & Connectivity",
    # --- Device & Hardware ---
    "laptop": "Device & Hardware",
    "monitor": "Device & Hardware",
    "keyboard": "Device & Hardware",
    "mouse": "Device & Hardware",
    "hardware": "Device & Hardware",
    "device": "Device & Hardware",
    "asset": "Device & Hardware",
    "equipment": "Device & Hardware",
    "docking": "Device & Hardware",
    "headset": "Device & Hardware",
    "peripheral": "Device & Hardware",
    "replacement": "Device & Hardware",
    "new hire": "Device & Hardware",
    "procurement": "Device & Hardware",
    "printer": "Device & Hardware",
    "print": "Device & Hardware",
    "scanning": "Device & Hardware",
    "scanner": "Device & Hardware",
    "fax": "Device & Hardware",
    "mfp": "Device & Hardware",
    "mobile": "Device & Hardware",
    "iphone": "Device & Hardware",
    "android": "Device & Hardware",
    "intune": "Device & Hardware",
    "mdm": "Device & Hardware",
    "airwatch": "Device & Hardware",
    "tablet": "Device & Hardware",
    "ipad": "Device & Hardware",
    # --- Software & Applications ---
    "install": "Software & Applications",
    "software": "Software & Applications",
    "application": "Software & Applications",
    "update": "Software & Applications",
    "license": "Software & Applications",
    "patch": "Software & Applications",
    "upgrade": "Software & Applications",
    "adobe": "Software & Applications",
    "office": "Software & Applications",
    "chrome": "Software & Applications",
    "java": "Software & Applications",
    # NOTE: no bare "app" pattern — as a substring it false-matches words like
    # "approval"/"appointment". The longer "application"/"software" patterns cover
    # the real cases.
    # --- Business Applications (line-of-business / ERP) ---
    "sap": "Business Applications",
    "sap gui": "Business Applications",
    "transaction": "Business Applications",
    "bw ": "Business Applications",
    "hana": "Business Applications",
    "fiori": "Business Applications",
    "abap": "Business Applications",
    "erp": "Business Applications",
    "oracle": "Business Applications",
    "salesforce": "Business Applications",
    "workday": "Business Applications",
    "jd edwards": "Business Applications",
    "jde": "Business Applications",
    "kyriba": "Business Applications",
    "maximo": "Business Applications",
    "concur": "Business Applications",
    # --- Data & Reporting ---
    "report": "Data & Reporting",
    "power bi": "Data & Reporting",
    "powerbi": "Data & Reporting",
    "tableau": "Data & Reporting",
    "dashboard": "Data & Reporting",
    "analytics": "Data & Reporting",
    "database": "Data & Reporting",
    "sql": "Data & Reporting",
    "data": "Data & Reporting",
    # --- Jobs & Batch Processing ---
    "job": "Jobs & Batch Processing",
    "batch": "Jobs & Batch Processing",
    "scheduler": "Jobs & Batch Processing",
    "control-m": "Jobs & Batch Processing",
    "controlm": "Jobs & Batch Processing",
    "autosys": "Jobs & Batch Processing",
    "abend": "Jobs & Batch Processing",
    "cron": "Jobs & Batch Processing",
    "long running": "Jobs & Batch Processing",
    "redwood": "Jobs & Batch Processing",
    # --- Monitoring & Alerts ---
    "alert": "Monitoring & Alerts",
    "netcool": "Monitoring & Alerts",
    "monitoring": "Monitoring & Alerts",
    "threshold": "Monitoring & Alerts",
    "snmp": "Monitoring & Alerts",
    "heartbeat": "Monitoring & Alerts",
    # --- Security & Compliance ---
    "security": "Security & Compliance",
    "virus": "Security & Compliance",
    "malware": "Security & Compliance",
    "phishing": "Security & Compliance",
    "encryption": "Security & Compliance",
    "bitlocker": "Security & Compliance",
    "dlp": "Security & Compliance",
    "compliance": "Security & Compliance",
    "suspicious": "Security & Compliance",
    "threat": "Security & Compliance",
    # --- Infrastructure & Servers ---
    "server": "Infrastructure & Servers",
    "backup": "Infrastructure & Servers",
    "storage": "Infrastructure & Servers",
    "aws": "Infrastructure & Servers",
    "azure": "Infrastructure & Servers",
    "cloud": "Infrastructure & Servers",
    "vm ": "Infrastructure & Servers",
    "virtual machine": "Infrastructure & Servers",
    "vmware": "Infrastructure & Servers",
    # NOTE: "Service Requests" was intentionally removed as a macro-category. The
    # source data is a support-/service-request export, so *every* ticket is
    # already a service request — a category by that name is tautological and
    # just becomes a magnet for tickets (especially access provisioning) that
    # belong in a real functional bucket. The old ServiceNow/portal/catalog/RITM
    # keywords are dropped so those clusters fall through to the LLM, and
    # "provisioning" is now treated as access intent (see ACCESS_INTENT_RE) and
    # rolls up under Access & Authorization.
}

STANDARD_CATEGORIES = [
    "Access & Authorization",
    "Email & Collaboration",
    "Network & Connectivity",
    "Device & Hardware",
    "Software & Applications",
    "Business Applications",
    "Data & Reporting",
    "Jobs & Batch Processing",
    "Monitoring & Alerts",
    "Security & Compliance",
    "Infrastructure & Servers",
    "General IT Support",
]

# Connector / filler words dropped when comparing two category names for
# equivalence — they carry no distinguishing meaning at the macro-category level.
_CAT_STOPWORDS = {"and", "the", "of", "for", "a", "an", "related"}


def _cat_signature(text):
    """Order-, plural- and punctuation-insensitive token signature of a category
    name. Two names with the same signature are the *same* macro-category worded
    differently — e.g. "Application & Software", "Software & Applications" and
    "Business Application" / "Business Applications". Used to canonicalize the
    LLM's free-text category output so near-duplicate categories don't pile up.

    Note: the crude trailing-"s" strip can mangle non-plurals (access→acces),
    but every name is normalized the same way, so matching stays consistent.
    """
    text = re.sub(r"[&/,\-]", " ", str(text).lower())
    sig = set()
    for tok in re.findall(r"[a-z0-9]+", text):
        if tok in _CAT_STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        sig.add(tok)
    return frozenset(sig)


# Precomputed signatures for the standard taxonomy (built once at import).
_STANDARD_CATEGORY_SIGNATURES = [(_cat_signature(c), c) for c in STANDARD_CATEGORIES]

MIN_REPETITIVE_SIZE = 5

# --- Access/Authorization intent ------------------------------------------
# Access is an ACTION that spans every system (SAP access, SharePoint access,
# VPN access, network-drive access ...). Semantic clustering groups tickets by
# the *system* they name, so access work would otherwise scatter across
# Business Applications / Email / Network / etc. — exactly the leakage observed
# in the Diebold output. These patterns capture access/authorization *intent*
# and are checked FIRST in _match_category_template so intent wins over the
# system a ticket happens to mention. Word boundaries (\b) avoid matching inside
# unrelated words (e.g. "role" is not matched inside "payroll", "account" not
# inside "accounting"). Kept deliberately high-precision: ambiguous single words
# like bare "account" are only matched in an access phrase ("account creation",
# "account unlock", ...).
ACCESS_INTENT_RE = re.compile(r'''\b(?:
    access | permission | provision\w* | deprovision\w* | entitlement |
    password | passwd | unlock | lockout | locked\ out |
    login | log\ ?on | logon | log\ ?in | sign[\ -]?in |
    credential | authenticat\w* | authoriz\w* | authoris\w* |
    mfa | 2fa | otp | sso | single\ sign | multi[\ -]?factor |
    active\ directory | ad\ (?:account|group|id) | azure\ ad |
    onboard\w* | offboard\w* | joiner | leaver |
    roles? |
    account\ (?:creation|activation|deactivation|reactivation|extension|promotion|related|unlock|lock) |
    user\ (?:creation|id|account|access|termination|deactivation|activation) |
    (?:create|delete|deactivate|reactivate|terminate|modify)\ user |
    grant | revoke
)\b''', re.IGNORECASE | re.VERBOSE)

# Unambiguous, ticket-level credential signals used as a safety net for tickets
# HDBSCAN dropped into the noise bucket (Non-Repetitive) — see the GUI post-pass.
# These are only ever applied to individual noise tickets, so they are allowed to
# be a little broad (bare "mfa"/"2fa") without risking real clusters.
ACCESS_HARD_SIGNAL_RE = re.compile(
    r'\b(?:'
    r'password\s+reset|reset\s+password|reset\s+the\s+password|'
    r'forgot\s+(?:my\s+)?password|password\s+expir\w*|password\s+chang\w*|'
    r'account\s+unlock|unlock\s+(?:the\s+)?account|account\s+(?:is\s+)?locked|'
    r'unlock\s+(?:the\s+)?user|user\s+unlock|'
    r'mfa|multi[\s-]?factor|2fa'
    r')\b', re.IGNORECASE)

# Cache for embedding models, keyed (model_name, backend, precision).
#
# Bounded: these hold real weights (bge-base ~440 MB, Qwen3-Embedding multi-GB), so an
# unbounded dict meant trying four models in one session kept all four resident. We keep
# the most recent few and evict oldest-first; a re-load from the local snapshot is cheap
# compared to the memory.
_EMBEDDING_CACHE_MAX = 2
_embedding_model_cache = {}


def _cache_embedding_model(cache_key, model):
    """Store a model, evicting the oldest entries beyond _EMBEDDING_CACHE_MAX."""
    _embedding_model_cache[cache_key] = model
    while len(_embedding_model_cache) > _EMBEDDING_CACHE_MAX:
        evicted_key = next(iter(_embedding_model_cache))
        _embedding_model_cache.pop(evicted_key, None)
        logger.info(f"Evicted cached embedding model: {evicted_key}")


def clear_embedding_model_cache():
    """Drop every cached embedding model (frees their weights)."""
    n = len(_embedding_model_cache)
    _embedding_model_cache.clear()
    if n:
        logger.info(f"Cleared {n} cached embedding model(s)")
    return n


class _PrecomputedUMAP:
    """Wrapper that returns pre-computed UMAP embeddings for BERTopic."""
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.embeddings

    def fit_transform(self, X, y=None):
        return self.embeddings


class _PrecomputedHDBSCAN:
    """Wrapper that returns pre-computed cluster labels for BERTopic."""
    def __init__(self, labels):
        self.labels_ = labels

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X, y=None):
        return self.labels_


def _download_embedding_snapshot(repo_id, local_dir):
    """Download the *complete* model repo into *local_dir*.

    Uses ``huggingface_hub.snapshot_download`` rather than
    ``SentenceTransformer(...).save()`` because the latter lazily fetches modules
    and can silently omit small subfolder configs (e.g. ``1_Pooling/config.json``),
    leaving the model unloadable. snapshot_download fetches and verifies every
    repo file and writes real files (not symlinks) into ``local_dir`` so the app
    can detect and reuse the model offline. Large weight formats the app never
    loads are skipped to save bandwidth and disk (green-by-default).
    """
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=[
            "*.onnx", "onnx/*",
            "openvino/*", "openvino_model.*",
            "*.h5", "tf_model.*",
            "*.msgpack", "flax_model.*",
            "rust_model.ot", "*.tflite", "*.gguf",
            # Skip the PyTorch pickle weights when safetensors are present -
            # e.g. bge-base-en-v1.5 ships BOTH model.safetensors and
            # pytorch_model.bin (438 MB each); we only load safetensors.
            "pytorch_model.bin", "*.bin",
        ],
    )


def get_embedding_model_resolved(model_name="bge-base-en-v1.5 (High · Moderate — Recommended)",
                                 acceleration="pytorch", calib_docs=None, log=None):
    """Load an embedding model and resolve its inference backend.

    Returns (model, backend, precision). `acceleration` is "pytorch" (default)
    or "openvino_int8_cpu". OpenVINO falls back to PyTorch on any failure
    (not installed, incompatible model, quantization error). `calib_docs` is the
    text used to calibrate INT8 (built once per model, cached). Models are cached
    keyed on (model_name, backend, precision) so backends coexist.
    """
    _lazy_import_ml()

    # Resolve actual HuggingFace path
    if model_name in EMBEDDING_MODELS:
        model_path = EMBEDDING_MODELS[model_name]
    else:
        match = next((v for k, v in EMBEDDING_MODELS.items() if model_name in k), None)
        model_path = match if match else model_name

    from src import acceleration as accel
    backend, precision, reason = accel.resolve_backend(acceleration, model_path)
    if reason:
        logger.info(f"Embedding acceleration: {reason}")
        if log:
            try:
                log(f"Embedding acceleration: {reason}")
            except Exception:
                pass

    cache_key = (model_name, backend, precision)
    if cache_key in _embedding_model_cache:
        logger.debug(f"Using cached embedding model: {cache_key}")
        return _embedding_model_cache[cache_key], backend, precision

    local_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'embedding', model_path.replace("/", "_")
    )

    try:
        # Ensure the COMPLETE model exists locally (needed by both backends).
        # snapshot_download fetches every repo file — unlike
        # SentenceTransformer(...).save(), which lazily skips the tiny subfolder
        # module configs (e.g. 1_Pooling/config.json) and leaves the model
        # unloadable. embedding_dir_complete() also flags earlier partial
        # downloads as incomplete so they are re-fetched.
        if not embedding_dir_complete(local_dir):
            logger.info(f"Downloading embedding model: {model_path}")
            try:
                _download_embedding_snapshot(model_path, local_dir)
                logger.info(f"Saved model to: {local_dir}")
            except Exception as e:
                logger.warning(f"Could not download model to {local_dir}: {e}")
        base_dir = local_dir if embedding_dir_complete(local_dir) else model_path

        # OpenVINO INT8 path — falls through to PyTorch on any failure.
        if backend == "openvino":
            ov_model = accel.load_accelerated_model(model_path, base_dir, calib_docs, log, _SentenceTransformer)
            if ov_model is not None:
                _cache_embedding_model(cache_key, ov_model)
                return ov_model, backend, precision
            logger.info("OpenVINO unavailable for this run — using PyTorch.")
            backend, precision = "pytorch", None
            cache_key = (model_name, backend, precision)
            if cache_key in _embedding_model_cache:
                return _embedding_model_cache[cache_key], backend, precision

        # PyTorch path (also the base for MPS, and the reference MLX is checked
        # against). device is "mps" only when resolve_backend cleared it.
        st_device = "mps" if backend == "mps" else "cpu"
        logger.info(f"Loading embedding model from: {base_dir} (device={st_device})")
        # SECURITY: trust_remote_code=True executes any custom modeling code
        # shipped inside the model repo. It is required by some curated models
        # (e.g. Qwen3-Embedding) and is safe for the vetted EMBEDDING_MODELS
        # catalog. If custom/user-supplied repos are ever loaded here, gate this
        # to the known-safe set (model_name in EMBEDDING_MODELS) instead.
        import logging as _logging
        _tf_log = _logging.getLogger("transformers")
        _tf_prev = _tf_log.level
        _tf_log.setLevel(_logging.ERROR)
        try:
            model = _SentenceTransformer(base_dir, device=st_device, trust_remote_code=True)
        except Exception as e:
            if st_device == "cpu":
                raise
            # MPS can fail per-model (an unimplemented op). Retry on CPU rather
            # than dropping to the all-MiniLM fallback, which would silently
            # swap the user's chosen model for a weaker one.
            logger.warning(f"MPS load failed ({e}); retrying on CPU.")
            if log:
                try:
                    log("Apple GPU (MPS) unavailable for this model — using CPU.")
                except Exception:
                    pass
            backend, precision = "pytorch", None
            cache_key = (model_name, backend, precision)
            model = _SentenceTransformer(base_dir, device="cpu", trust_remote_code=True)
        finally:
            _tf_log.setLevel(_tf_prev)

        # MLX embeddings are verified against this freshly loaded PyTorch model
        # before being used; load_embedding_model returns None unless the numbers
        # match, so a rejection costs a little time and changes no results.
        if backend == "mlx":
            from src import mlx_backend as mlxb
            mlx_model = mlxb.load_embedding_model(
                model_path, base_dir, calib_docs, log, reference_model=model)
            if mlx_model is not None:
                _cache_embedding_model((model_name, "mlx", None), mlx_model)
                return mlx_model, "mlx", None
            backend, precision = "pytorch", None
            cache_key = (model_name, backend, precision)

        _cache_embedding_model(cache_key, model)
        return model, backend, precision

    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        fallback = "sentence-transformers/all-MiniLM-L6-v2"
        logger.warning(f"Falling back to: {fallback}")
        # Prefer the bundled local copy so the fallback loads fully offline and
        # is not exposed to the same partial-download failure.
        fallback_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'embedding', fallback.replace("/", "_")
        )
        fallback_src = fallback_dir if embedding_dir_complete(fallback_dir) else fallback
        # trust_remote_code: see note on the main PyTorch load above. The fallback
        # (all-MiniLM-L6-v2) is a standard sentence-transformer with no custom code.
        _tf_log = _logging.getLogger("transformers")
        _tf_prev = _tf_log.level
        _tf_log.setLevel(_logging.ERROR)
        try:
            model = _SentenceTransformer(fallback_src, device="cpu", trust_remote_code=True)
        finally:
            _tf_log.setLevel(_tf_prev)
        _cache_embedding_model((model_name, "pytorch", None), model)
        return model, "pytorch", None


def get_embedding_model(model_name="bge-base-en-v1.5 (High · Moderate — Recommended)",
                        acceleration="pytorch", calib_docs=None, log=None):
    """Load embedding model by name with caching (backward-compatible wrapper)."""
    model, _backend, _precision = get_embedding_model_resolved(model_name, acceleration, calib_docs, log)
    return model


def _fallback_gguf_for(selection):
    """Map an LLM selection to something resolve_llm_path can handle.

    An MLX label is not a GGUF path, so if MLX could not load we must not pass
    the label through — resolve_llm_path would treat it as a custom file path,
    fail to find it, and report "no LLM" when a perfectly good curated GGUF was
    available. Substitute the default curated model instead.
    """
    from src.config import LLM_MODELS, MLX_LLM_MODELS
    if selection in MLX_LLM_MODELS:
        return next(iter(LLM_MODELS), None)
    return selection


def resolve_llm_path(selection, log=None):
    """Resolve an LLM dropdown selection to a local .gguf file path.

    `selection` is either a curated label (a key of config.LLM_MODELS) or an
    absolute path to a custom .gguf file. Curated models are downloaded into
    models/ on demand (mirrors get_embedding_model's behaviour). Returns the
    local path, or None if it can't be resolved/downloaded (caller falls back
    to keyword-only labels).
    """
    if not selection:
        return None

    from src.config import LLM_MODELS

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    # Curated model: ensure the GGUF is present locally, downloading if missing.
    spec = LLM_MODELS.get(selection)
    if spec:
        local = os.path.join(models_dir, spec["filename"])
        if os.path.exists(local):
            return local
        try:
            # Log the bare filename (never the "·" marking) to stay cp1252-safe.
            if log:
                log(f"Downloading LLM: {spec['filename']} (one-time)...", 0.12)
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=models_dir,
            )
        except Exception as e:
            logger.error(f"Could not download LLM {spec['filename']}: {e}")
            if log:
                log(f"LLM download failed ({spec['filename']}); using keyword labels.")
            return None

    # Custom file path (from the "Custom file…" picker).
    if isinstance(selection, str) and selection.lower().endswith(".gguf") and os.path.exists(selection):
        return selection

    return None


class TicketClusterer:
    def __init__(self, llm_model_path=None, embedding_model_name=None,
                 settings=None):
        self.llm_model_path = llm_model_path
        self.embedding_model_name = embedding_model_name or "bge-base-en-v1.5 (High · Moderate — Recommended)"
        self.topic_model = None
        self.llm = None
        self.discovered_categories = set()

        # Merge caller-supplied settings over defaults
        from src.config import DEFAULTS
        self._settings = settings or {}
        self._preprocess_settings = {**DEFAULTS["preprocessing"],
                                      **self._settings.get("preprocessing", {})}
        self._clustering_settings = {**DEFAULTS["clustering"],
                                      **self._settings.get("clustering", {})}
        self._stopword_settings = {**DEFAULTS["stopwords"],
                                    **self._settings.get("stopwords", {})}
        self._embedding_settings = {**DEFAULTS["embedding"],
                                    **self._settings.get("embedding", {})}
        self._acceleration = self._embedding_settings.get("acceleration", "pytorch")
        self._llm_settings = {**DEFAULTS["llm"], **self._settings.get("llm", {})}
        cache_cfg = {**DEFAULTS["cache"], **self._settings.get("cache", {})}
        self._cache_enabled = cache_cfg.get("enabled", True)
        self._cache_dir = resolve_cache_dir(cache_cfg.get("directory", "cache"))

    def close_llm(self):
        """Release the loaded GGUF model, if any.

        A Llama instance holds a multi-GB mmap. Relying on refcounting alone meant a
        re-run could load the new model while the previous one was still referenced by
        a finished generator (KBA/SOP/etc.), doubling peak RAM. Callers should invoke
        this on the *outgoing* clusterer before building a replacement. Safe to call
        more than once.
        """
        llm, self.llm = self.llm, None
        if llm is None:
            return False
        try:
            closer = getattr(llm, "close", None)
            if callable(closer):
                closer()
        except Exception as e:
            # Never let teardown break the caller — the model is dropped either way.
            logger.warning(f"Error closing the LLM (continuing): {e}")
        logger.info("Released the loaded LLM")
        return True

    def _is_cluster_repetitive(self, topic_id, topics):
        """Check if a cluster has enough tickets to be considered repetitive.

        Prefers the size table run() precomputes. The fallback scan is O(len(topics)) and
        this is called once per cluster, so relying on it made naming O(clusters x
        tickets) in pure Python — ~20M comparisons on 100k tickets / 200 clusters, for
        counts run() already has.
        """
        if topic_id == -1:
            return False
        sizes = getattr(self, "_topic_sizes", None)
        if sizes is not None:
            return sizes.get(topic_id, 0) >= MIN_REPETITIVE_SIZE
        return sum(1 for t in topics if t == topic_id) >= MIN_REPETITIVE_SIZE

    def _truncate_text(self, text, max_chars=200):
        """Truncate text to fit in LLM context"""
        if not text:
            return ""
        text = str(text)
        if len(text) > max_chars:
            return text[:max_chars].rsplit(' ', 1)[0] + "..."
        return text

    def _clean_llm_output(self, text, max_words=4):
        """Clean and normalize an LLM label.

        Keeps ':' and '/' so the analyst "Source : detail" / "A/B/Other" label
        style survives (e.g. "Nagios Alert : Disk Space", "Order: Release/Update/
        Delete/Other"). Title-cases plain words but preserves acronyms and given
        casing (ICM, SSRS, PowerBI, D.O.). ``max_words`` bounds the label length
        (subcategories pass a larger value than categories).
        """
        if not text:
            return ""

        text = text.strip()
        text = re.sub(r'^(label:|category:|name:|topic:|subcategory:|answer:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'^[-•*]\s*', '', text)
        text = re.sub(r'["\'\[\]\(\)]', '', text)
        # Strip markdown emphasis the LLM sometimes emits (e.g. "**User Access**").
        # Leave underscores alone — they can be legitimate (e.g. SAP_FICO).
        text = re.sub(r'[*`]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove sentence-like patterns (starts with "Here", "This", "The", etc.)
        text = re.sub(r'^(here\s+(is|are)|this\s+is|the\s+following|these\s+are|i\s+would|some\s+)\s*:?\s*', '', text, flags=re.IGNORECASE)

        # Take only the first line if multiple lines
        text = text.split('\n')[0].strip()

        words = text.strip().split()
        if len(words) > max_words:
            words = words[:max_words]

        # Title-case plain lowercase words, but preserve acronyms / embedded caps
        # (ICM, PowerBI, D.O.) and title-case each segment of a "/" enumeration
        # (release/update -> Release/Update).
        def _case(tok):
            return '/'.join(
                (seg[:1].upper() + seg[1:]) if seg.islower() else seg
                for seg in tok.split('/')
            )

        return ' '.join(_case(w) for w in words).strip()

    def _match_category_template(self, keywords, docs=None):
        """Map cluster keywords/label to a macro-category.

        Two-tier, priority-based — a rewrite of the old "first pattern that is a
        substring, longest pattern first" rule, which had two bugs:
          1. length, not meaning, decided the category, so a longer system noun
             ("sharepoint", "connectivity", "application") beat the "access"
             sitting right next to it — access tickets leaked into Email /
             Network / Software / etc.;
          2. it folded a raw sample ticket (``docs[0][:200]``) into the match, so
             one incidental word in one ticket could re-file a whole cluster.

        Now: (Tier 1) if the keywords/label carry access/authorization *intent*,
        return Access & Authorization — intent beats the named system. (Tier 2)
        otherwise score the remaining categories by how many distinct patterns
        hit, breaking ties by the longest matched pattern (most specific). Only
        the cluster keywords/label drive the decision; ``docs`` is accepted for
        backwards compatibility but intentionally ignored.
        """
        search_text = ' '.join(str(k) for k in keywords).lower()

        # Tier 1 — access/authorization intent wins over the system named.
        if ACCESS_INTENT_RE.search(search_text):
            return "Access & Authorization"

        # Tier 2 — score remaining categories: (distinct hits, longest match).
        # Access & Authorization is owned entirely by Tier 1, so its keyword
        # templates are skipped here — otherwise a prefix match like "account"
        # would wrongly fire on "accounting" and re-pull finance tickets.
        scores = {}
        for pattern, category in CATEGORY_TEMPLATES.items():
            if category == "Access & Authorization":
                continue
            pat = pattern.strip()
            # Prefix word-boundary keeps inflections ("report"→"reporting") while
            # dropping the mid-word false matches pure substring allowed.
            if re.search(rf'\b{re.escape(pat)}', search_text):
                hits, longest = scores.get(category, (0, 0))
                scores[category] = (hits + 1, max(longest, len(pat)))
        if scores:
            return max(scores.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]

        return None

    def _keywords_to_label(self, keywords, max_words=3):
        """Convert keywords to a readable label"""
        clean_kw = [str(k).strip() for k in keywords if k and len(str(k).strip()) > 2]
        if not clean_kw:
            return ""
        label = ' '.join(clean_kw[:max_words])
        return ' '.join(word.capitalize() for word in label.split())

    def _generate_subcategory_label(self, keywords, docs, topic_id=None, topics=None):
        """Generate a specific subcategory label"""
        if topic_id is not None and topics is not None:
            if not self._is_cluster_repetitive(topic_id, topics):
                return "Non-Repetitive"
        
        # Prepare fallback from keywords
        keyword_fallback = self._keywords_to_label(keywords, max_words=3)
        if not keyword_fallback:
            keyword_fallback = "General Issue"
        
        if not self.llm:
            return keyword_fallback

        top_keywords = [str(k) for k in keywords[:6] if k and len(str(k)) > 2]
        if not top_keywords:
            return keyword_fallback
            
        truncated_docs = [self._truncate_text(str(d), 150) for d in docs[:3] if d]

        # Single user message: the GGUF's embedded chat template handles role/turn
        # tokens, so this stays model-agnostic (Gemma, Qwen, Phi-3, Llama-3). No
        # system role — Gemma's template raises on one.
        user_content = f"""You are an IT service management analyst naming a recurring ticket theme.

Keywords: {', '.join(top_keywords)}

Sample tickets:
1. {truncated_docs[0] if len(truncated_docs) > 0 else 'N/A'}
2. {truncated_docs[1] if len(truncated_docs) > 1 else 'N/A'}
3. {truncated_docs[2] if len(truncated_docs) > 2 else 'N/A'}

Name the underlying recurring ISSUE, not a single incident. IGNORE machine-specific
details: hostnames, server/node names, IP addresses, drive letters, ticket numbers,
dates, user names and transaction IDs. For a monitoring alert, name it by source and
condition, e.g. "Nagios Alert : Disk Space".

Reply with ONLY the label (max 8 words, no sentence). You may use ":" to add a
source/parent and "/" to list close variants. Preserve acronyms (ICM, SSRS, VPN).
Examples: "Nagios Alert : Queue Backlog", "Disk Space Threshold Breach",
"Node Down / Unreachable", "Batch Job Failure : Control-M", "Database Connectivity Error",
"Certificate Expiry", "Password Reset / Account Unlock", "Report Generation Failure"

Label:"""

        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=self._llm_settings.get("max_tokens_subcategory", 20),
                stop=["\n\n"],
                temperature=self._llm_settings.get("temperature", 0.2)
            )
            label = output['choices'][0]['message']['content']
            cleaned = self._clean_llm_output(label, max_words=8)

            # Validate output
            if cleaned and len(cleaned) >= 3 and cleaned.lower() not in ['uncategorized', 'unknown', 'n/a', 'none']:
                return cleaned
            
            return keyword_fallback
            
        except Exception as e:
            logger.error(f"LLM error in subcategory: {e}")
            return keyword_fallback

    def _generate_category_label(self, subcategories, keywords, docs):
        """Generate a high-level category label"""
        # Try template match first
        template_match = self._match_category_template(keywords, docs)
        if template_match:
            return template_match
        
        # Prepare fallback
        keyword_fallback = self._keywords_to_label(keywords, max_words=2)
        if not keyword_fallback:
            keyword_fallback = "General IT Support"
        
        if not self.llm:
            return keyword_fallback

        sub_list = ', '.join([str(s) for s in subcategories[:5] if s and s != "Non-Repetitive"])
        top_keywords = [str(k) for k in keywords[:8] if k and len(str(k)) > 2]

        if not top_keywords and not sub_list:
            return keyword_fallback

        cat_options = '\n'.join([f"- {cat}" for cat in STANDARD_CATEGORIES])

        user_content = f"""You are an IT service management taxonomy expert. Assign this group of tickets to the single best macro-category.

Subcategories in this group: {sub_list if sub_list else 'N/A'}
Keywords: {', '.join(top_keywords) if top_keywords else 'N/A'}

Preferred categories (use one of these when it genuinely fits):
{cat_options}

Rules:
- Prefer the closest category above when it is a reasonable fit.
- If none reflects the business domain of these tickets, create a concise
  client-specific category of 1-3 words, e.g. "Order Related", "Policy Related",
  "Report Related", "Reversal/Cleanup Related".
- Do NOT answer "Service Request", "Service Requests", "Request", "Ticket" or
  "ServiceNow" — every ticket here is already a service request, so name what the
  request is ABOUT (the business domain), not that it is a request.
- Ignore hostnames, IDs, dates and other machine-specific details.
- Reply with ONLY the category name, nothing else.

Category:"""

        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=self._llm_settings.get("max_tokens_category", 15),
                stop=["\n\n"],
                temperature=self._llm_settings.get("temperature", 0.2)
            )
            label = output['choices'][0]['message']['content']
            cleaned = self._clean_llm_output(label, max_words=5)

            # Check for vague / disallowed responses. "Service Request(s)" et al.
            # are rejected here too: the category was removed (every ticket is a
            # service request), so if the LLM still emits it we fall back rather
            # than reintroduce the tautological bucket.
            if cleaned.lower() in ['mixed', 'unclear', 'various', 'miscellaneous',
                                   'other', 'uncategorized', 'unknown', 'n/a', 'none', '',
                                   'service request', 'service requests', 'request',
                                   'requests', 'servicenow', 'service now', 'ticket']:
                return keyword_fallback

            # Canonicalize to a standard category. First an order-/plural-insensitive
            # token-set match, so variants like "Application & Software" or
            # "Business Application" fold onto "Software & Applications" /
            # "Business Applications" instead of becoming near-duplicate categories.
            sig = _cat_signature(cleaned)
            if sig:
                for std_sig, std_cat in _STANDARD_CATEGORY_SIGNATURES:
                    if sig == std_sig:
                        return std_cat
            # Then the looser substring match (bare "Applications", "Security", …).
            for std_cat in STANDARD_CATEGORIES:
                if std_cat.lower() in cleaned.lower() or cleaned.lower() in std_cat.lower():
                    return std_cat

            if cleaned and len(cleaned) >= 3:
                if cleaned not in STANDARD_CATEGORIES:
                    self.discovered_categories.add(cleaned)
                return cleaned
            
            return keyword_fallback

        except Exception as e:
            logger.error(f"LLM error in category: {e}")
            return keyword_fallback

    def suggest_min_cluster_size(self, docs, use_preprocessing=True, candidates=None,
                                callback=None, precomputed_embeddings=None,
                                return_reduced=False, should_stop=None,
                                current_size=None):
        """Recommend a Min Cluster Size: embed + reduce once, then sweep candidate
        sizes through HDBSCAN and score each by DBCV / cluster count / noise.

        Returns {"recommended": int, "candidates": [{size,n_clusters,noise_pct,dbcv,valid}…],
        "reason": str}. Reuses the same embedding cache key as run().

        precomputed_embeddings: if provided (numpy array from a just-completed run()),
        the embedding phase is skipped entirely — saves minutes when called from the
        on_embeddings_ready hook mid-run.

        return_reduced: also return the UMAP projection under "reduced_embeddings", so
        the caller can hand it back to run() instead of paying for the same reduction
        twice. Same UMAP params *and* the same transform procedure as run(), so the
        projection is interchangeable.

        should_stop: cooperative cancellation, checked between candidate fits. Without
        it a Stop press had to wait out every remaining HDBSCAN fit.

        current_size: the size currently in effect, guaranteed a row in the sweep. Pass
        it when it may differ from settings["min_cluster_size"].
        """
        _lazy_import_ml()
        cs = self._clustering_settings

        def log(msg, progress=None):
            logger.info(msg)
            if callback:
                callback(msg, progress) if progress is not None else callback(msg)

        def _ck():
            if should_stop and should_stop():
                raise ClusteringCancelled("Cancelled during the Min Cluster Size sweep.")

        if precomputed_embeddings is not None:
            # Caller already has embeddings — jump straight to the sweep.
            embeddings = precomputed_embeddings
            log("Suggest: using pre-computed embeddings.", 0.45)
        else:
            # 1. Preprocess + prefix — identical to run() so the embedding cache key matches.
            log("Suggest: preparing documents…", 0.05)
            if use_preprocessing:
                processed = preprocess_documents(docs, settings=self._preprocess_settings)
            else:
                processed = [str(d) if d else "" for d in docs]
            model_path = EMBEDDING_MODELS.get(self.embedding_model_name, self.embedding_model_name)
            prefix = MODEL_PREFIXES.get(model_path, "")
            docs_for_embedding = [prefix + d for d in processed] if prefix else processed
            # Resolve backend (PyTorch / OpenVINO INT8) and load; calib uses these docs.
            embedding_model, _backend, _precision = get_embedding_model_resolved(
                self.embedding_model_name, acceleration=self._acceleration,
                calib_docs=docs_for_embedding, log=log)

            # 2. Embeddings (reuse cache; key includes backend so PyTorch/INT8 don't collide).
            cache_key = (_embedding_cache_key(docs_for_embedding, self.embedding_model_name, _backend, _precision)
                         if self._cache_enabled else None)
            embeddings = _load_cached_embeddings(cache_key, self._cache_dir) if cache_key else None
            if embeddings is not None:
                log("Suggest: loaded embeddings from cache.", 0.45)
            else:
                log("Suggest: generating embeddings…", 0.1)
                parts, bs, tot = [], self._embedding_settings.get("batch_size", 32), len(docs_for_embedding)
                for i in range(0, tot, bs):
                    parts.append(embedding_model.encode(docs_for_embedding[i:i + bs], show_progress_bar=False))
                    if (i // bs) % 5 == 0:
                        done = min(i + bs, tot)
                        log(f"Suggest: embedding {done:,}/{tot:,}…", 0.1 + 0.35 * done / max(1, tot))
                embeddings = _np.vstack(parts)
                del parts        # see the note in run(): vstack already copied
                if cache_key:
                    try:
                        _save_embeddings_cache(cache_key, embeddings, self._cache_dir)
                    except Exception as e:
                        logger.warning(f"Could not cache embeddings (suggest): {e}")

        # 3. UMAP reduce once (same params as run()).
        log("Suggest: reducing dimensions (UMAP)…", 0.55)
        n = len(embeddings)
        use_sampling = n > 50000
        umap_rs, umap_jobs, umap_label = _umap_parallel(cs.get("umap_parallel_mode", "auto"), n)
        log(f"Suggest: UMAP {umap_label}.")
        umap_model = _UMAP(
            n_neighbors=min(10, cs.get("umap_n_neighbors", 15)) if use_sampling else cs.get("umap_n_neighbors", 15),
            n_components=cs.get("umap_n_components", 5),
            min_dist=cs.get("umap_min_dist", 0.0),
            metric=cs.get("umap_metric", "euclidean"),
            random_state=umap_rs, low_memory=True, n_jobs=umap_jobs, verbose=False,
        )
        if use_sampling:
            idx = _sample_indices(n, min(50000, n))
            umap_model.fit(embeddings[idx])
            # Same helper run() uses: a single full transform here would not match
            # run()'s batched one, so a reused projection would change the clustering.
            reduced = _umap_transform_all(umap_model, embeddings, log=log)
        else:
            reduced = umap_model.fit_transform(embeddings)
        _ck()

        # 4. Sweep candidate sizes. `current` comes from the caller when given, so the
        # size actually in effect is always one of the rows the user sees scored (the
        # settings value can differ from the min_cluster_size run() was handed).
        current = int(current_size if current_size else cs.get("min_cluster_size", 15))
        sizes = candidates or _candidate_sizes(n, current)
        method = cs.get("hdbscan_cluster_selection_method", "eom")
        epsilon = cs.get("hdbscan_cluster_selection_epsilon", 0.0)
        min_samples = cs.get("hdbscan_min_samples", None)
        log(f"Suggest: evaluating {len(sizes)} candidate sizes…", 0.6)
        results = []
        for k, size in enumerate(sizes):
            # Outside the try: a cancellation must propagate, not be swallowed as a
            # failed candidate and logged as a warning.
            _ck()
            try:
                kwargs = dict(min_cluster_size=int(size), metric='euclidean',
                              cluster_selection_method=method, cluster_selection_epsilon=epsilon,
                              core_dist_n_jobs=-1, algorithm='best', gen_min_span_tree=True)
                if min_samples is not None:
                    kwargs["min_samples"] = min_samples
                model = _HDBSCAN(**kwargs)
                labels = model.fit_predict(reduced)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise = float(_np.mean(labels == -1)) if len(labels) else 1.0
                try:
                    dbcv = float(model.relative_validity_)
                    if dbcv != dbcv:  # NaN
                        dbcv = None
                except Exception:
                    dbcv = None
                results.append({"size": int(size), "n_clusters": int(n_clusters),
                                "noise_pct": round(noise * 100, 1), "dbcv": dbcv,
                                "valid": n_clusters >= 2 and noise <= 0.6})
                log(f"  → size {size}: {n_clusters} clusters, {noise*100:.0f}% noise"
                    + (f", DBCV {dbcv:.3f}" if dbcv is not None else ""),
                    0.6 + 0.35 * (k + 1) / len(sizes))
            except Exception as e:
                logger.warning(f"Suggest: size {size} failed: {e}")

        # 5. Recommend.
        valids = [r for r in results if r["valid"]]
        scored = [r for r in valids if r["dbcv"] is not None]
        if scored:
            top = max(r["dbcv"] for r in scored)
            # Near-tie bias: among sizes within a small DBCV margin of the best, prefer the
            # LARGEST (more docs per cluster → more robust, less fragmented).
            near = [r for r in scored if r["dbcv"] >= top - 0.02]
            best = max(near, key=lambda r: r["size"])
            tied = " (largest among near-tied DBCV)" if len(near) > 1 else ""
            reason = (f"Best density-based validity (DBCV {best['dbcv']:.3f}) → "
                      f"{best['n_clusters']} clusters{tied}.")
        elif valids:
            best = min(valids, key=lambda r: r["noise_pct"])
            reason = f"Most coverage: {best['n_clusters']} clusters, {best['noise_pct']}% noise."
        elif results:
            best = max(results, key=lambda r: (r["n_clusters"], -r["noise_pct"]))
            reason = "No clean separation found; chose the size yielding the most clusters."
        else:
            best = {"size": current}
            reason = "Could not evaluate candidates; keeping the current value."
        log(f"Suggest: recommended Min Cluster Size = {best['size']}", 1.0)
        out = {"recommended": int(best["size"]), "candidates": results, "reason": reason}
        if return_reduced:
            # Only handed back when explicitly asked for — a caller that ships this dict
            # across a Qt signal shouldn't be dragged into keeping an (n × 5) array alive.
            out["reduced_embeddings"] = reduced
        return out

    def run(self, docs, min_cluster_size=15, callback=None, intermediate_callback=None,
            use_preprocessing=True, label_categories=True, should_stop=None,
            on_embeddings_ready=None):
        """Main clustering method.

        label_categories: when False, skip Level-2 category labeling (macro_map stays
        empty besides Non-Repetitive). Used by the resolution-text clustering pass,
        where the symptom-oriented STANDARD_CATEGORIES taxonomy is meaningless.

        on_embeddings_ready: optional callable(embeddings, current_min_cluster_size),
        invoked right after embeddings are computed and before UMAP/HDBSCAN, so a caller
        can choose the cluster size once the slow, size-independent work is already done.
        It may return either an int (the chosen size) or a (size, reduced_embeddings)
        tuple; a returned projection of the right length is used in place of this
        method's own UMAP pass. Anything unusable — None, a non-int, a size below 2, a
        wrong-length projection, or an exception — leaves the run exactly as it was.
        """
        # Ensure all ML libraries are imported before use
        _lazy_import_ml()

        cs = self._clustering_settings
        # Allow per-call override but fall back to settings
        effective_min_cluster = min_cluster_size or cs.get("min_cluster_size", 15)
        umap_n_neighbors = cs.get("umap_n_neighbors", 15)
        umap_n_components = cs.get("umap_n_components", 5)
        umap_min_dist = cs.get("umap_min_dist", 0.0)
        umap_metric = cs.get("umap_metric", "euclidean")
        umap_parallel_mode = cs.get("umap_parallel_mode", "auto")
        hdbscan_min_samples = cs.get("hdbscan_min_samples", None)
        hdbscan_method = cs.get("hdbscan_cluster_selection_method", "eom")
        hdbscan_epsilon = cs.get("hdbscan_cluster_selection_epsilon", 0.0)

        def log(msg, progress=None):
            logger.info(msg)
            if callback:
                callback(msg, progress) if progress is not None else callback(msg)

        def _ck():
            # Cooperative cancellation: raise at a safe point when the caller (the
            # GUI Stop button) requests a stop. Native calls (UMAP/HDBSCAN/embedding/
            # LLM) can't be interrupted mid-flight, so we check between steps and
            # inside the batch/label loops.
            if should_stop and should_stop():
                raise ClusteringCancelled("Clustering cancelled by user.")

        _ck()
        log("--- Starting Clustering ---", 0.05)
        log(f"Using embedding model: {self.embedding_model_name}", 0.06)
        
        original_docs = [str(d) if d else "" for d in docs]
        
        # 0. Preprocess
        if use_preprocessing:
            log("Step 1/7: Preprocessing text...", 0.08)
            processed_docs = preprocess_documents(
                docs, settings=self._preprocess_settings)
        else:
            processed_docs = original_docs
        
        _ck()
        # 1. Load LLM
        if self.llm_model_path and self.llm is None:
            log("Step 2/7: Loading LLM...", 0.12)
            # self.llm_model_path may be a curated label or a custom path; resolve
            # (downloading the curated GGUF on demand) to a real file path.
            # MLX models are HF repos of MLX weights, not GGUF files, so they
            # bypass resolve_llm_path entirely. Falls through to llama.cpp if
            # mlx-lm is missing or the load fails.
            from src.config import MLX_LLM_MODELS
            mlx_spec = MLX_LLM_MODELS.get(self.llm_model_path)
            if mlx_spec:
                from src import mlx_backend as mlxb
                self.llm = mlxb.load_llm(mlx_spec["repo_id"], log=log)
                if self.llm is None:
                    log("MLX unavailable; falling back to the default GGUF model.")

            if self.llm is None:
                resolved_llm = resolve_llm_path(
                    _fallback_gguf_for(self.llm_model_path), log=log)
                if not resolved_llm:
                    log("No LLM available; using keyword labels only.")
                    self.llm = None
                else:
                    try:
                        from llama_cpp import Llama
                        from src import mlx_backend as mlxb
                        # -1 on Apple Silicon offloads every layer to the GPU.
                        # Harmless on a CPU-only build (llama.cpp just keeps them
                        # on the CPU), and run.sh installs the Metal wheel there.
                        n_gpu = mlxb.llama_n_gpu_layers(
                            self._llm_settings.get("n_gpu_layers"))
                        if n_gpu:
                            log("Using Apple Metal GPU offload for the LLM.")
                        self.llm = Llama(
                            model_path=resolved_llm,
                            n_gpu_layers=n_gpu,
                            n_ctx=self._llm_settings.get("n_ctx", 2048),
                            n_batch=self._llm_settings.get("n_batch", 512),
                            n_threads=os.cpu_count(),  # use all CPU cores for token generation
                            verbose=False,
                            use_mmap=True
                        )
                    except Exception as e:
                        log(f"LLM Error: {e}. Using keywords only.")
                        self.llm = None

        # 2. Load Embedding Model
        log(f"Step 3/7: Loading Embeddings ({self.embedding_model_name.split('(')[0].strip()})...", 0.2)

        # Determine prefix and build the docs we'll embed first (INT8 calib uses them).
        model_path = EMBEDDING_MODELS.get(self.embedding_model_name, self.embedding_model_name)
        prefix = MODEL_PREFIXES.get(model_path, "")
        if prefix:
            log(f"Applying prefix for model: '{prefix}'", 0.22)
            docs_for_embedding = [prefix + doc for doc in processed_docs]
        else:
            docs_for_embedding = processed_docs

        # Resolve backend (PyTorch / OpenVINO INT8) and load.
        embedding_model, _backend, _precision = get_embedding_model_resolved(
            self.embedding_model_name, acceleration=self._acceleration,
            calib_docs=docs_for_embedding, log=log)

        # --- Check embedding cache (key includes backend so PyTorch/INT8 don't collide) ---
        cache_key = None
        cached_embeddings = None
        if self._cache_enabled:
            cache_key = _embedding_cache_key(docs_for_embedding, self.embedding_model_name, _backend, _precision)
            cached_embeddings = _load_cached_embeddings(cache_key, self._cache_dir)

        # Pre-compute embeddings with progress tracking
        total_docs = len(docs_for_embedding)
        batch_size = self._embedding_settings.get("batch_size", 32)
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        # Initial time estimate
        est_total, est_breakdown = estimate_processing_time(total_docs, self.embedding_model_name)
        if cached_embeddings is not None:
            log("Step 4/7: Loaded embeddings from cache!", 0.35)
        else:
            log(f"Step 4/7: Generating embeddings... (Est. {format_time(est_breakdown['embedding'])})", 0.25)
        
        if cached_embeddings is not None:
            embeddings = cached_embeddings
        else:
            # Track actual speed for live updates
            embedding_start_time = time.time()
            docs_processed = 0
            actual_docs_per_sec = est_breakdown['docs_per_sec']  # Start with estimate
            
            # Encode with progress updates and live time remaining
            all_embeddings = []
            for i in range(0, total_docs, batch_size):
                _ck()
                batch_start = time.time()
                batch = docs_for_embedding[i:i + batch_size]
                batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
                batch_time = time.time() - batch_start
                
                docs_processed += len(batch)
                batch_idx = i // batch_size
                
                # Calculate actual speed (weighted average with previous estimate)
                if batch_time > 0:
                    current_speed = len(batch) / batch_time
                    # Smooth the speed estimate (70% current, 30% previous)
                    actual_docs_per_sec = 0.7 * current_speed + 0.3 * actual_docs_per_sec
                
                # Calculate remaining time
                docs_remaining = total_docs - docs_processed
                time_remaining = docs_remaining / actual_docs_per_sec if actual_docs_per_sec > 0 else 0
                
                # Update progress
                progress = 0.25 + (0.10 * (batch_idx + 1) / total_batches)
                percent_done = int((batch_idx + 1) / total_batches * 100)
                
                # Update every 5 batches or at start/end to avoid too many updates
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    log(f"Embedding: {percent_done}% ({docs_processed:,}/{total_docs:,}) - {format_time(time_remaining)} remaining", progress)
            
            embedding_total_time = time.time() - embedding_start_time
            embeddings = _np.vstack(all_embeddings)
            # vstack copies, so the per-batch list is now redundant. Holding both kept
            # ~2x the embedding matrix alive through UMAP/HDBSCAN/BERTopic — the stages
            # that need the memory most.
            del all_embeddings
            log(f"Embeddings complete: {len(embeddings):,} docs in {format_time(embedding_total_time)} ({int(total_docs/embedding_total_time)} docs/sec)", 0.35)

            # Save to cache
            if self._cache_enabled and cache_key:
                try:
                    _save_embeddings_cache(cache_key, embeddings, self._cache_dir)
                    logger.debug("Embeddings saved to cache.")
                except Exception as e:
                    logger.warning(f"Could not cache embeddings: {e}")

        # Give the caller a chance to pick a different min_cluster_size before UMAP runs.
        # The callback receives the embeddings so it can run a fast HDBSCAN sweep without
        # re-embedding. It may return either an int (the chosen size) or a
        # (size, reduced_embeddings) tuple — the sweep runs the same UMAP, with the same
        # transform procedure, so handing that projection back skips a duplicate pass.
        #
        # Checked BEFORE invoking the hook: if a stop is already pending there is nothing
        # to decide, and asking anyway would put a modal dialog on screen for a run that
        # is about to end (leaving it orphaned once the cancellation lands).
        precomputed_reduced = None
        _ck()
        if on_embeddings_ready is not None:
            chosen = None
            try:
                outcome = on_embeddings_ready(embeddings, effective_min_cluster)
                # Unpacking lives inside the try so a malformed return can't kill the
                # run — the documented contract is that anything unusable is ignored.
                if isinstance(outcome, tuple):
                    chosen, precomputed_reduced = outcome
                else:
                    chosen = outcome
            except ClusteringCancelled:
                raise                      # a stop request must not be downgraded
            except Exception as e:
                logger.warning(f"on_embeddings_ready gave nothing usable; "
                               f"keeping {effective_min_cluster}: {e}")
                chosen, precomputed_reduced = None, None
            if isinstance(chosen, (int, _np.integer)) and int(chosen) >= 2:
                effective_min_cluster = int(chosen)
                log(f"Min Cluster Size for this run: {effective_min_cluster}")
        _ck()                              # the hook may have taken a long time

        # 3. Setup and run clustering components separately for progress tracking
        _ck()
        log("Step 5/7: Clustering documents...", 0.4)

        # --- UMAP Dimensionality Reduction (Optimized for large datasets) ---
        n_samples = len(embeddings)
        use_sampling = n_samples > 50000  # Use sampling for large datasets

        # len() is guarded: a caller could hand back something unsized, and the contract
        # is that anything unusable falls back to reducing here rather than failing.
        try:
            reuse_projection = (precomputed_reduced is not None
                                and len(precomputed_reduced) == n_samples)
        except TypeError:
            reuse_projection = False
        if not reuse_projection and precomputed_reduced is not None:
            # Say so: otherwise the cluster counts the user picked from silently stop
            # applying, because this reduction is a different (re-optimised) projection.
            log("  → The size sweep's projection was unusable; reducing again. The "
                "cluster counts shown in the picker may not hold exactly.")

        if reuse_projection:
            # The sweep already reduced these exact embeddings with these exact UMAP
            # params and the same transform procedure; redoing it costs the same minutes
            # for the same answer.
            reduced_embeddings = precomputed_reduced
            log("  → Reusing the UMAP projection from the size sweep.", 0.52)
        else:
            if use_sampling:
                sample_size = min(50000, n_samples)  # Train on 50K samples max
                log(f"  → Large dataset detected ({n_samples:,} docs). Using optimized UMAP...", 0.42)
                log(f"  → Training UMAP on {sample_size:,} sample documents...", 0.43)

                # Random sample for training
                sample_indices = _sample_indices(n_samples, sample_size)
                sample_embeddings = embeddings[sample_indices]
            else:
                log("  → Running UMAP (reducing dimensions)...", 0.42)

            umap_start = time.time()

            # Adaptive parallelism: parallel on large data (or when forced), single-core
            # + reproducible otherwise. Parallel UMAP must drop random_state.
            umap_rs, umap_jobs, umap_label = _umap_parallel(umap_parallel_mode, n_samples)
            # Common to both branches, so it must sit above the sampling branch's 0.43.
            log(f"  → UMAP: {umap_label}.", 0.44)

            # Optimized UMAP settings for large datasets
            umap_model = _UMAP(
                n_neighbors=min(10, umap_n_neighbors) if use_sampling else umap_n_neighbors,
                n_components=umap_n_components,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=umap_rs,
                low_memory=True,
                n_jobs=umap_jobs,
                verbose=False
            )

            if use_sampling:
                # Fit on sample, transform all
                umap_model.fit(sample_embeddings)
                umap_fit_time = time.time() - umap_start
                log(f"  → UMAP training complete in {format_time(umap_fit_time)}. Transforming all docs...", 0.48)

                # Batches above the threshold to avoid a huge single transform.
                transform_start = time.time()
                reduced_embeddings = _umap_transform_all(
                    umap_model, embeddings, log=log,
                    progress_from=0.48, progress_to=0.52)

                transform_time = time.time() - transform_start
                total_umap_time = time.time() - umap_start
                log(f"  → UMAP complete in {format_time(total_umap_time)} (train: {format_time(umap_fit_time)}, transform: {format_time(transform_time)})", 0.52)
            else:
                reduced_embeddings = umap_model.fit_transform(embeddings)
                umap_time = time.time() - umap_start
                log(f"  → UMAP complete in {format_time(umap_time)}", 0.52)

        # --- HDBSCAN Clustering (Optimized) ---
        log("  → Running HDBSCAN (finding clusters)...", 0.54)
        hdbscan_start = time.time()
        
        # Optimized HDBSCAN settings
        hdbscan_kwargs = dict(
            min_cluster_size=effective_min_cluster, 
            metric='euclidean', 
            cluster_selection_method=hdbscan_method,
            cluster_selection_epsilon=hdbscan_epsilon,
            prediction_data=False,
            core_dist_n_jobs=-1,  # all CPU cores; no reproducibility cost
            algorithm='best',
        )
        if hdbscan_min_samples is not None:
            hdbscan_kwargs["min_samples"] = hdbscan_min_samples
        hdbscan_model = _HDBSCAN(**hdbscan_kwargs)
        
        clusters = hdbscan_model.fit_predict(reduced_embeddings)
        hdbscan_time = time.time() - hdbscan_start
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        log(f"  → HDBSCAN complete in {format_time(hdbscan_time)} - Found {num_clusters} clusters", 0.60)
        
        # --- Build BERTopic model with pre-computed components ---
        log("  → Extracting topic keywords...", 0.62)

        vectorizer_model = _CountVectorizer(
            stop_words=get_stopwords(self._stopword_settings),
            ngram_range=(1, 2),
            min_df=2 if n_samples > 50000 else 1,
            max_df=0.95 if n_samples > 50000 else 1.0
        )

        ctfidf_model = _ClassTfidfTransformer(reduce_frequent_words=True)

        # Create wrappers with pre-computed results to prevent
        # BERTopic from re-running UMAP and HDBSCAN
        precomputed_umap = _PrecomputedUMAP(reduced_embeddings)
        precomputed_hdbscan = _PrecomputedHDBSCAN(clusters)

        # Create BERTopic with pre-computed UMAP and HDBSCAN results
        self.topic_model = _BERTopic(
            embedding_model=embedding_model,
            umap_model=precomputed_umap,
            hdbscan_model=precomputed_hdbscan,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            verbose=False
        )

        # Use fit_transform with pre-computed embeddings
        log("  → Building topic representations...", 0.64)
        topics, probs = self.topic_model.fit_transform(docs_for_embedding, embeddings)
        
        log(f"  → Topic modeling complete! Found {num_clusters} topics.", 0.68)

        # One pass for every cluster-size question asked below (repetitiveness, label
        # harmonisation). Keys stay whatever `topics` holds — numpy ints hash equal to
        # their Python counterparts, so lookups by either work.
        self._topic_sizes = Counter(topics)
        
        # Map topics to original docs
        topic_to_original_docs = {}
        for idx, topic_id in enumerate(topics):
            if topic_id not in topic_to_original_docs:
                topic_to_original_docs[topic_id] = []
            if len(topic_to_original_docs[topic_id]) < 5:
                topic_to_original_docs[topic_id].append(original_docs[idx])

        # Intermediate save
        if intermediate_callback:
            log("Saving intermediate results...", 0.69)
            topic_info_df = self.topic_model.get_topic_info()
            topic_map_kw = {}
            for topic_id in set(topics):
                if topic_id == -1:
                    topic_map_kw[topic_id] = "Non-Repetitive"
                else:
                    try:
                        name = topic_info_df[topic_info_df['Topic'] == topic_id]['Name'].values[0]
                        topic_map_kw[topic_id] = name
                    except (IndexError, KeyError):
                        topic_map_kw[topic_id] = f"Topic_{topic_id}"
            intermediate_callback(topics, topic_map_kw)

        # --- LEVEL 1: SUBCATEGORIES ---
        log("Step 6/7: Naming Subcategories...", 0.70)
        topic_info = self.topic_model.get_topic_info()
        subcategory_map = {-1: "Non-Repetitive"}
        topic_keywords = {}  # topic_id -> BERTopic keywords, reused for category naming

        valid_topics = topic_info[topic_info['Topic'] != -1]
        total_topics = len(valid_topics)
        
        for i, (_, row) in enumerate(valid_topics.iterrows()):
            _ck()
            topic_id = row['Topic']
            
            # Get keywords safely
            try:
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words:
                    keywords = [w[0] for w in topic_words[:6] if isinstance(w, tuple) and len(w) > 0]
                else:
                    keywords = ["general", "issue"]
            except (IndexError, KeyError, TypeError):
                keywords = ["general", "issue"]

            topic_keywords[topic_id] = keywords
            rep_docs = topic_to_original_docs.get(topic_id, [])

            label = self._generate_subcategory_label(keywords, rep_docs, topic_id, topics)
            subcategory_map[topic_id] = label if label else self._keywords_to_label(keywords) or "General Issue"
            
            if i % 5 == 0 and total_topics > 0:
                progress = 0.70 + (0.10 * i / total_topics)
                log(f"Naming Subcategories ({i+1}/{total_topics})...", progress)

        # --- LEVEL 2: CATEGORIES ---
        # Classify each subcategory directly into a standard category. We used to
        # K-Means the topic embeddings into ~10 macro-groups and stamp one label onto
        # every subcategory in a group, but a heterogeneous group then forced a single
        # wrong label onto unrelated subcategories (e.g. PowerBI / ADF / batch tickets
        # all becoming "Email & Collaboration"). Classifying each subcategory on its own
        # lets its descriptive name drive the result while still rolling everything up
        # into the fixed STANDARD_CATEGORIES.
        log("Step 7/7: Grouping into Categories...", 0.8)
        macro_map = {-1: "Non-Repetitive"}

        real_topics = sorted(set(t for t in topics if t != -1))
        # Resolution-clustering passes skip category labeling: emptying real_topics
        # makes both loops below no-ops, so macro_map stays {-1: "Non-Repetitive"}.
        if not label_categories:
            real_topics = []
        total_cats = len(real_topics)

        for i, tid in enumerate(real_topics):
            _ck()
            subcat = subcategory_map.get(tid, "")
            keywords = topic_keywords.get(tid, [])
            # Feed the subcategory label into matching too — it is the single most
            # informative signal (e.g. "Powerbi Refresh Failure") and lets the
            # keyword/template fast-path resolve the right category on its own.
            if subcat and subcat != "Non-Repetitive":
                match_keywords = [subcat] + keywords
                subcats = [subcat]
            else:
                match_keywords = keywords
                subcats = []
            rep_docs = topic_to_original_docs.get(tid, [])

            label = self._generate_category_label(subcats, match_keywords, rep_docs)
            macro_map[tid] = label if label else "General IT Support"

            if i % 5 == 0 and total_cats > 0:
                progress = 0.8 + (0.18 * i / total_cats)
                log(f"Naming Categories ({i+1}/{total_cats})...", progress)

        # Harmonize duplicate labels. HDBSCAN can split one topic into several
        # clusters that the namer labels identically (e.g. two "Sap Password Reset"
        # clusters), and per-cluster matching can then file the same label under
        # different parents — which looks wrong in the pivot. For any label shared by
        # more than one cluster, re-classify the label on its own (the canonical
        # identity, free of one cluster's incidental keywords) and apply that single
        # category to all clusters carrying it. Classifying the label directly — vs a
        # raw ticket-count vote — resolves ties the right way (e.g. "Sap Password
        # Reset" → Access & Authorization, not whichever cluster happened to be larger).
        label_to_topics = {}
        for tid in real_topics:
            label = subcategory_map.get(tid, "")
            if label and label != "Non-Repetitive":
                label_to_topics.setdefault(label, []).append(tid)

        for label, tids in label_to_topics.items():
            if len(tids) < 2:
                continue
            if len({macro_map.get(tid) for tid in tids}) <= 1:
                continue  # already consistent
            canonical = self._generate_category_label([label], [label], [])
            if not canonical:
                # Fall back to the majority category by ticket count.
                weight = Counter()
                sizes = Counter(t for t in topics if t != -1)
                for tid in tids:
                    weight[macro_map.get(tid)] += sizes.get(tid, 0)
                canonical = weight.most_common(1)[0][0]
            for tid in tids:
                macro_map[tid] = canonical

        # Collapse category spellings that are token-equivalent (differ only by
        # word order, plural or punctuation) onto one canonical spelling, so
        # per-cluster naming variance can't leave near-duplicate categories in the
        # output. Prefer a standard category name; otherwise the spelling that
        # covers the most tickets.
        cluster_sizes = Counter(t for t in topics if t != -1)
        sig_groups = {}   # signature -> Counter({label: ticket_count})
        for tid in real_topics:
            lbl = macro_map.get(tid)
            if not lbl or lbl == "Non-Repetitive":
                continue
            sig = _cat_signature(lbl)
            if not sig:
                continue
            sig_groups.setdefault(sig, Counter())[lbl] += cluster_sizes.get(tid, 0)
        canonical_for_sig = {}
        for sig, counts in sig_groups.items():
            if len(counts) < 2:
                continue  # only one spelling — nothing to merge
            std = next((c for c in counts if c in STANDARD_CATEGORIES), None)
            canonical_for_sig[sig] = std or counts.most_common(1)[0][0]
        if canonical_for_sig:
            for tid in real_topics:
                lbl = macro_map.get(tid)
                sig = _cat_signature(lbl) if lbl else None
                if sig in canonical_for_sig:
                    macro_map[tid] = canonical_for_sig[sig]
            # Drop spellings that no longer survive so the summary log stays honest.
            surviving = set(macro_map.values())
            self.discovered_categories = {
                c for c in self.discovered_categories if c in surviving
            }

        if self.discovered_categories:
            log(f"New categories discovered: {', '.join(sorted(self.discovered_categories))}", 0.98)
        
        # Store results for downstream analysis modules
        self._last_topics = topics
        self._last_subcategory_map = subcategory_map
        self._last_macro_map = macro_map
        self._last_original_docs = original_docs
        self._last_topic_to_docs = topic_to_original_docs

        log("Done!", 1.0)
        return topics, subcategory_map, macro_map

    def get_cluster_data(self):
        """Return structured cluster data for downstream analysis modules.

        Must be called after run(). Returns a dict keyed by topic_id:
            {topic_id: {keywords, subcategory, category, sample_docs,
                        sample_doc_indices, topic_embedding}}
        """
        if not hasattr(self, "_last_topics") or self.topic_model is None:
            return {}

        _lazy_import_ml()
        cluster_data = {}
        topics = self._last_topics
        sub_map = self._last_subcategory_map
        macro_map = self._last_macro_map
        original_docs = self._last_original_docs

        # Build index of docs per topic
        topic_doc_indices = {}
        for idx, tid in enumerate(topics):
            if tid not in topic_doc_indices:
                topic_doc_indices[tid] = []
            topic_doc_indices[tid].append(idx)

        # Topic embeddings lookup
        all_topics_sorted = sorted(self.topic_model.get_topics().keys())
        topic_to_embed_idx = {t: i for i, t in enumerate(all_topics_sorted)}

        for tid in set(topics):
            # Keywords
            try:
                topic_words = self.topic_model.get_topic(tid)
                keywords = [w[0] for w in topic_words[:10] if isinstance(w, tuple)] if topic_words else []
            except Exception:
                keywords = []

            # Sample docs
            doc_indices = topic_doc_indices.get(tid, [])
            sample_indices = doc_indices[:5]
            sample_docs = [original_docs[i] for i in sample_indices]

            # Topic embedding
            embed_idx = topic_to_embed_idx.get(tid)
            topic_embedding = None
            if embed_idx is not None and embed_idx < len(self.topic_model.topic_embeddings_):
                topic_embedding = self.topic_model.topic_embeddings_[embed_idx]

            cluster_data[tid] = {
                "keywords": keywords,
                "subcategory": sub_map.get(tid, ""),
                "category": macro_map.get(tid, ""),
                "sample_docs": sample_docs,
                "sample_doc_indices": sample_indices,
                "topic_embedding": topic_embedding,
            }

        return cluster_data
