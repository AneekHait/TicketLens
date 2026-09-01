"""
Configuration management for the ticket clustering tool.
Handles default settings and user overrides (diff-only, vs DEFAULTS).

Saving and reloading a *run* -- clusters, labels, analysis results -- is src/session.py.
This module only deals with settings; the docstring used to claim session persistence
that lived nowhere.
"""

import copy
import json
import os

# Curated LLM (cluster-labeling) models for the GUI dropdown. Kept here (not in
# clustering.py) so the GUI can read it at startup without importing the heavy
# clustering module (which pulls in sklearn). All repos are ungated GGUFs; they
# are downloaded into models/ on demand by clustering.resolve_llm_path().
# Keys MUST match the GUI dropdown labels exactly.
LLM_CUSTOM_SENTINEL = "Custom file…"
LLM_MODELS = {
    "gemma-4-E2B-it (Good · Fast — Recommended)": {
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "size_gb": 2.9,
    },
    "gemma-4-E4B-it (High · Moderate)": {
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "size_gb": 4.0,
    },
    # Phi-4-mini (3.8B, MIT). Verified ungated with an embedded tokenizer.chat_template,
    # which is what makes it safe here: every call goes through create_chat_completion,
    # so llama-cpp applies each model's own template instead of the hardcoded Phi-3
    # markup this code used to emit at every model regardless.
    "phi-4-mini-instruct (High · Moderate)": {
        "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
        "filename": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.5,
    },
    "qwen2.5-0.5b-instruct (Basic · Fastest)": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_gb": 0.4,
    },
    "gemma-4-12b-it (Top · Slow — needs ~8 GB free)": {
        "repo_id": "unsloth/gemma-4-12B-it-GGUF",
        "filename": "gemma-4-12b-it-Q4_K_M.gguf",
        "size_gb": 8.0,
    },
}

# Embedding models for the GUI dropdown (label -> HuggingFace path). Kept here so
# the GUI can read it at startup without importing clustering (which pulls in
# sklearn). Dict order = dropdown order; keep bge-base first (startup default).
# Keys MUST match clustering.MODEL_SPEED_ESTIMATES keys exactly.
EMBEDDING_MODELS = {
    "bge-base-en-v1.5 (High · Moderate — Recommended)": "BAAI/bge-base-en-v1.5",
    # Microsoft Harrier (2026, MIT): SOTA on multilingual MTEB v2. Decoder-only
    # (last-token pooling) => PyTorch-only, 32k context, no document prefix needed.
    # The 27B variant is GPU-only and intentionally omitted (this tool is CPU-only).
    "harrier-270m (Top · Slow · 32k ctx)": "microsoft/harrier-oss-v1-270m",
    "harrier-0.6b (Top · Slowest · 32k ctx)": "microsoft/harrier-oss-v1-0.6b",
    # Google's EmbeddingGemma, via Unsloth's ungated mirror (google/* is license-gated).
    "embeddinggemma-300m (Top · Slow)": "unsloth/embeddinggemma-300m",
    # Qwen3-Embedding: top small model, Apache-2.0, ungated. Larger/slower than Gemma.
    "qwen3-embedding-0.6b (Top · Slowest)": "Qwen/Qwen3-Embedding-0.6B",
    "bge-large-en-v1.5 (High · Slow)": "BAAI/bge-large-en-v1.5",
    # GTE v1.5 (Alibaba): transformer++ backbone (BERT+RoPE+GLU), 8192 context,
    # needs trust_remote_code (already set). Supersedes the 2023 thenlper/gte-base.
    "gte-large-en-v1.5 (High · Slow · 8k ctx)": "Alibaba-NLP/gte-large-en-v1.5",
    "gte-base-en-v1.5 (High · Moderate · 8k ctx)": "Alibaba-NLP/gte-base-en-v1.5",
    "bge-small-en-v1.5 (Good · Fast)": "BAAI/bge-small-en-v1.5",
    "gte-small (Good · Fast)": "thenlper/gte-small",
    "all-MiniLM-L6-v2 (Basic · Fastest)": "sentence-transformers/all-MiniLM-L6-v2",
}

# Embedding acceleration backends (experimental). Label -> internal value.
# PyTorch is the reproducible default; OpenVINO INT8 is ~3.6-4.2x faster on
# CPUs with VNNI but shifts exact cluster assignments slightly (opt-in).
ACCELERATION_OPTIONS = {
    "PyTorch (default · reproducible)": "pytorch",
    "OpenVINO INT8 (CPU · experimental)": "openvino_int8_cpu",
    # Apple Silicon only. Both are inert elsewhere: acceleration.resolve_backend
    # forces PyTorch when the hardware or the package is missing, so a
    # user_settings.json copied from a Mac cannot select them on Windows.
    "Apple Metal / MPS (Apple Silicon)": "mps",
    "MLX (Apple Silicon · experimental)": "mlx",
}

# HF paths that work turnkey with the OpenVINO sentence-transformers backend.
# BERT-style encoders + Qwen3. EmbeddingGemma (Gemma3) is EXCLUDED — its OV
# export needs position_ids that the feature-extraction wrapper doesn't pass,
# so it stays on PyTorch.
ACCEL_COMPATIBLE_MODELS = {
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    # NOTE: gte-*-en-v1.5 (custom transformer++ code) and Harrier (decoder-only)
    # are NOT listed — they run PyTorch-only. Their OV export is unverified.
}

# HF paths with a known-good mlx-embeddings conversion. Deliberately the small
# BERT-family subset: mlx-embeddings reimplements pooling, and these models
# differ in whether they L2-normalise (gte-base-en-v1.5's modules.json is
# Transformer + Pooling with NO Normalize), while clustering runs on euclidean
# distance. A model whose pooling is not reproduced exactly yields quietly
# different clusters rather than an error, so the allowlist is opt-in and the
# runtime parity check in mlx_backend.load_embedding_model gates it again.
MLX_COMPATIBLE_MODELS = {
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
}

# MLX LLMs for cluster labelling on Apple Silicon (mlx-community, pre-quantized).
# These are NOT GGUF: mlx-lm loads an HF repo of MLX weights, so they are listed
# separately from LLM_MODELS rather than mixed into it. Every one must ship an
# embedded chat template — MLXChatAdapter renders with the tokenizer's own
# template and never invents markup, same rule as the llama.cpp path.
MLX_LLM_MODELS = {
    "MLX · qwen2.5-3b-instruct (Good · Fast)": {
        "repo_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "size_gb": 1.7,
    },
    "MLX · phi-4-mini-instruct (High · Moderate)": {
        "repo_id": "mlx-community/Phi-4-mini-instruct-4bit",
        "size_gb": 2.2,
    },
    "MLX · qwen2.5-0.5b-instruct (Basic · Fastest)": {
        "repo_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "size_gb": 0.3,
    },
}

# Default configuration
DEFAULTS = {
    # Preprocessing toggles
    "preprocessing": {
        "remove_emails": True,
        "remove_urls": True,
        "remove_ticket_ids": True,
        "remove_timestamps": True,
        "remove_phone_numbers": True,
        "remove_ip_addresses": True,
        "remove_file_paths": True,
        "remove_special_chars": True,
        "remove_boilerplate": True,
        "min_token_length": 3,
        "custom_regex_patterns": [],
        "custom_boilerplate_patterns": [],
    },
    # Stopwords
    "stopwords": {
        "use_english_stopwords": True,
        "use_it_stopwords": True,
        "custom_stopwords": [],
    },
    # Clustering parameters
    "clustering": {
        "min_cluster_size": 15,
        "umap_n_neighbors": 15,
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "umap_metric": "euclidean",
        # UMAP threading: "auto" (parallel only on large data), "parallel" (all cores,
        # not reproducible), or "reproducible" (1 core, fixed seed).
        "umap_parallel_mode": "auto",
        "hdbscan_min_samples": None,  # None = defaults to min_cluster_size
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_cluster_selection_epsilon": 0.0,
        # Pause mid-run once embeddings exist and let the user pick a Min Cluster Size
        # from a scored sweep. Embedding doesn't depend on the size, so this is the
        # cheapest point to decide; uncheck on the Clustering page to run straight
        # through with min_cluster_size above.
        "pick_size_after_embedding": True,
    },
    # Embedding
    "embedding": {
        "model_name": "bge-base-en-v1.5 (High · Moderate — Recommended)",
        "batch_size": 32,
        # Inference backend: "pytorch" (default) or "openvino_int8_cpu" (experimental).
        "acceleration": "pytorch",
    },
    # LLM (used by TicketClusterer for cluster labelling)
    "llm": {
        "enabled": True,          # initialises the "Enable AI Naming" checkbox
        "n_ctx": 2048,            # context window for the labelling prompts
        "n_batch": 512,
        "temperature": 0.2,
        "max_tokens_subcategory": 32,   # room for "Source : detail" style labels
        "max_tokens_category": 15,
    },
    # Cache
    "cache": {
        "enabled": True,
        "directory": "cache",
    },
    # Metadata column mapping (user maps Excel columns to these roles)
    "metadata_mapping": {
        "priority_col": None,
        "resolution_time_col": None,
        "sla_status_col": None,
        "assignment_group_col": None,
        "created_date_col": None,
        "resolved_date_col": None,
        "category_col": None,
        "resolution_notes_col": None,
        # Automation-opportunity (disposition) signals. None = auto-detect by name.
        "effort_col": None,            # handling time (e.g. business_duration, seconds)
        "reopen_col": None,            # reopen_count
        "reassignment_col": None,      # reassignment_count
        "close_code_col": None,        # close_code / closure code
        "symptom_col": None,           # short_description (for RCA prompt context)
    },
    # Ticket quality audit settings
    "audit": {
        "min_description_length": 10,
        "vague_word_threshold": 0.5,
        "completeness_weight": 0.4,
        "categorization_weight": 0.3,
        "resolution_weight": 0.3,
    },
    # KBA article generation
    "kba": {
        "max_sample_tickets": 5,
        "max_keywords": 8,
    },
    # SOP generation
    "sop": {
        "max_clusters_per_sop": 10,
    },
    # Impact analysis
    "analysis": {
        "top_n_clusters": 10,
    },
    # Automation-opportunity analysis (Root Cause + Resolution + Disposition).
    # Clusters the resolution text and classifies each cluster Automate / Reimagine
    # / Eradicate / Retain.
    "disposition": {
        "min_cluster_size": 15,          # min tickets per resolution cluster
        "max_sample_tickets": 5,         # sample symptom/resolution pairs per LLM call
        "max_keywords": 10,
        "effort_seconds_per_hour": 3600, # business_duration is seconds; /3600 -> hours
    },
    # AI category audit (re-checks how each cluster is filed into a macro-category).
    # The GUI reads this section, so without it here CategoryAuditor always received
    # {} and these knobs were unreachable from config/user_settings.json.
    "category_audit": {
        "max_sample_tickets": 3,   # sample tickets shown to the model per cluster
        "max_keywords": 8,
        "temperature": 0.1,        # low: this is a pick-from-a-list classification
        "max_tokens": 16,          # a category name only
    },
}

# Substring patterns (case-insensitive) used to auto-detect the resolution-text
# columns to cluster on, across ServiceNow export naming styles (technical names,
# display names like "Comments and Work notes", inc_-prefixed names).
RESOLUTION_TEXT_PATTERNS = [
    "close_notes", "close notes", "comments and work notes", "work_notes",
    "work notes", "resolution_notes", "resolution notes", "resolution", "solution",
]

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_USER_CONFIG_PATH = os.path.join(_CONFIG_DIR, "user_settings.json")


def embedding_dir_complete(local_dir):
    """Return True only if *local_dir* holds a fully-downloaded SentenceTransformer model.

    A model is "complete" when it has ``modules.json``, a root ``config.json``, a
    weight file, and — critically — the ``config.json`` of every Pooling/Dense
    sub-module it declares. Partial downloads often fetch the large weights but
    skip these tiny subfolder configs; a missing ``1_Pooling/config.json`` then
    makes SentenceTransformer crash on load with ``Pooling.__init__() missing 1
    required positional argument: 'embedding_dimension'``. Such a directory must
    be treated as not-yet-downloaded so it is re-fetched. Pure stdlib (os/json)
    so the GUI can call it at startup without importing the heavy clustering
    module.
    """
    if not (os.path.isdir(local_dir) and os.listdir(local_dir)):
        return False
    modules_path = os.path.join(local_dir, "modules.json")
    if not (os.path.exists(modules_path)
            and os.path.exists(os.path.join(local_dir, "config.json"))):
        return False
    if not any(f.endswith((".safetensors", ".bin")) for f in os.listdir(local_dir)):
        return False
    try:
        with open(modules_path, encoding="utf-8") as f:
            modules = json.load(f)
    except (OSError, ValueError):
        return False
    for module in modules:
        sub_path = module.get("path") or ""
        module_type = module.get("type", "")
        if sub_path and ("Pooling" in module_type or "Dense" in module_type):
            if not os.path.exists(os.path.join(local_dir, sub_path, "config.json")):
                return False
    return True


def _deep_merge(base, override):
    """Merge override dict into base dict recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Load configuration: defaults merged with user overrides."""
    config = copy.deepcopy(DEFAULTS)

    if os.path.exists(_USER_CONFIG_PATH):
        try:
            with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config = _deep_merge(config, user_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load user config: {e}")

    return config


def save_config(config):
    """Save user configuration to disk (diff-only: just what differs from DEFAULTS)."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)

    # Only save values that differ from defaults
    diff = _compute_diff(DEFAULTS, config)
    # An empty diff must still be written. Skipping the write left the *previous*
    # override file in place, and load_config() merges it straight back — so "Reset to
    # defaults" (and reverting any single setting by hand) appeared to work until the
    # next launch, when the old value returned.
    with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2)


def _compute_diff(defaults, config):
    """Compute the difference between config and defaults (only changed values)."""
    diff = {}
    for key, value in config.items():
        if key not in defaults:
            diff[key] = value
        elif isinstance(value, dict) and isinstance(defaults.get(key), dict):
            sub_diff = _compute_diff(defaults[key], value)
            if sub_diff:
                diff[key] = sub_diff
        elif value != defaults.get(key):
            diff[key] = value
    return diff


def reset_config():
    """Reset to default configuration by removing user config file."""
    if os.path.exists(_USER_CONFIG_PATH):
        os.remove(_USER_CONFIG_PATH)
    return copy.deepcopy(DEFAULTS)
