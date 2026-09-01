"""Apple Silicon acceleration: MLX, plus the Metal/MPS paths that need no extra deps.

This is the ONLY module that imports mlx / mlx_lm / mlx_embeddings, and every
such import is lazy and guarded — so the app runs normally on Windows, on Linux,
on Intel Macs, and on an Apple Silicon machine where MLX was never installed.
The public surface never raises; every failure returns None or False and the
caller falls back to the llama.cpp / PyTorch defaults.

Three separate things live here, in increasing order of risk:

1. ``mps_available()`` — PyTorch's Metal backend. No extra dependency; torch
   already ships it. Used to run the embedding model on the GPU.
2. ``load_llm()`` — mlx-lm as an alternative to llama.cpp for cluster labelling.
   Low risk: a bad label is visible, and every call goes through the same
   ``create_chat_completion`` contract the rest of the app already uses.
3. ``load_embedding_model()`` — mlx-embeddings. HIGHEST risk, and gated hardest.
   A wrong embedding does not raise; it silently produces different clusters.
   See the parity check in ``_verify_parity`` for why this path refuses to run
   unless it can prove it matches PyTorch numerically.
"""
import os
import platform
import sys

from src.logger import get_logger

logger = get_logger()

# Cached probes — these imports are expensive, so only pay for them once.
_mlx_lm_available = None
_mlx_emb_available = None
_mps_available = None

# How close an MLX embedding must be to the PyTorch reference to be used at all.
# Cosine guards direction; the norm ratio guards magnitude. Both matter: the
# curated embedding models do NOT all L2-normalise (gte-base-en-v1.5's
# modules.json is Transformer + Pooling with no Normalize step), and UMAP/HDBSCAN
# run on euclidean distance — so a backend that silently normalises when the
# reference does not would shift every cluster boundary while looking perfect
# to a cosine-only check.
_PARITY_MIN_COSINE = 0.999
_PARITY_NORM_TOLERANCE = 0.02  # |1 - mlx_norm/torch_norm|
_PARITY_SAMPLE = 8


# ---------------------------------------------------------------------------
# Platform / availability probes
# ---------------------------------------------------------------------------
def is_apple_silicon():
    """True on an arm64 Mac. MLX and Metal exist nowhere else."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mps_available():
    """True if PyTorch can use the Apple GPU via Metal Performance Shaders.

    Needs no extra install: the pinned torch (pulled in by sentence-transformers)
    already ships MPS on macOS.
    """
    global _mps_available
    if _mps_available is None:
        _mps_available = False
        if is_apple_silicon():
            try:
                import torch
                _mps_available = bool(torch.backends.mps.is_available())
            except Exception:
                _mps_available = False
    return _mps_available


def mlx_lm_available():
    """True if mlx-lm can be imported. Never raises."""
    global _mlx_lm_available
    if _mlx_lm_available is None:
        _mlx_lm_available = False
        if is_apple_silicon():
            try:
                import mlx_lm  # noqa: F401
                _mlx_lm_available = True
            except Exception:
                _mlx_lm_available = False
    return _mlx_lm_available


def mlx_embeddings_available():
    """True if mlx-embeddings can be imported. Never raises."""
    global _mlx_emb_available
    if _mlx_emb_available is None:
        _mlx_emb_available = False
        if is_apple_silicon():
            try:
                import mlx_embeddings  # noqa: F401
                _mlx_emb_available = True
            except Exception:
                _mlx_emb_available = False
    return _mlx_emb_available


def describe():
    """One-line human summary for the About dialog / logs. Never raises."""
    if not is_apple_silicon():
        return "Apple Silicon acceleration: not applicable on this platform."
    bits = []
    bits.append("Metal/MPS: yes" if mps_available() else "Metal/MPS: no")
    bits.append("mlx-lm: yes" if mlx_lm_available() else "mlx-lm: no")
    bits.append("mlx-embeddings: yes" if mlx_embeddings_available() else "mlx-embeddings: no")
    return "Apple Silicon — " + ", ".join(bits)


def llama_n_gpu_layers(requested=None):
    """How many llama.cpp layers to offload to the GPU.

    -1 (all) on Apple Silicon, 0 elsewhere. Note this only does anything if
    llama-cpp-python was built with Metal: the whl/cpu index is CPU-only, so
    run.sh installs from whl/metal on arm64 Macs. Offloading is harmless when
    the build has no Metal support — llama.cpp just keeps the layers on CPU.
    """
    if requested is not None:
        return int(requested)
    return -1 if is_apple_silicon() else 0


# ---------------------------------------------------------------------------
# LLM: mlx-lm behind the llama-cpp create_chat_completion contract
# ---------------------------------------------------------------------------
def _apply_stop(text, stop):
    """Truncate at the first stop sequence, mirroring llama.cpp's `stop`.

    mlx-lm has no equivalent parameter, and the call sites rely on this: the
    label prompts pass stop=["\\n\\n"] to cut the model off after the label
    instead of letting it ramble into an explanation.
    """
    if not stop:
        return text
    cut = len(text)
    for s in stop:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class MLXChatAdapter:
    """Exposes mlx-lm through the exact surface llama-cpp's Llama provides.

    Every LLM caller in src/ uses ``create_chat_completion(messages=...)`` and
    reads ``["choices"][0]["message"]["content"]``, so matching that shape is
    the whole job — nothing else in the app needs to know which engine ran.

    The prompt is rendered with the tokenizer's OWN chat template. That is not
    incidental: hardcoding prompt markup is what previously mislabelled Gemma,
    and it is why the rest of the codebase routes everything through
    create_chat_completion in the first place.
    """

    def __init__(self, model, tokenizer, repo_id=""):
        self._model = model
        self._tokenizer = tokenizer
        self.repo_id = repo_id
        self._gen_kind = None  # resolved on first use

    # -- prompt rendering ---------------------------------------------------
    def _render(self, messages):
        tok = self._tokenizer
        apply_template = getattr(tok, "apply_chat_template", None)
        if apply_template is not None and getattr(tok, "chat_template", None):
            try:
                return apply_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"[mlx] chat template failed ({e}); using plain text.")
        # No embedded template. Concatenate rather than invent markup: a wrong
        # template is worse than none, because it looks like it worked.
        return "\n\n".join(m.get("content", "") for m in messages)

    # -- generation ---------------------------------------------------------
    def _generate(self, prompt, max_tokens, temperature):
        """Call mlx_lm.generate across its API revisions.

        mlx-lm moved sampling from a ``temp=`` argument to a ``sampler=``
        object. Both spellings are still in the wild, and this module cannot be
        exercised on the machine it was written on, so pick by introspection
        rather than by pinning a version and hoping.
        """
        import inspect

        from mlx_lm import generate as mlx_generate

        kwargs = {"max_tokens": int(max_tokens), "verbose": False}
        try:
            params = inspect.signature(mlx_generate).parameters
        except (TypeError, ValueError):
            params = {}

        if "sampler" in params:
            try:
                from mlx_lm.sample_utils import make_sampler
                kwargs["sampler"] = make_sampler(temp=float(temperature))
            except Exception:
                pass
        elif "temp" in params:
            kwargs["temp"] = float(temperature)
        elif "temperature" in params:
            kwargs["temperature"] = float(temperature)

        if "verbose" not in params:
            kwargs.pop("verbose", None)

        return mlx_generate(self._model, self._tokenizer, prompt=prompt, **kwargs)

    # -- the contract -------------------------------------------------------
    def create_chat_completion(self, messages, max_tokens=256, temperature=0.2,
                               stop=None, **_ignored):
        """Same signature and return shape as llama_cpp.Llama.create_chat_completion."""
        prompt = self._render(messages)
        text = self._generate(prompt, max_tokens, temperature)
        if not isinstance(text, str):
            text = str(text)
        text = _apply_stop(text, stop)
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    # A real Llama is also callable for raw completions. No caller in src/ uses
    # it, but conftest's FakeLLM keeps the form, so stay faithful to the API.
    def __call__(self, prompt, max_tokens=256, temperature=0.2, stop=None, **_ignored):
        text = _apply_stop(self._generate(prompt, max_tokens, temperature), stop)
        return {"choices": [{"text": text}]}


def load_llm(repo_id, log=None, cache_dir=None):
    """Load an MLX LLM and wrap it in the llama-cpp contract.

    Returns an MLXChatAdapter, or None to fall back to llama.cpp. `repo_id` is
    an mlx-community HF repo (already-quantized MLX weights), not a GGUF path.
    """
    def _log(msg):
        logger.info(msg)
        if log:
            try:
                log(msg)
            except Exception:
                pass

    if not mlx_lm_available():
        return None
    try:
        from mlx_lm import load as mlx_load
        _log(f"MLX: loading {repo_id} (first run downloads the weights)...")
        kwargs = {}
        if cache_dir:
            # mlx-lm defers to huggingface_hub, which honours this env var. Set
            # it only for the load so we do not disturb the rest of the process.
            os.environ.setdefault("HF_HOME", cache_dir)
        model, tokenizer = mlx_load(repo_id, **kwargs)
        _log(f"MLX: {repo_id} ready (Apple GPU).")
        return MLXChatAdapter(model, tokenizer, repo_id=repo_id)
    except Exception as e:
        logger.warning(f"MLX LLM load failed ({e}); falling back to llama.cpp.")
        if log:
            try:
                log(f"MLX unavailable ({e}). Using llama.cpp instead.")
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Embeddings: mlx-embeddings, gated behind a numeric parity check
# ---------------------------------------------------------------------------
class MLXEmbedder:
    """Minimal SentenceTransformer stand-in: ``.encode(list) -> ndarray``.

    Only the surface clustering.py actually uses is implemented — two call
    sites, both ``encode(batch, show_progress_bar=False)``.
    """

    def __init__(self, model, processor, hf_path=""):
        self._model = model
        self._processor = processor
        self.hf_path = hf_path

    def encode(self, sentences, show_progress_bar=False, batch_size=32, **_ignored):
        import numpy as np

        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        if not sentences:
            return np.zeros((0, 0), dtype="float32")

        out = []
        for i in range(0, len(sentences), batch_size):
            out.append(self._encode_batch(sentences[i:i + batch_size]))
        return np.vstack(out).astype("float32")

    def _encode_batch(self, batch):
        import numpy as np

        from mlx_embeddings import generate as mlx_generate

        result = mlx_generate(self._model, self._processor, batch)
        # mlx-embeddings returns an object with text_embeds (pooled) and
        # last_hidden_state. Prefer the pooled output; accept a bare array too,
        # since the return type has moved between releases.
        vec = getattr(result, "text_embeds", None)
        if vec is None:
            vec = getattr(result, "pooler_output", None)
        if vec is None:
            vec = result
        return np.array(vec, dtype="float32")


def _verify_parity(mlx_model, reference_model, docs, log=None):
    """Prove the MLX embedder matches PyTorch before trusting it with real data.

    This exists because a wrong embedding backend has no visible failure mode:
    it does not raise, it just returns different numbers, and the user gets a
    plausible-looking set of wrong clusters. Both direction (cosine) and
    magnitude (norm ratio) are checked — the curated models do not all
    L2-normalise, and the pipeline clusters on euclidean distance, so a backend
    that normalises when the reference does not would pass a cosine-only check
    while moving every cluster boundary.

    Returns (ok, detail).
    """
    import numpy as np

    sample = [d for d in (docs or []) if d][:_PARITY_SAMPLE]
    if len(sample) < 2:
        return False, "not enough sample text to verify MLX output"

    try:
        a = np.asarray(mlx_model.encode(sample, show_progress_bar=False), dtype="float64")
        b = np.asarray(reference_model.encode(sample, show_progress_bar=False), dtype="float64")
    except Exception as e:
        return False, f"could not run the comparison ({e})"

    if a.shape != b.shape:
        return False, f"shape mismatch: MLX {a.shape} vs PyTorch {b.shape}"

    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if not np.all(np.isfinite(a)) or np.any(na == 0):
        return False, "MLX produced empty or non-finite vectors"

    cos = float(np.min(np.sum(a * b, axis=1) / (na * nb)))
    norm_err = float(np.max(np.abs(1.0 - (na / nb))))

    if cos < _PARITY_MIN_COSINE:
        return False, f"cosine {cos:.5f} < {_PARITY_MIN_COSINE} vs PyTorch"
    if norm_err > _PARITY_NORM_TOLERANCE:
        return False, (f"vector magnitudes differ by {norm_err:.1%} vs PyTorch "
                       f"(the model's own pooling/normalisation was not reproduced)")
    return True, f"cosine {cos:.5f}, magnitude within {norm_err:.2%}"


def load_embedding_model(hf_path, base_model_dir, calib_docs, log, reference_model=None):
    """Load an MLX embedding model, or None to stay on PyTorch.

    `reference_model` is a loaded PyTorch SentenceTransformer used to verify the
    MLX output numerically. Without one there is nothing to check against, so
    this refuses to run rather than risk silently different clusters.
    """
    def _log(msg):
        logger.info(msg)
        if log:
            try:
                log(msg)
            except Exception:
                pass

    if not mlx_embeddings_available():
        return None
    if reference_model is None:
        _log("MLX embeddings: no PyTorch reference to verify against — using PyTorch.")
        return None

    try:
        from mlx_embeddings import load as mlx_load
        _log("MLX embeddings: loading model...")
        model, processor = mlx_load(hf_path)
        candidate = MLXEmbedder(model, processor, hf_path=hf_path)
    except Exception as e:
        logger.warning(f"MLX embedding load failed ({e}); using PyTorch.")
        _log(f"MLX embeddings unavailable ({e}). Using PyTorch.")
        return None

    ok, detail = _verify_parity(candidate, reference_model, calib_docs, log=log)
    if not ok:
        _log(f"MLX embeddings REJECTED — {detail}. Using PyTorch (clusters unchanged).")
        logger.warning(f"MLX embedding parity check failed: {detail}")
        return None

    _log(f"MLX embeddings verified against PyTorch ({detail}).")
    return candidate
