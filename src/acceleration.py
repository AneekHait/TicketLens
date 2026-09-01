"""Embedding-backend selection, and the experimental OpenVINO INT8 path.

``resolve_backend`` is the single funnel every acceleration request goes
through: PyTorch (default), OpenVINO INT8, Apple MPS, or MLX. The Apple Silicon
backends live in src/mlx_backend.py; only the routing decision is here.

This is the ONLY module that imports openvino / optimum-intel / nncf / datasets,
and every such import is lazy and guarded — so the app runs normally when the
OpenVINO stack isn't installed. The clustering pipeline calls into here through
a small, never-raising public surface; any failure falls back to PyTorch.

INT8 quantization (~3.6-4.2x faster embeddings on CPUs with VNNI) is built once
per model from a calibration sample of the user's own ticket text (fully offline)
and cached under models/embedding/_ov/.
"""
import hashlib
import os

from src.config import ACCEL_COMPATIBLE_MODELS, MLX_COMPATIBLE_MODELS
from src.logger import get_logger

logger = get_logger()

_EMBEDDING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "embedding")

# Cached availability probe (import is relatively expensive; only do it once).
_ov_available = None


def openvino_available():
    """True if the OpenVINO + optimum-intel stack can be imported. Never raises."""
    global _ov_available
    if _ov_available is None:
        try:
            import openvino  # noqa: F401
            import optimum.intel  # noqa: F401
            _ov_available = True
        except Exception:
            _ov_available = False
    return _ov_available


def model_supports_acceleration(hf_path):
    """True if this HF model path works turnkey with the OpenVINO ST backend.
    BERT-style encoders + Qwen3; EmbeddingGemma (Gemma3) is excluded."""
    return hf_path in ACCEL_COMPATIBLE_MODELS


def model_supports_mlx(hf_path):
    """True if this HF model path has a known-good mlx-embeddings conversion.

    Deliberately a small allowlist rather than a try-it-and-see: mlx-embeddings
    reimplements pooling, and the curated models differ in whether they
    L2-normalise, so an unlisted model is far more likely to produce quietly
    different vectors than an outright error."""
    return hf_path in MLX_COMPATIBLE_MODELS


def resolve_backend(acceleration, hf_path):
    """Central, graceful funnel mapping a requested acceleration to what will
    actually run. Returns (backend, precision, reason).

    backend is "pytorch", "openvino", "mps" or "mlx"; reason is a human note
    when the request was downgraded (None otherwise). This is the backstop that
    forces PyTorch for incompatible models / missing deps / the wrong hardware,
    regardless of what was persisted in user_settings.json — a config file that
    travelled from an Apple Silicon Mac to a Windows box must not select MLX.
    """
    if acceleration in (None, "", "pytorch"):
        return "pytorch", None, None
    if acceleration == "openvino_int8_cpu":
        if not openvino_available():
            return "pytorch", None, "OpenVINO not installed"
        if not model_supports_acceleration(hf_path):
            return "pytorch", None, "model not OpenVINO-compatible (using PyTorch)"
        return "openvino", "int8", None
    if acceleration in ("mps", "mlx"):
        from src import mlx_backend as mlx
        if not mlx.is_apple_silicon():
            return "pytorch", None, f"'{acceleration}' needs Apple Silicon (using PyTorch)"
        if acceleration == "mps":
            if not mlx.mps_available():
                return "pytorch", None, "PyTorch MPS not available (using PyTorch CPU)"
            return "mps", None, None
        # MLX embeddings are the highest-risk backend in the app: a wrong vector
        # does not raise, it silently reshapes every cluster. Two gates before it
        # is even attempted, and a numeric parity check at load time on top.
        if not mlx.mlx_embeddings_available():
            return "pytorch", None, "mlx-embeddings not installed (run install_mlx.sh)"
        if not model_supports_mlx(hf_path):
            return "pytorch", None, "model not MLX-verified (using PyTorch)"
        return "mlx", None, None
    return "pytorch", None, f"unknown acceleration '{acceleration}'"


def _sanitize(hf_path):
    return hf_path.replace("/", "_").replace("\\", "_")


def _accel_cache_dir(hf_path, precision="int8"):
    return os.path.join(_EMBEDDING_DIR, "_ov", f"{_sanitize(hf_path)}_{precision}")


def _calibration_signature(calib_docs):
    """sha256 over a capped sample of the calibration docs (informational; stored
    in a .calib sidecar so we can show what a cached model was calibrated on)."""
    h = hashlib.sha256()
    for d in (calib_docs or [])[:300]:
        h.update((d or "").encode("utf-8"))
    h.update(str(len(calib_docs or [])).encode("utf-8"))
    return h.hexdigest()


def _is_built(ov_dir):
    return os.path.isfile(os.path.join(ov_dir, "openvino_model.xml"))


def ensure_int8_model(hf_path, base_model_dir, calib_docs, log=None, force=False):
    """Build (or reuse) an INT8-quantized OpenVINO IR of the model backbone.

    Returns the directory containing the quantized IR, or None on failure.
    Quantization is offline: calibration data comes from the user's own ticket
    text. Cached per model; rebuilt only when missing (or force=True).
    """
    def _log(msg):
        logger.info(msg)
        if log:
            try:
                log(msg)
            except Exception:
                pass

    ov_dir = _accel_cache_dir(hf_path, "int8")
    if _is_built(ov_dir) and not force:
        _log(f"OpenVINO INT8 model ready (cached): {os.path.basename(ov_dir)}")
        return ov_dir

    if not calib_docs:
        _log("OpenVINO INT8: no calibration text available yet — using PyTorch for now.")
        return None

    try:
        from optimum.intel import (
            OVModelForFeatureExtraction, OVQuantizer, OVConfig, OVQuantizationConfig,
        )
        from transformers import AutoTokenizer
        from datasets import Dataset

        _log("OpenVINO INT8: building quantized model… (one-time, ~30s-2min)")
        sample = [d for d in calib_docs if d][:300]
        n = max(8, min(len(sample), 300))
        tok = AutoTokenizer.from_pretrained(base_model_dir)
        calib = Dataset.from_dict({"text": sample}).map(
            lambda ex: tok(ex["text"], padding="max_length", truncation=True, max_length=128),
            remove_columns=["text"],
        )
        # Standalone FP32 export -> quantize -> reload (the verified pattern that
        # reliably writes openvino_model.xml).
        ov_fp32 = OVModelForFeatureExtraction.from_pretrained(base_model_dir, export=True)
        os.makedirs(os.path.dirname(ov_dir), exist_ok=True)
        OVQuantizer.from_pretrained(ov_fp32).quantize(
            calibration_dataset=calib,
            save_directory=ov_dir,
            ov_config=OVConfig(quantization_config=OVQuantizationConfig(num_samples=n)),
        )
        if not _is_built(ov_dir):
            _log("OpenVINO INT8: quantization produced no IR — using PyTorch.")
            return None
        with open(os.path.join(ov_dir, ".calib"), "w", encoding="utf-8") as f:
            f.write(f"{_calibration_signature(calib_docs)}\nsamples={n}\nmodel={hf_path}\n")
        _log(f"OpenVINO INT8: build complete ({n} calibration samples).")
        return ov_dir
    except Exception as e:
        logger.warning(f"OpenVINO INT8 build failed ({e}); falling back to PyTorch.")
        if log:
            try:
                log(f"OpenVINO INT8 build failed: {e}. Using PyTorch.")
            except Exception:
                pass
        return None


def load_accelerated_model(hf_path, base_model_dir, calib_docs, log, st_class, force=False):
    """Return a SentenceTransformer whose backbone is the INT8 OpenVINO IR (with
    the model's normal pooling/dense pipeline intact), or None to fall back to
    PyTorch. `st_class` is the SentenceTransformer class (injected so this module
    never imports sentence-transformers itself)."""
    ov_dir = ensure_int8_model(hf_path, base_model_dir, calib_docs, log=log, force=force)
    if not ov_dir:
        return None
    try:
        from optimum.intel import OVModelForFeatureExtraction
        # Load the full ST pipeline on the OpenVINO backend (FP32 export), then
        # swap the transformer backbone for the quantized IR — keeps the model's
        # correct pooling/dense/normalize while running INT8 inference.
        # SECURITY: trust_remote_code runs custom code from the model repo; safe
        # for the curated EMBEDDING_MODELS catalog (see clustering.get_embedding
        # note). Gate to the known-safe set if custom repos become loadable here.
        model = st_class(base_model_dir, backend="openvino", device="cpu", trust_remote_code=True)
        model[0].model = OVModelForFeatureExtraction.from_pretrained(ov_dir)  # auto_model -> self.model
        return model
    except Exception as e:
        logger.warning(f"OpenVINO INT8 load failed ({e}); falling back to PyTorch.")
        return None
