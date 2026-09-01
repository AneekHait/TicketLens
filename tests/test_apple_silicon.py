"""Apple Silicon acceleration: MPS, Metal offload, and the MLX backends.

None of this can be executed on non-Apple hardware, and the CI matrix has no
arm64 macOS runner, so these tests cover the parts that are hardware-independent:
the routing decisions, the llama-cpp contract the MLX adapter has to satisfy, and
the parity gate that stops a wrong embedding backend from reaching real data.

The parity gate is the important one. Every other failure here is loud -- a model
does not load, a label comes out empty. A wrong embedding backend is silent: it
returns plausible numbers, and the user gets a plausible set of *different*
clusters with nothing to indicate anything went wrong.
"""
import numpy as np
import pytest

from src import mlx_backend as mlxb
from src.acceleration import resolve_backend
from src.config import (ACCELERATION_OPTIONS, EMBEDDING_MODELS, LLM_MODELS,
                        MLX_COMPATIBLE_MODELS, MLX_LLM_MODELS)


# ---------------------------------------------------------------------------
# Routing: nothing Apple-specific may engage on non-Apple hardware
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("acceleration", ["mps", "mlx"])
def test_apple_backends_downgrade_off_apple_silicon(monkeypatch, acceleration):
    """user_settings.json is portable. A config written on a Mac and opened on
    Windows must not select a backend that cannot exist there."""
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: False)
    backend, precision, reason = resolve_backend(acceleration, "BAAI/bge-base-en-v1.5")
    assert backend == "pytorch"
    assert precision is None
    assert "Apple Silicon" in reason


def test_mps_downgrades_when_torch_has_no_metal(monkeypatch):
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlxb, "mps_available", lambda: False)
    backend, _p, reason = resolve_backend("mps", "BAAI/bge-base-en-v1.5")
    assert backend == "pytorch"
    assert "MPS" in reason


def test_mps_is_selected_when_everything_lines_up(monkeypatch):
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlxb, "mps_available", lambda: True)
    assert resolve_backend("mps", "BAAI/bge-base-en-v1.5") == ("mps", None, None)


def test_mlx_downgrades_when_the_package_is_missing(monkeypatch):
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlxb, "mlx_embeddings_available", lambda: False)
    backend, _p, reason = resolve_backend("mlx", "BAAI/bge-base-en-v1.5")
    assert backend == "pytorch"
    assert "install_mlx" in reason


def test_mlx_downgrades_for_a_model_not_on_the_verified_list(monkeypatch):
    """Harrier is decoder-only and EmbeddingGemma is Gemma3; neither has a
    verified mlx-embeddings conversion, so both must stay on PyTorch."""
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlxb, "mlx_embeddings_available", lambda: True)
    for hf in ("microsoft/harrier-oss-v1-270m", "unsloth/embeddinggemma-300m"):
        backend, _p, reason = resolve_backend("mlx", hf)
        assert backend == "pytorch", f"{hf} should not run on MLX"
        assert "MLX-verified" in reason


def test_mlx_is_selected_when_everything_lines_up(monkeypatch):
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlxb, "mlx_embeddings_available", lambda: True)
    assert resolve_backend("mlx", "BAAI/bge-base-en-v1.5") == ("mlx", None, None)


def test_every_dropdown_acceleration_resolves_to_a_real_backend():
    """resolve_backend is called with whatever is in the dropdown, so no option
    may fall through to the unknown-acceleration branch."""
    for value in ACCELERATION_OPTIONS.values():
        backend, _p, reason = resolve_backend(value, "BAAI/bge-base-en-v1.5")
        assert backend in ("pytorch", "openvino", "mps", "mlx")
        assert not (reason or "").startswith("unknown acceleration")


# ---------------------------------------------------------------------------
# Metal offload for llama.cpp
# ---------------------------------------------------------------------------
def test_gpu_offload_is_requested_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    assert mlxb.llama_n_gpu_layers() == -1
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: False)
    assert mlxb.llama_n_gpu_layers() == 0


def test_an_explicit_n_gpu_layers_setting_wins(monkeypatch):
    """The value is user-overridable through user_settings.json, so an explicit
    setting must not be silently replaced by the platform default."""
    monkeypatch.setattr(mlxb, "is_apple_silicon", lambda: True)
    assert mlxb.llama_n_gpu_layers(0) == 0
    assert mlxb.llama_n_gpu_layers(12) == 12


# ---------------------------------------------------------------------------
# The MLX chat adapter must be indistinguishable from llama_cpp.Llama
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Stands in for an mlx-lm tokenizer that carries its own chat template."""

    chat_template = "{% for m in messages %}<{{m.role}}>{{m.content}}{% endfor %}"

    def __init__(self):
        self.seen = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen = messages
        return "TEMPLATED:" + "|".join(m["content"] for m in messages)


def _adapter(monkeypatch, reply="Password Reset", tok=None):
    tok = tok if tok is not None else _FakeTokenizer()
    a = mlxb.MLXChatAdapter(object(), tok, repo_id="mlx-community/test")
    monkeypatch.setattr(a, "_generate", lambda prompt, mt, temp: reply)
    return a, tok


def test_the_adapter_returns_the_exact_llama_cpp_response_shape(monkeypatch):
    """Six call sites read output['choices'][0]['message']['content']. If the
    shape differs by one key the failure is a KeyError deep inside labelling."""
    a, _ = _adapter(monkeypatch, reply="Password Reset")
    out = a.create_chat_completion(
        messages=[{"role": "user", "content": "name this cluster"}],
        max_tokens=20, temperature=0.2)
    assert out["choices"][0]["message"]["content"] == "Password Reset"


def test_the_adapter_renders_with_the_models_own_chat_template(monkeypatch):
    """Hardcoded prompt markup is what previously mislabelled Gemma. The adapter
    must defer to the tokenizer's template, never invent one."""
    tok = _FakeTokenizer()
    a = mlxb.MLXChatAdapter(object(), tok, repo_id="x")
    captured = {}
    monkeypatch.setattr(a, "_generate",
                        lambda prompt, mt, temp: captured.setdefault("p", prompt) or "ok")
    a.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
    assert captured["p"].startswith("TEMPLATED:")
    assert tok.seen == [{"role": "user", "content": "hello"}]


def test_the_adapter_does_not_invent_markup_when_there_is_no_template(monkeypatch):
    """A model with no embedded template gets plain concatenation. Guessing a
    template that looks right is worse than none, because it fails silently."""
    class _NoTemplate:
        chat_template = None

    a, _ = _adapter(monkeypatch, reply="ok", tok=_NoTemplate())
    captured = {}
    monkeypatch.setattr(a, "_generate",
                        lambda prompt, mt, temp: captured.setdefault("p", prompt) or "ok")
    a.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
    assert captured["p"] == "hello"
    for marker in ("<|", "[INST]", "<start_of_turn>", "###"):
        assert marker not in captured["p"]


def test_the_adapter_honours_stop_sequences(monkeypatch):
    """The label prompts pass stop=['\\n\\n'] to cut the model off after the
    label. mlx-lm has no stop parameter, so the adapter has to apply it."""
    a, _ = _adapter(monkeypatch, reply="VPN Drops\n\nHere is why that happens...")
    out = a.create_chat_completion(
        messages=[{"role": "user", "content": "x"}], stop=["\n\n"])
    assert out["choices"][0]["message"]["content"] == "VPN Drops"


def test_the_adapter_tolerates_extra_llama_cpp_kwargs(monkeypatch):
    """Callers pass whatever llama-cpp accepts; unknown kwargs must not raise."""
    a, _ = _adapter(monkeypatch, reply="ok")
    out = a.create_chat_completion(
        messages=[{"role": "user", "content": "x"}],
        max_tokens=10, temperature=0.1, top_p=0.9, repeat_penalty=1.1, seed=7)
    assert out["choices"][0]["message"]["content"] == "ok"


def test_stop_truncation_handles_the_awkward_cases():
    assert mlxb._apply_stop("abc", None) == "abc"
    assert mlxb._apply_stop("abc", []) == "abc"
    assert mlxb._apply_stop("abc", [""]) == "abc"          # empty stop is not "match at 0"
    assert mlxb._apply_stop("a\n\nb\n\nc", ["\n\n"]) == "a"  # first match wins
    assert mlxb._apply_stop("axxbyyc", ["yy", "xx"]) == "a"  # earliest of several


# ---------------------------------------------------------------------------
# The parity gate: the only thing standing between MLX and silently wrong clusters
# ---------------------------------------------------------------------------
class _StubEncoder:
    def __init__(self, matrix):
        self._m = np.asarray(matrix, dtype="float64")

    def encode(self, sentences, show_progress_bar=False, **kw):
        return self._m[:len(sentences)]


_DOCS = ["password reset", "vpn drops", "printer jam", "disk full"]


def test_parity_accepts_a_backend_that_matches_pytorch():
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.5], [0.2, 0.1, 9.0], [1.0, 1.0, 1.0]])
    ok, detail = _verify(ref * 1.0000001, ref)
    assert ok, detail


def test_parity_rejects_a_backend_that_normalises_when_the_reference_does_not():
    """The regression this whole gate exists for. bge/gte models here do NOT all
    L2-normalise, and clustering runs on euclidean distance -- so a backend that
    normalises has perfect cosine agreement and completely different clusters.
    A cosine-only check would wave this straight through."""
    ref = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0], [1.0, 0.0, 0.0], [2.0, 2.0, 1.0]])
    normalised = ref / np.linalg.norm(ref, axis=1, keepdims=True)

    cos = np.min(np.sum(ref * normalised, axis=1) /
                 (np.linalg.norm(ref, axis=1) * np.linalg.norm(normalised, axis=1)))
    assert cos > 0.9999, "cosine alone cannot see this difference -- that is the point"

    ok, detail = _verify(normalised, ref)
    assert not ok
    assert "magnitude" in detail


def test_parity_rejects_a_backend_that_disagrees_in_direction():
    ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    wrong = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    ok, detail = _verify(wrong, ref)
    assert not ok
    assert "cosine" in detail


def test_parity_rejects_a_shape_mismatch():
    ref = np.ones((4, 8))
    ok, detail = _verify(np.ones((4, 4)), ref)
    assert not ok
    assert "shape" in detail


def test_parity_rejects_non_finite_or_empty_vectors():
    ref = np.ones((4, 3))
    ok, detail = _verify(np.full((4, 3), np.nan), ref)
    assert not ok
    ok, detail = _verify(np.zeros((4, 3)), ref)
    assert not ok


def test_parity_refuses_when_there_is_not_enough_sample_text():
    """No evidence is not the same as passing."""
    ref = np.ones((4, 3))
    ok, detail = mlxb._verify_parity(_StubEncoder(ref), _StubEncoder(ref), [])
    assert not ok
    assert "sample" in detail


def _verify(candidate, reference):
    return mlxb._verify_parity(_StubEncoder(candidate), _StubEncoder(reference), _DOCS)


def test_the_mlx_embedder_is_not_used_without_a_reference_to_check_against(monkeypatch):
    """No reference means the parity gate cannot run, so the answer is PyTorch."""
    monkeypatch.setattr(mlxb, "mlx_embeddings_available", lambda: True)
    assert mlxb.load_embedding_model(
        "BAAI/bge-base-en-v1.5", "/tmp/x", _DOCS, None, reference_model=None) is None


# ---------------------------------------------------------------------------
# Catalogue coherence
# ---------------------------------------------------------------------------
def test_every_mlx_llm_entry_names_a_repo():
    assert MLX_LLM_MODELS, "the MLX catalogue should not be empty"
    for label, spec in MLX_LLM_MODELS.items():
        assert spec.get("repo_id"), f"{label} has no repo_id"
        assert "/" in spec["repo_id"], f"{label} repo_id is not an HF path"
        assert label.startswith("MLX"), f"{label} is not identifiable as MLX in the dropdown"


def test_mlx_llm_labels_do_not_collide_with_gguf_labels():
    """Both sets share one dropdown; a duplicate key would make the selection
    ambiguous and route the user to the wrong engine."""
    assert not (set(MLX_LLM_MODELS) & set(LLM_MODELS))


def test_mlx_verified_embedding_models_are_all_real_catalogue_entries():
    """A typo here silently disables MLX for that model rather than erroring."""
    known = set(EMBEDDING_MODELS.values())
    assert MLX_COMPATIBLE_MODELS <= known, MLX_COMPATIBLE_MODELS - known


def test_an_mlx_selection_falls_back_to_a_real_gguf_not_to_no_llm():
    """If MLX cannot load, the MLX label must not be handed to resolve_llm_path
    as though it were a custom .gguf path -- that would report "no LLM" while a
    perfectly good curated model was sitting there."""
    from src.clustering import _fallback_gguf_for

    for label in MLX_LLM_MODELS:
        fallback = _fallback_gguf_for(label)
        assert fallback in LLM_MODELS, f"{label} fell back to {fallback!r}"
    # Non-MLX selections pass through untouched.
    assert _fallback_gguf_for("/models/custom.gguf") == "/models/custom.gguf"


def test_apple_options_are_described_as_apple_only_in_the_dropdown():
    """The labels are the only hint a Windows user gets about why the option is
    greyed out."""
    for label, value in ACCELERATION_OPTIONS.items():
        if value in ("mps", "mlx"):
            assert "Apple" in label, f"{label!r} does not say it needs Apple hardware"


def test_describe_never_raises_on_any_platform():
    assert isinstance(mlxb.describe(), str)
