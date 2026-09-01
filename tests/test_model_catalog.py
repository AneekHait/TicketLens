"""The curated model catalogs must stay well-formed.

Nothing referenced LLM_MODELS from the tests before, so a malformed entry would only
surface as a KeyError deep inside resolve_llm_path or the download button -- after the
user had already picked the model.
"""
import pytest

from src.config import (LLM_MODELS, LLM_CUSTOM_SENTINEL, EMBEDDING_MODELS,
                        ACCELERATION_OPTIONS)


@pytest.mark.parametrize("label", list(LLM_MODELS))
def test_llm_entry_is_well_formed(label):
    spec = LLM_MODELS[label]
    assert set(spec) == {"repo_id", "filename", "size_gb"}, (
        f"{label}: unexpected keys {sorted(spec)}")
    assert spec["repo_id"].count("/") == 1, f"{label}: repo_id must be owner/name"
    assert spec["filename"].endswith(".gguf"), f"{label}: filename must be a .gguf"
    assert isinstance(spec["size_gb"], (int, float)) and spec["size_gb"] > 0


def test_llm_filenames_are_unique():
    """auto_detect_model matches a downloaded file back to a label by filename, so a
    duplicate would make the selection ambiguous."""
    names = [s["filename"] for s in LLM_MODELS.values()]
    assert len(names) == len(set(names)), f"duplicate filenames: {names}"


def test_the_custom_sentinel_is_not_a_curated_entry():
    """It is appended to the dropdown separately and routes to a file picker."""
    assert LLM_CUSTOM_SENTINEL not in LLM_MODELS


def test_labels_carry_the_quality_speed_marking():
    """The dropdown labels are the catalog keys, so the convention lives here."""
    for label in LLM_MODELS:
        assert "(" in label and ")" in label, f"{label}: missing the (Quality · Speed) part"


def test_the_recommended_model_is_first_and_marked():
    """auto_detect_model falls back to next(iter(LLM_MODELS)), so order is load-bearing:
    the first entry is what a fresh install gets."""
    first = next(iter(LLM_MODELS))
    assert "Recommended" in first, f"the first entry ({first}) should be the recommended one"


def test_embedding_catalog_points_at_hf_paths():
    for label, path in EMBEDDING_MODELS.items():
        assert path.count("/") == 1, f"{label}: {path} is not an owner/name HF path"


def test_acceleration_options_map_to_known_backends():
    """Exact set on purpose: every value here must have a branch in
    acceleration.resolve_backend, or the option silently falls through to the
    unknown-acceleration case and the user gets PyTorch with no explanation.
    tests/test_apple_silicon.py asserts the resolving half."""
    assert set(ACCELERATION_OPTIONS.values()) == {
        "pytorch", "openvino_int8_cpu", "mps", "mlx"}
