"""Tests for src.config merge/diff helpers."""
from src.config import _deep_merge, _compute_diff


def test_deep_merge_recurses_and_overrides():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 20, "z": 30}, "c": 3}
    out = _deep_merge(base, override)
    assert out == {"a": 1, "b": {"x": 1, "y": 20, "z": 30}, "c": 3}


def test_deep_merge_does_not_mutate_base():
    base = {"b": {"x": 1}}
    _deep_merge(base, {"b": {"x": 99}})
    assert base == {"b": {"x": 1}}


def test_compute_diff_only_changed_values():
    defaults = {"a": 1, "b": {"x": 1, "y": 2}}
    config = {"a": 1, "b": {"x": 1, "y": 9}}
    assert _compute_diff(defaults, config) == {"b": {"y": 9}}


def test_compute_diff_identical_is_empty():
    d = {"a": 1, "b": {"x": 1}}
    assert _compute_diff(d, {"a": 1, "b": {"x": 1}}) == {}


def test_compute_diff_includes_extra_keys():
    assert _compute_diff({"a": 1}, {"a": 1, "new": 5}) == {"new": 5}
