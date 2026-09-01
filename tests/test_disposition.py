"""Tests for disposition pure helpers (_normalize_disposition, _heuristic).

Both are called without constructing the analyzer: _normalize_disposition is a
staticmethod, and _heuristic never touches `self`, so passing None as self is safe.
"""
from src.disposition import DispositionAnalyzer, DISPOSITIONS


# ---- _normalize_disposition -------------------------------------------------

def test_normalize_maps_keywords_to_buckets():
    n = DispositionAnalyzer._normalize_disposition
    assert n("we should automate this") == "Automate"
    assert n("redesign as self-service") == "Reimagine"
    assert n("eliminate the root cause permanently") == "Eradicate"
    assert n("keep this manual, needs human judgement") == "Retain"


def test_normalize_returns_none_for_empty_or_unknown():
    n = DispositionAnalyzer._normalize_disposition
    assert n("") is None
    assert n(None) is None
    assert n("banana") is None


# ---- _heuristic -------------------------------------------------------------

def _heur(metrics, keywords, samples):
    # _heuristic does not use self; None is a safe stand-in.
    return DispositionAnalyzer._heuristic(None, metrics, keywords, samples)


def test_heuristic_automate_on_stable_scriptable_fix():
    metrics = {"reopen_pct": 0, "reassign_avg": 0, "dominant_close_share": 100, "dominant_close_code": ""}
    prior, score = _heur(metrics, ["reset"], ["please reset the account"])
    assert prior == "Automate"
    assert score == 1.0


def test_heuristic_reimagine_on_high_reassignment():
    metrics = {"reassign_avg": 4}
    prior, _ = _heur(metrics, ["something"], ["escalated repeatedly"])
    assert prior == "Reimagine"


def test_heuristic_reimagine_on_demand_word():
    prior, _ = _heur({}, ["access"], ["user needs access to the portal"])
    assert prior == "Reimagine"


def test_heuristic_reimagine_on_close_code_hint():
    prior, _ = _heur({"dominant_close_code": "User Training"}, ["x"], ["y"])
    assert prior == "Reimagine"


def test_heuristic_retain_default():
    prior, _ = _heur({}, ["review"], ["manual handling required by an analyst"])
    assert prior == "Retain"


def test_heuristic_prior_always_valid():
    prior, score = _heur({}, ["misc"], ["something happened"])
    assert prior in DISPOSITIONS
    assert 0.0 <= score <= 1.0
