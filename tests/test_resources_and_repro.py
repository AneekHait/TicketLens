"""Tests for the Phase 2 resource-management and reproducibility fixes."""
import logging
import os

import pytest

import src.clustering as clustering
import src.logger as app_logger
from src.config import DEFAULTS


# --- embedding model cache is bounded --------------------------------------
def test_embedding_cache_is_bounded():
    """Previously unbounded: trying several models in one session kept every set of
    weights resident (bge-base ~440 MB, Qwen3 multi-GB)."""
    original = dict(clustering._embedding_model_cache)
    try:
        clustering._embedding_model_cache.clear()
        for i in range(clustering._EMBEDDING_CACHE_MAX + 3):
            clustering._cache_embedding_model((f"model-{i}", "pytorch", None), object())
        assert len(clustering._embedding_model_cache) == clustering._EMBEDDING_CACHE_MAX
    finally:
        clustering._embedding_model_cache.clear()
        clustering._embedding_model_cache.update(original)


def test_embedding_cache_evicts_oldest_first():
    original = dict(clustering._embedding_model_cache)
    try:
        clustering._embedding_model_cache.clear()
        first, last = ("first", "pytorch", None), ("last", "pytorch", None)
        clustering._cache_embedding_model(first, object())
        for i in range(clustering._EMBEDDING_CACHE_MAX):
            clustering._cache_embedding_model((f"filler-{i}", "pytorch", None), object())
        clustering._cache_embedding_model(last, object())
        assert first not in clustering._embedding_model_cache
        assert last in clustering._embedding_model_cache
    finally:
        clustering._embedding_model_cache.clear()
        clustering._embedding_model_cache.update(original)


def test_clear_embedding_model_cache_reports_and_empties():
    original = dict(clustering._embedding_model_cache)
    try:
        clustering._embedding_model_cache.clear()
        clustering._cache_embedding_model(("a", "pytorch", None), object())
        assert clustering.clear_embedding_model_cache() == 1
        assert clustering._embedding_model_cache == {}
        assert clustering.clear_embedding_model_cache() == 0    # idempotent
    finally:
        clustering._embedding_model_cache.clear()
        clustering._embedding_model_cache.update(original)


# --- Llama release ----------------------------------------------------------
def test_close_llm_releases_and_is_idempotent():
    closed = []

    class FakeLlama:
        def close(self):
            closed.append(True)

    c = clustering.TicketClusterer()
    c.llm = FakeLlama()
    assert c.close_llm() is True
    assert closed == [True]
    assert c.llm is None
    assert c.close_llm() is False, "second call should be a no-op"


def test_close_llm_survives_a_model_without_close():
    c = clustering.TicketClusterer()
    c.llm = object()            # no .close attribute
    assert c.close_llm() is True
    assert c.llm is None


def test_close_llm_swallows_a_failing_close():
    class Angry:
        def close(self):
            raise RuntimeError("native teardown blew up")

    c = clustering.TicketClusterer()
    c.llm = Angry()
    assert c.close_llm() is True     # must not propagate
    assert c.llm is None


def test_close_llm_with_no_model_loaded():
    assert clustering.TicketClusterer().close_llm() is False


# --- reproducibility --------------------------------------------------------
def test_sample_indices_is_deterministic(monkeypatch):
    """The >50k training subsample used the global numpy RNG, so even the explicit
    'reproducible' mode differed run to run.

    Injects numpy directly rather than calling _lazy_import_ml(), which would also
    pull in bertopic/umap just to exercise one sampling helper.
    """
    numpy = pytest.importorskip("numpy")
    monkeypatch.setattr(clustering, "_np", numpy)
    a = clustering._sample_indices(1000, 50)
    b = clustering._sample_indices(1000, 50)
    assert list(a) == list(b)
    assert len(set(a.tolist())) == 50, "indices must be distinct"
    assert all(0 <= i < 1000 for i in a.tolist())


def test_reproducible_mode_keeps_a_fixed_seed():
    state, n_jobs, label = clustering._umap_parallel("reproducible", 1_000_000)
    assert state == 42 and n_jobs == 1
    assert "reproducible" in label.lower()


def test_parallel_mode_label_states_the_reproducibility_cost():
    """'auto' silently becomes non-reproducible above the threshold, so the label the
    run log prints has to say so."""
    state, n_jobs, label = clustering._umap_parallel("parallel", 10)
    assert state is None and n_jobs == -1
    assert "not" in label.lower() and "reproducible" in label.lower()

    _s, _j, auto_label = clustering._umap_parallel(
        "auto", clustering.UMAP_PARALLEL_THRESHOLD)
    assert "not" in auto_label.lower()


def test_auto_mode_stays_reproducible_below_the_threshold():
    state, n_jobs, label = clustering._umap_parallel(
        "auto", clustering.UMAP_PARALLEL_THRESHOLD - 1)
    assert state == 42 and n_jobs == 1
    assert "reproducible" in label.lower()


# --- logging ----------------------------------------------------------------
def test_file_handler_is_size_capped():
    """A count cap alone left 15 *unbounded* files, i.e. still unbounded."""
    lg = app_logger.get_logger()
    rotating = [h for h in lg.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert rotating, "the log file handler must be size-capped"
    assert rotating[0].maxBytes == app_logger.MAX_LOG_BYTES
    assert app_logger.MAX_LOG_BYTES > 0


def test_prune_reserves_room_for_the_incoming_file(tmp_path, monkeypatch):
    """Pruning runs before the new handler opens its file, so it must keep
    MAX_LOG_FILES-1 to settle at the cap rather than one over it."""
    monkeypatch.setattr(app_logger, "_LOG_DIR", str(tmp_path))
    for i in range(20):
        (tmp_path / f"clustering_2026010{i % 10}_{i:06d}.log").write_text("x")

    app_logger._prune_old_logs(keep=5)
    remaining = sorted(p.name for p in tmp_path.glob("*.log"))
    assert len(remaining) == 4, remaining      # 5 - 1 reserved for the new file


def test_prune_also_sweeps_non_clustering_logs(tmp_path, monkeypatch):
    """get_latest_log_file() matches any *.log, so pruning must too — otherwise a
    stray log accumulates forever yet can be reported as 'the latest'."""
    monkeypatch.setattr(app_logger, "_LOG_DIR", str(tmp_path))
    (tmp_path / "aaa_stray.log").write_text("x")
    for i in range(6):
        (tmp_path / f"clustering_2026010{i}_000000.log").write_text("x")

    app_logger._prune_old_logs(keep=3)
    assert len(list(tmp_path.glob("*.log"))) == 2
    assert not (tmp_path / "aaa_stray.log").exists(), "oldest-first should remove it"


# --- config -----------------------------------------------------------------
def test_category_audit_section_exists_in_defaults():
    """The GUI reads config['category_audit'], so an absent section meant the
    auditor's tuning knobs were silently unreachable."""
    section = DEFAULTS.get("category_audit")
    assert section, "category_audit must be declared in DEFAULTS"
    for key in ("max_sample_tickets", "max_keywords", "temperature", "max_tokens"):
        assert key in section, key


def test_category_auditor_honours_the_configured_values():
    from src.category_audit import CategoryAuditor

    a = CategoryAuditor(llm=None, cluster_data={}, settings=DEFAULTS["category_audit"])
    assert a.max_samples == DEFAULTS["category_audit"]["max_sample_tickets"]
    assert a.max_keywords == DEFAULTS["category_audit"]["max_keywords"]
    assert a.temperature == DEFAULTS["category_audit"]["temperature"]


# --- offline plotly ---------------------------------------------------------
def test_plotly_output_does_not_depend_on_a_cdn():
    """This is an offline-first tool: include_plotlyjs='cdn' rendered every diagram
    as a blank page on an air-gapped machine.

    Checks the actual write_html calls, not prose — the surrounding comments mention
    "cdn" precisely to explain why it isn't used.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("src/fishbone.py", "src/impact_analysis.py"):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            calls = [ln.strip() for ln in f
                     if "write_html(" in ln and not ln.lstrip().startswith("#")]
        assert calls, f"expected at least one write_html call in {rel}"
        for call in calls:
            assert "cdn" not in call, f"{rel}: {call}"
            assert "include_plotlyjs=True" in call, f"{rel}: {call}"
