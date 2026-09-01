"""Tests for the mid-run Min Cluster Size picker.

Embedding is the slow phase of a run and it does not depend on min_cluster_size at
all, so the honest moment to choose that size is *after* embeddings exist: the sweep
then costs only HDBSCAN passes, and the cluster counts shown are real numbers for
this dataset rather than a guess made before any work happened.

Two mechanisms carry that, and both are easy to break silently:

1. ``run(on_embeddings_ready=...)`` fires between embedding and UMAP, and its return
   value replaces min_cluster_size for the rest of the run.
2. ``suggest_min_cluster_size(precomputed_embeddings=...)`` skips embedding entirely,
   and with ``return_reduced=True`` also hands back its UMAP projection so ``run``
   doesn't repeat an identical reduction.

These use fakes rather than the real UMAP/HDBSCAN: the contract under test is the
plumbing (is the hook called at the right point, is its answer honoured, is the
projection reused), not the clustering maths.
"""
import ast
import io
import os
import threading

import pytest

from src.config import DEFAULTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Qt is needed by the widget-level tests below. conftest sets QT_QPA_PLATFORM=offscreen
# before this import, so it works headless; skip cleanly if PySide6 isn't installed.
pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtWidgets import QDialog, QLabel, QPushButton   # noqa: E402


# --- the config flag --------------------------------------------------------
def test_pick_after_embedding_defaults_on():
    """The feature is the requested default flow; the checkbox is the opt-out."""
    assert DEFAULTS["clustering"]["pick_size_after_embedding"] is True


# --- run() calls the hook at the right point, and honours its answer --------
def _hook_call_site():
    """The source of run() between the embedding cache-save and the UMAP block."""
    with io.open(os.path.join(REPO, "src", "clustering.py"), encoding="utf-8") as f:
        src = f.read()
    return src


def test_hook_fires_before_umap_not_after():
    """If the hook ran after UMAP, the chosen size would arrive too late to matter
    for the projection and the sweep's reduction could not be reused."""
    src = _hook_call_site()
    hook = src.index("on_embeddings_ready(embeddings, effective_min_cluster)")
    umap = src.index("# --- UMAP Dimensionality Reduction")
    hdbscan = src.index("min_cluster_size=effective_min_cluster")
    assert hook < umap < hdbscan, "the hook must fire before UMAP and before HDBSCAN"


def test_run_signature_accepts_the_hook():
    src = _hook_call_site()
    tree = ast.parse(src)
    run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    args = [a.arg for a in run.args.args] + [a.arg for a in run.args.kwonlyargs]
    assert "on_embeddings_ready" in args


def test_suggest_signature_accepts_precomputed_and_return_reduced():
    src = _hook_call_site()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "suggest_min_cluster_size"
    )
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "precomputed_embeddings" in args
    assert "return_reduced" in args


# --- suggest_min_cluster_size with pre-computed embeddings ------------------
@pytest.fixture
def clusterer_for_sweep(monkeypatch):
    """A TicketClusterer whose UMAP/HDBSCAN are cheap fakes.

    Patches the module-level ML handles that _lazy_import_ml() would otherwise fill,
    and neutralises the import so it can't overwrite them.
    """
    np = pytest.importorskip("numpy")
    import src.clustering as clustering

    monkeypatch.setattr(clustering, "_lazy_import_ml", lambda: None)
    monkeypatch.setattr(clustering, "_np", np, raising=False)

    class FakeUMAP:
        """Reduces to n_components dims by truncation, and records that it ran."""
        instances = []

        def __init__(self, **kwargs):
            self.n_components = kwargs.get("n_components", 5)
            FakeUMAP.instances.append(self)

        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)[:, : self.n_components]

        def fit_transform(self, X):
            return self.transform(X)

    class FakeHDBSCAN:
        """Splits into 3 groups; noise fraction shrinks as min_cluster_size grows so
        the scoring code has something monotonic to chew on."""

        def __init__(self, **kwargs):
            self.min_cluster_size = kwargs.get("min_cluster_size", 15)
            self.relative_validity_ = 0.5

        def fit_predict(self, X):
            n = len(X)
            labels = np.array([i % 3 for i in range(n)])
            # Mark a couple of rows as noise.
            if n > 4:
                labels[0] = -1
                labels[1] = -1
            return labels

    FakeUMAP.instances = []
    monkeypatch.setattr(clustering, "_UMAP", FakeUMAP, raising=False)
    monkeypatch.setattr(clustering, "_HDBSCAN", FakeHDBSCAN, raising=False)

    TicketClusterer = clustering.TicketClusterer
    c = TicketClusterer(embedding_model_name="bge-base-en-v1.5 (High · Moderate — Recommended)")
    c._cache_enabled = False
    return c, FakeUMAP, np


def test_precomputed_embeddings_skip_the_embedding_phase(clusterer_for_sweep,
                                                         monkeypatch):
    """Passing embeddings in must not load a model or touch the docs — that is the
    entire point of calling this mid-run."""
    c, _FakeUMAP, np = clusterer_for_sweep
    import src.clustering as clustering

    def boom(*a, **k):
        raise AssertionError("the embedding model must not be loaded")

    monkeypatch.setattr(clustering, "get_embedding_model_resolved", boom)
    monkeypatch.setattr(clustering, "preprocess_documents", boom)

    embeddings = np.random.default_rng(0).random((60, 8)).astype("float32")
    # docs=None proves nothing reads the documents on this path.
    result = c.suggest_min_cluster_size(docs=None, precomputed_embeddings=embeddings)

    assert isinstance(result["recommended"], int)
    assert result["candidates"], "the sweep should have evaluated candidate sizes"


def test_return_reduced_hands_back_the_projection(clusterer_for_sweep):
    c, _FakeUMAP, np = clusterer_for_sweep
    embeddings = np.random.default_rng(1).random((60, 8)).astype("float32")

    result = c.suggest_min_cluster_size(
        docs=None, precomputed_embeddings=embeddings, return_reduced=True)

    reduced = result["reduced_embeddings"]
    assert reduced is not None
    assert len(reduced) == len(embeddings), "one reduced row per input row"


def test_reduced_is_withheld_unless_asked_for(clusterer_for_sweep):
    """The normal Suggest button ships this dict across a Qt signal; it shouldn't be
    made to keep an (n x 5) array alive for no reason."""
    c, _FakeUMAP, np = clusterer_for_sweep
    embeddings = np.random.default_rng(2).random((40, 8)).astype("float32")

    result = c.suggest_min_cluster_size(docs=None, precomputed_embeddings=embeddings)
    assert "reduced_embeddings" not in result


# --- the hook's answer actually changes the run ------------------------------
def _run_with_stubs(monkeypatch, hook, min_cluster_size=15, n=60):
    """Drive TicketClusterer.run() far enough to observe HDBSCAN's min_cluster_size.

    Everything expensive is stubbed; the return value is the min_cluster_size that
    reached HDBSCAN plus whether UMAP was constructed.
    """
    np = pytest.importorskip("numpy")
    import src.clustering as clustering

    seen = {"hdbscan_min_cluster_size": None, "umap_built": 0}

    monkeypatch.setattr(clustering, "_lazy_import_ml", lambda: None)
    monkeypatch.setattr(clustering, "_np", np, raising=False)
    monkeypatch.setattr(clustering, "preprocess_documents",
                        lambda docs, settings=None: [str(d) for d in docs])

    embeddings = np.random.default_rng(3).random((n, 8)).astype("float32")

    class FakeEmbedder:
        def encode(self, batch, **kwargs):
            return embeddings[: len(batch)]

    monkeypatch.setattr(clustering, "get_embedding_model_resolved",
                        lambda *a, **k: (FakeEmbedder(), "pytorch", None))

    class FakeUMAP:
        def __init__(self, **kwargs):
            seen["umap_built"] += 1
            self.n_components = kwargs.get("n_components", 5)

        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)[:, : self.n_components]

        def fit_transform(self, X):
            return self.transform(X)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            seen["hdbscan_min_cluster_size"] = kwargs.get("min_cluster_size")
            self.relative_validity_ = 0.5

        def fit_predict(self, X):
            return np.array([i % 3 for i in range(len(X))])

    monkeypatch.setattr(clustering, "_UMAP", FakeUMAP, raising=False)
    monkeypatch.setattr(clustering, "_HDBSCAN", FakeHDBSCAN, raising=False)

    c = clustering.TicketClusterer()
    c._cache_enabled = False

    docs = [f"ticket text {i}" for i in range(n)]
    try:
        c.run(docs, min_cluster_size=min_cluster_size, use_preprocessing=False,
              label_categories=False, on_embeddings_ready=hook)
    except Exception:
        # BERTopic and the labelling stages are out of scope; the assertions below
        # only need the values captured before that point.
        pass
    return seen


def test_hook_return_value_replaces_min_cluster_size(monkeypatch):
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: 42, min_cluster_size=15)
    assert seen["hdbscan_min_cluster_size"] == 42


def test_hook_returning_a_tuple_also_sets_the_size(monkeypatch):
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: (33, None),
                           min_cluster_size=15)
    assert seen["hdbscan_min_cluster_size"] == 33


def test_hook_returning_none_keeps_the_original_size(monkeypatch):
    """Cancelling the picker must leave the value the user already set."""
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: None,
                           min_cluster_size=15)
    assert seen["hdbscan_min_cluster_size"] == 15


def test_hook_raising_does_not_abort_the_run(monkeypatch):
    """A failed sweep is not a failed run."""
    def boom(emb, cur):
        raise RuntimeError("sweep exploded")

    seen = _run_with_stubs(monkeypatch, hook=boom, min_cluster_size=15)
    assert seen["hdbscan_min_cluster_size"] == 15


def test_nonsense_hook_values_are_ignored(monkeypatch):
    for bad in ("30", 1, 0, -5, None, 3.7):
        seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur, b=bad: b,
                               min_cluster_size=15)
        assert seen["hdbscan_min_cluster_size"] == 15, f"accepted {bad!r}"


def test_malformed_tuples_do_not_kill_the_run(monkeypatch):
    """The docstring promises anything unusable is ignored. The unpack and the len()
    used to sit outside the try, so a 1- or 3-tuple raised straight out of run()."""
    for bad in [(20,), (20, None, "extra"), (20, 5), ("x", None)]:
        seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur, b=bad: b,
                               min_cluster_size=15, n=60)
        # Reached HDBSCAN at all => the run survived.
        assert seen["hdbscan_min_cluster_size"] is not None, f"run died on {bad!r}"
        # And it reduced for itself rather than trusting a bad projection.
        assert seen["umap_built"] == 1, f"trusted a bad projection from {bad!r}"


def test_a_two_tuple_with_a_non_sized_projection_still_sets_the_size(monkeypatch):
    """(20, 5) is malformed only in its second element — the size is still usable."""
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: (20, 5),
                           min_cluster_size=15, n=60)
    assert seen["hdbscan_min_cluster_size"] == 20
    assert seen["umap_built"] == 1


def test_hook_receives_the_current_size(monkeypatch):
    """The picker pre-selects the current value, so it has to be told what it is."""
    got = {}

    def hook(emb, cur):
        got["current"] = cur
        return None

    _run_with_stubs(monkeypatch, hook=hook, min_cluster_size=23)
    assert got["current"] == 23


def test_hook_receives_embeddings_it_can_sweep(monkeypatch):
    got = {}

    def hook(emb, cur):
        got["n"] = len(emb)
        return None

    _run_with_stubs(monkeypatch, hook=hook, n=60)
    assert got["n"] == 60


# --- the reused UMAP projection --------------------------------------------
def test_returned_projection_replaces_runs_own_umap(monkeypatch):
    """Handing back a correctly-sized projection must skip run()'s UMAP entirely —
    the sweep already did that identical reduction."""
    np = pytest.importorskip("numpy")
    reduced = np.zeros((60, 5), dtype="float32")
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: (20, reduced), n=60)
    assert seen["umap_built"] == 0, "run() rebuilt UMAP despite being handed one"
    assert seen["hdbscan_min_cluster_size"] == 20


def test_no_projection_means_run_still_does_its_own_umap(monkeypatch):
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: (20, None), n=60)
    assert seen["umap_built"] == 1


def test_wrong_length_projection_is_rejected(monkeypatch):
    """A mismatched projection would silently mislabel every row, so run() must
    fall back to reducing the embeddings itself rather than trusting it."""
    np = pytest.importorskip("numpy")
    wrong = np.zeros((7, 5), dtype="float32")     # 7 != 60 rows
    seen = _run_with_stubs(monkeypatch, hook=lambda emb, cur: (20, wrong), n=60)
    assert seen["umap_built"] == 1, "a wrong-sized projection was accepted"


def test_hook_is_absent_when_the_option_is_off(monkeypatch):
    """With no hook, run() behaves exactly as before this feature existed."""
    seen = _run_with_stubs(monkeypatch, hook=None, min_cluster_size=15)
    assert seen["hdbscan_min_cluster_size"] == 15
    assert seen["umap_built"] == 1


# --- the shared transform helper -------------------------------------------
# The sweep used to do one full transform where run() batches "to avoid memory
# issues". Since UMAP's transform re-optimises per call, that made the reused
# projection a different array from the one run() would have produced.
class _CountingUMAP:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.calls = []

    def transform(self, X):
        import numpy as np
        self.calls.append(len(X))
        return np.asarray(X)[:, : self.n_components]


def test_transform_helper_is_a_single_call_below_the_threshold():
    np = pytest.importorskip("numpy")
    from src.clustering import _umap_transform_all, UMAP_TRANSFORM_BATCH_THRESHOLD

    m = _CountingUMAP()
    n = 1000
    out = _umap_transform_all(m, np.zeros((n, 8), dtype="float32"))
    assert m.calls == [n]
    assert len(out) == n
    assert n <= UMAP_TRANSFORM_BATCH_THRESHOLD


def test_transform_helper_batches_above_the_threshold():
    np = pytest.importorskip("numpy")
    from src.clustering import (_umap_transform_all, UMAP_TRANSFORM_BATCH_SIZE,
                                UMAP_TRANSFORM_BATCH_THRESHOLD)

    m = _CountingUMAP()
    n = UMAP_TRANSFORM_BATCH_THRESHOLD + 1
    out = _umap_transform_all(m, np.zeros((n, 8), dtype="float32"))
    assert len(m.calls) > 1, "did not batch above the threshold"
    assert max(m.calls) <= UMAP_TRANSFORM_BATCH_SIZE
    assert sum(m.calls) == n, "batches must cover every row exactly once"
    assert len(out) == n


def test_both_call_sites_use_the_shared_helper():
    """Guards the fix: an inline transform in either place reintroduces the divergence."""
    with io.open(os.path.join(REPO, "src", "clustering.py"), encoding="utf-8") as f:
        src = f.read()
    assert src.count("_umap_transform_all(") >= 3, "helper defined + used by both sites"
    # The old inline batching must be gone from run().
    assert "reduced_parts = []" not in src, "run() still batches inline"


# --- the abort button ------------------------------------------------------
def test_picker_has_no_abort_button_by_default(qapp):
    """The Suggest-button path must keep its two-button shape."""
    from src.gui import MinClusterSizePickerDialog

    dlg = MinClusterSizePickerDialog(15, [{"size": 15, "n_clusters": 3,
                                           "noise_pct": 10.0, "dbcv": 0.4,
                                           "valid": True}])
    assert dlg.aborted() is False
    labels = [b.text() for b in dlg.findChildren(QPushButton)]
    assert not any("Stop" in t for t in labels), labels
    assert any("Cancel" == t for t in labels), labels


def test_picker_abort_button_appears_and_reports_mid_run(qapp):
    from src.gui import MinClusterSizePickerDialog

    dlg = MinClusterSizePickerDialog(15, [{"size": 15, "n_clusters": 3,
                                           "noise_pct": 10.0, "dbcv": 0.4,
                                           "valid": True}], allow_abort=True)
    labels = [b.text() for b in dlg.findChildren(QPushButton)]
    assert any("Stop the run" == t for t in labels), labels
    # "Cancel" is reworded, because mid-run it continues the run rather than undoing it.
    assert not any("Cancel" == t for t in labels), labels
    assert any("Keep current value" == t for t in labels), labels

    assert dlg.aborted() is False
    dlg._on_abort()
    assert dlg.aborted() is True
    # Abort closes as Rejected, so callers must use aborted() to tell them apart.
    assert dlg.result() == QDialog.DialogCode.Rejected


def test_mid_run_wording_says_the_run_continues(qapp):
    """The old text ("Cancel to keep your current value") was written for the Suggest
    button and hid the fact that mid-run every exit starts a full clustering run."""
    from src.gui import MinClusterSizePickerDialog

    cands = [{"size": 15, "n_clusters": 3, "noise_pct": 10.0, "dbcv": 0.4, "valid": True}]
    mid = MinClusterSizePickerDialog(15, cands, allow_abort=True)
    text = " ".join(l.text() for l in mid.findChildren(QLabel))
    assert "paused" in text.lower()
    assert "continues the run" in text.lower()


def test_real_hook_abort_sets_the_stop_event_and_cancels(app_window, monkeypatch, qapp):
    """The picker's Stop button must end the run through the existing cancellation
    path, not by inventing a second one."""
    import src.gui as gui_mod
    from src.clustering import ClusteringCancelled

    class AbortingDialog:
        def __init__(self, rec, candidates, reason, parent=None, allow_abort=False):
            assert allow_abort, "mid-run must enable the abort button"

        def setWindowTitle(self, _t):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def aborted(self):
            return True

        def selected_size(self):
            return 99          # must be ignored when aborting

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", AbortingDialog)
    app_window._stop_event.clear()
    app_window.clustering_page.spin_cluster_size.setValue(15)
    hook = app_window._make_embeddings_ready_hook(_SweepStub(), lambda m, p=None: None)

    box = {}

    def worker():
        try:
            hook([0] * 10, 15)
        except ClusteringCancelled:
            box["cancelled"] = True
        except BaseException as e:              # noqa: BLE001 - report anything else
            box["other"] = repr(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(1500, qapp.quit)
    qapp.exec()
    t.join(timeout=5)

    assert not t.is_alive()
    assert box.get("cancelled") is True, box
    assert app_window._stop_event.is_set(), "abort must use the normal stop path"
    # The ignored size must not have been written to the UI.
    assert app_window.clustering_page.spin_cluster_size.value() == 15
    app_window._stop_event.clear()


# --- stop pressed during the sweep ----------------------------------------
def test_stop_during_the_sweep_never_shows_the_dialog(app_window, monkeypatch, qapp):
    """The orphan-modal bug: the sweep ran to completion, the dialog was posted, then
    the run ended underneath it — leaving an app-modal dialog blocking everything."""
    import src.gui as gui_mod
    from src.clustering import ClusteringCancelled

    class ExplodingDialog:
        def __init__(self, *a, **k):
            raise AssertionError("the picker was shown for an already-cancelled run")

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", ExplodingDialog)

    class StopMidSweep:
        """Simulates Stop landing while the sweep is working."""

        def suggest_min_cluster_size(self, docs=None, precomputed_embeddings=None,
                                     callback=None, return_reduced=False,
                                     should_stop=None, current_size=None):
            app_window._stop_event.set()
            # A real sweep raises from its own _ck(); mirror that.
            if should_stop and should_stop():
                raise ClusteringCancelled("cancelled in sweep")
            raise AssertionError("the sweep ignored should_stop")

    app_window._stop_event.clear()
    hook = app_window._make_embeddings_ready_hook(StopMidSweep(),
                                                  lambda m, p=None: None)
    box = {}

    def worker():
        try:
            hook([0] * 10, 15)
        except ClusteringCancelled:
            box["cancelled"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(800, qapp.quit)
    qapp.exec()
    t.join(timeout=5)

    assert not t.is_alive()
    assert box.get("cancelled") is True, "cancellation was swallowed as a failed sweep"
    app_window._stop_event.clear()


def test_stop_between_sweep_and_dialog_never_shows_the_dialog(app_window, monkeypatch,
                                                             qapp):
    """Stop can land in the gap after a successful sweep — still no dialog."""
    import src.gui as gui_mod
    from src.clustering import ClusteringCancelled

    class ExplodingDialog:
        def __init__(self, *a, **k):
            raise AssertionError("the picker was shown for an already-cancelled run")

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", ExplodingDialog)

    class SweepThenStop(_SweepStub):
        def suggest_min_cluster_size(self, **kw):
            out = super().suggest_min_cluster_size(**kw)
            app_window._stop_event.set()      # Stop lands just as the sweep returns
            return out

    app_window._stop_event.clear()
    hook = app_window._make_embeddings_ready_hook(SweepThenStop(),
                                                 lambda m, p=None: None)
    box = {}

    def worker():
        try:
            hook([0] * 10, 15)
        except ClusteringCancelled:
            box["cancelled"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(800, qapp.quit)
    qapp.exec()
    t.join(timeout=5)

    assert not t.is_alive()
    assert box.get("cancelled") is True, box
    app_window._stop_event.clear()


def test_run_checks_for_a_stop_before_calling_the_hook():
    """A _ck() must bracket the hook, so a pending stop skips it entirely."""
    with io.open(os.path.join(REPO, "src", "clustering.py"), encoding="utf-8") as f:
        src = f.read()
    call = src.index("on_embeddings_ready(embeddings, effective_min_cluster)")
    before = src.rindex("_ck()", 0, call)
    cache_save = src.index("Could not cache embeddings:")
    assert before > cache_save, "no _ck() between the embedding phase and the hook"


def test_sweep_accepts_should_stop_and_checks_it_per_candidate():
    with io.open(os.path.join(REPO, "src", "clustering.py"), encoding="utf-8") as f:
        src = f.read()
    fn = src[src.index("def suggest_min_cluster_size"):src.index("    def run(self, docs")]
    assert "should_stop" in fn
    # The check must be outside the per-candidate try, or it is logged as a failed
    # candidate and the sweep carries on. Compare *code* lines only — an earlier version
    # of this test matched the word "try:" inside a comment and reported a false failure.
    loop = fn[fn.index("for k, size in enumerate(sizes):"):]
    code = [ln.strip() for ln in loop.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    assert "_ck()" in code, code[:6]
    assert code.index("_ck()") < code.index("try:"), \
        "the stop check is inside the try and gets swallowed"


# --- the GUI side: the worker blocks, the dialog runs on the main thread ----
def test_clustering_page_has_the_opt_out_checked_by_default(qapp):
    """Built for real rather than grepped: the checkbox has to exist on the page and
    start on, since the mid-run picker is the requested default flow."""
    from src.gui import ClusteringPage

    page = ClusteringPage()
    assert page.chk_pick_after_embed.isChecked() is True
    # It sits with the Min Cluster Size control it governs, not off in Settings.
    assert "Min Cluster Size" in page.chk_pick_after_embed.text()


def test_gui_gates_the_hook_on_the_checkbox():
    """Regression guard: reading the QCheckBox from the worker thread is illegal, so
    the value must come from the settings dict captured on the GUI thread."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    assert 'settings["clustering"].get("pick_size_after_embedding"' in gui
    assert "if pick_after_embed else None" in gui
    # And the worker must not reach for the widget directly.
    worker = gui[gui.index("def run_clustering"):]
    worker = worker[: worker.index("def _on_clustering_finished")] if \
        "def _on_clustering_finished" in worker else worker
    assert "chk_pick_after_embed" not in worker, \
        "run_clustering must not touch the checkbox from the worker thread"


def test_gui_hook_blocks_the_worker_until_the_dialog_returns():
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    hook = gui[gui.index("def _make_embeddings_ready_hook"):]
    hook = hook[: hook.index("\n    # ---------------------------------------------------------------\n    # Clustering")]
    # The dialog must be shown via _post (main thread), never constructed inline.
    assert "self._post(_show)" in hook
    assert "ready.wait(" in hook, "the worker must wait for the user's choice"
    # ready.set() has to be unconditional or the worker hangs forever.
    assert "finally:" in hook and "ready.set()" in hook
    # And the wait must be escapable.
    assert "self._stop_event.is_set()" in hook


def test_gui_hook_requests_the_projection_and_returns_it():
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    hook = gui[gui.index("def _make_embeddings_ready_hook"):]
    hook = hook[: hook.index("def start_clustering_thread")]
    assert "return_reduced=True" in hook
    assert "precomputed_embeddings=embeddings" in hook
    assert "return chosen_box[0], reduced" in hook


def test_gui_hook_squeezes_sweep_progress_into_a_forward_only_band():
    """The sweep reports 0->1 of its own work; passed through raw it would drive the
    run's bar to 100% and then snap back to 40%."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    hook = gui[gui.index("def _make_embeddings_ready_hook"):]
    hook = hook[: hook.index("def start_clustering_thread")]
    assert "0.35 + 0.05 *" in hook


def _drive_hook(qapp, hook, embeddings, current, wait_ms=1500):
    """Run `hook` on a worker thread while pumping the main event loop, the way a real
    run does. Returns (result, thread_finished)."""
    from PySide6.QtCore import QTimer

    box = {}

    def worker():
        box["out"] = hook(embeddings, current)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    QTimer.singleShot(wait_ms, qapp.quit)
    qapp.exec()
    t.join(timeout=5)
    return box.get("out"), (not t.is_alive())


class _SweepStub:
    """Stands in for a TicketClusterer during the mid-run sweep."""

    def __init__(self, candidates=None, recommended=25, raises=False):
        self.candidates = candidates if candidates is not None else [
            {"size": 25, "n_clusters": 4, "noise_pct": 10.0, "dbcv": 0.4, "valid": True},
        ]
        self.recommended = recommended
        self.raises = raises
        self.got = {}

    def suggest_min_cluster_size(self, docs=None, precomputed_embeddings=None,
                                 callback=None, return_reduced=False,
                                 should_stop=None, current_size=None):
        if self.raises:
            raise RuntimeError("sweep exploded")
        self.got["docs"] = docs
        self.got["embeddings_passed"] = precomputed_embeddings is not None
        self.got["return_reduced"] = return_reduced
        # The hook must hand these through, or the sweep is uncancellable and can pin
        # the wrong "current" size into the candidate table.
        self.got["should_stop"] = should_stop
        self.got["current_size"] = current_size
        if callback:
            callback("sweeping…", 0.5)      # exercises the progress remap
        out = {"recommended": self.recommended, "reason": "test reason",
               "candidates": self.candidates}
        if return_reduced:
            out["reduced_embeddings"] = "REDUCED"
        return out


def _patch_dialog(monkeypatch, accepted=True, size=25):
    """Replace the picker with a stub that answers immediately."""
    import src.gui as gui_mod
    from PySide6.QtWidgets import QDialog

    class FakeDialog:
        def __init__(self, rec, candidates, reason, parent=None, allow_abort=False):
            self.rec = rec
            self.allow_abort = allow_abort

        def setWindowTitle(self, _t):
            pass

        def exec(self):
            return (QDialog.DialogCode.Accepted if accepted
                    else QDialog.DialogCode.Rejected)

        def aborted(self):
            return False        # this stub covers accept/cancel, not abort

        def selected_size(self):
            return size

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", FakeDialog)


@pytest.fixture
def app_window(qapp):
    from src.gui import ClusterApp
    return ClusterApp()


def test_real_hook_applies_the_choice_and_returns_the_projection(app_window,
                                                                 monkeypatch, qapp):
    """The whole handshake, for real: worker thread -> _post -> dialog on the main
    thread -> worker resumes with the answer."""
    _patch_dialog(monkeypatch, accepted=True, size=25)
    stub = _SweepStub(recommended=25)
    hook = app_window._make_embeddings_ready_hook(stub, lambda m, p=None: None)

    out, finished = _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert finished, "the worker never unblocked — a real run would freeze here"
    assert out == (25, "REDUCED")
    # The sweep must be handed the embeddings and asked for the projection.
    assert stub.got["embeddings_passed"] is True
    assert stub.got["return_reduced"] is True
    assert stub.got["docs"] is None, "docs must not be re-read on this path"
    # And it must be cancellable, and told the size actually in effect.
    assert callable(stub.got["should_stop"]), "the sweep was left uncancellable"
    assert stub.got["current_size"] == 15
    # The UI has to show what actually ran.
    assert app_window.clustering_page.spin_cluster_size.value() == 25


def test_real_hook_cancel_keeps_the_current_size(app_window, monkeypatch, qapp):
    """Cancel means "run with what I already set", not "abort the run"."""
    _patch_dialog(monkeypatch, accepted=False)
    hook = app_window._make_embeddings_ready_hook(_SweepStub(), lambda m, p=None: None)
    app_window.clustering_page.spin_cluster_size.setValue(15)

    out, finished = _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert finished
    assert out == (15, "REDUCED")
    assert app_window.clustering_page.spin_cluster_size.value() == 15


def test_real_hook_survives_a_failed_sweep(app_window, monkeypatch, qapp):
    """No dialog, no hang, and the run continues with the user's size."""
    _patch_dialog(monkeypatch)
    hook = app_window._make_embeddings_ready_hook(_SweepStub(raises=True),
                                                  lambda m, p=None: None)

    out, finished = _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert finished
    assert out == (15, None), "a failed sweep must not claim to have a projection"


def test_real_hook_skips_the_dialog_when_there_is_nothing_to_choose(app_window,
                                                                    monkeypatch, qapp):
    """An empty sweep must not interrupt the run with an empty picker."""
    shown = {"n": 0}
    import src.gui as gui_mod

    class ExplodingDialog:
        def __init__(self, *a, **k):
            shown["n"] += 1
            raise AssertionError("the picker was shown with no candidates")

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", ExplodingDialog)
    hook = app_window._make_embeddings_ready_hook(_SweepStub(candidates=[]),
                                                  lambda m, p=None: None)

    out, finished = _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert finished
    assert shown["n"] == 0
    assert out == (15, "REDUCED"), "the projection is still worth reusing"


def test_real_hook_unblocks_even_if_the_dialog_raises(app_window, monkeypatch, qapp):
    """ready.set() lives in a finally: for exactly this case — without it the worker
    thread would block forever and the run would never finish."""
    import src.gui as gui_mod

    class ExplodingDialog:
        def __init__(self, *a, **k):
            raise RuntimeError("dialog construction failed")

    monkeypatch.setattr(gui_mod, "MinClusterSizePickerDialog", ExplodingDialog)
    hook = app_window._make_embeddings_ready_hook(_SweepStub(), lambda m, p=None: None)

    out, finished = _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert finished, "the worker hung when the dialog raised"
    assert out == (15, "REDUCED")


def test_real_hook_keeps_sweep_progress_inside_the_run_band(app_window, monkeypatch,
                                                            qapp):
    """The sweep's own 0->1 progress must be compressed, not passed through raw."""
    _patch_dialog(monkeypatch)
    seen = []
    hook = app_window._make_embeddings_ready_hook(
        _SweepStub(), lambda m, p=None: seen.append(p) if p is not None else None)

    _drive_hook(qapp, hook, embeddings=[0] * 10, current=15)

    assert seen, "the sweep reported no progress at all"
    # 0.5 from the sweep must land in the embedding->clustering gap, not at 50%.
    assert all(0.35 <= p <= 0.40 for p in seen), seen


def test_the_dialog_is_reachable_from_a_worker_thread_pattern(qapp):
    """Exercises the actual handshake: a worker posts to the main thread, blocks on an
    Event, and the main thread sets it. Proves the pattern can't deadlock, which is
    the failure mode that would freeze a real run.
    """
    from PySide6.QtCore import QObject, Signal

    class Bridge(QObject):
        call_on_main = Signal(object)

    bridge = Bridge()
    bridge.call_on_main.connect(lambda fn: fn())

    ready = threading.Event()
    box = [None]

    def worker():
        def on_main():
            try:
                box[0] = 77
            finally:
                ready.set()

        bridge.call_on_main.emit(on_main)
        ready.wait(5)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Pump the event loop so the queued emission is delivered.
    from PySide6.QtCore import QTimer
    QTimer.singleShot(400, qapp.quit)
    qapp.exec()
    t.join(timeout=5)

    assert not t.is_alive(), "the worker never unblocked (deadlock)"
    assert box[0] == 77
