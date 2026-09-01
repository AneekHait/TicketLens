"""Tests for the guard that stops other features from voiding a clustering run.

The clustering worker reads the text out of ``self.df`` when it starts and writes labels
back minutes later, refusing to write onto a frame it did not read (``gui.py``:
"The loaded data changed while clustering was running"). Anything that replaces or
rewrites ``self.df`` in the meantime therefore destroys the run — and the error message
blames the data, which the user knows they never touched.

The LLM lease cannot cover this. Quality Audit replaces the frame wholesale
(``self.df = audit_df``) and takes no lease at all, so before ``_run_in_progress`` it was
the one click that could throw away a 20-minute run. The mid-run size picker made this
materially more likely by manufacturing a guaranteed idle-at-the-keyboard window.
"""
import io
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("PySide6", reason="PySide6 not installed")


@pytest.fixture
def app_window(qapp):
    from src.gui import ClusterApp
    return ClusterApp()


@pytest.fixture
def modals(monkeypatch):
    """Capture QMessageBox calls instead of showing them.

    Mandatory here, not a convenience: these are application-modal and block on an
    internal event loop, so an unpatched one hangs the test run indefinitely rather
    than failing. Returns the list of (kind, title, text) that were raised.
    """
    import src.gui as gui_mod

    seen = []

    def rec(kind, default):
        def _f(parent, title, text, *a, **k):
            seen.append((kind, title, text))
            return default
        return staticmethod(_f)

    monkeypatch.setattr(gui_mod.QMessageBox, "information", rec("info", None))
    monkeypatch.setattr(gui_mod.QMessageBox, "warning", rec("warn", None))
    monkeypatch.setattr(gui_mod.QMessageBox, "critical", rec("crit", None))
    monkeypatch.setattr(gui_mod.QMessageBox, "question",
                        rec("ask", gui_mod.QMessageBox.StandardButton.No))
    return seen


# --- the flag's lifecycle ---------------------------------------------------
def test_flag_starts_clear(app_window):
    assert app_window._run_in_progress is False


def test_finished_clears_the_flag_on_every_outcome(app_window, modals):
    """Success, cancel and error all funnel through one slot; a stuck flag would lock
    the user out of Quality Audit and file loading for the rest of the session."""
    for success, error in [(True, None), (False, "__CANCELLED__"), (False, "boom")]:
        app_window._run_in_progress = True
        try:
            app_window._on_clustering_finished(success, error)
        except Exception:
            # Rendering may fail with no real results loaded; the flag is the contract.
            pass
        assert app_window._run_in_progress is False, f"{success!r}/{error!r} left it set"


def test_flag_is_cleared_before_anything_that_could_raise():
    """It must be the first statement in the slot, not the last."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    body = gui[gui.index("def _on_clustering_finished"):]
    body = body[: body.index("\n    def ", 10)]
    code = [ln.strip() for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith(("#", '"""'))]
    assert "self._run_in_progress = False" in code
    # Only the def line and the `cp = ...` alias may precede it.
    assert code.index("self._run_in_progress = False") <= 2, code[:5]


# --- the guard refuses the dangerous features -------------------------------
@pytest.mark.parametrize("method,label", [
    ("_run_quality_audit", "Audit"),
    ("load_file", "Open Excel"),
    ("_run_disposition_analysis", "Automation Opportunities"),
    ("_run_category_audit", "Category Audit"),
])
def test_entry_points_are_refused_during_a_run(app_window, monkeypatch, modals,
                                               method, label):
    """Each of these replaces or rewrites self.df, so each must decline mid-run."""
    import src.gui as gui_mod

    # The file dialog is modal too; reaching it means the guard didn't fire.
    monkeypatch.setattr(gui_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("reached the file dialog during a run"))))

    pd = pytest.importorskip("pandas")
    app_window.df = pd.DataFrame({"a": [1, 2]})
    app_window._run_in_progress = True

    getattr(app_window, method)()

    titles = [t for _kind, t, _text in modals]
    assert label in titles, modals
    text = next(x for k, t, x in modals if t == label)
    assert "clustering run is in progress" in text
    app_window._run_in_progress = False


def test_the_guard_is_transparent_when_no_run_is_active(app_window):
    """The guard must not block normal use — returning False when idle."""
    app_window._run_in_progress = False
    assert app_window._busy_with_run("Audit") is False
    assert app_window._busy_with_run("Impact Analysis", read_only=True) is False


# --- read-only analysis features also decline, for a different reason --------
@pytest.mark.parametrize("method,label", [
    ("_run_impact_analysis", "Impact Analysis"),
    ("_run_main_theme", "Overview"),
    ("_run_fishbone", "Fishbone"),
])
def test_read_only_analysis_is_refused_during_a_run(app_window, modals, method, label):
    """These don't replace self.df, so they can't void the run — but they read
    cluster_data that _on_clustering_finished is about to replace, so the report
    would be stale on arrival. _run_impact_analysis is additionally threaded, so it
    can read a mismatched df/cluster_data pair."""
    pd = pytest.importorskip("pandas")
    app_window.df = pd.DataFrame({"Cluster_ID": [0, 1]})
    app_window.cluster_data = {0: {"keywords": ["vpn"], "subcategory": "s"}}
    app_window._run_in_progress = True

    getattr(app_window, method)()

    titles = [t for _kind, t, _text in modals]
    assert label in titles, modals
    text = next(x for k, t, x in modals if t == label)
    assert "clustering run is in progress" in text
    # The reason must be the honest read-only one, not "results would be discarded".
    assert "previous" in text.lower()
    assert "discarded" not in text.lower()
    app_window._run_in_progress = False


def test_the_two_guard_reasons_are_distinct(app_window, modals):
    """A read-only feature must not tell the user their run will be discarded."""
    app_window._run_in_progress = True
    app_window._busy_with_run("Mutating", read_only=False)
    app_window._busy_with_run("ReadOnly", read_only=True)
    app_window._run_in_progress = False
    mutating = next(x for k, t, x in modals if t == "Mutating")
    readonly = next(x for k, t, x in modals if t == "ReadOnly")
    assert "discarded" in mutating.lower()
    assert "discarded" not in readonly.lower()
    assert mutating != readonly


def test_quality_audit_still_replaces_the_frame():
    """Documents *why* the guard exists. If this ever stops being true the guard's
    rationale changes, and this test should be revisited rather than deleted."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    assert "self.df = audit_df" in gui


def test_quality_audit_takes_no_llm_lease():
    """The other frame-writers are gated by _llm_acquire; this one is not, which is
    what made it the dangerous one."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    body = gui[gui.index("def _run_quality_audit"):]
    body = body[: body.index("\n    def ", 10)]
    assert "_llm_acquire" not in body
    assert "_busy_with_run" in body, "so it must rely on the run guard instead"


# --- the import lock covers every route to load_file ------------------------
def test_file_menu_action_is_locked_with_the_import_controls(app_window):
    """A disabled Import button does not disable File > Open Excel."""
    assert app_window.act_open is not None
    app_window._set_import_enabled(False)
    assert app_window.act_open.isEnabled() is False
    assert app_window.import_page.btn_load.isEnabled() is False
    app_window._set_import_enabled(True)
    assert app_window.act_open.isEnabled() is True


def test_sheet_dropdown_is_not_re_enabled_during_a_run():
    """load_file used to end with dd.setEnabled(True), undoing the run's own lock."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    body = gui[gui.index("def load_file"):]
    body = body[: body.index("\n    def ", 10)]
    assert "dd.setEnabled(True)" not in body, "unconditional re-enable is back"
    assert "dd.setEnabled(not self._run_in_progress)" in body


def test_listing_sheets_does_not_load_one_as_a_side_effect():
    """clear()/addItems() emit currentTextChanged, so populating the dropdown used to
    fire on_sheet_selected and read a sheet nobody asked for."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    body = gui[gui.index("def load_file"):]
    body = body[: body.index("\n    def ", 10)]
    add = body.index("dd.addItems(self.sheet_names)")
    assert "dd.blockSignals(True)" in body[:add], "addItems is not signal-blocked"
    assert "dd.blockSignals(False)" in body[add:], "signals never unblocked"
