"""Regression guard for worker -> GUI thread marshalling.

Background: four features (Quality Audit, KBA, SOP, Impact) used to hop back to the
GUI thread with ``QTimer.singleShot(0, fn)`` from inside a ``threading.Thread``. A
QTimer created on a thread with no event loop never starts, so those callbacks were
silently dropped: results never rendered, buttons stayed disabled, and — worst — the
shared ``_llm_busy`` lease was never released, locking every AI feature out until
restart.

These tests pin the behaviour that made the fix necessary (QTimer does NOT work) and
the mechanism that replaced it (a Signal DOES work), so the pattern cannot be
reintroduced unnoticed. Kept deliberately dependency-light: a plain QApplication on
the offscreen platform, no pytest-qt required.
"""
import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtCore import QObject, QTimer, Signal          # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """A process-wide QApplication (Qt forbids more than one)."""
    app = QApplication.instance() or QApplication([])
    yield app


class _Bridge(QObject):
    """Mirrors WorkerSignals.call_on_main: emit a callable, run it on the GUI thread."""
    call_on_main = Signal(object)


def _drain(app, timeout_ms=1500):
    """Run the event loop briefly so queued cross-thread deliveries can land."""
    QTimer.singleShot(timeout_ms, app.quit)
    app.exec()


def test_signal_from_worker_thread_is_delivered(qapp):
    """The replacement mechanism works: a Signal emitted from a worker runs on the GUI thread."""
    bridge = _Bridge()
    seen = []
    bridge.call_on_main.connect(lambda fn: fn())

    def worker():
        bridge.call_on_main.emit(lambda: seen.append(threading.current_thread().name))

    # Start the worker only once the loop is running - the real app scenario.
    QTimer.singleShot(50, lambda: threading.Thread(target=worker, daemon=True).start())
    _drain(qapp)

    assert seen == ["MainThread"], (
        "call_on_main must deliver, and must run the callable on the GUI thread"
    )


def test_qtimer_from_worker_thread_never_fires(qapp):
    """The trap this guards against. If this ever starts passing, Qt/PySide6 semantics
    changed; the production code should still use the signal, but the comments
    explaining why would need revisiting."""
    fired = []

    def worker():
        QTimer.singleShot(0, lambda: fired.append("fired"))

    QTimer.singleShot(50, lambda: threading.Thread(target=worker, daemon=True).start())
    _drain(qapp)

    assert fired == [], (
        "QTimer.singleShot scheduled from a thread with no event loop should never fire"
    )


def test_gui_uses_signal_not_qtimer_in_worker_bodies():
    """Static guard: no `QTimer.singleShot(0, ...)` anywhere in src/gui.py.

    Every zero-delay singleShot in that file was a worker-thread hop (i.e. dead code).
    Non-zero delays are fine and still used on the GUI thread (e.g. resetting a
    'Copied' button label), so only the 0 ms form is banned.
    """
    gui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gui.py"
    )
    with open(gui_path, encoding="utf-8") as f:
        lines = f.readlines()

    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(lines, 1)
        if "QTimer.singleShot(0," in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "QTimer.singleShot(0, ...) never fires from a worker thread - use self._post(fn):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Worker threads must not *read* widgets either
# ---------------------------------------------------------------------------
# The QTimer trap above is about writing back to the GUI. The mirror-image bug is
# reading: _gather_metadata_mapping() calls QComboBox.currentText() and was being
# called from four worker threads. Changing sheet rebuilds those combos (dd.clear()),
# so a worker iterating them raced a live edit. run_clustering even carried comments
# explaining that touching a QWidget from a worker is illegal, while doing exactly
# that a couple of dozen lines further down.

# Every function body that executes on a worker thread, i.e. everything reachable as a
# threading.Thread target in src/gui.py. Note "_audit_thread" and "_worker" each name
# more than one function (Quality Audit / Category Audit; three download workers) — the
# AST walk matches by name, so all of them are covered.
WORKER_BODIES = ("run_clustering", "_run_suggest", "_audit_thread", "_kba_thread",
                 "_sop_thread", "_impact_thread", "_disp_thread", "_worker")

# Reads that only make sense against a live widget.
WIDGET_READS = ("currentText", "isChecked", "currentIndex", "currentData",
                "_gather_metadata_mapping", "_gather_settings")


def _worker_functions(tree):
    """Every FunctionDef in the tree whose name is a known worker body."""
    import ast

    return [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in WORKER_BODIES]


def test_no_worker_body_reads_a_widget():
    import ast

    gui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gui.py"
    )
    with open(gui_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    workers = _worker_functions(tree)
    assert workers, f"none of {WORKER_BODIES} found - were they renamed?"

    offenders = []
    for fn in workers:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            attr = getattr(node.func, "attr", None)
            if attr in WIDGET_READS:
                offenders.append(f"{fn.name} (line {node.lineno}): .{attr}()")

    assert not offenders, (
        "worker threads must not read Qt widgets - capture the value on the GUI thread "
        "and pass it in (see `settings` / `mapping`):\n" + "\n".join(offenders)
    )


def test_the_worker_bodies_this_guards_still_exist():
    """If a worker body is renamed the guard above silently stops covering it, so make
    the rename fail loudly here instead."""
    import ast

    gui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gui.py"
    )
    with open(gui_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found = {fn.name for fn in _worker_functions(tree)}
    missing = set(WORKER_BODIES) - found
    assert not missing, f"worker bodies no longer present under these names: {missing}"


def test_run_clustering_receives_the_mapping_instead_of_reading_it():
    import ast

    gui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gui.py"
    )
    with open(gui_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_clustering")
    args = [a.arg for a in fn.args.args]
    assert "mapping" in args, "run_clustering must be handed the mapping, not gather it"
    assert "settings" in args


# --- deferred error dialogs must not close over `except ... as e` -----------
# Python deletes the exception binding when the except block exits. _post defers the
# lambda to the GUI thread, so `str(e)` inside it raised NameError *instead of* showing
# the dialog: on a failure in Quality Audit, KBA, SOP or Impact Analysis the user saw no
# error at all, just a re-enabled button and a stderr traceback.
def test_the_exception_binding_is_dead_after_the_block():
    """Pin the language behaviour this guards against, so the guard's reason stays clear."""
    queued = []
    try:
        raise ValueError("boom")
    except ValueError as e:                     # noqa: F841 - the point of the test
        queued.append(lambda: str(e))    # noqa: F821 - demonstrating the bug
    with pytest.raises(NameError):
        queued[0]()


def test_deferred_error_dialogs_bind_the_message_eagerly():
    """No _post'd lambda may reference the except-binding directly.

    Scans the worker bodies for `_post(lambda: ... str(e) ...)`, which is the exact shape
    that silently swallowed four features' error dialogs.
    """
    import io
    import os
    import re

    src = io.open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "src", "gui.py"),
        encoding="utf-8").read()

    offenders = [
        line.strip() for line in src.splitlines()
        if re.search(r"_post\(lambda:.*\bstr\(e\)", line)
        or re.search(r"_post\(lambda:.*[^a-zA-Z_]e\b(?!\w)", line)
    ]
    assert not offenders, (
        "these lambdas run after `e` is deleted, so they raise NameError instead of "
        f"showing the dialog: {offenders}")
