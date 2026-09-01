"""Shared test fixtures.

Two things this unlocks that the suite previously lacked:

1. **Qt widget testing.** No test imported ``src.gui`` before, so 46% of the codebase
   had zero coverage. The ``qapp`` fixture provides the single process-wide
   ``QApplication`` Qt requires, on the offscreen platform so it works headless.
   We deliberately do *not* depend on pytest-qt: it isn't installed in the project
   venv, and nothing here needs ``qtbot`` beyond a running event loop.

2. **A shared LLM stub.** Every caller now goes through
   ``llm.create_chat_completion(messages=...)`` returning
   ``choices[0]["message"]["content"]`` — clustering, kba, sop, disposition and
   category_audit alike. ``FakeLLM`` also keeps the plain-callable form a real
   ``Llama`` supports, so the stub stays faithful to the API, but nothing in ``src/``
   exercises it any more: kba/sop were switched off raw completions (they had
   hardcoded Phi-3 markup that never matched the Gemma/Qwen models actually shipped).
"""
import os

import pytest

# Must be set before any Qt import so widgets work without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    """The one QApplication for the whole session (Qt forbids a second)."""
    pytest.importorskip("PySide6", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # Intentionally not calling app.quit()/deleteLater(): tearing down the
    # QApplication mid-session can crash subsequent Qt tests in the same process.


@pytest.fixture
def drain(qapp):
    """Return a callable that pumps the event loop so queued cross-thread
    signal deliveries land before assertions run."""
    from PySide6.QtCore import QTimer

    def _drain(timeout_ms=800):
        QTimer.singleShot(timeout_ms, qapp.quit)
        qapp.exec()

    return _drain


# ---------------------------------------------------------------------------
# LLM stub
# ---------------------------------------------------------------------------
class FakeLLM:
    """Stand-in for a llama-cpp ``Llama``, supporting both calling conventions.

    ``replies`` maps a substring of the prompt to the text to return; the first
    match wins, otherwise ``default`` is used. ``calls`` counts invocations so a
    test can assert the model was (or was not) consulted.
    """

    def __init__(self, replies=None, default=""):
        self.replies = replies or {}
        self.default = default
        self.calls = 0
        self.prompts = []

    def _reply_for(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        return self.default

    # Raw-completion form. A real Llama supports it; no caller in src/ uses it.
    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": self._reply_for(prompt)}]}

    # The form every caller in src/ uses.
    def create_chat_completion(self, messages, **kwargs):
        prompt = messages[0]["content"] if messages else ""
        return {"choices": [{"message": {"content": self._reply_for(prompt)}}]}


@pytest.fixture
def fake_llm():
    """Factory so a test can build a FakeLLM with its own canned replies."""
    return FakeLLM


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
@pytest.fixture
def tickets_df():
    """A small, realistic ticket frame with the metadata columns the analysis
    modules look for (resolution times, reopen/reassign counts, priority)."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        "number": [f"INC{i:04d}" for i in range(1, 13)],
        "short_description": [
            "password reset required", "password reset required",
            "cannot login to portal", "VPN keeps dropping",
            "VPN keeps dropping", "printer out of toner",
            "SAP GUI crashes on launch", "SAP GUI crashes on launch",
            "disk space alert on server", "disk space alert on server",
            "new laptop request", "mailbox full",
        ],
        "close_notes": [
            "reset the password", "reset the password",
            "unlocked the account", "reconnected the tunnel",
            "reconnected the tunnel", "replaced the toner",
            "reinstalled the client", "reinstalled the client",
            "cleared old logs", "cleared old logs",
            "provisioned the device", "archived old mail",
        ],
        "Cluster_ID": [0, 0, 0, 1, 1, -1, 2, 2, 3, 3, -1, -1],
        "Repetitive Category": [
            "Access & Authorization", "Access & Authorization", "Access & Authorization",
            "Network & Connectivity", "Network & Connectivity", "Non-Repetitive",
            "Business Applications", "Business Applications",
            "Infrastructure & Servers", "Infrastructure & Servers",
            "Non-Repetitive", "Non-Repetitive",
        ],
        "Repetitive Subcategory": [
            "Password Reset", "Password Reset", "Password Reset",
            "VPN Drops", "VPN Drops", "Non-Repetitive",
            "SAP GUI Crash", "SAP GUI Crash",
            "Disk Space Alert", "Disk Space Alert",
            "Non-Repetitive", "Non-Repetitive",
        ],
        "priority": ["3 - Moderate"] * 6 + ["2 - High"] * 6,
        "business_duration": [3600, 5400, 1800, 7200, 3600, 900,
                              10800, 9000, 2700, 3600, 14400, 1200],
        "reopen_count": [0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0],
        "reassignment_count": [0, 0, 1, 3, 2, 0, 1, 0, 0, 1, 4, 0],
        "close_code": ["Solved (Permanently)"] * 8 + ["Solved (Work Around)"] * 4,
    })


@pytest.fixture
def cluster_data():
    """Matches the shape TicketClusterer.get_cluster_data() returns."""
    return {
        0: {"keywords": ["password", "reset"], "subcategory": "Password Reset",
            "category": "Access & Authorization",
            "sample_docs": ["password reset required"], "sample_doc_indices": [0]},
        1: {"keywords": ["vpn", "drop"], "subcategory": "VPN Drops",
            "category": "Network & Connectivity",
            "sample_docs": ["VPN keeps dropping"], "sample_doc_indices": [3]},
        2: {"keywords": ["sap", "gui"], "subcategory": "SAP GUI Crash",
            "category": "Business Applications",
            "sample_docs": ["SAP GUI crashes on launch"], "sample_doc_indices": [6]},
        3: {"keywords": ["disk", "space"], "subcategory": "Disk Space Alert",
            "category": "Infrastructure & Servers",
            "sample_docs": ["disk space alert on server"], "sample_doc_indices": [8]},
    }
