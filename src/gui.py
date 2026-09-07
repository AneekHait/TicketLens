"""
Modern PySide6 GUI for TicketLens (local AI ticket analytics).
Features: sidebar navigation, wizard workflow, dashboard layout, dark theme.
"""

import os
import sys
import copy
import html
import random
import logging
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QDialog,
    QCheckBox, QTextEdit, QFrame, QStackedWidget, QSplitter,
)
from PySide6.QtCore import Qt, Signal, QObject, QSize
from PySide6.QtGui import QFont, QColor, QPalette

# Ensure project root is in path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import load_config, save_config, DEFAULTS, LLM_MODELS, LLM_CUSTOM_SENTINEL, EMBEDDING_MODELS, ACCELERATION_OPTIONS, embedding_dir_complete
from src.logger import get_logger, get_latest_log_file
from src.validation import validate_clustering_settings, validate_model_path, validate_excel_file

logger = get_logger()

# Lazy imports for faster startup
_pd = None
_TicketClusterer = None


def get_pandas():
    """Lazy load pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def get_clusterer_class():
    """Lazy load TicketClusterer (loads heavy ML libs)."""
    global _TicketClusterer
    if _TicketClusterer is None:
        from src.clustering import TicketClusterer
        _TicketClusterer = TicketClusterer
    return _TicketClusterer


# ---------------------------------------------------------------
# Thread-safe signal bridge
# ---------------------------------------------------------------
class WorkerSignals(QObject):
    """Signals for background thread communication."""
    status = Signal(str)
    progress = Signal(float)
    finished = Signal(bool, str)
    result = Signal(object)
    # Proactive model downloads (Settings page)
    model_download_status = Signal(str)
    model_download_done = Signal(bool, str)
    embed_download_done = Signal(bool, str)
    accel_build_done = Signal(bool, str)  # OpenVINO INT8 build finished (ok, label)
    suggest_done = Signal(object)  # Min-Cluster-Size suggestion result dict
    # Automation-opportunity (disposition) flow — emitted from its worker thread.
    disposition_progress = Signal(str, float)   # (status message, 0..1 progress)
    disposition_done = Signal(object, object)   # (results, summary)
    disposition_error = Signal(str)             # error message
    # LLM category-audit flow — emitted from its worker thread.
    category_audit_progress = Signal(str, float)  # (status message, 0..1 progress)
    category_audit_done = Signal(object)          # list[proposal dict]
    category_audit_error = Signal(str)            # error message
    # Generic marshaller: emit a zero-arg callable to run it on the GUI thread.
    # Use this instead of QTimer.singleShot(0, fn) from a worker thread — a QTimer
    # created on a thread with no event loop never starts, so those calls never fire.
    call_on_main = Signal(object)


class _LogEmitter(QObject):
    """Bridges logging records (any thread) to the GUI thread via a signal."""
    line = Signal(str, str)  # (levelname, formatted_message)


class QtLogHandler(logging.Handler):
    """A logging.Handler that streams every log record into the live console.

    emit() may run on a worker thread; the Signal delivers to the GUI thread
    via a queued connection (same pattern as WorkerSignals)."""
    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            self.emitter.line.emit(record.levelname, self.format(record))
        except Exception:
            self.handleError(record)


class _StreamEmitter(QObject):
    """Carries raw stdout/stderr text (any thread) to the GUI console."""
    text = Signal(str)


class _StreamTee:
    """Wraps a stdout/stderr stream: writes pass through to the real stream
    AND mirror into the live console (so tqdm download bars, model-loading
    progress, and any print() show 'like a terminal'). isatty() is False so
    libraries don't emit ANSI control codes."""
    def __init__(self, original, emitter):
        self._original = original
        self._emitter = emitter

    def write(self, s):
        try:
            if self._original is not None:
                self._original.write(s)
        except Exception:
            pass
        try:
            if s:
                self._emitter.text.emit(s)
        except Exception:
            pass
        return len(s) if s else 0

    def flush(self):
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Delegate everything else (encoding, fileno, buffer, …) to the real stream.
        return getattr(self._original, name)


# ---------------------------------------------------------------
# Presentation layer (src/ui/) — see src/ui/__init__.py for the split.
# Imported by name rather than with a star so the controller's dependencies stay
# readable, and so ruff can still flag anything that stops being used.
# ---------------------------------------------------------------
from src.ui.theme import (APP_NAME, APP_TAGLINE, APP_VERSION, APP_LICENSE, AUTHOR,
                          AUTHOR_URL, PROJECT_URL, ISSUES_URL,
                          COLORS, MONO, MONO_FAMILY, STYLESHEET,
                          apply_dark_titlebar, install_wheel_guard, make_metric_card,
                          styled_button)
from src.ui.widgets import ConsolePanel, Sidebar, _PivotTreeItem
from src.ui.dialogs import (CategoryAuditReviewDialog, InfoDialog,
                            MinClusterSizePickerDialog)
from src.ui.pages import AnalysisPage, ClusteringPage, ImportPage, SettingsPage



# ---------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------
class ClusterApp(QMainWindow):
    """Main application window with sidebar navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_TAGLINE}")
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(1200, max(960, int(avail.width() * 0.88)))
            target_h = min(800, max(600, int(avail.height() * 0.85)))
            self.resize(target_w, target_h)
        else:
            self.resize(1200, 800)
        self.setMinimumSize(960, 580)

        # State
        self.df = None
        self.excel_file = None
        self.sheet_names = []
        self.model_path = ""
        self.source_file_path = ""
        self.clusterer = None
        self.cluster_data = None
        self.analysis_results = {}
        self.kba_articles = []
        self.sop_documents = []
        self.selected_text_cols = []
        # Serializes the LLM-backed analyses (KBA/SOP/Fishbone/Disposition/Audit):
        # llama-cpp is not safe to call from two threads at once, and these
        # features each spawn their own worker. Only one may hold the model.
        self._llm_busy = False

        # Load config
        self.config = load_config()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar (full height, left)
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self._on_page_changed)
        self.sidebar.btn_cache.clicked.connect(self.clear_cache)
        main_layout.addWidget(self.sidebar)

        # Page stack
        self.pages = QStackedWidget()
        self.import_page = ImportPage()
        self.settings_page = SettingsPage(self.config)
        self.clustering_page = ClusteringPage()
        self.analysis_page = AnalysisPage()
        self.pages.addWidget(self.import_page)
        self.pages.addWidget(self.settings_page)
        self.pages.addWidget(self.clustering_page)
        self.pages.addWidget(self.analysis_page)

        # Live activity log docked under the pages (IDE-style output panel)
        self.console_panel = ConsolePanel()
        self.sidebar.btn_logs.clicked.connect(self._show_console)

        # Pages + a contextual Back/Next wizard footer
        pages_container = QWidget()
        pc_layout = QVBoxLayout(pages_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(0)
        pc_layout.addWidget(self.pages, 1)

        nav = QFrame()
        nav.setStyleSheet(
            f"background: {COLORS['bg_secondary']}; border-top: 1px solid {COLORS['border']};"
        )
        nav_row = QHBoxLayout(nav)
        nav_row.setContentsMargins(16, 8, 16, 8)
        self.btn_back = styled_button("◂  Back", COLORS['text_muted'])
        self.btn_next = styled_button("Next  ▸", COLORS['accent'])
        self.btn_back.clicked.connect(lambda: self._nav_step(-1))
        self.btn_next.clicked.connect(lambda: self._nav_step(1))
        nav_row.addWidget(self.btn_back)
        nav_row.addStretch(1)
        nav_row.addWidget(self.btn_next)
        pc_layout.addWidget(nav)

        self.right_split = QSplitter(Qt.Orientation.Vertical)
        self.right_split.addWidget(pages_container)
        self.right_split.addWidget(self.console_panel)
        self.right_split.setStretchFactor(0, 1)
        self.right_split.setStretchFactor(1, 0)
        self.right_split.setCollapsible(0, False)   # never hide the pages
        self.right_split.setCollapsible(1, True)     # console may collapse
        self.right_split.setSizes([680, 100])
        main_layout.addWidget(self.right_split, 1)
        self._update_nav_buttons(0)

        # Thread-safe signals
        self._stop_event = threading.Event()   # set by the Stop button to cancel a run
        # True from the moment a clustering run starts until it finishes, however it
        # finishes. Gates every feature that could replace or rewrite self.df: the
        # clustering worker reads the text at the start and writes labels back minutes
        # later, and it refuses to write onto a frame it didn't read — so a concurrent
        # Quality Audit (which swaps self.df wholesale) would throw the whole run away
        # with an error message blaming the data, which the user knows they never touched.
        # The LLM lease can't cover this: Quality Audit doesn't take one.
        self._run_in_progress = False
        self._signals = WorkerSignals()
        self._signals.status.connect(self._on_worker_status)
        self._signals.progress.connect(self._on_worker_progress)
        self._signals.finished.connect(self._on_clustering_finished)
        self._signals.disposition_progress.connect(self._on_disposition_progress)
        self._signals.disposition_done.connect(self._on_disposition_done)
        self._signals.disposition_error.connect(self._on_disposition_error)
        self._signals.category_audit_progress.connect(self._on_cat_audit_progress)
        self._signals.category_audit_done.connect(self._on_cat_audit_done)
        self._signals.category_audit_error.connect(self._on_cat_audit_error)
        # Runs the emitted callable on the GUI thread (Qt queues cross-thread emissions).
        self._signals.call_on_main.connect(lambda fn: fn())

        # Stream ALL logger output into the live console. Drop any stale
        # QtLogHandler first so repeated construction never stacks handlers.
        for _h in [h for h in logger.handlers if isinstance(h, QtLogHandler)]:
            logger.removeHandler(_h)
        self._log_handler = QtLogHandler()
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.emitter.line.connect(self.console_panel.append_line)
        logger.addHandler(self._log_handler)

        # Mirror stdout/stderr (tqdm download bars, model-loading progress, prints)
        # into the console so it behaves like a terminal. The logger's own
        # StreamHandler captured the original stderr before this, so logger lines
        # aren't duplicated. Install once (offscreen tests may re-construct).
        if not isinstance(sys.stdout, _StreamTee):
            self._stream_emitter = _StreamEmitter()
            self._stream_emitter.text.connect(self.console_panel.append_stream)
            sys.stdout = _StreamTee(sys.stdout, self._stream_emitter)
            sys.stderr = _StreamTee(sys.stderr, self._stream_emitter)
        logger.info("Console attached — ready.")

        # Connect UI actions
        self._connect_signals()

        # Auto-detect model
        self.auto_detect_model()
        self._update_embedding_status()
        self._update_accel_options()

        # Menu bar (File / View / Help)
        self._build_menu_bar()

        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)

        # Stop the mouse wheel from changing combo/spin/slider values (scroll the page instead).
        # Installed per-widget (not app-wide) so it adds no overhead to ordinary events.
        install_wheel_guard(self)

        # Dark Windows title bar to match the terminal theme.
        apply_dark_titlebar(self)

    def minimumSizeHint(self):
        return QSize(960, 580)

    # ---------------------------------------------------------------
    # Menu bar + Help/About dialogs
    # ---------------------------------------------------------------
    def _build_menu_bar(self):
        """Top menu bar: File / View / Help (wired to existing methods + Help dialogs)."""
        mb = self.menuBar()

        # Keep Python references so the QMenu wrappers aren't garbage-collected.
        self.menu_file = mb.addMenu("File")
        # Kept as an attribute so _set_import_enabled can lock it during a run — the
        # button on the Import page isn't the only route to load_file.
        self.act_open = self.menu_file.addAction("Open Excel…", self.load_file)
        self.menu_file.addAction("Save Results…", self.save_file)
        self.menu_file.addSeparator()
        # Sessions carry the whole run (clusters, labels, LLM-derived results), unlike
        # "Save Results…" which writes the frame only — reloading that drops the
        # clusters and disables most of the Analysis page.
        self.act_save_session = self.menu_file.addAction(
            "Save Session…", self._save_session)
        self.act_open_session = self.menu_file.addAction(
            "Open Session…", self._open_session)
        self.menu_file.addSeparator()
        self.menu_file.addAction("Exit", self.close)

        self.menu_view = mb.addMenu("View")
        self.menu_view.addAction("Toggle Activity Log", self.console_panel.toggle)
        self.menu_view.addAction("Clear Cache", self.clear_cache)

        self.menu_help = mb.addMenu("Help")
        self.menu_help.addAction("Quick Start", self._show_quickstart)
        self.menu_help.addAction("How Clustering Works", self._show_concepts)
        self.menu_help.addAction("Troubleshooting", self._show_troubleshooting)
        self.menu_help.addAction("System Info", self._show_system_info)
        self.menu_help.addSeparator()
        self.menu_help.addAction(f"About {APP_NAME}", self._show_about)

    def _html_doc(self, title, inner):
        """Wrap section HTML in a consistent themed document."""
        return (
            f"<div style='font-family:{MONO}; color:{COLORS['text']};'>"
            f"<h2 style='color:{COLORS['accent']}; margin:0 0 8px 0;'>{title}</h2>"
            f"{inner}</div>"
        )

    def _about_html(self):
        a, g, m = COLORS['accent'], COLORS['text_secondary'], COLORS['text_muted']

        def _accel_summary():
            """What acceleration this machine can actually use. Never raises —
            the About dialog must open even if a probe misbehaves."""
            try:
                from src import acceleration as accel
                from src import mlx_backend as mlxb
                if mlxb.is_apple_silicon():
                    return mlxb.describe()
                return ("Acceleration: OpenVINO INT8 available."
                        if accel.openvino_available()
                        else "Acceleration: PyTorch CPU (OpenVINO not installed).")
            except Exception:
                return ""

        banner = (
            "&nbsp;┌─────────────────────────────┐<br>"
            "&nbsp;│ &gt;_ TICKETLENS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │<br>"
            "&nbsp;└─────────────────────────────┘"
        )
        badge = (
            f"<p style='margin:12px 0 0 0; padding:6px 10px; border-radius:4px; "
            f"background:#1a2a3a; color:#6fb3e0; font-size:11px;'>"
            f"<b>Free and open source</b> — licensed under {APP_LICENSE}.<br>"
            f"<span style='font-size:10px; color:{m};'>"
            f"No telemetry and no analytics. The only network traffic is the "
            f"one-time model download from Hugging Face.</span></p>"
        )
        return self._html_doc(
            f"{APP_NAME}",
            f"<pre style='color:{a}; margin:0 0 10px 0;'>{banner}</pre>"
            f"<p><b>Version</b> {APP_VERSION}</p>"
            f"<p style='color:{g};'>"
            f"A privacy-first desktop analytics tool that uses local AI to automatically cluster IT "
            f"support tickets — or any free-text dataset — into meaningful, labelled groups. "
            f"<b>100% local</b> — runs fully offline on your CPU; no data ever leaves your machine."
            f"</p>"
            f"<p><b>What you can do</b><br>"
            f"<span style='color:{g};'>"
            f"&bull; <b>Import</b> any Excel file, pick your sheet and text columns, preview cleaned data<br>"
            f"&bull; <b>Cluster</b> tickets into themed groups with AI-generated labels — cancel anytime with the Stop button<br>"
            f"&bull; <b>Quality Audit</b> — score each cluster for consistency and flag outliers<br>"
            f"&bull; <b>KBA Articles</b> — auto-generate Knowledge Base Articles per cluster<br>"
            f"&bull; <b>SOPs</b> — produce Standard Operating Procedures from recurring patterns<br>"
            f"&bull; <b>Fishbone Analysis</b> — visualise root-cause breakdowns for any cluster<br>"
            f"&bull; <b>Impact Analysis</b> — quantify volume, SLA risk, and business impact per group<br>"
            f"&bull; <b>Automation Disposition</b> — classify clusters by automation opportunity<br>"
            f"&bull; <b>Category Audit</b> — let the AI re-check category assignments and "
            f"approve suggested reassignments<br>"
            f"&bull; <b>Word Cloud</b> — visual summary of dominant terms per cluster<br>"
            f"&bull; <b>Export</b> results to Excel with a generated filename<br>"
            f"</span></p>"
            f"<p><b>AI Label Models (LLM)</b><br>"
            f"<span style='color:{g};'>Controls how clusters are named and content is generated.<br>"
            f"Change in <b>Settings &rarr; AI Model</b>.</span><br>"
            f"&bull; Gemma 4 E2B &nbsp;— default; compact and fast on CPU<br>"
            f"&bull; Gemma 4 E4B / 12B &nbsp;— higher quality, slower and larger<br>"
            f"&bull; Phi-4-mini &nbsp;— 3.8 B, strong instruction following at ~2.5 GB<br>"
            f"&bull; Qwen2.5 0.5B &nbsp;— fastest; for a quick first pass<br>"
            f"&bull; <i>Any GGUF model</i> &nbsp;— pick <b>Custom file…</b> in the "
            f"dropdown and browse to any <code>.gguf</code> with an embedded chat "
            f"template</p>"
            f"<p><b>Embedding Models</b><br>"
            f"<span style='color:{g};'>Controls how text similarity is measured.<br>"
            f"Change in <b>Settings &rarr; Embedding Model</b>.</span><br>"
            f"&bull; bge-base-en-v1.5 &nbsp;— recommended; fast and accurate<br>"
            f"&bull; bge-large-en-v1.5 &nbsp;— higher accuracy, slower<br>"
            f"&bull; gte-base-en-v1.5 / gte-large &nbsp;— strong multilingual support<br>"
            f"&bull; Harrier 270 M / 0.6 B &nbsp;— 2026 SOTA decoder-only, 32 k context<br>"
            f"&bull; Qwen3 Embedding &nbsp;— instruction-tuned for nuanced retrieval<br>"
            f"&bull; MiniLM L6 &nbsp;— lightest model, fastest inference</p>"
            f"<p><b>Author</b><br>{AUTHOR} "
            f"(<a style='color:{a};' href='{AUTHOR_URL}'>{AUTHOR_URL}</a>)</p>"
            f"<p><b>Project</b><br>"
            f"<a style='color:{a};' href='{PROJECT_URL}'>{PROJECT_URL}</a><br>"
            f"<span style='color:{g};'>Report a bug or request a feature via "
            f"<a style='color:{a};' href='{ISSUES_URL}'>GitHub Issues</a>.</span></p>"
            f"<p style='color:{m};'>Built with PySide6 · llama.cpp · BERTopic · UMAP · HDBSCAN · "
            f"sentence-transformers · OpenVINO and MLX (optional acceleration).</p>"
            f"<p style='color:{m};'>{_accel_summary()}</p>"
            f"{badge}"
        )

    def _quickstart_html(self):
        g = COLORS['text_secondary']
        return self._html_doc(
            "Quick Start",
            f"<ol style='color:{g}; line-height:1.6;'>"
            "<li><b>Import Data</b> — load an Excel file and pick the sheet.</li>"
            "<li><b>Select columns</b> — check the text columns to cluster "
            "(e.g. description, short description).</li>"
            "<li><b>Settings</b> — choose the <b>Embedding Model</b> and, for AI cluster names, "
            "enable <b>AI Naming</b> and pick an <b>LLM</b>. Use <b>Download now</b> to fetch a model "
            "ahead of time. Set the <b>min cluster size</b>.</li>"
            "<li><b>Clustering</b> — click <b>[ ▶ RUN CLUSTERING ]</b>. Progress streams live into the "
            "<b>activity.log</b> console at the bottom.</li>"
            "<li><b>Save</b> — when complete, save the labelled results to a new Excel file; explore the "
            "<b>Analysis</b> tab for audits, KBAs, SOPs and word clouds.</li>"
            "</ol>"
        )

    # Hoisted out of the f-strings below: a backslash escape inside an f-string
    # *expression* is only legal from Python 3.12 (PEP 701), while run.bat and the
    # README both accept 3.11. Ruff's 3.11 target flags it as a syntax error.
    _H_UMAP = "UMAP — Uniform Manifold Approximation and Projection"
    _H_HDBSCAN = "HDBSCAN — density-based clustering"

    def _concepts_html(self):
        a, g, m = COLORS['accent'], COLORS['text_secondary'], COLORS['text_muted']
        grn = COLORS['accent_green']

        def h(t):
            return f"<p style='color:{a}; font-weight:bold; margin:14px 0 4px 0;'>{t}</p>"

        return self._html_doc(
            "How Clustering Works",
            f"<p style='color:{g};'>TicketLens groups tickets by <b>meaning</b>, not keywords. "
            f"Your text passes through four stages:</p>"
            f"<p style='color:{grn}; margin:2px 0 6px 0;'>Embeddings &rarr; UMAP &rarr; HDBSCAN "
            f"&rarr; LLM labels</p>"
            f"<p style='color:{g};'>1. turn text into numbers &middot; 2. simplify &middot; "
            f"3. group &middot; 4. name each group.</p>"

            f"{h('Embeddings (sentence-transformers)')}"
            f"<p style='color:{g};'>A <b>sentence-transformer</b> model reads each ticket and turns it "
            f"into a vector \u2014 a list of numbers (often 384 or 768 of them) that captures its meaning. "
            f"Tickets about the same problem get similar vectors even when the wording differs "
            f"(\u201ccan\u2019t log in\u201d \u2248 \u201cpassword not working\u201d). "
            f"<b style='color:{a};'>Biggest quality lever:</b> a stronger embedding model yields cleaner, "
            f"more meaningful clusters but is slower on CPU.</p>"

            f"{h(self._H_UMAP)}"
            f"<p style='color:{g};'>Embeddings have hundreds of dimensions, where density-based grouping "
            f"breaks down (the \u201ccurse of dimensionality\u201d). UMAP compresses them to a handful of "
            f"dimensions while keeping similar tickets close, so HDBSCAN can find dense groups reliably.</p>"
            f"<ul style='color:{g}; line-height:1.6; margin:2px 0;'>"
            f"<li><b>n_neighbors</b> (default 15) \u2014 local vs. global focus. Low (~5) emphasises fine "
            f"detail \u2192 more, smaller, specific clusters; high (~50) captures broad structure \u2192 "
            f"fewer, larger themes.</li>"
            f"<li><b>n_components</b> (default 5) \u2014 how many dimensions to reduce to. More keeps detail "
            f"but can reintroduce the sparsity that hurts clustering; ~5 is a good balance.</li>"
            f"<li><b>min_dist</b> (default 0.0) \u2014 how tightly points may pack. <b>0.0</b> is best for "
            f"clustering (lets dense clumps form); higher values spread points out (good for plots, worse "
            f"for clusters).</li>"
            f"<li><b>metric</b> (default euclidean) \u2014 how distance is measured. <i>euclidean</i> is the "
            f"safe default; <i>cosine</i> compares by orientation and can suit some embeddings.</li>"
            f"<li><b>UMAP CPU threads</b> \u2014 Reproducible (1 core, identical results every run) vs. "
            f"Parallel (all cores, faster but slightly different each run). A speed/reproducibility "
            f"trade-off, not a quality one.</li>"
            f"</ul>"

            f"{h(self._H_HDBSCAN)}"
            f"<p style='color:{g};'>Unlike k-means, HDBSCAN doesn\u2019t need you to pick the number of "
            f"clusters up front, finds clusters of different size and density, and labels genuine one-offs "
            f"as <b>noise</b> instead of forcing them into a group.</p>"
            f"<ul style='color:{g}; line-height:1.6; margin:2px 0;'>"
            f"<li><b>min cluster size</b> (the main slider) \u2014 the smallest number of tickets that counts "
            f"as a cluster. <b style='color:{a};'>Biggest tuning lever:</b> small \u2192 many narrow "
            f"clusters; large \u2192 fewer, broader ones.</li>"
            f"<li><b>min_samples</b> (default auto) \u2014 how conservative clustering is. Higher \u2192 "
            f"stricter, denser cores and more noise; lower \u2192 more inclusive. Blank = auto (matches "
            f"min cluster size).</li>"
            f"<li><b>selection</b> (eom vs leaf) \u2014 <i>eom</i> tends to give fewer, larger clusters; "
            f"<i>leaf</i> gives more, finer-grained ones.</li>"
            f"</ul>"
            f"<p style='color:{m}; font-size:11px;'>These parameters live under "
            f"<b>Settings \u2192 Advanced UMAP &amp; HDBSCAN Settings</b>.</p>"

            f"{h('Quick tuning cheatsheet')}"
            f"<ul style='color:{g}; line-height:1.6; margin:2px 0;'>"
            f"<li>Too many tiny clusters? \u2014 raise <b>min cluster size</b> or <b>n_neighbors</b>.</li>"
            f"<li>One giant blurry cluster? \u2014 lower <b>min cluster size</b> or <b>n_neighbors</b>.</li>"
            f"<li>Too much \u201cnoise\u201d / Non-Repetitive? \u2014 lower <b>min_samples</b> or "
            f"<b>min cluster size</b>.</li>"
            f"</ul>"
        )

    def _troubleshooting_html(self):
        g, code = COLORS['text_secondary'], COLORS['accent_orange']
        def c(t):
            return f"<code style='color:{code};'>{t}</code>"
        return self._html_doc(
            "Troubleshooting",
            f"<ul style='color:{g}; line-height:1.6;'>"
            f"<li><b>llama-cpp-python install fails</b> — install the prebuilt CPU wheel:<br>"
            f"{c('pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu')}</li>"
            "<li><b>Model won't download</b> — some Hugging Face models are license-gated and need an "
            "HF token. This app uses ungated mirrors; if a download stalls, check your network/proxy.</li>"
            "<li><b>No cluster names / keyword-only labels</b> — enable <b>AI Naming</b> and make sure an "
            "LLM is selected and downloaded (Settings).</li>"
            "<li><b>Too few / too many clusters</b> — adjust <b>min cluster size</b> in Settings.</li>"
            f"<li><b>Logs</b> — everything streams to the <b>activity.log</b> console; full files are in "
            f"the {c('logs/')} folder (Help isn't needed — use <b>View → Toggle Activity Log</b>).</li>"
            "</ul>"
        )

    def _system_info_html(self):
        from importlib.metadata import version, PackageNotFoundError

        def pkg(dist):
            try:
                return version(dist)
            except PackageNotFoundError:
                return "n/a"

        g = COLORS['text_secondary']
        py = sys.version.split()[0]
        llm_sel = self.settings_page.llm_dropdown.currentText()
        emb_sel = self.settings_page.embedding_dropdown.currentText()
        emb_present, _ = self._embedding_present(emb_sel)
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        spec = LLM_MODELS.get(llm_sel)
        llm_present = bool(spec) and os.path.exists(os.path.join(models_dir, spec["filename"]))
        log_path = get_latest_log_file() or "n/a"

        rows = [
            ("App version", APP_VERSION),
            ("Python", py),
            ("PySide6", pkg("PySide6")),
            ("llama-cpp-python", pkg("llama-cpp-python")),
            ("sentence-transformers", pkg("sentence-transformers")),
            ("transformers", pkg("transformers")),
            ("bertopic", pkg("bertopic")),
            ("LLM (selected)", f"{llm_sel}  [{'downloaded' if llm_present else 'not downloaded'}]"),
            ("Embedding (selected)", f"{emb_sel}  [{'downloaded' if emb_present else 'not downloaded'}]"),
            ("Latest log", log_path),
        ]
        body = "".join(
            f"<tr><td style='color:{COLORS['accent']}; padding:2px 16px 2px 0; "
            f"vertical-align:top; white-space:nowrap;'>{html.escape(k)}</td>"
            f"<td style='color:{g};'>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        return self._html_doc("System Info", f"<table>{body}</table>")

    def _show_about(self):
        InfoDialog(f"About {APP_NAME}", self._about_html(), self).exec()

    def _show_quickstart(self):
        InfoDialog("Quick Start", self._quickstart_html(), self).exec()

    def _show_concepts(self):
        InfoDialog("How Clustering Works", self._concepts_html(), self).exec()

    def _show_troubleshooting(self):
        InfoDialog("Troubleshooting", self._troubleshooting_html(), self).exec()

    def _show_system_info(self):
        InfoDialog("System Info", self._system_info_html(), self).exec()

    # ---------------------------------------------------------------
    # Signal connections
    # ---------------------------------------------------------------
    def _connect_signals(self):
        """Wire up all UI signals to handler methods."""
        p = self.import_page
        p.btn_load.clicked.connect(self.load_file)
        p.sheet_dropdown.currentTextChanged.connect(self.on_sheet_selected)
        p.btn_preview.clicked.connect(self.show_preview)

        s = self.settings_page
        s.llm_dropdown.currentTextChanged.connect(self.on_llm_selection_changed)
        s.btn_download_model.clicked.connect(self.download_selected_model)
        s.embedding_dropdown.currentTextChanged.connect(self.on_embedding_selection_changed)
        s.btn_download_embed.clicked.connect(self.download_selected_embedding)
        s.accel_dropdown.currentTextChanged.connect(self.on_accel_selection_changed)
        s.btn_build_accel.clicked.connect(self.build_accelerated_model)
        s.btn_install_ov.clicked.connect(self.install_openvino_support)
        s.btn_install_mlx.clicked.connect(self.install_mlx_support)
        s.btn_manage_sw.clicked.connect(self.manage_stopwords)
        s.chk_llm.stateChanged.connect(self.toggle_llm_ui)
        s.btn_suggest_cs.clicked.connect(self._suggest_min_cluster_size)
        self._signals.model_download_status.connect(self._on_model_download_status)
        self._signals.model_download_done.connect(self._on_model_download_done)
        self._signals.embed_download_done.connect(self._on_embed_download_done)
        self._signals.accel_build_done.connect(self._on_accel_build_done)
        self._signals.suggest_done.connect(self._on_suggest_done)

        c = self.clustering_page
        c.btn_run.clicked.connect(self.start_clustering_thread)
        c.btn_stop.clicked.connect(self.stop_clustering)
        c.btn_save.clicked.connect(self.save_file)
        c.btn_wordcloud.clicked.connect(self._open_wordcloud_studio)
        # Min Cluster Size is mirrored on both pages. Seed from the Settings page (the
        # config-backed one) so they can't start out of sync, then keep them in lockstep.
        # No recursion guard needed: QSpinBox.setValue doesn't re-emit valueChanged when
        # the value is unchanged, so the echo stops on the second hop.
        c.spin_cluster_size.setValue(s.spin_cluster_size.value())
        c.spin_cluster_size.valueChanged.connect(s.spin_cluster_size.setValue)
        s.spin_cluster_size.valueChanged.connect(c.spin_cluster_size.setValue)
        c.btn_suggest_cs.clicked.connect(self._suggest_min_cluster_size)
        # Restore the persisted mid-run picker preference (the page defaults it to on).
        c.chk_pick_after_embed.setChecked(
            self.config.get("clustering", {}).get("pick_size_after_embedding", True))

        a = self.analysis_page
        a.btn_overview.clicked.connect(self._run_main_theme)
        a.btn_audit.clicked.connect(self._run_quality_audit)
        a.btn_audit_export.clicked.connect(self._export_audit)
        a.btn_gen_kba.clicked.connect(self._run_kba_generation)
        a.btn_kba_export.clicked.connect(self._export_kba)
        a.kba_selector.currentTextChanged.connect(self._on_kba_selected)
        a.btn_gen_sop.clicked.connect(self._run_sop_generation)
        a.btn_sop_export.clicked.connect(self._export_sop)
        a.sop_selector.currentTextChanged.connect(self._on_sop_selected)
        a.btn_fishbone_gen.clicked.connect(self._run_fishbone)
        a.btn_fishbone_save.clicked.connect(self._save_fishbone_html)
        a.btn_impact.clicked.connect(self._run_impact_analysis)
        a.btn_impact_export.clicked.connect(self._export_impact)
        a.btn_disp.clicked.connect(self._run_disposition_analysis)
        a.btn_disp_export.clicked.connect(self._export_disposition)
        a.btn_wc_open_studio.clicked.connect(self._open_wordcloud_studio)
        a.btn_cat_audit.clicked.connect(self._run_category_audit)
        a.btn_pivot_export.clicked.connect(self._export_category_pivot)
        # Recompute the pivot whenever its tab is opened, so it reflects any
        # Category Audit reassignments made since the last view.
        for _b in a.tab_buttons:
            if _b.text() == "Category Pivot":
                _b.clicked.connect(self._refresh_category_pivot)

    _PAGE_NAMES = ["Import Data", "Settings", "Clustering", "Analysis"]

    def _on_page_changed(self, index):
        self.pages.setCurrentIndex(index)
        if hasattr(self, "btn_next"):
            self._update_nav_buttons(index)

    def _nav_step(self, delta):
        """Back/Next wizard navigation between pages (keeps the sidebar in sync)."""
        new = max(0, min(self.pages.count() - 1, self.pages.currentIndex() + delta))
        if new != self.pages.currentIndex():
            self.sidebar._on_nav_clicked(new)  # updates sidebar highlight + switches page

    def _update_nav_buttons(self, index):
        last = self.pages.count() - 1
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setEnabled(index < last)
        self.btn_back.setText(f"◂  {self._PAGE_NAMES[index - 1]}" if index > 0 else "◂  Back")
        self.btn_next.setText(f"{self._PAGE_NAMES[index + 1]}  ▸" if index < last else "Done")

    def _show_console(self):
        """Expand the live console and scroll to the latest line (View Logs button)."""
        self.console_panel.set_collapsed(False)
        try:
            self.right_split.setSizes([520, 280])
        except Exception:
            pass
        sb = self.console_panel.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_worker_status(self, msg):
        self.clustering_page.lbl_status.setText(f"› {msg}")

    def _on_worker_progress(self, value):
        self.clustering_page.progress.setValue(int(value * 100))

    # ---------------------------------------------------------------
    # Settings gathering
    # ---------------------------------------------------------------
    def _gather_settings(self):
        """Collect all UI settings into a dict compatible with TicketClusterer."""
        s = self.settings_page
        hdb_raw = s.edit_hdbscan_samples.text().strip()
        hdb_samples = int(hdb_raw) if hdb_raw.isdigit() else None

        # Keys with no widget behind them. Each block below is rebuilt wholesale from the
        # UI, so anything not listed there is dropped — which meant these four could be
        # set in config/user_settings.json and never reach the engine. Same treatment the
        # llm/cache blocks already get, just applied per-key inside mixed blocks.
        cfg_pp = self.config.get("preprocessing", {})
        cfg_sw = self.config.get("stopwords", {})
        cfg_cl = self.config.get("clustering", {})

        # Start from the loaded config so sections with no UI at all survive. save_config
        # writes _compute_diff(DEFAULTS, this dict) over the whole file, and _compute_diff
        # only walks the keys it is given — so a section missing from here was deleted from
        # user_settings.json on every RUN. That silently discarded hand-edited `audit`,
        # `kba`, `sop`, `analysis`, `disposition` and `category_audit` overrides. Same
        # defence as _gather_metadata_mapping: seed from config, then let the UI win.
        gathered = copy.deepcopy(self.config)
        gathered.update({
            "preprocessing": {
                "remove_emails": s.chk_rm_emails.isChecked(),
                "remove_urls": s.chk_rm_urls.isChecked(),
                "remove_ticket_ids": s.chk_rm_ticket_ids.isChecked(),
                "remove_timestamps": s.chk_rm_timestamps.isChecked(),
                "remove_phone_numbers": s.chk_rm_phones.isChecked(),
                "remove_ip_addresses": s.chk_rm_ips.isChecked(),
                "remove_file_paths": s.chk_rm_paths.isChecked(),
                "remove_special_chars": s.chk_rm_special.isChecked(),
                "remove_boilerplate": s.chk_rm_boilerplate.isChecked(),
                "min_token_length": s.spin_min_token.value(),
                # No widget: edited in config/user_settings.json only.
                "custom_regex_patterns": list(cfg_pp.get("custom_regex_patterns", [])),
                "custom_boilerplate_patterns": list(
                    cfg_pp.get("custom_boilerplate_patterns", [])),
            },
            "stopwords": {
                "use_english_stopwords": s.chk_english_sw.isChecked(),
                "use_it_stopwords": s.chk_it_sw.isChecked(),
                # No widget: the "Manage Custom Stopwords" dialog writes the file, and
                # this list is the config-side equivalent.
                "custom_stopwords": list(cfg_sw.get("custom_stopwords", [])),
            },
            "clustering": {
                # Persisted so the value survives a restart, and so
                # suggest_min_cluster_size() sees the real current value (it reads
                # `current` from here to pin it into the candidate sweep).
                "min_cluster_size": s.spin_cluster_size.value(),
                "umap_n_neighbors": s.spin_umap_neighbors.value(),
                "umap_n_components": s.spin_umap_components.value(),
                "umap_min_dist": s.spin_umap_mindist.value(),
                "umap_metric": s.combo_umap_metric.currentText(),
                "umap_parallel_mode": s._umap_parallel_modes[s.combo_umap_parallel.currentIndex()],
                "hdbscan_min_samples": hdb_samples,
                "hdbscan_cluster_selection_method": s.combo_hdbscan_method.currentText(),
                # No widget: edited in config/user_settings.json only.
                "hdbscan_cluster_selection_epsilon": cfg_cl.get(
                    "hdbscan_cluster_selection_epsilon", 0.0),
                # Lives on the Clustering page, not Settings — it's a per-run workflow
                # choice, but persisted so unchecking it sticks across restarts.
                "pick_size_after_embedding": self.clustering_page.chk_pick_after_embed.isChecked(),
            },
            "embedding": {
                "model_name": s.embedding_dropdown.currentText(),
                "acceleration": ACCELERATION_OPTIONS.get(s.accel_dropdown.currentText(), "pytorch"),
                "batch_size": self.config.get("embedding", {}).get("batch_size", 32),
            },
            # Non-UI blocks carried through from the loaded config so edits in
            # config/user_settings.json (llm params, cache directory) reach the
            # clusterer instead of being silently dropped.
            "llm": dict(self.config.get("llm", {})),
            "cache": {
                "enabled": True,
                "directory": self.config.get("cache", {}).get("directory", "cache"),
            },
            # Persisted so the column mapping survives a restart. The DEFAULTS section
            # existed but nothing ever wrote to or read from it, so users had to re-map
            # priority / resolution-time / reopen columns on every launch.
            "metadata_mapping": self._gather_metadata_mapping(),
        })
        return gathered

    def _gather_metadata_mapping(self):
        """Collect the metadata column mapping: config first, dropdowns on top.

        Only 8 of the 13 roles have a dropdown. The rest — notably ``reopen_col`` and
        ``reassignment_col``, which the disposition Automatability score reads — are
        config-only, so they must be carried through from ``config["metadata_mapping"]``
        or they would be silently dropped every time this mapping is saved.

        Must be called on the GUI thread (it reads QComboBox.currentText()).
        """
        # Start from every known role so consumers always see the full set of keys.
        mapping = {role: None for role in DEFAULTS.get("metadata_mapping", {})}
        for role, col in (self.config.get("metadata_mapping") or {}).items():
            mapping[role] = col
        # The dropdowns are the live source of truth for the roles they cover.
        for role_key, dd in self.settings_page.metadata_dropdowns.items():
            val = dd.currentText()
            mapping[role_key] = val if val != "-- Not mapped --" else None
        return mapping

    def _update_metadata_dropdowns(self):
        """Populate metadata dropdowns with current DataFrame columns."""
        if self.df is None:
            return
        cols = ["-- Not mapped --"] + list(self.df.columns)
        auto_map = {
            "priority_col": ["priority", "prio", "severity"],
            "resolution_time_col": ["resolution time", "resolve time", "ttr", "time to resolve", "resolution_time"],
            "sla_status_col": ["sla", "sla status", "sla_status", "sla breach"],
            "assignment_group_col": ["assignment group", "assigned group", "team", "group", "assignment_group"],
            "created_date_col": ["created", "created date", "open date", "opened", "created_date"],
            "resolved_date_col": ["resolved", "resolved date", "close date", "closed", "resolved_date"],
            "category_col": ["category", "incident category", "type", "ticket type"],
            "resolution_notes_col": ["resolution", "resolution notes", "close notes", "solution", "resolution_notes"],
        }
        # A previously saved mapping wins over auto-detection: the user chose it
        # deliberately, and auto_map's name patterns can't know about a column the
        # heuristics don't recognise. Only honoured for columns the new sheet actually
        # has, so switching files falls back to auto-detection rather than going blank.
        saved = self.config.get("metadata_mapping", {}) or {}

        for role_key, dd in self.settings_page.metadata_dropdowns.items():
            dd.clear()
            dd.addItems(cols)
            dd.setEnabled(True)

            remembered = saved.get(role_key)
            if remembered and remembered in self.df.columns:
                dd.setCurrentText(remembered)
                continue

            matched = False
            for pattern in auto_map.get(role_key, []):
                for col in self.df.columns:
                    if pattern == col.lower().strip():
                        dd.setCurrentText(col)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                dd.setCurrentText("-- Not mapped --")

    # ---------------------------------------------------------------
    # File & Data
    # ---------------------------------------------------------------
    def load_file(self):
        """Open an Excel file and populate sheet dropdown."""
        # Reachable from the File menu as well as the Import button, so the guard lives
        # here rather than relying on the button being disabled.
        if self._busy_with_run("Open Excel"):
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)"
        )
        if not file_path:
            return
        ok, err = validate_excel_file(file_path)
        if not ok:
            QMessageBox.warning(self, "Open Excel File", err)
            return
        try:
            pd = get_pandas()
            # Open the workbook BEFORE committing it as the source: if this throws
            # (corrupt / locked / password-protected), source_file_path must keep
            # pointing at the previous file rather than the rejected one, since the
            # intermediate-save and default export name are both derived from it.
            excel_file = pd.ExcelFile(file_path)
            # Close the previous workbook now that the new one opened successfully.
            # pd.ExcelFile keeps a ZipFile handle open for the whole session (deliberately
            # — on_sheet_selected re-reads from it), so replacing it without closing
            # leaked one handle per file loaded and kept every previously-opened .xlsx
            # locked on Windows, which shows up as an unexplained "file in use" in Excel.
            self._close_excel_file()
            self.source_file_path = file_path
            self.excel_file = excel_file
            self.sheet_names = self.excel_file.sheet_names

            self.import_page.lbl_file.setText(os.path.basename(file_path))
            self.import_page.lbl_file.setStyleSheet(f"color: {COLORS['text']}; border: none;")

            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.import_page.lbl_file_info.setText(
                f"Path: {file_path}   |   Size: {size_mb:.1f} MB   |   Sheets: {len(self.sheet_names)}"
            )
            self.import_page.file_info_frame.show()

            dd = self.import_page.sheet_dropdown
            # blockSignals: clear()/addItems() emit currentTextChanged even on a disabled
            # widget, which would fire on_sheet_selected and reload self.df as a side
            # effect of merely listing the sheets. The explicit selection below is what
            # should load a sheet.
            dd.blockSignals(True)
            dd.clear()
            dd.addItems(self.sheet_names)
            dd.blockSignals(False)
            # Don't hand the lock back mid-run (unreachable today thanks to the guard in
            # load_file, but this line was previously the way the lock got undone).
            dd.setEnabled(not self._run_in_progress)

            if len(self.sheet_names) == 1:
                dd.setCurrentIndex(0)
                self.on_sheet_selected(self.sheet_names[0])   # resets derived state itself
            else:
                dd.insertItem(0, "-- Select a sheet --")
                dd.setCurrentIndex(0)
                self.df = None
                # No sheet chosen yet, so anything derived from the old file is stale.
                self._reset_derived_state()
                self.clustering_page.btn_run.setEnabled(False)

            self.clustering_page.set_status(
                f"File loaded. {len(self.sheet_names)} sheet(s) found."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _close_excel_file(self):
        """Release the open workbook handle, if any. Safe to call repeatedly."""
        old = getattr(self, "excel_file", None)
        if old is None:
            return
        try:
            old.close()
        except Exception as e:
            logger.debug(f"Could not close the previous workbook: {e}")
        self.excel_file = None

    def _reset_derived_state(self):
        """Drop everything derived from the previously loaded data.

        Without this, loading a new file keeps the old clusters around: the analysis
        guards only check `cluster_data`/`df` for truthiness, so Overview, Impact and
        Quality Audit would happily score the NEW rows against the OLD clusters and
        report plausible-looking nonsense.
        """
        self.clusterer = None
        self.cluster_data = None
        self.analysis_results = {}
        self.kba_articles = []
        self.sop_documents = []
        self.selected_text_cols = []
        self.clustering_page.results_card.hide()
        try:
            self._refresh_category_pivot()   # falls back to its "run clustering" placeholder
        except Exception as e:
            logger.warning(f"Could not reset the category pivot: {e}")

    def on_sheet_selected(self, sheet_name):
        """Load the selected sheet and populate columns."""
        if not sheet_name or sheet_name.startswith("--"):
            return
        try:
            pd = get_pandas()
            self.df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            # New rows => every previously derived result is stale.
            self._reset_derived_state()
            self._populate_columns()
            self._update_metadata_dropdowns()
            self.clustering_page.btn_run.setEnabled(True)
            self.import_page.btn_preview.setEnabled(True)
            self.import_page.set_ready(
                True, f"Sheet '{sheet_name}' loaded \u2014 {len(self.df):,} rows, {len(self.df.columns)} columns"
            )
            self.clustering_page.update_metric_card(
                self.clustering_page.lbl_docs, f"{len(self.df):,}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sheet: {e}")

    def _populate_columns(self):
        """Populate column checkboxes from the loaded DataFrame."""
        for cb in self.import_page.checkboxes:
            cb.setParent(None)
        self.import_page.checkboxes = []

        while self.import_page.col_layout.count() > 0:
            item = self.import_page.col_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        for col in self.df.columns:
            cb = QCheckBox(col)
            cb.setObjectName("colCheck")  # styled via global QSS (cheap; no per-widget re-polish)
            self.import_page.col_layout.addWidget(cb)
            self.import_page.checkboxes.append(cb)

        self.import_page.col_layout.addStretch()
        self._populate_disposition_columns()

    def _populate_disposition_columns(self):
        """Populate the Automation-Opportunities resolution-column checkboxes,
        pre-checking columns whose name looks like resolution / work / close notes."""
        from src.config import RESOLUTION_TEXT_PATTERNS
        a = self.analysis_page
        if not hasattr(a, "disp_col_layout"):
            return
        for cb in a.disp_checkboxes:
            cb.setParent(None)
        a.disp_checkboxes = []
        while a.disp_col_layout.count() > 0:
            item = a.disp_col_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for col in self.df.columns:
            lc = str(col).lower()
            cb = QCheckBox(col)
            cb.setObjectName("colCheck")
            cb.setChecked(any(p in lc for p in RESOLUTION_TEXT_PATTERNS))
            a.disp_col_layout.addWidget(cb)
            a.disp_checkboxes.append(cb)
        a.disp_col_layout.addStretch()

    def show_preview(self):
        """Show a sample of documents before and after preprocessing."""
        if self.df is None:
            return
        selected_cols = [cb.text() for cb in self.import_page.checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "Preview", "Select at least one column first.")
            return

        from src.preprocessing import preprocess_document
        docs = self.df[selected_cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()
        n = min(8, len(docs))
        indices = random.sample(range(len(docs)), n)
        settings = self._gather_settings()["preprocessing"]

        lines = []
        for idx in indices:
            original = docs[idx][:200]
            cleaned = preprocess_document(docs[idx], settings=settings)[:200]
            lines.append(f"--- Doc #{idx + 1} ---")
            lines.append(f"BEFORE: {original}")
            lines.append(f"AFTER : {cleaned}\n")

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Preview: Before / After Cleaning")
        dlg.setDetailedText("\n".join(lines))
        dlg.setText(f"Showing {n} random samples from {len(docs)} documents.\nClick 'Show Details...' to see them.")
        dlg.exec()

    # ---------------------------------------------------------------
    # Model & Utility
    # ---------------------------------------------------------------
    def auto_detect_model(self):
        """Select the curated LLM whose .gguf is already in models/, else default to
        the Recommended one (it will download on first run)."""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        present = set()
        if os.path.exists(models_dir):
            present = {f for f in os.listdir(models_dir) if f.endswith('.gguf')}

        chosen = next(
            (label for label, spec in LLM_MODELS.items() if spec["filename"] in present),
            next(iter(LLM_MODELS)),  # fallback: the Recommended (first) entry
        )
        dd = self.settings_page.llm_dropdown
        dd.blockSignals(True)
        dd.setCurrentText(chosen)
        dd.blockSignals(False)
        self.model_path = ""  # curated selection; the path is resolved at run time
        self._update_llm_status()

    def select_model(self):
        """Open a file dialog for a custom .gguf (triggered by the 'Custom file…' item)."""
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Models (*.gguf)")
        if path:
            self.model_path = path
            self._update_llm_status()
        else:
            # Cancelled: revert to the Recommended curated model.
            dd = self.settings_page.llm_dropdown
            dd.blockSignals(True)
            dd.setCurrentText(next(iter(LLM_MODELS)))
            dd.blockSignals(False)
            self.model_path = ""
            self._update_llm_status()

    def on_llm_selection_changed(self, text):
        """React to LLM dropdown changes: open the file picker for the sentinel,
        otherwise treat it as a curated selection (path resolved at run time)."""
        if text == LLM_CUSTOM_SENTINEL:
            self.select_model()
        else:
            self.model_path = ""
            self._update_llm_status()

    def _update_llm_status(self):
        """Show the resolved LLM filename + download state, and reveal the
        'Download now' button only for a curated model that isn't present yet."""
        sp = self.settings_page
        lbl = sp.lbl_model_status
        btn = sp.btn_download_model
        sel = sp.llm_dropdown.currentText()
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        if sel == LLM_CUSTOM_SENTINEL:
            btn.setVisible(False)
            if self.model_path and os.path.exists(self.model_path):
                lbl.setText(os.path.basename(self.model_path))
                lbl.setStyleSheet(f"color: {COLORS['accent_green']}; border: none;")
            else:
                lbl.setText("No custom file selected")
                lbl.setStyleSheet(f"color: {COLORS['warning']}; border: none;")
            return

        # MLX models are HF repos fetched by mlx-lm on first use, not GGUF files
        # in models/, so there is nothing local to look for and no Download
        # button to offer. Say that, rather than leaving the label blank.
        from src.config import MLX_LLM_MODELS
        mlx_spec = MLX_LLM_MODELS.get(sel)
        if mlx_spec:
            btn.setVisible(False)
            lbl.setText(f"{mlx_spec['repo_id']} — downloads on first run "
                        f"(~{mlx_spec['size_gb']:.1f} GB, Apple GPU)")
            lbl.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            return

        spec = LLM_MODELS.get(sel)
        if not spec:
            lbl.setText("")
            btn.setVisible(False)
            return
        if os.path.exists(os.path.join(models_dir, spec["filename"])):
            lbl.setText(f"{spec['filename']} — downloaded")
            lbl.setStyleSheet(f"color: {COLORS['accent_green']}; border: none;")
            btn.setVisible(False)
        else:
            lbl.setText(f"{spec['filename']} — not downloaded (~{spec['size_gb']:.0f} GB)")
            lbl.setStyleSheet(f"color: {COLORS['warning']}; border: none;")
            btn.setText("Download now")
            btn.setEnabled(True)
            btn.setVisible(True)

    def download_selected_model(self):
        """Download the selected curated LLM now (background thread) instead of
        waiting for the first clustering run."""
        sp = self.settings_page
        sel = sp.llm_dropdown.currentText()
        if sel == LLM_CUSTOM_SENTINEL:
            return  # custom file: nothing to fetch
        spec = LLM_MODELS.get(sel)
        if not spec:
            return
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if os.path.exists(os.path.join(models_dir, spec["filename"])):
            self._update_llm_status()  # already there
            return

        sp.btn_download_model.setEnabled(False)
        sp.btn_download_model.setText("Downloading…")
        sp.lbl_model_status.setText(f"Downloading {spec['filename']} (~{spec['size_gb']:.0f} GB)…")
        sp.lbl_model_status.setStyleSheet(f"color: {COLORS['warning']}; border: none;")

        def _worker(label=sel):
            from src.clustering import resolve_llm_path
            try:
                def _cb(msg, progress=None):
                    self._signals.model_download_status.emit(msg)
                path = resolve_llm_path(label, log=_cb)
                self._signals.model_download_done.emit(bool(path), label)
            except Exception as e:
                logger.error(f"Model download failed: {e}")
                self._signals.model_download_done.emit(False, label)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_download_status(self, msg):
        """Worker-thread download progress text (main thread)."""
        self.settings_page.lbl_model_status.setText(msg)

    def _on_model_download_done(self, success, label):
        """Re-enable the button and refresh status when a download finishes."""
        sp = self.settings_page
        sp.btn_download_model.setText("Download now")
        sp.btn_download_model.setEnabled(True)
        self._update_llm_status()
        if not success:
            QMessageBox.warning(
                self, "Download failed",
                "Could not download the model. Check your network connection and try again.",
            )

    # --- Embedding model: same proactive download as the LLM -----------------
    def _embedding_present(self, label):
        """(present, local_dir) for an embedding label — mirrors get_embedding_model.

        Uses embedding_dir_complete() so a partial download (missing the tiny
        module configs) is reported as *not* present and re-fetched, instead of
        showing a misleading "Downloaded".
        """
        hf = EMBEDDING_MODELS.get(label)
        if not hf:
            return False, None
        local_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'embedding', hf.replace('/', '_'),
        )
        return embedding_dir_complete(local_dir), local_dir

    def on_embedding_selection_changed(self, text):
        self._update_embedding_status()
        self._update_accel_options()

    def _update_embedding_status(self):
        """Show whether the selected embedding model is local, and reveal the
        'Download now' button only when it isn't."""
        sp = self.settings_page
        lbl = sp.lbl_embed_status
        btn = sp.btn_download_embed
        present, _ = self._embedding_present(sp.embedding_dropdown.currentText())
        if present:
            lbl.setText("Downloaded")
            lbl.setStyleSheet(f"color: {COLORS['accent_green']}; border: none;")
            btn.setVisible(False)
        else:
            lbl.setText("Not downloaded — fetched automatically on first run")
            lbl.setStyleSheet(f"color: {COLORS['warning']}; border: none;")
            btn.setText("Download now")
            btn.setEnabled(True)
            btn.setVisible(True)

    def download_selected_embedding(self):
        """Download the selected embedding model now (background thread)."""
        sp = self.settings_page
        label = sp.embedding_dropdown.currentText()
        present, _ = self._embedding_present(label)
        if present:
            self._update_embedding_status()
            return

        sp.btn_download_embed.setEnabled(False)
        sp.btn_download_embed.setText("Downloading…")
        sp.lbl_embed_status.setText(f"Downloading {label.split(' (')[0]}…")
        sp.lbl_embed_status.setStyleSheet(f"color: {COLORS['warning']}; border: none;")

        def _worker(lbl=label):
            from src.clustering import get_embedding_model
            try:
                get_embedding_model(lbl)  # downloads + saves into models/embedding/
                self._signals.embed_download_done.emit(True, lbl)
            except Exception as e:
                logger.error(f"Embedding download failed: {e}")
                self._signals.embed_download_done.emit(False, lbl)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_embed_download_done(self, success, label):
        sp = self.settings_page
        sp.btn_download_embed.setText("Download now")
        sp.btn_download_embed.setEnabled(True)
        self._update_embedding_status()
        if not success:
            QMessageBox.warning(
                self, "Download failed",
                "Could not download the embedding model. Check your network connection and try again.",
            )

    # --- Acceleration (experimental OpenVINO INT8) ---------------------------
    def _update_accel_options(self):
        """Grey the INT8 option when OpenVINO isn't installed or the selected
        model isn't OV-compatible; show the install / build buttons accordingly."""
        sp = self.settings_page
        from src import acceleration as accel
        from src.config import ACCELERATION_OPTIONS
        ov = accel.openvino_available()
        hf = EMBEDDING_MODELS.get(sp.embedding_dropdown.currentText())
        compatible = accel.model_supports_acceleration(hf) if hf else False
        int8_enabled = ov and compatible

        # Apple Silicon options: MPS needs the hardware; MLX needs the hardware,
        # the package, and a model on the verified list.
        from src import mlx_backend as mlxb
        apple = mlxb.is_apple_silicon()
        mlx_pkg = mlxb.mlx_embeddings_available()
        mlx_model = accel.model_supports_mlx(hf or "")
        enabled_for = {
            "openvino_int8_cpu": int8_enabled,
            "mps": apple and mlxb.mps_available(),
            "mlx": apple and mlx_pkg and mlx_model,
        }
        for idx, label in enumerate(sp._accel_labels):
            value = ACCELERATION_OPTIONS[label]
            if value not in enabled_for:
                continue
            try:
                sp.accel_dropdown.model().item(idx).setEnabled(enabled_for[value])
            except Exception:
                pass

        sel_value = ACCELERATION_OPTIONS.get(sp.accel_dropdown.currentText(), "pytorch")
        if apple:
            sp.btn_install_ov.setVisible(False)
            if sel_value == "mps":
                sp.lbl_accel_status.setText(
                    "Apple GPU (Metal). Falls back to CPU per model if an op is unsupported."
                    if enabled_for["mps"] else "Metal/MPS not available — using PyTorch CPU.")
                sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
                sp.btn_install_mlx.setVisible(False)
                sp.btn_build_accel.setVisible(False)
                return

            if sel_value == "mlx":
                if not mlx_pkg:
                    sp.lbl_accel_status.setText("MLX not installed — using PyTorch.")
                    sp.btn_install_mlx.setVisible(True)
                elif not mlx_model:
                    sp.lbl_accel_status.setText("This model is not MLX-verified — using PyTorch.")
                    sp.btn_install_mlx.setVisible(False)
                else:
                    sp.lbl_accel_status.setText(
                        "Experimental. Verified against PyTorch on load; rejected if it disagrees.")
                    sp.btn_install_mlx.setVisible(False)
                sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
                sp.btn_build_accel.setVisible(False)
                return

            # Default / PyTorch on Apple Silicon
            sp.lbl_accel_status.setText("Experimental. PyTorch is the reproducible default.")
            sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            sp.btn_install_mlx.setVisible(not mlx_pkg)
            sp.btn_build_accel.setVisible(False)
            return

        # Non-Apple platforms (Windows / Linux):
        sp.btn_install_mlx.setVisible(False)
        if not ov:
            sp.lbl_accel_status.setText("OpenVINO not installed — using PyTorch.")
            sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            sp.btn_install_ov.setVisible(True)
        elif not compatible:
            sp.lbl_accel_status.setText("This model runs on PyTorch (OpenVINO INT8 not supported for it).")
            sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            sp.btn_install_ov.setVisible(False)
        else:
            sp.lbl_accel_status.setText("Experimental. PyTorch is the reproducible default.")
            sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            sp.btn_install_ov.setVisible(False)

        sp.btn_build_accel.setVisible(
            sel_value == "openvino_int8_cpu" and int8_enabled and self.df is not None)

    def on_accel_selection_changed(self, text):
        self._update_accel_options()

    def build_accelerated_model(self):
        """Build + cache the OpenVINO INT8 model now, calibrated on the loaded data
        (background thread; progress streams to the activity log)."""
        sp = self.settings_page
        from src.config import ACCELERATION_OPTIONS
        cols = [cb.text() for cb in self.import_page.checkboxes if cb.isChecked()]
        if self.df is None or not cols:
            QMessageBox.information(
                self, "Load data first",
                "Load a ticket file and select text column(s) before building the accelerated model.")
            return
        if ACCELERATION_OPTIONS.get(sp.accel_dropdown.currentText()) != "openvino_int8_cpu":
            return
        label = sp.embedding_dropdown.currentText()
        sample = self.df[cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()[:300]

        sp.btn_build_accel.setEnabled(False)
        sp.btn_build_accel.setText("Building…")
        self._show_console()

        def _worker(lbl=label, docs=sample):
            from src.clustering import get_embedding_model_resolved
            try:
                _model, backend, _prec = get_embedding_model_resolved(
                    lbl, acceleration="openvino_int8_cpu", calib_docs=docs)
                self._signals.accel_build_done.emit(backend == "openvino", lbl)
            except Exception as e:
                logger.error(f"Accelerated build failed: {e}")
                self._signals.accel_build_done.emit(False, lbl)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_accel_build_done(self, success, label):
        sp = self.settings_page
        sp.btn_build_accel.setText("Build accelerated model")
        sp.btn_build_accel.setEnabled(True)
        self._update_accel_options()
        if success:
            sp.lbl_accel_status.setText("INT8 model ready — clustering will use OpenVINO.")
            sp.lbl_accel_status.setStyleSheet(f"color: {COLORS['accent_green']}; border: none;")
        else:
            QMessageBox.warning(
                self, "Build failed",
                "Could not build the OpenVINO INT8 model. Clustering will use PyTorch.")

    def install_openvino_support(self):
        """Launch install_openvino.bat in its own console window."""
        import subprocess
        if sys.platform != "win32":
            QMessageBox.information(
                self, "OpenVINO not supported on this OS",
                "OpenVINO acceleration is configured for Windows in TicketLens. "
                "On Apple Silicon Macs, use Apple Metal / MPS or MLX acceleration instead.")
            return
        bat = os.path.join(os.path.dirname(os.path.dirname(__file__)), "install_openvino.bat")
        if not os.path.isfile(bat):
            QMessageBox.warning(self, "Installer missing",
                                "install_openvino.bat was not found in the app folder.")
            return
        try:
            subprocess.Popen(["cmd", "/c", "start", "TicketLens - Install OpenVINO", bat])
            QMessageBox.information(
                self, "Installing OpenVINO",
                "OpenVINO is installing in a new window. When it finishes, restart TicketLens "
                "to enable INT8 acceleration.")
        except Exception as e:
            QMessageBox.warning(self, "Install failed", f"Could not launch the installer: {e}")

    def install_mlx_support(self):
        """Launch install_mlx.sh in its own Terminal window."""
        import subprocess
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cmd_path = os.path.join(base_dir, "install_mlx.command")
        sh_path = os.path.join(base_dir, "install_mlx.sh")
        if not os.path.isfile(sh_path):
            QMessageBox.warning(self, "Installer missing",
                                "install_mlx.sh was not found in the app folder.")
            return
        try:
            try:
                os.chmod(sh_path, 0o755)
            except Exception:
                pass
            if os.path.isfile(cmd_path):
                try:
                    os.chmod(cmd_path, 0o755)
                except Exception:
                    pass
                subprocess.Popen(["open", cmd_path])
            else:
                subprocess.Popen(["open", "-a", "Terminal", sh_path])
            QMessageBox.information(
                self, "Installing MLX",
                "MLX installation has opened in a new Terminal window. "
                "When it finishes, restart TicketLens to use MLX acceleration.")
        except Exception as e:
            QMessageBox.warning(self, "Install failed", f"Could not launch the installer: {e}")

    def toggle_llm_ui(self):
        """Enable/disable the LLM dropdown based on the AI Naming toggle."""
        enabled = self.settings_page.chk_llm.isChecked()
        self.settings_page.llm_dropdown.setEnabled(enabled)

    def manage_stopwords(self):
        """Open dialog to view/edit custom stopwords."""
        from src.stopwords import _load_custom_stopwords_file, save_custom_stopwords

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Custom Stopwords")
        dlg.setText("Edit custom stopwords (one per line):")

        current = _load_custom_stopwords_file()
        txt = QTextEdit()
        txt.setMinimumSize(360, 300)
        txt.setPlainText("\n".join(current))
        dlg.layout().addWidget(txt, 1, 0, 1, dlg.layout().columnCount())
        dlg.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel)

        if dlg.exec() == QMessageBox.StandardButton.Save:
            raw = txt.toPlainText().strip()
            words = [w.strip() for w in raw.split("\n") if w.strip()]
            save_custom_stopwords(words)
            QMessageBox.information(self, "Saved", f"{len(words)} custom stopwords saved.")

    def clear_cache(self):
        """Remove all cached embeddings."""
        from src.clustering import resolve_cache_dir
        cache_dir = resolve_cache_dir(self.config.get("cache", {}).get("directory", "cache"))
        if not os.path.exists(cache_dir):
            QMessageBox.information(self, "Cache", "No cache directory found.")
            return
        files = [f for f in os.listdir(cache_dir) if f.endswith(".npy")]
        if not files:
            QMessageBox.information(self, "Cache", "Cache is already empty.")
            return
        reply = QMessageBox.question(
            self, "Clear Cache",
            f"Delete {len(files)} cached embedding file(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for f in files:
                os.remove(os.path.join(cache_dir, f))
            # Also drop the in-memory embedding models. Previously "Clear Cache" only
            # deleted the .npy files on disk and left the loaded weights resident.
            from src.clustering import clear_embedding_model_cache
            freed = clear_embedding_model_cache()
            extra = f" {freed} loaded model(s) released." if freed else ""
            QMessageBox.information(self, "Cache", f"Cache cleared.{extra}")

    def _default_save_name(self):
        """Auto-generated output name: dataset_embedding_AImodel_minCluster.xlsx."""
        def slug(text):
            # Keep the short model name (drop the " (High · Moderate…)" suffix),
            # turn spaces/slashes into '-', drop punctuation Windows dislikes.
            base = (text or "").split(" (")[0].strip()
            out = [(ch if (ch.isalnum() or ch in "._-") else "-")
                   for ch in base if ch.isalnum() or ch in "._- /\\"]
            return "".join(out).strip("-_.")

        sp = self.settings_page
        dataset = (slug(os.path.splitext(os.path.basename(self.source_file_path))[0])
                   if self.source_file_path else "results") or "results"
        embedding = slug(sp.embedding_dropdown.currentText()) or "emb"
        if sp.chk_llm.isChecked():
            sel = sp.llm_dropdown.currentText()
            if sel == LLM_CUSTOM_SENTINEL and self.model_path:
                ai = slug(os.path.splitext(os.path.basename(self.model_path))[0]) or "llm"
            else:
                ai = slug(sel) or "llm"
        else:
            ai = "keyword"
        return f"{dataset}_{embedding}_{ai}_{sp.spin_cluster_size.value()}.xlsx"

    def save_file(self):
        """Save clustered results to Excel."""
        if self.df is None:
            QMessageBox.warning(self, "Save Results", "Load and cluster data first.")
            return
        base_dir = os.path.dirname(self.source_file_path) if self.source_file_path else ""
        default_path = os.path.join(base_dir, self._default_save_name())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", default_path, "Excel Files (*.xlsx)"
        )
        if path:
            try:
                self.df.to_excel(path, index=False)
            except Exception as e:
                logger.error(f"Save failed: {e}")
                QMessageBox.critical(self, "Save Results", f"Could not save file:\n{e}")
                return
            QMessageBox.information(self, "Saved", "File saved.")

    # ---------------------------------------------------------------
    # Sessions (save/reload a whole run)
    # ---------------------------------------------------------------
    def _save_session(self):
        """Write the whole run to a .tsz so it can be reopened without re-running."""
        from src.session import SESSION_EXT, SessionError, save_session

        if self.df is None:
            QMessageBox.warning(self, "Save Session", "Load data first.")
            return
        if not self.cluster_data:
            QMessageBox.information(
                self, "Save Session",
                "Run clustering first — a session stores the clusters and labels, "
                "which is what makes reopening it worthwhile.")
            return
        if self._busy_with_run("Save Session", read_only=True):
            return

        base = os.path.splitext(self._default_save_name())[0] + SESSION_EXT
        base_dir = os.path.dirname(self.source_file_path) if self.source_file_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", os.path.join(base_dir, base),
            f"TicketLens Session (*{SESSION_EXT})")
        if not path:
            return
        if not path.lower().endswith(SESSION_EXT):
            path += SESSION_EXT

        try:
            dropped = save_session(
                path,
                df=self.df,
                cluster_data=self.cluster_data,
                analysis_results=self.analysis_results,
                kba_articles=self.kba_articles,
                sop_documents=self.sop_documents,
                selected_text_cols=self.selected_text_cols,
                settings=self.config,
                source_file=self.source_file_path,
                sheet_name=self.import_page.sheet_dropdown.currentText() or None,
                app_version=APP_VERSION,
            )
        except (SessionError, OSError) as e:
            logger.error(f"Session save failed: {e}")
            QMessageBox.critical(self, "Save Session", f"Could not save the session:\n{e}")
            return

        note = ""
        if dropped:
            # Say so rather than leaving the user to discover the holes later.
            logger.warning(f"Session save dropped unserialisable fields: {dropped}")
            note = f"\n\n{len(dropped)} field(s) could not be stored and will be "
            note += "recomputed on load."
        QMessageBox.information(
            self, "Session Saved",
            f"Saved {len(self.df):,} tickets and {len(self.cluster_data)} clusters.{note}")

    def _open_session(self):
        """Restore a saved run: clusters, labels and LLM-derived results."""
        from src.session import SESSION_EXT, SessionError, load_session

        if self._busy_with_run("Open Session"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", f"TicketLens Session (*{SESSION_EXT})")
        if not path:
            return

        try:
            data = load_session(path)
        except SessionError as e:
            QMessageBox.critical(self, "Open Session", str(e))
            return
        except Exception as e:
            logger.error(f"Session load failed: {e}")
            QMessageBox.critical(
                self, "Open Session", f"Could not open the session:\n{e}")
            return

        # A session replaces the loaded workbook, so let go of the old handle and clear
        # everything derived from it before installing the restored state.
        self._close_excel_file()
        self.sheet_names = []
        self.source_file_path = path
        self._reset_derived_state()

        self.df = data["df"]
        self.cluster_data = data["cluster_data"]
        self.analysis_results = dict(data["analysis_results"])
        self.kba_articles = data["kba_articles"]
        self.sop_documents = data["sop_documents"]
        self.selected_text_cols = data["selected_text_cols"]

        self._apply_restored_session(data["manifest"])

    def _apply_restored_session(self, manifest):
        """Bring the UI to the same state a finished run leaves it in.

        Deliberately reuses the run-success path's three steps rather than inventing a
        second one, so a restored session and a fresh run cannot drift apart.
        """
        ip = self.import_page
        cp = self.clustering_page
        name = manifest.get("source_file") or os.path.basename(self.source_file_path)
        ip.lbl_file.setText(f"{name} (session)")
        ip.lbl_file_info.setText(
            f"Restored session · {len(self.df):,} rows · "
            f"{len(self.cluster_data)} clusters · saved {manifest.get('saved_at', '?')}")
        ip.set_ready(True, f"Session restored — {len(self.df):,} rows, "
                           f"{len(self.df.columns)} columns")
        # No workbook is open, so the sheet picker has nothing to offer.
        dd = ip.sheet_dropdown
        dd.blockSignals(True)
        dd.clear()
        dd.blockSignals(False)
        dd.setEnabled(False)
        self._populate_columns()
        ip.btn_preview.setEnabled(True)

        cp.set_status("Session restored.")
        cp.results_card.show()
        cp.update_metric_card(cp.lbl_docs, f"{len(self.df):,}")
        cp.update_metric_card(cp.lbl_cols, str(len(self.selected_text_cols)))
        self._populate_fishbone_scope()
        self._refresh_category_pivot()

        if self.analysis_results.get("audit_summary"):
            try:
                self._display_audit_results(self.analysis_results["audit_summary"])
            except Exception as e:
                logger.warning(f"Could not redraw the restored audit summary: {e}")
        if self.kba_articles:
            self._display_kba_results()
        if self.sop_documents:
            self._display_sop_results()

        # The LLM is not part of a session (it is a live native handle), so be explicit
        # about which features need a run rather than letting the user find out.
        recompute = ", ".join(manifest.get("recompute_on_load") or []) or "none"
        logger.info(f"Session restored from {self.source_file_path}; "
                    f"recompute on demand: {recompute}")
        QMessageBox.information(
            self, "Session Restored",
            f"Restored {len(self.df):,} tickets and {len(self.cluster_data)} clusters.\n\n"
            "Overview, Quality Audit, Impact Analysis, Fishbone, Category Pivot and the "
            "exports are ready to use.\n\n"
            "KBA, SOP and Category Audit need a loaded model, so they will ask you to "
            "run clustering — any articles already generated were restored.")

    # ---------------------------------------------------------------
    # Suggest Min Cluster Size
    # ---------------------------------------------------------------
    def _set_suggest_enabled(self, enabled, text=None):
        """Toggle BOTH Suggest buttons (Settings page + Clustering page) together so
        the two mirrors can't drift out of sync. ``text`` is applied when given."""
        for b in (self.settings_page.btn_suggest_cs, self.clustering_page.btn_suggest_cs):
            b.setEnabled(enabled)
            if text is not None:
                b.setText(text)

    def _suggest_min_cluster_size(self):
        """Analyze the loaded data and recommend a Min Cluster Size (background thread)."""
        cols = [cb.text() for cb in self.import_page.checkboxes if cb.isChecked()]
        if self.df is None or not cols:
            QMessageBox.warning(self, "Suggest",
                                "Load data and select at least one text column first.")
            return
        settings = self._gather_settings()
        embedding_model = self.settings_page.embedding_dropdown.currentText()
        use_preprocessing = self.settings_page.chk_preprocess.isChecked()
        self._set_suggest_enabled(False, "Analyzing…")
        # Also block Run for the duration: the sweep builds its own clusterer and shares
        # the status/progress signals with clustering, so overlapping the two interleaves
        # the progress bar and loads the embedding model twice. Clustering already
        # disables Suggest, so this closes the race from both sides.
        self.clustering_page.btn_run.setEnabled(False)
        self._show_console()
        threading.Thread(
            target=self._run_suggest,
            args=(cols, settings, embedding_model, use_preprocessing),
            daemon=True,
        ).start()

    def _run_suggest(self, cols, settings, embedding_model, use_preprocessing):
        """Worker: build docs, run the suggestion sweep, emit the result."""
        try:
            def update_status(msg, progress=None):
                self._signals.status.emit(msg)
                if progress is not None:
                    self._signals.progress.emit(progress)
            docs = self.df[cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()
            TicketClusterer = get_clusterer_class()
            clusterer = TicketClusterer(embedding_model_name=embedding_model, settings=settings)
            result = clusterer.suggest_min_cluster_size(
                docs, use_preprocessing=use_preprocessing, callback=update_status)
            self._signals.suggest_done.emit(result)
        except Exception as e:
            logger.error(f"Suggest failed: {e}")
            self._signals.suggest_done.emit({"error": str(e)})

    def _on_suggest_done(self, result):
        """Show the candidate picker; apply only the size the user confirms (main thread).

        Deliberately does NOT auto-apply the recommendation — the recommended row is
        pre-selected in the picker, so confirming is one click, while Cancel leaves the
        current value untouched."""
        self._set_suggest_enabled(True, "💡  Suggest")
        # Restore Run from the current loaded-data state (the same condition sheet
        # loading gates it on) — placed before the early return so the error path
        # can't leave Run stuck disabled.
        self.clustering_page.btn_run.setEnabled(self.df is not None)
        if not result or "error" in result:
            msg = result.get("error", "unknown error") if result else "no result"
            QMessageBox.warning(self, "Suggest", f"Could not suggest a value:\n{msg}")
            return
        rec = result.get("recommended")
        candidates = result.get("candidates") or []
        reason = result.get("reason", "")

        if not candidates:
            # The sweep couldn't evaluate anything (suggest() fell back to the current
            # value) — there's nothing to choose between, so just report and apply.
            if rec:
                self.clustering_page.spin_cluster_size.setValue(int(rec))
            QMessageBox.information(
                self, "Suggest",
                (f"Suggested Min Cluster Size: {rec}\n\n{reason}" if rec
                 else reason or "No candidates could be evaluated."),
            )
            return

        dlg = MinClusterSizePickerDialog(rec, candidates, reason, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return   # Cancel — leave the current value as-is.
        size = dlg.selected_size()
        if size:
            # Setting one spinbox mirrors to the other via the sync in _connect_signals.
            self.clustering_page.spin_cluster_size.setValue(int(size))

    # ---------------------------------------------------------------
    # Mid-run size picker (called from worker thread after embeddings are ready)
    # ---------------------------------------------------------------
    def _make_embeddings_ready_hook(self, clusterer, update_status):
        """Return a closure suitable for clusterer.run(on_embeddings_ready=...).

        The closure runs in the worker thread: it sweeps HDBSCAN candidate sizes using
        the already-computed embeddings (no re-embedding), then asks the main thread to
        show MinClusterSizePickerDialog and blocks until the user dismisses it. Returns
        (chosen_size, reduced_embeddings) so run() can also skip its own UMAP pass, or
        raises ClusteringCancelled if the user chooses to stop.
        """
        # Free here: this factory is only called from run_clustering, after
        # get_clusterer_class() has already imported the module.
        from src.clustering import ClusteringCancelled

        def _hook(embeddings, current_size):
            # The sweep reports its own 0.05→1.0 progress. Squeeze it into the slice of
            # the run between embedding (0.35) and clustering (0.40) so the bar keeps
            # moving forward instead of racing to the end and snapping back.
            def sweep_status(msg, progress=None):
                if progress is None:
                    update_status(msg)
                else:
                    update_status(msg, 0.35 + 0.05 * max(0.0, min(1.0, progress)))

            try:
                result = clusterer.suggest_min_cluster_size(
                    docs=None,
                    precomputed_embeddings=embeddings,
                    callback=sweep_status,
                    return_reduced=True,
                    should_stop=self._stop_event.is_set,
                    current_size=current_size,
                )
            except ClusteringCancelled:
                # Stop pressed during the sweep: propagate so run() ends now. Returning
                # normally here would show the picker for a run that is already over.
                raise
            except Exception as e:
                # A failed sweep must not fail the run — fall through with the size the
                # user already set and let run() do its own UMAP.
                logger.warning(f"Mid-run size sweep failed; keeping {current_size}: {e}")
                return current_size, None

            reduced = result.get("reduced_embeddings")
            rec = result.get("recommended", current_size)
            candidates = result.get("candidates") or []
            reason = result.get("reason", "")

            if not candidates:
                # Nothing to choose between — don't interrupt the run with an empty dialog.
                logger.info("Mid-run sweep evaluated no candidates; keeping the current size.")
                return current_size, reduced

            # Second check: the sweep may have finished just as Stop was pressed. Showing
            # an app-modal dialog now would strand it on screen after the run ends.
            if self._stop_event.is_set():
                logger.info("Stop requested before the size picker opened; not showing it.")
                raise ClusteringCancelled("Clustering cancelled by user.")

            chosen_box = [current_size]     # mutable cell written by the main thread
            aborted_box = [False]
            ready = threading.Event()

            def _show():
                # Runs as a Qt slot: an exception escaping here would surface as an
                # unhandled exception in the event loop, so it's contained and logged.
                # The finally: matters just as much — without it a raising dialog would
                # leave the worker blocked on ready.wait() and the run would never end.
                try:
                    dlg = MinClusterSizePickerDialog(rec, candidates, reason, self,
                                                     allow_abort=True)
                    # The run is blocked on this dialog, so say so in the title bar —
                    # the default ("…Suggestion") reads as dismissible advice.
                    dlg.setWindowTitle("Choose Min Cluster Size — clustering is paused")
                    accepted = dlg.exec() == QDialog.DialogCode.Accepted
                    if dlg.aborted():
                        # Reuse the normal cancellation path: run() raises at its next
                        # checkpoint, so Stop-from-here behaves exactly like the button.
                        aborted_box[0] = True
                        self._stop_event.set()
                    elif accepted:
                        size = dlg.selected_size()
                        if size:
                            chosen_box[0] = int(size)
                            # Mirror into the spinboxes so the UI reflects what ran.
                            self.clustering_page.spin_cluster_size.setValue(chosen_box[0])
                except Exception as e:
                    logger.error(
                        f"Could not show the Min Cluster Size picker; continuing with "
                        f"{chosen_box[0]}: {e}"
                    )
                finally:
                    ready.set()

            msg = f"Waiting for you to choose a Min Cluster Size (suggested: {rec})…"
            logger.info(msg)        # so the wait is visible in the console and log file
            update_status(msg)
            self._post(_show)
            # Poll rather than block outright purely as a backstop for the case where the
            # queued call never runs at all. It cannot observe a Stop pressed *while* the
            # dialog is up: the dialog is application-modal, so btn_stop is unclickable,
            # and nothing else sets _stop_event. The dialog's own "Stop the run" button is
            # what makes stopping possible here.
            while not ready.wait(0.25):
                if self._stop_event.is_set():
                    logger.info("Stop requested before the size picker appeared.")
                    break
            if aborted_box[0]:
                raise ClusteringCancelled("Clustering cancelled from the size picker.")
            return chosen_box[0], reduced

        return _hook

    # ---------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------
    def start_clustering_thread(self):
        """Validate settings and launch clustering in a background thread."""
        selected_cols = [cb.text() for cb in self.import_page.checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "Warning", "Select at least one column.")
            return

        final_model_path = None
        if self.settings_page.chk_llm.isChecked():
            llm_selection = self.settings_page.llm_dropdown.currentText()
            if llm_selection == LLM_CUSTOM_SENTINEL:
                ok, err = validate_model_path(self.model_path)
                if not ok:
                    QMessageBox.warning(
                        self, "Warning",
                        f"{err}\nPlease choose a valid custom .gguf file or pick a built-in model."
                    )
                    return
                final_model_path = self.model_path
            else:
                # Curated model: confirm a pending multi-GB download before proceeding.
                spec = LLM_MODELS.get(llm_selection)
                if spec:
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                    if not os.path.exists(os.path.join(models_dir, spec["filename"])):
                        reply = QMessageBox.question(
                            self, "Download model?",
                            f"{spec['filename']} (~{spec['size_gb']:.0f} GB) will be "
                            f"downloaded now. Continue?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        )
                        if reply != QMessageBox.StandardButton.Yes:
                            return
                # Pass the label; resolve_llm_path() downloads/resolves in the worker.
                final_model_path = llm_selection

        settings = self._gather_settings()
        min_size = self.settings_page.spin_cluster_size.value()
        num_docs = len(self.df)
        embedding_selection = self.settings_page.embedding_dropdown.currentText()

        ok, err = validate_clustering_settings(
            num_docs, min_size, settings["clustering"]["umap_n_neighbors"]
        )
        if not ok:
            QMessageBox.critical(self, "Invalid Settings", err)
            return

        cp = self.clustering_page
        cp.update_metric_card(cp.lbl_docs, f"{num_docs:,}")
        cp.update_metric_card(cp.lbl_cols, str(len(selected_cols)))
        cp.update_metric_card(cp.lbl_model, embedding_selection.split(" (")[0][:12])

        try:
            from src.clustering import estimate_processing_time, format_time
            est_total, breakdown = estimate_processing_time(num_docs, embedding_selection)
            cp.update_metric_card(cp.lbl_estimate, format_time(est_total))

            if num_docs > 50000:
                # The estimate models one clustering pass. With the mid-run picker on,
                # the size sweep adds several more HDBSCAN fits plus however long the
                # user takes to answer, so say so rather than quoting a low number.
                extra = ""
                if settings["clustering"].get("pick_size_after_embedding", True):
                    extra = ("\nPlus the Min Cluster Size sweep (several extra clustering\n"
                             "passes), then it waits for you to pick a size.\n")
                reply = QMessageBox.question(
                    self, "Large Dataset",
                    f"Processing {num_docs:,} documents.\n\n"
                    f"Estimated time: {format_time(est_total)}\n"
                    f"  - Embeddings: ~{format_time(breakdown['embedding'])}\n"
                    f"  - Clustering: ~{format_time(breakdown['clustering'])}\n"
                    f"{extra}\n"
                    f"Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
        except Exception as e:
            cp.update_metric_card(cp.lbl_estimate, "~")
            logger.warning(f"Could not estimate time: {e}")

        # Clustering drives the LLM itself (topic naming), so it must hold the same
        # lease as the analysis features — llama-cpp is not safe to call from two
        # threads at once. Claim it before any UI state changes so a refusal is a
        # clean no-op. Released in _on_clustering_finished (all outcomes).
        self._clustering_holds_llm = bool(final_model_path)
        if self._clustering_holds_llm and not self._llm_acquire("Clustering"):
            self._clustering_holds_llm = False
            return

        # Navigate to clustering page
        self.pages.setCurrentIndex(2)
        self.sidebar._on_nav_clicked(2)

        self._stop_event.clear()
        self._run_in_progress = True    # cleared in _on_clustering_finished (all outcomes)
        cp.btn_run.setEnabled(False)
        cp.btn_run.setVisible(False)
        cp.btn_stop.setEnabled(True)
        cp.btn_stop.setVisible(True)
        self._set_suggest_enabled(False)
        self._set_import_enabled(False)
        cp.progress.setValue(0)
        cp.results_card.hide()

        # Fresh live console for this run, expanded so the stream is visible.
        self.console_panel.clear()
        self.console_panel.set_collapsed(False)
        self.right_split.setSizes([520, 300])

        save_intermediate = self.settings_page.chk_intermediate.isChecked()
        use_preprocessing = self.settings_page.chk_preprocess.isChecked()

        try:
            save_config(settings)
            # Keep the in-memory copy in step, so switching sheet later in this session
            # re-seeds the dropdowns from what the user actually used rather than from
            # the value loaded at launch.
            self.config["metadata_mapping"] = dict(settings["metadata_mapping"])
        except Exception as e:
            # Non-fatal: the run can still proceed with in-memory settings, but
            # the user's choices won't persist. Surface it via the console log
            # rather than swallowing it silently.
            logger.warning(f"Could not save settings (continuing this run): {e}")

        # Read the dropdowns here, on the GUI thread — the worker must not touch them.
        mapping = self._gather_metadata_mapping()

        thread = threading.Thread(
            target=self.run_clustering,
            args=(selected_cols, final_model_path, min_size, save_intermediate,
                  use_preprocessing, settings, mapping),
            daemon=True,
        )
        thread.start()

    def run_clustering(self, cols, model_path, min_size, save_intermediate,
                       use_preprocessing, settings, mapping=None):
        """Execute the clustering pipeline (runs in background thread).

        mapping: the metadata column mapping, read off the dropdowns on the GUI thread by
        the caller. Passed in rather than gathered here for the same reason as `settings`
        — reading a QWidget from this thread is illegal.
        """
        from src.clustering import ClusteringCancelled
        try:
            self.selected_text_cols = cols

            def update_status(msg, progress=None):
                self._signals.status.emit(msg)
                if progress is not None:
                    self._signals.progress.emit(progress)

            def save_intermediate_file(topics, topic_map):
                try:
                    update_status("Saving intermediate file...", 0.55)
                    temp_df = self.df.copy()
                    temp_df['Cluster_ID'] = topics
                    temp_df['Cluster_Name'] = temp_df['Cluster_ID'].map(topic_map)
                    source_dir = os.path.dirname(self.source_file_path)
                    base_name = os.path.splitext(os.path.basename(self.source_file_path))[0]
                    save_path = os.path.join(source_dir, f"{base_name}_keywords_only.xlsx")
                    temp_df.to_excel(save_path, index=False)
                    update_status(f"Intermediate file saved: {save_path}", 0.58)
                except Exception as e:
                    logger.error(f"Failed to save intermediate: {e}")

            update_status("Preparing data...", 0.0)
            # Remember exactly which frame these docs came from. The results are written
            # back minutes later, and writing them onto a *different* frame would silently
            # mislabel every row (pandas only raises if the row count also differs).
            df_at_start = self.df
            docs = df_at_start[cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()

            update_status("Loading ML libraries...", 0.02)
            TicketClusterer = get_clusterer_class()

            # Release the previous run's GGUF before the new one mmaps another multi-GB
            # file, otherwise a re-run briefly holds two models resident.
            if self.clusterer is not None:
                try:
                    self.clusterer.close_llm()
                except Exception as e:
                    logger.warning(f"Could not release the previous LLM: {e}")

            # Pass the full label; EMBEDDING_MODELS/MODEL_PREFIXES key on it exactly so
            # the model resolves and (for Gemma/Qwen3) the prompt prefix actually applies.
            # Taken from `settings` (captured on the GUI thread by _gather_settings)
            # rather than read off the dropdown here: touching a QWidget from a worker
            # thread is illegal, and it let a mid-run dropdown change swap the model.
            embedding_model = settings["embedding"]["model_name"]
            clusterer = TicketClusterer(
                llm_model_path=model_path,
                embedding_model_name=embedding_model,
                settings=settings,
            )
            self.clusterer = clusterer

            # Read off `settings` (captured on the GUI thread) rather than the checkbox
            # itself — touching a QWidget from this thread is illegal.
            pick_after_embed = settings["clustering"].get("pick_size_after_embedding", True)

            topics, sub_map, macro_map = clusterer.run(
                docs,
                min_cluster_size=min_size,
                callback=update_status,
                intermediate_callback=save_intermediate_file if save_intermediate else None,
                use_preprocessing=use_preprocessing,
                should_stop=self._stop_event.is_set,
                on_embeddings_ready=(
                    self._make_embeddings_ready_hook(clusterer, update_status)
                    if pick_after_embed else None
                ),
            )

            # Refuse to write results onto a frame the run didn't read. The import
            # controls are locked during a run, so this should be unreachable — it's the
            # backstop that turns a silent mislabelling into a clear error.
            if self.df is not df_at_start:
                raise RuntimeError(
                    "The loaded data changed while clustering was running, so the "
                    "results were discarded to avoid mislabelling rows. "
                    "Please re-run clustering."
                )

            self.df['Cluster_ID'] = topics
            # Both columns get the same fallback: the subcategory previously had no
            # fillna, so a Cluster_ID missing from the map left a blank cell here while
            # the category column said "Uncategorized".
            self.df['Repetitive Subcategory'] = self.df['Cluster_ID'].map(sub_map).fillna("Uncategorized")
            self.df['Repetitive Category'] = self.df['Cluster_ID'].map(macro_map).fillna("Uncategorized")

            # Safety net: run every noise ticket through the template matcher so
            # any recognisable category keyword rescues it from "Non-Repetitive".
            # Scoped to Cluster_ID==-1 rows only — real clusters are never touched.
            try:
                import pandas as pd
                from src.clustering import ACCESS_HARD_SIGNAL_RE
                noise_mask = self.df['Repetitive Category'] == 'Non-Repetitive'
                if noise_mask.any() and self.clusterer is not None:
                    text_series = pd.Series(docs, index=self.df.index)
                    noise_texts = text_series[noise_mask]

                    matched = noise_texts.apply(
                        lambda t: self.clusterer._match_category_template([str(t)])
                    )
                    fix_idx = matched.index[matched.notna()]

                    if len(fix_idx):
                        cats = matched[fix_idx]

                        # Specific label for hard-signal access; generic otherwise.
                        hard = text_series[fix_idx].str.contains(ACCESS_HARD_SIGNAL_RE)
                        subcats = cats.apply(lambda c: f'Individual {c} Ticket')
                        subcats = subcats.mask(
                            hard & (cats == 'Access & Authorization'),
                            'Password / MFA / Account Unlock',
                        )

                        self.df.loc[fix_idx, 'Repetitive Category']    = cats.values
                        self.df.loc[fix_idx, 'Repetitive Subcategory'] = subcats.values

                        summary_str = ', '.join(
                            f'{c}: {n}'
                            for c, n in sorted(cats.value_counts().to_dict().items())
                        )
                        logger.info(
                            f"Safety-net: re-tagged {len(fix_idx)} noise tickets "
                            f"→ {summary_str}"
                        )
            except Exception as e:
                logger.warning(f"Safety-net skipped: {e}")

            try:
                self.cluster_data = clusterer.get_cluster_data()
                res_col = (mapping or {}).get("resolution_notes_col")
                if res_col and res_col in self.df.columns:
                    for _cid, cdata in self.cluster_data.items():
                        indices = cdata.get("sample_doc_indices", [])
                        res_notes = []
                        for idx in indices:
                            val = str(self.df.iloc[idx].get(res_col, "")).strip()
                            if val and val.lower() not in ("nan", "none", ""):
                                res_notes.append(val)
                        cdata["resolution_notes"] = res_notes
            except Exception as e:
                logger.warning(f"Could not extract cluster data: {e}")
                self.cluster_data = {}

            self._signals.finished.emit(True, "")
        except ClusteringCancelled:
            logger.info("Clustering cancelled by user.")
            self._signals.finished.emit(False, "__CANCELLED__")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            self._signals.finished.emit(False, str(e))

    def stop_clustering(self):
        """Request cooperative cancellation of the running clustering job.

        Sets the flag the worker checks between steps / inside its loops. Native
        calls already in flight (UMAP/HDBSCAN/embedding/LLM) finish first, so the
        stop takes effect at the next safe checkpoint.
        """
        self._stop_event.set()
        cp = self.clustering_page
        cp.btn_stop.setEnabled(False)
        cp.lbl_status.setText("Stopping — finishing the current step…")
        cp.lbl_status.setStyleSheet(f"color: {COLORS['accent_orange']};")
        logger.info("Stop requested; clustering will cancel at the next checkpoint.")

    def _on_clustering_finished(self, success, error):
        """Handle clustering completion on the main thread."""
        cp = self.clustering_page
        # First statement: every later line can raise, and leaving this set would lock
        # the user out of Quality Audit and file loading for the rest of the session.
        self._run_in_progress = False
        cp.btn_stop.setVisible(False)
        cp.btn_run.setVisible(True)
        cp.btn_run.setEnabled(True)
        self._set_suggest_enabled(True)
        # Single completion slot for success / cancel / error — release the lease and
        # re-open the import controls here so no outcome can leave them stuck.
        if getattr(self, "_clustering_holds_llm", False):
            self._llm_release()
            self._clustering_holds_llm = False
        self._set_import_enabled(True)

        if success:
            cp.progress.setValue(100)
            cp.lbl_status.setText("Clustering complete!")
            cp.lbl_status.setStyleSheet(f"color: {COLORS['accent_green']};")
            cp.results_card.show()
            self._populate_fishbone_scope()
            self._refresh_category_pivot()
            QMessageBox.information(
                self, "Success",
                "Clustering complete. Analysis tools are now available."
            )
        elif error == "__CANCELLED__":
            cp.progress.setValue(0)
            cp.lbl_status.setText("Stopped by user.")
            cp.lbl_status.setStyleSheet(f"color: {COLORS['accent_orange']};")
        else:
            cp.progress.setValue(100)
            cp.lbl_status.setText(f"Error: {error}")
            cp.lbl_status.setStyleSheet(f"color: {COLORS['accent_red']};")
            QMessageBox.critical(self, "Error", error)

    # ---------------------------------------------------------------
    # Analysis: Overview (overall main theme)
    # ---------------------------------------------------------------
    def _run_main_theme(self):
        """Compute and render the overall main-theme summary (deterministic)."""
        if not self.cluster_data or self.df is None:
            QMessageBox.warning(
                self, "Overview",
                "Run clustering first \u2014 the overview summarises your clustered tickets."
            )
            return
        if self._busy_with_run("Overview", read_only=True):
            return
        try:
            from src.impact_analysis import ImpactAnalyzer
            analyzer = ImpactAnalyzer(
                self.df, self._gather_metadata_mapping(), self.cluster_data,
                settings=self.config.get("analysis", {}),
            )
            summary = analyzer.analyze_main_theme()
            self.analysis_results["main_theme"] = summary
            self._display_main_theme(summary)
            self.clustering_page.set_status("Overview summary generated.")
        except Exception as e:
            logger.error(f"Main-theme summary failed: {e}")
            QMessageBox.critical(self, "Overview Error", str(e))

    def _display_main_theme(self, summary):
        """Render the overall main-theme summary on the Overview tab."""
        a = self.analysis_page

        a.overview_headline.setText(summary.get("headline", "") or "No theme data available.")
        a.overview_headline.setStyleSheet(f"color: {COLORS['text']};")

        # KPI cards.
        while a.overview_cards_layout.count() > 1:
            item = a.overview_cards_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        dom_cat = summary.get("dominant_category") or "N/A"
        dom_pct = summary.get("dominant_category_pct", 0.0)
        dom_text = f"{dom_cat} ({dom_pct:.0f}%)" if summary.get("dominant_category") else "N/A"
        cards_data = [
            ("Total Tickets", f"{summary.get('total_tickets', 0):,}", COLORS['accent']),
            ("Themes", f"{summary.get('n_clusters', 0):,}", COLORS['accent_green']),
            ("Dominant Category", dom_text, COLORS['accent_purple']),
            ("One-off Tickets", f"{summary.get('noise_pct', 0.0):.0f}%", COLORS['accent_orange']),
        ]
        for label, value, color in cards_data:
            card = make_metric_card(label, value, color)
            a.overview_cards_layout.insertWidget(a.overview_cards_layout.count() - 1, card)

        # Detail lists.
        while a.overview_results_layout.count() > 1:
            item = a.overview_results_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        pos = 0
        top_subs = summary.get("top_subcategories", [])
        if top_subs:
            hdr = QLabel("Biggest Issues by Volume")
            hdr.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr.setStyleSheet(f"color: {COLORS['accent']};")
            a.overview_results_layout.insertWidget(pos, hdr)
            pos += 1
            for label, count, pct in top_subs:
                line = f"  {str(label)[:40]:<40} {count:>6,} tickets  ({pct:.1f}%)"
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                a.overview_results_layout.insertWidget(pos, lbl)
                pos += 1

        top_keywords = summary.get("top_keywords", [])
        if top_keywords:
            hdr2 = QLabel("Key Terms")
            hdr2.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr2.setStyleSheet(f"color: {COLORS['accent_green']};")
            a.overview_results_layout.insertWidget(pos, hdr2)
            pos += 1
            terms = ", ".join(term for term, _ in top_keywords)
            kw_lbl = QLabel(terms)
            kw_lbl.setWordWrap(True)
            kw_lbl.setFont(QFont(MONO_FAMILY, 10))
            kw_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
            a.overview_results_layout.insertWidget(pos, kw_lbl)

    # ---------------------------------------------------------------
    # Analysis: Quality Audit
    # ---------------------------------------------------------------
    def _run_quality_audit(self):
        """Run the ticket quality audit in a background thread."""
        if self.df is None:
            QMessageBox.warning(self, "Audit", "Load data first.")
            return
        # This one replaces self.df wholesale when it finishes, and unlike the LLM
        # features it takes no lease — so without this guard it is the one thing that
        # can silently void a long clustering run.
        if self._busy_with_run("Audit"):
            return
        a = self.analysis_page
        if not a.btn_audit.isEnabled():
            return          # already running — ignore the double-click
        a.btn_audit.setEnabled(False)

        def _apply(audit_df, summary):
            """GUI thread: adopt the scored frame, then render."""
            self.df = audit_df
            self.analysis_results["audit_summary"] = summary
            self._display_audit_results(summary)

        # Read on the GUI thread and closed over: _gather_metadata_mapping() calls
        # QComboBox.currentText(), which a worker thread must not touch. Changing sheet
        # rebuilds those combos (dd.clear()), so a worker iterating them races a live edit.
        mapping = self._gather_metadata_mapping()

        def _audit_thread():
            try:
                from src.quality_audit import TicketQualityAuditor
                audit_settings = self.config.get("audit", {})
                auditor = TicketQualityAuditor(
                    self.df, mapping,
                    cluster_results=self.cluster_data,
                    settings=audit_settings,
                )
                audit_df, summary = auditor.generate_report(
                    text_columns=self.selected_text_cols,
                )
                self._post(lambda: _apply(audit_df, summary))
            except Exception as e:
                # Bind the text now: `e` is deleted when the except block
                # exits, and _post runs this lambda later on the GUI thread,
                # so str(e) in there raised NameError and showed no dialog.
                msg = str(e)
                logger.error(f"Audit failed: {msg}")
                self._post(lambda: QMessageBox.critical(self, "Audit Error", msg))
            finally:
                self._post(lambda: a.btn_audit.setEnabled(True))

        threading.Thread(target=_audit_thread, daemon=True).start()

    def _display_audit_results(self, summary):
        """Update the Quality Audit tab with results."""
        a = self.analysis_page

        while a.audit_cards_layout.count() > 1:
            item = a.audit_cards_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        cards_data = [
            ("Overall Score", summary.get("overall_score"), COLORS['accent']),
            ("Completeness", summary.get("completeness_avg"), COLORS['accent_green']),
            ("Categorization", summary.get("categorization_avg"), COLORS['accent_purple']),
            ("Resolution", summary.get("resolution_avg"), COLORS['accent_orange']),
        ]
        for label, value, color in cards_data:
            val_text = f"{value:.1f}%" if value is not None else "N/A"
            card = make_metric_card(label, val_text, color)
            a.audit_cards_layout.insertWidget(a.audit_cards_layout.count() - 1, card)

        while a.audit_results_layout.count() > 1:
            item = a.audit_results_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        # "N/A" on every card means nothing could be measured, which is easy to mistake
        # for a bug. Say what to do about it. (This used to report a flat 50% instead,
        # which looked like a real result.)
        if not summary.get("tickets_scored"):
            why = QLabel(
                "Nothing could be scored yet. The audit needs either some mapped "
                "metadata columns (Settings → Column Mapping) or the text columns from "
                "a clustering run — run clustering first, or map a few columns."
            )
            why.setWordWrap(True)
            why.setStyleSheet(f"color: {COLORS['accent_orange']}; border: none;")
            a.audit_results_layout.insertWidget(0, why)

        clusters = summary.get("clusters", {})
        if clusters:
            hdr = QLabel("Per-Cluster Quality")
            hdr.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr.setStyleSheet(f"color: {COLORS['accent']};")
            a.audit_results_layout.insertWidget(0, hdr)

            header_text = f"{'Cluster':<10} {'Subcategory':<25} {'Count':<8} {'Quality':>8}"
            h_lbl = QLabel(header_text)
            h_lbl.setFont(QFont(MONO_FAMILY, 10, QFont.Weight.Bold))
            a.audit_results_layout.insertWidget(1, h_lbl)

            pos = 2
            for cid, info in sorted(clusters.items()):
                q = info.get("avg_quality")
                q_str = f"{q:.1f}%" if q is not None else "N/A"
                line = f"{cid:<10} {info.get('subcategory', '')[:24]:<25} {info.get('count', 0):<8} {q_str:>8}"
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                a.audit_results_layout.insertWidget(pos, lbl)
                pos += 1

        a.btn_audit_export.setEnabled(True)
        self.clustering_page.set_status("Quality audit complete.")

    def _export_audit(self):
        """Export audit report to Excel."""
        if "audit_summary" not in self.analysis_results:
            QMessageBox.information(self, "Export", "Run the audit first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Report", "quality_audit_report.xlsx",
            "Excel (*.xlsx)",
        )
        if path:
            from src.export import export_audit_to_excel
            self._do_export(
                export_audit_to_excel, self.df,
                self.analysis_results["audit_summary"], path,
                title="Export Audit Report", ok_msg="Audit report saved.",
            )

    # ---------------------------------------------------------------
    # Analysis: KBA
    # ---------------------------------------------------------------
    def _run_kba_generation(self):
        """Generate KBA articles in a background thread."""
        if not self.cluster_data:
            QMessageBox.warning(self, "KBA", "Run clustering first \u2014 KBA articles are generated from cluster data.")
            return
        if not self.settings_page.chk_llm.isChecked() or not self.clusterer or not self.clusterer.llm:
            QMessageBox.warning(self, "KBA", "KBA generation requires the LLM to be enabled and loaded.")
            return
        if not self._llm_acquire("KBA"):
            return

        self.analysis_page.btn_gen_kba.setEnabled(False)

        def _kba_thread():
            try:
                from src.kba_generator import KBAGenerator
                gen = KBAGenerator(
                    self.clusterer.llm, self.cluster_data,
                    settings=self.config.get("kba", {}),
                )

                def _cb(msg, prog):
                    self._signals.status.emit(msg)

                articles = gen.generate_all(callback=_cb)

                def _apply():
                    self.kba_articles = articles
                    self._display_kba_results()

                self._post(_apply)
            except Exception as e:
                # Bind the text now: `e` is deleted when the except block
                # exits, and _post runs this lambda later on the GUI thread,
                # so str(e) in there raised NameError and showed no dialog.
                msg = str(e)
                logger.error(f"KBA generation failed: {msg}")
                self._post(lambda: QMessageBox.critical(self, "KBA Error", msg))
            finally:
                self._post(self._llm_release)
                self._post(lambda: self.analysis_page.btn_gen_kba.setEnabled(True))

        threading.Thread(target=_kba_thread, daemon=True).start()

    def _display_kba_results(self):
        """Populate KBA selector and show first article."""
        if not self.kba_articles:
            return
        choices = [f"Cluster {art['cluster_id']}: {art['title']}" for art in self.kba_articles]
        a = self.analysis_page
        a.kba_selector.clear()
        a.kba_selector.addItems(choices)
        a.kba_selector.setCurrentIndex(0)
        self._on_kba_selected(choices[0])
        a.btn_kba_export.setEnabled(True)
        self.clustering_page.set_status(f"Generated {len(self.kba_articles)} KBA articles.")

    def _on_kba_selected(self, selection):
        """Display the selected KBA article in the preview pane."""
        if not self.kba_articles or not selection or selection.startswith("--"):
            return
        idx = 0
        for i, art in enumerate(self.kba_articles):
            if selection.startswith(f"Cluster {art['cluster_id']}"):
                idx = i
                break
        art = self.kba_articles[idx]
        text = (
            f"TITLE: {art.get('title', '')}\n"
            f"Category: {art.get('category', '')} > {art.get('subcategory', '')}\n"
            f"Keywords: {', '.join(str(k) for k in art.get('keywords', []))}\n"
            f"\n{'=' * 50}\n\n"
            f"SYMPTOMS:\n{art.get('symptoms', 'N/A')}\n\n"
            f"CAUSE:\n{art.get('cause', 'N/A')}\n\n"
            f"RESOLUTION:\n{art.get('resolution', 'N/A')}\n\n"
            f"PREVENTION:\n{art.get('prevention', 'N/A')}\n"
        )
        self.analysis_page.kba_preview.setPlainText(text)

    def _export_kba(self):
        """Export all KBA articles."""
        if not self.kba_articles:
            QMessageBox.information(self, "Export", "Generate KBA articles first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export KBA Articles", "kba_articles.docx",
            "Word Document (*.docx);;Excel (*.xlsx)",
        )
        if path:
            from src.export import export_kba_to_docx, export_kba_to_excel
            writer = export_kba_to_excel if path.endswith(".xlsx") else export_kba_to_docx
            self._do_export(writer, self.kba_articles, path,
                            title="Export KBA Articles", ok_msg="KBA articles saved.")

    # ---------------------------------------------------------------
    # Analysis: SOP
    # ---------------------------------------------------------------
    def _run_sop_generation(self):
        """Generate SOPs in a background thread."""
        if not self.cluster_data:
            QMessageBox.warning(self, "SOP", "Run clustering first \u2014 SOPs are generated from cluster data.")
            return
        if not self.settings_page.chk_llm.isChecked() or not self.clusterer or not self.clusterer.llm:
            QMessageBox.warning(self, "SOP", "SOP generation requires the LLM to be enabled and loaded.")
            return
        if not self._llm_acquire("SOP"):
            return

        self.analysis_page.btn_gen_sop.setEnabled(False)

        def _sop_thread():
            try:
                from src.sop_generator import SOPGenerator
                gen = SOPGenerator(
                    self.clusterer.llm, self.cluster_data,
                    settings=self.config.get("sop", {}),
                )

                def _cb(msg, prog):
                    self._signals.status.emit(msg)

                sops = gen.generate_all(callback=_cb)

                def _apply():
                    self.sop_documents = sops
                    self._display_sop_results()

                self._post(_apply)
            except Exception as e:
                # Bind the text now: `e` is deleted when the except block
                # exits, and _post runs this lambda later on the GUI thread,
                # so str(e) in there raised NameError and showed no dialog.
                msg = str(e)
                logger.error(f"SOP generation failed: {msg}")
                self._post(lambda: QMessageBox.critical(self, "SOP Error", msg))
            finally:
                self._post(self._llm_release)
                self._post(lambda: self.analysis_page.btn_gen_sop.setEnabled(True))

        threading.Thread(target=_sop_thread, daemon=True).start()

    def _display_sop_results(self):
        """Populate SOP selector and show first SOP."""
        if not self.sop_documents:
            return
        choices = [sop.get("title", sop.get("category", f"SOP {i}")) for i, sop in enumerate(self.sop_documents)]
        a = self.analysis_page
        a.sop_selector.clear()
        a.sop_selector.addItems(choices)
        a.sop_selector.setCurrentIndex(0)
        self._on_sop_selected(choices[0])
        a.btn_sop_export.setEnabled(True)
        self.clustering_page.set_status(f"Generated {len(self.sop_documents)} SOPs.")

    def _on_sop_selected(self, selection):
        """Display the selected SOP in the preview pane."""
        if not self.sop_documents or not selection or selection.startswith("--"):
            return
        idx = 0
        for i, sop in enumerate(self.sop_documents):
            if sop.get("title") == selection:
                idx = i
                break
        sop = self.sop_documents[idx]
        text = (
            f"TITLE: {sop.get('title', '')}\n"
            f"Category: {sop.get('category', '')}\n"
            f"\n{'=' * 50}\n\n"
            f"PURPOSE:\n{sop.get('purpose', 'N/A')}\n\n"
            f"SCOPE:\n{sop.get('scope', 'N/A')}\n\n"
            f"PREREQUISITES:\n{sop.get('prerequisites', 'N/A')}\n\n"
            f"PROCEDURE:\n{sop.get('procedure', 'N/A')}\n\n"
            f"ESCALATION:\n{sop.get('escalation', 'N/A')}\n\n"
            f"QUALITY CHECKS:\n{sop.get('quality_checks', 'N/A')}\n"
        )
        self.analysis_page.sop_preview.setPlainText(text)

    def _export_sop(self):
        """Export all SOPs."""
        if not self.sop_documents:
            QMessageBox.information(self, "Export", "Generate SOPs first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SOPs", "standard_operating_procedures.docx",
            "Word Document (*.docx);;Excel (*.xlsx)",
        )
        if path:
            from src.export import export_sop_to_docx, export_sop_to_excel
            writer = export_sop_to_excel if path.endswith(".xlsx") else export_sop_to_docx
            self._do_export(writer, self.sop_documents, path,
                            title="Export SOPs", ok_msg="SOPs saved.")

    # ---------------------------------------------------------------
    # Analysis: Fishbone
    # ---------------------------------------------------------------
    def _populate_fishbone_scope(self):
        """Populate the fishbone scope dropdown with clusters and categories.

        Always clears, even when there is nothing to offer. Bailing out early on an empty
        list left the *previous* run's items in the dropdown after _reset_derived_state,
        so selecting one looked up a cluster id that no longer existed: keywords came back
        empty, generate_diagram substituted "(no data)" for every bone, and the user got a
        complete-looking diagram — under the old label — built from nothing.
        """
        a = self.analysis_page
        choices = []
        for cid in sorted((self.cluster_data or {}).keys()):
            if cid == -1:
                continue
            subcat = self.cluster_data[cid].get("subcategory", f"Topic {cid}")
            choices.append(f"Cluster {cid}: {subcat}")
        cats = sorted(set(
            cd.get("category", "") for cd in (self.cluster_data or {}).values()
            if cd.get("category") and cd.get("category") != "Non-Repetitive"
        ))
        for cat in cats:
            choices.append(f"Category: {cat}")

        a.fishbone_scope.clear()
        if choices:
            a.fishbone_scope.addItems(choices)
            a.fishbone_scope.setCurrentIndex(0)
        else:
            # A "--" prefix is what _run_fishbone's guard checks for.
            a.fishbone_scope.addItem("-- Run clustering first --")

    def _run_fishbone(self):
        """Generate and open a fishbone diagram in the browser."""
        if not self.cluster_data:
            QMessageBox.warning(self, "Fishbone", "Run clustering first \u2014 fishbone diagrams use cluster keywords.")
            return
        if self._busy_with_run("Fishbone", read_only=True):
            return
        selection = self.analysis_page.fishbone_scope.currentText()
        if not selection or selection.startswith("--"):
            QMessageBox.warning(self, "Fishbone", "Select a cluster or category.")
            return

        # No LLM lease: classification is pure keyword matching (see FishboneGenerator).
        # Taking the shared lease here used to block every other AI feature for an
        # operation that never touched the model.
        try:
            from src.fishbone import FishboneGenerator
            gen = FishboneGenerator(self.cluster_data)

            if selection.startswith("Category: "):
                scope = selection.replace("Category: ", "")
            else:
                scope = int(selection.split(":")[0].replace("Cluster ", ""))

            fig = gen.generate_diagram(scope)
            self._last_fishbone_fig = fig
            gen.open_in_browser(fig)
            self.analysis_page.fishbone_status.setText(
                f"Fishbone diagram opened in browser for: {selection}"
            )
            self.analysis_page.fishbone_status.setStyleSheet(
                f"color: {COLORS['accent_green']};"
            )
        except Exception as e:
            # There was a finally: but no except:, so a failure (a stale dropdown label
            # that int() can't parse, a browser/plotly error) printed a traceback to
            # stderr and nothing at all appeared on screen.
            logger.error(f"Fishbone generation failed: {e}")
            self.analysis_page.fishbone_status.setText(
                f"Could not build the diagram ({type(e).__name__})."
            )
            self.analysis_page.fishbone_status.setStyleSheet(
                f"color: {COLORS['accent_red']};"
            )
            QMessageBox.critical(
                self, "Fishbone",
                f"Could not build the fishbone diagram:\n\n{e}",
            )

    def _save_fishbone_html(self):
        """Save the last fishbone diagram as HTML."""
        if not hasattr(self, "_last_fishbone_fig") or self._last_fishbone_fig is None:
            QMessageBox.information(self, "Save", "Generate a fishbone diagram first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fishbone Diagram", "fishbone_diagram.html",
            "HTML (*.html);;PNG Image (*.png)",
        )
        if path:
            from src.fishbone import FishboneGenerator
            gen = FishboneGenerator(self.cluster_data)
            writer = gen.export_image if path.endswith(".png") else gen.export_html
            self._do_export(writer, self._last_fishbone_fig, path,
                            title="Save Fishbone Diagram", ok_msg="Fishbone diagram saved.")

    # ---------------------------------------------------------------
    # Analysis: Impact
    # ---------------------------------------------------------------
    def _run_impact_analysis(self):
        """Run impact analysis in a background thread."""
        if self.df is None:
            QMessageBox.warning(self, "Impact", "Load data first.")
            return
        # Read-only, but threaded: it reads self.df and self.cluster_data inside the
        # worker, so a run finishing mid-analysis can swap one out from under the other
        # and produce a mismatched pair.
        if self._busy_with_run("Impact Analysis", read_only=True):
            return
        a = self.analysis_page
        if not a.btn_impact.isEnabled():
            return          # already running — ignore the double-click
        a.btn_impact.setEnabled(False)

        # GUI thread: see the note in _run_quality_audit.
        mapping = self._gather_metadata_mapping()

        def _impact_thread():
            try:
                from src.impact_analysis import ImpactAnalyzer
                analyzer = ImpactAnalyzer(
                    self.df, mapping, self.cluster_data,
                    settings=self.config.get("analysis", {}),
                )
                problem = analyzer.analyze_problem_clusters()
                process = analyzer.analyze_business_process()
                kpi = analyzer.analyze_kpi_impact()
                figs = analyzer.generate_visualizations(problem, process, kpi)

                def _apply():
                    self.analysis_results["problem_clusters"] = problem
                    self.analysis_results["business_process"] = process
                    self.analysis_results["kpi"] = kpi
                    self.analysis_results["impact_figs"] = figs
                    self._display_impact_results(problem, process, kpi, figs)

                self._post(_apply)
            except Exception as e:
                # Bind the text now: `e` is deleted when the except block
                # exits, and _post runs this lambda later on the GUI thread,
                # so str(e) in there raised NameError and showed no dialog.
                msg = str(e)
                logger.error(f"Impact analysis failed: {msg}")
                self._post(lambda: QMessageBox.critical(self, "Impact Error", msg))
            finally:
                self._post(lambda: a.btn_impact.setEnabled(True))

        threading.Thread(target=_impact_thread, daemon=True).start()

    def _display_impact_results(self, problem, process, kpi, figs):
        """Display impact analysis results."""
        a = self.analysis_page

        while a.impact_results_layout.count() > 1:
            item = a.impact_results_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        pos = 0

        hdr = QLabel("Problem Cluster Analysis")
        hdr.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {COLORS['accent']};")
        a.impact_results_layout.insertWidget(pos, hdr)
        pos += 1

        if "volume_stats" in problem:
            vol = problem["volume_stats"]
            top = vol.head(10)
            label_col = "subcategory" if "subcategory" in top.columns else "Cluster_ID"
            for _, row in top.iterrows():
                line = f"  Cluster {row['Cluster_ID']}: {row.get(label_col, '')} \u2014 {row['count']} tickets ({row['pct']}%)"
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                a.impact_results_layout.insertWidget(pos, lbl)
                pos += 1

        if "impact_ranking" in kpi:
            hdr2 = QLabel("KPI Impact Ranking")
            hdr2.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr2.setStyleSheet(f"color: {COLORS['accent']};")
            a.impact_results_layout.insertWidget(pos, hdr2)
            pos += 1
            rank = kpi["impact_ranking"].head(10)
            # isna, not `is not None`: Avg_Resolution is None for some clusters and a
            # float for others, so the DataFrame column is float64 and the Nones arrive
            # as NaN. NaN is not None, so this used to render "nanh" in the table.
            _pd = get_pandas()
            for _, row in rank.iterrows():
                _res = row.get("Avg_Resolution")
                res_str = "N/A" if _pd.isna(_res) else f"{_res:.1f}h"
                line = (
                    f"  {str(row.get('Subcategory', ''))[:30]:<30} "
                    f"Tickets: {row['Ticket_Count']:<6} "
                    f"Avg Res: {res_str:<8} "
                    f"SLA Breach: {row.get('SLA_Breach_Rate', 0):.1f}% "
                    f"Score: {row['Impact_Score']:.0f}"
                )
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                a.impact_results_layout.insertWidget(pos, lbl)
                pos += 1

        if "bottlenecks" in process:
            bn = process["bottlenecks"]
            if not bn.empty and "is_bottleneck" in bn.columns:
                bottleneck_rows = bn[bn["is_bottleneck"]]
                if not bottleneck_rows.empty:
                    hdr3 = QLabel("Bottlenecks")
                    hdr3.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
                    hdr3.setStyleSheet(f"color: {COLORS['accent_red']};")
                    a.impact_results_layout.insertWidget(pos, hdr3)
                    pos += 1
                    label_col = "subcategory" if "subcategory" in bottleneck_rows.columns else "Cluster_ID"
                    for _, row in bottleneck_rows.head(5).iterrows():
                        line = f"  {row.get(label_col, '')} \u2014 {row['count']} tickets, avg {row['avg_resolution']:.1f}h resolution"
                        lbl = QLabel(line)
                        lbl.setFont(QFont(MONO_FAMILY, 10))
                        lbl.setStyleSheet(f"color: {COLORS['accent_red']};")
                        a.impact_results_layout.insertWidget(pos, lbl)
                        pos += 1

        if figs:
            hdr4 = QLabel("Interactive Charts")
            hdr4.setFont(QFont(MONO_FAMILY, 12, QFont.Weight.Bold))
            hdr4.setStyleSheet(f"color: {COLORS['accent']};")
            a.impact_results_layout.insertWidget(pos, hdr4)
            pos += 1

            chart_row = QWidget()
            chart_layout = QHBoxLayout(chart_row)
            chart_layout.setContentsMargins(0, 0, 0, 0)
            for name, fig in figs.items():
                label = name.replace("_", " ").title()
                btn = styled_button(f"\U0001F4CA {label}", COLORS['accent'], small=True)
                btn.clicked.connect(lambda checked, f=fig: self._open_plotly_chart(f))
                chart_layout.addWidget(btn)
            chart_layout.addStretch()
            a.impact_results_layout.insertWidget(pos, chart_row)

        a.btn_impact_export.setEnabled(True)
        self.clustering_page.set_status("Impact analysis complete.")

    def _open_plotly_chart(self, fig):
        """Open a Plotly figure in the default browser."""
        from src.impact_analysis import ImpactAnalyzer
        ImpactAnalyzer.open_figure_in_browser(fig)

    def _export_impact(self):
        """Export impact analysis results to Excel."""
        if not self.analysis_results.get("kpi"):
            QMessageBox.information(self, "Export", "Run impact analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Impact Report", "impact_analysis_report.xlsx",
            "Excel (*.xlsx)",
        )
        if path:
            from src.export import export_impact_to_excel
            self._do_export(export_impact_to_excel, self.analysis_results, path,
                            title="Export Impact Report", ok_msg="Impact report saved.")

    # ---------------------------------------------------------------
    # Automation Opportunities (Root Cause + Resolution + Disposition)
    # ---------------------------------------------------------------
    def _run_disposition_analysis(self):
        """Assign automation dispositions to the already-clustered symptom topics."""
        logger.info("[disposition] Find Opportunities clicked.")
        if self.df is None:
            logger.info("[disposition] Guard: no data loaded.")
            QMessageBox.warning(self, "Automation Opportunities", "Load data first.")
            return
        # Writes five columns onto self.df; the run would then refuse its own results.
        if self._busy_with_run("Automation Opportunities"):
            return
        if not self.cluster_data or "Cluster_ID" not in self.df.columns:
            logger.info(
                f"[disposition] Guard: clustering not run "
                f"(cluster_data={bool(self.cluster_data)}, "
                f"has_col={'Cluster_ID' in self.df.columns})."
            )
            QMessageBox.warning(
                self, "Automation Opportunities",
                "Run clustering first — automation opportunities are analysed per subcategory.",
            )
            return
        a = self.analysis_page
        res_cols = [cb.text() for cb in a.disp_checkboxes if cb.isChecked()]
        if not res_cols:
            logger.info(
                f"[disposition] Guard: no resolution columns checked "
                f"({len(a.disp_checkboxes)} checkboxes available)."
            )
            QMessageBox.warning(
                self, "Automation Opportunities",
                "Select at least one resolution-text column (e.g. work_notes, close_notes).",
            )
            return
        if not self._llm_acquire("Automation Opportunities"):
            return

        a.btn_disp.setEnabled(False)
        a.disp_progress.setValue(0)
        a.disp_progress.setVisible(True)
        a.disp_status_label.setText("Starting analysis…")
        a.disp_status_label.setVisible(True)
        self.clustering_page.set_status("Finding automation opportunities…")

        # GUI thread: see the note in _run_quality_audit. res_cols above is read here for
        # the same reason.
        mapping = self._gather_metadata_mapping()

        def _disp_thread():
            try:
                from src.disposition import DispositionAnalyzer

                def status(msg, progress=None):
                    logger.info(f"[disposition] {msg}")
                    # Cross-thread Qt signal — QTimer from a worker thread never fires
                    # (no event loop on this thread), so use the signal mechanism.
                    self._signals.disposition_progress.emit(msg, progress if progress is not None else -1.0)

                disp_settings = self.config.get("disposition", {})
                n_clusters = len([c for c in self.cluster_data if c != -1])
                llm_loaded = self.clusterer is not None and self.clusterer.llm is not None
                logger.info(
                    f"[disposition] Starting: {n_clusters} clusters, res_cols={res_cols}, "
                    f"llm_loaded={llm_loaded}"
                )
                status(f"Analysing {n_clusters} subcategories…", 0.0)

                # Use the already-clustered symptom topics — no second clustering pass.
                # cluster_data keyed by topic_id; cluster_col is the Cluster_ID column
                # written by the main clustering run.
                analyzer = DispositionAnalyzer(
                    llm=(self.clusterer.llm if self.clusterer else None),
                    df=self.df,
                    cluster_data=self.cluster_data,
                    cluster_col="Cluster_ID",
                    mapping=mapping,
                    settings=disp_settings,
                    res_cols=res_cols,
                )
                results = analyzer.generate_all(callback=status)
                summary = analyzer.summarize(results)
                logger.info(f"[disposition] Finished: {len(results)} results")

                # Per-ticket columns: Cluster_ID already exists; add disposition/cause/
                # resolution/rationale/recommendation. The analyzer produces all five LLM
                # sections; "recommendation" used to reach only the disposition workbook
                # and the results card, so the main sheet was missing the one column that
                # says what to actually do about the cluster.
                disp_map  = {r["cluster_id"]: r["disposition"]  for r in results}
                cause_map = {r["cluster_id"]: r["root_cause"]   for r in results}
                res_map   = {r["cluster_id"]: r["resolution"]   for r in results}
                rat_map   = {r["cluster_id"]: r.get("rationale", "") for r in results}
                rec_map   = {r["cluster_id"]: r.get("recommendation", "") for r in results}
                _disp = self.df["Cluster_ID"].map(disp_map)
                # Safety-net tickets (Cluster_ID==-1 but re-tagged to a real category)
                # didn't form a repeating cluster, so they can't be automated as a group.
                # Give them "Retain" rather than misleading "Non-Repetitive".
                if "Repetitive Category" in self.df.columns:
                    _has_real_cat = self.df["Repetitive Category"].ne("Non-Repetitive")
                    _disp = _disp.mask(_disp.isna() & _has_real_cat, "Retain")
                self.df["Automation Disposition"] = _disp.fillna("Non-Repetitive")
                self.df["Root Cause"]             = self.df["Cluster_ID"].map(cause_map).fillna("")
                self.df["Resolution Provided"]    = self.df["Cluster_ID"].map(res_map).fillna("")
                self.df["Disposition Rationale"]  = self.df["Cluster_ID"].map(rat_map).fillna("")
                self.df["Recommendation"]         = self.df["Cluster_ID"].map(rec_map).fillna("")

                self.analysis_results["disposition"] = results
                self.analysis_results["disposition_summary"] = summary
                self._signals.disposition_done.emit(results, summary)
            except Exception as e:
                import traceback
                logger.error(f"Disposition analysis failed: {e}\n{traceback.format_exc()}")
                self._signals.disposition_error.emit(f"{type(e).__name__}: {e}")
            finally:
                # The done/error slots already release, but this guarantees it even if
                # the emit itself fails — a leaked lease locks out every AI feature.
                # _llm_release just clears a flag, so a double release is harmless.
                self._post(self._llm_release)

        threading.Thread(target=_disp_thread, daemon=True).start()

    def _on_disposition_progress(self, msg, progress):
        """Main-thread slot: update the Automation-Opportunities progress UI."""
        a = self.analysis_page
        a.disp_status_label.setText(msg)
        if progress is not None and progress >= 0:
            a.disp_progress.setValue(int(progress * 100))

    def _on_disposition_done(self, results, summary):
        """Main-thread slot: render results and reset the progress UI."""
        self._llm_release()
        a = self.analysis_page
        a.disp_progress.setValue(100)
        try:
            self._display_disposition_results(results, summary)
        finally:
            a.btn_disp.setEnabled(True)
            a.disp_progress.setVisible(False)
            a.disp_progress.setValue(0)
            a.disp_status_label.setVisible(False)
            a.disp_status_label.setText("")

    def _on_disposition_error(self, message):
        """Main-thread slot: report failure and reset the progress UI."""
        self._llm_release()
        a = self.analysis_page
        a.btn_disp.setEnabled(True)
        a.disp_progress.setVisible(False)
        a.disp_progress.setValue(0)
        a.disp_status_label.setVisible(False)
        a.disp_status_label.setText("")
        QMessageBox.critical(self, "Automation Opportunities Error", message)

    _DISPOSITION_COLORS = {
        "Automate": "accent_green",
        "Eradicate": "accent_red",
        "Reimagine": "accent_orange",
        "Retain": "text_muted",
    }

    def _display_disposition_results(self, results, summary):
        """Render the ranked automation opportunities."""
        logger.info(f"[disposition] Rendering {len(results)} results to GUI.")
        a = self.analysis_page
        # Results are already stored in self.analysis_results, so enable export up
        # front — even if rendering a row below fails, the report stays exportable.
        a.btn_disp_export.setEnabled(True)

        def _num(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        try:
            while a.disp_results_layout.count() > 1:
                item = a.disp_results_layout.takeAt(0)
                if item and item.widget():
                    item.widget().setParent(None)

            pos = 0
            hdr = QLabel("Summary by Disposition")
            hdr.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr.setStyleSheet(f"color: {COLORS['accent_green']};")
            a.disp_results_layout.insertWidget(pos, hdr); pos += 1
            for disp, info in (summary or {}).items():
                line = (f"  {disp:<10} {info.get('clusters', 0):>3} clusters   "
                        f"{info.get('tickets', 0):>5} tickets   "
                        f"{_num(info.get('roi_hours')):>10,.0f} effort-h")
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                lbl.setStyleSheet(f"color: {COLORS[self._DISPOSITION_COLORS.get(disp, 'text')]};")
                a.disp_results_layout.insertWidget(pos, lbl); pos += 1

            hdr2 = QLabel("Top Opportunities (ranked by total effort hours)")
            hdr2.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
            hdr2.setStyleSheet(f"color: {COLORS['accent']};")
            a.disp_results_layout.insertWidget(pos, hdr2); pos += 1
            head = (f"  {'Disposition':<11} {'Resolution Pattern':<26} {'Tkts':>5} "
                    f"{'Effort-h':>9} {'Auto':>5} {'Reopen':>7}")
            h_lbl = QLabel(head)
            h_lbl.setFont(QFont(MONO_FAMILY, 10, QFont.Weight.Bold))
            a.disp_results_layout.insertWidget(pos, h_lbl); pos += 1

            for r in results[:25]:
                disp = r.get("disposition", "") or ""
                line = (f"  {disp:<11} {str(r.get('resolution_pattern', ''))[:25]:<26} "
                        f"{r.get('volume', 0):>5} {str(r.get('roi', '')):>9} "
                        f"{str(r.get('automatability', '')):>5} {str(r.get('reopen_pct', '')):>7}")
                lbl = QLabel(line)
                lbl.setFont(QFont(MONO_FAMILY, 10))
                lbl.setStyleSheet(f"color: {COLORS[self._DISPOSITION_COLORS.get(disp, 'text')]};")
                lbl.setToolTip(
                    f"Root cause: {r.get('root_cause', '')}\n"
                    f"Resolution: {r.get('resolution', '')}\n"
                    f"Rationale: {r.get('rationale', '')}\n"
                    f"Recommendation: {r.get('recommendation', '')}"
                )
                a.disp_results_layout.insertWidget(pos, lbl); pos += 1

            self.clustering_page.set_status("Automation-opportunity analysis complete.")
            logger.info("[disposition] Render complete.")
        except Exception as e:
            import traceback
            logger.error(f"[disposition] Render failed: {e}\n{traceback.format_exc()}")
            QMessageBox.warning(
                self, "Automation Opportunities",
                f"Analysis finished but the on-screen summary could not be drawn "
                f"({type(e).__name__}). Use Export Report to save the full results.",
            )

    def _export_disposition(self):
        """Export automation opportunities to Excel."""
        results = self.analysis_results.get("disposition")
        if not results:
            QMessageBox.information(self, "Export", "Run the analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Automation Opportunities",
            "automation_opportunities.xlsx", "Excel (*.xlsx)",
        )
        if path:
            from src.export import export_disposition_to_excel
            self._do_export(
                export_disposition_to_excel, results, path,
                self.analysis_results.get("disposition_summary"),
                title="Export Automation Opportunities",
                ok_msg="Automation opportunities saved.",
            )

    # ---------------------------------------------------------------
    # Export helper (report failures instead of dying inside a Qt slot)
    # ---------------------------------------------------------------
    def _do_export(self, fn, *args, title="Export", ok_msg="Saved."):
        """Run an exporter, reporting success or failure in a dialog.

        Without this an exception (most often the target .xlsx being open in Excel →
        PermissionError) propagates out of a Qt slot: the traceback goes to stderr and
        the user sees nothing at all — no file and no error.
        """
        try:
            fn(*args)
        except Exception as e:
            logger.error(f"{title} failed: {e}")
            QMessageBox.critical(
                self, title,
                f"Could not save the file:\n{e}\n\n"
                "If it is open in Excel, close it and try again.",
            )
            return False
        QMessageBox.information(self, "Exported", ok_msg)
        return True

    # ---------------------------------------------------------------
    # Import lock (keeps self.df stable while a run reads it)
    # ---------------------------------------------------------------
    def _set_import_enabled(self, enabled):
        """Enable/disable the controls that can replace self.df.

        The clustering worker reads the text out of self.df at the start and writes
        Cluster_ID back minutes later; if the sheet is swapped in between and the new
        sheet happens to have the same row count, the write-back silently stamps one
        dataset's labels onto another's rows. Locking these for the run prevents it.
        """
        p = self.import_page
        p.btn_load.setEnabled(enabled)
        # Only re-enable the sheet picker if a workbook is actually loaded.
        p.sheet_dropdown.setEnabled(enabled and bool(self.sheet_names))
        # The File menu reaches load_file too, and a disabled button doesn't disable it.
        if getattr(self, "act_open", None) is not None:
            self.act_open.setEnabled(enabled)
        # Opening a session replaces df and cluster_data outright; saving one mid-run
        # would capture a half-built state. _busy_with_run also refuses both, but greying
        # them is the honest signal.
        for attr in ("act_open_session", "act_save_session"):
            act = getattr(self, attr, None)
            if act is not None:
                act.setEnabled(enabled)

    def _busy_with_run(self, feature, read_only=False):
        """True (with a notice) when a clustering run is in progress.

        Anything that replaces or rewrites self.df, or that reads cluster state being
        rebuilt, must call this first. Refusing here costs the user a click; letting it
        through costs them the entire run.

        ``read_only=True`` is for features that only *read* self.df / self.cluster_data
        (Overview, Impact, Fishbone). They cannot void the run, so the honest reason is
        different: they would analyse the previous run's clusters and present the result
        as current, moments before _on_clustering_finished replaces them.
        """
        if not self._run_in_progress:
            return False
        if read_only:
            why = ("It would analyse the *previous* run's clusters, which are about to "
                   "be replaced — so the report would be out of date the moment the run "
                   "finishes.")
        else:
            why = ("This would change the data the run is working on, so the run's "
                   "results would have to be discarded.")
        QMessageBox.information(
            self, feature,
            f"A clustering run is in progress.\n\n{why} Wait for it to finish "
            "(or press Stop) and try again.",
        )
        return True

    # ---------------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------------
    def closeEvent(self, event):
        """Release what the OS won't tidy up for us, then close.

        There was no closeEvent at all before this. Closing mid-run left the worker (a
        daemon thread) to be abandoned at interpreter teardown, often inside native
        UMAP/HDBSCAN, with the GGUF still mapped and the source workbook still locked.
        Signalling the stop and giving the worker a moment to reach its next checkpoint
        turns that into an ordinary cancellation. The wait is deliberately short: this is
        best-effort tidying, and the app must always actually close.
        """
        try:
            self._stop_event.set()
            if self._run_in_progress:
                logger.info("Closing while a run is in progress — cancelling it.")
            # Native calls can't be interrupted, so don't wait on them; just give a
            # cooperative checkpoint a brief chance to fire.
            for t in threading.enumerate():
                if t is not threading.current_thread() and t.name.startswith("Thread-"):
                    t.join(timeout=0.25)
            if self.clusterer is not None:
                try:
                    self.clusterer.close_llm()
                except Exception as e:
                    logger.debug(f"Could not release the LLM on close: {e}")
            self._close_excel_file()
        except Exception as e:
            # Never let tidying block the close.
            logger.debug(f"Shutdown cleanup raised: {e}")
        super().closeEvent(event)

    # ---------------------------------------------------------------
    # Worker -> GUI thread marshalling
    # ---------------------------------------------------------------
    def _post(self, fn):
        """Run ``fn`` on the GUI thread. Safe to call from a worker thread.

        Always use this (never QTimer.singleShot) to hop back from a worker: a
        QTimer created on a thread with no event loop never fires, so those
        callbacks are silently dropped.
        """
        self._signals.call_on_main.emit(fn)

    # ---------------------------------------------------------------
    # LLM mutual-exclusion (one AI analysis at a time)
    # ---------------------------------------------------------------
    def _llm_acquire(self, feature):
        """Claim the shared LLM for ``feature``; False (with a notice) if busy.

        llama-cpp isn't safe to call concurrently, so clustering / KBA / SOP /
        Fishbone / Disposition / Category Audit must run one at a time. Call this
        on the main thread after validation, immediately before starting work,
        and pair it with _llm_release() on every completion path.
        """
        if self._llm_busy:
            QMessageBox.information(
                self, feature,
                "Another AI task is still running. Please wait for it to finish "
                "before starting this one.",
            )
            return False
        self._llm_busy = True
        return True

    def _llm_release(self):
        self._llm_busy = False

    # ---------------------------------------------------------------
    # Category audit (LLM re-checks category assignments)
    # ---------------------------------------------------------------
    def _run_category_audit(self):
        """Ask the local LLM to review category assignments and propose moves."""
        logger.info("[category-audit] Audit Categories clicked.")
        if self.df is None:
            QMessageBox.warning(self, "Category Audit", "Load data first.")
            return
        # Rewrites Repetitive Category rows, and reads cluster_data the run is rebuilding.
        if self._busy_with_run("Category Audit"):
            return
        if not self.cluster_data or "Cluster_ID" not in self.df.columns:
            QMessageBox.warning(
                self, "Category Audit",
                "Run clustering first — the audit reviews the categories it produced.",
            )
            return
        if not (self.clusterer and self.clusterer.llm):
            QMessageBox.warning(
                self, "Category Audit",
                "The category audit needs the AI model. Turn on AI naming in Settings "
                "and run clustering, then try again.",
            )
            return
        if not self._llm_acquire("Category Audit"):
            return

        a = self.analysis_page
        a.btn_cat_audit.setEnabled(False)
        a.cat_audit_summary.setVisible(False)
        a.cat_audit_progress.setValue(0)
        a.cat_audit_progress.setVisible(True)
        a.cat_audit_status_label.setText("Starting audit…")
        a.cat_audit_status_label.setVisible(True)

        # Inject per-cluster ticket counts here (main thread) so proposals can show
        # sizes without the worker mutating cluster_data concurrently. int() normalizes
        # the count value from numpy int64 to a plain Python int.
        sizes = self.df["Cluster_ID"].value_counts().to_dict()
        for cid, cdata in self.cluster_data.items():
            cdata["size"] = int(sizes.get(cid, 0))

        def _audit_thread():
            try:
                from src.category_audit import CategoryAuditor

                def status(msg, progress=None):
                    self._signals.category_audit_progress.emit(
                        msg, progress if progress is not None else -1.0
                    )

                auditor = CategoryAuditor(
                    llm=self.clusterer.llm,
                    cluster_data=self.cluster_data,
                    settings=self.config.get("category_audit", {}),
                )
                # should_stop was accepted by audit() but never passed, so the audit
                # could not be interrupted despite having the plumbing for it.
                proposals = auditor.audit(callback=status,
                                          should_stop=self._stop_event.is_set)
                self._signals.category_audit_done.emit(proposals)
            except Exception as e:
                import traceback
                logger.error(f"Category audit failed: {e}\n{traceback.format_exc()}")
                self._signals.category_audit_error.emit(f"{type(e).__name__}: {e}")
            finally:
                # Matches _disp_thread: the done/error slots already release, but this
                # guarantees it even if the emit itself fails — a leaked lease locks out
                # every AI feature. _llm_release just clears a flag, so a double release
                # is harmless.
                self._post(self._llm_release)

        # _stop_event is shared with clustering and stays set after a cancelled run, so
        # clear it or the audit would cancel itself immediately. Safe: _busy_with_run
        # above means no clustering run can be in progress here.
        self._stop_event.clear()
        threading.Thread(target=_audit_thread, daemon=True).start()

    def _on_cat_audit_progress(self, msg, progress):
        a = self.analysis_page
        a.cat_audit_status_label.setText(msg)
        if progress is not None and progress >= 0:
            a.cat_audit_progress.setValue(int(progress * 100))

    def _on_cat_audit_done(self, proposals):
        """Main-thread slot: reset progress UI and open the review dialog."""
        self._llm_release()   # LLM work is finished; the review below is offline
        a = self.analysis_page
        a.btn_cat_audit.setEnabled(True)
        a.cat_audit_progress.setVisible(False)
        a.cat_audit_progress.setValue(0)
        a.cat_audit_status_label.setVisible(False)
        a.cat_audit_status_label.setText("")

        if not proposals:
            a.cat_audit_summary.setText(
                f"<span style='color:{COLORS['accent_green']};'>✓ No changes suggested — "
                "the current categorization looks consistent.</span>"
            )
            a.cat_audit_summary.setVisible(True)
            return

        dlg = CategoryAuditReviewDialog(proposals, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            a.cat_audit_summary.setText(
                f"<span style='color:{COLORS['text_muted']};'>Audit dismissed — "
                f"no changes applied ({len(proposals)} suggestion(s) discarded).</span>"
            )
            a.cat_audit_summary.setVisible(True)
            return

        approved = dlg.approved()
        if not approved:
            a.cat_audit_summary.setText(
                f"<span style='color:{COLORS['text_muted']};'>Nothing selected — "
                "no changes applied.</span>"
            )
            a.cat_audit_summary.setVisible(True)
            return

        self._apply_category_audit(approved)

    def _on_cat_audit_error(self, message):
        self._llm_release()
        a = self.analysis_page
        a.btn_cat_audit.setEnabled(True)
        a.cat_audit_progress.setVisible(False)
        a.cat_audit_progress.setValue(0)
        a.cat_audit_status_label.setVisible(False)
        a.cat_audit_status_label.setText("")
        QMessageBox.critical(self, "Category Audit Error", message)

    def _apply_category_audit(self, approved):
        """Apply approved reassignments to cluster_data, macro_map and the df.

        Surgical by design: only rows belonging to an approved cluster are
        rewritten. Unapproved clusters and noise rows (Cluster_ID == -1, which may
        carry safety-net category tags) are never touched, so the audit can't
        blank or drift categories the user didn't sign off on.
        """
        applied = 0
        for p in approved:
            cid = p["cluster_id"]
            new_cat = p["proposed_category"]
            if not new_cat or cid == -1:
                continue  # defensive: never write an empty category or touch noise
            if cid in self.cluster_data:
                self.cluster_data[cid]["category"] = new_cat
            # Keep the clusterer's map in sync so any later re-derivation agrees.
            if self.clusterer is not None and getattr(self.clusterer, "_last_macro_map", None) is not None:
                self.clusterer._last_macro_map[cid] = new_cat
            # Rewrite only this cluster's rows.
            self.df.loc[self.df["Cluster_ID"] == cid, "Repetitive Category"] = new_cat
            applied += 1

        # Refresh category-dependent UI (Fishbone / Overview scope dropdown).
        if hasattr(self, "_populate_fishbone_scope"):
            try:
                self._populate_fishbone_scope()
            except Exception as e:
                logger.warning(f"[category-audit] Could not refresh fishbone scope: {e}")

        moves = "".join(
            f"<li>{html.escape(p['subcategory'])}: "
            f"<span style='color:{COLORS['text_muted']};'>{html.escape(p['current_category'])}</span>"
            f" → <span style='color:{COLORS['accent_green']};'>{html.escape(p['proposed_category'])}</span></li>"
            for p in approved if p.get("proposed_category") and p.get("cluster_id") != -1
        )
        a = self.analysis_page
        a.cat_audit_summary.setText(
            f"<b style='color:{COLORS['accent_green']};'>✓ Applied {applied} reassignment(s)</b>"
            f" to the “Repetitive Category” column.<ul>{moves}</ul>"
        )
        a.cat_audit_summary.setVisible(True)
        self.clustering_page.set_status(
            f"Category audit applied — {applied} group(s) recategorized."
        )
        logger.info(f"[category-audit] Applied {applied} reassignment(s).")

        # Keep the Category Pivot in sync with the reassignments just applied.
        self._refresh_category_pivot()

    # ---------------------------------------------------------------
    # Category pivot (Category -> Subcategory ticket breakdown)
    # ---------------------------------------------------------------
    def _refresh_category_pivot(self):
        """Recompute the Category → Subcategory tree from self.df. Safe to call
        anytime; shows a placeholder until clustering has produced the columns."""
        a = self.analysis_page
        tree = a.pivot_tree
        if self.df is None or "Repetitive Category" not in self.df.columns:
            tree.clear()
            a.pivot_headline.setText("Run clustering to see the category breakdown.")
            a.btn_pivot_export.setEnabled(False)
            return

        from src.pivot import compute_category_pivot
        pivot = compute_category_pivot(self.df)
        if pivot["total_tickets"] == 0:
            tree.clear()
            a.pivot_headline.setText("No categorized tickets to show yet.")
            a.btn_pivot_export.setEnabled(False)
            return

        tree.setSortingEnabled(False)   # bulk insert without re-sorting each add
        tree.clear()
        for cat in pivot["categories"]:
            parent = _PivotTreeItem([cat["category"], f"{cat['count']:,}", f"{cat['pct']:.1f}%"])
            parent.setData(1, Qt.ItemDataRole.UserRole, cat["count"])
            parent.setData(2, Qt.ItemDataRole.UserRole, cat["pct"])
            parent.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
            parent.setTextAlignment(2, Qt.AlignmentFlag.AlignRight)
            f = parent.font(0); f.setBold(True); parent.setFont(0, f)
            for sub in cat["subcategories"]:
                child = _PivotTreeItem([sub["subcategory"], f"{sub['count']:,}", f"{sub['pct']:.1f}%"])
                child.setData(1, Qt.ItemDataRole.UserRole, sub["count"])
                child.setData(2, Qt.ItemDataRole.UserRole, sub["pct"])
                child.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
                child.setTextAlignment(2, Qt.AlignmentFlag.AlignRight)
                parent.addChild(child)
            tree.addTopLevelItem(parent)
        tree.expandAll()
        # Default view: most tickets first (categories and their subcategories).
        tree.setSortingEnabled(True)
        tree.sortItems(1, Qt.SortOrder.DescendingOrder)

        a.pivot_headline.setText(
            f"{pivot['total_tickets']:,} tickets · {pivot['n_categories']} categories "
            f"· {pivot['n_subcategories']} subcategories"
        )
        a.btn_pivot_export.setEnabled(True)

    def _export_category_pivot(self):
        """Export the Category → Subcategory pivot to Excel."""
        if self.df is None or "Repetitive Category" not in self.df.columns:
            QMessageBox.information(self, "Export", "Run clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Category Pivot", "category_pivot.xlsx", "Excel (*.xlsx)")
        if path:
            from src.export import export_category_pivot_to_excel
            self._do_export(export_category_pivot_to_excel, self.df, path,
                            title="Export Category Pivot", ok_msg="Category pivot saved.")

    # ---------------------------------------------------------------
    # Word Cloud
    # ---------------------------------------------------------------
    def _get_topic_word_weights(self):
        """Extract {topic_id: {word: weight}} from BERTopic model."""
        topic_model = self.clusterer.topic_model
        topic_words = {}
        for topic_id in topic_model.get_topics():
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id)
            if words:
                topic_words[topic_id] = {w: abs(score) for w, score in words if isinstance(w, str)}
        return topic_words

    def _open_wordcloud_studio(self):
        """Open the full Word Cloud Studio window (lazy import to avoid a circular import)."""
        if self.df is None and not self.cluster_data:
            QMessageBox.warning(self, "Word Cloud Studio",
                                "Load data (or run clustering) first.")
            return
        try:
            from src.wordcloud_studio import WordCloudStudio
        except Exception as e:
            QMessageBox.critical(self, "Word Cloud Studio", f"Could not open the studio:\n{e}")
            return
        self._wc_studio = WordCloudStudio(self)   # keep a reference (non-modal window)
        self._wc_studio.show()
        self._wc_studio.raise_()
        self._wc_studio.activateWindow()



def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS['bg']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS['text']))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS['bg_input']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS['bg_secondary']))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS['text']))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS['bg_card']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS['text']))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS['accent']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    app.setPalette(palette)

    window = ClusterApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

