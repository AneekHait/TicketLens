"""Comprehensive end-to-end tests for TicketLens.

Covers the entire application lifecycle:
1. Window sizing, responsive geometry across screen resolutions (including 1440x900 MacBook fullscreen).
2. Data creation and Excel loading via ImportPage.
3. Column selection and document preprocessing preview.
4. Settings page configuration.
5. Full clustering pipeline execution with cluster_data and df labeling.
6. Session persistence (save and reload).
7. Analysis dashboard tabs:
   - Overview (main theme & KPI cards)
   - Quality Audit (scoring & export)
   - KBA Articles generation & preview
   - SOPs generation & preview
   - Fishbone diagram generation & export
   - Impact analysis (KPI ranking, bottlenecks, interactive figures)
   - Automation Opportunities (disposition analysis & ROI)
   - Category Pivot tree hierarchy
   - Word Cloud studio launch wiring
"""
import os
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QFileDialog

from src.gui import ClusterApp
from src.session import save_session, load_session


def _drain_events(app, timeout_ms=600):
    """Pumps the event loop so cross-thread signal deliveries and timers finish."""
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(app.quit)
    timer.start(timeout_ms)
    app.exec()


def _sample_tickets_excel(tmp_path):
    """Create a realistic sample tickets Excel workbook."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "Incident ID": [f"INC{i:05d}" for i in range(1, 21)],
        "Short Description": [
            "User unable to login password expired",
            "Password reset required for ERP system",
            "Cannot login account locked out",
            "VPN connection keeps dropping after 5 minutes",
            "VPN tunnel disconnection error 412",
            "Remote VPN access denied",
            "Outlook search indexing stopped working",
            "Outlook crashes when opening attachments",
            "Mailbox quota full unable to receive emails",
            "SAP GUI freezes on transaction launch",
            "SAP slow response on payroll module",
            "SAP client error 1005 connection timed out",
            "Printer paper jam on 3rd floor",
            "Network printer not responding",
            "Disk space alert C drive 98% full",
            "Server memory threshold exceeded",
            "Request for new employee laptop setup",
            "Request software license for Adobe Acrobat",
            "Monitor flickering when connected via dock",
            "Keyboard keys sticking",
        ],
        "Close Notes": [
            "Reset password in Active Directory and verified login",
            "Provided temporary password and forced reset on next login",
            "Unlocked AD account and confirmed access",
            "Reconfigured VPN client adapter MTU size",
            "Reinstalled Cisco AnyConnect client and certificates",
            "Updated remote access profile and restarted tunnel",
            "Rebuilt Windows search index and verified Outlook",
            "Updated Outlook to latest build and disabled add-in",
            "Archived old email folders to reduce mailbox size",
            "Cleared SAP GUI local cache and restarted application",
            "Adjusted SAP GUI buffer settings and checked network latency",
            "Reconnected SAP application server endpoint",
            "Cleared paper path and ran test print",
            "Power cycled network printer and reset print spooler",
            "Cleaned temporary files and expanded virtual disk",
            "Restarted leaking background service to free memory",
            "Standard corporate laptop image deployed with base apps",
            "Assigned available license key in portal",
            "Replaced Thunderbolt dock cable",
            "Replaced keyboard with spare unit",
        ],
        "Category": [
            "Access & Authorization", "Access & Authorization", "Access & Authorization",
            "Network & Connectivity", "Network & Connectivity", "Network & Connectivity",
            "Software & Office", "Software & Office", "Software & Office",
            "Business Applications", "Business Applications", "Business Applications",
            "Hardware & Peripherals", "Hardware & Peripherals",
            "Infrastructure & Servers", "Infrastructure & Servers",
            "Service Request", "Service Request",
            "Hardware & Peripherals", "Hardware & Peripherals",
        ],
        "Priority": ["3 - Moderate"] * 10 + ["2 - High"] * 10,
        "Business Duration": [3600, 1800, 2400, 7200, 5400, 3600,
                              4800, 9000, 3600, 10800, 14400, 7200,
                              1800, 2400, 3600, 7200, 28800, 14400,
                              3600, 1800],
        "Reopen Count": [0, 1, 0, 2, 1, 0, 1, 2, 0, 3, 2, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "Reassignment Count": [0, 0, 1, 2, 1, 0, 1, 3, 0, 2, 3, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    })
    path = str(tmp_path / "sample_tickets.xlsx")
    df.to_excel(path, index=False, sheet_name="All Incidents")
    return path, df


@pytest.fixture
def mock_ui_dialogs(monkeypatch):
    """Mocks all blocking Qt message boxes and file dialogs."""
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "exec", lambda *a, **k: 0)
    monkeypatch.setattr(QMessageBox, "exec_", lambda *a, **k: 0)


def test_window_geometry_and_responsive_resizing(qapp):
    """App window must adapt to various screen resolutions without clipping."""
    win = ClusterApp()
    win.show()
    qapp.processEvents()

    # Verify minHint is within manageable laptop limits
    assert win.minimumSizeHint().width() <= 960, f"min width too large: {win.minimumSizeHint().width()}"

    # Test distinct display resolutions
    for w, h in [(960, 600), (1200, 800), (1440, 900), (1920, 1080)]:
        win.resize(w, h)
        qapp.processEvents()
        assert win.width() == w, f"Failed resizing width to {w}, got {win.width()}"
        assert win.height() == h, f"Failed resizing height to {h}, got {win.height()}"

    # Verify pages and tabs fit within window bounds
    assert win.pages.width() <= win.width()
    assert win.analysis_page.tab_scroll.horizontalScrollBar() is not None


def test_import_and_preview_workflow(qapp, tmp_path, mock_ui_dialogs, monkeypatch):
    """Test importing an Excel file, selecting columns, and showing preview."""
    file_path, df = _sample_tickets_excel(tmp_path)
    win = ClusterApp()

    # Mock file picker to return sample Excel
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (file_path, "Excel Files (*.xlsx *.xls)"))

    # Load file
    win.load_file()
    qapp.processEvents()

    assert win.source_file_path == file_path
    assert win.import_page.lbl_file.text() == os.path.basename(file_path)
    assert win.import_page.sheet_dropdown.currentText() == "All Incidents"
    assert len(win.import_page.checkboxes) == len(df.columns)

    # Select text columns
    for cb in win.import_page.checkboxes:
        if cb.text() in ("Short Description", "Close Notes"):
            cb.setChecked(True)
        else:
            cb.setChecked(False)

    # Show preview
    win.show_preview()
    assert win.import_page.btn_preview.isEnabled() is True
    assert "loaded" in win.import_page.ready_label.text().lower()


def test_end_to_end_clustering_and_analysis_tabs(qapp, tmp_path, mock_ui_dialogs, monkeypatch, fake_llm):
    """End-to-end execution of clustering followed by all 10 analysis dashboard features."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    import src.clustering as clustering
    import webbrowser

    # Avoid launching external web browser for plotly / fishbone
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

    file_path, raw_df = _sample_tickets_excel(tmp_path)
    win = ClusterApp()

    # Load file
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (file_path, "Excel Files (*.xlsx *.xls)"))
    win.load_file()
    qapp.processEvents()

    # Select columns
    for cb in win.import_page.checkboxes:
        cb.setChecked(cb.text() in ("Short Description", "Close Notes"))

    # Mock ML models for fast deterministic clustering run
    monkeypatch.setattr(clustering, "_lazy_import_ml", lambda: None)
    monkeypatch.setattr(clustering, "_np", np, raising=False)

    class FastUMAP:
        def __init__(self, **kw):
            self.n_components = 5
        def fit(self, X): return self
        def transform(self, X): return np.asarray(X)[:, :5]
        def fit_transform(self, X): return self.transform(X)

    class FastHDBSCAN:
        def __init__(self, **kw):
            self.relative_validity_ = 0.6
        def fit_predict(self, X):
            labels = np.array([i % 3 for i in range(len(X))])
            labels[-1] = -1
            return labels

    class FastTopicModel:
        def __init__(self, **kw):
            self.topic_embeddings_ = np.zeros((4, 16), dtype="float32")
        def fit_transform(self, docs, embeddings):
            topics = [i % 3 for i in range(len(docs))]
            topics[-1] = -1
            return topics, None
        def get_topic_info(self):
            return pd.DataFrame({"Topic": [-1, 0, 1, 2],
                                 "Name": ["-1_noise", "0_login_password", "1_vpn_network", "2_sap_freeze"]})
        def get_topic(self, tid):
            return [("login", 0.9), ("password", 0.8), ("account", 0.6)]
        def get_topics(self):
            return {-1: [("noise", 0.5)], 0: [("login", 0.9)], 1: [("vpn", 0.9)], 2: [("sap", 0.9)]}

    class FastVectorizer:
        def __init__(self, **kw): pass

    monkeypatch.setattr(clustering, "_UMAP", FastUMAP, raising=False)
    monkeypatch.setattr(clustering, "_HDBSCAN", FastHDBSCAN, raising=False)
    monkeypatch.setattr(clustering, "_BERTopic", FastTopicModel, raising=False)
    monkeypatch.setattr(clustering, "_CountVectorizer", FastVectorizer, raising=False)
    monkeypatch.setattr(clustering, "_ClassTfidfTransformer", FastVectorizer, raising=False)
    monkeypatch.setattr(clustering, "get_stopwords", lambda *a, **k: ["the", "a", "is", "for"])

    n_rows = len(win.df)
    mock_embeddings = np.random.default_rng(42).random((n_rows, 16)).astype("float32")

    class FastEmbModel:
        def encode(self, batch, **kw):
            return mock_embeddings[:len(batch)]

    monkeypatch.setattr(
        clustering, "get_embedding_model_resolved",
        lambda *a, **k: (FastEmbModel(), "pytorch", None)
    )

    llm_instance = fake_llm(
        default='{"category": "IT Support", "subcategory": "General Issue", "disposition": "Automate", "root_cause": "Configuration", "resolution": "Applied fix", "recommendation": "Self service"}',
        replies={
            "Naming Subcategories": '{"subcategory": "Login & Access Issues"}',
            "Macro-Category": '{"category": "Access & Authorization"}',
            "KBA": "Title: Resolving Login Issues\n\nSteps:\n1. Check network\n2. Reset credentials",
            "SOP": "Standard Operating Procedure for Account Unlock",
        }
    )

    # Configure clustering settings
    win.settings_page.chk_llm.setChecked(False)
    win.clustering_page.chk_pick_after_embed.setChecked(False)
    win.clustering_page.spin_cluster_size.setValue(3)

    # Execute clustering
    selected_cols = ["Short Description", "Close Notes"]
    mapping = {
        "priority_col": "Priority",
        "duration_col": "Business Duration",
        "reopen_col": "Reopen Count",
        "reassign_col": "Reassignment Count",
        "category_col": "Category",
        "resolution_notes_col": "Close Notes",
    }
    settings = win._gather_settings()

    win.run_clustering(
        cols=selected_cols,
        model_path="",
        min_size=3,
        save_intermediate=False,
        use_preprocessing=True,
        settings=settings,
        mapping=mapping,
    )
    _drain_events(qapp, 600)

    # Assert clustering results
    assert "Cluster_ID" in win.df.columns
    assert "Repetitive Subcategory" in win.df.columns
    assert "Repetitive Category" in win.df.columns
    assert win.cluster_data is not None
    assert len(win.cluster_data) > 0
    assert win.clustering_page.results_card.isVisibleTo(win.clustering_page)

    win.clusterer.llm = llm_instance
    win.settings_page.chk_llm.setChecked(True)

    # -------------------------------------------------------------
    # Test Session Round Trip
    # -------------------------------------------------------------
    session_file = str(tmp_path / "test_run.tlens")
    save_session(
        session_file,
        df=win.df,
        cluster_data=win.cluster_data,
        analysis_results=win.analysis_results,
        kba_articles=[],
        sop_documents=[],
        selected_text_cols=win.selected_text_cols,
        settings=win.config,
        source_file=win.source_file_path,
        sheet_name="All Incidents",
        app_version="1.2.0",
    )
    loaded = load_session(session_file)
    assert len(loaded["df"]) == len(win.df)
    assert set(loaded["cluster_data"].keys()) == set(win.cluster_data.keys())

    # -------------------------------------------------------------
    # Test Analysis Tab 0: Overview (Main Theme)
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(0)
    win._run_main_theme()
    _drain_events(qapp, 400)
    assert "main_theme" in win.analysis_results
    assert win.analysis_page.overview_headline.text() != ""
    assert win.analysis_page.overview_cards_layout.count() > 1

    # -------------------------------------------------------------
    # Test Analysis Tab 1: Quality Audit
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(1)
    win._run_quality_audit()
    _drain_events(qapp, 1000)
    assert "audit_summary" in win.analysis_results
    assert win.analysis_page.btn_audit_export.isEnabled() is True

    audit_export_path = str(tmp_path / "audit_report.xlsx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (audit_export_path, "Excel (*.xlsx)"))
    win._export_audit()
    assert os.path.exists(audit_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 2: KBA Articles
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(2)
    win._run_kba_generation()
    _drain_events(qapp, 1000)
    assert len(win.kba_articles) > 0
    assert win.analysis_page.kba_selector.count() > 0
    assert win.analysis_page.kba_preview.toPlainText() != ""

    kba_export_path = str(tmp_path / "kba.docx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (kba_export_path, "Word Document (*.docx)"))
    win._export_kba()
    assert os.path.exists(kba_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 3: SOP Documents
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(3)
    win._run_sop_generation()
    _drain_events(qapp, 1000)
    assert len(win.sop_documents) > 0
    assert win.analysis_page.sop_selector.count() > 0
    assert win.analysis_page.sop_preview.toPlainText() != ""

    sop_export_path = str(tmp_path / "sops.docx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (sop_export_path, "Word Document (*.docx)"))
    win._export_sop()
    assert os.path.exists(sop_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 4: Fishbone
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(4)
    win._populate_fishbone_scope()
    assert win.analysis_page.fishbone_scope.count() > 0
    win._run_fishbone()
    assert hasattr(win, "_last_fishbone_fig")
    assert win._last_fishbone_fig is not None

    fishbone_export_path = str(tmp_path / "fishbone.html")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (fishbone_export_path, "HTML (*.html)"))
    win._save_fishbone_html()
    assert os.path.exists(fishbone_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 5: Impact Analysis
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(5)
    win._run_impact_analysis()
    _drain_events(qapp, 1000)
    assert "kpi" in win.analysis_results
    assert "problem_clusters" in win.analysis_results
    assert win.analysis_page.btn_impact_export.isEnabled() is True

    impact_export_path = str(tmp_path / "impact.xlsx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (impact_export_path, "Excel (*.xlsx)"))
    win._export_impact()
    assert os.path.exists(impact_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 6: Automation Opportunities (Disposition)
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(6)
    for cb in win.analysis_page.disp_checkboxes:
        cb.setChecked(cb.text() == "Close Notes")
    win._run_disposition_analysis()
    _drain_events(qapp, 1500)
    assert "disposition" in win.analysis_results
    assert "Automation Disposition" in win.df.columns

    disp_export_path = str(tmp_path / "disposition.xlsx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (disp_export_path, "Excel (*.xlsx)"))
    win._export_disposition()
    assert os.path.exists(disp_export_path)

    # -------------------------------------------------------------
    # Test Analysis Tab 7: Word Cloud Studio launcher
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(7)
    assert win.analysis_page.btn_wc_open_studio is not None
    assert win.analysis_page.btn_wc_open_studio.isEnabled() is True

    # -------------------------------------------------------------
    # Test Analysis Tab 8: Category Audit
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(8)
    win._run_category_audit()
    _drain_events(qapp, 1000)
    assert win.analysis_page.cat_audit_summary.text() != ""

    # -------------------------------------------------------------
    # Test Analysis Tab 9: Category Pivot Tree
    # -------------------------------------------------------------
    win.analysis_page._switch_tab(9)
    win._refresh_category_pivot()
    assert win.analysis_page.pivot_tree.topLevelItemCount() > 0
    assert win.analysis_page.btn_pivot_export.isEnabled() is True

    pivot_export_path = str(tmp_path / "pivot.xlsx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (pivot_export_path, "Excel (*.xlsx)"))
    win._export_category_pivot()
    assert os.path.exists(pivot_export_path)
