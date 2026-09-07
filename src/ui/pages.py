"""The four main pages: Import, Settings, Clustering and Analysis.

These are widget construction with almost no logic -- the controller reaches in
by attribute (`self.settings_page.chk_llm`, ...) to read and connect them, so the
access pattern is unchanged by living here. The only behaviour is local: the
Settings page's advanced-section toggle and its reset-to-defaults button, plus the
Analysis page's tab switching.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QLineEdit, QTextEdit, QProgressBar, QScrollArea,
    QFrame, QStackedWidget, QSpinBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from src.config import (ACCELERATION_OPTIONS, DEFAULTS, EMBEDDING_MODELS,
                        LLM_CUSTOM_SENTINEL, LLM_MODELS, MLX_LLM_MODELS)
from src.mlx_backend import is_apple_silicon
from src.ui.theme import (COLORS, MONO_FAMILY, make_card, make_data_tree,
                          make_metric_card, make_section_label, styled_button)


# ---------------------------------------------------------------
# Page 0: Import Data
# ---------------------------------------------------------------
class ImportPage(QScrollArea):
    """Data import page with file loading, sheet selection, and column picking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(32, 20, 32, 20)
        self.main_layout.setSpacing(12)
        self.setWidget(container)

        # Page header
        header = QLabel("Import Your Data")
        header.setFont(QFont(MONO_FAMILY, 22, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {COLORS['text']};")
        self.main_layout.addWidget(header)

        desc = QLabel("Load an Excel file, select the sheet and columns to cluster.")
        desc.setFont(QFont(MONO_FAMILY, 12))
        desc.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.main_layout.addWidget(desc)
        self.main_layout.addSpacing(8)

        # Step 1: File
        card1, layout1 = make_card("Step 1 \u2014 Load Excel File")
        self.main_layout.addWidget(card1)

        row = QHBoxLayout()
        self.btn_load = styled_button("Browse for Excel File", COLORS['accent'], "\U0001F4C1")
        row.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setFont(QFont(MONO_FAMILY, 11))
        self.lbl_file.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row.addWidget(self.lbl_file, 1)
        layout1.addLayout(row)

        # File info card (hidden until file loaded)
        self.file_info_frame = QFrame()
        self.file_info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
                padding: 6px;
            }}
        """)
        self.file_info_layout = QHBoxLayout(self.file_info_frame)
        self.file_info_layout.setContentsMargins(12, 6, 12, 6)
        self.lbl_file_info = QLabel("")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.file_info_layout.addWidget(self.lbl_file_info)
        self.file_info_frame.hide()
        layout1.addWidget(self.file_info_frame)

        # Step 2: Sheet
        card2, layout2 = make_card("Step 2 \u2014 Select Sheet")
        self.main_layout.addWidget(card2)

        self.sheet_dropdown = QComboBox()
        self.sheet_dropdown.addItem("-- Load a file first --")
        self.sheet_dropdown.setEnabled(False)
        self.sheet_dropdown.setMinimumHeight(34)
        layout2.addWidget(self.sheet_dropdown)

        # Step 3: Columns
        card3, layout3 = make_card("Step 3 \u2014 Select Text Columns for Clustering")
        self.main_layout.addWidget(card3)

        hint = QLabel("Check the columns that contain ticket descriptions, summaries, or notes to analyze.")
        hint.setFont(QFont(MONO_FAMILY, 10))
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        hint.setWordWrap(True)
        layout3.addWidget(hint)

        # Scrollable column list
        self.col_scroll = QScrollArea()
        self.col_scroll.setWidgetResizable(True)
        self.col_scroll.setMinimumHeight(90)
        self.col_scroll.setMaximumHeight(260)
        self.col_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
            }}
        """)
        self.col_container = QWidget()
        self.col_container.setStyleSheet("background: transparent;")
        self.col_layout = QVBoxLayout(self.col_container)
        self.col_layout.setContentsMargins(12, 8, 12, 8)
        self.col_layout.setSpacing(4)
        self.col_layout.addStretch()
        self.col_scroll.setWidget(self.col_container)
        layout3.addWidget(self.col_scroll)

        self.checkboxes = []

        # Preview button + ready indicator in step 3
        preview_row = QHBoxLayout()
        self.btn_preview = styled_button("Preview Sample (before / after cleaning)", COLORS['bg_card'], "\U0001F50D")
        self.btn_preview.setEnabled(False)
        self.btn_preview.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
                padding: 8px 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {COLORS['sidebar_hover']}; color: {COLORS['text']}; }}
            QPushButton:disabled {{ color: {COLORS['text_muted']}; }}
        """)
        preview_row.addWidget(self.btn_preview)

        self.ready_label = QLabel("")
        self.ready_label.setFont(QFont(MONO_FAMILY, 11))
        self.ready_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        preview_row.addWidget(self.ready_label, 1)
        layout3.addLayout(preview_row)

        self.main_layout.addStretch()

    def set_ready(self, ready, msg=""):
        """Update the ready indicator."""
        if ready:
            self.ready_label.setText(f"\u2705  {msg}")
            self.ready_label.setStyleSheet(f"color: {COLORS['accent_green']};")
        else:
            self.ready_label.setText(msg)
            self.ready_label.setStyleSheet(f"color: {COLORS['text_muted']};")


# ---------------------------------------------------------------
# Page 1: Settings
# ---------------------------------------------------------------
class SettingsPage(QScrollArea):
    """All configuration in organized, collapsible sections."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(32, 24, 32, 24)
        self.main_layout.setSpacing(16)
        self.setWidget(container)

        # Header
        header = QLabel("Configuration")
        header.setFont(QFont(MONO_FAMILY, 22, QFont.Weight.Bold))
        self.main_layout.addWidget(header)

        desc = QLabel("Tune preprocessing, clustering, and AI model settings.")
        desc.setFont(QFont(MONO_FAMILY, 12))
        desc.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.main_layout.addWidget(desc)
        self.main_layout.addSpacing(8)

        # AI Model
        card_ai, layout_ai = make_card("AI Model")
        self.main_layout.addWidget(card_ai)

        # Curated LLM dropdown (+ "Custom file…"); resolved/downloaded on demand.
        # MLX entries are offered only on Apple Silicon. Listing them everywhere
        # would let a Windows user pick a model that can never load; the engine
        # would fall back silently and the dropdown would keep claiming MLX.
        _mlx_labels = list(MLX_LLM_MODELS.keys()) if is_apple_silicon() else []
        self.llm_options = list(LLM_MODELS.keys()) + _mlx_labels + [LLM_CUSTOM_SENTINEL]
        self.llm_dropdown = QComboBox()
        self.llm_dropdown.addItems(self.llm_options)
        self.llm_dropdown.setMinimumHeight(36)
        layout_ai.addWidget(self.llm_dropdown)

        row_status = QHBoxLayout()
        self.lbl_model_status = QLabel("")
        self.lbl_model_status.setWordWrap(True)
        self.lbl_model_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row_status.addWidget(self.lbl_model_status, 1)
        # "Download now" appears when the selected curated model isn't downloaded yet.
        self.btn_download_model = styled_button("Download now", COLORS['accent'], "⬇", small=True)
        self.btn_download_model.setVisible(False)
        row_status.addWidget(self.btn_download_model)
        layout_ai.addLayout(row_status)

        self.chk_llm = QCheckBox("Enable AI Naming (uses local LLM for cluster names)")
        self.chk_llm.setChecked(config["llm"]["enabled"])
        layout_ai.addWidget(self.chk_llm)

        self.chk_intermediate = QCheckBox("Save keywords-only intermediate file")
        self.chk_intermediate.setChecked(False)
        layout_ai.addWidget(self.chk_intermediate)

        # Embedding Model
        card_emb, layout_emb = make_card("Embedding Model")
        self.main_layout.addWidget(card_emb)

        # Single source of truth: dropdown order = EMBEDDING_MODELS dict order
        # (bge-base first, so it stays the startup default).
        self.embedding_options = list(EMBEDDING_MODELS.keys())
        self.embedding_dropdown = QComboBox()
        self.embedding_dropdown.addItems(self.embedding_options)
        self.embedding_dropdown.setMinimumHeight(36)
        layout_emb.addWidget(self.embedding_dropdown)

        row_emb_status = QHBoxLayout()
        self.lbl_embed_status = QLabel("")
        self.lbl_embed_status.setWordWrap(True)
        self.lbl_embed_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row_emb_status.addWidget(self.lbl_embed_status, 1)
        # "Download now" appears when the selected embedding model isn't local yet.
        self.btn_download_embed = styled_button("Download now", COLORS['accent'], "⬇", small=True)
        self.btn_download_embed.setVisible(False)
        row_emb_status.addWidget(self.btn_download_embed)
        layout_emb.addLayout(row_emb_status)

        # Restore the persisted embedding model selection (default = first item).
        _emb_cfg = config.get("embedding", {})
        _saved_emb = _emb_cfg.get("model_name")
        if _saved_emb and _saved_emb in self.embedding_options:
            self.embedding_dropdown.blockSignals(True)
            self.embedding_dropdown.setCurrentText(_saved_emb)
            self.embedding_dropdown.blockSignals(False)

        # --- Acceleration (experimental) ---------------------------------------
        layout_emb.addWidget(make_section_label("Acceleration"))
        self._accel_labels = list(ACCELERATION_OPTIONS.keys())  # label order = dropdown order
        self.accel_dropdown = QComboBox()
        self.accel_dropdown.addItems(self._accel_labels)
        self.accel_dropdown.setMinimumHeight(36)
        layout_emb.addWidget(self.accel_dropdown)
        # restore persisted acceleration choice (value -> label)
        _saved_accel = _emb_cfg.get("acceleration", "pytorch")
        _accel_label = next((l for l, v in ACCELERATION_OPTIONS.items() if v == _saved_accel),
                            self._accel_labels[0])
        self.accel_dropdown.blockSignals(True)
        self.accel_dropdown.setCurrentText(_accel_label)
        self.accel_dropdown.blockSignals(False)

        row_accel = QHBoxLayout()
        self.lbl_accel_status = QLabel("Experimental. PyTorch is the reproducible default.")
        self.lbl_accel_status.setWordWrap(True)
        self.lbl_accel_status.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row_accel.addWidget(self.lbl_accel_status, 1)
        # Optional one-time INT8 build (visible only when an OV option + dataset are ready).
        self.btn_build_accel = styled_button("Build accelerated model", COLORS['accent'], "⚡", small=True)
        self.btn_build_accel.setVisible(False)
        row_accel.addWidget(self.btn_build_accel)
        # Shown only when the OpenVINO stack isn't installed (Windows/Linux).
        self.btn_install_ov = styled_button("Install OpenVINO support…", COLORS['text_muted'], "⬇", small=True)
        self.btn_install_ov.setVisible(False)
        row_accel.addWidget(self.btn_install_ov)
        # Shown only on Apple Silicon when the MLX stack isn't installed.
        self.btn_install_mlx = styled_button("Install MLX support…", COLORS['text_muted'], "⬇", small=True)
        self.btn_install_mlx.setVisible(False)
        self.btn_install_mlx.setToolTip("Install MLX acceleration for Apple Silicon (runs install_mlx.sh)")
        row_accel.addWidget(self.btn_install_mlx)
        layout_emb.addLayout(row_accel)

        # Clustering Parameters
        card_clust, layout_clust = make_card("Clustering Parameters")
        self.main_layout.addWidget(card_clust)

        cs = config["clustering"]

        row_cs = QHBoxLayout()
        row_cs.addWidget(QLabel("Min Cluster Size:"))
        self.spin_cluster_size = QSpinBox()
        self.spin_cluster_size.setRange(2, 10000)
        self.spin_cluster_size.setValue(cs["min_cluster_size"])
        self.spin_cluster_size.setFixedWidth(100)
        row_cs.addWidget(self.spin_cluster_size)
        self.btn_suggest_cs = styled_button("Suggest", COLORS['accent'], "💡", small=True)
        row_cs.addWidget(self.btn_suggest_cs)
        hint_cs = QLabel("Minimum documents per cluster")
        hint_cs.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row_cs.addWidget(hint_cs, 1)
        layout_clust.addLayout(row_cs)
        hint_sg = QLabel("Suggest analyzes your data and recommends a value (streams to the console).")
        hint_sg.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        hint_sg.setWordWrap(True)
        layout_clust.addWidget(hint_sg)

        # Advanced Clustering (collapsible)
        self.adv_toggle = QPushButton("\u25B8  Advanced UMAP & HDBSCAN Settings")
        self.adv_toggle.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                text-align: left;
                padding: 6px 0;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {COLORS['text']}; }}
        """)
        self.adv_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        layout_clust.addWidget(self.adv_toggle)

        self.adv_frame = QFrame()
        self.adv_frame.setStyleSheet(f"QFrame {{ background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 10px; }}")
        adv_layout = QGridLayout(self.adv_frame)
        adv_layout.setSpacing(8)

        adv_layout.addWidget(QLabel("UMAP n_neighbors:"), 0, 0)
        self.spin_umap_neighbors = QSpinBox()
        self.spin_umap_neighbors.setRange(2, 200)
        self.spin_umap_neighbors.setValue(cs["umap_n_neighbors"])
        adv_layout.addWidget(self.spin_umap_neighbors, 0, 1)

        adv_layout.addWidget(QLabel("UMAP n_components:"), 1, 0)
        self.spin_umap_components = QSpinBox()
        self.spin_umap_components.setRange(2, 100)
        self.spin_umap_components.setValue(cs["umap_n_components"])
        adv_layout.addWidget(self.spin_umap_components, 1, 1)

        adv_layout.addWidget(QLabel("UMAP min_dist:"), 2, 0)
        self.spin_umap_mindist = QDoubleSpinBox()
        self.spin_umap_mindist.setRange(0.0, 1.0)
        self.spin_umap_mindist.setSingleStep(0.05)
        self.spin_umap_mindist.setValue(cs["umap_min_dist"])
        adv_layout.addWidget(self.spin_umap_mindist, 2, 1)

        adv_layout.addWidget(QLabel("UMAP metric:"), 3, 0)
        self.combo_umap_metric = QComboBox()
        self.combo_umap_metric.addItems(["euclidean", "cosine", "manhattan"])
        self.combo_umap_metric.setCurrentText(cs["umap_metric"])
        adv_layout.addWidget(self.combo_umap_metric, 3, 1)

        adv_layout.addWidget(QLabel("HDBSCAN min_samples:"), 4, 0)
        self.edit_hdbscan_samples = QLineEdit()
        self.edit_hdbscan_samples.setPlaceholderText("auto")
        if cs["hdbscan_min_samples"] is not None:
            self.edit_hdbscan_samples.setText(str(cs["hdbscan_min_samples"]))
        adv_layout.addWidget(self.edit_hdbscan_samples, 4, 1)

        adv_layout.addWidget(QLabel("HDBSCAN selection:"), 5, 0)
        self.combo_hdbscan_method = QComboBox()
        self.combo_hdbscan_method.addItems(["eom", "leaf"])
        self.combo_hdbscan_method.setCurrentText(cs["hdbscan_cluster_selection_method"])
        adv_layout.addWidget(self.combo_hdbscan_method, 5, 1)

        # UMAP CPU threading. Parallel UMAP drops its random seed (not reproducible),
        # so "Auto" only parallelizes once the data is large enough to be worth it.
        adv_layout.addWidget(QLabel("UMAP CPU threads:"), 6, 0)
        self.combo_umap_parallel = QComboBox()
        self._umap_parallel_modes = ["auto", "reproducible", "parallel"]
        self.combo_umap_parallel.addItems([
            "Auto (parallel on large data)",
            "Reproducible (1 core)",
            "Parallel (all cores)",
        ])
        _mode = cs.get("umap_parallel_mode", "auto")
        self.combo_umap_parallel.setCurrentIndex(
            self._umap_parallel_modes.index(_mode) if _mode in self._umap_parallel_modes else 0)
        adv_layout.addWidget(self.combo_umap_parallel, 6, 1)

        # Restore every advanced control above to the documented defaults.
        self.btn_reset_advanced = styled_button(
            "Reset to Defaults", COLORS['text_muted'], "\u21BA", small=True)
        self.btn_reset_advanced.setToolTip(
            "Reset the UMAP & HDBSCAN settings above to their default values")
        adv_layout.addWidget(self.btn_reset_advanced, 7, 1)
        self.btn_reset_advanced.clicked.connect(self._reset_advanced_defaults)

        self.adv_frame.hide()
        layout_clust.addWidget(self.adv_frame)

        self.adv_toggle.clicked.connect(self._toggle_advanced)
        self._adv_visible = False

        # Text Cleaning
        card_clean, layout_clean = make_card("Text Cleaning")
        self.main_layout.addWidget(card_clean)

        pp = config["preprocessing"]

        self.chk_preprocess = QCheckBox("Enable Text Cleaning")
        self.chk_preprocess.setChecked(True)
        self.chk_preprocess.setFont(QFont(MONO_FAMILY, 12, QFont.Weight.Bold))
        layout_clean.addWidget(self.chk_preprocess)

        clean_grid = QGridLayout()
        clean_grid.setSpacing(4)

        self.chk_rm_emails = QCheckBox("Remove emails")
        self.chk_rm_emails.setChecked(pp["remove_emails"])
        clean_grid.addWidget(self.chk_rm_emails, 0, 0)

        self.chk_rm_urls = QCheckBox("Remove URLs")
        self.chk_rm_urls.setChecked(pp["remove_urls"])
        clean_grid.addWidget(self.chk_rm_urls, 0, 1)

        self.chk_rm_ticket_ids = QCheckBox("Remove ticket IDs")
        self.chk_rm_ticket_ids.setChecked(pp["remove_ticket_ids"])
        clean_grid.addWidget(self.chk_rm_ticket_ids, 1, 0)

        self.chk_rm_timestamps = QCheckBox("Remove timestamps/dates")
        self.chk_rm_timestamps.setChecked(pp["remove_timestamps"])
        clean_grid.addWidget(self.chk_rm_timestamps, 1, 1)

        self.chk_rm_phones = QCheckBox("Remove phone numbers")
        self.chk_rm_phones.setChecked(pp["remove_phone_numbers"])
        clean_grid.addWidget(self.chk_rm_phones, 2, 0)

        self.chk_rm_ips = QCheckBox("Remove IP addresses")
        self.chk_rm_ips.setChecked(pp["remove_ip_addresses"])
        clean_grid.addWidget(self.chk_rm_ips, 2, 1)

        self.chk_rm_paths = QCheckBox("Remove file paths")
        self.chk_rm_paths.setChecked(pp["remove_file_paths"])
        clean_grid.addWidget(self.chk_rm_paths, 3, 0)

        self.chk_rm_special = QCheckBox("Remove special chars")
        self.chk_rm_special.setChecked(pp["remove_special_chars"])
        clean_grid.addWidget(self.chk_rm_special, 3, 1)

        self.chk_rm_boilerplate = QCheckBox("Remove boilerplate")
        self.chk_rm_boilerplate.setChecked(pp["remove_boilerplate"])
        clean_grid.addWidget(self.chk_rm_boilerplate, 4, 0)

        layout_clean.addLayout(clean_grid)

        row_tok = QHBoxLayout()
        row_tok.addWidget(QLabel("Min token length:"))
        self.spin_min_token = QSpinBox()
        self.spin_min_token.setRange(1, 20)
        self.spin_min_token.setValue(pp["min_token_length"])
        self.spin_min_token.setFixedWidth(70)
        row_tok.addWidget(self.spin_min_token)
        row_tok.addStretch()
        layout_clean.addLayout(row_tok)

        # Stopwords
        card_sw, layout_sw = make_card("Stopwords")
        self.main_layout.addWidget(card_sw)

        sw = config["stopwords"]
        self.chk_english_sw = QCheckBox("English stopwords")
        self.chk_english_sw.setChecked(sw["use_english_stopwords"])
        layout_sw.addWidget(self.chk_english_sw)

        self.chk_it_sw = QCheckBox("IT domain stopwords")
        self.chk_it_sw.setChecked(sw["use_it_stopwords"])
        layout_sw.addWidget(self.chk_it_sw)

        self.btn_manage_sw = styled_button("Manage Custom Stopwords", COLORS['text_muted'],"\u270F\uFE0F", small=True)
        layout_sw.addWidget(self.btn_manage_sw)

        # Metadata Column Mapping
        card_meta, layout_meta = make_card("Metadata Column Mapping")
        self.main_layout.addWidget(card_meta)

        meta_hint = QLabel("Map Excel columns for deeper analytics (optional).")
        meta_hint.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        meta_hint.setWordWrap(True)
        layout_meta.addWidget(meta_hint)

        self.metadata_dropdowns = {}
        roles = [
            ("priority_col", "Priority"),
            ("resolution_time_col", "Resolution Time"),
            ("sla_status_col", "SLA Status"),
            ("assignment_group_col", "Assignment Group"),
            ("created_date_col", "Created Date"),
            ("resolved_date_col", "Resolved Date"),
            ("category_col", "Original Category"),
            ("resolution_notes_col", "Resolution Notes"),
        ]

        meta_grid = QGridLayout()
        meta_grid.setSpacing(6)
        for i, (role_key, label) in enumerate(roles):
            meta_grid.addWidget(QLabel(f"{label}:"), i, 0)
            dd = QComboBox()
            dd.addItem("-- Not mapped --")
            dd.setEnabled(False)
            dd.setMinimumWidth(180)
            meta_grid.addWidget(dd, i, 1)
            self.metadata_dropdowns[role_key] = dd
        layout_meta.addLayout(meta_grid)

        self.main_layout.addStretch()

    def _toggle_advanced(self):
        """Toggle visibility of advanced settings."""
        self._adv_visible = not self._adv_visible
        self.adv_frame.setVisible(self._adv_visible)
        arrow = "\u25BE" if self._adv_visible else "\u25B8"
        self.adv_toggle.setText(f"{arrow}  Advanced UMAP & HDBSCAN Settings")

    def _reset_advanced_defaults(self):
        """Restore the advanced UMAP/HDBSCAN controls to the documented defaults."""
        d = DEFAULTS["clustering"]
        self.spin_umap_neighbors.setValue(d["umap_n_neighbors"])
        self.spin_umap_components.setValue(d["umap_n_components"])
        self.spin_umap_mindist.setValue(d["umap_min_dist"])
        self.combo_umap_metric.setCurrentText(d["umap_metric"])
        # None => "auto": clear the field so the placeholder shows.
        samples = d["hdbscan_min_samples"]
        self.edit_hdbscan_samples.setText("" if samples is None else str(samples))
        self.combo_hdbscan_method.setCurrentText(d["hdbscan_cluster_selection_method"])
        mode = d.get("umap_parallel_mode", "auto")
        self.combo_umap_parallel.setCurrentIndex(
            self._umap_parallel_modes.index(mode) if mode in self._umap_parallel_modes else 0)


# ---------------------------------------------------------------
# Page 2: Clustering
# ---------------------------------------------------------------
class ClusteringPage(QScrollArea):
    """Run clustering page with progress and status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 20, 32, 20)
        layout.setSpacing(14)
        self.setWidget(container)

        header = QLabel("Run Clustering")
        header.setFont(QFont(MONO_FAMILY, 22, QFont.Weight.Bold))
        layout.addWidget(header)

        desc = QLabel("Execute the clustering pipeline on your loaded data.")
        desc.setFont(QFont(MONO_FAMILY, 12))
        desc.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(desc)
        layout.addSpacing(16)

        # Status card
        card, card_layout = make_card()
        layout.addWidget(card)

        # Summary row
        self.summary_layout = QHBoxLayout()
        self.lbl_docs = make_metric_card("Documents", "\u2014", COLORS['accent'])
        self.summary_layout.addWidget(self.lbl_docs)
        self.lbl_cols = make_metric_card("Columns", "\u2014", COLORS['accent_green'])
        self.summary_layout.addWidget(self.lbl_cols)
        self.lbl_model = make_metric_card("Embedding", "\u2014", COLORS['accent_purple'])
        self.summary_layout.addWidget(self.lbl_model)
        self.lbl_estimate = make_metric_card("Est. Time", "\u2014", COLORS['accent_orange'])
        self.summary_layout.addWidget(self.lbl_estimate)
        self.summary_layout.addStretch()
        card_layout.addLayout(self.summary_layout)

        card_layout.addSpacing(8)

        # Min Cluster Size \u2014 mirrored from the Settings page (kept two-way synced in
        # ClusterApp._connect_signals) so the most impactful knob can be tuned right
        # here without round-tripping to Settings between runs.
        row_cs = QHBoxLayout()
        row_cs.addWidget(QLabel("Min Cluster Size:"))
        self.spin_cluster_size = QSpinBox()
        self.spin_cluster_size.setRange(2, 10000)
        self.spin_cluster_size.setFixedWidth(100)
        row_cs.addWidget(self.spin_cluster_size)
        self.btn_suggest_cs = styled_button("Suggest", COLORS['accent'], "\U0001F4A1", small=True)
        row_cs.addWidget(self.btn_suggest_cs)
        hint_cs = QLabel("Higher \u2192 fewer, larger clusters")
        hint_cs.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        row_cs.addWidget(hint_cs, 1)
        card_layout.addLayout(row_cs)

        # Mid-run picker. Embedding is the slow part and it doesn't depend on the cluster
        # size at all, so the honest moment to choose is once it's done: the sweep then
        # costs only HDBSCAN passes and the numbers shown are for this exact dataset.
        # Uncheck to run straight through with the value above (e.g. re-running a size
        # already settled on). Defaulted on here; ClusterApp._connect_signals restores
        # the persisted value, since this page has no access to config at construction.
        self.chk_pick_after_embed = QCheckBox(
            "Choose Min Cluster Size after embedding (shows scored options)")
        self.chk_pick_after_embed.setChecked(True)
        card_layout.addWidget(self.chk_pick_after_embed)

        card_layout.addSpacing(8)

        # Run button
        self.btn_run = styled_button("[ \u25B6 RUN CLUSTERING ]", COLORS['accent_green'])
        self.btn_run.setMinimumHeight(52)
        self.btn_run.setFont(QFont(MONO_FAMILY, 15, QFont.Weight.Bold))
        self.btn_run.setEnabled(False)
        card_layout.addWidget(self.btn_run)

        # Stop button \u2014 shown in place of Run while a clustering job is running.
        self.btn_stop = styled_button("[ \u25A0 STOP ]", COLORS['accent_red'])
        self.btn_stop.setMinimumHeight(52)
        self.btn_stop.setFont(QFont(MONO_FAMILY, 15, QFont.Weight.Bold))
        self.btn_stop.setVisible(False)
        card_layout.addWidget(self.btn_stop)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(24)
        card_layout.addWidget(self.progress)

        # Current-step line (terminal-style; detailed stream goes to the console)
        self.lbl_status = QLabel("\u203a Ready \u2014 load data and configure settings first")
        self.lbl_status.setFont(QFont(MONO_FAMILY, 11))
        self.lbl_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lbl_status.setWordWrap(True)
        card_layout.addWidget(self.lbl_status)

        hint = QLabel("Detailed progress streams to the activity.log console below.")
        hint.setFont(QFont(MONO_FAMILY, 9))
        hint.setStyleSheet(f"color: {COLORS['text_muted']};")
        card_layout.addWidget(hint)

        # Results bar (hidden until done)
        self.results_card = QFrame()
        self.results_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['success']};
                border-radius: 2px;
                padding: 16px;
            }}
        """)
        results_layout = QVBoxLayout(self.results_card)
        results_layout.setContentsMargins(16, 12, 16, 12)

        res_title = QLabel("\u2705  Clustering Complete")
        res_title.setFont(QFont(MONO_FAMILY, 14, QFont.Weight.Bold))
        res_title.setStyleSheet(f"color: {COLORS['accent_green']}; border: none;")
        results_layout.addWidget(res_title)

        btn_row = QHBoxLayout()
        self.btn_save = styled_button("Save Results", COLORS['accent'], "\U0001F4BE")
        btn_row.addWidget(self.btn_save)
        self.btn_wordcloud = styled_button("Word Cloud", COLORS['accent_purple'], "\u2601")
        btn_row.addWidget(self.btn_wordcloud)
        btn_row.addStretch()
        results_layout.addLayout(btn_row)

        self.results_card.hide()
        layout.addWidget(self.results_card)

        layout.addStretch()

    def update_metric_card(self, card_widget, value):
        """Update the value label inside a metric card."""
        for child in card_widget.findChildren(QLabel):
            if child.objectName() == "metricValue":
                child.setText(str(value))
                return
        # Fallback: first label
        labels = card_widget.findChildren(QLabel)
        if labels:
            labels[0].setText(str(value))

    def set_status(self, msg, progress=None):
        """Update status label and optional progress bar."""
        self.lbl_status.setText(msg)
        if progress is not None:
            self.progress.setValue(int(progress * 100))


# ---------------------------------------------------------------
# Page 3: Analysis
# ---------------------------------------------------------------
class AnalysisPage(QWidget):
    """Post-clustering analysis dashboard with sub-tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        header = QLabel("Analysis Dashboard")
        header.setFont(QFont(MONO_FAMILY, 22, QFont.Weight.Bold))
        layout.addWidget(header)

        # Tab buttons row inside a horizontal scroll area for responsive layout
        tab_container = QWidget()
        tab_container.setStyleSheet("background: transparent;")
        self.tab_bar = QHBoxLayout(tab_container)
        self.tab_bar.setContentsMargins(0, 0, 0, 0)
        self.tab_bar.setSpacing(6)
        self.tab_buttons = []
        tabs = [
            ("Overview", COLORS['accent']),
            ("Quality Audit", COLORS['accent_orange']),
            ("KBA Articles", COLORS['accent_green']),
            ("SOPs", COLORS['accent_green']),
            ("Fishbone", COLORS['accent_purple']),
            ("Impact Analysis", COLORS['accent_red']),
            ("Automation Opportunities", COLORS['accent_green']),
            ("Word Cloud", COLORS['accent']),
            ("Category Audit", COLORS['accent_purple']),
            ("Category Pivot", COLORS['accent_purple']),
        ]
        for i, (name, color) in enumerate(tabs):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(36)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 2px;
                    padding: 6px 14px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['sidebar_hover']};
                    color: {COLORS['text']};
                }}
                QPushButton:checked {{
                    background-color: {color};
                    color: white;
                    border-color: {color};
                }}
            """)
            btn.clicked.connect(lambda checked, idx=i: self._switch_tab(idx))
            self.tab_bar.addWidget(btn)
            self.tab_buttons.append(btn)
        self.tab_bar.addStretch()

        self.tab_scroll = QScrollArea()
        self.tab_scroll.setWidgetResizable(True)
        self.tab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tab_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tab_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.tab_scroll.setFixedHeight(46)
        self.tab_scroll.setMinimumWidth(100)
        self.tab_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.tab_scroll.setWidget(tab_container)
        layout.addWidget(self.tab_scroll)

        # Stacked content area
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._build_overview_tab()
        self._build_quality_tab()
        self._build_kba_tab()
        self._build_sop_tab()
        self._build_fishbone_tab()
        self._build_impact_tab()
        self._build_disposition_tab()
        self._build_wordcloud_tab()
        self._build_category_audit_tab()
        self._build_category_pivot_tab()

        # Set first tab active
        self.tab_buttons[0].setChecked(True)

    def minimumSizeHint(self):
        return QSize(600, 400)

    def _switch_tab(self, index):
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)
        if 0 <= index < len(self.tab_buttons):
            self.tab_scroll.ensureWidgetVisible(self.tab_buttons[index])

    def _build_overview_tab(self):
        """Overall main-theme summary across all clustered tickets."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.btn_overview = styled_button("Generate Summary", COLORS['accent'], "\U0001F4A1")
        top.addWidget(self.btn_overview)
        top.addStretch()
        layout.addLayout(top)

        self.overview_headline = QLabel(
            "Run clustering, then click \u201cGenerate Summary\u201d to see the overall "
            "main theme of your tickets."
        )
        self.overview_headline.setWordWrap(True)
        self.overview_headline.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.overview_headline)

        self.overview_cards_layout = QHBoxLayout()
        self.overview_cards_layout.setSpacing(12)
        self.overview_cards_layout.addStretch()
        layout.addLayout(self.overview_cards_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.overview_results_widget = QWidget()
        self.overview_results_widget.setStyleSheet("background: transparent;")
        self.overview_results_layout = QVBoxLayout(self.overview_results_widget)
        self.overview_results_layout.setContentsMargins(0, 0, 0, 0)
        self.overview_results_layout.addStretch()
        scroll.setWidget(self.overview_results_widget)
        layout.addWidget(scroll, 1)

        self.stack.addWidget(page)

    def _build_quality_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.btn_audit = styled_button("Run Quality Audit", COLORS['accent_orange'], "\U0001F50D")
        top.addWidget(self.btn_audit)
        self.btn_audit_export = styled_button("Export Report", COLORS['text_muted'],"\U0001F4E4", small=True)
        self.btn_audit_export.setEnabled(False)
        top.addWidget(self.btn_audit_export)
        top.addStretch()
        layout.addLayout(top)

        self.audit_cards_layout = QHBoxLayout()
        self.audit_cards_layout.setSpacing(12)
        self.audit_cards_layout.addStretch()
        layout.addLayout(self.audit_cards_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.audit_results_widget = QWidget()
        self.audit_results_widget.setStyleSheet("background: transparent;")
        self.audit_results_layout = QVBoxLayout(self.audit_results_widget)
        self.audit_results_layout.setContentsMargins(0, 0, 0, 0)
        self.audit_results_layout.addStretch()
        scroll.setWidget(self.audit_results_widget)
        layout.addWidget(scroll, 1)

        self.stack.addWidget(page)

    def _build_kba_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.btn_gen_kba = styled_button("Generate KBA Articles", COLORS['accent_green'], "\U0001F4DD")
        top.addWidget(self.btn_gen_kba)
        self.btn_kba_export = styled_button("Export All", COLORS['text_muted'],"\U0001F4E4", small=True)
        self.btn_kba_export.setEnabled(False)
        top.addWidget(self.btn_kba_export)
        top.addStretch()
        layout.addLayout(top)

        sel = QHBoxLayout()
        sel.addWidget(QLabel("Article:"))
        self.kba_selector = QComboBox()
        self.kba_selector.addItem("-- Generate first --")
        self.kba_selector.setMinimumWidth(220)
        sel.addWidget(self.kba_selector, 1)
        layout.addLayout(sel)

        self.kba_preview = QTextEdit()
        self.kba_preview.setReadOnly(True)
        layout.addWidget(self.kba_preview, 1)

        self.stack.addWidget(page)

    def _build_sop_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.btn_gen_sop = styled_button("Generate SOPs", COLORS['accent_green'], "\U0001F4CB")
        top.addWidget(self.btn_gen_sop)
        self.btn_sop_export = styled_button("Export All", COLORS['text_muted'],"\U0001F4E4", small=True)
        self.btn_sop_export.setEnabled(False)
        top.addWidget(self.btn_sop_export)
        top.addStretch()
        layout.addLayout(top)

        sel = QHBoxLayout()
        sel.addWidget(QLabel("SOP:"))
        self.sop_selector = QComboBox()
        self.sop_selector.addItem("-- Generate first --")
        self.sop_selector.setMinimumWidth(220)
        sel.addWidget(self.sop_selector, 1)
        layout.addLayout(sel)

        self.sop_preview = QTextEdit()
        self.sop_preview.setReadOnly(True)
        layout.addWidget(self.sop_preview, 1)

        self.stack.addWidget(page)

    def _build_fishbone_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(QLabel("Scope:"))
        self.fishbone_scope = QComboBox()
        self.fishbone_scope.addItem("-- Select --")
        self.fishbone_scope.setMinimumWidth(220)
        top.addWidget(self.fishbone_scope, 1)
        self.btn_fishbone_gen = styled_button("Generate", COLORS['accent_purple'], "\U0001F4C8")
        top.addWidget(self.btn_fishbone_gen)
        self.btn_fishbone_save = styled_button("Save as HTML", COLORS['text_muted'],"\U0001F4BE", small=True)
        top.addWidget(self.btn_fishbone_save)
        layout.addLayout(top)

        self.fishbone_status = QLabel("Select a cluster or category and click Generate.")
        self.fishbone_status.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.fishbone_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fishbone_status)
        layout.addStretch()

        self.stack.addWidget(page)

    def _build_impact_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.btn_impact = styled_button("Run Analysis", COLORS['accent_red'], "\U0001F4CA")
        top.addWidget(self.btn_impact)
        self.btn_impact_export = styled_button("Export Report", COLORS['text_muted'],"\U0001F4E4", small=True)
        self.btn_impact_export.setEnabled(False)
        top.addWidget(self.btn_impact_export)
        top.addStretch()
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.impact_results_widget = QWidget()
        self.impact_results_widget.setStyleSheet("background: transparent;")
        self.impact_results_layout = QVBoxLayout(self.impact_results_widget)
        self.impact_results_layout.setContentsMargins(0, 0, 0, 0)
        self.impact_results_layout.addStretch()
        scroll.setWidget(self.impact_results_widget)
        layout.addWidget(scroll, 1)

        self.stack.addWidget(page)

    def _build_disposition_tab(self):
        """Automation Opportunities: cluster resolution text, assign Automate /
        Reimagine / Eradicate / Retain per cluster with root cause + ROI."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        blurb = QLabel(
            "Analyses each subcategory (already clustered) and classifies it as "
            "Automate, Reimagine, Eradicate or Retain using the resolution / work / close notes "
            "of its tickets — ranked by total effort hours (ROI). Adds Root Cause / "
            "Resolution Provided / Automation Disposition columns to the dataset."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(blurb)

        # Resolution-text column picker — scrollable vertical list (datasets have
        # 100+ columns, so a horizontal row overflows off-screen).
        col_hdr = QLabel("Resolution text columns to cluster:")
        col_hdr.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        layout.addWidget(col_hdr)
        self.disp_col_scroll = QScrollArea()
        self.disp_col_scroll.setWidgetResizable(True)
        self.disp_col_scroll.setMinimumHeight(120)
        self.disp_col_scroll.setMaximumHeight(220)
        self.disp_col_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
            }}
        """)
        self.disp_col_container = QWidget()
        self.disp_col_container.setStyleSheet("background: transparent;")
        self.disp_col_layout = QVBoxLayout(self.disp_col_container)
        self.disp_col_layout.setContentsMargins(12, 8, 12, 8)
        self.disp_col_layout.setSpacing(4)
        self.disp_col_layout.addStretch()
        self.disp_col_scroll.setWidget(self.disp_col_container)
        self.disp_checkboxes = []
        layout.addWidget(self.disp_col_scroll)

        top = QHBoxLayout()
        self.btn_disp = styled_button("Find Opportunities", COLORS['accent_green'], "\U0001F916")
        top.addWidget(self.btn_disp)
        self.btn_disp_export = styled_button("Export Report", COLORS['text_muted'], "\U0001F4E4", small=True)
        self.btn_disp_export.setEnabled(False)
        top.addWidget(self.btn_disp_export)
        top.addStretch()
        layout.addLayout(top)

        self.disp_progress = QProgressBar()
        self.disp_progress.setRange(0, 100)
        self.disp_progress.setValue(0)
        self.disp_progress.setMinimumHeight(20)
        self.disp_progress.setVisible(False)
        layout.addWidget(self.disp_progress)

        self.disp_status_label = QLabel("")
        self.disp_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.disp_status_label.setWordWrap(True)
        self.disp_status_label.setVisible(False)
        layout.addWidget(self.disp_status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.disp_results_widget = QWidget()
        self.disp_results_widget.setStyleSheet("background: transparent;")
        self.disp_results_layout = QVBoxLayout(self.disp_results_widget)
        self.disp_results_layout.setContentsMargins(0, 0, 0, 0)
        self.disp_results_layout.addStretch()
        scroll.setWidget(self.disp_results_widget)
        layout.addWidget(scroll, 1)

        self.stack.addWidget(page)

    def _build_wordcloud_tab(self):
        """Launcher for the Word Cloud Studio (the full feature lives in its own window)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        blurb = QLabel(
            "Open the Word Cloud Studio to build presentation-ready word clouds from a "
            "text column or your cluster topics \u2014 with color schemes, shapes, fonts, a "
            "live word-frequency table, custom stopwords, and PNG / JPG / SVG / clipboard export."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(blurb)

        self.btn_wc_open_studio = styled_button("\U0001F3A8  Open Word Cloud Studio", COLORS['accent'])
        self.btn_wc_open_studio.setMinimumHeight(48)
        layout.addWidget(self.btn_wc_open_studio)
        layout.addStretch(1)

        self.stack.addWidget(page)

    def _build_category_audit_tab(self):
        """AI category audit: the LLM re-checks how each cluster is filed and
        proposes reassignments; the user reviews and approves before they apply."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        blurb = QLabel(
            "Uses the local AI model to review how each ticket group is filed into "
            "macro-categories and suggest a better-fitting category where one exists. "
            "You review every suggestion and approve the ones you want before anything "
            "changes \u2014 nothing is applied automatically. Merges near-duplicate categories "
            "and moves misfiled groups; updates the \u201cRepetitive Category\u201d column."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(blurb)

        top = QHBoxLayout()
        self.btn_cat_audit = styled_button("Audit Categories", COLORS['accent_purple'], "\U0001F9E0")
        top.addWidget(self.btn_cat_audit)
        top.addStretch()
        layout.addLayout(top)

        self.cat_audit_progress = QProgressBar()
        self.cat_audit_progress.setRange(0, 100)
        self.cat_audit_progress.setValue(0)
        self.cat_audit_progress.setMinimumHeight(20)
        self.cat_audit_progress.setVisible(False)
        layout.addWidget(self.cat_audit_progress)

        self.cat_audit_status_label = QLabel("")
        self.cat_audit_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.cat_audit_status_label.setWordWrap(True)
        self.cat_audit_status_label.setVisible(False)
        layout.addWidget(self.cat_audit_status_label)

        self.cat_audit_summary = QLabel("")
        self.cat_audit_summary.setWordWrap(True)
        self.cat_audit_summary.setTextFormat(Qt.TextFormat.RichText)
        self.cat_audit_summary.setStyleSheet(f"color: {COLORS['text']}; border: none;")
        self.cat_audit_summary.setVisible(False)
        layout.addWidget(self.cat_audit_summary)
        layout.addStretch(1)

        self.stack.addWidget(page)

    def _build_category_pivot_tab(self):
        """Category → Subcategory ticket breakdown as an interactive tree table.
        Auto-populated after clustering; no button needed to generate it."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        blurb = QLabel(
            "Ticket counts per macro-category and the subcategories within each. "
            "Click a column header to sort; updates automatically after clustering "
            "and after a Category Audit."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(blurb)

        top = QHBoxLayout()
        self.pivot_headline = QLabel("Run clustering to see the category breakdown.")
        self.pivot_headline.setFont(QFont(MONO_FAMILY, 11, QFont.Weight.Bold))
        self.pivot_headline.setStyleSheet(f"color: {COLORS['accent']}; border: none;")
        top.addWidget(self.pivot_headline)
        top.addStretch()
        self.btn_pivot_export = styled_button("Export Pivot", COLORS['text_muted'], "\U0001F4E4", small=True)
        self.btn_pivot_export.setEnabled(False)
        top.addWidget(self.btn_pivot_export)
        layout.addLayout(top)

        self.pivot_tree = make_data_tree(
            ["Category / Subcategory", "Count", "% of total"])
        self.pivot_tree.setSortingEnabled(True)
        layout.addWidget(self.pivot_tree, 1)

        self.stack.addWidget(page)
