"""Word Cloud Studio — a full-featured word-cloud generator window for TicketLens.

Mirrors the reference 'Text Analyzer Pro' studio: a 3-pane window (controls |
live preview | word-counts table) with rich appearance / shape / typography
controls, custom stopwords, and PNG/JPG/SVG/clipboard export. Adapted to PySide6
and TicketLens's terminal theme. Heavy libs (wordcloud, matplotlib) are imported
lazily inside methods.

The theme import below is from src/ui/theme.py, not gui.py: theme is the bottom of
the UI dependency chain and imports no TicketLens module, so there is no cycle to
tiptoe around (this file used to import the same names from gui.py, which worked
only because gui.py imports this one lazily).
"""
import io
import os
import re
import random
from collections import Counter

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QCursor
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSpinBox, QSlider, QLineEdit, QScrollArea, QFrame, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QApplication, QMenu,
    QPlainTextEdit,
)

from src.ui.theme import (COLORS, styled_button, make_section_label,
                          apply_dark_titlebar, install_wheel_guard)
from src.logger import get_logger

logger = get_logger()

RENDER_W, RENDER_H = 1600, 900

# --- Color schemes -----------------------------------------------------------
GRADIENT_COLORMAPS = {
    "Terminal Green": "Greens",
    "Corporate Blue": "Blues",
    "Ocean": "GnBu",
    "Sunset": "YlOrRd",
    "Fire": "OrRd",
    "Berry": "RdPu",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Magma": "magma",
    "Inferno": "inferno",
    "Cool": "cool",
    "Grayscale": "gray",
}
PALETTES = {
    "Terminal Green (multi)": ["#3fb950", "#56d364", "#2ea043", "#aff5b4", "#1a7f37"],
    "Violet": ["#8b5cf6", "#7c3aed", "#5b21b6", "#c4b5fd", "#a78bfa"],
    "Rainbow Mix": ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"],
    "Neon": ["#39ff14", "#ff073a", "#00f7ff", "#ff00ff", "#ffff00", "#ff6600"],
    "Pastel Mix": ["#a8e6cf", "#dcedc1", "#ffd3b6", "#ffaaa5", "#d5aaff", "#a0c4ff"],
    "Ocean Breeze": ["#0077b6", "#00b4d8", "#90e0ef", "#48cae4", "#023e8a"],
    "Tech": ["#00d4ff", "#7928ca", "#ff0080", "#00ff88", "#ff6b6b"],
}
BACKGROUNDS = {
    "White": "white", "Off-White": "#f8f8f8", "Light Gray": "#e0e0e0",
    "Dark Gray": "#2d2d2d", "Black": "black", "Navy": "#1a1a2e",
    "Cream": "#fffef0", "Light Blue": "#e3f2fd", "Transparent": None,
}
SHAPES = ["Rectangle", "Circle", "Oval", "Rounded Rect", "Diamond", "Heart",
          "Star", "Cloud", "Hexagon", "Triangle", "Custom Image…"]

_SCALE_DESC = [
    (0.05, "rank only"), (0.3, "mostly rank-based"), (0.6, "balanced"),
    (0.95, "mostly proportional"), (1.01, "fully proportional"),
]


def _scale_desc(v):
    for thr, txt in _SCALE_DESC:
        if v < thr:
            return txt
    return "fully proportional"


def _palette_color_func(palette):
    def f(*args, **kwargs):
        return random.choice(palette)
    return f


def shape_mask(shape, w=RENDER_W, h=RENDER_H):
    """Return a uint8 numpy mask (255 = excluded background, 0 = word area),
    or None for 'Rectangle' (no mask)."""
    if not shape or shape == "Rectangle":
        return None
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    inside = None

    if shape == "Circle":
        r = min(w, h) / 2.0 * 0.95
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    elif shape == "Oval":
        rx, ry = w / 2.0 * 0.96, h / 2.0 * 0.96
        inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    elif shape == "Rounded Rect":
        rx, ry = w / 2.0 * 0.96, h / 2.0 * 0.96
        inside = ((xx - cx) / rx) ** 4 + ((yy - cy) / ry) ** 4 <= 1.0  # superellipse
    elif shape == "Heart":
        X = (xx - cx) / (w * 0.34)
        Y = (cy - yy) / (h * 0.34)  # y up
        inside = (X ** 2 + Y ** 2 - 1) ** 3 - (X ** 2) * (Y ** 3) <= 0
    elif shape == "Cloud":
        def disc(ox, oy, rr):
            return ((xx - ox) ** 2 + (yy - oy) ** 2) <= rr ** 2
        base = ((xx - cx) / (w * 0.42)) ** 2 + ((yy - cy * 1.15) / (h * 0.22)) ** 2 <= 1.0
        bumps = (disc(cx, cy * 0.95, h * 0.26) | disc(cx - w * 0.20, cy * 1.05, h * 0.20)
                 | disc(cx + w * 0.20, cy * 1.05, h * 0.20))
        inside = base | bumps
    else:
        # Polygons via matplotlib Path (robust point-in-polygon).
        from matplotlib.path import Path as MplPath
        if shape == "Diamond":
            verts = [(cx, h * 0.04), (w * 0.96, cy), (cx, h * 0.96), (w * 0.04, cy)]
        elif shape == "Triangle":
            verts = [(cx, h * 0.05), (w * 0.95, h * 0.95), (w * 0.05, h * 0.95)]
        elif shape == "Hexagon":
            R = min(w, h) / 2.0 * 0.95
            verts = [(cx + R * np.cos(a), cy + R * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
        elif shape == "Star":
            R, r = min(w, h) / 2.0 * 0.95, min(w, h) / 2.0 * 0.42
            verts = []
            for i in range(10):
                ang = -np.pi / 2 + i * np.pi / 5
                rad = R if i % 2 == 0 else r
                verts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        else:
            return None
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(verts).contains_points(pts).reshape(h, w)

    mask = np.full((h, w), 255, dtype=np.uint8)
    if inside is not None:
        mask[inside] = 0
    return mask


def image_mask(path, w=RENDER_W, h=RENDER_H):
    """Build a mask from a user image (light pixels = excluded background)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((w, h))
    arr = np.array(img)
    mask = np.where(arr > 128, 255, 0).astype(np.uint8)
    return mask


def discover_fonts():
    """Return {display_name: font_path} for system .ttf fonts (+ Default)."""
    fonts = {"(Default)": None}
    fdir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    try:
        for f in sorted(os.listdir(fdir)):
            if f.lower().endswith(".ttf"):
                fonts[os.path.splitext(f)[0]] = os.path.join(fdir, f)
    except OSError:
        pass
    return fonts


class WordCloudStudio(QDialog):
    """3-pane Word Cloud Studio window."""

    def __init__(self, app):
        super().__init__(app)
        self._app = app                 # the ClusterApp main window (data + helpers)
        self.custom_stopwords = set()
        self._pixmap = None             # full-res QPixmap (export/copy)
        self._pil = None                # PIL image (export)
        self._wc = None                 # WordCloud object (SVG export)
        self._custom_mask_path = None
        self._fonts = discover_fonts()

        self.setWindowTitle("Word Cloud Studio")
        self.resize(1240, 820)
        # QDialogs have no maximize/minimize buttons by default — add them so the
        # window can be maximized (the preview scales on resize).
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMinimizeButtonHint
                            | Qt.WindowType.WindowMaximizeButtonHint)
        self.setStyleSheet(app.styleSheet())  # inherit terminal theme

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_controls())
        body.addWidget(self._build_preview(), 1)
        body.addWidget(self._build_word_table())
        root.addLayout(body, 1)

        self.lbl_stats = QLabel("Unique: 0  ·  Total: 0  ·  Texts: 0")
        self.lbl_stats.setStyleSheet(f"color: {COLORS['text_muted']};")
        root.addWidget(self.lbl_stats)

        self._populate_sources()
        self._on_source_changed()
        apply_dark_titlebar(self)
        install_wheel_guard(self)  # combos/spins here were covered by the old app-wide filter

    # ----- UI builders -------------------------------------------------------
    def _build_toolbar(self):
        row = QHBoxLayout()
        self.btn_generate = styled_button("▶  Generate", COLORS['accent_green'])
        self.btn_generate.clicked.connect(self.generate)
        self.btn_copy = styled_button("Copy", COLORS['text_muted'], small=True)
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        self.btn_copy.setEnabled(False)
        self.btn_save = styled_button("Save…", COLORS['text_muted'], small=True)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        row.addWidget(self.btn_generate)
        row.addWidget(self.btn_copy)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        return row

    def _combo(self, items):
        c = QComboBox()
        c.addItems(items)
        c.setMinimumWidth(180)
        return c

    def _build_controls(self):
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFixedWidth(300)
        panel.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(2, 2, 8, 2)
        v.setSpacing(6)

        # Source
        v.addWidget(make_section_label("Source"))
        self.cmb_source = self._combo(["Column", "Cluster topics (all)", "Specific cluster"])
        self.cmb_source.currentTextChanged.connect(self._on_source_changed)
        v.addWidget(self.cmb_source)
        self.cmb_target = self._combo([])
        v.addWidget(self.cmb_target)

        # Appearance
        v.addWidget(make_section_label("Appearance"))
        v.addWidget(QLabel("Font"))
        self.cmb_font = self._combo(list(self._fonts.keys()))
        v.addWidget(self.cmb_font)
        v.addWidget(QLabel("Colors"))
        self.cmb_colors = self._combo(list(PALETTES.keys()) + list(GRADIENT_COLORMAPS.keys()))
        v.addWidget(self.cmb_colors)
        v.addWidget(QLabel("Background"))
        self.cmb_bg = self._combo(list(BACKGROUNDS.keys()))
        v.addWidget(self.cmb_bg)

        # Shape
        v.addWidget(make_section_label("Shape"))
        self.cmb_shape = self._combo(SHAPES)
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        v.addWidget(self.cmb_shape)
        self.lbl_shape_file = QLabel("")
        self.lbl_shape_file.setStyleSheet(f"color: {COLORS['text_muted']};")
        v.addWidget(self.lbl_shape_file)

        # Typography
        v.addWidget(make_section_label("Typography"))
        grid = QGridLayout()
        grid.addWidget(QLabel("Max words"), 0, 0)
        self.spin_max_words = QSpinBox(); self.spin_max_words.setRange(50, 500)
        self.spin_max_words.setValue(200); self.spin_max_words.setSingleStep(10)
        grid.addWidget(self.spin_max_words, 0, 1)
        grid.addWidget(QLabel("Min font"), 1, 0)
        self.spin_min_font = QSpinBox(); self.spin_min_font.setRange(4, 40)
        self.spin_min_font.setValue(10); self.spin_min_font.setSingleStep(2)
        grid.addWidget(self.spin_min_font, 1, 1)
        grid.addWidget(QLabel("Max font (0=auto)"), 2, 0)
        self.spin_max_font = QSpinBox(); self.spin_max_font.setRange(0, 500)
        self.spin_max_font.setValue(0); self.spin_max_font.setSingleStep(10)
        grid.addWidget(self.spin_max_font, 2, 1)
        v.addLayout(grid)

        # Size proportionality
        v.addWidget(make_section_label("Size ∝ Count"))
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(0, 100)
        self.slider_scale.setValue(0)
        self.lbl_scale = QLabel("0.00  (rank only)")
        self.lbl_scale.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.slider_scale.valueChanged.connect(
            lambda x: self.lbl_scale.setText(f"{x/100:.2f}  ({_scale_desc(x/100)})"))
        v.addWidget(self.slider_scale)
        v.addWidget(self.lbl_scale)

        # Stopwords
        v.addWidget(make_section_label("Stopwords"))
        self.lbl_sw = QLabel("0 custom words excluded")
        self.lbl_sw.setStyleSheet(f"color: {COLORS['text_muted']};")
        v.addWidget(self.lbl_sw)
        sw_row = QHBoxLayout()
        b_edit = styled_button("Edit", COLORS['text_muted'], small=True)
        b_edit.clicked.connect(self._edit_stopwords)
        b_clear = styled_button("Clear", COLORS['text_muted'], small=True)
        b_clear.clicked.connect(self._clear_stopwords)
        sw_row.addWidget(b_edit); sw_row.addWidget(b_clear); sw_row.addStretch(1)
        v.addLayout(sw_row)

        v.addStretch(1)
        panel.setWidget(inner)
        return panel

    def _build_preview(self):
        wrap = QFrame()
        wrap.setStyleSheet(
            f"QFrame {{ background: {COLORS['bg_input']}; border: 1px solid {COLORS['border']}; }}")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(8, 8, 8, 8)
        self.preview = QLabel("Click  ▶ Generate  to render a word cloud.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        self.preview.setMinimumSize(480, 360)
        lay.addWidget(self.preview, 1)
        return wrap

    def _build_word_table(self):
        panel = QFrame()
        panel.setFixedWidth(250)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(make_section_label("Word Counts"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("search words…")
        self.search.textChanged.connect(self._filter_table)
        lay.addWidget(self.search)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Word", "Count", "%"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 55)
        self.table.setColumnWidth(2, 45)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        lay.addWidget(self.table, 1)
        hint = QLabel("right-click → add to stopwords")
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        lay.addWidget(hint)
        return panel

    # ----- source handling ---------------------------------------------------
    def _populate_sources(self):
        df = getattr(self._app, "df", None)
        cluster_data = getattr(self._app, "cluster_data", None)
        # Accept object/string dtypes ("object", "string", "str", "string[pyarrow]", …).
        self._columns = []
        if df is not None:
            for c in df.columns:
                dt = str(df[c].dtype)
                if dt == "object" or dt.startswith("str"):
                    self._columns.append(c)
        self._clusters = []
        if cluster_data:
            for cid in sorted(k for k in cluster_data.keys() if k != -1):
                sub = cluster_data[cid].get("subcategory", f"Topic {cid}")
                self._clusters.append((cid, f"Cluster {cid}: {sub}"))

    def _on_source_changed(self, *_):
        src = self.cmb_source.currentText()
        self.cmb_target.clear()
        if src == "Column":
            self.cmb_target.setVisible(True)
            self.cmb_target.addItems(self._columns or ["-- no text columns --"])
        elif src == "Specific cluster":
            self.cmb_target.setVisible(True)
            self.cmb_target.addItems([label for _, label in self._clusters] or ["-- run clustering --"])
        else:  # Cluster topics (all)
            self.cmb_target.setVisible(False)

    def _on_shape_changed(self, text):
        if text == "Custom Image…":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select mask image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if path:
                self._custom_mask_path = path
                self.lbl_shape_file.setText(os.path.basename(path))
            else:
                self.cmb_shape.setCurrentText("Rectangle")
        else:
            self._custom_mask_path = None
            self.lbl_shape_file.setText("")

    # ----- frequencies -------------------------------------------------------
    def _combined_stopwords(self):
        from wordcloud import STOPWORDS
        sw = set(STOPWORDS)
        try:
            from src.stopwords import get_stopwords
            # get_stopwords() reads use_english_stopwords / use_it_stopwords /
            # custom_stopwords, which live in the "stopwords" *section*. Passing the whole
            # config meant none of those keys were found and every one silently fell back
            # to its default — so turning IT stopwords off never affected the word cloud.
            sw |= set(get_stopwords(
                getattr(self._app, "config", {}).get("stopwords", {})))
        except Exception:
            pass
        sw |= {s.lower() for s in self.custom_stopwords}
        return sw

    def _frequencies(self):
        """Return (frequencies dict, is_count bool, n_texts int) for the current source."""
        src = self.cmb_source.currentText()
        sw = self._combined_stopwords()
        if src == "Column":
            df = getattr(self._app, "df", None)
            if df is None:
                raise ValueError("Load data first.")
            col = self.cmb_target.currentText()
            if not col or col.startswith("--") or col not in df.columns:
                raise ValueError("Select a valid text column.")
            series = df[col].dropna().astype(str)
            text = " ".join(series.tolist())
            tokens = [t for t in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()) if t not in sw]
            return dict(Counter(tokens)), True, len(series)
        # cluster topics
        app = self._app
        if not getattr(app, "clusterer", None) or not getattr(app.clusterer, "topic_model", None):
            raise ValueError("Run clustering first — cluster topics are not available.")
        topic_words = app._get_topic_word_weights()
        if not topic_words:
            raise ValueError("No topics found to visualize.")
        if src == "Specific cluster":
            label = self.cmb_target.currentText()
            cid = next((c for c, lbl in self._clusters if lbl == label), None)
            freqs = dict(topic_words.get(cid, {}))
            return freqs, False, 1
        merged = {}
        for ww in topic_words.values():
            for word, weight in ww.items():
                merged[word] = merged.get(word, 0) + weight
        return merged, False, len(topic_words)

    # ----- generation --------------------------------------------------------
    def generate(self):
        try:
            freqs, is_count, n_texts = self._frequencies()
        except ValueError as e:
            QMessageBox.warning(self, "Word Cloud", str(e))
            return
        if not freqs:
            QMessageBox.information(self, "Word Cloud", "No words to display.")
            return

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            logger.info(f"Generating word cloud ({len(freqs)} unique words)…")
            from wordcloud import WordCloud

            # Colors
            scheme = self.cmb_colors.currentText()
            colormap, color_func = None, None
            if scheme in GRADIENT_COLORMAPS:
                colormap = GRADIENT_COLORMAPS[scheme]
            else:
                color_func = _palette_color_func(PALETTES[scheme])

            # Mask
            shape = self.cmb_shape.currentText()
            if shape == "Custom Image…" and self._custom_mask_path:
                mask = image_mask(self._custom_mask_path)
            else:
                mask = shape_mask(shape)

            bg = BACKGROUNDS[self.cmb_bg.currentText()]
            max_font = self.spin_max_font.value() or None
            font_path = self._fonts.get(self.cmb_font.currentText())

            wc = WordCloud(
                width=RENDER_W, height=RENDER_H,
                max_words=self.spin_max_words.value(),
                colormap=colormap,
                background_color=bg,
                mask=mask,
                font_path=font_path,
                min_font_size=self.spin_min_font.value(),
                max_font_size=max_font,
                stopwords=set(),  # already filtered into freqs
                mode="RGBA" if bg is None else "RGB",
                prefer_horizontal=0.9,
                relative_scaling=self.slider_scale.value() / 100.0,
            ).generate_from_frequencies(freqs)
            if color_func is not None:
                wc.recolor(color_func=color_func)

            self._wc = wc
            self._pil = wc.to_image()
            buf = io.BytesIO()
            self._pil.save(buf, format="PNG")
            buf.seek(0)
            self._pixmap = QPixmap()
            self._pixmap.loadFromData(buf.read())
            self._update_preview()

            self._fill_table(freqs, is_count)
            total = sum(freqs.values())
            self.lbl_stats.setText(
                f"Unique: {len(freqs):,}  ·  Total: {total:,.0f}  ·  Texts: {n_texts:,}")
            self.btn_copy.setEnabled(True)
            self.btn_save.setEnabled(True)
            logger.info("Word cloud ready.")
        except Exception as e:
            logger.error(f"Word cloud generation failed: {e}")
            QMessageBox.critical(self, "Word Cloud", f"Generation failed:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _update_preview(self):
        if self._pixmap is None:
            return
        target = self.preview.size()
        self.preview.setPixmap(self._pixmap.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    # ----- word table --------------------------------------------------------
    def _fill_table(self, freqs, is_count):
        items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:150]
        total = sum(freqs.values()) or 1
        self.table.setRowCount(len(items))
        for r, (word, val) in enumerate(items):
            self.table.setItem(r, 0, QTableWidgetItem(word))
            cnt = f"{int(val)}" if is_count else f"{val:.3g}"
            it1 = QTableWidgetItem(cnt); it1.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 1, it1)
            it2 = QTableWidgetItem(f"{val/total*100:.1f}"); it2.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, it2)
        self._filter_table(self.search.text())

    def _filter_table(self, text):
        text = (text or "").lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            self.table.setRowHidden(r, bool(text) and (item is None or text not in item.text().lower()))

    def _table_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
        word = self.table.item(item.row(), 0).text()
        menu = QMenu(self)
        act = menu.addAction(f'Add "{word}" to stopwords')
        if menu.exec(self.table.mapToGlobal(pos)) == act:
            self.custom_stopwords.add(word)
            self._refresh_sw_label()
            self.generate()

    # ----- stopwords ---------------------------------------------------------
    def _refresh_sw_label(self):
        n = len(self.custom_stopwords)
        self.lbl_sw.setText(f"{n} custom word{'s' if n != 1 else ''} excluded")

    def _edit_stopwords(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Stopwords")
        dlg.resize(360, 420)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Custom stopwords (one per line):"))
        editor = QPlainTextEdit()
        editor.setPlainText("\n".join(sorted(self.custom_stopwords)))
        lay.addWidget(editor, 1)
        lay.addWidget(QLabel("Default English stopwords are always excluded."))
        row = QHBoxLayout(); row.addStretch(1)
        ok = styled_button("Save", COLORS['accent'])
        ok.clicked.connect(dlg.accept)
        row.addWidget(ok)
        lay.addLayout(row)
        if dlg.exec():
            words = re.findall(r"[a-zA-Z0-9']+", editor.toPlainText().lower())
            self.custom_stopwords = set(words)
            self._refresh_sw_label()

    def _clear_stopwords(self):
        self.custom_stopwords.clear()
        self._refresh_sw_label()

    # ----- export ------------------------------------------------------------
    def copy_to_clipboard(self):
        if self._pixmap is not None:
            QApplication.clipboard().setPixmap(self._pixmap)
            QMessageBox.information(self, "Copied", "Word cloud copied to clipboard.")

    def save_image(self):
        if self._pil is None:
            return
        src = self.cmb_source.currentText().split()[0].lower()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Word Cloud", f"wordcloud_{src}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Vector (*.svg)")
        if not path:
            return
        try:
            low = path.lower()
            if low.endswith(".svg"):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(RENDER_W / 100, RENDER_H / 100))
                ax.imshow(self._wc.to_array(), interpolation="bilinear")
                ax.axis("off")
                fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            else:
                img = self._pil
                if low.endswith((".jpg", ".jpeg")) and img.mode == "RGBA":
                    from PIL import Image
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                img.save(path, dpi=(300, 300))
            QMessageBox.information(self, "Saved", f"Word cloud saved to:\n{path}")
        except Exception as e:
            logger.error(f"Save word cloud failed: {e}")
            QMessageBox.critical(self, "Save failed", str(e))
