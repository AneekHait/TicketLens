"""Theme primitives: the palette, the global stylesheet, and the small widget
factories every page and dialog builds on.

Split out of gui.py so the UI modules (and wordcloud_studio) can share these
without importing the controller. Nothing here knows about TicketLens's data or
config -- it is pure presentation, which is what makes it safe to import from
anywhere.
"""
import os
import sys

from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QScrollArea, QAbstractSpinBox, QSlider, QTreeWidget, QHeaderView,
)
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QFont


# ---------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------
# Terminal / console palette — near-black base with muted phosphor accents.
# All keys are kept (≈140 call sites read them); only the values changed.
COLORS = {
    "bg": "#0a0e14",          # editor base
    "bg_secondary": "#0d1117",  # gutter / scrollbar trough
    "bg_card": "#11161f",       # panels / cards / groupboxes
    "bg_input": "#0d1117",      # inputs / console body
    "sidebar": "#080b10",       # darkest column
    "sidebar_active": "#13202a",  # active nav row (green-tinted)
    "sidebar_hover": "#0f1620",   # hover row
    "accent": "#3fb950",        # primary phosphor green (was blue)
    "accent_green": "#3fb950",  # run / success green
    "accent_orange": "#e3b341", # amber
    "accent_red": "#f85149",    # error red
    "accent_purple": "#a371f7", # secondary (fishbone / wordcloud / embedding)
    "text": "#c9d1d9",          # primary text
    "text_secondary": "#8b949e",  # secondary
    "text_muted": "#586069",      # muted
    "border": "#21323b",          # green-tinted slate border
    "success": "#2ea043",         # results-card border
    "warning": "#e3b341",         # amber (matches accent_orange)
}

# Monospace everywhere for the terminal aesthetic.
MONO = "'Cascadia Mono', 'Consolas', 'Courier New', monospace"   # CSS stack (stylesheet)
MONO_FAMILY = "Cascadia Mono"                                     # single family (QFont)

# Bundled SVG icons for QSS (combo caret + checkbox tick). Forward slashes for QSS;
# the project path can contain spaces, so the url() is quoted in the stylesheet.
#
# assets/ is at the repo root, i.e. two directories up from src/ui/theme.py. Qt
# silently ignores a url() it cannot open -- it does not warn or raise, it just
# renders a tickless checkbox and a caret-less combo. That is exactly what happened
# when this line moved here from src/gui.py one level shallower and kept counting
# two dirnames. tests/test_theme_assets.py now asserts every path resolves, so the
# next move fails a test instead of quietly un-styling the app.
_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets").replace("\\", "/")
_CHEVRON_SVG = f"{_ASSETS}/chevron-down.svg"
_CHEVRON_UP_SVG = f"{_ASSETS}/chevron-up.svg"
_CHECK_SVG = f"{_ASSETS}/check.svg"

# App identity (shown in the Help → About dialog). src/__init__.py is the single
# source of truth for the version — run.bat and build_dist.ps1 both read it too.
# The fallback is deliberately not a number: a stale hardcoded version silently
# misreports the build, which is worse than admitting we couldn't determine it.
try:
    from src import __version__ as APP_VERSION
except Exception:
    APP_VERSION = "unknown"
APP_NAME = "TicketLens"
APP_TAGLINE = "Local AI Ticket Analytics"
APP_LICENSE = "Apache-2.0"
AUTHOR = "Aneek Hait"
PROJECT_URL = "https://github.com/AneekHait/TicketLens"
AUTHOR_URL = "https://aneekhait.github.io"
ISSUES_URL = f"{PROJECT_URL}/issues"

STYLESHEET = f"""
    QMainWindow {{
        background-color: {COLORS['bg']};
    }}
    QWidget {{
        color: {COLORS['text']};
        font-family: {MONO};
        font-size: 13px;
    }}
    QLabel {{
        color: {COLORS['text']};
        background: transparent;
    }}
    QPushButton {{
        background-color: {COLORS['accent']};
        color: #0a0e14;
        border: 1px solid {COLORS['accent']};
        border-radius: 2px;
        padding: 8px 18px;
        font-weight: bold;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: #4fd964;
        border-color: #4fd964;
    }}
    QPushButton:pressed {{
        background-color: {COLORS['success']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_muted']};
        border: 1px solid {COLORS['border']};
    }}
    QComboBox {{
        background-color: {COLORS['bg_input']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        padding: 6px 10px;
        min-height: 28px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: url("{_CHEVRON_SVG}");
        width: 12px;
        height: 12px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        selection-background-color: {COLORS['accent']};
        selection-color: #0a0e14;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['bg_input']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        padding: 6px 10px;
        min-height: 26px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {COLORS['accent']};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid {COLORS['border']};
        background: {COLORS['bg_secondary']};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid {COLORS['border']};
        background: {COLORS['bg_secondary']};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {COLORS['sidebar_active']};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{_CHEVRON_UP_SVG}");
        width: 9px;
        height: 9px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{_CHEVRON_SVG}");
        width: 9px;
        height: 9px;
    }}
    QCheckBox {{
        spacing: 8px;
        color: {COLORS['text']};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 2px;
        border: 1px solid {COLORS['border']};
        background-color: {COLORS['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLORS['accent']};
        border-color: {COLORS['accent']};
        image: url("{_CHECK_SVG}");
    }}
    QCheckBox#colCheck {{
        padding: 4px 8px;
        border-radius: 4px;
    }}
    QCheckBox#colCheck:hover {{
        background-color: {COLORS['sidebar_hover']};
    }}
    QTextEdit {{
        background-color: {COLORS['bg_input']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        padding: 8px;
        font-family: {MONO};
        font-size: 12px;
    }}
    QPlainTextEdit {{
        background-color: {COLORS['bg_input']};
        color: {COLORS['text']};
        border: none;
        font-family: {MONO};
        font-size: 12px;
    }}
    QProgressBar {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        text-align: center;
        color: {COLORS['text']};
        min-height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['accent']};
        border-radius: 1px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: {COLORS['bg_secondary']};
        width: 14px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['border']};
        border-radius: 0;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLORS['accent']};
    }}
    QScrollBar::sub-line:vertical {{
        background: {COLORS['bg_secondary']};
        height: 14px;
        subcontrol-position: top;
        subcontrol-origin: margin;
        border: none;
    }}
    QScrollBar::add-line:vertical {{
        background: {COLORS['bg_secondary']};
        height: 14px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
        border: none;
    }}
    QScrollBar::sub-line:vertical:hover, QScrollBar::add-line:vertical:hover {{
        background: {COLORS['border']};
    }}
    QScrollBar::up-arrow:vertical {{
        image: url("{_CHEVRON_UP_SVG}");
        width: 9px;
        height: 9px;
    }}
    QScrollBar::down-arrow:vertical {{
        image: url("{_CHEVRON_SVG}");
        width: 9px;
        height: 9px;
    }}
    QScrollBar:horizontal {{
        background: {COLORS['bg_secondary']};
        height: 8px;
        border-radius: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {COLORS['border']};
        border-radius: 0;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {COLORS['accent']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QGroupBox {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        font-weight: bold;
        font-size: 13px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 12px;
        color: {COLORS['accent']};
    }}
    QSplitter::handle {{
        background-color: {COLORS['border']};
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    QFrame#separator {{
        background-color: {COLORS['border']};
        max-height: 1px;
    }}
    QMenuBar {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text']};
        border-bottom: 1px solid {COLORS['border']};
        padding: 2px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 12px;
        border-radius: 2px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLORS['sidebar_active']};
        color: {COLORS['accent']};
    }}
    QMenu {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 16px;
        border-radius: 2px;
    }}
    QMenu::item:selected {{
        background-color: {COLORS['sidebar_active']};
        color: {COLORS['accent']};
    }}
    QMenu::separator {{
        height: 1px;
        background: {COLORS['border']};
        margin: 4px 8px;
    }}
"""


# ---------------------------------------------------------------
# Wheel guard — stop the mouse wheel from changing combo/spin/slider values
# ---------------------------------------------------------------
class _WheelGuard(QObject):
    """Per-widget filter: a wheel over an unfocused combo/spinbox/slider should NOT
    change its value — forward it to the enclosing scroll area so the page scrolls.
    Installed only on those widgets (NOT app-wide) to avoid per-event overhead."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            p = obj.parentWidget()
            while p is not None and not isinstance(p, QScrollArea):
                p = p.parentWidget()
            if p is not None:
                try:
                    QApplication.sendEvent(p.viewport(), event)
                except Exception:
                    pass
            return True  # swallow it for the control either way (no value change)
        return False


_wheel_guard = None


def install_wheel_guard(root):
    """Install the no-scroll wheel guard on every combo/spinbox/slider under `root`.
    Per-widget (not app-wide), so it adds no overhead to ordinary events elsewhere."""
    global _wheel_guard
    if _wheel_guard is None:
        _wheel_guard = _WheelGuard()
    for cls in (QComboBox, QAbstractSpinBox, QSlider):
        for w in root.findChildren(cls):
            w.installEventFilter(_wheel_guard)


def apply_dark_titlebar(widget):
    """Make the Windows title bar dark (DWM immersive dark mode) so it matches the
    terminal theme. No-op off Windows or if the call isn't supported."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        val = ctypes.c_int(1)
        # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 on Win10 2004+/Win11, 19 on 1809-1909.
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(hwnd), ctypes.c_int(attr),
                    ctypes.byref(val), ctypes.sizeof(val)) == 0:
                break
    except Exception:
        pass


# ---------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------
def make_card(title=None, parent=None):
    """Create a styled card frame."""
    card = QFrame(parent)
    card.setObjectName("card")
    card.setStyleSheet(f"""
        QFrame#card {{
            background-color: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 2px;
        }}
        QFrame#card > QLabel {{
            border: none;
            background: transparent;
        }}
        QFrame#card > QProgressBar {{
            border: 1px solid {COLORS['border']};
            border-radius: 2px;
        }}
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(10)
    if title:
        lbl = QLabel(title)
        lbl.setFont(QFont(MONO_FAMILY, 13, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {COLORS['accent']}; border: none; padding: 0;")
        layout.addWidget(lbl)
    return card, layout


def make_metric_card(label, value, color=COLORS['accent'], parent=None):
    """Create a compact metric display card."""
    card = QFrame(parent)
    card.setObjectName("metricCard")
    card.setMinimumSize(130, 80)
    card.setMaximumHeight(100)
    card.setStyleSheet(f"""
        QFrame#metricCard {{
            background-color: {COLORS['bg_card']};
            border: 1px solid {color};
            border-radius: 2px;
        }}
        QFrame#metricCard QLabel {{
            border: none;
            background: transparent;
        }}
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 10, 8, 10)
    layout.setSpacing(2)

    val_lbl = QLabel(str(value))
    val_lbl.setObjectName("metricValue")
    val_lbl.setFont(QFont(MONO_FAMILY, 22, QFont.Weight.Bold))
    val_lbl.setStyleSheet(f"color: {color};")
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(val_lbl)

    name_lbl = QLabel(label)
    name_lbl.setFont(QFont(MONO_FAMILY, 10))
    name_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
    name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(name_lbl)

    return card


def make_separator():
    """Create a horizontal line separator."""
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFixedHeight(1)
    return sep


def make_section_label(text):
    """Create a bold section header label with a CLI prompt cue."""
    lbl = QLabel(f"› {text}")  # › prompt glyph
    lbl.setFont(QFont(MONO_FAMILY, 12, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {COLORS['accent']}; padding: 4px 0;")
    return lbl


def make_data_tree(headers, stretch_col=0):
    """Create a themed, monospace QTreeWidget for tabular data.

    ``stretch_col`` stretches to fill; every other column sizes to its contents.
    Sorting is left to the caller (the Category Pivot toggles it around bulk
    inserts, the Min Cluster Size picker enables it once afterwards).
    """
    tree = QTreeWidget()
    tree.setColumnCount(len(headers))
    tree.setHeaderLabels(list(headers))
    tree.setUniformRowHeights(True)
    tree.setAlternatingRowColors(False)
    tree.setFont(QFont(MONO_FAMILY, 10))
    hdr = tree.header()
    for col in range(len(headers)):
        hdr.setSectionResizeMode(
            col,
            QHeaderView.ResizeMode.Stretch if col == stretch_col
            else QHeaderView.ResizeMode.ResizeToContents,
        )
    tree.setStyleSheet(f"""
        QTreeWidget {{
            background-color: {COLORS['bg_card']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 2px;
        }}
        QHeaderView::section {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
            border: none;
            padding: 6px;
            font-weight: bold;
        }}
        QTreeWidget::item {{ padding: 3px 0; }}
    """)
    return tree


def styled_button(text, color=COLORS['accent'], icon_text=None, small=False):
    """Create a styled button with optional emoji icon."""
    btn = QPushButton(f"{icon_text}  {text}" if icon_text else text)
    h = 32 if small else 40
    btn.setMinimumHeight(h)
    # Normalize #rgb -> #rrggbb so the hover/pressed 8-digit RGBA (base + "cc"/"aa")
    # is always valid (a 3-digit base like "#555" would produce an invalid "#555cc").
    base = color
    if isinstance(base, str) and base.startswith("#") and len(base) == 4:
        base = "#" + "".join(ch * 2 for ch in base[1:])
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {base};
            color: white;
            border: none;
            border-radius: 2px;
            padding: {'6px 12px' if small else '8px 18px'};
            font-weight: bold;
            font-size: {'12px' if small else '13px'};
        }}
        QPushButton:hover {{
            background-color: {base}cc;
        }}
        QPushButton:pressed {{
            background-color: {base}aa;
        }}
        QPushButton:disabled {{
            background-color: {COLORS['bg_card']};
            color: {COLORS['text_muted']};
        }}
    """)
    return btn
