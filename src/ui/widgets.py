"""Composite widgets: the sidebar, the activity-log console, and the numeric-sort
tree item.

These are stateful widgets rather than the one-call factories in `theme`, but they
are still presentation-only -- the controller wires their signals up.
"""
import re
import html

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QPlainTextEdit,
    QTreeWidgetItem,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor

from src.ui.theme import COLORS, MONO_FAMILY, make_separator


# ---------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------
class SidebarButton(QPushButton):
    """A navigation button for the sidebar."""

    def __init__(self, text, icon_text="", page_index=0, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self.setText(f"  {icon_text}   {text}")
        self.setCheckable(True)
        self.setMinimumHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 2px;
                text-align: left;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: {COLORS['sidebar_hover']};
                color: {COLORS['text']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['sidebar_active']};
                color: white;
                font-weight: bold;
                border-left: 3px solid {COLORS['accent']};
            }}
        """)


class Sidebar(QFrame):
    """Left sidebar with navigation buttons."""
    page_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['sidebar']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # App title
        title = QLabel("  TicketLens")
        title.setFont(QFont(MONO_FAMILY, 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent']}; border: none; padding: 8px 0;")
        title.setMinimumHeight(50)
        layout.addWidget(title)

        subtitle = QLabel("  Local AI · offline")
        subtitle.setFont(QFont(MONO_FAMILY, 9))
        subtitle.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Navigation buttons
        self.buttons = []
        nav_items = [
            ("Import Data", "\U0001F4C2", 0),
            ("Settings", "\u2699\uFE0F", 1),
            ("Clustering", "\U0001F680", 2),
            ("Analysis", "\U0001F4CA", 3),
        ]

        for text, icon, idx in nav_items:
            btn = SidebarButton(text, icon, idx, self)
            btn.clicked.connect(lambda checked, i=idx: self._on_nav_clicked(i))
            layout.addWidget(btn)
            self.buttons.append(btn)

        layout.addStretch()

        # Utility section
        layout.addWidget(make_separator())
        layout.addSpacing(8)

        self.btn_logs = SidebarButton("View Logs", "\U0001F4DD", -1, self)
        self.btn_logs.setCheckable(False)
        self.btn_logs.setStyleSheet(self.btn_logs.styleSheet().replace("font-size: 14px", "font-size: 12px"))
        layout.addWidget(self.btn_logs)

        self.btn_cache = SidebarButton("Clear Cache", "\U0001F5D1\uFE0F", -1, self)
        self.btn_cache.setCheckable(False)
        self.btn_cache.setStyleSheet(self.btn_cache.styleSheet().replace("font-size: 14px", "font-size: 12px"))
        layout.addWidget(self.btn_cache)

        # Set first button active
        self.buttons[0].setChecked(True)

    def _on_nav_clicked(self, index):
        for btn in self.buttons:
            btn.setChecked(btn.page_index == index)
        self.page_changed.emit(index)


# ---------------------------------------------------------------
# Global live activity log (docked at the bottom of the content area)
# ---------------------------------------------------------------
class ConsolePanel(QFrame):
    """Collapsible IDE-style 'Output' panel that streams the live activity log."""
    COLLAPSED_H = 30
    EXPANDED_H = 200

    _LEVEL_COLORS = {
        "DEBUG": COLORS['text_muted'],
        "INFO": COLORS['text'],
        "WARNING": COLORS['warning'],
        "ERROR": COLORS['accent_red'],
        "CRITICAL": COLORS['accent_red'],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed = False
        self.setStyleSheet(
            f"ConsolePanel {{ background: {COLORS['bg_input']}; "
            f"border-top: 1px solid {COLORS['border']}; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar: toggle + title + clear
        header = QFrame()
        header.setFixedHeight(self.COLLAPSED_H)
        header.setStyleSheet(f"background: {COLORS['sidebar']};")
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(10, 0, 8, 0)
        hrow.setSpacing(8)
        self.toggle_btn = QPushButton("▾ activity.log")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {COLORS['accent']}; "
            f"font-weight: bold; padding: 0; text-align: left; }}"
            f"QPushButton:hover {{ color: {COLORS['text']}; }}"
        )
        self.toggle_btn.clicked.connect(self.toggle)
        hrow.addWidget(self.toggle_btn)
        hrow.addStretch(1)
        self.clear_btn = QPushButton("clear")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {COLORS['text_muted']}; "
            f"padding: 0 4px; }}"
            f"QPushButton:hover {{ color: {COLORS['text']}; }}"
        )
        self.clear_btn.clicked.connect(self.clear)
        hrow.addWidget(self.clear_btn)
        outer.addWidget(header)

        # Body: the log view
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(2000)  # ring buffer; drop oldest lines
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(self.view, 1)

    _ANSI = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')

    def append_line(self, level, msg):
        color = self._LEVEL_COLORS.get(level, COLORS['text'])
        safe = html.escape(msg)
        self.view.appendHtml(f'<span style="color:{color}; white-space:pre">{safe}</span>')
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())  # autoscroll to bottom

    def append_stream(self, text):
        """Mirror raw stdout/stderr text, honouring terminal \\r (in-place line
        updates) and \\n, with ANSI escape codes stripped."""
        text = self._ANSI.sub('', text).replace('\r\n', '\n')
        if not text:
            return
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for part in re.split(r'(\r|\n)', text):
            if part == '\n':
                cursor.insertText('\n')
            elif part == '\r':
                # Carriage return: clear the current line so the next text overwrites it.
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            elif part:
                cursor.insertText(part)
        self.view.setTextCursor(cursor)
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self.view.clear()

    def set_collapsed(self, collapsed):
        self._collapsed = collapsed
        self.view.setVisible(not collapsed)
        self.toggle_btn.setText(("▸ " if collapsed else "▾ ") + "activity.log")
        if collapsed:
            self.setMaximumHeight(self.COLLAPSED_H)
        else:
            self.setMaximumHeight(16777215)
            self.setMinimumHeight(self.COLLAPSED_H + 60)

    def toggle(self):
        self.set_collapsed(not self._collapsed)


class _PivotTreeItem(QTreeWidgetItem):
    """Tree item that sorts columns by a number stashed in UserRole instead of
    lexicographically. Used by the Category Pivot tab and the Min Cluster Size
    picker; any column with a UserRole value sorts numerically.

    Note: the text fallback compares self.text() directly rather than calling
    super().__lt__() — in PySide6 the latter re-enters this Python override and
    recurses infinitely.
    """

    def __lt__(self, other):
        tw = self.treeWidget()
        col = tw.sortColumn() if tw else 0
        a = self.data(col, Qt.ItemDataRole.UserRole)
        b = other.data(col, Qt.ItemDataRole.UserRole)
        if a is not None and b is not None:
            return a < b
        return self.text(col) < other.text(col)
