"""Modal dialogs.

Each one is constructed, `exec()`d and read by the controller; none of them touch
the ticket frame or the config, so they move out of gui.py cleanly.
"""
import html

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QScrollArea, QFrame, QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

from src.ui.theme import (COLORS, MONO_FAMILY, apply_dark_titlebar, make_data_tree,
                          styled_button)
from src.ui.widgets import _PivotTreeItem


# ---------------------------------------------------------------
# Help / About dialog (themed, renders an HTML body)
# ---------------------------------------------------------------
class InfoDialog(QDialog):
    """Modal dialog that renders an HTML body — used for Help / About."""
    def __init__(self, title, body_html, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setHtml(body_html)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = styled_button("Close", COLORS['accent'])
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        lay.addLayout(row)
        apply_dark_titlebar(self)


# ---------------------------------------------------------------
# Category-audit review dialog (approve / reject each reassignment)
# ---------------------------------------------------------------
class CategoryAuditReviewDialog(QDialog):
    """Modal dialog listing the LLM's proposed category reassignments, each with
    a checkbox. ``approved()`` returns the proposals the user kept ticked."""

    def __init__(self, proposals, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Category Audit")
        self.resize(720, 560)
        self._checks = []   # [(QCheckBox, proposal)]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        head = QLabel(
            f"The AI suggests {len(proposals)} category reassignment(s). "
            "Untick any you disagree with, then apply."
        )
        head.setWordWrap(True)
        head.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        lay.addWidget(head)

        # Select all / none row.
        sel_row = QHBoxLayout()
        btn_all = styled_button("Select all", COLORS['bg_card'], small=True)
        btn_none = styled_button("Select none", COLORS['bg_card'], small=True)
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch(1)
        lay.addLayout(sel_row)

        # Scrollable list of proposals.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(6)
        for p in proposals:
            card = QFrame()
            card.setStyleSheet(
                f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; "
                f"border-radius: 6px;"
            )
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(10, 8, 10, 8)
            card_lay.setSpacing(2)
            size_txt = f"  ·  {p['size']} tickets" if p.get("size") else ""
            cb = QCheckBox(f"{p['subcategory']}{size_txt}")
            cb.setChecked(True)
            cb.setStyleSheet("border: none;")
            card_lay.addWidget(cb)
            # Show whether the suggestion is a deterministic rule or the AI model.
            if p.get("source") == "rule":
                tag = f"<span style='color:{COLORS['accent']};'>[rule]</span> "
            else:
                tag = f"<span style='color:{COLORS['text_muted']};'>[AI]</span> "
            move = QLabel(
                f"{tag}"
                f"<span style='color:{COLORS['text_muted']};'>{html.escape(p['current_category'])}</span>"
                f"  →  <span style='color:{COLORS['accent_green']};'>{html.escape(p['proposed_category'])}</span>"
            )
            move.setStyleSheet("border: none;")
            card_lay.addWidget(move)
            inner_lay.addWidget(card)
            self._checks.append((cb, p))
        inner_lay.addStretch(1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        # Action buttons.
        row = QHBoxLayout()
        row.addStretch(1)
        cancel_btn = styled_button("Cancel", COLORS['text_muted'])
        cancel_btn.clicked.connect(self.reject)
        apply_btn = styled_button("Apply Selected", COLORS['accent_green'])
        apply_btn.clicked.connect(self.accept)
        row.addWidget(cancel_btn)
        row.addWidget(apply_btn)
        lay.addLayout(row)
        apply_dark_titlebar(self)

    def _set_all(self, state):
        for cb, _ in self._checks:
            cb.setChecked(state)

    def approved(self):
        """Proposals whose checkbox is still ticked."""
        return [p for cb, p in self._checks if cb.isChecked()]


# ---------------------------------------------------------------
# Min Cluster Size picker (choose from the Suggest sweep's candidates)
# ---------------------------------------------------------------
class MinClusterSizePickerDialog(QDialog):
    """Lets the user choose a Min Cluster Size from the Suggest sweep's candidate
    table rather than silently accepting the one recommended value. The sweep already
    computed every row (size → clusters, noise %, DBCV), so surfacing them costs
    nothing extra. ``selected_size()`` returns the chosen size."""

    def __init__(self, recommended, candidates, reason="", parent=None,
                 allow_abort=False):
        """allow_abort: mid-run use. Adds a "Stop the run" button and rewords the
        instructions, because mid-run the consequences of each button differ from the
        Suggest-button case: there, Cancel changes nothing; mid-run, Cancel immediately
        continues clustering with the current value. ``aborted()`` reports the choice."""
        super().__init__(parent)
        self.setWindowTitle("Min Cluster Size Suggestion")
        self.resize(620, 540)
        self._recommended = recommended
        self._aborted = False
        self._allow_abort = allow_abort

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        head = QLabel(f"Suggested Min Cluster Size: {recommended}")
        head.setFont(QFont(MONO_FAMILY, 15, QFont.Weight.Bold))
        head.setStyleSheet(f"color: {COLORS['accent']}; border: none;")
        lay.addWidget(head)

        if reason:
            why = QLabel(reason)
            why.setWordWrap(True)
            why.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
            lay.addWidget(why)

        pick = QLabel(
            # Mid-run every exit starts a full clustering run, so the wording has to
            # say so — "Cancel" alone reads as "abort", which is the opposite.
            "Clustering is paused here. The recommended size is pre-selected; pick a "
            "different row to trade off cluster count against noise, then confirm. "
            "Keep current value continues the run with the size you already set. "
            "Stop the run ends it now — the embeddings stay cached, so starting again "
            "is quick."
            if allow_abort else
            "The recommended size is pre-selected. Pick a different row to trade off "
            "cluster count against noise, then confirm — or Cancel to keep your current value."
        )
        pick.setWordWrap(True)
        pick.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        lay.addWidget(pick)

        self.tree = make_data_tree(["Size", "Clusters", "Noise %", "DBCV"])
        self.tree.setRootIsDecorated(False)     # flat table — no expand arrows

        rec_item = None
        for c in candidates:
            size = int(c["size"])
            n_clusters = int(c.get("n_clusters", 0))
            noise = c.get("noise_pct", 0)
            dbcv = c.get("dbcv")
            item = _PivotTreeItem([
                str(size), f"{n_clusters:,}", f"{noise}%",
                "—" if dbcv is None else f"{dbcv:.3f}",
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, size)
            item.setData(1, Qt.ItemDataRole.UserRole, n_clusters)
            item.setData(2, Qt.ItemDataRole.UserRole, float(noise))
            # Unscored candidates sort below every real DBCV rather than mixing in.
            item.setData(3, Qt.ItemDataRole.UserRole,
                         float(dbcv) if dbcv is not None else float("-inf"))
            for col in (1, 2, 3):
                item.setTextAlignment(col, Qt.AlignmentFlag.AlignRight)
            if size == recommended:
                f = item.font(0)
                f.setBold(True)
                for col in range(4):
                    item.setFont(col, f)
                    item.setForeground(col, QColor(COLORS['accent']))
                rec_item = item
            elif not c.get("valid", True):
                # Fewer than 2 clusters, or >60% noise — a poor trade-off; mute it.
                for col in range(4):
                    item.setForeground(col, QColor(COLORS['text_muted']))
            self.tree.addTopLevelItem(item)

        # Sort after the bulk insert (see _PivotTreeItem on numeric sorting).
        self.tree.setSortingEnabled(True)
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        if rec_item is not None:
            self.tree.setCurrentItem(rec_item)
        elif self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())
        lay.addWidget(self.tree, 1)

        note = QLabel(
            "DBCV = density-based cluster validity (higher is better). Higher size → "
            "fewer, larger clusters. Greyed rows separate poorly (under 2 clusters, or "
            "over 60% noise)."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; border: none;")
        lay.addWidget(note)

        row = QHBoxLayout()
        if allow_abort:
            # Left-aligned and away from the confirm button: it ends the run, so it
            # shouldn't sit under a mis-aimed click meant for "Use Selected Size".
            abort_btn = styled_button("Stop the run", COLORS['accent_red'])
            abort_btn.clicked.connect(self._on_abort)
            row.addWidget(abort_btn)
        row.addStretch(1)
        cancel_btn = styled_button(
            "Keep current value" if allow_abort else "Cancel", COLORS['text_muted'])
        cancel_btn.clicked.connect(self.reject)
        use_btn = styled_button("Use Selected Size", COLORS['accent_green'])
        use_btn.clicked.connect(self.accept)
        row.addWidget(cancel_btn)
        row.addWidget(use_btn)
        lay.addLayout(row)
        apply_dark_titlebar(self)

    def _on_abort(self):
        """Record the abort, then close via reject() so the caller's Accepted/Rejected
        handling is unchanged — aborted() is what distinguishes this from Cancel."""
        self._aborted = True
        self.reject()

    def aborted(self):
        """True when the user chose to end the run (mid-run mode only)."""
        return self._aborted

    def selected_size(self):
        """The size on the highlighted row, falling back to the recommendation."""
        item = self.tree.currentItem()
        if item is None:
            return self._recommended
        val = item.data(0, Qt.ItemDataRole.UserRole)
        return int(val) if val is not None else self._recommended
