"""The src/ui/ split is only worth anything if the dependency direction holds.

gui.py went from 5,818 lines to ~3,580 by moving the pages, dialogs and theme
primitives into `src/ui/`. What makes that a real separation rather than four files
that happen to sit in a folder is the one-way import direction: gui.py imports from
src.ui, never the reverse.

The specific thing being guarded: `wordcloud_studio` used to do
`from src.gui import COLORS, styled_button, ...`, which only worked because gui.py
imports wordcloud_studio *lazily inside a method*. That is a circular import held
apart by call ordering -- it survives until someone moves the import to module
scope. Now theme.py sits at the bottom of the chain and imports no TicketLens
module at all, so the cycle cannot form.

These tests parse the source with `ast` instead of importing, so they need no
QApplication and report the offending module by name.
"""
import ast
import os

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
_UI = os.path.join(_SRC, "ui")

UI_MODULES = sorted(f for f in os.listdir(_UI) if f.endswith(".py"))


def _imports(path):
    """Every module a file imports, as top-level-qualified dotted names."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


def test_the_ui_package_is_not_empty():
    """A glob-driven test that silently passes on zero files is worse than no test."""
    assert len(UI_MODULES) >= 5, UI_MODULES
    for expected in ("__init__.py", "theme.py", "widgets.py", "dialogs.py", "pages.py"):
        assert expected in UI_MODULES


@pytest.mark.parametrize("module", UI_MODULES)
def test_nothing_in_src_ui_imports_the_controller(module):
    """The whole point of the split. gui.py owns the data and the threads; the UI
    modules are constructed *by* it and must not reach back."""
    found = _imports(os.path.join(_UI, module))
    offenders = {m for m in found if m == "src.gui" or m.startswith("src.gui.")}
    assert not offenders, f"src/ui/{module} imports the controller: {offenders}"


def test_theme_is_the_bottom_of_the_chain():
    """theme.py may import PySide6 and `src` (for __version__ in the About dialog),
    and nothing else. This is the property wordcloud_studio relies on."""
    found = _imports(os.path.join(_UI, "theme.py"))
    ticketlens = {m for m in found if m == "src" or m.startswith("src.")}
    assert ticketlens == {"src"}, (
        f"theme.py must stay dependency-free; it imports {ticketlens}")


def test_widgets_and_pages_do_not_depend_on_each_other():
    """They are siblings. A page needing a composite widget would be fine, but a
    widget reaching into pages would mean the layers are inverted."""
    assert "src.ui.pages" not in _imports(os.path.join(_UI, "widgets.py"))
    assert "src.ui.dialogs" not in _imports(os.path.join(_UI, "widgets.py"))


def test_wordcloud_studio_takes_the_theme_directly():
    """It used to import these from gui.py. If that regresses, the lazy-import
    workaround in gui.py becomes load-bearing again."""
    found = _imports(os.path.join(_SRC, "wordcloud_studio.py"))
    assert "src.ui.theme" in found
    assert "src.gui" not in found


# --- no class got left behind, or duplicated ---------------------------------
MOVED = {
    "InfoDialog": "ui/dialogs.py",
    "CategoryAuditReviewDialog": "ui/dialogs.py",
    "MinClusterSizePickerDialog": "ui/dialogs.py",
    "ImportPage": "ui/pages.py",
    "SettingsPage": "ui/pages.py",
    "ClusteringPage": "ui/pages.py",
    "AnalysisPage": "ui/pages.py",
    "Sidebar": "ui/widgets.py",
    "SidebarButton": "ui/widgets.py",
    "ConsolePanel": "ui/widgets.py",
    "_PivotTreeItem": "ui/widgets.py",
}


def _classes_in(rel):
    tree = ast.parse(open(os.path.join(_SRC, rel), encoding="utf-8").read())
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


@pytest.mark.parametrize("name,home", sorted(MOVED.items()))
def test_each_moved_class_is_defined_exactly_once(name, home):
    """A half-applied revert would leave two definitions, and the one gui.py's own
    code sees would depend on import order."""
    homes = [rel for rel in ("gui.py", "ui/theme.py", "ui/widgets.py",
                             "ui/dialogs.py", "ui/pages.py")
             if name in _classes_in(rel)]
    assert homes == [home], f"{name} is defined in {homes}, expected only {home}"


def test_the_controller_still_re_exports_what_callers_import():
    """`from src.gui import ClusteringPage` appears in the test suite and
    `from src.gui import main` in main.py. The split must not break either."""
    import src.gui as gui_mod

    for name in ("ClusterApp", "ClusteringPage", "MinClusterSizePickerDialog", "main"):
        assert hasattr(gui_mod, name), name


def test_the_controller_actually_shrank():
    """Not a style rule -- a tripwire. If gui.py creeps back past ~4,500 lines the
    extraction has been undone or bypassed, and this is the cheapest place to notice.
    Raise the bound deliberately if the controller genuinely needs to grow."""
    with open(os.path.join(_SRC, "gui.py"), encoding="utf-8") as f:
        n = sum(1 for _ in f)
    assert n < 4500, f"gui.py is {n} lines; it was 3,580 after the src/ui/ split"
