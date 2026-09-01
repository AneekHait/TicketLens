"""Every url() in the stylesheet must point at a file that exists.

Qt does not complain about a stylesheet `url()` it cannot open. It applies the rest
of the rule and draws nothing where the image should be, so a wrong asset path
produces a checkbox with no tick and a combo box with no caret -- visible only by
looking at the running app, with no log line and no exception.

That is not hypothetical: `_ASSETS` was `dirname(dirname(__file__))` back when it
lived in src/gui.py, and moving it to src/ui/theme.py put it one directory deeper,
so it silently resolved to a nonexistent src/assets/. These tests are the cheap
check that would have caught it.
"""
import os
import re

import pytest

from src.ui import theme

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The QSS selectors that depend on these, so a failure below says what breaks.
USED_BY = {
    "_CHECK_SVG": "QCheckBox::indicator:checked (the tick)",
    "_CHEVRON_SVG": "QComboBox::down-arrow / QSpinBox::down-button (the caret)",
    "_CHEVRON_UP_SVG": "QSpinBox::up-button (the caret)",
}


def test_the_assets_directory_resolves():
    assert os.path.isdir(theme._ASSETS), (
        f"_ASSETS points at {theme._ASSETS}, which does not exist. It must resolve "
        f"to {os.path.join(REPO, 'assets')} -- check the number of dirname() calls "
        f"against how deep this module sits.")


@pytest.mark.parametrize("name", sorted(USED_BY))
def test_each_icon_file_exists(name):
    path = getattr(theme, name)
    assert os.path.isfile(path), f"{name} -> {path} is missing; breaks {USED_BY[name]}"


def test_the_assets_dir_is_the_one_at_the_repo_root():
    """Not just *a* directory named assets -- the tracked one that build_dist.ps1
    ships and asserts on."""
    assert os.path.normcase(os.path.normpath(theme._ASSETS)) == \
        os.path.normcase(os.path.normpath(os.path.join(REPO, "assets")))


def test_every_url_in_the_stylesheet_resolves():
    """Catches an icon added to the QSS without being added to assets/, which the
    per-name tests above would miss."""
    urls = re.findall(r'url\("([^"]+)"\)', theme.STYLESHEET)
    assert urls, "no url() found in STYLESHEET - did the quoting change?"
    missing = [u for u in urls if not os.path.isfile(u)]
    assert not missing, f"stylesheet references files that do not exist: {missing}"


def test_the_paths_use_forward_slashes():
    """QSS treats a backslash as an escape, so a Windows path must be normalised
    before it goes into url()."""
    for name in USED_BY:
        assert "\\" not in getattr(theme, name), f"{name} still has a backslash"


def test_the_icons_are_actually_svg():
    """A zero-byte or HTML-error-page placeholder would pass an existence check and
    still render nothing."""
    for name in USED_BY:
        with open(getattr(theme, name), encoding="utf-8") as f:
            head = f.read(400)
        assert "<svg" in head, f"{name} does not look like an SVG"
