"""Guards on the distributable build.

The zip shipped every recipient the developer's own config/user_settings.json:
min_cluster_size 15->20, umap_parallel_mode auto->parallel (explicitly
not bit-for-bit reproducible), use_it_stopwords off, and a different embedding
model. None of that is visible to the recipient, who sees an app that behaves
unlike the documented defaults.

These tests read build_dist.ps1 as text rather than running it, so they work
without PowerShell and without a multi-minute real build.
"""
import io
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(REPO, "build_dist.ps1")


@pytest.fixture(scope="module")
def script():
    with io.open(BUILD_SCRIPT, encoding="utf-8") as f:
        return f.read()


def test_the_build_script_exists():
    assert os.path.isfile(BUILD_SCRIPT)


@pytest.mark.parametrize("path", [
    r"config\user_settings.json",
    r"config\custom_stopwords.txt",
])
def test_per_user_state_is_excluded_from_the_archive(script, path):
    """Both files are gitignored per-user state. Nothing in $excludeDirs or
    $excludeExt covered them, so they were packaged."""
    assert path in script, f"{path} is not excluded by build_dist.ps1"


def test_the_exclusion_is_actually_applied_to_the_file_list(script):
    """Declaring $excludeFiles is not enough — the Where-Object must consult it."""
    assert "$excludeFiles" in script
    # The filter block must reference it, not just the declaration.
    filter_region = script.split("Get-ChildItem")[1].split("if (-not $files)")[0]
    assert "$excludeFiles" in filter_region, \
        "$excludeFiles is declared but the file filter never uses it"


def test_the_archive_is_verified_to_not_contain_per_user_state(script):
    """A post-build check, so a future refactor of the filter can't silently
    reintroduce the leak."""
    assert "MUST NOT be in the archive" in script
    assert "config/user_settings.json" in script


@pytest.mark.parametrize("d", [".claude", ".vscode", ".idea"])
def test_editor_and_agent_tooling_state_is_excluded(script, d):
    """Not part of the app, and it accumulates over time. .claude/ was shipping
    in v1.2.0's first build."""
    assert d in script, f"{d} is not excluded by build_dist.ps1"


def test_gitignore_and_build_script_agree_on_per_user_state():
    """If a file is gitignored as per-user state, it must not ship either."""
    with io.open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        ignored = f.read()
    with io.open(BUILD_SCRIPT, encoding="utf-8") as f:
        build = f.read()
    for name in ("user_settings.json", "custom_stopwords.txt"):
        assert name in ignored, f"{name} should be gitignored"
        assert name in build, f"{name} is gitignored but build_dist.ps1 ships it"


def test_version_is_the_single_source_of_truth():
    """build_dist.ps1 reads __version__ from src/__init__.py, so nothing else
    may hardcode a version that could drift."""
    with io.open(os.path.join(REPO, "src", "__init__.py"), encoding="utf-8") as f:
        init = f.read()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    assert m, "src/__init__.py must define __version__"
    version = m.group(1)
    # Semantic version, so the archive name sorts and compares sensibly.
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), \
        f"expected MAJOR.MINOR.PATCH, got {version!r}"


def test_pytest_pins_a_basetemp():
    """Without --basetemp, pytest's teardown hits the pytest-current symlink under
    LOCALAPPDATA and raises PermissionError on Windows, so a fully passing suite
    still exits 1 and CI cannot read pass/fail from the exit code."""
    with io.open(os.path.join(REPO, "pytest.ini"), encoding="utf-8") as f:
        ini = f.read()
    assert "--basetemp" in ini


def test_the_pinned_basetemp_is_gitignored():
    with io.open(os.path.join(REPO, "pytest.ini"), encoding="utf-8") as f:
        ini = f.read()
    m = re.search(r"--basetemp=(\S+)", ini)
    assert m
    basetemp = m.group(1).strip().rstrip("/\\")
    with io.open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        ignored = f.read()
    assert basetemp.lstrip("./") in ignored, \
        f"{basetemp} holds test scratch files and must be gitignored"


# ---------------------------------------------------------------------------
# Open-source distribution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["LICENSE", "NOTICE", "README.md",
                                  "CONTRIBUTING.md", "SECURITY.md",
                                  "CODE_OF_CONDUCT.md"])
def test_the_open_source_paperwork_is_present(name):
    assert os.path.isfile(os.path.join(REPO, name)), f"{name} is missing from the repo"


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE", "README.md"])
def test_the_license_files_ship_in_the_archive(script, name):
    """Apache-2.0 section 4 requires the license and NOTICE to travel with any
    redistribution. build_dist.ps1 must fail, not warn, if one goes missing."""
    assert f"'{name}'" in script, f"{name} is not in build_dist.ps1's $required list"


def test_the_license_is_apache_2_and_names_a_copyright_holder():
    with io.open(os.path.join(REPO, "LICENSE"), encoding="utf-8") as f:
        text = f.read()
    assert "Apache License" in text and "Version 2.0" in text
    # The boilerplate placeholder left unfilled is a real and common mistake.
    assert "[yyyy]" not in text and "[name of copyright owner]" not in text
    assert re.search(r"Copyright \d{4}", text), "LICENSE has no filled-in copyright line"


# ---------------------------------------------------------------------------
# The Unix launcher
# ---------------------------------------------------------------------------
RUN_SH = os.path.join(REPO, "run.sh")


def test_the_unix_launcher_exists_and_ships():
    assert os.path.isfile(RUN_SH), "run.sh is missing"
    with io.open(BUILD_SCRIPT, encoding="utf-8") as f:
        assert "'run.sh'" in f.read(), "run.sh is not in build_dist.ps1's $required list"


def test_run_sh_has_no_crlf_line_endings():
    """A CRLF shebang makes Linux report 'bad interpreter: /usr/bin/env bash^M',
    which reads like a missing bash rather than a line-ending problem. Editing
    run.sh on Windows is exactly how that gets introduced, and nothing else in
    the suite would notice."""
    with io.open(RUN_SH, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw, "run.sh contains CRLF line endings; it must be LF-only"
    assert raw.startswith(b"#!/usr/bin/env bash"), "run.sh must start with a bash shebang"


def test_gitattributes_pins_shell_scripts_to_lf():
    """Belt and braces: even if someone commits a CRLF run.sh, the checkout rule
    keeps it LF for the next person."""
    with io.open(os.path.join(REPO, ".gitattributes"), encoding="utf-8") as f:
        attrs = f.read()
    assert re.search(r"^\*\.sh\s+text\s+eol=lf", attrs, re.M), \
        ".gitattributes must pin *.sh to eol=lf"


def test_run_sh_is_pure_ascii():
    """Same rationale as the .ps1 rule: a stray em dash in a launcher that runs
    before any locale is set is not worth the risk."""
    with io.open(RUN_SH, "rb") as f:
        raw = f.read()
    non_ascii = [b for b in raw if b > 127]
    assert not non_ascii, f"run.sh has {len(non_ascii)} non-ASCII byte(s)"


def test_both_launchers_report_the_same_version_source():
    """run.bat and run.sh must both read the version from src/__init__.py rather
    than hardcoding it, or the two platforms print different versions."""
    for launcher in ("run.bat", "run.sh"):
        with io.open(os.path.join(REPO, launcher), encoding="utf-8") as f:
            text = f.read()
        assert "__version__" in text and "src" in text, \
            f"{launcher} does not read the version from src/__init__.py"
