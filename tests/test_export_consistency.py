"""Tests for cross-workbook consistency of the exports, and for the config
containing no dead keys.

Motivation: the side exports used to label the join key "Cluster ID" (with a space)
while the main results sheet writes "Cluster_ID" (underscore), so a VLOOKUP or Power
Query between the two silently failed to match. And DEFAULTS advertised seven settings
that nothing in src/ ever read, which reads as supported configuration but does nothing.
"""
import ast
import io
import os
import re

import pandas as pd
import pytest

from src.config import DEFAULTS
from src.export import (
    export_audit_to_excel,
    export_category_pivot_to_excel,
    export_disposition_to_excel,
    export_kba_to_excel,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The name the clustering pipeline writes onto the main sheet (src/gui.py).
JOIN_KEY = "Cluster_ID"


# --- the join key is spelled the same everywhere ----------------------------
def test_no_export_uses_the_spaced_cluster_id():
    """Guards the rename: a stray "Cluster ID" breaks joins against the main sheet."""
    with io.open(os.path.join(REPO, "src", "export.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    spaced = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "Cluster ID"
    ]
    assert not spaced, 'export.py must use "Cluster_ID", not "Cluster ID"'


def test_kba_export_join_key(tmp_path):
    out = tmp_path / "kba.xlsx"
    export_kba_to_excel(
        [{"cluster_id": 3, "title": "T", "category": "c", "subcategory": "s",
          "keywords": ["k"], "symptoms": "", "cause": "", "resolution": "",
          "prevention": ""}],
        str(out),
    )
    assert JOIN_KEY in pd.read_excel(out).columns


def test_audit_cluster_breakdown_join_key(tmp_path):
    out = tmp_path / "audit.xlsx"
    df = pd.DataFrame({"Cluster_ID": [0, 0], "Quality_Score": [80.0, 90.0]})
    summary = {
        "overall_score": 85, "completeness_avg": 80, "categorization_avg": 90,
        "resolution_avg": 85, "total_tickets": 2, "tickets_scored": 2,
        "clusters": {0: {"subcategory": "Password Reset", "count": 2,
                         "avg_quality": 85.0}},
    }
    export_audit_to_excel(df, summary, str(out))
    breakdown = pd.read_excel(out, sheet_name="Cluster Breakdown")
    assert JOIN_KEY in breakdown.columns


def test_all_five_disposition_outputs_reach_the_main_sheet():
    """The analyzer produces five LLM sections; "recommendation" used to reach only
    the disposition workbook and the results card, so the main sheet lacked the one
    column saying what to actually do about each cluster."""
    with io.open(os.path.join(REPO, "src", "gui.py"), encoding="utf-8") as f:
        gui = f.read()
    for col in ("Automation Disposition", "Root Cause", "Resolution Provided",
                "Disposition Rationale", "Recommendation"):
        assert f'self.df["{col}"]' in gui, f"{col} is never written to the sheet"


def test_disposition_export_join_key(tmp_path):
    out = tmp_path / "disp.xlsx"
    export_disposition_to_excel(
        [{"cluster_id": 1, "subcategory": "VPN Drops", "disposition": "Automate",
          "count": 5, "root_cause": "rc", "resolution": "res", "rationale": "r",
          "recommendation": "do the thing", "keywords": ["vpn"]}],
        str(out),
    )
    opps = pd.read_excel(out, sheet_name="Opportunities")
    assert JOIN_KEY in opps.columns
    # Recommendation must survive into the workbook too.
    assert "Recommendation" in opps.columns
    assert opps["Recommendation"].iloc[0] == "do the thing"


def test_every_excel_export_that_has_a_cluster_key_spells_it_consistently(tmp_path):
    """One assertion covering all of them, so a new export can't reintroduce the drift."""
    kba = tmp_path / "k.xlsx"
    export_kba_to_excel([{"cluster_id": 1, "title": "t", "keywords": []}], str(kba))

    disp = tmp_path / "d.xlsx"
    export_disposition_to_excel([{"cluster_id": 1, "disposition": "Retain",
                                  "keywords": []}], str(disp))

    piv = tmp_path / "p.xlsx"
    export_category_pivot_to_excel(
        pd.DataFrame({"Repetitive Category": ["A"], "Repetitive Subcategory": ["b"]}),
        str(piv),
    )

    for path in (kba, disp, piv):
        for _sheet, frame in pd.read_excel(path, sheet_name=None).items():
            for col in frame.columns:
                assert col != "Cluster ID", f"{path.name}: spaced join key"


# --- DEFAULTS carries no dead settings --------------------------------------
DEAD_KEYS = [
    "n_macro_categories_max",      # leftover from the replaced K-Means macro grouping
    "n_macro_categories_divisor",
    "require_resolution_notes",    # was self-labelled "NOT YET WIRED"
    "include_resolution_steps",    # was self-labelled "NOT YET WIRED"
    "resolution_text_cols",        # the GUI builds res_cols from checkboxes instead
    "sla_target_hours",            # breach comes from sla_status_col, not a threshold
]


@pytest.mark.parametrize("key", DEAD_KEYS)
def test_dead_key_is_gone_from_defaults(key):
    for section, values in DEFAULTS.items():
        if isinstance(values, dict):
            assert key not in values, f"DEFAULTS['{section}']['{key}'] is unused"


def test_llm_model_path_is_gone():
    """The model path comes from the constructor arg / UI selection; _llm_settings
    only ever reads n_ctx / n_batch / temperature / max_tokens_*."""
    assert "model_path" not in DEFAULTS["llm"]


def test_still_live_settings_survived_the_cleanup():
    assert DEFAULTS["clustering"]["min_cluster_size"] == 15
    assert DEFAULTS["clustering"]["hdbscan_cluster_selection_method"] == "eom"
    assert DEFAULTS["llm"]["n_ctx"] == 2048
    assert DEFAULTS["analysis"]["top_n_clusters"] == 10
    assert DEFAULTS["disposition"]["max_sample_tickets"] == 5
    assert DEFAULTS["category_audit"]["max_keywords"] == 8


# Keys consumed via the metadata-mapping roles or the UI rather than by name,
# or whose name intentionally collides with another section's key.
UNREAD_EXEMPT = {
    "enabled",          # cache.enabled / llm.enabled are set by the GUI checkboxes
    "directory",        # cache.directory, read via resolve_cache_dir(...)
    "custom_stopwords", "custom_regex_patterns", "custom_boilerplate_patterns",
    "min_cluster_size", # disposition.min_cluster_size collides with clustering's;
                        # both sections use the same string literal in source
}


def _src_blob():
    """Every src/*.py except config.py (where DEFAULTS is declared)."""
    sources = []
    for name in os.listdir(os.path.join(REPO, "src")):
        if name.endswith(".py") and name != "config.py":
            with io.open(os.path.join(REPO, "src", name), encoding="utf-8") as f:
                sources.append(f.read())
    return "\n".join(sources)


def _unread_keys(defaults, blob, exempt=UNREAD_EXEMPT):
    """DEFAULTS leaf keys that no src/ module reads as a quoted dict access.

    A bare ``key in blob`` substring test green-lit genuinely dead keys:
    "remove_boilerplate" matched the *function* of that name in preprocessing.py, and
    disposition's "min_cluster_size" matched clustering.py's unrelated key. Requiring
    the quoted form excludes function names, comments and prose.
    """
    unread = []
    for section, values in defaults.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            if key in exempt or key.endswith("_col"):
                continue
            pattern = re.compile(r"""["']""" + re.escape(key) + r"""["']""")
            if not pattern.search(blob):
                unread.append(f"{section}.{key}")
    return unread


def test_every_defaults_key_is_read_somewhere_in_src():
    """Catches the next dead setting before it ships."""
    unread = _unread_keys(DEFAULTS, _src_blob())
    assert not unread, f"DEFAULTS keys nothing in src/ reads: {unread}"


def test_the_dead_key_check_actually_catches_a_dead_key():
    """The check itself was broken: it matched substrings anywhere in src/, so it
    passed for keys nothing consumed. Prove it now fires."""
    fake = {"clustering": {"a_key_no_module_will_ever_read": 1}}
    assert _unread_keys(fake, _src_blob()) == [
        "clustering.a_key_no_module_will_ever_read"
    ]


def test_the_dead_key_check_is_not_fooled_by_a_bare_substring():
    """A key whose name appears in src/ only as part of a longer identifier (a
    function name, another key) must still be reported as dead."""
    # "remove_boilerplate" is a real function name in preprocessing.py, which is
    # exactly what fooled the old substring check. As a *config key* it is read
    # from settings, so it appears quoted too — use a name that only occurs bare.
    blob = 'def some_orphan_setting_helper(): pass\n'
    assert _unread_keys(
        {"s": {"some_orphan_setting": 1}}, blob, exempt=set()
    ) == ["s.some_orphan_setting"]


def test_the_dead_key_check_accepts_a_genuinely_read_key():
    """Guard against the pattern being so strict that nothing ever matches."""
    blob = 'value = settings["a_live_setting"]\n'
    assert _unread_keys({"s": {"a_live_setting": 1}}, blob, exempt=set()) == []
