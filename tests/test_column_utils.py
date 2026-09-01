"""Tests for the shared column resolver used by the analysis modules."""
import pandas as pd

from src.column_utils import resolve_column

DF = pd.DataFrame(columns=["Short Description", "business_duration", "Reopen Count"])


def test_explicit_mapping_wins():
    assert resolve_column(DF, {"effort_col": "business_duration"}, "effort_col") == "business_duration"


def test_mapping_ignored_when_column_absent():
    # Mapped to a non-existent column, no patterns -> None (mapping-only behaviour).
    assert resolve_column(DF, {"effort_col": "not_here"}, "effort_col") is None


def test_no_patterns_is_mapping_only():
    assert resolve_column(DF, {}, "effort_col") is None


def test_exact_name_match_case_insensitive():
    assert resolve_column(DF, {}, "effort_col", "BUSINESS_DURATION") == "business_duration"


def test_substring_match_after_exact():
    # No exact "reopen", but "Reopen Count" contains it.
    assert resolve_column(DF, {}, "reopen_col", "reopen") == "Reopen Count"


def test_exact_preferred_over_substring():
    df = pd.DataFrame(columns=["duration", "business_duration"])
    # "duration" exact-matches before "business_duration" substring-matches.
    assert resolve_column(df, {}, "effort_col", "duration") == "duration"


def test_returns_none_when_nothing_matches():
    assert resolve_column(DF, {}, "x", "zzz_no_such_column") is None


# --- the cluster-label columns live in exactly one place --------------------
def test_label_columns_are_defined_only_here():
    """Duplicated copies of these constants caused the same bug twice: a call site
    looking only for the bare "Subcategory"/"Category", which the pipeline never
    writes. impact_analysis and quality_audit must import, not redefine."""
    import io
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    definers = []
    for name in sorted(os.listdir(os.path.join(root, "src"))):
        if not name.endswith(".py") or name == "column_utils.py":
            continue
        with io.open(os.path.join(root, "src", name), encoding="utf-8") as f:
            for line in f:
                if line.startswith(("CLUSTER_CATEGORY_COLS =",
                                    "CLUSTER_SUBCATEGORY_COLS =")):
                    definers.append(f"{name}: {line.strip()}")
    assert not definers, f"redefined outside column_utils: {definers}"


def test_the_modules_that_need_them_import_them():
    """Guard the other direction: if the import is dropped the constants would resolve
    to nothing and the lookups would NameError at runtime."""
    from src import impact_analysis, quality_audit
    from src.column_utils import CLUSTER_CATEGORY_COLS, CLUSTER_SUBCATEGORY_COLS

    for mod in (impact_analysis, quality_audit):
        assert mod.CLUSTER_CATEGORY_COLS is CLUSTER_CATEGORY_COLS
        assert mod.CLUSTER_SUBCATEGORY_COLS is CLUSTER_SUBCATEGORY_COLS


def test_resolve_label_column_prefers_the_pipeline_name():
    """The pipeline writes "Repetitive Subcategory"; the bare name is only a fallback
    for externally-prepared sheets. The preference order is the whole point."""
    from src.column_utils import CLUSTER_SUBCATEGORY_COLS, resolve_label_column

    both = pd.DataFrame({"Repetitive Subcategory": ["a"], "Subcategory": ["b"]})
    assert resolve_label_column(both, CLUSTER_SUBCATEGORY_COLS) == "Repetitive Subcategory"

    bare = pd.DataFrame({"Subcategory": ["b"]})
    assert resolve_label_column(bare, CLUSTER_SUBCATEGORY_COLS) == "Subcategory"

    neither = pd.DataFrame({"Cluster_ID": [0]})
    assert resolve_label_column(neither, CLUSTER_SUBCATEGORY_COLS) is None
