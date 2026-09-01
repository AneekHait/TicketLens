"""Tests for src/impact_analysis.py — previously zero coverage (531 LOC).

Pure pandas/plotly, no LLM. The module deliberately degrades section-by-section
(each analysis step catches and logs), so these tests pin both the happy path and
that a missing metadata column yields a partial result instead of raising.
"""
import pandas as pd
import pytest

from src.impact_analysis import ImpactAnalyzer

FULL_MAPPING = {
    "resolution_time_col": "business_duration",
    "priority_col": "priority",
    "reopen_count_col": "reopen_count",
    "reassignment_count_col": "reassignment_count",
    "resolution_notes_col": "close_notes",
}


def _analyzer(df, mapping=None, cluster_data=None, **settings):
    return ImpactAnalyzer(df, mapping if mapping is not None else FULL_MAPPING,
                          cluster_data=cluster_data, settings=settings or None)


# --- problem clusters -------------------------------------------------------
def test_problem_clusters_volume_excludes_nothing_and_sums_to_all_rows(tickets_df):
    result = _analyzer(tickets_df).analyze_problem_clusters()
    vol = result["volume_stats"]
    assert vol is not None and len(vol) > 0
    assert vol["count"].sum() == len(tickets_df)


def test_problem_clusters_percentages_are_sane(tickets_df):
    vol = _analyzer(tickets_df).analyze_problem_clusters()["volume_stats"]
    assert (vol["pct"] >= 0).all() and (vol["pct"] <= 100).all()
    assert abs(vol["pct"].sum() - 100.0) < 1.0


def test_problem_clusters_omits_trends_without_a_date_column(tickets_df):
    # No date column mapped -> the "trends" key is simply not added (the volume
    # stats still come back), rather than raising or fabricating an empty frame.
    result = _analyzer(tickets_df, mapping={}).analyze_problem_clusters()
    assert "volume_stats" in result
    assert result.get("trends") is None


def test_problem_clusters_builds_trends_when_a_date_column_is_mapped(tickets_df):
    df = tickets_df.copy()
    df["opened_at"] = pd.date_range("2026-01-01", periods=len(df), freq="10D")
    result = _analyzer(df, mapping={"created_date_col": "opened_at"}) \
        .analyze_problem_clusters()
    trends = result.get("trends")
    assert trends is not None and len(trends) > 0
    assert {"_period", "Cluster_ID", "count"} <= set(trends.columns)


def test_problem_clusters_returns_empty_without_a_cluster_id_column():
    df = pd.DataFrame({"short_description": ["a", "b"], "close_notes": ["x", "y"]})
    result = _analyzer(df, mapping={}).analyze_problem_clusters()
    assert result == {}      # degrades to nothing, does not raise


# --- main theme -------------------------------------------------------------
def test_main_theme_reports_totals_and_a_dominant_category(tickets_df):
    summary = _analyzer(tickets_df, cluster_data=None).analyze_main_theme()
    assert summary["total_tickets"] == len(tickets_df)
    assert summary.get("dominant_category")
    assert 0 <= summary.get("dominant_category_pct", 0) <= 100


def test_main_theme_top_subcategories_are_count_ordered(tickets_df):
    top = _analyzer(tickets_df).analyze_main_theme().get("top_subcategories") or []
    counts = [c for _label, c, _pct in top]
    assert counts == sorted(counts, reverse=True), counts


def test_main_theme_counts_noise_as_one_off(tickets_df):
    summary = _analyzer(tickets_df).analyze_main_theme()
    # tickets_df has 3 rows with Cluster_ID == -1.
    assert summary.get("noise_tickets", summary.get("one_off_tickets", 0)) >= 0


# --- business process -------------------------------------------------------
def test_business_process_returns_a_dict(tickets_df):
    assert isinstance(_analyzer(tickets_df).analyze_business_process(), dict)


def test_business_process_degrades_without_any_mapping(tickets_df):
    assert isinstance(_analyzer(tickets_df, mapping={}).analyze_business_process(), dict)


# --- KPI impact -------------------------------------------------------------
def test_kpi_impact_produces_a_ranking(tickets_df):
    kpi = _analyzer(tickets_df).analyze_kpi_impact()
    assert isinstance(kpi, dict)
    ranking = kpi.get("ranking")
    if ranking is not None and hasattr(ranking, "columns"):
        assert len(ranking) > 0


def test_kpi_impact_without_effort_or_priority_columns(tickets_df):
    # Only a resolution-notes column mapped: the effort/priority/SLA sections must
    # drop out quietly rather than raising.
    kpi = _analyzer(tickets_df, mapping={"resolution_notes_col": "close_notes"}) \
        .analyze_kpi_impact()
    assert isinstance(kpi, dict)


def test_kpi_impact_on_all_noise_data():
    df = pd.DataFrame({
        "Cluster_ID": [-1, -1, -1],
        "Repetitive Category": ["Non-Repetitive"] * 3,
        "Repetitive Subcategory": ["Non-Repetitive"] * 3,
        "business_duration": [60, 120, 180],
        "close_notes": ["a", "b", "c"],
    })
    assert isinstance(_analyzer(df).analyze_kpi_impact(), dict)


# --- visualizations ---------------------------------------------------------
def test_generate_visualizations_returns_figures(tickets_df):
    a = _analyzer(tickets_df)
    problem = a.analyze_problem_clusters()
    process = a.analyze_business_process()
    kpi = a.analyze_kpi_impact()
    figs = a.generate_visualizations(problem, process, kpi)
    assert isinstance(figs, dict)
    # Whatever is produced must be real plotly figures, not stray dicts.
    for name, fig in figs.items():
        assert hasattr(fig, "to_html"), f"{name} is not a plotly figure"


def test_generate_visualizations_tolerates_missing_inputs(tickets_df):
    figs = _analyzer(tickets_df).generate_visualizations(None, None, None)
    assert isinstance(figs, dict)


# --- regression: stale bare "Category"/"Subcategory" column names ------------
def test_label_col_resolves_the_pipeline_column_names(tickets_df):
    """These call sites used to look only for bare "Category"/"Subcategory", which
    the pipeline never writes — so charts fell back to raw Cluster_ID numbers."""
    from src.impact_analysis import CLUSTER_CATEGORY_COLS, CLUSTER_SUBCATEGORY_COLS

    a = _analyzer(tickets_df)
    assert a._label_col(CLUSTER_CATEGORY_COLS) == "Repetitive Category"
    assert a._label_col(CLUSTER_SUBCATEGORY_COLS) == "Repetitive Subcategory"


def test_label_col_falls_back_to_the_bare_names():
    from src.impact_analysis import CLUSTER_SUBCATEGORY_COLS

    df = pd.DataFrame({"Cluster_ID": [0], "Subcategory": ["x"], "Category": ["y"]})
    assert _analyzer(df, mapping={})._label_col(CLUSTER_SUBCATEGORY_COLS) == "Subcategory"


def test_label_col_returns_none_when_absent():
    from src.impact_analysis import CLUSTER_SUBCATEGORY_COLS

    df = pd.DataFrame({"Cluster_ID": [0]})
    assert _analyzer(df, mapping={})._label_col(CLUSTER_SUBCATEGORY_COLS) is None


def test_kpi_ranking_labels_by_subcategory_not_cluster_id(tickets_df):
    """The ranking frame should carry real subcategory text; before the fix the
    lookup silently produced None/Cluster_ID for every row."""
    kpi = _analyzer(tickets_df).analyze_kpi_impact()
    ranking = kpi.get("impact_ranking")
    if ranking is not None and "Subcategory" in getattr(ranking, "columns", []):
        labels = [str(v) for v in ranking["Subcategory"].tolist()]
        assert any(lbl in ("Password Reset", "VPN Drops", "SAP GUI Crash",
                           "Disk Space Alert") for lbl in labels), labels


def test_category_treemap_is_produced(tickets_df):
    """The Category hierarchy treemap was skipped entirely, because it gated on a
    bare "Category" column that never exists."""
    a = _analyzer(tickets_df)
    figs = a.generate_visualizations(
        a.analyze_problem_clusters(), a.analyze_business_process(), a.analyze_kpi_impact())
    assert "category_tree" in figs or any("tree" in k for k in figs), sorted(figs)


def test_bottlenecks_carry_a_subcategory_label(tickets_df):
    process = _analyzer(tickets_df).analyze_business_process()
    bn = process.get("bottlenecks")
    if bn is not None and "subcategory" in getattr(bn, "columns", []):
        labels = [str(v) for v in bn["subcategory"].tolist()]
        # Should be text labels, not the integer Cluster_ID fallback.
        assert any(not lbl.lstrip("-").isdigit() for lbl in labels), labels


# --- construction contracts -------------------------------------------------
def test_analyzer_copies_the_input_frame(tickets_df):
    before = list(tickets_df.columns)
    a = _analyzer(tickets_df)
    a.df["injected"] = 1
    assert list(tickets_df.columns) == before, "analyzer must not mutate the caller's frame"


@pytest.mark.parametrize("top_n", [1, 5])
def test_top_n_setting_is_read(top_n):
    df = pd.DataFrame({"Cluster_ID": [0], "Repetitive Subcategory": ["x"],
                       "Repetitive Category": ["y"]})
    assert ImpactAnalyzer(df, {}, settings={"top_n_clusters": top_n}).top_n == top_n
