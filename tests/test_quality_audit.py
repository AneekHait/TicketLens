"""Tests for src/quality_audit.py — previously zero coverage (268 LOC).

Pure pandas, no LLM, so every scoring dimension is directly assertable.
"""
import pandas as pd
import pytest

from src.quality_audit import TicketQualityAuditor


def _audit(df, mapping=None, **settings):
    return TicketQualityAuditor(df, mapping or {}, settings=settings or None)


# --- resolution quality -----------------------------------------------------
def test_resolution_quality_returns_none_without_a_mapped_column():
    df = pd.DataFrame({"x": ["a", "b"]})
    assert _audit(df).audit_resolution_quality() == [None, None]


def test_resolution_quality_scores_empty_as_zero():
    df = pd.DataFrame({"close_notes": ["", None, "nan"]})
    scores = _audit(df, {"resolution_notes_col": "close_notes"}).audit_resolution_quality()
    assert scores == [0.0, 0.0, 0.0]


def test_resolution_quality_penalises_generic_notes():
    df = pd.DataFrame({"close_notes": ["Resolved.", "n/a", "ticket closed"]})
    scores = _audit(df, {"resolution_notes_col": "close_notes"}).audit_resolution_quality()
    # Presence (30) + a little length, but the "not generic" 30 is withheld.
    assert all(s < 70 for s in scores), scores


def test_resolution_quality_rewards_a_detailed_note():
    detailed = " ".join(["step"] * 25)      # >= 20 words earns full length marks
    df = pd.DataFrame({"close_notes": [detailed]})
    (score,) = _audit(df, {"resolution_notes_col": "close_notes"}).audit_resolution_quality()
    assert score == 100.0


def test_resolution_quality_is_bounded_0_to_100():
    df = pd.DataFrame({"close_notes": ["x", " ".join(["w"] * 500), "Done."]})
    scores = _audit(df, {"resolution_notes_col": "close_notes"}).audit_resolution_quality()
    assert all(0.0 <= s <= 100.0 for s in scores), scores


# --- categorization ---------------------------------------------------------
def test_categorization_needs_both_a_mapped_col_and_an_assigned_col():
    # Mapped source column present but no assigned-category column to compare against.
    df = pd.DataFrame({"incident category": ["Network"]})
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [None]


def test_categorization_reads_the_column_the_pipeline_actually_writes():
    """Regression: this used to look only for a bare "Category", which the pipeline
    never creates (it writes "Repetitive Category"), so Categorization_Score was
    silently blank for every row on every real run."""
    df = pd.DataFrame({
        "incident category": ["Network", "Access"],
        "Repetitive Category": ["Network", "Something Else"],
    })
    scores = _audit(df, {"category_col": "incident category"}).audit_categorization()
    assert scores[0] == 100.0, "exact match against Repetitive Category should score 100"
    assert scores[1] is not None, "must be scored, not skipped"


def test_categorization_still_accepts_a_bare_Category_column():
    """Externally-prepared sheets may already have a plain "Category" column."""
    df = pd.DataFrame({"incident category": ["Network"], "Category": ["Network"]})
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [100.0]


def test_repetitive_category_wins_when_both_columns_exist():
    df = pd.DataFrame({
        "incident category": ["Network"],
        "Repetitive Category": ["Network"],      # matches -> 100
        "Category": ["Totally Different"],       # would score ~0 if preferred
    })
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [100.0]


def test_generate_report_populates_categorization_score(tickets_df):
    """End-to-end guard for the same bug: the column must no longer be all-blank
    when the frame carries Repetitive Category and a mapped source category."""
    df = tickets_df.copy()
    df["incident category"] = df["Repetitive Category"]      # a perfect match
    out, summary = TicketQualityAuditor(
        df, {"category_col": "incident category", "resolution_notes_col": "close_notes"},
    ).generate_report(text_columns=["short_description"])

    assert "Categorization_Score" in out.columns
    scored = out["Categorization_Score"].dropna()
    assert len(scored) == len(df), "every row should now be scored"
    assert summary.get("categorization_avg") not in (None, "")


def test_categorization_exact_match_scores_100():
    df = pd.DataFrame({"incident category": ["Network"], "Category": ["Network"]})
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [100.0]


def test_categorization_substring_match_scores_75():
    df = pd.DataFrame({"incident category": ["Network"],
                       "Category": ["Network & Connectivity"]})
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [75.0]


def test_categorization_blank_original_is_unscored():
    df = pd.DataFrame({"incident category": [None], "Category": ["Network"]})
    assert _audit(df, {"category_col": "incident category"}).audit_categorization() == [None]


def test_categorization_disjoint_values_score_low():
    df = pd.DataFrame({"incident category": ["Hardware"], "Category": ["Email"]})
    (score,) = _audit(df, {"category_col": "incident category"}).audit_categorization()
    assert score == 0.0


# --- completeness -----------------------------------------------------------
def test_completeness_is_none_when_nothing_is_checkable():
    """Used to return a flat 50.0, which read as a real measurement and was
    indistinguishable from a genuinely mediocre dataset. Reachable by loading a file and
    going straight to the audit: selected_text_cols is empty until a clustering run, so
    with no metadata mapped there is nothing to score at all.

    None instead lets the composite renormalise it away and the UI show "N/A".
    """
    df = pd.DataFrame({"unmapped": ["a", "b"]})
    assert _audit(df).audit_completeness() == [None, None]


def test_completeness_rewards_populated_metadata():
    df = pd.DataFrame({
        "priority": ["High", None],
        "assignment_group": ["Svc Desk", None],
    })
    mapping = {"priority_col": "priority", "assignment_group_col": "assignment_group"}
    full, empty = _audit(df, mapping).audit_completeness()
    assert full > empty, (full, empty)


def test_completeness_penalises_a_vague_description():
    df = pd.DataFrame({"short_description": [
        "Server 12 refuses TLS handshake after the certificate rotation",
        "broken please fix asap",
    ]})
    good, vague = _audit(df, {}).audit_completeness(text_columns=["short_description"])
    assert good > vague, (good, vague)


# --- generate_report --------------------------------------------------------
def test_generate_report_adds_score_columns_without_dropping_rows(tickets_df):
    mapping = {"resolution_notes_col": "close_notes", "priority_col": "priority"}
    auditor = TicketQualityAuditor(tickets_df, mapping)
    out, summary = auditor.generate_report(text_columns=["short_description"])

    assert len(out) == len(tickets_df), "row count must be preserved"
    added = set(out.columns) - set(tickets_df.columns)
    assert added, "expected new score columns"
    assert "overall_score" in summary
    assert 0 <= summary["overall_score"] <= 100


def test_generate_report_does_not_mutate_the_caller_frame(tickets_df):
    before = list(tickets_df.columns)
    TicketQualityAuditor(tickets_df, {"resolution_notes_col": "close_notes"}) \
        .generate_report(text_columns=["short_description"])
    assert list(tickets_df.columns) == before, "auditor must work on a copy"


def test_generate_report_survives_an_empty_frame():
    empty = pd.DataFrame({"short_description": [], "close_notes": []})
    out, summary = TicketQualityAuditor(empty, {"resolution_notes_col": "close_notes"}) \
        .generate_report(text_columns=["short_description"])
    assert len(out) == 0
    assert isinstance(summary, dict)


def test_generate_report_reports_per_cluster_when_clusters_exist(tickets_df):
    auditor = TicketQualityAuditor(
        tickets_df, {"resolution_notes_col": "close_notes"},
        cluster_results={0: "Password Reset", 1: "VPN Drops"},
    )
    _out, summary = auditor.generate_report(text_columns=["short_description"])
    clusters = summary.get("clusters")
    if clusters:          # only populated when Cluster_ID is present
        assert all("count" in info for info in clusters.values())
        assert sum(i["count"] for i in clusters.values()) <= len(tickets_df)


@pytest.mark.parametrize("min_len", [5, 50])
def test_min_description_length_setting_is_honoured(min_len):
    df = pd.DataFrame({"short_description": ["a short one"]})
    a = TicketQualityAuditor(df, {}, settings={"min_description_length": min_len})
    assert a.min_desc_len == min_len
    scores = a.audit_completeness(text_columns=["short_description"])
    assert len(scores) == 1
