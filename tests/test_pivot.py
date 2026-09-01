"""Tests for the Category/Subcategory pivot computation and Excel export."""
import pandas as pd
import pytest

from src.pivot import (
    compute_category_pivot,
    pivot_to_rows,
    pivot_to_category_totals,
)
from src.export import export_category_pivot_to_excel


def _df():
    return pd.DataFrame({
        "Repetitive Category": ["A&A", "A&A", "A&A", "SW", "SW", "Net"],
        "Repetitive Subcategory": ["Password", "Password", "Unlock", "SAP", "SAP", "VPN"],
    })


def test_counts_and_totals():
    p = compute_category_pivot(_df())
    assert p["total_tickets"] == 6
    assert p["n_categories"] == 3
    assert p["n_subcategories"] == 4   # Password, Unlock, SAP, VPN


def test_category_order_is_count_desc():
    p = compute_category_pivot(_df())
    assert [c["category"] for c in p["categories"]] == ["A&A", "SW", "Net"]


def test_subcategory_order_is_count_desc():
    p = compute_category_pivot(_df())
    aa = p["categories"][0]
    assert [s["subcategory"] for s in aa["subcategories"]] == ["Password", "Unlock"]
    assert aa["subcategories"][0]["count"] == 2


def test_percentages_sum_to_100():
    p = compute_category_pivot(_df())
    assert abs(sum(c["pct"] for c in p["categories"]) - 100.0) < 0.5
    aa = p["categories"][0]
    assert aa["pct"] == 50.0        # 3 of 6
    assert aa["subcategories"][0]["pct"] == pytest.approx(33.3, abs=0.1)


@pytest.mark.parametrize("bad", [None, pd.DataFrame(), pd.DataFrame({"x": [1, 2]})])
def test_missing_or_empty_is_safe(bad):
    p = compute_category_pivot(bad)
    assert p["total_tickets"] == 0
    assert p["categories"] == []


def test_nan_values_become_placeholder():
    df = pd.DataFrame({
        "Repetitive Category": ["A&A", None],
        "Repetitive Subcategory": ["Password", None],
    })
    p = compute_category_pivot(df)
    cats = [c["category"] for c in p["categories"]]
    assert "(blank)" in cats and p["total_tickets"] == 2


def test_flatten_helpers():
    p = compute_category_pivot(_df())
    rows = pivot_to_rows(p)
    assert len(rows) == 4
    assert set(rows[0]) == {"Category", "Subcategory", "Count", "% of Total"}
    totals = pivot_to_category_totals(p)
    assert len(totals) == 3
    assert sum(r["Count"] for r in totals) == 6


def test_export_roundtrip(tmp_path):
    out = tmp_path / "pivot.xlsx"
    export_category_pivot_to_excel(_df(), str(out))
    assert out.exists()
    sheets = pd.read_excel(out, sheet_name=None)
    assert set(sheets) == {"Category Pivot", "Category Totals"}
    pivot_sheet = sheets["Category Pivot"]
    assert list(pivot_sheet.columns) == ["Category", "Subcategory", "Count", "% of Total"]
    assert pivot_sheet["Count"].sum() == 6
    assert sheets["Category Totals"]["Count"].sum() == 6


def test_export_empty_df_does_not_crash(tmp_path):
    out = tmp_path / "empty.xlsx"
    export_category_pivot_to_excel(pd.DataFrame({"x": [1]}), str(out))
    assert out.exists()   # writes placeholder rows rather than raising
