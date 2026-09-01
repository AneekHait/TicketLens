"""
Category / Subcategory pivot computation.

Pure (Qt-free, IO-free) so it can be unit-tested and reused by both the GUI
tree view and the Excel exporter. Reads the two columns the clustering pipeline
writes onto the results DataFrame: "Repetitive Category" and
"Repetitive Subcategory".
"""

CATEGORY_COL = "Repetitive Category"
SUBCATEGORY_COL = "Repetitive Subcategory"


def compute_category_pivot(df, category_col=CATEGORY_COL, subcategory_col=SUBCATEGORY_COL):
    """Build an ordered Category -> Subcategory count breakdown.

    Returns a dict:
        {
          "total_tickets": int,
          "n_categories": int,
          "n_subcategories": int,
          "categories": [
              {"category": str, "count": int, "pct": float,
               "subcategories": [{"subcategory": str, "count": int, "pct": float}, ...]},
              ...
          ],
        }
    Categories are sorted by total count desc; subcategories within each category
    by count desc. ``pct`` is percent of the grand total. Missing columns or an
    empty frame yield an all-empty result (never raises).
    """
    empty = {"total_tickets": 0, "n_categories": 0, "n_subcategories": 0, "categories": []}
    if df is None or category_col not in getattr(df, "columns", []) \
            or subcategory_col not in df.columns:
        return empty

    # Count (category, subcategory) pairs. NaNs -> a readable placeholder so they
    # still show up rather than silently vanishing.
    pairs = (
        df[[category_col, subcategory_col]]
        .fillna("(blank)")
        .astype(str)
        .groupby([category_col, subcategory_col])
        .size()
    )
    total = int(pairs.sum())
    if total == 0:
        return empty

    def pct(n):
        return round(100.0 * n / total, 1)

    # Aggregate to category level and order.
    cat_totals = {}
    cat_subs = {}
    for (cat, sub), n in pairs.items():
        n = int(n)
        cat_totals[cat] = cat_totals.get(cat, 0) + n
        cat_subs.setdefault(cat, []).append((sub, n))

    categories = []
    n_subcategories = 0
    for cat in sorted(cat_totals, key=lambda c: (-cat_totals[c], c)):
        subs = sorted(cat_subs[cat], key=lambda kv: (-kv[1], kv[0]))
        n_subcategories += len(subs)
        categories.append({
            "category": cat,
            "count": cat_totals[cat],
            "pct": pct(cat_totals[cat]),
            "subcategories": [
                {"subcategory": s, "count": n, "pct": pct(n)} for s, n in subs
            ],
        })

    return {
        "total_tickets": total,
        "n_categories": len(categories),
        "n_subcategories": n_subcategories,
        "categories": categories,
    }


def pivot_to_rows(pivot):
    """Flatten a compute_category_pivot() result into a tidy list of dicts
    (one row per Category+Subcategory pair) for tabular export."""
    rows = []
    for cat in pivot.get("categories", []):
        for sub in cat["subcategories"]:
            rows.append({
                "Category": cat["category"],
                "Subcategory": sub["subcategory"],
                "Count": sub["count"],
                "% of Total": sub["pct"],
            })
    return rows


def pivot_to_category_totals(pivot):
    """Flatten to one row per Category (totals) for export."""
    return [
        {"Category": c["category"], "Count": c["count"], "% of Total": c["pct"]}
        for c in pivot.get("categories", [])
    ]
