"""
Shared DataFrame column-resolution helpers.

Used by the analysis modules (disposition, impact_analysis, quality_audit) so
the "resolve a metadata role to a real column" logic lives in one place.

Two different questions get asked about columns here:

* ``resolve_column`` -- which real column fills a *metadata role* (priority,
  resolution time, ...), from the user's mapping plus name patterns.
* ``resolve_label_column`` -- which real column holds a *cluster label*. The
  pipeline writes "Repetitive Category"/"Repetitive Subcategory"; the bare names
  are accepted so an externally-prepared sheet still analyses.

The label constants and lookup lived in both impact_analysis and quality_audit,
and drifting copies caused the same bug twice: a call site looking only for the
bare name, which never exists in practice. Impact charts silently fell back to
raw Cluster_ID numbers (fixed in 81c234a) and the audit's Subcategory column was
always blank (fixed in f18c8e5). One copy now.
"""

# Order matters: the pipeline's own name first, the bare fallback second.
CLUSTER_CATEGORY_COLS = ("Repetitive Category", "Category")
CLUSTER_SUBCATEGORY_COLS = ("Repetitive Subcategory", "Subcategory")


def resolve_label_column(df, candidates):
    """First of `candidates` present on `df`, else None."""
    return next((c for c in candidates if c in df.columns), None)


def resolve_column(df, mapping, role, *patterns):
    """Resolve a DataFrame column for a metadata role.

    Priority:
      1. explicit mapping[role] if it names a present column;
      2. then, for each pattern in order: a case-insensitive EXACT column-name
         match, then a case-insensitive SUBSTRING match.
    Returns the column name, or None if nothing matches.

    With no patterns this is a mapping-only lookup (mapping → None), which is the
    behaviour the impact-analysis and quality-audit callers rely on. With
    patterns it also covers ServiceNow export naming styles: technical names
    (``work_notes``), display names (``Comments and Work notes``) and prefixed
    names (``inc_short_description``).
    """
    col = (mapping or {}).get(role)
    if col and col in df.columns:
        return col
    if not patterns:
        return None
    lcols = [(c, str(c).lower()) for c in df.columns]
    for pat in patterns:
        p = str(pat).lower()
        for c, lc in lcols:
            if lc == p:
                return c
        for c, lc in lcols:
            if p in lc:
                return c
    return None
