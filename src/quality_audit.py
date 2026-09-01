"""
Ticket quality audit engine.
Assesses completeness, categorization accuracy, and resolution quality.
"""

import re
from src.logger import get_logger
from src.column_utils import (CLUSTER_CATEGORY_COLS, CLUSTER_SUBCATEGORY_COLS,
                              resolve_column, resolve_label_column)

logger = get_logger()



def _is_missing(value):
    """True for None and for NaN.

    NaN needs its own test: it is not None, and it is the form None takes as soon as it
    lands in a numeric DataFrame column — so `x is not None` lets it through and the mean
    of a list containing it is NaN. `value != value` is only true for NaN, and avoids
    importing pandas into this module just for isna().
    """
    return value is None or value != value

# Patterns indicating vague/generic ticket descriptions
VAGUE_PATTERNS = [
    r'\b(not working|doesnt work|doesn\'t work|broken|issue|problem|help|error)\b',
    r'\b(something wrong|cannot|can\'t|unable)\b',
    r'\b(please fix|please help|asap|urgent)\b',
]

# Patterns indicating generic/unhelpful resolution notes
GENERIC_RESOLUTION_PATTERNS = [
    r'^(resolved|done|fixed|closed|completed|ok|okay)\.?$',
    r'^(issue resolved|problem fixed|ticket closed|case closed)\.?$',
    r'^(no action needed|not applicable|n/a|na)\.?$',
]


class TicketQualityAuditor:
    """Audits ticket data quality across three dimensions.

    Args:
        df: DataFrame with ticket data and clustering results.
        metadata_mapping: Dict mapping role names to column names.
        cluster_results: Dict mapping topic_id -> subcategory label.
        settings: Audit settings from config.
    """

    def __init__(self, df, metadata_mapping, cluster_results=None, settings=None):
        self.df = df.copy()
        self.mapping = metadata_mapping or {}
        self.cluster_results = cluster_results or {}
        self.settings = settings or {}
        self.min_desc_len = self.settings.get("min_description_length", 10)
        self.vague_threshold = self.settings.get("vague_word_threshold", 0.5)

    def _get_mapped_col(self, role):
        """Return the column name for a metadata role, or None."""
        return resolve_column(self.df, self.mapping, role)

    def audit_completeness(self, text_columns=None):
        """Score each ticket on field completeness and description quality.

        Returns a Series of scores 0-100.
        """
        n = len(self.df)
        scores = [0.0] * n
        checks_count = 0

        # Check mapped metadata columns for nulls/empty
        meta_roles = [
            "priority_col", "assignment_group_col",
            "created_date_col", "resolution_notes_col",
        ]
        mapped_cols = []
        for role in meta_roles:
            col = self._get_mapped_col(role)
            if col:
                mapped_cols.append(col)

        if mapped_cols:
            checks_count += len(mapped_cols)
            # Read each column once instead of self.df.iloc[idx][c] per row (the
            # latter rebuilds a row Series for every row × every pass).
            col_values = {c: self.df[c].tolist() for c in mapped_cols}
            for idx in range(n):
                filled = sum(
                    1 for c in mapped_cols
                    if col_values[c][idx] is not None
                    and str(col_values[c][idx]).strip() not in ("", "nan", "None")
                )
                scores[idx] += (filled / len(mapped_cols)) * 100

        # Combined text per row, computed once and reused by both text passes.
        if text_columns:
            col_lists = {
                c: (self.df[c].tolist() if c in self.df.columns else None)
                for c in text_columns
            }
            combined_text = [
                " ".join(
                    str(col_lists[c][idx]) if col_lists[c] is not None else ""
                    for c in text_columns
                ).strip()
                for idx in range(n)
            ]

            # Description length scoring
            checks_count += 1
            for idx in range(n):
                word_count = len(combined_text[idx].split())
                if word_count >= self.min_desc_len:
                    scores[idx] += 100
                elif word_count > 0:
                    scores[idx] += (word_count / self.min_desc_len) * 100

            # Vague language penalty
            checks_count += 1
            compiled = [re.compile(p, re.IGNORECASE) for p in VAGUE_PATTERNS]
            for idx in range(n):
                combined = combined_text[idx]
                if not combined:
                    continue
                words = combined.split()
                # Count vague *words*, not how many of the patterns matched at least
                # once. With the old `sum(1 for p in compiled if p.search(...))` the
                # numerator could never exceed len(VAGUE_PATTERNS), so the ratio could
                # only reach the 0.5 threshold on descriptions of ~6 words or fewer —
                # every longer description scored a full 100 no matter how vague.
                vague_hits = sum(len(p.findall(combined)) for p in compiled)
                vague_ratio = vague_hits / max(len(words), 1)
                if vague_ratio < self.vague_threshold:
                    scores[idx] += 100
                else:
                    scores[idx] += max(0, (1 - vague_ratio) * 100)

        if checks_count == 0:
            # Nothing was actually measured (no metadata columns mapped and no text
            # columns known — reachable by loading a file and going straight to the
            # audit, since selected_text_cols is only set by a clustering run).
            # Returning 50.0 invented a score that looked like a real measurement and
            # was indistinguishable from a genuinely mediocre dataset.
            return [None] * n

        return [round(s / checks_count, 1) for s in scores]

    def audit_categorization(self):
        """Compare original category column against cluster-assigned categories.

        Returns a Series of scores 0-100.
        """
        cat_col = self._get_mapped_col("category_col")
        # The pipeline writes "Repetitive Category"; this used to look for a bare
        # "Category" that never exists, so the whole dimension silently scored None
        # for every row and Categorization_Score was always blank.
        assigned_col = resolve_label_column(self.df, CLUSTER_CATEGORY_COLS)
        if not cat_col or not assigned_col:
            return [None] * len(self.df)

        orig_vals = self.df[cat_col].tolist()
        assigned_vals = self.df[assigned_col].tolist()
        scores = []
        for idx in range(len(self.df)):
            original = str(orig_vals[idx]).strip().lower()
            assigned = str(assigned_vals[idx]).strip().lower()
            if not original or original in ("nan", "none", ""):
                scores.append(None)
                continue
            if original == assigned:
                scores.append(100.0)
            elif original in assigned or assigned in original:
                scores.append(75.0)
            else:
                # Simple word overlap ratio
                orig_words = set(original.split())
                assigned_words = set(assigned.split())
                if orig_words and assigned_words:
                    overlap = len(orig_words & assigned_words)
                    total = len(orig_words | assigned_words)
                    scores.append(round((overlap / total) * 100, 1))
                else:
                    scores.append(0.0)
        return scores

    def audit_resolution_quality(self):
        """Score resolution notes on presence, length, and meaningfulness.

        Returns a Series of scores 0-100.
        """
        res_col = self._get_mapped_col("resolution_notes_col")
        if not res_col:
            return [None] * len(self.df)

        compiled_generic = [
            re.compile(p, re.IGNORECASE) for p in GENERIC_RESOLUTION_PATTERNS
        ]
        res_vals = self.df[res_col].tolist()
        scores = []
        for idx in range(len(self.df)):
            val = str(res_vals[idx]).strip()
            if not val or val.lower() in ("nan", "none", ""):
                scores.append(0.0)
                continue

            score = 0.0
            # Presence: 30 points
            score += 30.0

            # Length: up to 40 points (>= 20 words = full marks)
            word_count = len(val.split())
            score += min(40.0, (word_count / 20) * 40)

            # Not generic: 30 points
            is_generic = any(p.match(val) for p in compiled_generic)
            if not is_generic:
                score += 30.0

            scores.append(round(min(score, 100.0), 1))
        return scores

    def generate_report(self, text_columns=None):
        """Run all audits and return augmented DataFrame + summary dict.

        Args:
            text_columns: List of column names used for ticket text.

        Returns:
            (df, summary) tuple.
        """
        weights = {
            "completeness": self.settings.get("completeness_weight", 0.4),
            "categorization": self.settings.get("categorization_weight", 0.3),
            "resolution": self.settings.get("resolution_weight", 0.3),
        }

        comp_scores = self.audit_completeness(text_columns)
        cat_scores = self.audit_categorization()
        res_scores = self.audit_resolution_quality()

        self.df["Completeness_Score"] = comp_scores
        self.df["Categorization_Score"] = cat_scores
        self.df["Resolution_Score"] = res_scores

        # Composite quality score
        quality = []
        for i in range(len(self.df)):
            parts = []
            w_total = 0
            if comp_scores[i] is not None:
                parts.append(comp_scores[i] * weights["completeness"])
                w_total += weights["completeness"]
            if cat_scores[i] is not None:
                parts.append(cat_scores[i] * weights["categorization"])
                w_total += weights["categorization"]
            if res_scores[i] is not None:
                parts.append(res_scores[i] * weights["resolution"])
                w_total += weights["resolution"]
            if w_total > 0:
                quality.append(round(sum(parts) / w_total, 1))
            else:
                quality.append(None)

        self.df["Quality_Score"] = quality

        valid_quality = [q for q in quality if q is not None]
        valid_comp = [c for c in comp_scores if c is not None]
        valid_cat = [c for c in cat_scores if c is not None]
        valid_res = [r for r in res_scores if r is not None]

        summary = {
            "overall_score": round(sum(valid_quality) / len(valid_quality), 1) if valid_quality else None,
            "completeness_avg": round(sum(valid_comp) / len(valid_comp), 1) if valid_comp else None,
            "categorization_avg": round(sum(valid_cat) / len(valid_cat), 1) if valid_cat else None,
            "resolution_avg": round(sum(valid_res) / len(valid_res), 1) if valid_res else None,
            "total_tickets": len(self.df),
            "tickets_scored": len(valid_quality),
        }

        # Per-cluster summary
        if "Cluster_ID" in self.df.columns:
            # The pipeline writes "Repetitive Subcategory"; this looked only for a bare
            # "Subcategory", which never exists — so the label was always "" in the
            # per-cluster table and in the exported Cluster Breakdown sheet.
            sub_col = resolve_label_column(self.df, CLUSTER_SUBCATEGORY_COLS)
            cluster_summary = {}
            for cid in sorted(self.df["Cluster_ID"].unique()):
                mask = self.df["Cluster_ID"] == cid
                subset = self.df[mask]
                cq = [q for q in subset["Quality_Score"] if not _is_missing(q)]
                cluster_summary[int(cid)] = {
                    "count": int(mask.sum()),
                    "avg_quality": round(sum(cq) / len(cq), 1) if cq else None,
                    "subcategory": str(subset[sub_col].iloc[0]) if sub_col else "",
                }
            summary["clusters"] = cluster_summary

        logger.info(f"Quality audit complete: overall={summary['overall_score']}")
        return self.df, summary
