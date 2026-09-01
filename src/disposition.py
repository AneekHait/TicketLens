"""
Automation-opportunity analysis (Root Cause + Resolution + Disposition).

Consumes a *resolution-text* clustering pass (tickets grouped by HOW they were
resolved) and, per cluster, computes demand-reduction metrics and assigns one of
four dispositions using a hybrid of those metrics and the local LLM:

    Eradicate  - recurring defect/config/data problem with a fixable root cause;
                 permanently removing the cause eliminates the demand.
    Automate   - deterministic, repeatable fix with stable steps (reset, restart,
                 unlock, rerun, grant, clear). Route to self-heal / bot / script.
    Reimagine  - demand reveals a broken process or missing self-service; redesign
                 it (shift-left to a portal, fix routing, knowledge enablement).
    Retain     - genuine human judgement, or low-volume / high-variance; stay manual.

Mirrors the (llm, cluster_data, settings) generator pattern of kba_generator.py
and the (df, mapping, cluster_data) metric pattern of impact_analysis.py.
"""

import re

import pandas as pd

from src.logger import get_logger
from src.llm_utils import parse_llm_sections
from src.column_utils import resolve_column
from src.preprocessing import clean_resolution_text

logger = get_logger()

DISPOSITIONS = ["Automate", "Reimagine", "Eradicate", "Retain"]
# Ranking priority when ROI ties (lower = surfaced first).
_DISPOSITION_PRIORITY = {"Automate": 0, "Eradicate": 1, "Reimagine": 2, "Retain": 3}

_DISPOSITION_HEADERS = ["ROOT_CAUSE", "RESOLUTION", "DISPOSITION", "RATIONALE", "RECOMMENDATION"]

# Resolution verbs that signal a deterministic, scriptable fix.
_AUTOMATABLE_VERBS = [
    "reset", "restart", "reboot", "unlock", "rerun", "re-run", "clear", "grant",
    "enable", "disable", "reprocess", "resubmit", "release", "bounce", "flush",
    "purge", "reconnect", "recycle", "trigger", "reschedule", "kill", "cancel",
]
# Words that signal demand better served by self-service / process redesign.
_REIMAGINE_WORDS = ["access", "password", "request", "onboard", "provision", "training", "guidance"]

# close_code (lowercased, substring) -> disposition prior.
_CLOSE_CODE_HINTS = {
    "work around": "Eradicate",        # repeated workaround => fix the root cause
    "user training": "Reimagine",
    "resolved by caller": "Reimagine",
    "resolved by request": "Reimagine",
    "not reproducible": "Retain",
}

# Disposition definitions injected verbatim into the LLM prompt.
_DISPOSITION_GUIDE = (
    "- Eradicate: a recurring defect / config / data problem with a fixable root cause; "
    "permanently removing the cause eliminates the ticket. Route to problem management.\n"
    "- Automate: a deterministic, repeatable fix with stable steps (reset, restart, unlock, "
    "rerun, grant, clear). Route to self-heal / bot / script.\n"
    "- Reimagine: the demand reveals a broken process or missing self-service; redesign it "
    "(shift-left to a portal, fix routing, knowledge enablement).\n"
    "- Retain: genuinely needs human judgement, or is low-volume / high-variance; keep manual."
)


class DispositionAnalyzer:
    """Score resolution-clusters and assign an automation disposition.

    Args:
        llm: llama-cpp Llama instance (or None for metrics + heuristic only).
        df: DataFrame with a per-row resolution-cluster id column.
        cluster_data: Dict from TicketClusterer.get_cluster_data() of the
            resolution-text pass (keywords, subcategory=resolution pattern,
            sample_docs=resolution notes, sample_doc_indices).
        cluster_col: Name of the per-row cluster id column in df (e.g.
            "Resolution_Cluster").
        mapping: Metadata column mapping (effort/reopen/reassignment/close_code/symptom).
        settings: config["disposition"] block.
    """

    def __init__(self, llm, df, cluster_data, cluster_col, mapping=None, settings=None, res_cols=None):
        self.llm = llm
        self.df = df
        self.cluster_data = cluster_data or {}
        self.cluster_col = cluster_col
        self.mapping = mapping or {}
        self.settings = settings or {}
        self.res_cols = list(res_cols) if res_cols else []
        self.max_samples = self.settings.get("max_sample_tickets", 5)
        self.max_keywords = self.settings.get("max_keywords", 10)
        # ServiceNow business/calendar duration is exported in seconds; /3600 -> hours.
        # If a deployment exports milliseconds the absolute hours are off but the ROI
        # *ranking* is unaffected (every cluster scales identically).
        self.effort_divisor = self.settings.get("effort_seconds_per_hour", 3600)
        self._groups = None  # lazily-built {cluster_id: subframe} cache

    # ------------------------------------------------------------------
    # Column resolution
    # ------------------------------------------------------------------
    def _col(self, role, *patterns):
        """Resolve a column for a role across ServiceNow export naming styles.
        See column_utils.resolve_column for the resolution rules."""
        return resolve_column(self.df, self.mapping, role, *patterns)

    # ------------------------------------------------------------------
    # Per-cluster metrics
    # ------------------------------------------------------------------
    def _cluster_groups(self):
        """Group the frame by cluster once. generate() is called per cluster, so
        without this each call re-scanned the whole frame (O(clusters x rows))."""
        if self._groups is None:
            self._groups = dict(tuple(self.df.groupby(self.cluster_col, sort=False)))
        return self._groups

    def _metrics(self, cluster_id):
        """Volume / effort / reopen / reassignment / closure consistency + ROI."""
        sub = self._cluster_groups().get(cluster_id)
        if sub is None:
            sub = self.df.iloc[0:0]
        volume = int(len(sub))

        effort_hours = None        # avg per ticket
        total_effort_hours = None  # cluster total (the prize for eliminating it)
        effort_col = self._col(
            "effort_col", "business_duration", "business duration",
            "calendar_duration", "calendar duration", "time_worked", "time worked", "duration",
        )
        if effort_col is not None:
            secs = pd.to_numeric(sub[effort_col], errors="coerce")
            if secs.notna().any():
                effort_hours = round(float(secs.mean()) / self.effort_divisor, 2)
                total_effort_hours = round(float(secs.sum()) / self.effort_divisor, 1)

        reopen_pct = None
        reopen_col = self._col("reopen_col", "reopen_count", "reopen count", "reopened")
        if reopen_col is not None:
            rc = pd.to_numeric(sub[reopen_col], errors="coerce").fillna(0)
            if len(rc):
                reopen_pct = round(float((rc > 0).mean()) * 100, 1)

        reassign_avg = None
        reassign_col = self._col("reassignment_col", "reassignment_count", "reassignment count", "reassign")
        if reassign_col is not None:
            ra = pd.to_numeric(sub[reassign_col], errors="coerce")
            if ra.notna().any():
                reassign_avg = round(float(ra.mean()), 2)

        dominant_close = None
        dominant_share = None
        close_col = self._col("close_code_col", "close_code", "close code", "closure code", "resolution code")
        if close_col is not None:
            cc = sub[close_col].dropna().astype(str)
            cc = cc[~cc.str.strip().isin(["", "nan", "None"])]
            if len(cc):
                vc = cc.value_counts()
                dominant_close = str(vc.index[0]).strip().strip("'\"")
                dominant_share = round(float(vc.iloc[0]) / len(cc) * 100, 1)

        # ROI = total human-hours spent on the cluster (fallback to volume).
        roi = total_effort_hours if total_effort_hours is not None else float(volume)

        return {
            "volume": volume,
            "effort_hours": effort_hours,
            "total_effort_hours": total_effort_hours,
            "roi": round(roi, 1),
            "reopen_pct": reopen_pct,
            "reassign_avg": reassign_avg,
            "dominant_close_code": dominant_close,
            "dominant_close_share": dominant_share,
        }

    # ------------------------------------------------------------------
    # Heuristic prior (seeds the LLM + fallback when the LLM is unavailable)
    # ------------------------------------------------------------------
    def _heuristic(self, metrics, keywords, sample_texts):
        """Return (suggested_disposition, automatability_score 0..1)."""
        text = " ".join([str(k) for k in keywords] + [str(t) for t in sample_texts]).lower()
        has_auto_verb = any(re.search(r"\b" + re.escape(v) + r"\b", text) for v in _AUTOMATABLE_VERBS)

        # None means "no such column in the data", which is NOT the same as a measured
        # zero. `or 0.0` conflated them, and zero is the *best* possible value for reopen
        # and reassign — so a dataset with no reopen/reassignment columns (very common)
        # scored full marks on both, and the Automate gate below collapsed to
        # has_auto_verb alone. Automatability was systematically optimistic and
        # indistinguishable from a genuinely stable cluster.
        reopen = metrics.get("reopen_pct")
        reassign = metrics.get("reassign_avg")
        share = metrics.get("dominant_close_share")

        # Automatability: deterministic verb + stable resolution (low reopen/reassign,
        # consistent closure code). Each component contributes only if it was measured;
        # the total is renormalised over the components present, so a missing column
        # neither inflates nor deflates the result (same approach as quality_audit's
        # composite score).
        score, weight_total = 0.0, 0.0

        score += 0.40 if has_auto_verb else 0.0
        weight_total += 0.40                      # always measurable from the text

        if reopen is not None:
            score += 0.20 * max(0.0, 1.0 - reopen / 20.0)     # reopen < 20% is good
            weight_total += 0.20
        if reassign is not None:
            score += 0.20 * max(0.0, 1.0 - reassign / 3.0)    # < 3 reassignments is good
            weight_total += 0.20
        if share is not None:
            score += 0.20 * (share / 100.0)
            weight_total += 0.20

        score = round(min(score / weight_total, 1.0), 2) if weight_total else 0.0

        # An unmeasured metric must not *disqualify* a cluster either, or a dataset
        # without these columns could never be Automate. Unknown is treated as
        # "no evidence against", which the renormalised score above already reflects.
        stable = (reopen is None or reopen < 20) and (reassign is None or reassign < 3)
        if has_auto_verb and stable:
            prior = "Automate"
        else:
            prior = None
            dc = (metrics.get("dominant_close_code") or "").lower()
            for key, disp in _CLOSE_CODE_HINTS.items():
                if key in dc:
                    prior = disp
                    break
            if prior is None:
                if reassign and reassign >= 3:
                    prior = "Reimagine"
                elif any(re.search(r"\b" + w + r"\b", text) for w in _REIMAGINE_WORDS):
                    prior = "Reimagine"
                else:
                    prior = "Retain"
        return prior, score

    # ------------------------------------------------------------------
    # LLM: root cause + resolution + disposition
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_disposition(text):
        """Fuzzy-map free LLM text to one of the four buckets, or None."""
        t = (text or "").strip().lower()
        if not t:
            return None
        checks = [
            ("Automate", ["automat"]),
            ("Reimagine", ["reimagine", "redesign", "self-service", "self service", "shift-left", "shift left", "portal"]),
            ("Eradicate", ["eradicat", "eliminat", "permanent", "problem management", "root cause fix", "root-cause fix"]),
            ("Retain", ["retain", "manual", "keep as", "no change", "human judg"]),
        ]
        for disp, needles in checks:
            if any(n in t for n in needles):
                return disp
        return None

    def _build_prompt(self, subcategory, keywords, samples, metrics, prior):
        kw = ", ".join(str(k) for k in keywords) or "N/A"
        sample_lines = []
        for i, (symptom, resolution) in enumerate(samples, 1):
            symptom = (symptom or "").strip() or "N/A"
            resolution = (resolution or "").strip() or "N/A"
            sample_lines.append(f"{i}. SYMPTOM: {symptom}\n   RESOLVED BY: {resolution}")
        sample_str = "\n".join(sample_lines) if sample_lines else "N/A"

        def fmt(v, suffix=""):
            return f"{v}{suffix}" if v is not None else "N/A"

        metric_line = (
            f"Tickets: {metrics['volume']}; "
            f"avg handling time: {fmt(metrics['effort_hours'], ' h')}; "
            f"total effort: {fmt(metrics['total_effort_hours'], ' h')}; "
            f"reopened: {fmt(metrics['reopen_pct'], '%')}; "
            f"avg reassignments: {fmt(metrics['reassign_avg'])}; "
            f"most common closure: {fmt(metrics['dominant_close_code'])} "
            f"({fmt(metrics['dominant_close_share'], '%')})."
        )

        return f"""You are an IT service-management transformation analyst. The tickets below were all resolved the same way. Decide how to permanently reduce this demand.

Resolution pattern: {subcategory or 'N/A'}
Keywords: {kw}
Metrics: {metric_line}

Sample tickets:
{sample_str}

Disposition options (choose exactly ONE):
{_DISPOSITION_GUIDE}

A metrics-based heuristic suggests: {prior}. Treat it only as a hint.

Reply in EXACTLY this format, one line each:
ROOT_CAUSE: (one sentence — the underlying cause of these tickets)
RESOLUTION: (one sentence — the common fix that was applied)
DISPOSITION: (exactly one of: Automate, Reimagine, Eradicate, Retain)
RATIONALE: (one sentence — why that disposition)
RECOMMENDATION: (one sentence — the concrete next step)"""

    def _llm_analyze(self, subcategory, keywords, samples, metrics, prior):
        """Return (root_cause, resolution, disposition, rationale, recommendation)."""
        kw_str = ", ".join(str(k) for k in keywords)
        if not self.llm:
            # Heuristic-only fallback.
            return (
                f"Recurring issue related to: {kw_str}." if kw_str else "",
                samples[0][1] if samples and samples[0][1] else "",
                prior,
                "Assigned from resolution metrics (LLM disabled).",
                "",
            )

        prompt = self._build_prompt(subcategory, keywords, samples, metrics, prior)
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=320,
                temperature=0.2,
            )
            raw = output["choices"][0]["message"]["content"].strip()
            sections = parse_llm_sections(raw, _DISPOSITION_HEADERS)
            disposition = self._normalize_disposition(sections.get("DISPOSITION", "")) or prior
            return (
                sections.get("ROOT_CAUSE", "").strip(),
                sections.get("RESOLUTION", "").strip(),
                disposition,
                sections.get("RATIONALE", "").strip(),
                sections.get("RECOMMENDATION", "").strip(),
            )
        except Exception as e:
            logger.error(f"Disposition LLM failed for cluster {subcategory}: {e}")
            return (
                f"Recurring issue related to: {kw_str}." if kw_str else "",
                "",
                prior,
                "Assigned from resolution metrics (LLM error).",
                "",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, cluster_id):
        """Analyze one resolution cluster -> dict (metrics + RCA + disposition)."""
        cdata = self.cluster_data.get(cluster_id, {}) or {}
        keywords = (cdata.get("keywords", []) or [])[:self.max_keywords]
        subcategory = cdata.get("subcategory", "") or ""
        indices = (cdata.get("sample_doc_indices", []) or [])
        res_samples = (cdata.get("sample_docs", []) or [])

        symptom_col = self._col(
            "symptom_col", "short_description", "short description",
            "inc_short_description", "description",
        )
        # Resolution text columns available in df (fetched by row index).
        df_res_cols = [c for c in self.res_cols if c in self.df.columns]

        samples = []
        n = min(self.max_samples, max(len(indices), len(res_samples)))
        for i in range(n):
            row = self.df.iloc[indices[i]] if i < len(indices) and 0 <= indices[i] < len(self.df) else None

            symptom = ""
            if row is not None and symptom_col is not None:
                symptom = str(row.get(symptom_col, "") or "")[:160]

            # Prefer live resolution text from the df row; fall back to cluster_data
            # sample_docs (which may be symptom text when called without res_cols).
            # clean_resolution_text strips ServiceNow journal metadata first — entry
            # headers like "2024-01-15 10:23:45 - J Smith (Work notes)" and _x000D_
            # artifacts. Without it those eat a large share of the 240-char budget below,
            # so the LLM saw timestamps and author names instead of the resolution.
            if row is not None and df_res_cols:
                resolution = clean_resolution_text(
                    " ".join(str(row.get(c, "") or "") for c in df_res_cols))[:240]
            elif i < len(res_samples):
                resolution = clean_resolution_text(str(res_samples[i]))[:240]
            else:
                resolution = ""

            samples.append((symptom, resolution))

        metrics = self._metrics(cluster_id)
        prior, auto_score = self._heuristic(metrics, keywords, [r for _, r in samples])
        metrics["automatability"] = auto_score

        root_cause, resolution, disposition, rationale, recommendation = self._llm_analyze(
            subcategory, keywords, samples, metrics, prior
        )

        return {
            "cluster_id": cluster_id,
            "resolution_pattern": subcategory,
            "keywords": keywords,
            "disposition": disposition,
            "heuristic_disposition": prior,
            "root_cause": root_cause,
            "resolution": resolution,
            "rationale": rationale,
            "recommendation": recommendation,
            **metrics,
        }

    def generate_all(self, callback=None):
        """Analyze every real resolution cluster; return list ranked by ROI desc.

        Skips the -1 (Non-Repetitive) bucket — one-off resolutions are not a pattern.
        """
        cluster_ids = sorted(cid for cid in self.cluster_data if cid != -1)
        total = len(cluster_ids)
        results = []
        for i, cid in enumerate(cluster_ids):
            if callback:
                callback(f"Analyzing resolution cluster {i + 1}/{total}...", i / total if total else 1.0)
            results.append(self.generate(cid))

        results.sort(
            key=lambda r: (-(r.get("roi") or 0), _DISPOSITION_PRIORITY.get(r.get("disposition"), 9))
        )
        if callback:
            callback(f"Analyzed {len(results)} resolution clusters.", 1.0)
        logger.info(f"Disposition analysis complete: {len(results)} clusters")
        return results

    @staticmethod
    def summarize(results):
        """Aggregate counts, ticket volume and total ROI hours by disposition."""
        summary = {}
        for disp in DISPOSITIONS:
            rows = [r for r in results if r.get("disposition") == disp]
            summary[disp] = {
                "clusters": len(rows),
                "tickets": int(sum(r.get("volume") or 0 for r in rows)),
                "roi_hours": round(sum(r.get("roi") or 0 for r in rows), 1),
            }
        return summary
