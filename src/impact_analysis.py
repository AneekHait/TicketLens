"""
Impact analysis engine.
Provides problem cluster analysis, business process impact, and KPI impact
analysis with interactive Plotly visualizations.
"""

import webbrowser
import tempfile
import os
import atexit
from pathlib import Path
from src.logger import get_logger
from src.column_utils import (CLUSTER_CATEGORY_COLS, CLUSTER_SUBCATEGORY_COLS,
                              resolve_column, resolve_label_column)

logger = get_logger()


def _remove_quietly(path):
    """Delete a temp file, ignoring the case where it is already gone."""
    try:
        os.remove(path)
    except OSError:
        pass



class ImpactAnalyzer:
    """Analyzes ticket impact across problem clusters, business processes, and KPIs.

    Args:
        df: DataFrame with ticket data and clustering results (Cluster_ID,
            Repetitive Subcategory, Repetitive Category columns).
        metadata_mapping: Dict mapping role names to column names.
        cluster_data: Dict from TicketClusterer.get_cluster_data().
        settings: Analysis settings from config.
    """

    def __init__(self, df, metadata_mapping, cluster_data=None, settings=None):
        self.df = df.copy()
        self.mapping = metadata_mapping or {}
        self.cluster_data = cluster_data or {}
        self.settings = settings or {}
        self.top_n = self.settings.get("top_n_clusters", 10)
        # No SLA target here: the breach rate is read from the source sla_status_col
        # (see analyze_kpi_impact), not derived from a duration threshold.

    def _col(self, role):
        """Return column name for a metadata role if mapped and present."""
        return resolve_column(self.df, self.mapping, role)

    def _label_col(self, candidates, frame=None):
        """First of `candidates` present on the frame (default self.df), else None."""
        return resolve_label_column(self.df if frame is None else frame, candidates)

    def _slice(self, *cols):
        """A copy of self.df narrowed to `cols` (deduped, missing ones skipped).

        The methods below each add one derived column (_res_time, _breached, ...) before
        grouping, so they must not mutate self.df. They used to copy the *entire* frame
        to do it — four full copies per run on top of the constructor's, which on a wide
        ticket export is most of the memory this class uses. Only a couple of columns
        are ever read.
        """
        seen, keep = set(), []
        for c in cols:
            if c and c in self.df.columns and c not in seen:
                seen.add(c)
                keep.append(c)
        return self.df[keep].copy()

    # ------------------------------------------------------------------
    # Problem Cluster Analysis
    # ------------------------------------------------------------------

    def analyze_problem_clusters(self):
        """Compute cluster volume statistics and inter-cluster relationships.

        Returns dict with:
            volume_stats: DataFrame (cluster_id, subcategory, count, pct)
            trends: DataFrame or None (date, cluster_id, count) if date column mapped
        """
        import pandas as pd

        results = {}

        if "Cluster_ID" not in self.df.columns:
            return results

        # Volume distribution. Uses _label_col so a sheet carrying the bare
        # "Subcategory"/"Category" names is handled too — hardcoding the Repetitive*
        # names here meant such a sheet fell through to Cluster_ID, filling the
        # subcategory column with raw cluster numbers. Worse, the downstream guard in
        # analyze_main_theme tests `"subcategory" in themed.columns`, which was then
        # always true, so it could never detect that the fallback had happened.
        _sub = self._label_col(CLUSTER_SUBCATEGORY_COLS)
        _cat = self._label_col(CLUSTER_CATEGORY_COLS)
        vol = (
            self.df.groupby("Cluster_ID")
            .agg(
                count=("Cluster_ID", "size"),
                subcategory=(_sub, "first") if _sub else ("Cluster_ID", "first"),
                category=(_cat, "first") if _cat else ("Cluster_ID", "first"),
            )
            .reset_index()
        )
        vol["pct"] = (vol["count"] / vol["count"].sum() * 100).round(1)
        vol = vol.sort_values("count", ascending=False)
        results["volume_stats"] = vol

        # Temporal trends
        date_col = self._col("created_date_col")
        if date_col:
            try:
                temp = self._slice(date_col, "Cluster_ID")
                temp["_date"] = pd.to_datetime(temp[date_col], errors="coerce")
                temp = temp.dropna(subset=["_date"])
                if not temp.empty:
                    temp["_period"] = temp["_date"].dt.to_period("M").astype(str)
                    trends = (
                        temp.groupby(["_period", "Cluster_ID"])
                        .size()
                        .reset_index(name="count")
                    )
                    results["trends"] = trends
            except Exception as e:
                logger.warning(f"Trend analysis failed: {e}")

        return results

    # ------------------------------------------------------------------
    # Overall Main Theme
    # ------------------------------------------------------------------

    def analyze_main_theme(self, problem=None):
        """Summarise the single overall theme across all tickets.

        Deterministic (no LLM): reuses :meth:`analyze_problem_clusters` volume
        statistics plus per-cluster keywords to surface the dominant category,
        the largest sub-themes by volume and the most representative terms. All
        percentages are relative to the total ticket count.

        Args:
            problem: Optional pre-computed :meth:`analyze_problem_clusters`
                result, reused to avoid recomputing the volume stats.

        Returns:
            dict with: ``total_tickets``, ``n_clusters``, ``dominant_category``,
            ``dominant_category_pct``, ``noise_pct`` (share of one-off tickets),
            ``top_subcategories`` (list of ``(label, count, pct)``),
            ``top_keywords`` (list of ``(term, weight)``) and a one-line
            ``headline`` string. Safe on empty/unclustered data.
        """
        summary = {
            "total_tickets": int(len(self.df)),
            "n_clusters": 0,
            "dominant_category": None,
            "dominant_category_pct": 0.0,
            "noise_pct": 0.0,
            "top_subcategories": [],
            "top_keywords": [],
            "headline": "",
        }
        if "Cluster_ID" not in self.df.columns or len(self.df) == 0:
            return summary

        if problem is None:
            problem = self.analyze_problem_clusters()
        vol = problem.get("volume_stats")
        if vol is None or vol.empty:
            return summary

        total = int(vol["count"].sum()) or 1

        # Themed clusters exclude the Non-Repetitive bucket (Cluster_ID == -1).
        themed = vol[vol["Cluster_ID"] != -1]
        summary["n_clusters"] = int(len(themed))
        noise = vol[vol["Cluster_ID"] == -1]["count"].sum()
        summary["noise_pct"] = round(float(noise) / total * 100, 1)
        if themed.empty:
            themed = vol

        # Dominant category by ticket volume (ignore empty / Non-Repetitive).
        if "category" in themed.columns:
            cat = themed.groupby("category")["count"].sum().sort_values(ascending=False)
            cat = cat[[str(c).strip() not in ("", "Non-Repetitive") for c in cat.index]]
            if not cat.empty:
                summary["dominant_category"] = str(cat.index[0])
                summary["dominant_category_pct"] = round(cat.iloc[0] / total * 100, 1)

        # Largest sub-themes by volume.
        label_col = "subcategory" if "subcategory" in themed.columns else "Cluster_ID"
        for _, row in themed.head(self.top_n).iterrows():
            label = str(row.get(label_col, row["Cluster_ID"])).strip() or f"Cluster {row['Cluster_ID']}"
            summary["top_subcategories"].append(
                (label, int(row["count"]), float(row["pct"]))
            )

        # Aggregate keywords across clusters, weighted by cluster volume and rank.
        # Same frame, so the two columns are equal length by construction;
        # strict=True makes that an assertion rather than an assumption.
        counts_by_tid = dict(zip(vol["Cluster_ID"], vol["count"], strict=True))
        weights = {}
        for tid, cdata in self.cluster_data.items():
            if tid == -1:
                continue
            vol_weight = float(counts_by_tid.get(tid, 0)) or 1.0
            keywords = cdata.get("keywords", []) or []
            for rank, kw in enumerate(keywords[:10]):
                term = str(kw).strip().lower()
                if not term:
                    continue
                weights[term] = weights.get(term, 0.0) + vol_weight * (10 - rank)
        top_keywords = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:12]
        summary["top_keywords"] = [(term, round(weight, 1)) for term, weight in top_keywords]

        # One-line headline.
        parts = [f"{summary['total_tickets']:,} tickets across {summary['n_clusters']} themes"]
        if summary["dominant_category"]:
            parts.append(
                f"led by \u201c{summary['dominant_category']}\u201d "
                f"({summary['dominant_category_pct']:.0f}% of tickets)"
            )
        if summary["top_subcategories"]:
            lead = summary["top_subcategories"][0]
            parts.append(
                f"the single biggest issue is \u201c{lead[0]}\u201d "
                f"({lead[1]:,} tickets, {lead[2]:.0f}%)"
            )
        summary["headline"] = "; ".join(parts) + "."

        return summary

    # ------------------------------------------------------------------
    # Business Process Impact
    # ------------------------------------------------------------------

    def analyze_business_process(self):
        """Analyze workload distribution and bottlenecks.

        Returns dict with:
            workload: DataFrame (assignment_group, cluster_count, ticket_count)
            bottlenecks: DataFrame (cluster_id, subcategory, count, avg_resolution)
        """
        import pandas as pd
        results = {}

        ag_col = self._col("assignment_group_col")
        if ag_col and "Cluster_ID" in self.df.columns:
            workload = (
                self.df.groupby(ag_col)
                .agg(
                    ticket_count=(ag_col, "size"),
                    unique_clusters=("Cluster_ID", "nunique"),
                )
                .reset_index()
                .sort_values("ticket_count", ascending=False)
            )
            results["workload"] = workload

        # Bottleneck detection: high volume + high resolution time
        res_col = self._col("resolution_time_col")
        if res_col and "Cluster_ID" in self.df.columns:
            try:
                sub_col = self._label_col(CLUSTER_SUBCATEGORY_COLS)
                temp = self._slice(res_col, "Cluster_ID", sub_col)
                temp["_res_time"] = pd.to_numeric(temp[res_col], errors="coerce")
                bottlenecks = (
                    temp.groupby("Cluster_ID")
                    .agg(
                        count=("Cluster_ID", "size"),
                        avg_resolution=("_res_time", "mean"),
                        subcategory=(sub_col, "first") if sub_col else ("Cluster_ID", "first"),
                    )
                    .reset_index()
                )
                bottlenecks = bottlenecks.dropna(subset=["avg_resolution"])
                bottlenecks["avg_resolution"] = bottlenecks["avg_resolution"].round(1)
                # Flag: above-median volume AND above-median resolution time
                if not bottlenecks.empty:
                    med_count = bottlenecks["count"].median()
                    med_res = bottlenecks["avg_resolution"].median()
                    bottlenecks["is_bottleneck"] = (
                        (bottlenecks["count"] >= med_count)
                        & (bottlenecks["avg_resolution"] >= med_res)
                    )
                    bottlenecks = bottlenecks.sort_values(
                        "avg_resolution", ascending=False
                    )
                results["bottlenecks"] = bottlenecks
            except Exception as e:
                logger.warning(f"Bottleneck analysis failed: {e}")

        return results

    # ------------------------------------------------------------------
    # KPI Impact Analysis
    # ------------------------------------------------------------------

    def analyze_kpi_impact(self):
        """Compute KPI metrics per cluster and rank by composite impact.

        Returns dict with:
            resolution_stats: DataFrame per cluster
            sla_stats: DataFrame per cluster
            priority_dist: DataFrame (cluster_id, priority, count)
            impact_ranking: DataFrame sorted by composite score
        """
        import pandas as pd
        results = {}

        if "Cluster_ID" not in self.df.columns:
            return results

        cluster_ids = sorted(self.df["Cluster_ID"].unique())

        # Resolution time stats
        res_col = self._col("resolution_time_col")
        if res_col:
            try:
                temp = self._slice(res_col, "Cluster_ID")
                temp["_res"] = pd.to_numeric(temp[res_col], errors="coerce")
                stats = (
                    temp.groupby("Cluster_ID")["_res"]
                    .agg(["mean", "median", lambda x: x.quantile(0.95)])
                    .reset_index()
                )
                stats.columns = ["Cluster_ID", "avg_resolution", "median_resolution", "p95_resolution"]
                stats = stats.round(1)
                results["resolution_stats"] = stats
            except Exception as e:
                logger.warning(f"Resolution stats failed: {e}")

        # SLA breach rate
        sla_col = self._col("sla_status_col")
        if sla_col:
            try:
                temp = self._slice(sla_col, "Cluster_ID")
                sla_lower = temp[sla_col].astype(str).str.lower()
                temp["_breached"] = sla_lower.isin(["breached", "missed", "failed", "no", "false", "0"])
                sla_stats = (
                    temp.groupby("Cluster_ID")
                    .agg(
                        total=(sla_col, "size"),
                        breached=("_breached", "sum"),
                    )
                    .reset_index()
                )
                sla_stats["breach_rate"] = (sla_stats["breached"] / sla_stats["total"] * 100).round(1)
                results["sla_stats"] = sla_stats
            except Exception as e:
                logger.warning(f"SLA analysis failed: {e}")

        # Priority distribution
        pri_col = self._col("priority_col")
        if pri_col:
            try:
                pri_dist = (
                    self.df.groupby(["Cluster_ID", pri_col])
                    .size()
                    .reset_index(name="count")
                    .rename(columns={pri_col: "priority"})
                )
                results["priority_dist"] = pri_dist
            except Exception as e:
                logger.warning(f"Priority analysis failed: {e}")

        # Composite impact ranking. Precompute per-cluster volume and the first
        # (positional) Subcategory once, instead of re-scanning the whole frame
        # (self.df["Cluster_ID"] == cid) for every cluster.
        counts = self.df["Cluster_ID"].value_counts()
        _sub_col = self._label_col(CLUSTER_SUBCATEGORY_COLS)
        subcat_by_cid = (
            self.df.drop_duplicates("Cluster_ID", keep="first")
            .set_index("Cluster_ID")[_sub_col]
            if _sub_col else None
        )
        impact_rows = []
        for cid in cluster_ids:
            count = int(counts.get(cid, 0))
            subcat = subcat_by_cid.get(cid, "") if subcat_by_cid is not None else ""

            avg_res = None
            if "resolution_stats" in results:
                row = results["resolution_stats"]
                row = row[row["Cluster_ID"] == cid]
                if not row.empty:
                    avg_res = row["avg_resolution"].iloc[0]

            breach_rate = 0.0
            if "sla_stats" in results:
                row = results["sla_stats"]
                row = row[row["Cluster_ID"] == cid]
                if not row.empty:
                    breach_rate = row["breach_rate"].iloc[0]

            # Composite: normalized volume * avg_resolution * (1 + breach_rate/100)
            score = count
            if avg_res is not None and avg_res > 0:
                score *= avg_res
            score *= (1 + breach_rate / 100)

            impact_rows.append({
                "Cluster_ID": cid,
                "Subcategory": subcat,
                "Ticket_Count": count,
                "Avg_Resolution": avg_res,
                "SLA_Breach_Rate": breach_rate,
                "Impact_Score": round(score, 1),
            })

        ranking = pd.DataFrame(impact_rows).sort_values("Impact_Score", ascending=False)
        results["impact_ranking"] = ranking

        return results

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def generate_visualizations(self, problem_data=None, process_data=None, kpi_data=None):
        """Generate interactive Plotly figures for all analyses.

        Returns dict of Plotly Figure objects keyed by name.
        """
        import plotly.graph_objects as go
        import plotly.express as px

        figs = {}

        # 1. Volume bar chart
        if problem_data and "volume_stats" in problem_data:
            vol = problem_data["volume_stats"].head(self.top_n)
            label_col = "subcategory" if "subcategory" in vol.columns else "Cluster_ID"
            fig = px.bar(
                vol, x=label_col, y="count",
                title="Top Clusters by Ticket Volume",
                labels={label_col: "Cluster", "count": "Tickets"},
                color="count", color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_tickangle=-45)
            figs["volume"] = fig

        # 2. Trend line chart
        if problem_data and "trends" in problem_data:
            trends = problem_data["trends"]
            # Top 5 clusters only
            top_ids = trends.groupby("Cluster_ID")["count"].sum().nlargest(5).index
            filtered = trends[trends["Cluster_ID"].isin(top_ids)]
            fig = px.line(
                filtered, x="_period", y="count",
                color="Cluster_ID",
                title="Ticket Volume Trends (Top 5 Clusters)",
                labels={"_period": "Period", "count": "Tickets"},
            )
            figs["trends"] = fig

        # 3. Workload stacked bar
        if process_data and "workload" in process_data:
            wl = process_data["workload"].head(15)
            ag_col = [c for c in wl.columns if c not in ("ticket_count", "unique_clusters")][0]
            fig = px.bar(
                wl, x=ag_col, y="ticket_count",
                title="Ticket Workload by Assignment Group",
                labels={ag_col: "Assignment Group", "ticket_count": "Tickets"},
                color="unique_clusters",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickangle=-45)
            figs["workload"] = fig

        # 4. Bottleneck scatter (volume vs resolution time)
        if process_data and "bottlenecks" in process_data:
            bn = process_data["bottlenecks"]
            if not bn.empty and "avg_resolution" in bn.columns:
                label_col = "subcategory" if "subcategory" in bn.columns else "Cluster_ID"
                fig = px.scatter(
                    bn, x="count", y="avg_resolution",
                    size="count", color="is_bottleneck" if "is_bottleneck" in bn.columns else None,
                    hover_name=label_col,
                    title="Volume vs. Resolution Time (Bottleneck Detection)",
                    labels={"count": "Ticket Count", "avg_resolution": "Avg Resolution Time"},
                )
                figs["bottlenecks"] = fig

        # 5. KPI Impact Pareto chart
        if kpi_data and "impact_ranking" in kpi_data:
            rank = kpi_data["impact_ranking"].head(self.top_n)
            label_col = "Subcategory" if "Subcategory" in rank.columns else "Cluster_ID"
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rank[label_col], y=rank["Impact_Score"],
                name="Impact Score",
                marker_color="#e74c3c",
            ))
            # Cumulative line
            cumulative = rank["Impact_Score"].cumsum()
            total = rank["Impact_Score"].sum()
            if total > 0:
                cum_pct = (cumulative / total * 100).round(1)
                fig.add_trace(go.Scatter(
                    x=rank[label_col], y=cum_pct,
                    name="Cumulative %",
                    yaxis="y2",
                    mode="lines+markers",
                    marker_color="#3498db",
                ))
            fig.update_layout(
                title="KPI Impact Ranking (Pareto)",
                yaxis=dict(title="Impact Score"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                xaxis_tickangle=-45,
            )
            figs["pareto"] = fig

        # 6. Priority distribution (grouped bar)
        if kpi_data and "priority_dist" in kpi_data:
            pri = kpi_data["priority_dist"]
            # Only top clusters
            if "impact_ranking" in kpi_data:
                top_ids = kpi_data["impact_ranking"]["Cluster_ID"].head(self.top_n).tolist()
                pri = pri[pri["Cluster_ID"].isin(top_ids)]
            if not pri.empty:
                fig = px.bar(
                    pri, x="Cluster_ID", y="count", color="priority",
                    title="Priority Distribution by Cluster",
                    barmode="group",
                )
                figs["priority"] = fig

        # 7. Category treemap
        _cat_col = self._label_col(CLUSTER_CATEGORY_COLS)
        _sub_col_t = self._label_col(CLUSTER_SUBCATEGORY_COLS)
        if _cat_col and _sub_col_t:
            tree_data = (
                self.df[self.df["Cluster_ID"] != -1]
                .groupby([_cat_col, _sub_col_t])
                .size()
                .reset_index(name="count")
            )
            if not tree_data.empty:
                fig = px.treemap(
                    tree_data,
                    path=[_cat_col, _sub_col_t],
                    values="count",
                    title="Category Hierarchy (Ticket Volume)",
                    color="count",
                    color_continuous_scale="Viridis",
                )
                figs["treemap"] = fig

        return figs

    @staticmethod
    def open_figure_in_browser(fig):
        """Save a Plotly figure to a temp HTML file, open it in the browser, and
        schedule the file for deletion when the process exits."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="impact_"
        )
        tmp.close()
        # Embed plotly.js instead of linking the CDN: this tool is offline-first, and
        # a CDN reference renders as a blank page on an air-gapped machine.
        fig.write_html(tmp.name, include_plotlyjs=True)
        # Path.as_uri() builds a valid file:///C:/... URL on Windows;
        # a bare f"file://{path}" produces file://C:\... which browsers reject.
        webbrowser.open(Path(tmp.name).as_uri())
        atexit.register(_remove_quietly, tmp.name)
        return tmp.name
