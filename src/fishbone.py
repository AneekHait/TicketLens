"""
Fishbone (Ishikawa) diagram generator.
Classifies cluster keywords into standard cause categories and renders
interactive Plotly diagrams.
"""

import webbrowser
import tempfile
import os
import re
import atexit
from pathlib import Path
from src.logger import get_logger

logger = get_logger()


def _remove_quietly(path):
    """Delete a temp file, ignoring the case where it is already gone."""
    try:
        os.remove(path)
    except OSError:
        pass

# Standard Ishikawa cause categories with keyword patterns
# Ordered dict — more specific categories checked first to avoid
# ambiguous keywords being swallowed by broader categories.
ISHIKAWA_CATEGORIES = {
    "People": [
        "user", "team", "training", "staff", "knowledge", "skill",
        "employee", "agent", "analyst", "manager", "human", "person",
        "awareness", "experience", "communication",
    ],
    "Data": [
        "data", "report", "access", "permission", "database",
        "record", "log", "file", "backup", "restore", "migration",
        "integrity", "quality", "accuracy",
    ],
    "Management": [
        "priority", "resource", "capacity", "planning", "budget",
        "leadership", "decision", "oversight", "compliance", "audit",
        "reporting", "governance", "strategy",
    ],
    "Environment": [
        "office", "remote", "vpn", "location", "infrastructure",
        "workstation", "facility", "cloud", "datacenter", "site",
        "building", "floor", "region",
    ],
    "Process": [
        "workflow", "procedure", "approval", "escalation", "sla",
        "process", "policy", "standard", "guideline", "handoff",
        "routing", "queue", "assignment", "backlog",
    ],
    "Technology": [
        "system", "software", "hardware", "network", "server",
        "application", "tool", "platform", "api",
        "integration", "configuration", "update", "patch", "bug",
        "crash", "error", "timeout", "latency", "outage",
    ],
}


def classify_keyword(keyword):
    """Classify a single keyword into an Ishikawa category.

    Matching is by whole word, and an exact match wins over a partial one.

    The previous rule was ``pattern in kw_lower or kw_lower in pattern`` over the dict in
    insertion order, which misfiled real keywords: "backlog" is listed under Process but
    Data's "log" matched it first; "reporting" is listed under Management but Data's
    "report" matched first. The second clause (keyword is a substring of the pattern) was
    worse still — it filed "man" as People via "manager", "use" via "user", "cat" via
    "communication".

    Args:
        keyword: String keyword to classify.

    Returns:
        Category name or "Technology" as default.
    """
    kw_lower = str(keyword).lower().strip()
    if not kw_lower:
        return "Technology"

    # Pass 1: the keyword *is* one of the patterns. Wins outright, so an explicitly
    # listed keyword always lands in the category that lists it.
    for cat, patterns in ISHIKAWA_CATEGORIES.items():
        if kw_lower in patterns:
            return cat

    # Pass 2: a pattern appears as a whole word inside a multi-word keyword
    # ("vpn tunnel" -> Environment). Longest pattern first, so the most specific match
    # wins regardless of which category happens to be declared first.
    words = set(re.findall(r"[a-z0-9]+", kw_lower))
    best = None
    for cat, patterns in ISHIKAWA_CATEGORIES.items():
        for pattern in patterns:
            if pattern in words and (best is None or len(pattern) > best[0]):
                best = (len(pattern), cat)
    if best:
        return best[1]

    return "Technology"


class FishboneGenerator:
    """Generates fishbone (Ishikawa) diagrams from cluster data.

    Classification is purely deterministic keyword matching — no LLM is involved. The
    constructor used to accept an ``llm`` argument, store it and never use it, and the
    GUI took the shared LLM lease before calling this, blocking every other AI feature
    for an operation that never touched the model.

    Args:
        cluster_data: Dict from TicketClusterer.get_cluster_data().
    """

    def __init__(self, cluster_data):
        self.cluster_data = cluster_data

    def classify_causes(self, cluster_id):
        """Classify cluster keywords into Ishikawa categories.

        Args:
            cluster_id: Cluster/topic ID, or a category name string.

        Returns:
            Dict: {category_name: [keywords]}.
        """
        # Collect keywords
        if isinstance(cluster_id, str):
            # Aggregate keywords across clusters in this category
            keywords = []
            for cid, cdata in self.cluster_data.items():
                if cid == -1:
                    continue
                if cdata.get("category", "") == cluster_id:
                    keywords.extend(cdata.get("keywords", []))
        else:
            cdata = self.cluster_data.get(cluster_id, {})
            keywords = cdata.get("keywords", [])

        # Classify each keyword
        classified = {cat: [] for cat in ISHIKAWA_CATEGORIES}
        seen = set()
        for kw in keywords:
            kw_str = str(kw).strip()
            if not kw_str or kw_str in seen:
                continue
            seen.add(kw_str)
            cat = classify_keyword(kw_str)
            classified[cat].append(kw_str)

        # Remove empty categories
        return {k: v for k, v in classified.items() if v}

    def generate_diagram(self, cluster_id_or_category, problem_label=None):
        """Generate a Plotly fishbone diagram figure.

        Args:
            cluster_id_or_category: Cluster ID (int) or category name (str).
            problem_label: Label for the fish head. Auto-derived if None.

        Returns:
            plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        causes = self.classify_causes(cluster_id_or_category)

        # Determine problem label
        if problem_label is None:
            if isinstance(cluster_id_or_category, str):
                problem_label = cluster_id_or_category
            else:
                cdata = self.cluster_data.get(cluster_id_or_category, {})
                problem_label = cdata.get("subcategory", f"Cluster {cluster_id_or_category}")

        # Layout constants
        spine_y = 0
        spine_x_start = -1.0
        spine_x_end = 6.0
        head_x = spine_x_end + 0.5

        categories = list(causes.keys())
        n_cats = len(categories)
        if n_cats == 0:
            categories = list(ISHIKAWA_CATEGORIES.keys())
            causes = {c: ["(no data)"] for c in categories}
            n_cats = len(categories)

        # Position categories: upper and lower, evenly spaced
        upper = categories[:((n_cats + 1) // 2)]
        lower = categories[((n_cats + 1) // 2):]

        fig = go.Figure()

        # Draw spine
        fig.add_trace(go.Scatter(
            x=[spine_x_start, spine_x_end],
            y=[spine_y, spine_y],
            mode="lines",
            line=dict(color="#2c3e50", width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Fish head
        fig.add_annotation(
            x=head_x, y=spine_y,
            text=f"<b>{problem_label}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor="#e74c3c",
            ax=-40, ay=0,
            font=dict(size=14, color="#e74c3c"),
            bgcolor="white",
            bordercolor="#e74c3c",
            borderwidth=2,
            borderpad=6,
        )

        colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]

        def _draw_branch(cat_name, keywords, x_pos, y_dir, color_idx):
            """Draw a single fishbone branch with sub-branches."""
            color = colors[color_idx % len(colors)]
            branch_y = y_dir * 2.0

            # Main branch line
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[spine_y, branch_y],
                mode="lines",
                line=dict(color=color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Category label
            fig.add_annotation(
                x=x_pos, y=branch_y + y_dir * 0.3,
                text=f"<b>{cat_name}</b>",
                showarrow=False,
                font=dict(size=12, color=color),
            )

            # Sub-branches (keywords)
            for j, kw in enumerate(keywords[:5]):
                sub_y = spine_y + y_dir * (0.4 + j * 0.3)
                sub_x_offset = 0.6 if y_dir > 0 else -0.6
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos + sub_x_offset],
                    y=[sub_y, sub_y + y_dir * 0.15],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_annotation(
                    x=x_pos + sub_x_offset * 1.1,
                    y=sub_y + y_dir * 0.15,
                    text=kw,
                    showarrow=False,
                    font=dict(size=9, color="#555"),
                    xanchor="left" if sub_x_offset > 0 else "right",
                )

        # Draw upper branches
        for i, cat in enumerate(upper):
            x_pos = 0.5 + i * (4.5 / max(len(upper), 1))
            _draw_branch(cat, causes.get(cat, []), x_pos, 1, i)

        # Draw lower branches
        for i, cat in enumerate(lower):
            x_pos = 0.5 + i * (4.5 / max(len(lower), 1))
            _draw_branch(cat, causes.get(cat, []), x_pos, -1, len(upper) + i)

        fig.update_layout(
            title=dict(
                text=f"Fishbone Diagram: {problem_label}",
                font=dict(size=16),
            ),
            xaxis=dict(visible=False, range=[-1.5, head_x + 1]),
            yaxis=dict(visible=False, range=[-3.5, 3.5]),
            plot_bgcolor="white",
            height=600,
            width=1000,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        return fig

    def export_html(self, fig, output_path):
        """Save a Plotly figure as interactive HTML.

        plotly.js is embedded rather than pulled from the CDN: this is an
        offline-first, air-gapped-capable tool, and with include_plotlyjs="cdn"
        every diagram opened without internet access rendered as a blank page.
        Inlining costs ~3 MB per file but makes the output self-contained.
        """
        fig.write_html(output_path, include_plotlyjs=True)
        logger.info(f"Fishbone diagram exported to {output_path}")

    def export_image(self, fig, output_path):
        """Save a Plotly figure as PNG."""
        fig.write_image(output_path, engine="kaleido")
        logger.info(f"Fishbone image exported to {output_path}")

    def open_in_browser(self, fig):
        """Save to a temp HTML file, open it in the default browser, and
        schedule the file for deletion when the process exits."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="fishbone_"
        )
        tmp.close()
        self.export_html(fig, tmp.name)
        # Path.as_uri() builds a valid file:///C:/... URL on Windows;
        # a bare f"file://{path}" produces file://C:\... which browsers reject.
        webbrowser.open(Path(tmp.name).as_uri())
        atexit.register(_remove_quietly, tmp.name)
        return tmp.name
