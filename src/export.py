"""
Unified export utilities for KBA articles, SOPs, and audit reports.
Supports Word (.docx), Excel (.xlsx), and HTML output.
"""

from src.logger import get_logger

logger = get_logger()


def export_kba_to_docx(articles, output_path):
    """Export KBA articles to a Word document.

    Args:
        articles: List of article dicts from KBAGenerator.
        output_path: Output .docx file path.
    """
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)

    doc.add_heading("Knowledge Base Articles", level=0)
    doc.add_paragraph(f"Total articles: {len(articles)}")
    doc.add_page_break()

    for i, art in enumerate(articles):
        doc.add_heading(art.get("title", f"Article {i+1}"), level=1)

        meta = doc.add_paragraph()
        meta.add_run("Category: ").bold = True
        meta.add_run(art.get("category", ""))
        meta.add_run("  |  Subcategory: ").bold = True
        meta.add_run(art.get("subcategory", ""))

        if art.get("keywords"):
            kw_para = doc.add_paragraph()
            kw_para.add_run("Keywords: ").bold = True
            kw_para.add_run(", ".join(str(k) for k in art["keywords"]))

        for section, heading in [
            ("symptoms", "Symptoms"),
            ("cause", "Cause"),
            ("resolution", "Resolution"),
            ("prevention", "Prevention"),
        ]:
            content = art.get(section, "").strip()
            if content:
                doc.add_heading(heading, level=2)
                doc.add_paragraph(content)

        if i < len(articles) - 1:
            doc.add_page_break()

    doc.save(output_path)
    logger.info(f"KBA exported to {output_path}")


def export_kba_to_excel(articles, output_path):
    """Export KBA articles to an Excel spreadsheet.

    Args:
        articles: List of article dicts from KBAGenerator.
        output_path: Output .xlsx file path.
    """
    import pandas as pd

    rows = []
    for art in articles:
        rows.append({
            "Cluster_ID": art.get("cluster_id", ""),
            "Title": art.get("title", ""),
            "Category": art.get("category", ""),
            "Subcategory": art.get("subcategory", ""),
            "Keywords": ", ".join(str(k) for k in art.get("keywords", [])),
            "Symptoms": art.get("symptoms", ""),
            "Cause": art.get("cause", ""),
            "Resolution": art.get("resolution", ""),
            "Prevention": art.get("prevention", ""),
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, sheet_name="KBA Articles")
    logger.info(f"KBA exported to {output_path}")


def export_sop_to_docx(sops, output_path):
    """Export SOPs to a Word document.

    Args:
        sops: List of SOP dicts from SOPGenerator.
        output_path: Output .docx file path.
    """
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)

    doc.add_heading("Standard Operating Procedures", level=0)
    doc.add_paragraph(f"Total SOPs: {len(sops)}")
    doc.add_page_break()

    for i, sop in enumerate(sops):
        doc.add_heading(sop.get("title", f"SOP {i+1}"), level=1)

        for section, heading in [
            ("purpose", "Purpose"),
            ("scope", "Scope"),
            ("prerequisites", "Prerequisites"),
            ("procedure", "Procedure"),
            ("escalation", "Escalation"),
            ("quality_checks", "Quality Checks"),
        ]:
            content = sop.get(section, "").strip()
            if content:
                doc.add_heading(heading, level=2)
                doc.add_paragraph(content)

        if sop.get("subcategories"):
            doc.add_heading("Related Subcategories", level=2)
            for sub in sop["subcategories"]:
                if sub:
                    doc.add_paragraph(sub, style="List Bullet")

        if i < len(sops) - 1:
            doc.add_page_break()

    doc.save(output_path)
    logger.info(f"SOPs exported to {output_path}")


def export_sop_to_excel(sops, output_path):
    """Export SOPs to an Excel spreadsheet."""
    import pandas as pd

    rows = []
    for sop in sops:
        rows.append({
            "Category": sop.get("category", ""),
            "Title": sop.get("title", ""),
            "Purpose": sop.get("purpose", ""),
            "Scope": sop.get("scope", ""),
            "Prerequisites": sop.get("prerequisites", ""),
            "Procedure": sop.get("procedure", ""),
            "Escalation": sop.get("escalation", ""),
            "Quality Checks": sop.get("quality_checks", ""),
            "Subcategories": ", ".join(s for s in sop.get("subcategories", []) if s),
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, sheet_name="SOPs")
    logger.info(f"SOPs exported to {output_path}")


def export_audit_to_excel(audit_df, summary, output_path):
    """Export audit report to a multi-sheet Excel file.

    Args:
        audit_df: DataFrame with quality score columns.
        summary: Summary dict from TicketQualityAuditor.generate_report().
        output_path: Output .xlsx file path.
    """
    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Full ticket scores
        audit_df.to_excel(writer, sheet_name="Ticket Scores", index=False)

        # Sheet 2: Summary
        summary_rows = [
            {"Metric": "Overall Quality Score", "Value": summary.get("overall_score", "")},
            {"Metric": "Completeness Average", "Value": summary.get("completeness_avg", "")},
            {"Metric": "Categorization Average", "Value": summary.get("categorization_avg", "")},
            {"Metric": "Resolution Quality Average", "Value": summary.get("resolution_avg", "")},
            {"Metric": "Total Tickets", "Value": summary.get("total_tickets", "")},
            {"Metric": "Tickets Scored", "Value": summary.get("tickets_scored", "")},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 3: Cluster breakdown
        clusters = summary.get("clusters", {})
        if clusters:
            cluster_rows = []
            for cid, info in sorted(clusters.items()):
                cluster_rows.append({
                    "Cluster_ID": cid,
                    "Subcategory": info.get("subcategory", ""),
                    "Ticket Count": info.get("count", 0),
                    "Avg Quality Score": info.get("avg_quality", ""),
                })
            pd.DataFrame(cluster_rows).to_excel(
                writer, sheet_name="Cluster Breakdown", index=False
            )

    logger.info(f"Audit report exported to {output_path}")


def export_category_pivot_to_excel(df, output_path):
    """Export the Category → Subcategory ticket pivot to a two-sheet Excel file.

    Args:
        df: results DataFrame carrying "Repetitive Category" /
            "Repetitive Subcategory" columns.
        output_path: Output .xlsx file path.
    """
    import pandas as pd
    from src.pivot import compute_category_pivot, pivot_to_rows, pivot_to_category_totals

    pivot = compute_category_pivot(df)
    rows = pivot_to_rows(pivot) or [{"Category": "", "Subcategory": "", "Count": 0, "% of Total": 0}]
    totals = pivot_to_category_totals(pivot) or [{"Category": "", "Count": 0, "% of Total": 0}]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Category Pivot", index=False)
        pd.DataFrame(totals).to_excel(writer, sheet_name="Category Totals", index=False)

    logger.info(f"Category pivot exported to {output_path}")


def export_impact_to_excel(analysis_results, output_path):
    """Export impact analysis results to a multi-sheet Excel file.

    Args:
        analysis_results: Dict with problem_clusters, business_process, kpi keys.
        output_path: Output .xlsx file path.
    """
    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Problem cluster volume
        vol = analysis_results.get("problem_clusters", {}).get("volume_stats")
        if vol is not None and hasattr(vol, "to_excel"):
            vol.to_excel(writer, sheet_name="Cluster Volume")

        # KPI impact ranking
        ranking = analysis_results.get("kpi", {}).get("impact_ranking")
        if ranking is not None and hasattr(ranking, "to_excel"):
            ranking.to_excel(writer, sheet_name="KPI Impact Ranking")

        # Business process
        workload = analysis_results.get("business_process", {}).get("workload")
        if workload is not None and hasattr(workload, "to_excel"):
            workload.to_excel(writer, sheet_name="Workload Distribution")

        # Bottlenecks
        bottlenecks = analysis_results.get("business_process", {}).get("bottlenecks")
        if bottlenecks is not None and hasattr(bottlenecks, "to_excel"):
            bottlenecks.to_excel(writer, sheet_name="Bottlenecks", index=False)

    logger.info(f"Impact analysis exported to {output_path}")


def export_disposition_to_excel(results, output_path, summary=None):
    """Export automation-opportunity (disposition) analysis to a multi-sheet Excel.

    Args:
        results: List of dicts from DispositionAnalyzer.generate_all().
        output_path: Output .xlsx file path.
        summary: Optional dict from DispositionAnalyzer.summarize(); recomputed if None.
    """
    import pandas as pd
    from src.disposition import DispositionAnalyzer

    rows = []
    for r in results:
        rows.append({
            "Cluster_ID": r.get("cluster_id", ""),
            "Resolution Pattern": r.get("resolution_pattern", ""),
            "Disposition": r.get("disposition", ""),
            "Tickets": r.get("volume", ""),
            "Avg Handling (h)": r.get("effort_hours", ""),
            "Total Effort (h) / ROI": r.get("roi", ""),
            "Reopen %": r.get("reopen_pct", ""),
            "Avg Reassignments": r.get("reassign_avg", ""),
            "Automatability": r.get("automatability", ""),
            "Dominant Close Code": r.get("dominant_close_code", ""),
            "Root Cause": r.get("root_cause", ""),
            "Resolution": r.get("resolution", ""),
            "Rationale": r.get("rationale", ""),
            "Recommendation": r.get("recommendation", ""),
            "Keywords": ", ".join(str(k) for k in r.get("keywords", [])),
        })

    if summary is None:
        summary = DispositionAnalyzer.summarize(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Opportunities", index=False)
        summary_rows = [
            {
                "Disposition": disp,
                "Clusters": info.get("clusters", 0),
                "Tickets": info.get("tickets", 0),
                "Total Effort (h)": info.get("roi_hours", 0),
            }
            for disp, info in summary.items()
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary by Disposition", index=False)

    logger.info(f"Disposition analysis exported to {output_path}")
