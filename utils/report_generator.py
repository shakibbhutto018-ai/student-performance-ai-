"""
PDF and CSV Report Generator for StudentAI.
Uses ReportLab to generate a structured AI Academic Performance Report PDF
from real dataset statistics, ML metrics, risk analysis, and AI explanations.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def _sanitize_text(text: Any) -> str:
    """Safely cleans text for ReportLab XML Paragraph rendering."""
    if text is None:
        return "N/A"
    s = str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s


def generate_pdf_report(
    report_data: Dict[str, Any]
) -> bytes:
    """
    Generates a complete AI Academic Performance Report PDF in memory and returns raw PDF bytes.
    All statistics, ML metrics, predictions, and AI explanations come from actual application results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="StudentAI Academic Performance Report",
        author="StudentAI Analytics Platform",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=body_style,
        leftIndent=12,
        spaceAfter=3,
    )

    elements = []

    # Header
    project_title = report_data.get("project_title", "StudentAI — AI Academic Performance Report")
    generated_at = report_data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    user_email = report_data.get("user_email", "Authenticated User")

    elements.append(Paragraph(_sanitize_text(project_title), title_style))
    elements.append(
        Paragraph(
            f"<b>Generated:</b> {_sanitize_text(generated_at)} &nbsp;|&nbsp; <b>Prepared For:</b> {_sanitize_text(user_email)}",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

    def _build_kv_table(rows_data: List[List[str]]) -> Table:
        tbl = Table(rows_data, colWidths=[65 * mm, 109 * mm])
        tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        return tbl

    # 1. Dataset Summary & Cleaning Summary
    elements.append(Paragraph("1. Dataset &amp; Data Cleaning Summary", section_style))
    ds_info = report_data.get("dataset_summary", {})
    clean_info = report_data.get("cleaning_summary", {})
    ds_rows = [
        ["Metric / Property", "Actual Value"],
        ["Dataset Filename", _sanitize_text(ds_info.get("filename", "N/A"))],
        ["Total Rows (Raw -> Cleaned)", f"{ds_info.get('raw_rows', 'N/A')} -> {ds_info.get('cleaned_rows', ds_info.get('rows', 'N/A'))}"],
        ["Total Columns", _sanitize_text(ds_info.get("columns", "N/A"))],
        ["Duplicate Records Removed", _sanitize_text(clean_info.get("duplicates_removed", 0))],
        ["Missing Values Handled", _sanitize_text(clean_info.get("missing_values_handled", 0))],
        ["Outliers Adjusted", _sanitize_text(clean_info.get("outliers_adjusted", 0))],
        ["Numeric Columns", _sanitize_text(", ".join(ds_info.get("numeric_columns", [])) or "N/A")],
        ["Categorical Columns", _sanitize_text(", ".join(ds_info.get("categorical_columns", [])) or "None")],
    ]
    elements.append(_build_kv_table(ds_rows))
    elements.append(Spacer(1, 6))

    # 2. Important Statistics & Visual Analytics Summary
    elements.append(Paragraph("2. Key Academic Statistics &amp; Visual Analytics Summary", section_style))
    kpis = report_data.get("kpis", {})
    kpi_rows = [
        ["Analytical Indicator", "Computed Dataset Value"],
        ["Total Students Analyzed", _sanitize_text(kpis.get("total_students", "N/A"))],
        ["Average Academic Score", _sanitize_text(kpis.get("average_score", "N/A"))],
        ["Average Attendance (%)", f"{kpis.get('average_attendance', 'N/A')}%"],
        ["Overall Pass Rate (%)", f"{kpis.get('pass_rate', 'N/A')}%"],
        ["At-Risk Students Identified", _sanitize_text(kpis.get("at_risk_students", "N/A"))],
    ]
    elements.append(_build_kv_table(kpi_rows))

    analytics_notes = report_data.get("visual_analytics_summary", [])
    if analytics_notes:
        elements.append(Spacer(1, 4))
        for note in analytics_notes:
            elements.append(Paragraph(f"&bull; {_sanitize_text(note)}", bullet_style))

    # 3. Machine Learning Model & Evaluation Metrics
    elements.append(Paragraph("3. Machine Learning Model &amp; Evaluation Metrics", section_style))
    ml_info = report_data.get("ml_summary", {})
    if ml_info:
        ml_rows = [
            ["ML Configuration / Metric", "Result"],
            ["Model Algorithm", _sanitize_text(ml_info.get("model_name", "N/A"))],
            ["Task Type", _sanitize_text(ml_info.get("task_type", "N/A")).title()],
            ["Target Variable", _sanitize_text(ml_info.get("target_column", "N/A"))],
            ["Train / Test Split", f"{ml_info.get('train_size', 'N/A')} train rows / {ml_info.get('test_size', 'N/A')} test rows"],
        ]
        metrics_dict = ml_info.get("metrics", {})
        for m_k, m_v in metrics_dict.items():
            if isinstance(m_v, (int, float)):
                ml_rows.append([_sanitize_text(m_k), f"{m_v:.4f}"])
        elements.append(_build_kv_table(ml_rows))

        # Feature Importance
        feat_imp = ml_info.get("feature_importance", [])
        if feat_imp:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("<b>Top Influential Features (Model Importance):</b>", body_style))
            fi_rows = [["Feature Name", "Relative Importance Score"]]
            for item in feat_imp[:6]:
                fi_rows.append([
                    _sanitize_text(item.get("feature", "")),
                    f"{float(item.get('importance', 0.0)):.4f}",
                ])
            elements.append(_build_kv_table(fi_rows))
    else:
        elements.append(Paragraph("No Machine Learning model was trained prior to generating this report.", body_style))

    # 4. Student Performance Prediction & Risk Analysis
    elements.append(Paragraph("4. Student Prediction Results &amp; Risk Analysis", section_style))
    pred_info = report_data.get("prediction_summary", {})
    if pred_info:
        pred_rows = [
            ["Prediction Output", "Value"],
            ["Model Used", _sanitize_text(pred_info.get("model_name", "N/A"))],
            ["Predicted Output / Score", _sanitize_text(pred_info.get("predicted_display", "N/A"))],
            ["Performance Category", _sanitize_text(pred_info.get("performance_category", "N/A"))],
            ["Assigned Risk Level", _sanitize_text(pred_info.get("risk_level", "N/A"))],
        ]
        if pred_info.get("confidence_score") is not None:
            pred_rows.append(["Model Probability / Confidence", f"{float(pred_info['confidence_score']) * 100:.1f}%"])
        elements.append(_build_kv_table(pred_rows))

        reasons = pred_info.get("risk_reasons", [])
        if reasons:
            elements.append(Spacer(1, 4))
            elements.append(Paragraph("<b>Risk Assessment Rationale:</b>", body_style))
            for r in reasons:
                elements.append(Paragraph(f"&bull; {_sanitize_text(r)}", bullet_style))
    else:
        elements.append(Paragraph("No individual student prediction run was selected for this report.", body_style))

    # 5. Generative AI Explanation & Recommendations
    elements.append(Paragraph("5. AI-Generated Interpretation &amp; Academic Recommendations", section_style))
    ai_info = report_data.get("ai_explanation", {})
    if ai_info:
        summary_txt = ai_info.get("performance_summary") or ai_info.get("summary")
        if summary_txt:
            elements.append(Paragraph(f"<b>Executive Summary:</b> {_sanitize_text(summary_txt)}", body_style))

        why_txt = ai_info.get("why_model_predicted_this")
        if why_txt:
            elements.append(Paragraph(f"<b>Why the Model Predicted This:</b> {_sanitize_text(why_txt)}", body_style))

        for label, key in [
            ("Identified Strengths", "strengths"),
            ("Areas for Improvement", "areas_for_improvement"),
            ("Recommended Academic Actions", "recommended_actions"),
        ]:
            items = ai_info.get(key, [])
            if items and isinstance(items, list):
                elements.append(Paragraph(f"<b>{label}:</b>", body_style))
                for item in items:
                    elements.append(Paragraph(f"&bull; {_sanitize_text(item)}", bullet_style))
    else:
        elements.append(Paragraph("No AI insight narrative was attached to this report.", body_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_predictions_csv(predictions_list: List[Dict[str, Any]]) -> bytes:
    """
    Converts prediction history records into downloadable CSV bytes.
    """
    if not predictions_list:
        empty_df = pd.DataFrame(columns=[
            "created_at", "model_name", "task_type", "predicted_label",
            "predicted_score", "performance_category", "risk_level"
        ])
        return empty_df.to_csv(index=False).encode("utf-8")

    flat_rows = []
    for item in predictions_list:
        row = {
            "timestamp": item.get("created_at", datetime.now().isoformat()),
            "model_name": item.get("model_name", ""),
            "task_type": item.get("task_type", ""),
            "predicted_label": item.get("predicted_label", ""),
            "predicted_score": item.get("predicted_score", ""),
            "performance_category": item.get("performance_category", ""),
            "risk_level": item.get("risk_level", ""),
            "confidence_score": item.get("confidence_score", ""),
        }
        inputs = item.get("input_features", {})
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                row[f"input_{k}"] = v
        flat_rows.append(row)

    df_out = pd.DataFrame(flat_rows)
    return df_out.to_csv(index=False).encode("utf-8")
