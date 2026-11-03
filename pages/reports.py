"""
Reports Generation & Export Page (pages/reports.py).
Generates the comprehensive 'AI Academic Performance Report' PDF using ReportLab
and exports prediction history as CSV. Persists report metadata to Supabase PostgreSQL.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from services.supabase_service import SupabaseService
from utils.validators import smart_detect_columns
from utils.helpers import compute_dashboard_kpis
from utils.report_generator import generate_pdf_report, generate_predictions_csv


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">AI Academic Performance Reports &amp; Exports</div>
            <div class="saas-page-subtitle">Generate downloadable PDF executive reports and CSV prediction logs from real session &amp; database results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.info("Please upload or load a dataset on the **Dataset** page first to generate an academic report.")
        if st.button("Go to Dataset Page"):
            st.session_state["current_page"] = "Dataset"
            st.rerun()
        return

    user = st.session_state.get("auth_user") or {}
    mapping = st.session_state.get("column_mapping") or smart_detect_columns(df)
    role_to_col = mapping.get("role_to_col", {})

    trained_models = st.session_state.get("trained_models", {})
    active_key = st.session_state.get("active_model_key")
    best_model = trained_models.get(active_key) if active_key in trained_models else (
        list(trained_models.values())[-1] if trained_models else None
    )

    kpis = compute_dashboard_kpis(df, role_to_col, best_model)
    inspection = st.session_state.get("dataset_inspection") or {}
    clean_rep = st.session_state.get("cleaning_report") or {}
    latest_pred = st.session_state.get("latest_prediction") or {}
    latest_ai = st.session_state.get("latest_ai_insight") or {}

    # Compute visual analytics takeaways from actual dataset correlations
    visual_notes = []
    score_col = role_to_col.get("final_score")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if score_col and score_col in num_cols and len(num_cols) >= 2:
        corr_s = df[num_cols].corr()[score_col].drop(labels=[score_col], errors="ignore").dropna()
        for f_name, c_val in corr_s.abs().sort_values(ascending=False).head(3).items():
            visual_notes.append(f"Pearson correlation between {f_name} and {score_col}: r = {corr_s[f_name]:.3f}")
    if kpis.get("pass_rate") is not None:
        visual_notes.append(f"Dataset Pass Rate: {kpis['pass_rate']:.1f}% ({kpis.get('at_risk_students', 0)} at-risk students identified).")

    report_title_in = st.text_input(
        "Report Title",
        value="AI Academic Performance Report — StudentAI Analytics",
    )

    report_payload = {
        "project_title": report_title_in,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_email": user.get("email", "Authenticated Analyst"),
        "dataset_summary": {
            "filename": st.session_state.get("dataset_filename", "dataset.csv"),
            "raw_rows": clean_rep.get("rows_before", len(df)),
            "cleaned_rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": inspection.get("numeric_columns", num_cols),
            "categorical_columns": inspection.get("categorical_columns", []),
        },
        "cleaning_summary": clean_rep,
        "kpis": kpis,
        "visual_analytics_summary": visual_notes,
        "ml_summary": best_model if best_model else {},
        "prediction_summary": latest_pred,
        "ai_explanation": latest_ai,
    }

    c_pdf, c_csv = st.columns(2)

    with c_pdf:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("1. Download PDF Executive Report")
        st.write(
            "Compiles dataset summary, cleaning logs, statistical KPIs, trained ML metrics, "
            "feature importances, latest student risk assessment, and AI recommendations into a formatted PDF."
        )
        try:
            pdf_bytes = generate_pdf_report(report_payload)
            if st.download_button(
                label="Download PDF (AI Academic Performance Report)",
                data=pdf_bytes,
                file_name=f"StudentAI_Academic_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            ):
                # Save report metadata to Supabase
                if sb.is_configured() and user.get("access_token"):
                    sb.save_report_metadata(
                        user_id=user["id"],
                        access_token=user["access_token"],
                        dataset_id=st.session_state.get("dataset_id"),
                        prediction_id=st.session_state.get("latest_prediction_id"),
                        report_title=report_title_in,
                        model_used=best_model.get("model_name", "N/A") if best_model else "N/A",
                        risk_level=latest_pred.get("risk_level", "N/A"),
                        summary_snapshot=kpis,
                    )
        except Exception as exc:
            st.error(f"PDF generation error: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_csv:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("2. Download Prediction History CSV")
        preds_list = st.session_state.get("session_predictions", [])
        st.write(
            f"Export all **{len(preds_list)}** student performance prediction record(s) from your account history as a structured CSV file."
        )
        csv_bytes = generate_predictions_csv(preds_list)
        st.download_button(
            label="Download Prediction CSV",
            data=csv_bytes,
            file_name=f"StudentAI_Predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Report Content Preview")
    with st.expander("Inspect Live Data Included in Report", expanded=True):
        st.json({
            "Title": report_payload["project_title"],
            "Generated_At": report_payload["generated_at"],
            "Dataset": report_payload["dataset_summary"],
            "KPIs": report_payload["kpis"],
            "ML_Model": best_model.get("model_name") if best_model else "None trained yet",
            "Latest_Prediction": latest_pred.get("predicted_display", "None run yet"),
            "Risk_Level": latest_pred.get("risk_level", "N/A"),
        })
