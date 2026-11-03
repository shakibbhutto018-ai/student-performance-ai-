"""
Dashboard Page (pages/dashboard.py) for StudentAI.
Displays 6 real KPI cards (Total Students, Average Score, Average Attendance,
Pass Rate, At-Risk Students, Best Model Score) and 6 real interactive Plotly charts.
Shows a proper empty state with 'Upload Dataset' and 'Download Sample Dataset' when no dataset is loaded.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from services.supabase_service import SupabaseService
from utils.helpers import compute_dashboard_kpis
from utils.validators import smart_detect_columns
from utils.data_cleaner import inspect_dataset, clean_dataset


def _get_sample_csv_bytes() -> bytes:
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_students.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            return f.read()
    return b""


def _load_sample_into_session(sb: SupabaseService) -> None:
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_students.csv")
    if os.path.exists(sample_path):
        raw_df = pd.read_csv(sample_path)
        cleaned_df, clean_rep = clean_dataset(raw_df)
        mapping = smart_detect_columns(cleaned_df)
        inspection = inspect_dataset(raw_df, "sample_students.csv")

        st.session_state["raw_df"] = raw_df
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["dataset_filename"] = "sample_students.csv"
        st.session_state["dataset_inspection"] = inspection
        st.session_state["cleaning_report"] = clean_rep
        st.session_state["column_mapping"] = mapping

        user = st.session_state.get("auth_user") or {}
        if sb.is_configured() and user.get("access_token"):
            ok, row, _ = sb.save_dataset_metadata(
                user_id=user["id"],
                access_token=user["access_token"],
                filename="sample_students.csv",
                raw_rows=len(raw_df),
                cleaned_rows=len(cleaned_df),
                column_count=len(cleaned_df.columns),
                missing_values_count=inspection["missing_values"],
                duplicates_removed=clean_rep["duplicates_removed"],
                numeric_columns=inspection["numeric_columns"],
                categorical_columns=inspection["categorical_columns"],
                detected_mapping=mapping["role_to_col"],
                summary_stats={"rows_after_cleaning": len(cleaned_df)},
            )
            if ok and row and isinstance(row, dict):
                st.session_state["dataset_id"] = row.get("id")


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Academic Intelligence Dashboard</div>
            <div class="saas-page-subtitle">Real-time student performance metrics, risk distribution, and machine learning insights</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.markdown(
            """
            <div class="empty-state-box" style="padding: 2.25rem 1.5rem; border: 2px dashed #CBD5E1; border-radius: 16px; background: #FFFFFF; text-align: center; margin: 1rem 0 1.5rem 0;">
                <div style="font-size: 46px; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.35rem;">No Dataset Loaded Yet</div>
                <div style="font-size: 0.94rem; color: #64748B; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                    StudentAI operates strictly on real academic records. Click <strong>"Load Sample Dataset"</strong> below to explore 300+ student records with 6 KPI cards, 10 interactive charts, and trained ML models immediately.
                </div>
                <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1.75rem; flex-wrap: wrap;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.9rem 1.25rem; text-align: left; width: 190px;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB;">STEP 1</div>
                        <div style="font-weight: 600; font-size: 0.9rem; color: #1E293B; margin-top: 2px;">Load Dataset</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">Auto-cleaning, outlier handling &amp; 6 KPIs</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.9rem 1.25rem; text-align: left; width: 190px;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB;">STEP 2</div>
                        <div style="font-weight: 600; font-size: 0.9rem; color: #1E293B; margin-top: 2px;">Train ML Models</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">Random Forest &amp; Linear/Logistic models</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.9rem 1.25rem; text-align: left; width: 190px;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB;">STEP 3</div>
                        <div style="font-weight: 600; font-size: 0.9rem; color: #1E293B; margin-top: 2px;">Predict &amp; Risk Alert</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">Predict final score &amp; get AI explanation</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([1.25, 1, 1])
        with c1:
            if st.button("⚡ Load Sample Dataset (Instant Demo)", type="primary", use_container_width=True):
                _load_sample_into_session(sb)
                st.rerun()
        with c2:
            if st.button("📁 Upload Your Own CSV", use_container_width=True):
                st.session_state["current_page"] = "Dataset"
                st.rerun()
        with c3:
            sample_bytes = _get_sample_csv_bytes()
            if sample_bytes:
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=sample_bytes,
                    file_name="sample_students.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        return

    mapping = st.session_state.get("column_mapping") or smart_detect_columns(df)
    role_to_col = mapping.get("role_to_col", {})

    # Find best trained model if any
    trained_models = st.session_state.get("trained_models", {})
    best_model = None
    if trained_models:
        active_key = st.session_state.get("active_model_key")
        if active_key and active_key in trained_models:
            best_model = trained_models[active_key]
        else:
            best_model = list(trained_models.values())[-1]

    kpis = compute_dashboard_kpis(df, role_to_col, best_model)

    # Render 6 KPI Cards
    cols = st.columns(6)
    cards_spec = [
        ("Total Students", f"{kpis['total_students']:,}" if kpis["total_students"] is not None else "N/A", f"Dataset: {st.session_state.get('dataset_filename', 'CSV')}"),
        ("Average Score", f"{kpis['average_score']:.1f}" if kpis["average_score"] is not None else "N/A", role_to_col.get("final_score", "Score column")),
        ("Average Attendance", f"{kpis['average_attendance']:.1f}%" if kpis["average_attendance"] is not None else "N/A", role_to_col.get("attendance", "Attendance column")),
        ("Pass Rate", f"{kpis['pass_rate']:.1f}%" if kpis["pass_rate"] is not None else "N/A", "Passing benchmark"),
        ("At-Risk Students", f"{kpis['at_risk_students']:,}" if kpis["at_risk_students"] is not None else "N/A", "Score < 50 or Attendance < 65%"),
        ("Best Model Score", kpis["best_model_score"] or "Not Trained", kpis["best_model_label"]),
    ]

    for idx, (label, val, sub) in enumerate(cards_spec):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    score_col = role_to_col.get("final_score")
    att_col = role_to_col.get("attendance")
    study_col = role_to_col.get("study_hours")
    res_col = role_to_col.get("result")
    date_col = role_to_col.get("date")

    # Row 1 Charts: Performance Distribution & Pass/Fail Distribution
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if score_col and score_col in df.columns:
            fig_dist = px.histogram(
                df,
                x=score_col,
                nbins=22,
                color=res_col if (res_col and res_col in df.columns) else None,
                title=f"Performance Distribution ({score_col})",
                color_discrete_sequence=["#2563EB", "#EF4444", "#10B981"],
                template="plotly_white",
            )
            fig_dist.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No final score column mapped for Performance Distribution chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r1_c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if res_col and res_col in df.columns:
            counts = df[res_col].value_counts().reset_index()
            counts.columns = [res_col, "Count"]
            fig_pf = px.pie(
                counts,
                names=res_col,
                values="Count",
                hole=0.52,
                title=f"Pass / Fail Distribution ({res_col})",
                color_discrete_sequence=["#10B981", "#EF4444", "#3B82F6"],
                template="plotly_white",
            )
            fig_pf.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_pf, use_container_width=True)
        else:
            st.info("No result/pass-fail column available in the current dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2 Charts: Attendance vs Performance & Study Hours vs Performance
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if att_col and score_col and att_col in df.columns and score_col in df.columns:
            fig_att = px.scatter(
                df,
                x=att_col,
                y=score_col,
                color=res_col if (res_col and res_col in df.columns) else None,
                title=f"Attendance vs Performance ({att_col} vs {score_col})",
                opacity=0.8,
                color_discrete_sequence=["#2563EB", "#EF4444"],
                template="plotly_white",
            )
            fig_att.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_att, use_container_width=True)
        else:
            st.info("Both Attendance and Score columns are required to display Attendance vs Performance.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r2_c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if study_col and score_col and study_col in df.columns and score_col in df.columns:
            fig_sh = px.scatter(
                df,
                x=study_col,
                y=score_col,
                color=res_col if (res_col and res_col in df.columns) else None,
                title=f"Study Hours vs Performance ({study_col} vs {score_col})",
                opacity=0.8,
                color_discrete_sequence=["#0EA5E9", "#F97316"],
                template="plotly_white",
            )
            fig_sh.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_sh, use_container_width=True)
        else:
            st.info("Both Study Hours and Score columns are required to display Study Hours vs Performance.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3 Charts: Performance Trends (when date exists) & Feature Importance (after model training)
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if date_col and date_col in df.columns and score_col and score_col in df.columns:
            temp_df = df[[date_col, score_col]].copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
            temp_df = temp_df.dropna(subset=[date_col, score_col]).sort_values(date_col)
            if len(temp_df) >= 3:
                temp_df["Period"] = temp_df[date_col].dt.to_period("M").astype(str)
                trend_df = temp_df.groupby("Period", as_index=False)[score_col].mean()
                fig_trend = px.line(
                    trend_df,
                    x="Period",
                    y=score_col,
                    markers=True,
                    title=f"Performance Trend Over Time (Monthly Avg {score_col})",
                    template="plotly_white",
                )
                fig_trend.update_traces(line_color="#2563EB", line_width=3)
                fig_trend.update_layout(height=330, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Insufficient valid date entries for Performance Trend visualization.")
        else:
            st.info("No date/time column detected in this dataset for Performance Trends.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r3_c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if best_model and best_model.get("feature_importance"):
            fi_df = pd.DataFrame(best_model["feature_importance"]).sort_values("importance", ascending=True)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature Importance ({best_model['model_name']})",
                color="importance",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_fi.update_layout(height=330, margin=dict(l=20, r=20, t=45, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature Importance will appear here after you train a model on the **ML Models** page.")
        st.markdown("</div>", unsafe_allow_html=True)
