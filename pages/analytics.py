"""
Visual Analytics Page (pages/analytics.py) for StudentAI.
Renders up to 10 meaningful interactive Plotly visualizations strictly from columns
that actually exist in the uploaded dataset. Never creates fake charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from services.supabase_service import SupabaseService
from utils.validators import smart_detect_columns


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Deep Academic Visual Analytics</div>
            <div class="saas-page-subtitle">10-point statistical exploration of score distributions, attendance correlations, assessments, and temporal trends</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.info("No dataset available for visual analytics. Please upload or load a dataset on the **Dataset** page.")
        if st.button("Go to Dataset Page"):
            st.session_state["current_page"] = "Dataset"
            st.rerun()
        return

    mapping = st.session_state.get("column_mapping") or smart_detect_columns(df)
    role_to_col = mapping.get("role_to_col", {})

    score_col = role_to_col.get("final_score")
    att_col = role_to_col.get("attendance")
    study_col = role_to_col.get("study_hours")
    assign_col = role_to_col.get("assignment_score")
    intern_col = role_to_col.get("internal_marks")
    res_col = role_to_col.get("result")
    course_col = role_to_col.get("course") or role_to_col.get("subject") or role_to_col.get("semester")
    date_col = role_to_col.get("date")
    color_col = res_col if (res_col and res_col in df.columns) else None

    # 1. Score Distribution & 2. Attendance Distribution
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if score_col and score_col in df.columns:
            fig1 = px.histogram(
                df,
                x=score_col,
                nbins=24,
                color=color_col,
                marginal="box",
                title=f"1. Score Distribution ({score_col})",
                color_discrete_sequence=["#2563EB", "#EF4444"],
                template="plotly_white",
            )
            fig1.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("1. Score Distribution: No score column mapped in current dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if att_col and att_col in df.columns:
            fig2 = px.histogram(
                df,
                x=att_col,
                nbins=24,
                color=color_col,
                marginal="box",
                title=f"2. Attendance Distribution ({att_col})",
                color_discrete_sequence=["#0EA5E9", "#F97316"],
                template="plotly_white",
            )
            fig2.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("2. Attendance Distribution: No attendance column detected in current dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Attendance vs Score & 4. Study Hours vs Score
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if att_col and score_col and att_col in df.columns and score_col in df.columns:
            fig3 = px.scatter(
                df,
                x=att_col,
                y=score_col,
                color=color_col,
                title=f"3. Attendance vs Score ({att_col} vs {score_col})",
                opacity=0.8,
                color_discrete_sequence=["#2563EB", "#EF4444"],
                template="plotly_white",
            )
            fig3.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("3. Attendance vs Score: Requires both attendance and score columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if study_col and score_col and study_col in df.columns and score_col in df.columns:
            fig4 = px.scatter(
                df,
                x=study_col,
                y=score_col,
                color=color_col,
                title=f"4. Study Hours vs Score ({study_col} vs {score_col})",
                opacity=0.8,
                color_discrete_sequence=["#10B981", "#EF4444"],
                template="plotly_white",
            )
            fig4.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("4. Study Hours vs Score: Requires both study hours and score columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Assignment Score Analysis & 6. Internal Marks Analysis
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if assign_col and assign_col in df.columns:
            if score_col and score_col in df.columns:
                fig5 = px.scatter(
                    df,
                    x=assign_col,
                    y=score_col,
                    color=color_col,
                    title=f"5. Assignment Score vs Final Performance ({assign_col})",
                    color_discrete_sequence=["#6366F1", "#EF4444"],
                    template="plotly_white",
                )
            else:
                fig5 = px.histogram(df, x=assign_col, title=f"5. Assignment Score Analysis ({assign_col})", template="plotly_white")
            fig5.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("5. Assignment Score Analysis: No assignment score column detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if intern_col and intern_col in df.columns:
            if score_col and score_col in df.columns:
                fig6 = px.scatter(
                    df,
                    x=intern_col,
                    y=score_col,
                    color=color_col,
                    title=f"6. Internal Marks vs Final Performance ({intern_col})",
                    color_discrete_sequence=["#0284C7", "#EF4444"],
                    template="plotly_white",
                )
            else:
                fig6 = px.histogram(df, x=intern_col, title=f"6. Internal Marks Analysis ({intern_col})", template="plotly_white")
            fig6.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("6. Internal Marks Analysis: No internal marks column detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 7. Pass/Fail Distribution & 8. Correlation Heatmap
    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if res_col and res_col in df.columns:
            vc = df[res_col].value_counts().reset_index()
            vc.columns = [res_col, "Students"]
            fig7 = px.bar(
                vc,
                x=res_col,
                y="Students",
                color=res_col,
                text="Students",
                title=f"7. Pass / Fail Outcome Breakdown ({res_col})",
                color_discrete_sequence=["#10B981", "#EF4444", "#3B82F6"],
                template="plotly_white",
            )
            fig7.update_layout(height=370, margin=dict(l=20, r=20, t=45, b=20), showlegend=False)
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("7. Pass/Fail Distribution: No result column mapped in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c8:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr().round(2)
            fig8 = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="8. Feature Correlation Heatmap (Pearson r)",
                template="plotly_white",
            )
            fig8.update_layout(height=370, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("8. Correlation Heatmap: Requires at least 2 numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 9. Category / Course Comparison & 10. Performance Trends Over Time
    c9, c10 = st.columns(2)
    with c9:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if course_col and course_col in df.columns and score_col and score_col in df.columns:
            fig9 = px.box(
                df,
                x=course_col,
                y=score_col,
                color=course_col,
                title=f"9. Category / Course Comparison ({course_col} vs {score_col})",
                template="plotly_white",
            )
            fig9.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20), showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)
        else:
            st.info("9. Category/Course Comparison: Requires a categorical course/subject/semester column and a score column.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c10:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if date_col and date_col in df.columns and score_col and score_col in df.columns:
            t_df = df[[date_col, score_col]].copy()
            t_df[date_col] = pd.to_datetime(t_df[date_col], errors="coerce")
            t_df = t_df.dropna(subset=[date_col, score_col]).sort_values(date_col)
            if len(t_df) >= 3:
                t_df["Week"] = t_df[date_col].dt.to_period("W").apply(lambda r: r.start_time.strftime("%Y-%m-%d"))
                wk_avg = t_df.groupby("Week", as_index=False)[score_col].mean()
                fig10 = px.line(
                    wk_avg,
                    x="Week",
                    y=score_col,
                    markers=True,
                    title=f"10. Temporal Performance Trend ({score_col} by Week)",
                    template="plotly_white",
                )
                fig10.update_traces(line_color="#2563EB", line_width=2.5)
                fig10.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig10, use_container_width=True)
            else:
                st.info("10. Performance Trends: Insufficient valid date values in column.")
        else:
            st.info("10. Performance Trends: No date/timestamp column present in this dataset.")
        st.markdown("</div>", unsafe_allow_html=True)
