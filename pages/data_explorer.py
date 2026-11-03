"""
Data Explorer Page (pages/data_explorer.py) for StudentAI.
Provides interactive multi-column filtering, descriptive statistical profiling,
percentile inspection, and filtered CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np

from services.supabase_service import SupabaseService


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Interactive Data Explorer</div>
            <div class="saas-page-subtitle">Filter, sort, search, and inspect descriptive statistics across your cleaned student dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.info("No dataset loaded yet. Please visit the **Dataset** page to upload or load a student dataset.")
        if st.button("Go to Dataset Upload"):
            st.session_state["current_page"] = "Dataset"
            st.rerun()
        return

    filtered_df = df.copy()
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in filtered_df.columns if c not in numeric_cols]

    with st.expander("Filter & Slice Records", expanded=True):
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            selected_cat_cols = st.multiselect(
                "Filter by Categorical Columns",
                options=categorical_cols,
                default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols,
            )
            for c_col in selected_cat_cols:
                unique_vals = sorted(filtered_df[c_col].dropna().astype(str).unique().tolist())
                if 1 < len(unique_vals) <= 30:
                    chosen_vals = st.multiselect(f"Values for `{c_col}`", unique_vals, default=unique_vals)
                    filtered_df = filtered_df[filtered_df[c_col].astype(str).isin(chosen_vals)]

        with f_col2:
            selected_num_cols = st.multiselect(
                "Filter by Numeric Ranges",
                options=numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
            )
            for n_col in selected_num_cols:
                s = filtered_df[n_col].dropna()
                if len(s) > 0 and float(s.min()) < float(s.max()):
                    min_v, max_v = float(s.min()), float(s.max())
                    r_low, r_high = st.slider(
                        f"Range for `{n_col}`",
                        min_value=min_v,
                        max_value=max_v,
                        value=(min_v, max_v),
                    )
                    filtered_df = filtered_df[(filtered_df[n_col] >= r_low) & (filtered_df[n_col] <= r_high)]

    st.markdown(f"**Showing `{len(filtered_df):,}` of `{len(df):,}` student records**")
    st.dataframe(filtered_df, use_container_width=True, height=380)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Records (CSV)",
        data=csv_bytes,
        file_name="filtered_student_records.csv",
        mime="text/csv",
    )

    st.subheader("Descriptive Statistical Summary (Numeric Features)")
    if numeric_cols and len(filtered_df) > 0:
        desc_df = filtered_df[numeric_cols].describe().T.reset_index()
        desc_df = desc_df.rename(columns={"index": "Column"})
        st.dataframe(desc_df.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns available to compute descriptive statistics.")
