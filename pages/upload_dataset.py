"""
Dataset Upload, Smart Column Detection, and Data Cleaning Page (pages/upload_dataset.py).
Displays real dataset metrics, provides automatic & manual column mapping UI,
executes genuine preprocessing with before/after counts, and allows downloading cleaned CSV.
"""

import os
import io
import streamlit as st
import pandas as pd

from services.supabase_service import SupabaseService
from utils.validators import validate_uploaded_dataframe, smart_detect_columns
from utils.data_cleaner import inspect_dataset, clean_dataset


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Dataset Management, Smart Mapping &amp; Data Cleaning</div>
            <div class="saas-page-subtitle">Upload student CSV records, verify educational column detection, and execute automated preprocessing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_sample = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Student Performance CSV Dataset",
            type=["csv"],
            help="Upload any educational CSV containing attendance, study hours, marks, or exam results.",
        )

    with col_sample:
        st.markdown("**Demonstration & Testing Dataset**")
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_students.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                sample_bytes = f.read()
            st.download_button(
                "Download Sample Dataset (CSV)",
                data=sample_bytes,
                file_name="sample_students.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("Load Sample Dataset into Workspace", type="secondary", use_container_width=True):
                raw_df = pd.read_csv(sample_path)
                _process_loaded_dataframe(raw_df, "sample_students.csv", sb)
                st.success("Loaded `sample_students.csv` (304 raw rows with realistic missing values & duplicates for cleaning demonstration).")

    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            raw_df = pd.read_csv(io.BytesIO(raw_bytes))
            if st.session_state.get("dataset_filename") != uploaded_file.name or st.session_state.get("raw_df") is None:
                _process_loaded_dataframe(raw_df, uploaded_file.name, sb)
        except Exception as exc:
            st.error(f"Could not parse uploaded CSV file: {exc}")
            return

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Upload a CSV file above or click **Load Sample Dataset into Workspace** to inspect dataset statistics.")
        return

    filename = st.session_state.get("dataset_filename", "dataset.csv")
    inspection = st.session_state.get("dataset_inspection") or inspect_dataset(raw_df, filename)

    # 1. Real Dataset Inspection Metrics
    st.subheader(f"1. Raw Dataset Profile — `{inspection['filename']}`")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows", f"{inspection['rows']:,}")
    m2.metric("Columns", f"{inspection['columns']}")
    m3.metric("Missing Values", f"{inspection['missing_values']:,}")
    m4.metric("Duplicate Records", f"{inspection['duplicate_records']:,}")
    m5.metric("Numeric Columns", f"{len(inspection['numeric_columns'])}")
    m6.metric("Categorical Columns", f"{len(inspection['categorical_columns'])}")

    with st.expander("View Column Data Types, Missing Counts & Raw Preview", expanded=True):
        st.dataframe(pd.DataFrame(inspection["column_profile"]), use_container_width=True, hide_index=True)
        st.markdown("**Raw Dataset Preview (First 10 Rows):**")
        st.dataframe(raw_df.head(10), use_container_width=True)

    # 2. Data Cleaning Pipeline
    st.subheader("2. Automated Data Cleaning & Preprocessing")
    c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
    with c_opt1:
        missing_strat = st.selectbox(
            "Missing Value Strategy",
            options=["median_mode", "mean_mode", "drop_rows"],
            format_func=lambda x: {
                "median_mode": "Median (Numeric) + Mode (Categorical)",
                "mean_mode": "Mean (Numeric) + Mode (Categorical)",
                "drop_rows": "Drop Rows with Missing Values",
            }[x],
        )
    with c_opt2:
        rem_dups = st.checkbox("Remove Duplicate Rows", value=True)
    with c_opt3:
        fix_types = st.checkbox("Auto-Convert Numeric Strings", value=True)
    with c_opt4:
        clip_out = st.checkbox("Handle Domain & IQR Outliers", value=True)

    if st.button("Run Data Cleaning Pipeline", type="primary"):
        cleaned_df, clean_rep = clean_dataset(
            raw_df,
            missing_strategy=missing_strat,
            remove_duplicates=rem_dups,
            fix_data_types=fix_types,
            handle_outliers=clip_out,
        )
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["cleaning_report"] = clean_rep
        st.session_state["column_mapping"] = smart_detect_columns(cleaned_df)
        st.success("Data cleaning completed with real calculated statistics.")

    clean_rep = st.session_state.get("cleaning_report")
    cleaned_df = st.session_state.get("cleaned_df")

    if clean_rep and cleaned_df is not None:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Data Cleaning Verification Summary")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Rows Before Cleaning", f"{clean_rep['rows_before']:,}")
        r2.metric("Duplicates Removed", f"{clean_rep['duplicates_removed']:,}")
        r3.metric("Missing Values Handled", f"{clean_rep['missing_values_handled']:,}")
        r4.metric("Outliers Adjusted", f"{clean_rep['outliers_adjusted']:,}")
        r5.metric("Rows After Cleaning", f"{clean_rep['rows_after']:,}")

        for log_line in clean_rep.get("changes_log", []):
            st.caption(f"• {log_line}")

        csv_clean_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=csv_clean_bytes,
            file_name=f"cleaned_{filename}",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Smart Column Detection & Manual Mapping UI
    active_df = cleaned_df if cleaned_df is not None else raw_df
    mapping = st.session_state.get("column_mapping") or smart_detect_columns(active_df)
    role_to_col = mapping.get("role_to_col", {})

    st.subheader("3. Smart Educational Column Detection & Manual Mapping")
    conf_level = mapping.get("detection_confidence", "medium").upper()
    st.caption(
        f"Automatic Schema Detection Confidence: **{conf_level}** "
        f"(Matched {mapping.get('matched_core_count', 0)} canonical educational roles). "
        "You can verify or customize the mapping below:"
    )

    all_cols = ["(None)"] + list(active_df.columns)
    roles_to_configure = [
        ("final_score", "Final Score / Numerical Target"),
        ("result", "Result / Pass-Fail Target"),
        ("attendance", "Attendance (%)"),
        ("study_hours", "Study Hours"),
        ("internal_marks", "Internal Marks"),
        ("assignment_score", "Assignment Score"),
        ("previous_score", "Previous Score / Marks"),
        ("course", "Course / Program"),
        ("semester", "Semester"),
        ("subject", "Subject"),
        ("gender", "Gender"),
        ("date", "Assessment Date"),
    ]

    with st.form("manual_column_mapping_form"):
        grid_cols = st.columns(3)
        updated_role_to_col = {}
        for idx, (role_key, role_label) in enumerate(roles_to_configure):
            current_mapped = role_to_col.get(role_key, "(None)")
            if current_mapped not in all_cols:
                current_mapped = "(None)"
            with grid_cols[idx % 3]:
                chosen = st.selectbox(
                    role_label,
                    options=all_cols,
                    index=all_cols.index(current_mapped),
                    key=f"map_{role_key}",
                )
                if chosen != "(None)":
                    updated_role_to_col[role_key] = chosen

        save_map_btn = st.form_submit_button("Save Column Mapping")
        if save_map_btn:
            mapping["role_to_col"] = updated_role_to_col
            if "final_score" in updated_role_to_col:
                mapping["suggested_regression_target"] = updated_role_to_col["final_score"]
            if "result" in updated_role_to_col:
                mapping["suggested_classification_target"] = updated_role_to_col["result"]
            st.session_state["column_mapping"] = mapping
            st.success("Column mapping updated and saved for Analytics and ML Training.")


def _process_loaded_dataframe(raw_df: pd.DataFrame, filename: str, sb: SupabaseService) -> None:
    is_valid, msg, _ = validate_uploaded_dataframe(raw_df, filename)
    if not is_valid:
        st.error(msg)
        return

    inspection = inspect_dataset(raw_df, filename)
    cleaned_df, clean_rep = clean_dataset(raw_df)
    mapping = smart_detect_columns(cleaned_df)

    st.session_state["raw_df"] = raw_df
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["dataset_filename"] = filename
    st.session_state["dataset_inspection"] = inspection
    st.session_state["cleaning_report"] = clean_rep
    st.session_state["column_mapping"] = mapping

    user = st.session_state.get("auth_user") or {}
    if sb.is_configured() and user.get("access_token"):
        ok, row, _ = sb.save_dataset_metadata(
            user_id=user["id"],
            access_token=user["access_token"],
            filename=filename,
            raw_rows=len(raw_df),
            cleaned_rows=len(cleaned_df),
            column_count=len(cleaned_df.columns),
            missing_values_count=inspection["missing_values"],
            duplicates_removed=clean_rep["duplicates_removed"],
            numeric_columns=inspection["numeric_columns"],
            categorical_columns=inspection["categorical_columns"],
            detected_mapping=mapping["role_to_col"],
            summary_stats=clean_rep,
        )
        if ok and row and isinstance(row, dict):
            st.session_state["dataset_id"] = row.get("id")
