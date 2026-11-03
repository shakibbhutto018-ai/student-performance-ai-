"""
Machine Learning Models Training & Evaluation Page (pages/ml_models.py).
Supports Classification (Random Forest Classifier, Logistic Regression)
and Regression (Linear Regression, Random Forest Regressor) with zero data leakage,
real evaluation metrics, confusion matrix, actual vs predicted plots, feature importance,
and BCA Viva explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from services.supabase_service import SupabaseService
from utils.validators import smart_detect_columns
from utils.helpers import get_viva_explanations
from ml.preprocessing import detect_task_type, prepare_train_test_data
from ml.classification import train_classification_model
from ml.regression import train_regression_model


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Machine Learning Training &amp; Evaluation Studio</div>
            <div class="saas-page-subtitle">Leakage-free Scikit-learn pipelines for Classification &amp; Regression with real test-set verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.info("No dataset loaded. Please upload or load a dataset on the **Dataset** page before training ML models.")
        if st.button("Go to Dataset Page"):
            st.session_state["current_page"] = "Dataset"
            st.rerun()
        return

    mapping = st.session_state.get("column_mapping") or smart_detect_columns(df)
    role_to_col = mapping.get("role_to_col", {})
    id_cols = set(mapping.get("id_columns", []))
    if "date" in role_to_col:
        id_cols.add(role_to_col["date"])

    candidate_targets = [c for c in df.columns if c not in id_cols]
    default_target = (
        mapping.get("suggested_regression_target")
        or mapping.get("suggested_classification_target")
        or candidate_targets[-1]
    )

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("1. Configure ML Pipeline")

    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        target_col = st.selectbox(
            "Target Column (Variable to Predict)",
            options=candidate_targets,
            index=candidate_targets.index(default_target) if default_target in candidate_targets else 0,
        )

    detected_task = detect_task_type(df, target_col)

    with c2:
        if detected_task == "classification":
            algo_options = ["Random Forest Classifier", "Logistic Regression"]
        else:
            algo_options = ["Random Forest Regressor", "Linear Regression"]
        selected_algo = st.selectbox(
            f"ML Algorithm (Detected Task: {detected_task.title()})",
            options=algo_options,
        )

    with c3:
        test_pct = st.slider("Test Split (%)", min_value=15, max_value=35, value=20, step=5)

    # Feature selection excluding target, IDs, and the alternate target column to prevent target leakage
    alternate_targets = set()
    if target_col == role_to_col.get("final_score") and role_to_col.get("result"):
        alternate_targets.add(role_to_col["result"])
    elif target_col == role_to_col.get("result") and role_to_col.get("final_score"):
        alternate_targets.add(role_to_col["final_score"])

    available_features = [c for c in df.columns if c != target_col and c not in id_cols]
    default_features = [
        c for c in mapping.get("suggested_features", available_features)
        if c in available_features and c not in alternate_targets
    ]
    if not default_features:
        default_features = [c for c in available_features if c not in alternate_targets] or available_features

    selected_features = st.multiselect(
        "Feature Columns (Input Variables X)",
        options=available_features,
        default=default_features,
        help="Note: If predicting Pass/Fail Result, exclude Final Score (and vice versa) to prevent target leakage.",
    )

    train_col1, train_col2 = st.columns([1, 1])
    with train_col1:
        train_single_btn = st.button(f"Train {selected_algo}", type="primary", use_container_width=True)
    with train_col2:
        train_both_btn = st.button(f"Train & Compare Both {detected_task.title()} Models", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if train_single_btn or train_both_btn:
        if not selected_features:
            st.error("Please select at least one feature column to train the model.")
            return
        try:
            split_data = prepare_train_test_data(
                df=df,
                target_col=target_col,
                feature_cols=selected_features,
                test_size=float(test_pct) / 100.0,
                random_state=42,
            )
            algos_to_run = algo_options if train_both_btn else [selected_algo]
            for alg in algos_to_run:
                if detected_task == "classification":
                    result_bundle = train_classification_model(split_data, algorithm=alg, random_state=42)
                else:
                    result_bundle = train_regression_model(split_data, algorithm=alg, random_state=42)

                model_key = f"{alg} ({target_col})"
                st.session_state["trained_models"][model_key] = result_bundle
                st.session_state["active_model_key"] = model_key

                # Persist analysis record to session and Supabase
                user = st.session_state.get("auth_user") or {}
                analysis_entry = {
                    "dataset_name": st.session_state.get("dataset_filename", "dataset.csv"),
                    "task_type": detected_task,
                    "target_column": target_col,
                    "feature_columns": selected_features,
                    "model_name": alg,
                    "metrics": result_bundle["metrics"],
                    "feature_importance": result_bundle["feature_importance"],
                    "train_size": result_bundle["train_size"],
                    "test_size": result_bundle["test_size"],
                }
                st.session_state["session_analyses"].insert(0, analysis_entry)

                if sb.is_configured() and user.get("access_token"):
                    sb.save_dataset_analysis(
                        user_id=user["id"],
                        access_token=user["access_token"],
                        dataset_id=st.session_state.get("dataset_id"),
                        **analysis_entry,
                    )
            st.success(f"Trained {', '.join(algos_to_run)} on {split_data['X_train'].shape[0]} training records and evaluated on {split_data['X_test'].shape[0]} unseen test records.")
        except Exception as exc:
            st.error(f"ML Training Error: {exc}")
            return

    trained_models = st.session_state.get("trained_models", {})
    if not trained_models:
        st.info("Configure your target and features above, then click **Train** to compute real model evaluation metrics.")
        return

    # Model Selector & Comparison Table if multiple models are trained
    model_keys = list(trained_models.keys())
    if len(model_keys) > 1:
        st.subheader("2. Trained Model Comparison")
        comp_rows = []
        for k, b in trained_models.items():
            row = {
                "Model": b["model_name"],
                "Task": b["task_type"].title(),
                "Target": b["target_column"],
                "Train Rows": b["train_size"],
                "Test Rows": b["test_size"],
            }
            for mk, mv in b["metrics"].items():
                row[mk] = mv
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    active_key = st.selectbox(
        "Select Trained Model to Inspect & Use for Prediction",
        options=model_keys,
        index=model_keys.index(st.session_state["active_model_key"]) if st.session_state.get("active_model_key") in model_keys else 0,
    )
    st.session_state["active_model_key"] = active_key
    bundle = trained_models[active_key]

    # Display Real Evaluation Metrics
    st.subheader(f"3. Test-Set Evaluation Metrics — {bundle['model_name']}")
    st.caption(
        f"Evaluated strictly on **{bundle['test_size']} unseen test samples** "
        f"(Trained on **{bundle['train_size']} training samples**, `random_state=42`)."
    )

    metrics = bundle["metrics"]
    m_cols = st.columns(len(metrics))
    for idx, (m_name, m_val) in enumerate(metrics.items()):
        if bundle["task_type"] == "classification":
            m_cols[idx].metric(m_name, f"{m_val * 100:.2f}%", help=f"Raw score: {m_val:.4f}")
        else:
            m_cols[idx].metric(m_name, f"{m_val:.4f}")

    # Visualizations: Confusion Matrix / Actual vs Predicted & Feature Importance
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        if bundle["task_type"] == "classification" and bundle.get("confusion_matrix"):
            cm_data = bundle["confusion_matrix"]
            fig_cm = px.imshow(
                cm_data["matrix"],
                x=cm_data["labels"],
                y=cm_data["labels"],
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                title=f"Confusion Matrix ({bundle['model_name']})",
                template="plotly_white",
            )
            fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            avp_df = pd.DataFrame(bundle["actual_vs_predicted"])
            fig_avp = px.scatter(
                avp_df,
                x="Actual",
                y="Predicted",
                hover_data=["Sample_Index", "Residual_Error"],
                title=f"Actual vs Predicted Test Values ({bundle['model_name']})",
                color_discrete_sequence=["#2563EB"],
                template="plotly_white",
            )
            min_v = float(min(avp_df["Actual"].min(), avp_df["Predicted"].min()))
            max_v = float(max(avp_df["Actual"].max(), avp_df["Predicted"].max()))
            fig_avp.add_shape(
                type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                line=dict(color="#EF4444", dash="dash"),
            )
            fig_avp.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_avp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with v_col2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        fi_list = bundle.get("feature_importance", [])
        if fi_list:
            fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=True)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature Importance ({bundle['model_name']})",
                color="importance",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_fi.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View Test Set Actual vs Predicted Table"):
        st.dataframe(pd.DataFrame(bundle["actual_vs_predicted"]), use_container_width=True, hide_index=True)

    with st.expander("BCA Viva Guide: Understanding These Machine Learning Metrics"):
        explanations = get_viva_explanations()
        for k, desc in explanations.items():
            st.markdown(f"- **{k}:** {desc}")
