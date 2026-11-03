"""
Student Performance Prediction & Risk Analysis Page (pages/prediction.py).
Provides interactive feature inputs (Attendance, Study Hours, Internal Marks,
Assignment Score, Previous Score, etc.), button 'ANALYZE PERFORMANCE',
real ML model inference, Risk Stratification (LOW/MEDIUM/HIGH RISK),
and automated Generative AI explanation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from services.supabase_service import SupabaseService
from services.ai_router import AIRouter
from utils.validators import smart_detect_columns
from utils.helpers import compute_dashboard_kpis
from ml.prediction import predict_student_performance


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Student Performance Predictor &amp; Risk Analyzer</div>
            <div class="saas-page-subtitle">Enter student academic indicators to compute real ML predictions, risk stratification, and AI explanations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trained_models = st.session_state.get("trained_models", {})
    if not trained_models:
        st.info("No trained ML model found in this session. Please train a model on the **ML Models** page first.")
        if st.button("Go to ML Models Studio", type="primary"):
            st.session_state["current_page"] = "ML Models"
            st.rerun()
        return

    model_keys = list(trained_models.keys())
    active_key = st.selectbox(
        "Select Trained ML Model for Inference",
        options=model_keys,
        index=model_keys.index(st.session_state["active_model_key"]) if st.session_state.get("active_model_key") in model_keys else 0,
    )
    st.session_state["active_model_key"] = active_key
    bundle = trained_models[active_key]

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("raw_df")
    mapping = st.session_state.get("column_mapping") or (smart_detect_columns(df) if df is not None else {"role_to_col": {}})
    role_to_col = mapping.get("role_to_col", {})

    feature_cols = bundle["feature_columns"]
    feature_meta = bundle["feature_metadata"]

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Student Academic Parameters")

    input_values = {}
    with st.form("student_prediction_form"):
        grid = st.columns(3)
        for idx, col_name in enumerate(feature_cols):
            meta = feature_meta.get(col_name, {"type": "numeric", "min": 0.0, "max": 100.0, "median": 70.0})
            with grid[idx % 3]:
                if meta["type"] == "numeric":
                    col_low = col_name.lower()
                    if "hour" in col_low:
                        min_v, max_v, step_v = 0.0, 16.0, 0.5
                    elif any(k in col_low for k in ("attendance", "score", "mark", "pct", "percent")):
                        min_v, max_v, step_v = 0.0, 100.0, 1.0
                    else:
                        min_v = float(min(0.0, meta["min"]))
                        max_v = float(max(100.0, meta["max"]))
                        step_v = 1.0
                    default_v = float(min(max(meta["median"], min_v), max_v))
                    input_values[col_name] = st.number_input(
                        f"{col_name}",
                        min_value=min_v,
                        max_value=max_v,
                        value=round(default_v, 1),
                        step=step_v,
                    )
                else:
                    opts = meta.get("options", ["Unknown"])
                    default_opt = meta.get("default", opts[0])
                    input_values[col_name] = st.selectbox(
                        f"{col_name}",
                        options=opts,
                        index=opts.index(default_opt) if default_opt in opts else 0,
                    )

        analyze_btn = st.form_submit_button("ANALYZE PERFORMANCE", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn:
        try:
            kpis = compute_dashboard_kpis(df, role_to_col, bundle)
            pred_res = predict_student_performance(
                trained_model_bundle=bundle,
                input_features=input_values,
                role_to_col=role_to_col,
                dataset_stats=kpis,
            )
            st.session_state["latest_prediction"] = pred_res
            st.session_state["session_predictions"].insert(0, pred_res)

            user = st.session_state.get("auth_user") or {}
            pred_id = None
            if sb.is_configured() and user.get("access_token"):
                ok, saved_row, _ = sb.save_prediction(
                    user_id=user["id"],
                    access_token=user["access_token"],
                    dataset_id=st.session_state.get("dataset_id"),
                    prediction_result=pred_res,
                )
                if ok and isinstance(saved_row, dict):
                    pred_id = saved_row.get("id")
            st.session_state["latest_prediction_id"] = pred_id

            # Generate AI Explanation via AIRouter
            router = AIRouter(
                primary_provider=st.session_state.get("ai_primary_provider"),
                fallback_provider=st.session_state.get("ai_fallback_provider"),
                openrouter_model=st.session_state.get("openrouter_model"),
            )
            ai_insight = router.generate_prediction_explanation(pred_res, {"kpis": kpis})
            st.session_state["latest_ai_insight"] = ai_insight

            if sb.is_configured() and user.get("access_token"):
                sb.save_ai_insight(
                    user_id=user["id"],
                    access_token=user["access_token"],
                    prediction_id=pred_id,
                    dataset_id=st.session_state.get("dataset_id"),
                    insight_type="prediction_explanation",
                    provider_used=ai_insight.get("provider_used", "AI Router"),
                    model_used=ai_insight.get("model_used", ""),
                    prompt_context=input_values,
                    response_content=ai_insight,
                )
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

    pred_res = st.session_state.get("latest_prediction")
    if not pred_res:
        return

    # Display Output Cards
    st.subheader("Prediction & Academic Risk Assessment")
    risk_lvl = pred_res["risk_level"]
    badge_class = (
        "risk-badge-low" if risk_lvl == "LOW RISK"
        else ("risk-badge-medium" if risk_lvl == "MEDIUM RISK" else "risk-badge-high")
    )

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        lbl_title = "Predicted Score" if pred_res["task_type"] == "regression" else "Predicted Outcome"
        st.metric(lbl_title, pred_res["predicted_display"])
    with o2:
        st.metric("Performance Category", pred_res["performance_category"])
    with o3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Risk Level</div>
                <div style="margin-top:0.35rem;"><span class="{badge_class}">{risk_lvl}</span></div>
                <div class="kpi-sub">Evidence-based stratification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o4:
        conf_val = pred_res.get("confidence_score")
        if conf_val is not None:
            st.metric("Model Probability", f"{conf_val * 100:.1f}%", help=f"Model Used: {pred_res['model_name']}")
        else:
            st.metric("Model Used", pred_res["model_name"])

    st.markdown("<br>", unsafe_allow_html=True)

    r_col1, r_col2 = st.columns([1.2, 1.0])
    with r_col1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(f"#### Why Was `{risk_lvl}` Assigned?")
        for reason in pred_res.get("risk_reasons", []):
            st.markdown(f"- {reason}")
        st.markdown("#### Identified Positive Academic Indicators")
        for s_item in pred_res.get("academic_strengths", []):
            st.markdown(f"- {s_item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with r_col2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        factors = pred_res.get("important_factors", [])
        if factors:
            fi_df = pd.DataFrame(factors).sort_values("importance", ascending=True)
            fig_f = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Important Model Factors ({pred_res['model_name']})",
                color="importance",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_f.update_layout(height=280, margin=dict(l=15, r=15, t=40, b=15), coloraxis_showscale=False)
            st.plotly_chart(fig_f, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("Visit the **AI Insights** page for the full generative AI interpretation and dataset Q&A, or **Reports** to download a PDF report.")
