"""
AI Insights & Dataset Chat Page (pages/ai_insights.py).
Displays structured Generative AI explanations of actual ML predictions:
- Performance Summary
- Strengths (2-4 points)
- Areas for Improvement (2-4 points)
- Recommended Actions
- Why Did The Model Predict This?
And provides 'ASK AI ABOUT THIS DATASET' interactive Q&A grounded in real statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from services.supabase_service import SupabaseService
from services.ai_router import AIRouter
from utils.validators import smart_detect_columns
from utils.helpers import compute_dashboard_kpis


def _build_dataset_context_payload(df: pd.DataFrame) -> dict:
    mapping = st.session_state.get("column_mapping") or smart_detect_columns(df)
    role_to_col = mapping.get("role_to_col", {})

    trained_models = st.session_state.get("trained_models", {})
    active_key = st.session_state.get("active_model_key")
    best_model = trained_models.get(active_key) if active_key in trained_models else (
        list(trained_models.values())[-1] if trained_models else None
    )

    kpis = compute_dashboard_kpis(df, role_to_col, best_model)

    # Compute top correlations with score column
    score_col = role_to_col.get("final_score")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    top_corrs = []
    if score_col and score_col in num_cols and len(num_cols) >= 2:
        corr_s = df[num_cols].corr()[score_col].drop(labels=[score_col], errors="ignore").dropna()
        for feat_name, c_val in corr_s.abs().sort_values(ascending=False).items():
            top_corrs.append({"feature": feat_name, "correlation": round(float(corr_s[feat_name]), 4)})

    ml_summary = {}
    if best_model:
        ml_summary = {
            "model_name": best_model.get("model_name"),
            "task_type": best_model.get("task_type"),
            "target_column": best_model.get("target_column"),
            "train_size": best_model.get("train_size"),
            "test_size": best_model.get("test_size"),
            "metrics": best_model.get("metrics"),
            "feature_importance": best_model.get("feature_importance"),
        }

    return {
        "dataset_info": {
            "filename": st.session_state.get("dataset_filename", "dataset.csv"),
            "rows": int(len(df)),
            "columns": list(df.columns),
        },
        "kpis": kpis,
        "top_correlations": top_corrs,
        "ml_summary": ml_summary,
    }


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Generative AI Insights &amp; Dataset Intelligence</div>
            <div class="saas-page-subtitle">Explainable AI narratives grounded strictly in Scikit-learn ML outputs and dataset statistics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    router = AIRouter(
        primary_provider=st.session_state.get("ai_primary_provider"),
        fallback_provider=st.session_state.get("ai_fallback_provider"),
        openrouter_model=st.session_state.get("openrouter_model"),
    )

    tab_pred, tab_chat = st.tabs(["Student Prediction AI Explanation", "ASK AI ABOUT THIS DATASET"])

    # TAB 1: AI Explanation of Student Prediction
    with tab_pred:
        pred_res = st.session_state.get("latest_prediction")
        ai_insight = st.session_state.get("latest_ai_insight")

        if not pred_res:
            st.info(
                "Run a student prediction on the **Prediction** page first so the AI Router can explain the actual ML prediction and risk factors."
            )
            if st.button("Go to Prediction Page", type="primary"):
                st.session_state["current_page"] = "Prediction"
                st.rerun()
        else:
            if not ai_insight or st.button("Regenerate AI Explanation"):
                df = st.session_state.get("cleaned_df")
                if df is None:
                    df = st.session_state.get("raw_df")
                ctx = _build_dataset_context_payload(df) if df is not None else {}
                ai_insight = router.generate_prediction_explanation(pred_res, ctx)
                st.session_state["latest_ai_insight"] = ai_insight

            st.caption(
                f"Provider Used: **{ai_insight.get('provider_used', 'AI Router')}** "
                f"(`{ai_insight.get('model_used', '')}`) | "
                f"ML Prediction Explained: **{pred_res.get('predicted_display')}** ({pred_res.get('risk_level')})"
            )

            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Performance Summary")
            st.write(ai_insight.get("performance_summary", "No summary available."))
            st.markdown("</div>", unsafe_allow_html=True)

            c_str, c_imp = st.columns(2)
            with c_str:
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                st.markdown("#### Strengths")
                for item in ai_insight.get("strengths", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            with c_imp:
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                st.markdown("#### Areas for Improvement")
                for item in ai_insight.get("areas_for_improvement", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Recommended Actions")
            for act in ai_insight.get("recommended_actions", []):
                st.markdown(f"- {act}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Why Did The Model Predict This?")
            st.write(ai_insight.get("why_model_predicted_this", ""))
            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: ASK AI ABOUT THIS DATASET
    with tab_chat:
        df = st.session_state.get("cleaned_df")
        if df is None:
            df = st.session_state.get("raw_df")

        if df is None or df.empty:
            st.info("Please upload or load a dataset on the **Dataset** page to ask AI questions about your data.")
            return

        ctx_payload = _build_dataset_context_payload(df)

        st.markdown("#### ASK AI ABOUT THIS DATASET")
        st.caption("Select an analytical question below or type your own question. Answers are strictly grounded in your dataset's actual statistics and ML metrics.")

        sample_questions = [
            "What factors are associated with low performance?",
            "Which factors have the strongest relationship with scores?",
            "Explain the model performance.",
            "What should students improve?",
            "Summarize this dataset.",
            "Explain this graph.",
        ]

        q_cols = st.columns(3)
        triggered_question = None
        for idx, q_text in enumerate(sample_questions):
            with q_cols[idx % 3]:
                if st.button(q_text, key=f"sample_q_{idx}", use_container_width=True):
                    triggered_question = q_text

        custom_q = st.chat_input("Ask a question about this dataset or trained ML model...")
        active_q = triggered_question or custom_q

        if active_q:
            res = router.answer_dataset_question(active_q, ctx_payload)
            chat_entry = {
                "question": active_q,
                "answer": res["answer"],
                "provider": res["provider_used"],
                "model": res["model_used"],
            }
            st.session_state["dataset_chat_history"].insert(0, chat_entry)

            user = st.session_state.get("auth_user") or {}
            if sb.is_configured() and user.get("access_token"):
                sb.save_ai_insight(
                    user_id=user["id"],
                    access_token=user["access_token"],
                    prediction_id=st.session_state.get("latest_prediction_id"),
                    dataset_id=st.session_state.get("dataset_id"),
                    insight_type="dataset_chat",
                    provider_used=res["provider_used"],
                    model_used=res["model_used"],
                    prompt_context={"question": active_q},
                    response_content={"answer": res["answer"]},
                )

        for entry in st.session_state.get("dataset_chat_history", []):
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(entry["answer"])
            st.caption(f"Source Engine: {entry['provider']} ({entry['model']})")
            st.markdown("</div>", unsafe_allow_html=True)
