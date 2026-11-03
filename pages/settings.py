"""
Settings, Cloud Diagnostics, Admin Console & BCA Viva Reference Page (pages/settings.py).
Allows configuring AI Router primary/fallback providers, checking Supabase PostgreSQL status,
viewing separated Admin user/activity logs, and reviewing BCA Viva questions & answers.
"""

import streamlit as st
import pandas as pd

from services.supabase_service import SupabaseService
from services.ai_router import AIRouter


def render(sb: SupabaseService) -> None:
    st.markdown(
        """
        <div class="saas-page-header">
            <div class="saas-page-title">Platform Settings, Cloud Diagnostics &amp; Admin Console</div>
            <div class="saas-page-subtitle">Configure AI Router providers, verify Supabase PostgreSQL persistence, and review BCA Viva documentation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.get("auth_user") or {}
    is_admin = user.get("role") == "ADMIN"

    tabs = ["AI Router & Database Status", "Account History", "BCA Viva Reference"]
    if is_admin:
        tabs.append("Admin Console (ADMIN Only)")

    rendered_tabs = st.tabs(tabs)

    # TAB 1: AI Router & Database Status
    with rendered_tabs[0]:
        c_db, c_ai = st.columns(2)
        with c_db:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.subheader("Supabase PostgreSQL Status")
            is_ok, status_msg = sb.check_connection()
            if is_ok:
                st.success(status_msg)
            else:
                st.warning(status_msg)
            st.markdown(
                """
                - **Database Engine:** Supabase PostgreSQL (Cloud Relational DB)
                - **Local SQLite Dependency:** None (100% removed)
                - **Row Level Security (RLS):** Enabled per `user_id` (`auth.uid() = user_id`)
                - **Secrets Protection:** Loaded strictly from `.env` / Streamlit Secrets
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c_ai:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.subheader("Generative AI Router Configuration")
            router = AIRouter(
                primary_provider=st.session_state.get("ai_primary_provider"),
                fallback_provider=st.session_state.get("ai_fallback_provider"),
                openrouter_model=st.session_state.get("openrouter_model"),
            )
            p_status = router.get_provider_status()

            s1, s2, s3 = st.columns(3)
            s1.metric("OpenAI API", "Configured" if p_status["openai"] else "Not Set")
            s2.metric("Gemini API", "Configured" if p_status["gemini"] else "Not Set")
            s3.metric("OpenRouter API", "Configured" if p_status["openrouter"] else "Not Set")

            providers_list = ["gemini", "openai", "openrouter"]
            curr_prim = st.session_state.get("ai_primary_provider", "gemini")
            curr_fall = st.session_state.get("ai_fallback_provider", "openrouter")

            new_prim = st.selectbox(
                "Primary AI Provider",
                options=providers_list,
                index=providers_list.index(curr_prim) if curr_prim in providers_list else 0,
            )
            new_fall = st.selectbox(
                "Fallback AI Provider",
                options=providers_list,
                index=providers_list.index(curr_fall) if curr_fall in providers_list else 2,
            )
            new_or_model = st.text_input(
                "OpenRouter Model ID (`OPENROUTER_MODEL`)",
                value=st.session_state.get("openrouter_model", "openai/gpt-4o-mini"),
            )

            if st.button("Save AI Router Settings", type="primary"):
                st.session_state["ai_primary_provider"] = new_prim
                st.session_state["ai_fallback_provider"] = new_fall
                st.session_state["openrouter_model"] = new_or_model.strip() or "openai/gpt-4o-mini"
                st.success("AI Router provider preferences updated for this session.")
            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: User's Isolated History
    with rendered_tabs[1]:
        st.subheader(f"Private Activity & Analysis History ({user.get('email', 'User')})")
        st.caption("Every record below is strictly scoped to your authenticated user account.")
        preds = st.session_state.get("session_predictions", [])
        analyses = st.session_state.get("session_analyses", [])

        st.markdown(f"**Saved Predictions ({len(preds)}):**")
        if preds:
            st.dataframe(pd.DataFrame(preds)[["model_name", "task_type", "predicted_label", "performance_category", "risk_level"]], use_container_width=True, hide_index=True)
        else:
            st.info("No predictions recorded yet.")

        st.markdown(f"**Saved Model Training Runs ({len(analyses)}):**")
        if analyses:
            st.dataframe(pd.DataFrame(analyses)[["dataset_name", "model_name", "task_type", "target_column", "train_size", "test_size"]], use_container_width=True, hide_index=True)
        else:
            st.info("No model training runs recorded yet.")

    # TAB 3: BCA Viva Questions & Answers Reference
    with rendered_tabs[2]:
        st.subheader("BCA 3rd Semester Viva-Voce Quick Preparation Guide")
        viva_qa = [
            ("Why Python?", "Python offers rich, mature scientific computing and machine learning ecosystems (Pandas, NumPy, Scikit-learn, Plotly) along with clean syntax and rapid web application deployment via Streamlit."),
            ("Why Pandas & NumPy?", "NumPy provides fast vectorized array operations and linear algebra routines, while Pandas provides DataFrame structures for tabular CSV ingestion, missing-value imputation, duplicate removal, and group-by aggregations."),
            ("Why Scikit-learn?", "Scikit-learn provides standardized, leakage-free `Pipeline` and `ColumnTransformer` objects alongside battle-tested implementations of Random Forest, Logistic Regression, and Linear Regression."),
            ("Why Random Forest, Logistic Regression, and Linear Regression?", "Random Forest captures non-linear feature interactions and reduces variance through bootstrap aggregation (bagging) of decision trees. Logistic Regression provides interpretable probability estimates for binary classification (Pass/Fail). Linear Regression models continuous numerical relationships (Final Score) via Ordinary Least Squares."),
            ("What is Train/Test Split & Data Leakage?", "Train/test split separates data into training samples (e.g., 80%) to fit parameters and unseen test samples (20%) to evaluate generalization. Fitting scalers/imputers only on the training split prevents data leakage from the test set."),
            ("Why use Supabase PostgreSQL instead of SQLite?", "PostgreSQL is an enterprise relational database. Supabase provides managed cloud PostgreSQL with built-in authentication and Row Level Security (RLS). Cloud persistence ensures multi-user data survives application restarts and Streamlit redeployments, unlike local ephemeral SQLite files."),
            ("How does AI differ from ML in this project?", "The Scikit-learn Machine Learning pipeline is the mathematical prediction engine that computes numerical scores, probabilities, and feature importances. Generative AI (OpenAI, Gemini, OpenRouter) acts strictly as the natural-language interpretation layer that explains the ML outputs and recommends academic actions without inventing numbers."),
        ]
        for q, a in viva_qa:
            with st.expander(q, expanded=False):
                st.write(a)

    # TAB 4: Admin Console (Only rendered when role == 'ADMIN')
    if is_admin and len(rendered_tabs) > 3:
        with rendered_tabs[3]:
            st.subheader("Administrator Governance Console")
            st.caption("Separated ADMIN view: inspect registered platform profiles and system-wide audit logs.")
            if sb.is_configured() and user.get("access_token"):
                profiles = sb.get_admin_profiles(user["access_token"])
                logs = sb.get_activity_logs(user["id"], user["access_token"], is_admin=True)
                st.markdown(f"**Registered Profiles ({len(profiles)}):**")
                if profiles:
                    st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)
                st.markdown(f"**Platform Audit Logs ({len(logs)}):**")
                if logs:
                    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
            else:
                st.info("Connect your Supabase PostgreSQL instance in `.env` to view live multi-user `public.profiles` and `public.activity_logs` records.")
