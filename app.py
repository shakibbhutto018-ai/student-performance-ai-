"""
StudentAI — AI-Powered Academic Performance Intelligence
Main Application Entry Point (app.py)
Run command: streamlit run app.py
"""

import os
import sys

# Ensure project root is always at the top of sys.path for Streamlit Cloud deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="StudentAI | AI-Powered Academic Performance Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from services.supabase_service import SupabaseService, _get_config_value
from pages import (
    dashboard,
    upload_dataset,
    data_explorer,
    analytics,
    ml_models,
    prediction,
    ai_insights,
    reports,
    settings,
)


def load_custom_css() -> None:
    """Loads custom SaaS CSS from assets/styles.css using relative path."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initializes all required Streamlit session state keys safely."""
    defaults: Dict[str, Any] = {
        "authenticated": False,
        "auth_user": None,
        "current_page": "Dashboard",
        "raw_df": None,
        "cleaned_df": None,
        "dataset_filename": None,
        "dataset_id": None,
        "dataset_inspection": None,
        "cleaning_report": None,
        "column_mapping": None,
        "trained_models": {},
        "active_model_key": None,
        "latest_prediction": None,
        "latest_prediction_id": None,
        "latest_ai_insight": None,
        "session_predictions": [],
        "session_analyses": [],
        "session_reports": [],
        "dataset_chat_history": [],
        "ai_primary_provider": _get_config_value("DEFAULT_AI_PROVIDER", "gemini"),
        "ai_fallback_provider": _get_config_value("FALLBACK_AI_PROVIDER", "openrouter"),
        "openrouter_model": _get_config_value("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sync_user_cloud_history(sb: SupabaseService) -> None:
    """Loads persistent prediction, analysis, and report history for the logged-in user from Supabase."""
    user = st.session_state.get("auth_user")
    if not user or not sb.is_configured():
        return
    uid = user.get("id")
    token = user.get("access_token")
    if not uid or not token:
        return

    if not st.session_state.get("cloud_history_loaded"):
        preds = sb.get_user_predictions(uid, token)
        if preds:
            st.session_state["session_predictions"] = preds
        analyses = sb.get_user_analyses(uid, token)
        if analyses:
            st.session_state["session_analyses"] = analyses
        reps = sb.get_user_reports(uid, token)
        if reps:
            st.session_state["session_reports"] = reps
        st.session_state["cloud_history_loaded"] = True


def render_login_landing(sb: SupabaseService) -> None:
    """Renders a sleek, modern, user-friendly StudentAI Authentication interface."""
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 1.45, 1])

    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div class="auth-header-icon">
                    <span>🎓</span>
                </div>
                <div class="auth-header-title">StudentAI</div>
                <div class="auth-header-subtitle">Academic Performance Intelligence &amp; Prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="saas-auth-card">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email_in = st.text_input("Email Address", placeholder="student@university.edu")
                password_in = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In to Portal", type="primary", use_container_width=True)

                if submit_login:
                    ok, msg, user_data = sb.login_user(email_in, password_in)
                    if ok and user_data:
                        st.session_state["authenticated"] = True
                        st.session_state["auth_user"] = user_data
                        st.session_state["cloud_history_loaded"] = False
                        sync_user_cloud_history(sb)
                        st.toast(msg, icon="✅")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_register:
            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="e.g. Mohammed Shakib")
                reg_email = st.text_input("Email Address", placeholder="student@university.edu")
                reg_password = st.text_input("Password", type="password", placeholder="Minimum 4 characters")
                role_choice = st.selectbox(
                    "Account Role",
                    ["🎓 Student (Learner)", "👨‍🏫 Faculty / Evaluator (Admin)"],
                    index=0,
                )
                reg_role = "ADMIN" if "Faculty" in role_choice else "USER"
                submit_reg = st.form_submit_button("Create Account & Sign In", type="primary", use_container_width=True)

                if submit_reg:
                    ok, msg, user_data = sb.register_user(reg_email, reg_password, reg_name, reg_role)
                    if ok and user_data:
                        st.session_state["authenticated"] = True
                        st.session_state["auth_user"] = user_data
                        st.session_state["cloud_history_loaded"] = False
                        sync_user_cloud_history(sb)
                        st.toast(f"Account created! Welcome, {user_data.get('full_name')}!", icon="🎉")
                        st.rerun()
                    else:
                        st.error(msg)

        # Quick Demo One-Click Access
        st.markdown(
            """
            <div class="auth-divider-line">
                <span class="auth-divider-text">Or Quick Demo Access</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("🎓 Demo Student", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "email": "student@studentai.edu",
                    "full_name": "BCA Demo Student",
                    "role": "USER",
                    "access_token": None,
                    "is_cloud_authenticated": False,
                }
                st.rerun()
        with col_d2:
            if st.button("👨‍🏫 Faculty (Admin)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = {
                    "id": "00000000-0000-0000-0000-000000000002",
                    "email": "admin.faculty@studentai.edu",
                    "full_name": "BCA Project Evaluator",
                    "role": "ADMIN",
                    "access_token": None,
                    "is_cloud_authenticated": False,
                }
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Minimalist feature highlights
        st.markdown(
            """
            <div style="display: flex; justify-content: center; gap: 0.6rem; margin-top: 1.5rem; flex-wrap: wrap;">
                <div class="auth-feature-pill">⚡ ML Score Prediction</div>
                <div class="auth-feature-pill">📊 Visual Analytics</div>
                <div class="auth-feature-pill">🎯 Risk Stratification</div>
                <div class="auth-feature-pill">💡 AI Explanations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_authenticated_app(sb: SupabaseService) -> None:
    """Renders the sidebar navigation and active workspace view."""
    user = st.session_state.get("auth_user") or {}
    sync_user_cloud_history(sb)

    nav_options = [
        "Dashboard",
        "Dataset",
        "Data Explorer",
        "Analytics",
        "ML Models",
        "Prediction",
        "AI Insights",
        "Reports",
        "Settings",
    ]

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;">StudentAI</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.6rem;">Academic Performance Intelligence</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: #1E293B;">{user.get('full_name', 'User')}</div>
                <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 0.35rem;">{user.get('email', '')}</div>
                <span class="role-pill">{user.get('role', 'USER')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current = st.session_state.get("current_page", "Dashboard")
        if current not in nav_options:
            current = "Dashboard"

        selected_page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(current),
            label_visibility="collapsed",
        )
        if selected_page != current:
            st.session_state["current_page"] = selected_page
            st.rerun()

        st.markdown("<hr style='margin: 1rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

        # Quick dataset status pill in sidebar
        active_df = st.session_state.get("cleaned_df")
        if active_df is None:
            active_df = st.session_state.get("raw_df")
        if active_df is not None:
            st.caption(
                f"Active Dataset: **{st.session_state.get('dataset_filename', 'Loaded')}** "
                f"({len(active_df)} rows)"
            )
        else:
            st.caption("No dataset loaded yet.")

        if st.button("Logout", use_container_width=True):
            sb.logout_user(user.get("access_token"), user.get("id"))
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Route to selected page module
    page = st.session_state.get("current_page", "Dashboard")
    if page == "Dashboard":
        dashboard.render(sb)
    elif page == "Dataset":
        upload_dataset.render(sb)
    elif page == "Data Explorer":
        data_explorer.render(sb)
    elif page == "Analytics":
        analytics.render(sb)
    elif page == "ML Models":
        ml_models.render(sb)
    elif page == "Prediction":
        prediction.render(sb)
    elif page == "AI Insights":
        ai_insights.render(sb)
    elif page == "Reports":
        reports.render(sb)
    elif page == "Settings":
        settings.render(sb)


def main() -> None:
    load_custom_css()
    init_session_state()
    sb = SupabaseService()

    if not st.session_state.get("authenticated"):
        render_login_landing(sb)
    else:
        render_authenticated_app(sb)


if __name__ == "__main__":
    main()
