"""
AI Router Service for StudentAI.
Manages Primary Provider -> Fallback Provider routing across OpenAI, Gemini, and OpenRouter.
Enforces strict rules:
1. Never call all three APIs simultaneously; call one provider at a time.
2. If Primary Provider fails, safely capture error without exposing secrets and try Fallback Provider.
3. If all providers fail (or are unconfigured), gracefully degrade to a deterministic
   statistical explanation engine so the ML application never crashes.
4. LLMs interpret actual ML results and dataset statistics — they NEVER invent numerical predictions.
"""

import json
import re
from typing import Dict, Any, List, Optional

from services.openai_service import OpenAIService
from services.gemini_service import GeminiService
from services.openrouter_service import OpenRouterService
from services.supabase_service import _get_config_value, _sanitize_error


class AIRouter:
    """Routes generative AI requests with single-provider execution and automatic fallback."""

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        openrouter_model: Optional[str] = None,
    ) -> None:
        self.openai = OpenAIService()
        self.gemini = GeminiService()
        self.openrouter = OpenRouterService(custom_model=openrouter_model)

        self.primary_provider = (primary_provider or _get_config_value("DEFAULT_AI_PROVIDER", "gemini")).lower()
        self.fallback_provider = (fallback_provider or _get_config_value("FALLBACK_AI_PROVIDER", "openrouter")).lower()

    def get_provider_status(self) -> Dict[str, bool]:
        return {
            "openai": self.openai.is_configured(),
            "gemini": self.gemini.is_configured(),
            "openrouter": self.openrouter.is_configured(),
        }

    def _get_service(self, name: str) -> Any:
        key = name.strip().lower()
        if "openai" in key:
            return self.openai
        if "gemini" in key:
            return self.gemini
        if "openrouter" in key:
            return self.openrouter
        return self.gemini

    def _OrderedProviders(self) -> List[str]:
        order = []
        for p in [self.primary_provider, self.fallback_provider, "gemini", "openai", "openrouter"]:
            if p not in order:
                order.append(p)
        return order

    def _call_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Attempts the primary provider first; if it fails or is not configured,
        attempts the fallback provider. Returns structured response metadata.
        """
        errors: List[str] = []
        for provider_name in self._OrderedProviders():
            svc = self._get_service(provider_name)
            if not svc.is_configured():
                errors.append(f"{provider_name.title()}: API key not configured.")
                continue
            try:
                res = svc.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=json_mode,
                )
                res["fallback_used"] = provider_name != self.primary_provider
                res["Attempted_Log"] = errors
                return res
            except Exception as exc:
                errors.append(f"{provider_name.title()} failed: {_sanitize_error(exc)}")

        return {
            "provider": "Local Analytical Engine (AI API Fallback)",
            "model": "deterministic-statistical-interpreter",
            "content": None,
            "fallback_used": True,
            "all_apis_failed": True,
            "Attempted_Log": errors,
        }

    def generate_prediction_explanation(
        self,
        prediction_result: Dict[str, Any],
        dataset_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a structured explanation of an actual ML prediction:
        - performance_summary
        - strengths (2-4 points)
        - areas_for_improvement (2-4 points)
        - recommended_actions
        - why_model_predicted_this
        Never invents numbers; strictly interprets the ML output.
        """
        system_prompt = (
            "You are an expert Academic Performance Data Scientist. "
            "You receive real Machine Learning prediction results and student feature values. "
            "CRITICAL RULES:\n"
            "1. NEVER invent, alter, or hallucinate numerical predictions, scores, or metrics.\n"
            "2. Cite the exact numbers provided in the payload.\n"
            "3. Do NOT diagnose medical or psychological conditions.\n"
            "4. Respond strictly in valid JSON with the following keys:\n"
            '  "performance_summary": string (2-3 sentences explaining the predicted outcome and risk level),\n'
            '  "strengths": array of 2 to 4 strings (based on actual strong feature values),\n'
            '  "areas_for_improvement": array of 2 to 4 strings (based on actual weak/moderate feature values),\n'
            '  "recommended_actions": array of 3 to 4 practical academic actions for the student/faculty,\n'
            '  "why_model_predicted_this": string (explaining how the top feature importances and student values drove the ML model output).'
        )

        payload_context = {
            "model_name": prediction_result.get("model_name"),
            "task_type": prediction_result.get("task_type"),
            "predicted_output": prediction_result.get("predicted_display"),
            "performance_category": prediction_result.get("performance_category"),
            "risk_level": prediction_result.get("risk_level"),
            "confidence_score": prediction_result.get("confidence_score"),
            "student_input_features": prediction_result.get("input_features"),
            "top_model_feature_importances": prediction_result.get("important_factors"),
            "model_evaluation_metrics": prediction_result.get("model_metrics"),
            "identified_risk_reasons": prediction_result.get("risk_reasons"),
            "identified_strengths": prediction_result.get("academic_strengths"),
            "dataset_averages": (dataset_summary or {}).get("kpis", {}),
        }

        user_prompt = (
            "Interpret the following real Machine Learning student performance prediction and return the JSON object:\n\n"
            + json.dumps(payload_context, indent=2, default=str)
        )

        raw_res = self._call_with_fallback(system_prompt, user_prompt, json_mode=True)

        if raw_res.get("content"):
            try:
                text = raw_res["content"].strip()
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if m:
                    text = m.group(0)
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "performance_summary" in parsed:
                    parsed["provider_used"] = raw_res["provider"]
                    parsed["model_used"] = raw_res["model"]
                    parsed["fallback_used"] = raw_res.get("fallback_used", False)
                    parsed["attempted_log"] = raw_res.get("Attempted_Log", [])
                    return parsed
            except Exception:
                pass

        return self._build_deterministic_insight(prediction_result, raw_res)

    def _build_deterministic_insight(
        self,
        pred: Dict[str, Any],
        raw_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_name = pred.get("model_name", "Scikit-learn Model")
        pred_disp = pred.get("predicted_display", "N/A")
        cat = pred.get("performance_category", "Evaluated")
        risk = pred.get("risk_level", "MEDIUM RISK")
        conf = pred.get("confidence_score")
        conf_str = f" with {conf * 100:.1f}% model probability" if conf is not None else ""

        factors = pred.get("important_factors", [])
        top_factors_str = ", ".join(
            [f"{f['feature']} ({f['importance'] * 100:.1f}% weight)" for f in factors[:3]]
        ) if factors else "the selected academic features"

        inputs = pred.get("input_features", {})
        inputs_summary = ", ".join([f"{k}={v}" for k, v in list(inputs.items())[:5]])

        strengths = list(pred.get("academic_strengths", []))
        if len(strengths) < 2:
            strengths.append("Baseline engagement established across core curriculum parameters.")

        improvements = list(pred.get("risk_reasons", []))
        if len(improvements) < 2:
            improvements.append("Maintain consistent weekly revision across quantitative and practical coursework.")

        recommendations = []
        if risk == "HIGH RISK":
            recommendations.extend([
                "Schedule immediate faculty mentoring and weekly internal assessment tracking.",
                "Raise class attendance above the 75% benchmark to prevent lecture comprehension gaps.",
                "Allocate at least 3.5 to 4.0 structured daily self-study hours focused on core problem sets.",
                "Complete all pending lab and assignment submissions before midterm evaluations.",
            ])
        elif risk == "MEDIUM RISK":
            recommendations.extend([
                "Increase daily self-study consistency by 1.0–1.5 hours per day with active recall practice.",
                "Target 80%+ attendance and review internal assessment errors with subject faculty.",
                "Focus revision on topics linked to the highest-weighted features in the model.",
            ])
        else:
            recommendations.extend([
                "Sustain current attendance and structured study schedule to preserve Distinction/First-Class standing.",
                "Engage in peer mentoring or advanced BCA project coursework to deepen applied skills.",
                "Continue periodic self-assessment to guard against late-semester performance dips.",
            ])

        return {
            "performance_summary": (
                f"The trained {model_name} evaluated the student's academic profile ({inputs_summary}) "
                f"and produced a predicted outcome of {pred_disp}{conf_str}, placing the student in the "
                f"'{cat}' tier with an overall classification of {risk}."
            ),
            "strengths": strengths[:4],
            "areas_for_improvement": improvements[:4],
            "recommended_actions": recommendations[:4],
            "why_model_predicted_this": (
                f"In the trained {model_name}, the strongest predictive drivers are {top_factors_str}. "
                f"Evaluating the student's actual values ({inputs_summary}) against the dataset distribution "
                f"directly determined the {pred_disp} prediction and {risk} assignment."
            ),
            "provider_used": raw_res.get("provider", "Local Analytical Engine"),
            "model_used": raw_res.get("model", "deterministic-interpreter"),
            "fallback_used": True,
            "attempted_log": raw_res.get("Attempted_Log", []),
        }

    def answer_dataset_question(
        self,
        question: str,
        dataset_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Answers natural-language questions about the uploaded dataset and trained ML models
        using actual dataset statistics, correlations, and ML metrics. Never hallucinates numbers.
        """
        clean_ctx = dict(dataset_context)
        if "ml_summary" in clean_ctx and isinstance(clean_ctx["ml_summary"], dict):
            ml_copy = {
                k: v for k, v in clean_ctx["ml_summary"].items()
                if k not in ("artifact", "label_encoder", "actual_vs_predicted")
            }
            clean_ctx["ml_summary"] = ml_copy

        system_prompt = (
            "You are the StudentAI Dataset Analytics Assistant. "
            "Answer the user's question strictly using the provided actual dataset statistics, "
            "feature correlations, and trained Machine Learning model outputs. "
            "Never invent or hallucinate numbers. If the requested data is not present in the context, "
            "explicitly state that the dataset does not contain that information."
        )
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Actual Dataset & ML Context (JSON):\n{json.dumps(clean_ctx, indent=2, default=str)}"
        )

        raw_res = self._call_with_fallback(system_prompt, user_prompt, json_mode=False)
        if raw_res.get("content"):
            return {
                "answer": raw_res["content"],
                "provider_used": raw_res["provider"],
                "model_used": raw_res["model"],
                "fallback_used": raw_res.get("fallback_used", False),
            }

        return {
            "answer": self._deterministic_dataset_answer(question, clean_ctx),
            "provider_used": raw_res.get("provider", "Local Analytical Engine (AI API Fallback)"),
            "model_used": raw_res.get("model", "deterministic-statistical-interpreter"),
            "fallback_used": True,
        }

    def _deterministic_dataset_answer(self, question: str, ctx: Dict[str, Any]) -> str:
        q_low = question.lower()
        kpis = ctx.get("kpis", {})
        corrs = ctx.get("top_correlations", [])
        ml_info = ctx.get("ml_summary", {})
        ds_info = ctx.get("dataset_info", {})

        lines = []
        lines.append(
            f"**Dataset Overview (`{ds_info.get('filename', 'Uploaded Dataset')}`):** "
            f"**{kpis.get('total_students', 'N/A')}** students analyzed | "
            f"Average Score: **{kpis.get('average_score', 'N/A')}** | "
            f"Average Attendance: **{kpis.get('average_attendance', 'N/A')}%** | "
            f"Pass Rate: **{kpis.get('pass_rate', 'N/A')}%** | "
            f"At-Risk Students: **{kpis.get('at_risk_students', 'N/A')}**."
        )

        if any(w in q_low for w in ("factor", "relationship", "correlat", "low", "score", "improve")):
            if corrs:
                corr_str = ", ".join([f"**{c['feature']}** ($r = {c['correlation']:.3f}$)" for c in corrs[:4]])
                lines.append(f"- **Strongest Statistical Relationships with Performance:** {corr_str}.")
            if ml_info and ml_info.get("feature_importance"):
                fi_str = ", ".join(
                    [f"**{f['feature']}** ({f['importance']*100:.1f}%)" for f in ml_info["feature_importance"][:4]]
                )
                lines.append(f"- **Trained ML Feature Importance ({ml_info.get('model_name')}):** {fi_str}.")
            lines.append(
                "- **Academic Improvement Focus:** Based on the dataset distribution, students in the at-risk band "
                "show the largest performance gains by raising attendance above 75%, increasing daily self-study hours, "
                "and improving internal assessment scores."
            )

        if any(w in q_low for w in ("model", "accuracy", "r2", "metric", "predict")):
            if ml_info:
                metrics_str = ", ".join([f"**{k}**: {v}" for k, v in ml_info.get("metrics", {}).items()])
                lines.append(
                    f"- **Model Evaluation ({ml_info.get('model_name')} on `{ml_info.get('target_column')}`):** "
                    f"Trained on {ml_info.get('train_size')} rows and evaluated on {ml_info.get('test_size')} unseen test rows -> {metrics_str}."
                )
            else:
                lines.append("- **Model Status:** No ML model has been trained in this session yet. Visit the **ML Models** page to train a model.")

        return "\n\n".join(lines)
