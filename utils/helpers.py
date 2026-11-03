"""
Helper functions, Dashboard KPI calculators, Academic Risk Analysis Engine,
and BCA Viva explanations for StudentAI.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def compute_dashboard_kpis(
    df: Optional[pd.DataFrame],
    role_to_col: Dict[str, str],
    best_model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes 100% genuine dashboard KPIs from the loaded DataFrame and trained ML models.
    Never uses hardcoded values.
    """
    if df is None or df.empty:
        return {
            "has_data": False,
            "total_students": None,
            "average_score": None,
            "average_attendance": None,
            "pass_rate": None,
            "at_risk_students": None,
            "best_model_score": None,
            "best_model_label": "Not Trained Yet",
        }

    total_students = int(len(df))

    # 1. Average Score
    score_col = role_to_col.get("final_score")
    if not score_col:
        # fallback to internal_marks or previous_score if final_score isn't mapped
        score_col = role_to_col.get("internal_marks") or role_to_col.get("previous_score")
    avg_score: Optional[float] = None
    if score_col and score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        avg_score = round(float(df[score_col].dropna().mean()), 2)

    # 2. Average Attendance
    att_col = role_to_col.get("attendance")
    avg_attendance: Optional[float] = None
    if att_col and att_col in df.columns and pd.api.types.is_numeric_dtype(df[att_col]):
        avg_attendance = round(float(df[att_col].dropna().mean()), 2)

    # 3. Pass Rate
    res_col = role_to_col.get("result")
    pass_rate: Optional[float] = None
    if res_col and res_col in df.columns:
        series_str = df[res_col].dropna().astype(str).str.strip().str.lower()
        if len(series_str) > 0:
            pass_mask = series_str.isin(["pass", "passed", "1", "true", "qualified", "p", "yes"])
            pass_rate = round(float(pass_mask.mean() * 100.0), 2)
    elif score_col and score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        valid_scores = df[score_col].dropna()
        if len(valid_scores) > 0:
            pass_rate = round(float((valid_scores >= 50.0).mean() * 100.0), 2)

    # 4. At-Risk Students count (based on real dataset rows)
    at_risk_mask = pd.Series(False, index=df.index)
    if score_col and score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        at_risk_mask = at_risk_mask | (df[score_col] < 50.0)
    if att_col and att_col in df.columns and pd.api.types.is_numeric_dtype(df[att_col]):
        at_risk_mask = at_risk_mask | (df[att_col] < 65.0)
    if res_col and res_col in df.columns:
        fail_mask = df[res_col].astype(str).str.strip().str.lower().isin(["fail", "failed", "0", "false", "f", "no"])
        at_risk_mask = at_risk_mask | fail_mask

    at_risk_students = int(at_risk_mask.sum())

    # 5. Best Model Score
    best_model_score: Optional[str] = None
    best_model_label = "Train Model in ML Page"
    if best_model_info and isinstance(best_model_info, dict):
        m_name = best_model_info.get("model_name", "Trained Model")
        metric_name = best_model_info.get("primary_metric_name", "Score")
        metric_val = best_model_info.get("primary_metric_value")
        if metric_val is not None:
            if metric_name == "Accuracy":
                best_model_score = f"{round(float(metric_val) * 100, 1)}%"
            else:
                best_model_score = f"{round(float(metric_val), 3)} ({metric_name})"
            best_model_label = m_name

    return {
        "has_data": True,
        "total_students": total_students,
        "average_score": avg_score,
        "average_attendance": avg_attendance,
        "pass_rate": pass_rate,
        "at_risk_students": at_risk_students,
        "best_model_score": best_model_score,
        "best_model_label": best_model_label,
    }


def evaluate_academic_risk(
    input_features: Dict[str, Any],
    role_to_col: Dict[str, str],
    predicted_score: Optional[float] = None,
    predicted_label: Optional[str] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates student academic risk level (LOW RISK, MEDIUM RISK, HIGH RISK)
    strictly from actual model predictions and academic feature values.
    Never diagnoses medical or psychological conditions.
    """
    reasons: List[str] = []
    strengths: List[str] = []
    risk_points = 0

    # Helper to look up canonical role value from input_features
    def get_role_val(role_key: str) -> Optional[float]:
        col_name = role_to_col.get(role_key)
        if col_name and col_name in input_features:
            try:
                return float(input_features[col_name])
            except (ValueError, TypeError):
                return None
        # Fallback: direct key match
        for k, v in input_features.items():
            if role_key in k.lower():
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
        return None

    attendance = get_role_val("attendance")
    study_hours = get_role_val("study_hours")
    internal_marks = get_role_val("internal_marks")
    assignment_score = get_role_val("assignment_score")
    previous_score = get_role_val("previous_score")

    # 1. Check predicted score or classification outcome
    if predicted_score is not None:
        if predicted_score < 50.0:
            risk_points += 3
            reasons.append(f"Predicted final score ({predicted_score:.1f}) falls below the standard passing threshold of 50.0.")
        elif predicted_score < 65.0:
            risk_points += 1
            reasons.append(f"Predicted final score ({predicted_score:.1f}) is in the borderline/moderate academic band (50–65).")
        else:
            strengths.append(f"Strong predicted academic score ({predicted_score:.1f}).")

    if predicted_label is not None:
        lbl_lower = str(predicted_label).strip().lower()
        if lbl_lower in ("fail", "failed", "0", "false", "at-risk", "low"):
            risk_points += 3
            reasons.append(f"Classification model predicts an outcome of '{predicted_label}'.")
        elif lbl_lower in ("pass", "passed", "1", "true"):
            strengths.append(f"Classification model predicts a '{predicted_label}' outcome.")

    # 2. Check Attendance
    if attendance is not None:
        if attendance < 60.0:
            risk_points += 2
            reasons.append(f"Critically low class attendance ({attendance:.1f}% < 60% minimum benchmark).")
        elif attendance < 75.0:
            risk_points += 1
            reasons.append(f"Attendance ({attendance:.1f}%) is below the standard 75% collegiate requirement.")
        else:
            strengths.append(f"Consistent class attendance ({attendance:.1f}%).")

    # 3. Check Internal Marks
    if internal_marks is not None:
        if internal_marks < 45.0:
            risk_points += 2
            reasons.append(f"Low internal assessment marks ({internal_marks:.1f}), indicating gaps in continuous evaluation.")
        elif internal_marks < 60.0:
            risk_points += 1
            reasons.append(f"Moderate internal assessment marks ({internal_marks:.1f}) with room for improvement.")
        else:
            strengths.append(f"Solid internal assessment performance ({internal_marks:.1f}).")

    # 4. Check Study Hours
    if study_hours is not None:
        if study_hours < 2.0:
            risk_points += 2
            reasons.append(f"Low self-study allocation ({study_hours:.1f} hrs/day < 2.0 hrs/day recommended minimum).")
        elif study_hours < 3.5:
            risk_points += 1
            reasons.append(f"Moderate self-study schedule ({study_hours:.1f} hrs/day); increasing to 4+ hrs/day can boost retention.")
        else:
            strengths.append(f"Dedicated self-study routine ({study_hours:.1f} hrs/day).")

    # 5. Check Assignment Score
    if assignment_score is not None:
        if assignment_score < 50.0:
            risk_points += 1
            reasons.append(f"Low assignment completion score ({assignment_score:.1f}).")
        elif assignment_score >= 70.0:
            strengths.append(f"High coursework and assignment score ({assignment_score:.1f}).")

    # 6. Check Previous Score
    if previous_score is not None:
        if previous_score < 50.0:
            risk_points += 1
            reasons.append(f"Low prior academic baseline score ({previous_score:.1f}).")
        elif previous_score >= 70.0:
            strengths.append(f"Strong prior academic foundation ({previous_score:.1f}).")

    # Determine Risk Level
    if risk_points >= 4 or (predicted_score is not None and predicted_score < 48.0) or (
        predicted_label is not None and str(predicted_label).strip().lower() in ("fail", "failed", "0")
    ):
        risk_level = "HIGH RISK"
    elif risk_points >= 2 or (predicted_score is not None and predicted_score < 65.0):
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    # Determine Performance Category
    if predicted_score is not None:
        if predicted_score >= 80.0:
            performance_category = "Distinction / Excellent"
        elif predicted_score >= 65.0:
            performance_category = "First Class / Good"
        elif predicted_score >= 50.0:
            performance_category = "Second Class / Satisfactory"
        else:
            performance_category = "Needs Academic Intervention (Below Pass)"
    else:
        if risk_level == "LOW RISK":
            performance_category = "Proficient / On Track"
        elif risk_level == "MEDIUM RISK":
            performance_category = "Moderate / Needs Monitoring"
        else:
            performance_category = "At-Risk / Needs Academic Support"

    if not reasons:
        reasons.append("All evaluated academic indicators meet or exceed standard institutional benchmarks.")

    if not strengths:
        strengths.append("Opportunity to establish positive momentum through structured study habits and regular attendance.")

    return {
        "risk_level": risk_level,
        "risk_points": risk_points,
        "performance_category": performance_category,
        "risk_reasons": reasons,
        "academic_strengths": strengths,
    }


def get_viva_explanations() -> Dict[str, str]:
    """
    Returns clear, accurate BCA Viva explanations for ML metrics and concepts.
    """
    return {
        "Accuracy": "The proportion of total predictions (both Pass and Fail) that the classification model predicted correctly: (TP + TN) / Total.",
        "Precision": "Out of all students predicted as Positive/Pass by the model, what percentage actually passed: TP / (TP + FP).",
        "Recall": "Out of all students who actually passed in reality, what percentage did the model correctly identify: TP / (TP + FN).",
        "F1 Score": "The harmonic mean of Precision and Recall (2 * Precision * Recall / (Precision + Recall)), balancing false positives and false negatives.",
        "Confusion Matrix": "A 2x2 table showing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) on the unseen test set.",
        "MAE (Mean Absolute Error)": "The average absolute difference between actual student scores and predicted scores. Lower MAE means more accurate numerical predictions.",
        "RMSE (Root Mean Squared Error)": "The square root of the average squared prediction errors. Penalizes large prediction mistakes more heavily than MAE.",
        "R² (Coefficient of Determination)": "Measures the proportion of variance in the student score explained by the model features (from 0 to 1, where closer to 1.0 indicates stronger fit).",
        "Train/Test Split": "Dividing the dataset into a training set (e.g., 80%) to fit the model and a separate unseen test set (e.g., 20%) to evaluate generalization and prevent data leakage.",
        "Overfitting": "When a model memorizes noise in the training data and achieves high training score but performs poorly on unseen test data.",
    }
