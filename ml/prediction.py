"""
Single-Student and Batch Prediction Engine for StudentAI.
Executes genuine predictions using the trained ML model artifact,
calculates authentic probabilities only when supported by the model,
and runs academic risk evaluation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from utils.helpers import evaluate_academic_risk


def predict_student_performance(
    trained_model_bundle: Dict[str, Any],
    input_features: Dict[str, Any],
    role_to_col: Dict[str, str],
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs a real prediction using the trained ML model bundle on the user-provided student features.
    Never invents predictions or confidence scores.
    """
    if not trained_model_bundle or "artifact" not in trained_model_bundle:
        raise ValueError("No trained ML model is available. Please train a model on the ML Models page first.")

    feature_cols: List[str] = trained_model_bundle["feature_columns"]
    task_type: str = trained_model_bundle["task_type"]
    model_name: str = trained_model_bundle["model_name"]
    artifact: Dict[str, Any] = trained_model_bundle["artifact"]
    label_classes: Optional[List[str]] = trained_model_bundle.get("label_classes")

    # Build a single-row DataFrame matching the training feature columns
    row_dict = {}
    for col in feature_cols:
        row_dict[col] = input_features.get(col, np.nan)
    X_single = pd.DataFrame([row_dict])

    confidence_score: Optional[float] = None
    predicted_score: Optional[float] = None
    predicted_label: str = ""

    if artifact["type"] == "sklearn":
        pipeline = artifact["pipeline"]
        raw_pred = pipeline.predict(X_single)[0]
        if task_type == "classification":
            pred_idx = int(raw_pred)
            if label_classes and 0 <= pred_idx < len(label_classes):
                predicted_label = str(label_classes[pred_idx])
            else:
                predicted_label = str(raw_pred)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_single)[0]
                confidence_score = round(float(np.max(probs)), 4)
        else:
            predicted_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)
            predicted_label = f"{predicted_score:.1f} / 100"
    else:
        prep = artifact["preprocessor"]
        est = artifact["estimator"]
        X_mat = prep.transform(X_single)
        raw_pred = est.predict(X_mat)[0]
        if task_type == "classification":
            pred_idx = int(raw_pred)
            if label_classes and 0 <= pred_idx < len(label_classes):
                predicted_label = str(label_classes[pred_idx])
            else:
                predicted_label = str(raw_pred)
            if hasattr(est, "predict_proba"):
                probs = est.predict_proba(X_mat)[0]
                confidence_score = round(float(np.max(probs)), 4)
        else:
            predicted_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)
            predicted_label = f"{predicted_score:.1f} / 100"

    # Evaluate Academic Risk from actual prediction + input features
    risk_result = evaluate_academic_risk(
        input_features=input_features,
        role_to_col=role_to_col,
        predicted_score=predicted_score,
        predicted_label=predicted_label if task_type == "classification" else None,
        dataset_stats=dataset_stats,
    )

    important_factors = trained_model_bundle.get("feature_importance", [])[:5]

    return {
        "model_name": model_name,
        "task_type": task_type,
        "target_column": trained_model_bundle.get("target_column", ""),
        "input_features": input_features,
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "predicted_display": f"{predicted_score:.2f}" if predicted_score is not None else predicted_label,
        "confidence_score": confidence_score,
        "performance_category": risk_result["performance_category"],
        "risk_level": risk_result["risk_level"],
        "risk_reasons": risk_result["risk_reasons"],
        "academic_strengths": risk_result["academic_strengths"],
        "important_factors": important_factors,
        "model_metrics": trained_model_bundle.get("metrics", {}),
    }
