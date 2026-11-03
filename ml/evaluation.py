"""
Model Evaluation and Feature Importance Extraction Module for StudentAI.
Computes genuine Classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
and Regression metrics (MAE, RMSE, R²), plus Actual vs Predicted and Feature Importance.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes genuine Accuracy, Precision, Recall, F1 Score, Confusion Matrix,
    and Actual vs Predicted records on the test set.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(unique_labels) <= 2

    if HAS_SKLEARN_METRICS:
        acc = float(accuracy_score(y_true, y_pred))
        avg_mode = "binary" if is_binary else "weighted"
        pos_lbl = unique_labels[-1] if is_binary else 1
        prec = float(precision_score(y_true, y_pred, average=avg_mode, pos_label=pos_lbl, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=avg_mode, pos_label=pos_lbl, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=avg_mode, pos_label=pos_lbl, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        acc = float(np.mean(y_true == y_pred))
        pos_class = unique_labels[-1]
        tp = int(np.sum((y_true == pos_class) & (y_pred == pos_class)))
        fp = int(np.sum((y_true != pos_class) & (y_pred == pos_class)))
        fn = int(np.sum((y_true == pos_class) & (y_pred != pos_class)))
        tn = int(np.sum((y_true != pos_class) & (y_pred != pos_class)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        cm = [[tn, fp], [fn, tp]]

    def _map_lbl(val: int) -> str:
        if label_classes and 0 <= int(val) < len(label_classes):
            return str(label_classes[int(val)])
        return str(val)

    actual_vs_pred = []
    for i in range(min(len(y_true), 100)):
        act_str = _map_lbl(int(y_true[i]))
        prd_str = _map_lbl(int(y_pred[i]))
        actual_vs_pred.append({
            "Sample_Index": i + 1,
            "Actual": act_str,
            "Predicted": prd_str,
            "Match": "Correct" if act_str == prd_str else "Mismatch",
        })

    return {
        "metrics": {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
        },
        "confusion_matrix": {
            "matrix": cm,
            "labels": label_classes if label_classes else [str(u) for u in unique_labels],
        },
        "actual_vs_predicted": actual_vs_pred,
    }


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes genuine MAE, RMSE, R², and Actual vs Predicted pairs on the test set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if HAS_SKLEARN_METRICS:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
    else:
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    actual_vs_pred = []
    for i in range(min(len(y_true), 100)):
        act_v = round(float(y_true[i]), 2)
        prd_v = round(float(y_pred[i]), 2)
        actual_vs_pred.append({
            "Sample_Index": i + 1,
            "Actual": act_v,
            "Predicted": prd_v,
            "Residual_Error": round(act_v - prd_v, 2),
        })

    return {
        "metrics": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
        },
        "actual_vs_predicted": actual_vs_pred,
    }


def extract_pipeline_feature_importance(
    pipeline_or_dict: Any,
    original_feature_cols: List[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Extracts normalized feature importance mapped back to the original feature names.
    Supports tree-based `feature_importances_` and linear `coef_` weights.
    """
    raw_scores: Dict[str, float] = {col: 0.0 for col in original_feature_cols}

    try:
        if isinstance(pipeline_or_dict, dict):
            prep = pipeline_or_dict["preprocessor"]
            est = pipeline_or_dict["estimator"]
            transformed_names = prep.feature_names_out_
        else:
            prep = pipeline_or_dict.named_steps["preprocessor"]
            est = pipeline_or_dict.named_steps["estimator"]
            transformed_names = []
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in X_train.columns if c not in num_cols]
            transformed_names.extend(num_cols)
            if cat_cols and "cat" in prep.named_transformers_:
                ohe = prep.named_transformers_["cat"].named_steps["encoder"]
                for col_idx, col_name in enumerate(cat_cols):
                    for cat_val in ohe.categories_[col_idx]:
                        transformed_names.append(f"{col_name}_{cat_val}")

        if hasattr(est, "feature_importances_") and est.feature_importances_ is not None:
            imp_vec = np.abs(np.asarray(est.feature_importances_, dtype=float).flatten())
        elif hasattr(est, "coef_") and est.coef_ is not None:
            coef_arr = np.asarray(est.coef_, dtype=float)
            imp_vec = np.mean(np.abs(coef_arr), axis=0) if coef_arr.ndim > 1 else np.abs(coef_arr)
        else:
            imp_vec = np.ones(len(transformed_names))

        # Aggregate one-hot features back to original column name
        for idx, t_name in enumerate(transformed_names):
            if idx >= len(imp_vec):
                break
            val = float(imp_vec[idx])
            if t_name in raw_scores:
                raw_scores[t_name] += val
            else:
                for orig_col in original_feature_cols:
                    if t_name.startswith(f"{orig_col}_"):
                        raw_scores[orig_col] += val
                        break
    except Exception:
        # Fallback: absolute correlation with target on training data
        for col in original_feature_cols:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                c_val = np.corrcoef(X_train[col].fillna(0).values, y_train)[0, 1]
                raw_scores[col] = abs(float(c_val)) if not np.isnan(c_val) else 0.05
            else:
                raw_scores[col] = 0.05

    total_imp = sum(raw_scores.values())
    if total_imp <= 0:
        total_imp = 1.0

    sorted_items = sorted(
        [{"feature": k, "importance": round(v / total_imp, 4)} for k, v in raw_scores.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return sorted_items
