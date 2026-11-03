"""
Validators and Smart Column Detection Module for StudentAI.
Handles CSV validation, untrusted input checks, and intelligent educational column mapping.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


# Canonical educational schema roles and their common alias patterns
CANONICAL_COLUMN_PATTERNS: Dict[str, List[str]] = {
    "student_id": [
        r"^student[_\s-]?id$", r"^roll[_\s-]?no$", r"^roll[_\s-]?number$",
        r"^uid$", r"^enrollment[_\s-]?no$", r"^reg[_\s-]?no$", r"^id$",
        r"^unnamed.*", r"^index$"
    ],
    "attendance": [
        r"attendance", r"attend[_\s-]?pct", r"attendance[_\s-]?percentage",
        r"attendance[_\s-]?rate", r"present[_\s-]?pct", r"classes[_\s-]?attended"
    ],
    "study_hours": [
        r"study[_\s-]?hours?", r"studyhours?", r"hours[_\s-]?studied",
        r"daily[_\s-]?study", r"weekly[_\s-]?study", r"self[_\s-]?study"
    ],
    "assignment_score": [
        r"assignment[_\s-]?scores?", r"assignment[_\s-]?marks?", r"assignments?",
        r"homework[_\s-]?scores?", r"project[_\s-]?scores?", r"lab[_\s-]?scores?"
    ],
    "internal_marks": [
        r"internal[_\s-]?marks?", r"internal[_\s-]?scores?", r"internals?",
        r"midterm[_\s-]?scores?", r"mid[_\s-]?sem", r"sessional[_\s-]?marks?",
        r"cia[_\s-]?marks?"
    ],
    "previous_score": [
        r"previous[_\s-]?scores?", r"previous[_\s-]?marks?", r"prev[_\s-]?scores?",
        r"past[_\s-]?scores?", r"prior[_\s-]?grade", r"last[_\s-]?sem[_\s-]?marks?",
        r"previous[_\s-]?percentage"
    ],
    "final_score": [
        r"final[_\s-]?scores?", r"final[_\s-]?marks?", r"total[_\s-]?scores?",
        r"total[_\s-]?marks?", r"exam[_\s-]?scores?", r"overall[_\s-]?scores?",
        r"grade[_\s-]?points?", r"percentage", r"marks[_\s-]?obtained", r"^score$"
    ],
    "result": [
        r"^results?$", r"^status$", r"pass[_\s-]?fail", r"outcome",
        r"grade[_\s-]?status", r"performance[_\s-]?status", r"qualified"
    ],
    "gender": [
        r"^gender$", r"^sex$"
    ],
    "course": [
        r"^course$", r"^program$", r"^branch$", r"^department$", r"^stream$", r"^major$"
    ],
    "semester": [
        r"^semester$", r"^sem$", r"^term$", r"^year$"
    ],
    "subject": [
        r"^subject$", r"^paper$", r"^module$", r"^topic$"
    ],
    "date": [
        r"date", r"timestamp", r"exam[_\s-]?date", r"assessment[_\s-]?date", r"recorded[_\s-]?at"
    ]
}


def normalize_col_name(col: str) -> str:
    """Normalizes a column name (camelCase, spaces, symbols) for matching."""
    if not isinstance(col, str):
        col = str(col)
    # Split camelCase (e.g., studyHours -> study_hours)
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col.strip())
    s2 = re.sub(r"[^a-zA-Z0-9]+", "_", s1).strip("_").lower()
    return s2


def validate_uploaded_dataframe(df: Optional[pd.DataFrame], filename: str = "dataset.csv") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates a loaded DataFrame for safety, size, and suitability for analysis & ML.
    Returns (is_valid, message, metadata).
    """
    if df is None:
        return False, "No dataset was loaded. Please upload a valid CSV file.", {}

    if not isinstance(df, pd.DataFrame):
        return False, "Uploaded object could not be parsed as a tabular dataset.", {}

    if df.empty or len(df.columns) == 0:
        return False, f"The uploaded file '{filename}' is empty and contains no records.", {}

    if len(df) < 10:
        return (
            False,
            f"The dataset '{filename}' has only {len(df)} rows. At least 10 student records are required for statistical analysis and machine learning train/test splitting.",
            {"rows": len(df), "columns": len(df.columns)}
        )

    if len(df.columns) < 2:
        return (
            False,
            f"The dataset '{filename}' contains only {len(df.columns)} column. At least 1 feature column and 1 target column are required.",
            {"rows": len(df), "columns": len(df.columns)}
        )

    # Check if there is at least one numeric column (or convertible numeric column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        # Check if any object columns can be parsed as numeric
        convertible = 0
        for col in df.columns:
            parsed = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
            if parsed.notna().mean() >= 0.6:
                convertible += 1
        if convertible == 0:
            return (
                False,
                f"The dataset '{filename}' contains no numeric columns suitable for quantitative analysis or ML modeling.",
                {"rows": len(df), "columns": len(df.columns)}
            )

    return True, "Dataset validated successfully.", {
        "filename": filename,
        "rows": int(len(df)),
        "columns": int(len(df.columns))
    }


def smart_detect_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligently detects canonical educational columns in any uploaded student dataset.
    Supports variations like 'Attendance', 'attendance_percentage', 'studyHours',
    'Study Hours', 'internal marks', 'Final Score', etc.
    Returns:
      - role_to_col: Dict mapping canonical role -> actual column name in df
      - col_to_role: Dict mapping actual column name -> canonical role
      - suggested_regression_target: Optional[str]
      - suggested_classification_target: Optional[str]
      - suggested_features: List[str]
      - detection_confidence: 'high' | 'medium' | 'low'
      - id_columns: List[str]
    """
    role_to_col: Dict[str, str] = {}
    col_to_role: Dict[str, str] = {}

    columns = list(df.columns)
    normalized_map = {col: normalize_col_name(col) for col in columns}

    # First pass: exact/regex pattern matching on normalized column names
    for role, patterns in CANONICAL_COLUMN_PATTERNS.items():
        for col in columns:
            if col in col_to_role:
                continue
            norm_col = normalized_map[col]
            raw_lower = str(col).strip().lower()
            for pat in patterns:
                if re.search(pat, norm_col) or re.search(pat, raw_lower):
                    role_to_col[role] = col
                    col_to_role[col] = role
                    break
            if role in role_to_col:
                break

    # Identify ID-like columns to exclude from ML features
    id_columns: List[str] = []
    if "student_id" in role_to_col:
        id_columns.append(role_to_col["student_id"])
    for col in columns:
        norm_col = normalized_map[col]
        if col not in id_columns and (norm_col.endswith("_id") or norm_col in ("id", "roll_no", "name", "student_name", "email")):
            id_columns.append(col)

    # Determine suggested regression target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    suggested_reg_target: Optional[str] = role_to_col.get("final_score")
    if not suggested_reg_target:
        # Look for any numeric column with score/marks/grade/gpa in name
        for col in numeric_cols:
            if col not in id_columns and any(k in normalized_map[col] for k in ("score", "mark", "grade", "gpa", "total", "pct", "percent")):
                suggested_reg_target = col
                break
    if not suggested_reg_target and numeric_cols:
        non_id_num = [c for c in numeric_cols if c not in id_columns]
        if non_id_num:
            suggested_reg_target = non_id_num[-1]

    # Determine suggested classification target
    suggested_clf_target: Optional[str] = role_to_col.get("result")
    if not suggested_clf_target:
        for col in columns:
            if col in id_columns:
                continue
            nunique = df[col].dropna().nunique()
            if 2 <= nunique <= 6 and (
                df[col].dtype == "object"
                or str(df[col].dtype) == "category"
                or any(k in normalized_map[col] for k in ("result", "pass", "status", "grade", "class", "target", "outcome"))
            ):
                suggested_clf_target = col
                break

    # Determine suggested features
    core_feature_roles = ["attendance", "study_hours", "assignment_score", "internal_marks", "previous_score"]
    detected_core_features = [role_to_col[r] for r in core_feature_roles if r in role_to_col]

    exclude_from_features = set(id_columns)
    if suggested_reg_target:
        exclude_from_features.add(suggested_reg_target)
    if suggested_clf_target:
        exclude_from_features.add(suggested_clf_target)
    if "date" in role_to_col:
        exclude_from_features.add(role_to_col["date"])

    all_candidate_features = [
        c for c in columns
        if c not in exclude_from_features and df[c].dropna().nunique() > 1
    ]

    # Order features so detected core academic features appear first
    suggested_features = list(detected_core_features)
    for c in all_candidate_features:
        if c not in suggested_features:
            suggested_features.append(c)

    # Calculate confidence
    matched_core_count = len(detected_core_features) + (1 if "final_score" in role_to_col or "result" in role_to_col else 0)
    if matched_core_count >= 4:
        confidence = "high"
    elif matched_core_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "role_to_col": role_to_col,
        "col_to_role": col_to_role,
        "id_columns": id_columns,
        "suggested_regression_target": suggested_reg_target,
        "suggested_classification_target": suggested_clf_target,
        "suggested_features": suggested_features,
        "detection_confidence": confidence,
        "matched_core_count": matched_core_count,
    }
