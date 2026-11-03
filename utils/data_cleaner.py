"""
Data Cleaning and Dataset Inspection Module for StudentAI.
Performs genuine preprocessing: duplicate removal, type fixing, missing value imputation,
domain range validation, and IQR outlier clipping with transparent before/after reporting.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


def inspect_dataset(df: pd.DataFrame, filename: str = "uploaded_dataset.csv") -> Dict[str, Any]:
    """
    Computes 100% genuine statistics from the given DataFrame.
    Never hardcodes any numbers.
    """
    rows, cols = df.shape
    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

    col_profile = []
    for col in df.columns:
        series = df[col]
        missing_cnt = int(series.isna().sum())
        missing_pct = round((missing_cnt / rows) * 100, 2) if rows > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        sample_val = series.dropna().iloc[0] if series.dropna().shape[0] > 0 else "N/A"
        col_profile.append({
            "Column": col,
            "Data Type": str(series.dtype),
            "Category": "Numeric" if col in numeric_cols else "Categorical/Text",
            "Missing Values": missing_cnt,
            "Missing (%)": f"{missing_pct}%",
            "Unique Values": unique_cnt,
            "Sample Value": str(sample_val),
        })

    return {
        "filename": filename,
        "rows": int(rows),
        "columns": int(cols),
        "missing_values": total_missing,
        "missing_by_column": {k: int(v) for k, v in missing_by_col.items()},
        "duplicate_records": duplicate_rows,
        "dtypes": dtypes_map,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_profile": col_profile,
    }


def clean_dataset(
    df: pd.DataFrame,
    missing_strategy: str = "median_mode",
    remove_duplicates: bool = True,
    fix_data_types: bool = True,
    handle_outliers: bool = True,
    iqr_multiplier: float = 2.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs genuine data cleaning on a copy of the uploaded DataFrame.
    Returns (cleaned_df, cleaning_report) with exact counts of what changed.
    """
    work_df = df.copy()
    rows_before = int(len(work_df))
    missing_before = int(work_df.isna().sum().sum())
    changes_log: List[str] = []

    # 1. Strip whitespace from column names and string cells
    work_df.columns = [str(c).strip() for c in work_df.columns]
    for col in work_df.select_dtypes(include=["object"]).columns:
        work_df[col] = work_df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        # Convert common placeholder strings to actual NaN
        work_df[col] = work_df[col].replace(
            {"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "None": np.nan, "?": np.nan, "-": np.nan}
        )

    # 2. Remove completely empty rows (invalid records)
    empty_rows_mask = work_df.isna().all(axis=1)
    invalid_records_removed = int(empty_rows_mask.sum())
    if invalid_records_removed > 0:
        work_df = work_df[~empty_rows_mask].reset_index(drop=True)
        changes_log.append(f"Removed {invalid_records_removed} completely empty/invalid row(s).")

    # 3. Remove duplicate rows
    duplicates_removed = 0
    if remove_duplicates:
        dup_count = int(work_df.duplicated().sum())
        if dup_count > 0:
            work_df = work_df.drop_duplicates().reset_index(drop=True)
            duplicates_removed = dup_count
            changes_log.append(f"Removed {duplicates_removed} duplicate record(s).")

    # 4. Fix incorrect data types (convert numeric-like strings such as '85%' or ' 72.5 ')
    type_conversions: List[str] = []
    if fix_data_types:
        for col in list(work_df.columns):
            if work_df[col].dtype == "object":
                col_lower = col.lower()
                # Skip obvious ID or date columns
                if any(k in col_lower for k in ("id", "roll", "name", "date", "time", "email")):
                    continue
                cleaned_str = (
                    work_df[col]
                    .dropna()
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                if len(cleaned_str) > 0:
                    converted = pd.to_numeric(cleaned_str, errors="coerce")
                    valid_ratio = converted.notna().mean()
                    if valid_ratio >= 0.80:
                        work_df[col] = pd.to_numeric(
                            work_df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip(),
                            errors="coerce"
                        )
                        type_conversions.append(col)
        if type_conversions:
            changes_log.append(f"Converted {len(type_conversions)} column(s) to numeric type: {', '.join(type_conversions)}.")

    # 5. Handle missing values
    missing_before_imputation = int(work_df.isna().sum().sum())
    missing_handled_by_col: Dict[str, int] = {}

    if missing_before_imputation > 0:
        if missing_strategy == "drop_rows":
            before_drop = len(work_df)
            work_df = work_df.dropna().reset_index(drop=True)
            dropped_na_rows = before_drop - len(work_df)
            missing_values_handled = missing_before_imputation
            changes_log.append(f"Dropped {dropped_na_rows} row(s) containing {missing_values_handled} missing value(s).")
        else:
            numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in work_df.columns:
                col_na = int(work_df[col].isna().sum())
                if col_na == 0:
                    continue
                missing_handled_by_col[col] = col_na
                if col in numeric_cols:
                    if missing_strategy == "mean_mode":
                        fill_val = float(work_df[col].mean()) if work_df[col].notna().any() else 0.0
                    else:
                        fill_val = float(work_df[col].median()) if work_df[col].notna().any() else 0.0
                    work_df[col] = work_df[col].fillna(round(fill_val, 2))
                else:
                    mode_series = work_df[col].mode(dropna=True)
                    fill_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
                    work_df[col] = work_df[col].fillna(fill_val)
            missing_values_handled = sum(missing_handled_by_col.values())
            changes_log.append(
                f"Imputed {missing_values_handled} missing value(s) across {len(missing_handled_by_col)} column(s) using {missing_strategy.replace('_', ' / ')}."
            )
    else:
        missing_values_handled = 0

    # 6. Handle domain bounds & statistical outliers on numeric features
    outliers_adjusted = 0
    if handle_outliers:
        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            col_lower = col.lower()
            if any(k in col_lower for k in ("id", "roll", "year", "sem")):
                continue
            series = work_df[col]
            # Domain bounds for percentages/marks (0 to 100)
            if any(k in col_lower for k in ("attendance", "score", "mark", "percent", "pct")):
                out_mask = (series < 0) | (series > 100)
                cnt = int(out_mask.sum())
                if cnt > 0:
                    work_df[col] = series.clip(0.0, 100.0)
                    outliers_adjusted += cnt
                    series = work_df[col]
            elif "hour" in col_lower:
                out_mask = (series < 0) | (series > 20)
                cnt = int(out_mask.sum())
                if cnt > 0:
                    work_df[col] = series.clip(0.0, 20.0)
                    outliers_adjusted += cnt
                    series = work_df[col]

            # IQR clipping for extreme values
            if len(series.dropna()) >= 10:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = round(q1 - iqr_multiplier * iqr, 2)
                    upper_bound = round(q3 + iqr_multiplier * iqr, 2)
                    extreme_mask = (series < lower_bound) | (series > upper_bound)
                    ext_cnt = int(extreme_mask.sum())
                    if ext_cnt > 0:
                        work_df[col] = series.clip(lower_bound, upper_bound)
                        outliers_adjusted += ext_cnt

        if outliers_adjusted > 0:
            changes_log.append(f"Clipped {outliers_adjusted} extreme outlier value(s) to valid domain/IQR bounds.")

    rows_after = int(len(work_df))
    missing_after = int(work_df.isna().sum().sum())

    if not changes_log:
        changes_log.append("Dataset was already clean; no rows or cells required modification.")

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "invalid_records_removed": invalid_records_removed,
        "missing_before": missing_before,
        "missing_values_handled": int(missing_values_handled),
        "missing_after": missing_after,
        "missing_handled_by_col": missing_handled_by_col,
        "type_conversions": type_conversions,
        "outliers_adjusted": outliers_adjusted,
        "changes_log": changes_log,
    }

    return work_df, report
