"""
ML Preprocessing and Feature Engineering Pipeline for StudentAI.
Prevents data leakage by fitting all imputers, scalers, and encoders strictly on the training split.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Determines whether a target column represents a 'classification' or 'regression' task
    based on data type and cardinality.
    """
    if target_col not in df.columns:
        raise ValueError(f"Selected target column '{target_col}' does not exist in dataset.")

    series = df[target_col].dropna()
    nunique = series.nunique()

    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    if nunique <= 6 and set(series.unique()).issubset({0, 1, 2, 3, 4, 5}):
        return "classification"

    return "regression"


def build_sklearn_preprocessor(X_train: pd.DataFrame) -> Tuple[Any, List[str], List[str]]:
    """
    Constructs a Scikit-learn ColumnTransformer that imputes and scales numeric features
    and imputes and one-hot encodes categorical features.
    Fitted ONLY on X_train to guarantee zero data leakage.
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    if not HAS_SKLEARN:
        return None, numeric_features, categorical_features

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", ohe),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


class PureNumpyPreprocessor:
    """
    Leakage-free preprocessor used only when scikit-learn is unavailable in minimal test environments.
    Fits medians, means, stds, and one-hot categories strictly on X_train.
    """

    def __init__(self) -> None:
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.medians: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.categories: Dict[str, List[str]] = {}
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "PureNumpyPreprocessor":
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = [c for c in X.columns if c not in self.numeric_features]
        self.feature_names_out_ = []

        for col in self.numeric_features:
            s = pd.to_numeric(X[col], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            s_filled = s.fillna(med)
            mean_v = float(s_filled.mean())
            std_v = float(s_filled.std(ddof=0))
            if std_v < 1e-8:
                std_v = 1.0
            self.medians[col] = med
            self.means[col] = mean_v
            self.stds[col] = std_v
            self.feature_names_out_.append(col)

        for col in self.categorical_features:
            s = X[col].fillna("Missing").astype(str)
            cats = sorted(s.unique().tolist())
            self.categories[col] = cats
            for cat_val in cats:
                self.feature_names_out_.append(f"{col}_{cat_val}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols_out = []
        n_rows = len(X)
        for col in self.numeric_features:
            if col in X.columns:
                s = pd.to_numeric(X[col], errors="coerce").fillna(self.medians[col])
            else:
                s = pd.Series(self.medians[col], index=range(n_rows))
            scaled = (s.values - self.means[col]) / self.stds[col]
            cols_out.append(scaled.reshape(-1, 1))

        for col in self.categorical_features:
            s = X[col].fillna("Missing").astype(str) if col in X.columns else pd.Series("Missing", index=range(n_rows))
            for cat_val in self.categories[col]:
                indicator = (s.values == cat_val).astype(float).reshape(-1, 1)
                cols_out.append(indicator)

        if not cols_out:
            return np.zeros((n_rows, 1))
        return np.hstack(cols_out)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Validates target and features, drops rows where target is missing,
    and performs a reproducible train/test split BEFORE fitting any preprocessing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if not valid_features:
        raise ValueError("Please select at least one valid feature column distinct from the target column.")

    clean_sub = df[valid_features + [target_col]].copy()
    clean_sub = clean_sub.dropna(subset=[target_col]).reset_index(drop=True)

    if len(clean_sub) < 12:
        raise ValueError(f"Insufficient valid rows ({len(clean_sub)}) after removing missing target values. Need at least 12 rows.")

    X = clean_sub[valid_features].copy()
    y_raw = clean_sub[target_col].copy()
    task_type = detect_task_type(clean_sub, target_col)

    label_classes: Optional[List[str]] = None
    label_encoder = None

    if task_type == "classification":
        y_str = y_raw.astype(str).str.strip()
        if y_str.nunique() < 2:
            raise ValueError(f"Target column '{target_col}' has only 1 unique class. Classification requires at least 2 classes.")
        if HAS_SKLEARN:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_str)
            label_classes = [str(c) for c in label_encoder.classes_]
        else:
            unique_sorted = sorted(y_str.unique().tolist())
            # Prefer putting 'Pass' or positive class as index 1 for binary classification
            if len(unique_sorted) == 2 and unique_sorted[0].lower() in ("pass", "passed", "true", "1", "yes"):
                unique_sorted = [unique_sorted[1], unique_sorted[0]]
            label_classes = unique_sorted
            class_to_idx = {c: i for i, c in enumerate(label_classes)}
            y = np.array([class_to_idx[val] for val in y_str], dtype=int)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y_num.notna()
        X = X[valid_mask].reset_index(drop=True)
        y = y_num[valid_mask].values.astype(float)
        if len(y) < 12:
            raise ValueError("Insufficient numeric target rows for regression training.")

    # Compute feature statistics on X for UI sliders/defaults
    feature_metadata: Dict[str, Dict[str, Any]] = {}
    for col in valid_features:
        if pd.api.types.is_numeric_dtype(X[col]):
            s = X[col].dropna()
            feature_metadata[col] = {
                "type": "numeric",
                "min": float(s.min()) if len(s) else 0.0,
                "max": float(s.max()) if len(s) else 100.0,
                "median": float(s.median()) if len(s) else 50.0,
                "mean": float(s.mean()) if len(s) else 50.0,
            }
        else:
            s = X[col].dropna().astype(str)
            options = sorted(s.unique().tolist())
            feature_metadata[col] = {
                "type": "categorical",
                "options": options if options else ["Unknown"],
                "default": s.mode().iloc[0] if len(s) else "Unknown",
            }

    if HAS_SKLEARN:
        stratify_arg = y if (task_type == "classification" and pd.Series(y).value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )
    else:
        rng = np.random.RandomState(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        n_test = max(2, int(round(len(X) * test_size)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": valid_features,
        "label_classes": label_classes,
        "label_encoder": label_encoder,
        "feature_metadata": feature_metadata,
    }
