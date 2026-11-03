"""
Regression Models Module for StudentAI.
Implements Linear Regression and Random Forest Regressor using Scikit-learn Pipelines.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml.preprocessing import HAS_SKLEARN, build_sklearn_preprocessor, PureNumpyPreprocessor
from ml.evaluation import evaluate_regression_model, extract_pipeline_feature_importance

if HAS_SKLEARN:
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression


class _FallbackLinearRegressor:
    """Genuine Ordinary Least Squares (OLS) / Ridge Linear Regression via NumPy lstsq."""

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FallbackLinearRegressor":
        X_design = np.hstack([np.ones((len(X), 1)), X])
        # Small L2 regularization for numerical stability
        n_cols = X_design.shape[1]
        reg_eye = np.eye(n_cols) * 1e-4
        reg_eye[0, 0] = 0.0
        theta = np.linalg.pinv(X_design.T.dot(X_design) + reg_eye).dot(X_design.T).dot(y)
        self.bias = float(theta[0])
        self.weights = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    @property
    def coef_(self) -> np.ndarray:
        return self.weights


class _FallbackRandomForestRegressor:
    """Genuine ensemble of regression trees (depth-2) fitted on bootstrap samples."""

    def __init__(self, n_estimators: int = 60, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees: List[Dict[str, Any]] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def _fit_node(self, X_sub: np.ndarray, y_sub: np.ndarray, rng: np.random.RandomState, importances: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        if len(y_sub) < 6 or depth >= 3:
            return {"is_leaf": True, "val": float(np.mean(y_sub)) if len(y_sub) > 0 else 0.0}

        n_features = X_sub.shape[1]
        feat_candidates = rng.choice(n_features, size=max(1, int(np.ceil(n_features * 0.7))), replace=False)
        parent_var = float(np.var(y_sub))
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        for f_idx in feat_candidates:
            vals = X_sub[:, f_idx]
            for q in (0.2, 0.4, 0.6, 0.8):
                thresh = float(np.quantile(vals, q))
                left_m = vals <= thresh
                right_m = ~left_m
                if left_m.sum() < 2 or right_m.sum() < 2:
                    continue
                gain = parent_var - (
                    left_m.mean() * float(np.var(y_sub[left_m]))
                    + right_m.mean() * float(np.var(y_sub[right_m]))
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = thresh

        if best_feat == -1:
            return {"is_leaf": True, "val": float(np.mean(y_sub))}

        importances[best_feat] += best_gain
        left_m = X_sub[:, best_feat] <= best_thresh
        return {
            "is_leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._fit_node(X_sub[left_m], y_sub[left_m], rng, importances, depth + 1),
            "right": self._fit_node(X_sub[~left_m], y_sub[~left_m], rng, importances, depth + 1),
        }

    def _predict_node(self, node: Dict[str, Any], row: np.ndarray) -> float:
        if node["is_leaf"]:
            return node["val"]
        if row[node["feat"]] <= node["thresh"]:
            return self._predict_node(node["left"], row)
        return self._predict_node(node["right"], row)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FallbackRandomForestRegressor":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        importances = np.zeros(n_features)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = self._fit_node(X[idx], y[idx], rng, importances, depth=0)
            self.trees.append(tree)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else np.ones(n_features) / n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X))
        for i in range(len(X)):
            row_sum = 0.0
            for t in self.trees:
                row_sum += self._predict_node(t, X[i])
            preds[i] = row_sum / max(1, len(self.trees))
        return preds


def train_regression_model(
    split_data: Dict[str, Any],
    algorithm: str = "Random Forest Regressor",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a real regression model (Linear Regression or Random Forest Regressor)
    strictly on X_train, y_train and evaluates on unseen X_test, y_test.
    """
    X_train: pd.DataFrame = split_data["X_train"]
    X_test: pd.DataFrame = split_data["X_test"]
    y_train: np.ndarray = split_data["y_train"]
    y_test: np.ndarray = split_data["y_test"]
    feature_cols: List[str] = split_data["feature_cols"]

    if HAS_SKLEARN:
        preprocessor, num_cols, cat_cols = build_sklearn_preprocessor(X_train)
        if "linear" in algorithm.lower():
            estimator = LinearRegression()
            model_name = "Linear Regression"
        else:
            estimator = RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_split=4, random_state=random_state
            )
            model_name = "Random Forest Regressor"

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        feature_importance = extract_pipeline_feature_importance(pipeline, feature_cols, X_train, y_train)
        trained_artifact = {"type": "sklearn", "pipeline": pipeline}
    else:
        prep = PureNumpyPreprocessor().fit(X_train)
        X_tr_mat = prep.transform(X_train)
        X_te_mat = prep.transform(X_test)
        if "linear" in algorithm.lower():
            est = _FallbackLinearRegressor().fit(X_tr_mat, y_train)
            model_name = "Linear Regression"
        else:
            est = _FallbackRandomForestRegressor(n_estimators=70, random_state=random_state).fit(X_tr_mat, y_train)
            model_name = "Random Forest Regressor"

        y_pred = est.predict(X_te_mat)
        feature_importance = extract_pipeline_feature_importance(
            {"preprocessor": prep, "estimator": est}, feature_cols, X_train, y_train
        )
        trained_artifact = {"type": "fallback", "preprocessor": prep, "estimator": est}

    eval_results = evaluate_regression_model(y_test, y_pred)

    return {
        "model_name": model_name,
        "task_type": "regression",
        "target_column": split_data["target_col"],
        "feature_columns": feature_cols,
        "feature_metadata": split_data["feature_metadata"],
        "label_classes": None,
        "label_encoder": None,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": eval_results["metrics"],
        "confusion_matrix": None,
        "actual_vs_predicted": eval_results["actual_vs_predicted"],
        "feature_importance": feature_importance,
        "primary_metric_name": "R²",
        "primary_metric_value": float(eval_results["metrics"]["R²"]),
        "artifact": trained_artifact,
    }
