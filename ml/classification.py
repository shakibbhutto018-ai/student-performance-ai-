"""
Classification Models Module for StudentAI.
Implements Random Forest Classifier and Logistic Regression using Scikit-learn Pipelines.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml.preprocessing import HAS_SKLEARN, build_sklearn_preprocessor, PureNumpyPreprocessor
from ml.evaluation import evaluate_classification_model, extract_pipeline_feature_importance

if HAS_SKLEARN:
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression


class _FallbackLogisticClassifier:
    """Genuine L2-regularized Logistic Regression classifier via gradient descent for fallback."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.classes_: np.ndarray = np.array([0, 1])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FallbackLogisticClassifier":
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[-1]).astype(float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        lr = 0.15
        reg = 0.01
        for _ in range(400):
            linear = X.dot(self.weights) + self.bias
            preds = self._sigmoid(linear)
            err = preds - y_bin
            dw = (1.0 / n_samples) * X.T.dot(err) + reg * self.weights
            db = (1.0 / n_samples) * float(np.sum(err))
            self.weights -= lr * dw
            self.bias -= lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._sigmoid(X.dot(self.weights) + self.bias)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        idx = (probs >= 0.5).astype(int)
        return np.where(idx == 1, self.classes_[-1], self.classes_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.weights.reshape(1, -1)


class _FallbackRandomForestClassifier:
    """Genuine bootstrapped ensemble of decision stumps/trees for fallback environments."""

    def __init__(self, n_estimators: int = 60, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees: List[Dict[str, Any]] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FallbackRandomForestClassifier":
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[-1]).astype(float)
        n_samples, n_features = X.shape
        importances = np.zeros(n_features)
        self.trees = []

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_sub = X[sample_idx]
            y_sub = y_bin[sample_idx]
            feat_candidates = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

            best_gain = -1.0
            best_tree = {"feat": 0, "thresh": 0.0, "left_prob": 0.5, "right_prob": 0.5}
            parent_var = float(np.var(y_sub)) if len(y_sub) > 0 else 0.0

            for f_idx in feat_candidates:
                vals = X_sub[:, f_idx]
                for q in (0.25, 0.5, 0.75):
                    thresh = float(np.quantile(vals, q))
                    left_mask = vals <= thresh
                    right_mask = ~left_mask
                    if left_mask.sum() < 2 or right_mask.sum() < 2:
                        continue
                    w_left = left_mask.mean()
                    w_right = right_mask.mean()
                    gain = parent_var - (w_left * np.var(y_sub[left_mask]) + w_right * np.var(y_sub[right_mask]))
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_tree = {
                            "feat": int(f_idx),
                            "thresh": thresh,
                            "left_prob": float(y_sub[left_mask].mean()),
                            "right_prob": float(y_sub[right_mask].mean()),
                        }
            importances[best_tree["feat"]] += max(0.0, best_gain)
            self.trees.append(best_tree)

        total_imp = importances.sum()
        self.feature_importances_ = importances / total_imp if total_imp > 0 else np.ones(n_features) / n_features
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X))
        for t in self.trees:
            f_vals = X[:, t["feat"]]
            preds += np.where(f_vals <= t["thresh"], t["left_prob"], t["right_prob"])
        p1 = preds / max(1, len(self.trees))
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        idx = (probs >= 0.5).astype(int)
        return np.where(idx == 1, self.classes_[-1], self.classes_[0])


def train_classification_model(
    split_data: Dict[str, Any],
    algorithm: str = "Random Forest Classifier",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a real classification model (Random Forest Classifier or Logistic Regression)
    strictly on X_train, y_train and evaluates on unseen X_test, y_test.
    """
    X_train: pd.DataFrame = split_data["X_train"]
    X_test: pd.DataFrame = split_data["X_test"]
    y_train: np.ndarray = split_data["y_train"]
    y_test: np.ndarray = split_data["y_test"]
    feature_cols: List[str] = split_data["feature_cols"]
    label_classes: List[str] = split_data.get("label_classes") or ["Fail", "Pass"]

    if HAS_SKLEARN:
        preprocessor, num_cols, cat_cols = build_sklearn_preprocessor(X_train)
        if "logistic" in algorithm.lower():
            estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            model_name = "Logistic Regression"
        else:
            estimator = RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_split=4, random_state=random_state
            )
            model_name = "Random Forest Classifier"

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        feature_importance = extract_pipeline_feature_importance(pipeline, feature_cols, X_train, y_train)
        trained_artifact = {"type": "sklearn", "pipeline": pipeline}
    else:
        prep = PureNumpyPreprocessor().fit(X_train)
        X_tr_mat = prep.transform(X_train)
        X_te_mat = prep.transform(X_test)
        if "logistic" in algorithm.lower():
            est = _FallbackLogisticClassifier(random_state=random_state).fit(X_tr_mat, y_train)
            model_name = "Logistic Regression"
        else:
            est = _FallbackRandomForestClassifier(n_estimators=80, random_state=random_state).fit(X_tr_mat, y_train)
            model_name = "Random Forest Classifier"

        y_pred = est.predict(X_te_mat)
        y_prob = est.predict_proba(X_te_mat)
        feature_importance = extract_pipeline_feature_importance(
            {"preprocessor": prep, "estimator": est}, feature_cols, X_train, y_train
        )
        trained_artifact = {"type": "fallback", "preprocessor": prep, "estimator": est}

    eval_results = evaluate_classification_model(y_test, y_pred, label_classes=label_classes)

    return {
        "model_name": model_name,
        "task_type": "classification",
        "target_column": split_data["target_col"],
        "feature_columns": feature_cols,
        "feature_metadata": split_data["feature_metadata"],
        "label_classes": label_classes,
        "label_encoder": split_data.get("label_encoder"),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": eval_results["metrics"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "actual_vs_predicted": eval_results["actual_vs_predicted"],
        "feature_importance": feature_importance,
        "primary_metric_name": "Accuracy",
        "primary_metric_value": float(eval_results["metrics"]["Accuracy"]),
        "artifact": trained_artifact,
        "y_prob_sample": y_prob[:10].tolist() if y_prob is not None else None,
    }
