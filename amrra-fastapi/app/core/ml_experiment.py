# app/core/ml_experiment.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from scipy.stats import permutation_test

RANDOM_SEED = 12345

def _bootstrap_ci(stats: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1 - alpha/2)
    return float(lo), float(hi)

def run_ml_experiment() -> Dict[str, Any]:
    ds = load_breast_cancer()
    X = ds.data
    y = ds.target
    feature_names = list(ds.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_SEED))
    ])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    )

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    p_lr = logreg.predict_proba(X_test)[:,1]
    p_rf = rf.predict_proba(X_test)[:,1]

    yhat_lr = (p_lr >= 0.5).astype(int)
    yhat_rf = (p_rf >= 0.5).astype(int)

    auc_lr = roc_auc_score(y_test, p_lr)
    auc_rf = roc_auc_score(y_test, p_rf)
    acc_lr = accuracy_score(y_test, yhat_lr)
    acc_rf = accuracy_score(y_test, yhat_rf)
    f1_lr = f1_score(y_test, yhat_lr)
    f1_rf = f1_score(y_test, yhat_rf)

    cm_lr = confusion_matrix(y_test, yhat_lr).tolist()
    cm_rf = confusion_matrix(y_test, yhat_rf).tolist()

    rng = np.random.default_rng(RANDOM_SEED)
    B = 500
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(y_test), size=len(y_test))
        diffs.append(roc_auc_score(y_test[idx], p_rf[idx]) - roc_auc_score(y_test[idx], p_lr[idx]))
    ci_lo, ci_hi = _bootstrap_ci(np.array(diffs), alpha=0.05)

    def stat(x, y):
        return roc_auc_score(y_test, x) - roc_auc_score(y_test, y)
    perm = permutation_test((p_rf, p_lr), stat, permutation_type="pairings",
                            alternative="greater", n_resamples=300, random_state=RANDOM_SEED)
    p_perm = float(perm.pvalue)

    return {
        "dataset": {
            "name": "sklearn_breast_cancer",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feature_names[:8] + (["…"] if len(feature_names) > 8 else []),
            "target_name": "malignant_vs_benign"
        },
        "metrics": {
            "logreg": {"auc": round(float(auc_lr),4), "accuracy": round(float(acc_lr),4), "f1": round(float(f1_lr),4), "confusion_matrix": cm_lr},
            "rf":     {"auc": round(float(auc_rf),4), "accuracy": round(float(acc_rf),4), "f1": round(float(f1_rf),4), "confusion_matrix": cm_rf}
        },
        "comparison": {
            "delta_auc_rf_minus_lr": round(float(auc_rf - auc_lr), 4),
            "delta_auc_bootstrap_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "permutation_test_p_greater": round(p_perm, 5)
        },
        "design": {
            "task": "binary_classification",
            "primary_metric": "roc_auc",
            "test_size": 0.25,
            "seed": RANDOM_SEED,
            "bootstrap_B": B,
            "permutation_n_resamples": 300
        }
    }