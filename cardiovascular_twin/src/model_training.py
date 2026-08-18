"""
model_training.py
=================
Rigorous training pipeline for the CardioCore risk model.

Designed around "how do I get the most *honest* accuracy?" rather than a
single lucky split:

1. **Data hygiene** – known anomalies in the UCI-combined dataset
   (RestingBP == 0, Cholesterol == 0) are treated as missing and imputed
   with medians *fitted on the training fold only* (no leakage).
2. **Stratified split** (80/20) – preserves the 55/45 class balance.
3. **Model bake-off** – logistic regression (interpretable baseline),
   random forest, and XGBoost compared with 5-fold stratified CV on ROC-AUC.
4. **Hyperparameter tuning** – grid search over the XGBoost grid, scored by
   CV AUC.
5. **Class imbalance** – ``scale_pos_weight`` / ``class_weight='balanced'``.
6. **Decision threshold** – chosen by Youden's J on *out-of-fold training
   predictions* (never on the test set).
7. **Probability calibration** – isotonic calibration; Brier score reported
   before/after on the untouched holdout.
8. **Honest reporting** – holdout ROC-AUC, PR-AUC, accuracy, sensitivity,
   specificity, F1 + permutation importances.

Also provides ``twin_feature_vector`` to feed a live ``DigitalTwin`` state
into the trained model, and save/load for artifacts.

Dataset: "Heart Failure Prediction Dataset" (fedesoriano) – the combined
UCI Heart Disease data (Cleveland + Hungarian + Switzerland + Statlog,
918 records after de-duplication). Educational use; not clinical grade.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "DATASET_URL",
    "ensure_dataset",
    "prepare_features",
    "train_risk_model",
    "save_artifact",
    "load_artifact",
    "twin_feature_vector",
]

# mirror of the fedesoriano "Heart Failure Prediction Dataset" (918 rows)
DATASET_URL = "https://raw.githubusercontent.com/clinicalml/TabLLM/main/datasets/heart/heart.csv"
TARGET_COL = "HeartDisease"
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
# zero is physiologically impossible for these -> treated as missing
ZERO_INVALID = ["RestingBP", "Cholesterol"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def ensure_dataset(path: Path | str = "data/raw/heart_uci_combined.csv",
                   url: str = DATASET_URL) -> Path:
    """Return the dataset path, downloading it if absent."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading dataset from {url}")
    urllib.request.urlretrieve(url, path)
    return path


def load_dataset(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(NUMERIC + CATEGORICAL + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset at {path} missing columns: {sorted(missing)}")
    return df


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Copy + replace impossible zeros with NaN (kept for imputer)."""
    out = df.copy()
    for col in ZERO_INVALID:
        if col in out.columns:
            out.loc[out[col] <= 0, col] = np.nan
    return out


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """X (raw columns, cleaned) and y arrays; encoding lives in the pipeline."""
    X = clean_raw(df[NUMERIC + CATEGORICAL])
    y = df[TARGET_COL].astype(int).values
    return X, y


def build_preprocessor() -> Pipeline:
    """Impute + encode; fitted within model pipelines on train data only."""
    from sklearn.compose import ColumnTransformer

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_candidates(random_state: int) -> Dict[str, Any]:
    xgb = None
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0, random_state=random_state,
        )
    except ImportError:  # pragma: no cover
        pass
    candidates: Dict[str, Any] = {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced", random_state=random_state)),
        ]),
    }
    if xgb is not None:
        candidates["xgboost"] = Pipeline([("prep", build_preprocessor()), ("clf", xgb)])
    return candidates


XGB_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.03, 0.08],
    "clf__subsample": [0.8, 1.0],
}


def _youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1."""
    thresholds = np.unique(np.round(proba, 4))
    best_t, best_j = 0.5, -np.inf
    for t in thresholds:
        pred = proba >= t
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return round(best_t, 4)


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "accuracy": round(float((tp + tn) / len(y_true)), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "threshold": round(float(threshold), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_risk_model(
    df: pd.DataFrame,
    tune: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Full bake-off + tuning + calibration. Returns artifact dict."""
    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ---- 1) bake-off ---------------------------------------------------- #
    candidates = _make_candidates(random_state)
    cv_scores: Dict[str, float] = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = round(float(scores.mean()), 4)
        print(f"[cv] {name:<20} ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ---- 2) tune XGBoost (or best available) ----------------------------- #
    base_name = max(cv_scores, key=cv_scores.get)
    best_name, best_model = base_name, candidates[base_name]
    grid_result: Dict[str, Any] = {"grid_used": False}
    if tune and "xgboost" in candidates:
        print("[tune] grid-searching XGBoost ...")
        gs = GridSearchCV(candidates["xgboost"], XGB_GRID, cv=cv,
                          scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        grid_result = {
            "grid_used": True,
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "best_cv_auc": round(float(gs.best_score_), 4),
        }
        print(f"[tune] best params {grid_result['best_params']} -> CV AUC {grid_result['best_cv_auc']:.4f}")
        best_name, best_model = "xgboost_tuned", gs.best_estimator_

    # ---- 3) decision threshold from out-of-fold train predictions -------- #
    oof = cross_val_predict(best_model, X_tr, y_tr, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = _youden_threshold(y_tr, oof)

    # ---- 4) holdout evaluation (uncalibrated) ---------------------------- #
    best_model.fit(X_tr, y_tr)
    proba_te = best_model.predict_proba(X_te)[:, 1]
    holdout = _metrics(y_te, proba_te, threshold)

    # ---- 5) probability calibration -------------------------------------- #
    calibrated = CalibratedClassifierCV(estimator=best_model, method="isotonic", cv=cv)
    calibrated.fit(X_tr, y_tr)
    proba_te_cal = calibrated.predict_proba(X_te)[:, 1]
    holdout_cal = _metrics(y_te, proba_te_cal, threshold)
    print(f"[holdout] {best_name}: AUC {holdout['roc_auc']:.4f} | "
          f"sens {holdout['sensitivity']:.3f} spec {holdout['specificity']:.3f} | "
          f"brier {holdout['brier']:.4f} -> calibrated brier {holdout_cal['brier']:.4f}")

    # ---- 6) refit calibrated model on ALL data --------------------------- #
    final_model = CalibratedClassifierCV(estimator=best_model, method="isotonic", cv=cv)
    final_model.fit(X, y)

    # ---- 7) permutation importance (holdout, calibrated) ----------------- #
    perm = permutation_importance(calibrated, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=random_state)
    importances = sorted(
        ({"feature": c, "importance": round(float(v), 4)}
         for c, v in zip(X_te.columns, perm.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )

    return {
        "model": final_model,               # calibrated, refit on all data
        "raw_model": best_model,            # uncalibrated (for SHAP/twin attach)
        "model_name": best_name,
        "threshold": threshold,
        "cv_scores": cv_scores,
        "tuning": grid_result,
        "holdout": holdout,
        "holdout_calibrated": holdout_cal,
        "importances": importances,
        "features": list(X.columns),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "random_state": random_state,
    }


# --------------------------------------------------------------------------- #
# Twin integration
# --------------------------------------------------------------------------- #
def twin_feature_vector(twin: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Build a model-input row from a DigitalTwin's current state.

    Mappings: Age<-profile.age, Sex<-profile.sex, RestingBP<-twin systolic
    state, Cholesterol<-latest labs, FastingBS<-diabetic flag. Exercise-test
    features (MaxHR, Oldpeak, angina, ECG, ST-slope, chest-pain type) are not
    measured by the wearable, so they fall back to training-set medians/modes
    stored in the artifact – this *dampens* individualization and is
    documented behaviour, not silent magic.
    """
    profile = twin.profile
    state = twin.state
    labs = twin.labs or {}
    row: Dict[str, Any] = {
        "Age": float(profile.get("age", 50)),
        "Sex": "M" if str(profile.get("sex", "male")).lower().startswith("m") else "F",
        "RestingBP": state.get("systolic_bp") or df["RestingBP"].median(),
        "Cholesterol": labs.get("total_cholesterol") or df["Cholesterol"].median(),
        "FastingBS": 1 if profile.get("diabetic") else 0,
    }
    defaults: Dict[str, Any] = {}
    for col in NUMERIC + CATEGORICAL:
        if col not in row:
            defaults[col] = df[col].mode(dropna=True).iloc[0] if col in CATEGORICAL else df[col].median()
    row.update(defaults)
    return pd.DataFrame([{c: row[c] for c in NUMERIC + CATEGORICAL}])


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def save_artifact(artifact: Dict[str, Any], out_dir: Path | str,
                  training_df: Optional[pd.DataFrame] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "risk_model.joblib"
    bundle = dict(artifact)
    if training_df is not None:
        # keep a small sample for twin defaults (medians/modes)
        bundle["data_sample"] = training_df.sample(min(200, len(training_df)), random_state=0)
    joblib.dump(bundle, model_path)

    meta = {k: v for k, v in bundle.items()
            if k in ("model_name", "threshold", "cv_scores", "tuning", "holdout",
                     "holdout_calibrated", "importances", "features", "n_train", "n_test")}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2, default=str))
    return model_path


def load_artifact(path: Path | str) -> Dict[str, Any]:
    return joblib.load(path)
