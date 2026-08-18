"""Tests for src/model_training.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.digital_twin import DigitalTwin
from src.model_training import (
    CATEGORICAL,
    NUMERIC,
    TARGET_COL,
    _metrics,
    _youden_threshold,
    prepare_features,
    save_artifact,
    train_risk_model,
    twin_feature_vector,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_uci_combined.csv"


def synthetic_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset with learnable signal, same schema as the real one."""
    rng = np.random.default_rng(seed)
    risk = (rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({
        "Age": 40 + risk * 15 + rng.normal(0, 6, n),
        "Sex": rng.choice(["M", "F"], n, p=[0.55, 0.45]),
        "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], n, p=[0.4, 0.2, 0.3, 0.1]),
        "RestingBP": 115 + risk * 20 + rng.normal(0, 8, n),
        "Cholesterol": 170 + risk * 45 + rng.normal(0, 25, n),
        "FastingBS": rng.choice([0, 1], n, p=[0.75, 0.25]) | risk * rng.integers(0, 2, n),
        "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
        "MaxHR": 165 - risk * 20 + rng.normal(0, 10, n),
        "ExerciseAngina": np.where(risk == 1, rng.choice(["N", "Y"], n, p=[0.35, 0.65]), "N"),
        "Oldpeak": risk * rng.uniform(0.5, 3.5, n) + rng.uniform(0, 0.8, n),
        "ST_Slope": rng.choice(["Up", "Flat", "Down"], n, p=[0.5, 0.3, 0.2]),
        TARGET_COL: risk,
    })


class TestPreparation:
    def test_prepare_features_shape(self):
        df = synthetic_df()
        X, y = prepare_features(df)
        assert list(X.columns) == NUMERIC + CATEGORICAL
        assert len(X) == len(y) == len(df)
        assert set(np.unique(y)) == {0, 1}

    def test_impossible_zeros_become_nan(self):
        df = synthetic_df(50, seed=1)
        df.loc[0, "Cholesterol"] = 0
        df.loc[1, "RestingBP"] = 0
        X, _ = prepare_features(df)
        assert pd.isna(X.loc[0, "Cholesterol"]) and pd.isna(X.loc[1, "RestingBP"])


class TestHelpers:
    def test_youden_threshold_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        t = _youden_threshold(y, p)
        assert ((p >= t).astype(int) == y).all()

    def test_metrics_ranges(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        p = np.array([0.1, 0.35, 0.65, 0.8, 0.9, 0.4])
        m = _metrics(y, p, 0.5)
        for key in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "brier"):
            assert 0.0 <= m[key] <= 1.0
        assert m["confusion"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}


class TestTraining:
    def test_train_on_synthetic(self, tmp_path):
        df = synthetic_df(400, seed=2)
        artifact = train_risk_model(df, tune=False, random_state=7)
        assert artifact["holdout"]["roc_auc"] > 0.75  # strong signal in synthetic data
        assert 0.0 < artifact["threshold"] < 1.0
        assert set(artifact["cv_scores"]) >= {"logistic_regression", "random_forest"}

        out = save_artifact(artifact, tmp_path, training_df=df)
        assert out.exists() and (tmp_path / "metrics.json").exists()

        from src.model_training import load_artifact
        loaded = load_artifact(out)
        row = df.iloc[[5]][NUMERIC + CATEGORICAL]
        p1 = artifact["model"].predict_proba(row)
        p2 = loaded["model"].predict_proba(row)
        assert np.allclose(p1, p2)


class TestTwinIntegration:
    def test_twin_feature_vector(self):
        twin = DigitalTwin({"age": 58, "sex": "male", "height_cm": 175, "weight_kg": 90,
                            "smoker": True, "diabetic": True, "family_history": False})
        twin.assimilate({"systolic_bp": 145.0}, labs={"total_cholesterol": 240.0, "hdl": 38, "triglycerides": 200})
        df = synthetic_df(100, seed=3)
        vec = twin_feature_vector(twin, df)
        assert list(vec.columns) == NUMERIC + CATEGORICAL
        assert vec.iloc[0]["Age"] == 58
        assert vec.iloc[0]["Sex"] == "M"
        assert vec.iloc[0]["RestingBP"] == 145.0
        assert vec.iloc[0]["Cholesterol"] == 240.0
        assert vec.iloc[0]["FastingBS"] == 1

    def test_female_profile_maps_to_f(self):
        twin = DigitalTwin({"age": 45, "sex": "female"})
        vec = twin_feature_vector(twin, synthetic_df(20, seed=4))
        assert vec.iloc[0]["Sex"] == "F"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="real dataset not downloaded")
class TestRealDataset:
    def test_quick_train_real_data(self):
        from src.model_training import load_dataset

        df = load_dataset(DATA_PATH).sample(400, random_state=0)
        artifact = train_risk_model(df, tune=False, random_state=0)
        assert artifact["holdout"]["roc_auc"] > 0.80
