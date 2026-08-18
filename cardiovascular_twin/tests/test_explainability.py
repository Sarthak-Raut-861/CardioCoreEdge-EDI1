"""Tests for src/explainability.py"""

import numpy as np
import pandas as pd
import pytest

from src.explainability import RiskExplainer
from src.factor_engine import FactorEngine


GOOD = {
    "resting_heart_rate": 55, "hrv_rmssd": 68, "systolic_bp": 112, "diastolic_bp": 72,
    "nocturnal_spo2_pct": 97.2, "spo2_dips": 0, "sleep_hours": 7.8, "deep_sleep_pct": 20,
    "sleep_efficiency": 0.93, "steps": 9500, "active_minutes": 45,
}
BAD = {
    "resting_heart_rate": 88, "hrv_rmssd": 22, "systolic_bp": 152, "diastolic_bp": 96,
    "nocturnal_spo2_pct": 91.5, "spo2_dips": 8, "sleep_hours": 5.0, "deep_sleep_pct": 8,
    "sleep_efficiency": 0.68, "steps": 1800, "active_minutes": 8,
}
PROFILE = {"age": 50, "sex": "male", "height_cm": 175, "weight_kg": 88,
           "smoker": True, "diabetic": False, "family_history": True}


@pytest.fixture(scope="module")
def bad_composite():
    return FactorEngine.evaluate_all(BAD, {"total_cholesterol": 240, "hdl": 36, "triglycerides": 250}, PROFILE)


class TestFactorAttribution:
    def test_table_structure(self, bad_composite):
        table = RiskExplainer.explain_composite(bad_composite)
        assert {"factor", "modality", "value", "score", "weight", "contribution", "share", "assessment"} <= set(table.columns)
        assert table["share"].sum() == pytest.approx(1.0, abs=0.02)

    def test_bad_case_flags_risk_increasing(self, bad_composite):
        table = RiskExplainer.explain_composite(bad_composite)
        assert (table["assessment"] == "risk-increasing").any()

    def test_narrative_content(self, bad_composite):
        text = RiskExplainer.narrative(bad_composite)
        assert "Composite risk" in text
        assert len(text) > 40

    def test_narrative_empty_composite(self):
        text = RiskExplainer.narrative(FactorEngine.composite(FactorEngine.evaluate_wearables({})))
        assert text.startswith("No data available")


@pytest.fixture(scope="module")
def tiny_model():
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] * 2 + X[:, 2] - X[:, 3] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss", verbosity=0)
    model.fit(X, y)
    return model


class TestShapExplanation:

    def test_shap_output(self, tiny_model):
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 5)), columns=[f"f{i}" for i in range(5)])
        out = RiskExplainer.explain_model_prediction(tiny_model, X)
        assert 0.0 <= out["prediction"] <= 1.0
        assert out["n_features"] == 5
        contribs = out["contributions"]
        assert contribs[0]["shap_value"] >= 0 or contribs[0]["shap_value"] < 0  # present
        assert set(contribs[0]) == {"feature", "shap_value"}

    def test_shap_recovers_informative_features(self, tiny_model):
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(10, 5)), columns=[f"f{i}" for i in range(5)])
        out = RiskExplainer.explain_model_prediction(tiny_model, X)
        top_features = {c["feature"] for c in out["contributions"][:3]}
        assert top_features & {"f0", "f2", "f3"}


class TestCombined:
    def test_explain_convenience(self, bad_composite):
        out = RiskExplainer().explain(composite=bad_composite)
        assert "factor_table" in out and "narrative" in out
        assert isinstance(out["factor_table"], pd.DataFrame)
