"""Tests for src/factor_engine.py"""

import pytest

from src.factor_engine import CompositeResult, FactorEngine

GOOD_OBS = {
    "resting_heart_rate": 55, "hrv_rmssd": 68, "systolic_bp": 112, "diastolic_bp": 72,
    "nocturnal_spo2_pct": 97.2, "spo2_dips": 0, "sleep_hours": 7.8, "deep_sleep_pct": 20,
    "sleep_efficiency": 0.93, "steps": 9500, "active_minutes": 45,
}
BAD_OBS = {
    "resting_heart_rate": 88, "hrv_rmssd": 22, "systolic_bp": 152, "diastolic_bp": 96,
    "nocturnal_spo2_pct": 91.5, "spo2_dips": 8, "sleep_hours": 5.0, "deep_sleep_pct": 8,
    "sleep_efficiency": 0.68, "steps": 1800, "active_minutes": 8,
}
GOOD_LABS = {"total_cholesterol": 175, "hdl": 62, "triglycerides": 95}
BAD_LABS = {"total_cholesterol": 245, "hdl": 34, "triglycerides": 280}
GOOD_PROFILE = {"age": 35, "sex": "female", "height_cm": 165, "weight_kg": 60,
                "smoker": False, "diabetic": False, "family_history": False}
BAD_PROFILE = {"age": 60, "sex": "male", "height_cm": 172, "weight_kg": 100,
               "smoker": True, "diabetic": True, "family_history": True}


class TestWearableFactors:
    def test_scores_bounded(self):
        factors = FactorEngine.evaluate_wearables(GOOD_OBS)
        for f in factors:
            assert 0.0 <= f.score <= 1.0

    def test_good_less_risky_than_bad(self):
        good = FactorEngine.composite(FactorEngine.evaluate_wearables(GOOD_OBS))
        bad = FactorEngine.composite(FactorEngine.evaluate_wearables(BAD_OBS))
        assert good.score < bad.score
        assert bad.score > 0.5

    def test_missing_keys_tolerated(self):
        partial = {"resting_heart_rate": 70, "systolic_bp": 130}
        result = FactorEngine.composite(FactorEngine.evaluate_wearables(partial))
        assert result.n_factors == 2
        assert result.coverage < 1.0
        assert 0.0 <= result.score <= 1.0

    def test_u_shaped_sleep(self):
        optimal = FactorEngine.composite(FactorEngine.evaluate_wearables({**GOOD_OBS, "sleep_hours": 8.0})).score
        short = FactorEngine.composite(FactorEngine.evaluate_wearables({**GOOD_OBS, "sleep_hours": 4.5})).score
        long_ = FactorEngine.composite(FactorEngine.evaluate_wearables({**GOOD_OBS, "sleep_hours": 10.5})).score
        assert short > optimal and long_ > optimal


class TestLabFactors:
    def test_no_labs_yields_unavailable(self):
        factors = FactorEngine.evaluate_labs(None)
        assert all(f.value is None for f in factors)

    def test_good_vs_bad_labs(self):
        good = FactorEngine.composite(FactorEngine.evaluate_labs(GOOD_LABS))
        bad = FactorEngine.composite(FactorEngine.evaluate_labs(BAD_LABS))
        assert good.score < 0.3
        assert bad.score > good.score


class TestDemographics:
    def test_smoker_worse_than_non_smoker(self):
        clean = FactorEngine.composite(FactorEngine.evaluate_demographics(GOOD_PROFILE))
        smoker = FactorEngine.composite(FactorEngine.evaluate_demographics({**GOOD_PROFILE, "smoker": True}))
        assert smoker.score > clean.score

    def test_bmi_computed_from_height_weight(self):
        factors = {f.key: f for f in FactorEngine.evaluate_demographics(GOOD_PROFILE)}
        assert factors["bmi"].value == pytest.approx(60 / 1.65 ** 2, abs=0.05)


class TestComposite:
    def test_evaluate_all_blends_modalities(self):
        good = FactorEngine.evaluate_all(GOOD_OBS, GOOD_LABS, GOOD_PROFILE)
        bad = FactorEngine.evaluate_all(BAD_OBS, BAD_LABS, BAD_PROFILE)
        assert good.score < bad.score
        assert good.coverage == 1.0 and good.n_factors == 20
        modalities = {f.modality for f in good.breakdown}
        assert modalities == {"wearable", "lab", "demographic"}

    def test_breakdown_sorted_by_contribution(self):
        result = FactorEngine.evaluate_all(BAD_OBS, BAD_LABS, BAD_PROFILE)
        contribs = [f.weight * f.score for f in result.breakdown if f.value is not None]
        assert contribs == sorted(contribs, reverse=True)

    def test_top_n(self):
        result: CompositeResult = FactorEngine.evaluate_all(BAD_OBS, BAD_LABS, BAD_PROFILE)
        assert len(result.top(3)) == 3
