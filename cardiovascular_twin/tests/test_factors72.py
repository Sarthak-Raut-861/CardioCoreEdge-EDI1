"""Tests for the 72-factor framework (src/factors72.py)."""

import pytest

from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource
from src.factors72 import FACTORS, FACTOR_IDS, FactorSpec, build_factor_vector, normalize


class TestRegistry:
    def test_exactly_72_factors(self):
        assert len(FACTORS) == 72
        assert FACTOR_IDS == [f"F{i}" for i in range(1, 73)]

    def test_specs_well_formed(self):
        for fid, spec in FACTORS.items():
            assert isinstance(spec, FactorSpec)
            assert spec.raw_min < spec.raw_max
            assert spec.name and spec.source and spec.category

    def test_categories_cover_doc_groups(self):
        cats = {s.category for s in FACTORS.values()}
        assert {"NIR", "PPG", "BP", "HRV", "SpO2", "ECG", "Activity",
                "Demographic", "History", "Diet", "Smoking", "Derived"} <= cats

    def test_documented_factor_ids_present(self):
        # spot-check ids named in the documentation
        for fid, name_frag in [("F1", "NIR 1040"), ("F18", "Pulse Pressure"), ("F30", "AF Probability"),
                               ("F39", "Daily Activity"), ("F48", "Age"), ("F56", "Cholesterol History"),
                               ("F58", "Clot"), ("F66", "Smoking"), ("F72", "Endothelial")]:
            assert name_frag.lower() in FACTORS[fid].name.lower()

    def test_documented_top_weights(self):
        assert FACTORS["F1"].weights["TC"][0] == 18
        assert FACTORS["F7"].weights["TC"][0] == 16
        assert FACTORS["F56"].weights["TC"][0] == 14
        assert FACTORS["F30"].weights["DD"][0] == 18
        assert FACTORS["F58"].weights["DD"][0] == 15
        assert FACTORS["F2"].weights["TG"][0] == 15
        assert FACTORS["F50"].weights["CRP"][0] == 12

    def test_directions(self):
        assert FACTORS["F21"].direction == "protective"    # SDNN protective
        assert FACTORS["F30"].direction == "risk-increasing"  # AF prob


class TestNormalization:
    def test_clip_and_scale(self):
        assert normalize(0.5, 0.0, 1.0) == 0.5
        assert normalize(-5, 0.0, 1.0) == 0.0
        assert normalize(9, 0.0, 1.0) == 1.0
        assert normalize(15, 10, 20) == 0.5


class TestFactorVector:
    def test_all_72_present_and_bounded(self):
        obs = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=1).next_day()
        vec = build_factor_vector(obs, PRESET_PROFILES["typical"].to_dict(), 0)
        assert len(vec) == 72
        assert all(0.0 <= v <= 1.0 for v in vec.values())

    def test_deterministic(self):
        p = PRESET_PROFILES["typical"].to_dict()
        src = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=3)
        obs_list = src.next_days(5)
        v1 = [build_factor_vector(o, p, i) for i, o in enumerate(obs_list)]
        src2 = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=3)
        obs_list2 = src2.next_days(5)
        v2 = [build_factor_vector(o, p, i) for i, o in enumerate(obs_list2)]
        assert v1 == v2

    def test_unhealthy_observation_raises_risk_factors(self):
        good = {"resting_heart_rate": 55, "hrv_rmssd": 70, "systolic_bp": 112, "diastolic_bp": 72,
                "nocturnal_spo2_pct": 97, "spo2_dips": 0, "sleep_hours": 7.8, "steps": 10000, "active_minutes": 50}
        bad = {"resting_heart_rate": 90, "hrv_rmssd": 22, "systolic_bp": 155, "diastolic_bp": 98,
               "nocturnal_spo2_pct": 92, "spo2_dips": 9, "sleep_hours": 5.0, "steps": 1500, "active_minutes": 5}
        p = PRESET_PROFILES["typical"].to_dict()
        vg, vb = build_factor_vector(good, p, 0), build_factor_vector(bad, p, 0)
        assert vb["F16"] > vg["F16"]   # SBP
        assert vb["F21"] > vg["F21"] or True  # SDNN from HRV: lower HRV -> lower normalized SDNN
        assert vg["F21"] > vb["F21"]   # protective factor higher when healthy
        assert vb["F46"] > vg["F46"]   # stress
        assert vg["F39"] > vb["F39"]   # activity

    def test_bp_variability_factor(self):
        p = PRESET_PROFILES["typical"].to_dict()
        obs = {"resting_heart_rate": 65, "systolic_bp": 125, "sleep_hours": 7}
        v_low = build_factor_vector(obs, p, 0, bp_sd=1.0)
        v_high = build_factor_vector(obs, p, 0, bp_sd=9.0)
        assert v_high["F20"] > v_low["F20"]
