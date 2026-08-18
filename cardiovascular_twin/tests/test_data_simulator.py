"""Tests for src/data_simulator.py"""

import pandas as pd
import pytest

from src.data_simulator import (
    PRESET_PROFILES,
    SimulatedWearableSource,
    UserProfile,
    simulate_history,
)


class TestSimulator:
    def test_seed_determinism(self):
        a = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=42).next_days(30)
        b = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=42).next_days(30)
        assert a == b

    def test_different_seeds_differ(self):
        a = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=1).next_days(10)
        b = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=2).next_days(10)
        assert a != b

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError):
            SimulatedWearableSource(scenario="chaos")

    def test_observation_shape_and_bounds(self):
        src = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=5, lab_interval_days=10)
        for obs in src.next_days(30):
            assert "day_index" in obs and "date" in obs
            assert 35 <= obs["resting_heart_rate"] <= 130
            assert 8 <= obs["hrv_rmssd"] <= 180
            assert 80 <= obs["systolic_bp"] <= 220
            assert 0 <= obs["sleep_hours"] <= 11
            assert obs["steps"] >= 0
            assert 88 <= obs["nocturnal_spo2_pct"] <= 99.5
            assert obs["spo2_dips"] >= 0

    def test_labs_appear_on_interval(self):
        src = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=1, lab_interval_days=10)
        days = src.next_days(31)
        lab_days = [d["day_index"] for d in days if "labs" in d]
        assert 0 in lab_days and 10 in lab_days and 30 in lab_days
        assert all(set(d["labs"]) == {"total_cholesterol", "hdl", "triglycerides"} for d in days if "labs" in d)

    def test_declining_scenario_drifts_physiology(self):
        healthy = SimulatedWearableSource(PRESET_PROFILES["typical"], scenario="declining", seed=7)
        df = simulate_history(90, PRESET_PROFILES["typical"], scenario="declining", seed=7)
        first, last = df.iloc[:7], df.iloc[-7:]
        assert last["resting_heart_rate"].mean() > first["resting_heart_rate"].mean()
        assert last["hrv_rmssd"].mean() < first["hrv_rmssd"].mean()


class TestHistoryFrame:
    def test_frame_shape_and_columns(self):
        df = simulate_history(50, PRESET_PROFILES["healthy"], seed=3)
        assert len(df) == 50
        for col in ("resting_heart_rate", "hrv_rmssd", "systolic_bp", "steps", "sleep_hours"):
            assert col in df.columns
        assert "labs" not in df.columns

    def test_labs_forward_filled(self):
        df = simulate_history(25, PRESET_PROFILES["typical"], seed=3, lab_interval_days=10)
        assert df["total_cholesterol"].notna().all()


class TestProfile:
    def test_bmi(self):
        p = UserProfile(height_cm=170, weight_kg=85)
        assert p.bmi == pytest.approx(29.4, abs=0.1)

    def test_presets_cover_range(self):
        assert PRESET_PROFILES["healthy"].baseline_systolic < PRESET_PROFILES["at_risk"].baseline_systolic
        assert PRESET_PROFILES["healthy"].baseline_hrv_rmssd > PRESET_PROFILES["at_risk"].baseline_hrv_rmssd
