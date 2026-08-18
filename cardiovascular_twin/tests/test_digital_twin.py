"""Tests for src/digital_twin.py"""

import pytest

from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource
from src.digital_twin import DigitalTwin


def feed(twin: DigitalTwin, obs: dict, n: int = 1):
    for _ in range(n):
        twin.assimilate(obs)


class TestAssimilation:
    def test_ewma_smooths_noise(self):
        twin = DigitalTwin(PRESET_PROFILES["healthy"].to_dict(), learning_rate=0.15)
        twin.assimilate({"resting_heart_rate": 60.0})
        twin.assimilate({"resting_heart_rate": 100.0})  # single outlier
        assert twin.state["resting_heart_rate"] < 70  # outlier only moves state partially

    def test_first_observation_sets_state(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        twin.assimilate({"systolic_bp": 120.0})
        assert twin.state["systolic_bp"] == 120.0

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError):
            DigitalTwin({"age": 40}, learning_rate=0.0)

    def test_days_seen_counts(self):
        twin = DigitalTwin({"age": 40})
        feed(twin, {"resting_heart_rate": 60}, n=5)
        assert twin.days_seen == 5


class TestRiskScoring:
    def test_declining_scenario_raises_risk(self):
        profile = PRESET_PROFILES["at_risk"].to_dict()
        twin = DigitalTwin(profile)
        source = SimulatedWearableSource(PRESET_PROFILES["at_risk"], scenario="declining", seed=3)
        first = None
        for obs in source.next_days(60):
            upd = twin.assimilate(obs, obs.get("labs"))
            if first is None:
                first = upd.risk_score
        assert twin.risk_score > first
        assert twin.risk_trend() == "declining"

    def test_improving_scenario_lowers_risk(self):
        twin = DigitalTwin(PRESET_PROFILES["at_risk"].to_dict())
        source = SimulatedWearableSource(PRESET_PROFILES["at_risk"], scenario="improving", seed=3)
        first = None
        for obs in source.next_days(60):
            upd = twin.assimilate(obs, obs.get("labs"))
            if first is None:
                first = upd.risk_score
        assert twin.risk_score < first

    def test_category_bands(self):
        twin = DigitalTwin(PRESET_PROFILES["healthy"].to_dict())
        source = SimulatedWearableSource(PRESET_PROFILES["healthy"], seed=1)
        for obs in source.next_days(20):
            twin.assimilate(obs, obs.get("labs"))
        assert twin.risk_category in {"Low", "Moderate", "High", "Very High"}
        assert 0.0 <= twin.risk_score <= 1.0


class TestAlerts:
    def test_hypertensive_crisis_alert(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        upd = twin.assimilate({"systolic_bp": 185.0, "diastolic_bp": 122.0, "date": "2026-01-01"})
        codes = [a["code"] for a in upd.alerts]
        assert "hypertensive_crisis" in codes

    def test_desaturation_alert(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        feed(twin, {"nocturnal_spo2_pct": 89.0}, n=3)
        upd = twin.assimilate({"nocturnal_spo2_pct": 89.0})
        assert any(a["code"] == "desaturation" for a in upd.alerts)

    def test_healthy_days_no_serious_alerts(self):
        twin = DigitalTwin(PRESET_PROFILES["healthy"].to_dict())
        source = SimulatedWearableSource(PRESET_PROFILES["healthy"], seed=1)
        alerts = []
        for obs in source.next_days(15):
            alerts += twin.assimilate(obs, obs.get("labs")).alerts
        # safety-relevant alert levels must stay silent for a healthy profile;
        # info-level personal-baseline anomaly notes are acceptable by design
        assert all(a["level"] == "info" and a["code"] == "anomaly_detected" for a in alerts)


class TestPersistence:
    def test_snapshot_roundtrip(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        source = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=9, lab_interval_days=5)
        for obs in source.next_days(30):
            twin.assimilate(obs, obs.get("labs"))
        restored = DigitalTwin.from_dict(twin.snapshot())
        assert restored.state == twin.state
        assert restored.labs == twin.labs
        assert restored.days_seen == twin.days_seen
        assert restored.risk_history == twin.risk_history
        assert restored.risk_score == twin.risk_score


class TestTwinUpdate:
    def test_to_dict_shape(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        upd = twin.assimilate({"resting_heart_rate": 60, "systolic_bp": 120, "date": "2026-01-01"})
        d = upd.to_dict()
        assert {"day_index", "date", "risk_score", "category", "alerts", "trend", "top_factors"} <= set(d)
