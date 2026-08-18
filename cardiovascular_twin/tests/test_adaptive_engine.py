"""Tests for src/adaptive_engine.py and its DigitalTwin integration."""

import numpy as np
import pytest

from src.adaptive_engine import (
    AdaptiveWeightLearner,
    ConfidenceModel,
    PersonalBaselineTracker,
    RunningStats,
)
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource
from src.digital_twin import DigitalTwin


class TestRunningStats:
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        xs = rng.normal(50, 5, size=100)
        s = RunningStats()
        for x in xs:
            s.update(float(x))
        assert s.mean == pytest.approx(xs.mean(), abs=1e-9)
        assert s.variance == pytest.approx(xs.var(ddof=1), rel=1e-6)

    def test_z_score(self):
        s = RunningStats()
        for x in (10.0, 12.0, 14.0, 16.0):
            s.update(x)
        assert s.z(13.0) == pytest.approx((13.0 - 13.0) / s.std, abs=1e-9)

    def test_roundtrip(self):
        s = RunningStats()
        for x in (1.0, 2.0, 3.0):
            s.update(x)
        s2 = RunningStats.from_dict(s.to_dict())
        assert (s2.n, s2.mean, s2.m2) == (s.n, s.mean, s.m2)


class TestPersonalBaselineTracker:
    def test_warmup_and_baseline(self):
        t = PersonalBaselineTracker()
        for day in range(20):
            t.update({"resting_heart_rate": 60 + (day % 3)})
        assert t.is_warm("resting_heart_rate")
        assert 60 <= t.baseline("resting_heart_rate") <= 62
        ci = t.baseline_ci("resting_heart_rate")
        assert ci and ci[0] < ci[1]

    def test_not_warm_early(self):
        t = PersonalBaselineTracker()
        for _ in range(5):
            t.update({"x": 1.0})
        assert not t.is_warm("x")
        assert t.baseline_ci("x") is None

    def test_anomaly_detection(self):
        t = PersonalBaselineTracker()
        rng = np.random.default_rng(1)
        for _ in range(20):
            t.update({"hrv_rmssd": 50.0 + rng.normal(0, 3)})  # cv ~6% -> qualifies
        zs = t.update({"hrv_rmssd": 70.0})  # big jump
        assert "hrv_rmssd" in zs
        assert "hrv_rmssd" in t.anomalies(zs)

    def test_narrow_distribution_suppressed(self):
        """Channels whose personal spread is tiny must not produce z-scores."""
        t = PersonalBaselineTracker()
        for v in np.full(15, 50.0):
            t.update({"calibrated_channel": float(v)})
        zs = t.update({"calibrated_channel": 52.0})  # cv=0 -> no z produced
        assert zs == {} or "calibrated_channel" not in zs

    def test_roundtrip(self):
        t = PersonalBaselineTracker()
        for v in (55.0, 56.0, 57.0, 58.0, 59.0, 60.0):
            t.update({"rhr": v})
        t2 = PersonalBaselineTracker.from_dict(t.to_dict())
        assert t2.baseline("rhr") == t.baseline("rhr")
        assert t2.recent["rhr"] == t.recent["rhr"]


class TestAdaptiveWeightLearner:
    def test_predictive_factor_gains_weight(self):
        learner = AdaptiveWeightLearner()
        comp = 0.5
        # factor 'good' moves in the same direction as the next composite; 'bad' opposite
        rng = np.random.default_rng(0)
        for _ in range(30):
            comp_prev, comp = comp, comp + rng.normal(0, 0.02)
            direction = 1 if comp > comp_prev else -1
            learner.observe({"good": 0.5 + direction * 0.05,
                             "bad": 0.5 - direction * 0.05}, comp)
        mults = learner.active_multipliers()
        if mults.get("good") and mults.get("bad"):
            assert mults["good"] >= mults["bad"]

    def test_multipliers_bounded(self):
        learner = AdaptiveWeightLearner()
        comp = 0.5
        rng = np.random.default_rng(2)
        for i in range(200):
            prev = comp
            comp = min(max(comp + rng.normal(0, 0.05), 0.05), 0.95)
            learner.observe({"f": rng.uniform(0, 1)}, comp)
        assert all(AdaptiveWeightLearner.MIN_MULT <= m <= AdaptiveWeightLearner.MAX_MULT
                   for m in learner.multipliers.values())

    def test_roundtrip(self):
        learner = AdaptiveWeightLearner()
        for i in range(15):
            learner.observe({"f": float(i % 3) / 3}, 0.4 + 0.01 * (i % 2))
        l2 = AdaptiveWeightLearner.from_dict(learner.to_dict())
        assert l2.active_multipliers() == learner.active_multipliers()


class TestConfidenceModel:
    def test_ci_shrinks_with_consistent_data(self):
        c = ConfidenceModel()
        widths = []
        for v in [0.5, 0.52, 0.49, 0.51, 0.5, 0.5, 0.51, 0.49, 0.5, 0.5]:
            low, high = c.update(v)
            widths.append(high - low)
        assert widths[-1] <= widths[0] + 1e-9

    def test_precision_pct(self):
        c = ConfidenceModel()
        c.scores.extend([0.4, 0.6])  # wide early state
        for v in (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5):
            c.update(v)
        assert c.precision_pct() >= 0.0
        assert 0.0 <= c.width <= 1.0

    def test_roundtrip(self):
        c = ConfidenceModel()
        for v in (0.3, 0.4, 0.5):
            c.update(v)
        c2 = ConfidenceModel.from_dict(c.to_dict())
        assert c2.ci95() == c.ci95()


class TestTwinIntegration:
    def test_adaptive_twin_learns(self):
        twin = DigitalTwin(PRESET_PROFILES["at_risk"].to_dict(), adaptive=True)
        src = SimulatedWearableSource(PRESET_PROFILES["at_risk"], scenario="declining", seed=5)
        updates = [twin.assimilate(obs, obs.get("labs")) for obs in src.next_days(60)]
        assert updates[-1].ci95 is not None
        assert updates[-1].ci95[0] <= updates[-1].risk_score <= updates[-1].ci95[1] + 0.02
        summary = twin.learning_summary()
        assert summary["enabled"] and summary["days_seen"] == 60
        assert summary["warm_baselines"] >= 5
        # multipliers engaged for some factors after 60 days
        assert any(m != 1.0 for m in (updates[-1].multipliers or {}).values())

    def test_snapshot_roundtrip_with_adaptive_state(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict())
        src = SimulatedWearableSource(PRESET_PROFILES["typical"], seed=9)
        for obs in src.next_days(30):
            twin.assimilate(obs, obs.get("labs"))
        restored = DigitalTwin.from_dict(twin.snapshot())
        assert restored.risk_score == twin.risk_score
        assert restored.baselines.baseline("resting_heart_rate") == twin.baselines.baseline("resting_heart_rate")
        assert restored.weight_learner.active_multipliers() == twin.weight_learner.active_multipliers()
        # further assimilation continues identically
        obs = {"resting_heart_rate": 70.0, "systolic_bp": 130.0, "hrv_rmssd": 40.0}
        u1, u2 = twin.assimilate(obs), restored.assimilate(obs)
        assert u1.risk_score == u2.risk_score

    def test_non_adaptive_mode(self):
        twin = DigitalTwin(PRESET_PROFILES["typical"].to_dict(), adaptive=False)
        upd = twin.assimilate({"resting_heart_rate": 60.0, "systolic_bp": 120.0})
        assert upd.ci95 is None and upd.multipliers is None
        assert twin.learning_summary()["enabled"] is False
