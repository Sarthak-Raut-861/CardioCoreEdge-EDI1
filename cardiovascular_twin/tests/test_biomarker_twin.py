"""Tests for the biomarker digital twin (src/biomarker_twin.py)."""

import numpy as np
import pandas as pd
import pytest

from src.biomarker_twin import BiomarkerClusterer, BiomarkerTwin, generate_cohort
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource


def run_twin(profile_name: str, scenario: str = "stable", seed: int = 42, days: int = 60) -> BiomarkerTwin:
    prof = PRESET_PROFILES[profile_name]
    twin = BiomarkerTwin(prof.to_dict())
    src = SimulatedWearableSource(prof, scenario=scenario, seed=seed)
    for i, obs in enumerate(src.next_days(days)):
        twin.update(obs, day_index=i)
    return twin


class TestUpdateCycle:
    def test_history_and_state(self):
        twin = run_twin("typical", days=30)
        assert twin.days_seen == 30
        assert len(twin.biomarker_history) == 30
        assert len(twin.factor_history) == 30
        assert len(twin.risk_history) == 30
        s = twin.state
        assert set(s.biomarkers) == {"TC", "HDL", "LDL", "TG", "CRP", "DD"}
        assert s.cluster is not None and "label" in s.cluster

    def test_factor_smoothing_reduces_jitter(self):
        twin = run_twin("typical", days=60)
        tc_series = [h["TC"] for h in twin.biomarker_history]
        assert np.std(tc_series[20:]) < 25  # smoothed, not whipsawing

    def test_rolling_history_capped(self):
        twin = run_twin("typical", days=120)
        assert len(twin.biomarker_history) <= 100


class TestTrend:
    def test_declining_increases(self):
        assert run_twin("at_risk", "declining").trend == "Increasing"

    def test_improving_decreases(self):
        assert run_twin("at_risk", "improving").trend == "Decreasing"

    def test_stable_stable(self):
        assert run_twin("healthy", "stable").trend == "Stable"

    def test_scenario_separation(self):
        dec = run_twin("at_risk", "declining").state.cvd_score
        imp = run_twin("at_risk", "improving").state.cvd_score
        assert dec > imp + 0.05


class TestBaselines:
    def test_baseline_table(self):
        twin = run_twin("typical", days=20)
        table = twin.baseline_table()
        assert len(table) == 6
        for row in table:
            assert row["n_readings"] == 20
            assert row["baseline_mean"] is not None
            assert row["current"] is not None

    def test_zscores_populate_after_warmup(self):
        twin = run_twin("typical", days=20)
        assert any(isinstance(z, float) for z in twin.state.zscores.values())


class TestAlerts:
    def test_at_risk_raises_biomarker_alerts(self):
        twin = run_twin("at_risk", "declining")
        codes = {a["code"] for a in twin.alerts_raised}
        assert codes & {"tc_high", "ldl_high", "tg_high", "crp_high", "dd_moderate",
                        "dd_high", "ctr_high", "hdl_low"}

    def test_healthy_minimal_alerts(self):
        twin = run_twin("healthy", "stable")
        assert len(twin.alerts_raised) <= 6

    def test_alert_dedupe(self):
        twin = run_twin("at_risk", "declining", days=60)
        # non-critical codes must not fire on consecutive days; critical ones
        # intentionally repeat daily until the condition resolves
        raised = twin.alerts_raised
        non_critical = [a["code"] for a in raised if a["level"] != "critical"]
        consecutive = sum(1 for a, b in zip(non_critical, non_critical[1:]) if a == b)
        assert consecutive <= 2


class TestClustering:
    def test_cohort_generation(self):
        X, cvd = generate_cohort(120, seed=1)
        assert X.shape == (120, 6)
        assert list(X.columns) == ["TC", "HDL", "LDL", "TG", "CRP", "DD"]
        assert X["TC"].between(140, 340).all()
        assert X["DD"].between(0.05, 3.2).all()

    def test_clusterer_fit_and_assign(self):
        X, cvd = generate_cohort(150, seed=3)
        cl = BiomarkerClusterer().fit(X, cvd)
        row = {"TC": 250, "HDL": 35, "LDL": 165, "TG": 280, "CRP": 5.0, "DD": 2.2}
        res = cl.assign(row)
        assert res["label"] in BiomarkerClusterer.RISK_LABELS
        assert 0.0 <= res["confidence"] <= 1.0
        assert "Very High Risk" == res["label"] or "High Risk" == res["label"]

    def test_healthy_patient_lands_low(self):
        X, cvd = generate_cohort(150, seed=3)
        cl = BiomarkerClusterer().fit(X, cvd)
        row = {"TC": 160, "HDL": 75, "LDL": 60, "TG": 100, "CRP": 0.5, "DD": 0.2}
        assert cl.assign(row)["label"] == "Low Risk"

    def test_cluster_ordering_matches_cvd(self):
        X, cvd = generate_cohort(200, seed=5)
        cl = BiomarkerClusterer().fit(X, cvd)
        order = sorted(cl.cluster_cvd_, key=cl.cluster_cvd_.get)
        labels = [cl.labels_map_[c] for c in order]
        assert labels == BiomarkerClusterer.RISK_LABELS  # Low ... Very High


class TestSnapshot:
    def test_snapshot_contents(self):
        twin = run_twin("typical", days=15)
        snap = twin.snapshot()
        assert {"profile", "state", "days_seen", "trend", "risk_history",
                "biomarker_history", "alerts_raised", "baselines"} <= set(snap)
        assert snap["state"]["top_contributions"]["TC"]
