"""Tests for the biomarker calculation engine (src/biomarker_engine.py)."""

import numpy as np
import pytest

from src.biomarker_engine import BiomarkerEngine, BiomarkerResult
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource
from src.factors72 import FACTOR_IDS, build_factor_vector


def vector_for(profile_name: str, seed: int = 42, day: int = 30) -> dict:
    prof = PRESET_PROFILES[profile_name]
    src = SimulatedWearableSource(prof, seed=seed)
    obs_list = src.next_days(day + 1)
    return build_factor_vector(obs_list[-1], prof.to_dict(), day)


class TestFormulas:
    def test_zero_factors_give_base_values(self):
        vec = {fid: 0.0 for fid in FACTOR_IDS}
        r = BiomarkerEngine.compute(vec)
        assert r.values["TC"] == 150.0
        assert r.values["TG"] == 80.0
        assert r.values["CRP"] == 0.2
        assert r.values["HDL"] == 80.0  # 80 - 0
        assert r.values["DD"] == 0.1

    def test_values_within_documented_ranges(self):
        for name in ("healthy", "typical", "at_risk"):
            r = BiomarkerEngine.compute(vector_for(name))
            assert 150 <= r.values["TC"] <= 320
            assert 20 <= r.values["HDL"] <= 80
            assert 80 <= r.values["TG"] <= 500
            assert 30 <= r.values["LDL"] <= 300
            assert 0.2 <= r.values["CRP"] <= 10
            assert 0.1 <= r.values["DD"] <= 3.0

    def test_friedewald_consistency(self):
        r = BiomarkerEngine.compute(vector_for("typical"))
        expected = r.values["TC"] - r.values["HDL"] - r.values["TG"] / 5
        assert r.values["LDL"] == pytest.approx(expected, abs=0.2)

    def test_profile_ordering(self):
        h = BiomarkerEngine.compute(vector_for("healthy"))
        a = BiomarkerEngine.compute(vector_for("at_risk"))
        assert a.values["TC"] > h.values["TC"]
        assert a.values["TG"] > h.values["TG"]
        assert a.values["CRP"] > h.values["CRP"]
        assert a.values["DD"] > h.values["DD"]
        assert a.values["HDL"] < h.values["HDL"]

    def test_dd_crp_linkage(self):
        """D-Dimer must rise when CRP-driving factors rise (doc §6.11)."""
        vec = vector_for("typical")
        r1 = BiomarkerEngine.compute(vec)
        vec2 = dict(vec)
        for fid in ("F34", "F35", "F36", "F46", "F50", "F55", "F66", "F72"):
            vec2[fid] = 1.0
        r2 = BiomarkerEngine.compute(vec2)
        assert r2.values["DD"] > r1.values["DD"]
        assert r2.values["CRP"] > r1.values["CRP"]


class TestCategories:
    def test_thresholds(self):
        r = BiomarkerEngine.__new__(BiomarkerResult)
        from src.biomarker_engine import (_categorize, TC_BANDS, HDL_BANDS, LDL_BANDS,
                                          TG_BANDS, CRP_BANDS, DD_BANDS)
        assert _categorize(180, TC_BANDS) == "Desirable"
        assert _categorize(220, TC_BANDS) == "Borderline High"
        assert _categorize(260, TC_BANDS) == "High"
        assert _categorize(65, HDL_BANDS) == "Protective"
        assert _categorize(35, HDL_BANDS) == "Low (Risk)"
        assert _categorize(90, LDL_BANDS) == "Optimal"
        assert _categorize(200, LDL_BANDS) == "Very High"
        assert _categorize(130, TG_BANDS) == "Normal"
        assert _categorize(0.5, CRP_BANDS) == "Low CV Risk"
        assert _categorize(0.7, DD_BANDS) == "Mild Elevation"

    def test_result_categories_present(self):
        r = BiomarkerEngine.compute(vector_for("at_risk"))
        assert set(r.categories) == {"TC", "HDL", "LDL", "TG", "CRP", "DD"}


class TestRiskScores:
    def test_scores_bounded(self):
        for name in ("healthy", "typical", "at_risk"):
            r = BiomarkerEngine.compute(vector_for(name))
            assert 0.0 <= r.cvd_score <= 1.0
            assert 0.0 <= r.ctr <= 1.0
            assert set(r.pathway_risk) == {"lipid", "inflammation", "thrombosis",
                                           "hemodynamic", "autonomic", "metabolic"}

    def test_at_risk_scores_higher(self):
        h = BiomarkerEngine.compute(vector_for("healthy"))
        a = BiomarkerEngine.compute(vector_for("at_risk"))
        assert a.cvd_score > h.cvd_score + 0.15
        assert a.ctr > h.ctr

    def test_derived_values(self):
        r = BiomarkerEngine.compute(vector_for("typical"))
        assert r.derived["VLDL"] == pytest.approx(r.values["TG"] / 5, abs=0.15)
        assert r.derived["Non_HDL"] == pytest.approx(r.values["TC"] - r.values["HDL"], abs=0.15)


class TestContributions:
    def test_sorted_and_signed(self):
        r = BiomarkerEngine.compute(vector_for("at_risk"))
        for bm, contribs in r.contributions.items():
            mags = [abs(c.contribution) for c in contribs]
            assert mags == sorted(mags, reverse=True)
            for c in contribs:
                if c.direction == "protective":
                    assert c.contribution <= 0
                else:
                    assert c.contribution >= 0

    def test_top_documented_factors_dominate_tc(self):
        r = BiomarkerEngine.compute(vector_for("at_risk"))
        names = [c.fid for c in r.top("TC", 8)]
        assert "F1" in names or "F7" in names   # NIR factors
