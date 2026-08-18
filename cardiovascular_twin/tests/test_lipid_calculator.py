"""Tests for src/lipid_calculator.py"""

import pytest

from src.lipid_calculator import LipidCalculator, LipidPanel


HEALTHY = LipidPanel(total_cholesterol=175, hdl=62, triglycerides=95)
AT_RISK = LipidPanel(total_cholesterol=240, hdl=35, triglycerides=260)


class TestLipidPanel:
    def test_from_dict_roundtrip(self):
        panel = LipidPanel.from_dict({"total_cholesterol": 200, "hdl": 50, "triglycerides": 150, "ldl_direct": 120})
        assert panel.total_cholesterol == 200
        assert panel.ldl_direct == 120

    def test_from_dict_missing_required_raises(self):
        with pytest.raises(ValueError):
            LipidPanel.from_dict({"total_cholesterol": 200, "hdl": 50})

    def test_nonpositive_values_rejected(self):
        with pytest.raises(ValueError):
            LipidPanel(total_cholesterol=0, hdl=50, triglycerides=150)
        with pytest.raises(ValueError):
            LipidPanel(total_cholesterol=-5, hdl=50, triglycerides=150)


class TestLDLEstimation:
    def test_friedewald_known_value(self):
        # TC 200, HDL 50, TG 150 -> 200 - 50 - 30 = 120
        panel = LipidPanel(total_cholesterol=200, hdl=50, triglycerides=150)
        assert LipidCalculator.friedewald_ldl(panel) == 120.0

    def test_sampson_matches_formula(self):
        tc, hdl, tg = 200.0, 50.0, 150.0
        expected = tc / 0.948 - hdl / 0.971 - (tg / 8.56 + tg * (tc - hdl) / 2140.0 - tg * tg / 16100.0) - 9.44
        panel = LipidPanel(total_cholesterol=tc, hdl=hdl, triglycerides=tg)
        assert LipidCalculator.sampson_ldl(panel) == pytest.approx(expected, abs=0.1)

    def test_sampson_close_to_friedewald_at_normal_tg(self):
        panel = LipidPanel(total_cholesterol=200, hdl=50, triglycerides=120)
        assert abs(LipidCalculator.sampson_ldl(panel) - LipidCalculator.friedewald_ldl(panel)) < 8

    def test_friedewald_invalid_high_tg(self):
        panel = LipidPanel(total_cholesterol=250, hdl=40, triglycerides=500)
        assert LipidCalculator.friedewald_ldl(panel) is None
        assert LipidCalculator.sampson_ldl(panel) is not None  # valid below 800

    def test_both_invalid_extreme_tg(self):
        panel = LipidPanel(total_cholesterol=300, hdl=35, triglycerides=900)
        assert LipidCalculator.friedewald_ldl(panel) is None
        assert LipidCalculator.sampson_ldl(panel) is None

    def test_best_ldl_prefers_direct(self):
        panel = LipidPanel(total_cholesterol=200, hdl=50, triglycerides=150, ldl_direct=115)
        ldl, method = LipidCalculator.best_ldl(panel)
        assert (ldl, method) == (115.0, "direct")

    def test_clamped_at_zero(self):
        panel = LipidPanel(total_cholesterol=90, hdl=80, triglycerides=50)
        assert LipidCalculator.friedewald_ldl(panel) == 0.0


class TestDerivedQuantities:
    def test_non_hdl(self):
        assert LipidCalculator.non_hdl(LipidPanel(total_cholesterol=200, hdl=50, triglycerides=150)) == 150.0

    def test_ratios(self):
        panel = LipidPanel(total_cholesterol=200, hdl=50, triglycerides=150)
        assert LipidCalculator.tc_hdl_ratio(panel) == 4.0
        assert LipidCalculator.tg_hdl_ratio(panel) == 3.0

    def test_atherogenic_index(self):
        import math
        panel = LipidPanel(total_cholesterol=200, hdl=50, triglycerides=100)
        assert LipidCalculator.atherogenic_index(panel) == pytest.approx(math.log10(2.0), abs=1e-3)


class TestCategoriesAndScore:
    def test_ldl_categories(self):
        assert LipidCalculator.ldl_category(60) == "optimal (low risk)"
        assert LipidCalculator.ldl_category(115) == "borderline high"
        assert LipidCalculator.ldl_category(250) == "very high"
        assert LipidCalculator.ldl_category(None) is None

    def test_non_hdl_categories(self):
        assert LipidCalculator.non_hdl_category(120) == "optimal"
        assert LipidCalculator.non_hdl_category(250) == "very high"

    def test_risk_score_range_and_ordering(self):
        h = LipidCalculator.lipid_risk_score(HEALTHY)
        b = LipidCalculator.lipid_risk_score(AT_RISK)
        assert 0.0 <= h <= 1.0 and 0.0 <= b <= 1.0
        assert b > h

    def test_analyze_full_output(self):
        result = LipidCalculator.analyze(HEALTHY)
        d = result.to_dict()
        assert result.ldl is not None
        assert result.ldl_method in {"direct", "sampson", "friedewald"}
        assert d["panel"]["hdl"] == 62
        assert result.ldl_category is not None
