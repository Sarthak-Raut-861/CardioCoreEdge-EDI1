"""
factor_engine.py
================
Converts multimodal daily observations into **normalized, weighted
cardiovascular risk factors**.

Every factor is scored on a 0..1 scale where 0 = optimal and 1 = high risk,
using piecewise-linear anchor tables derived from public-health reference
ranges (AHA/ESC-style bands). A weighted composite blends the available
factors and reports coverage, so partially-missing data degrades gracefully.

Factor families
---------------
* wearable  – resting HR, HRV (RMSSD), BP, nocturnal SpO2, sleep, activity
* lab       – lipid-panel factors (delegates to ``lipid_calculator``)
* demographic – age, sex, BMI, smoking, diabetes, family history

Educational/research use only – not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .lipid_calculator import LipidPanel, LipidCalculator, _piecewise  # reuse helpers

__all__ = ["Factor", "CompositeResult", "FactorEngine"]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Factor:
    """One evaluated risk factor."""

    key: str
    name: str
    modality: str            # wearable | lab | demographic
    value: Optional[float]   # raw measurement (None if unavailable)
    score: float             # 0..1 risk points
    weight: float            # relative importance in composite
    direction: str           # "higher_is_worse" | "lower_is_worse" | "u_shaped"
    reference: str           # human-readable optimal range

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompositeResult:
    """Weighted blend of evaluated factors."""

    score: float                     # 0..1 composite risk
    coverage: float                  # fraction of expected factors available
    n_factors: int
    breakdown: List[Factor]          # sorted by weighted contribution (desc)

    def top(self, n: int = 5) -> List[Factor]:
        return self.breakdown[:n]

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "coverage": self.coverage,
            "n_factors": self.n_factors,
            "breakdown": [f.to_dict() for f in self.breakdown],
        }


# --------------------------------------------------------------------------- #
# Anchor tables (value -> 0..1 risk points)
# --------------------------------------------------------------------------- #
_ANCHORS = {
    "resting_heart_rate": [(40, 0.00), (50, 0.03), (60, 0.12), (70, 0.32), (80, 0.58), (90, 0.82), (100, 1.00), (130, 1.00)],
    "hrv_rmssd":          [(15, 1.00), (20, 0.92), (30, 0.75), (40, 0.52), (55, 0.28), (70, 0.12), (90, 0.02), (120, 0.00)],  # lower is worse
    "systolic_bp":        [(95, 0.00), (110, 0.04), (120, 0.14), (130, 0.36), (140, 0.56), (160, 0.80), (180, 1.00), (220, 1.00)],
    "diastolic_bp":       [(60, 0.00), (70, 0.04), (80, 0.16), (90, 0.42), (100, 0.66), (110, 0.85), (120, 1.00), (140, 1.00)],
    "nocturnal_spo2_pct": [(88, 1.00), (90, 0.88), (92, 0.68), (94, 0.38), (95, 0.22), (96, 0.12), (98, 0.02), (100, 0.00)],  # lower is worse
    "spo2_dips":          [(0, 0.00), (2, 0.15), (5, 0.45), (10, 0.75), (20, 1.00), (50, 1.00)],
    "sleep_efficiency":   [(0.55, 1.00), (0.65, 0.80), (0.75, 0.55), (0.85, 0.28), (0.92, 0.10), (1.00, 0.00)],               # lower is worse
    "age_years":          [(20, 0.00), (30, 0.04), (40, 0.12), (50, 0.26), (60, 0.46), (70, 0.70), (80, 0.92), (90, 1.00)],
    "bmi":                [(17, 0.60), (18.5, 0.30), (21, 0.05), (25, 0.18), (27.5, 0.42), (30, 0.62), (35, 0.85), (45, 1.00)],  # u-shaped
    "tc_hdl_ratio":       [(2.5, 0.00), (3.5, 0.10), (5.0, 0.40), (6.5, 0.70), (8.0, 0.90), (10.0, 1.00)],
    "non_hdl":            [(80, 0.00), (130, 0.15), (160, 0.40), (190, 0.65), (220, 0.85), (260, 1.00)],
    "ldl":                [(50, 0.00), (70, 0.05), (100, 0.20), (130, 0.45), (160, 0.70), (190, 0.90), (250, 1.00)],
    "triglycerides":      [(75, 0.00), (150, 0.25), (200, 0.50), (300, 0.75), (500, 1.00)],
    "hdl":                [(25, 1.00), (35, 0.75), (45, 0.45), (55, 0.15), (70, 0.00)],                                      # lower is worse
}


def _u_shaped_sleep_hours(hours: float) -> float:
    """7-9 h optimal; risk rises for short *and* long sleep."""
    if hours >= 7.0 and hours <= 9.0:
        return 0.02
    if hours < 7.0:
        return min(1.0, 0.02 + (7.0 - hours) * 0.28)
    return min(1.0, 0.02 + (hours - 9.0) * 0.35)


def _u_shaped_deep_sleep(pct: float) -> float:
    if pct >= 13.0 and pct <= 25.0:
        return 0.05
    if pct < 13.0:
        return min(1.0, 0.05 + (13.0 - pct) * 0.09)
    return min(1.0, 0.05 + (pct - 25.0) * 0.06)


def _activity_score(steps: float, active_minutes: Optional[float]) -> float:
    """Steps dominate; active minutes modulate (higher activity = lower risk)."""
    s = _piecewise([(0, 1.00), (2500, 0.82), (5000, 0.58), (7500, 0.34), (10000, 0.14), (13000, 0.00)], steps)
    if active_minutes is not None:
        a = _piecewise([(0, 1.00), (15, 0.75), (30, 0.45), (45, 0.22), (60, 0.08), (90, 0.00)], active_minutes)
        return round(0.6 * s + 0.4 * a, 3)
    return round(s, 3)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class FactorEngine:
    """Evaluates and blends risk factors from any combination of inputs."""

    # (key, name, weight, direction, reference)
    WEARABLE_SPECS: Sequence[Tuple[str, str, float, str, str]] = (
        ("resting_heart_rate", "Resting heart rate", 1.30, "higher_is_worse", "50-65 bpm"),
        ("hrv_rmssd", "HRV (RMSSD)", 1.25, "lower_is_worse", ">45 ms"),
        ("systolic_bp", "Systolic blood pressure", 1.60, "higher_is_worse", "<120 mmHg"),
        ("diastolic_bp", "Diastolic blood pressure", 1.00, "higher_is_worse", "<80 mmHg"),
        ("nocturnal_spo2_pct", "Nocturnal SpO2", 1.20, "lower_is_worse", ">95 %"),
        ("spo2_dips", "SpO2 dips per night", 0.80, "higher_is_worse", "0-2"),
        ("sleep_hours", "Sleep duration", 0.90, "u_shaped", "7-9 h"),
        ("deep_sleep_pct", "Deep sleep share", 0.60, "u_shaped", "13-25 %"),
        ("sleep_efficiency", "Sleep efficiency", 0.60, "lower_is_worse", ">88 %"),
        ("activity", "Daily activity", 0.90, "lower_is_worse", ">=8k steps / >=30 min active"),
    )
    LAB_SPECS: Sequence[Tuple[str, str, float, str, str]] = (
        ("ldl", "LDL cholesterol", 1.40, "higher_is_worse", "<100 mg/dL"),
        ("non_hdl", "Non-HDL cholesterol", 1.10, "higher_is_worse", "<130 mg/dL"),
        ("tc_hdl_ratio", "TC/HDL ratio", 1.00, "higher_is_worse", "<4.5"),
        ("triglycerides", "Triglycerides", 0.70, "higher_is_worse", "<150 mg/dL"),
        ("hdl", "HDL cholesterol", 0.70, "lower_is_worse", ">55 mg/dL"),
    )
    DEMOGRAPHIC_SPECS: Sequence[Tuple[str, str, float, str, str]] = (
        ("age_years", "Age", 0.90, "higher_is_worse", "-"),
        ("bmi", "Body-mass index", 0.70, "u_shaped", "18.5-24.9"),
        ("smoking", "Smoking status", 1.60, "higher_is_worse", "non-smoker"),
        ("diabetes", "Diabetes", 1.30, "higher_is_worse", "no"),
        ("family_history", "Family history of CVD", 0.80, "higher_is_worse", "no"),
    )

    # ------------------------------------------------------------------ #
    # Wearables
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate_wearables(cls, obs: Mapping[str, float]) -> List[Factor]:
        factors: List[Factor] = []
        for key, name, weight, direction, ref in cls.WEARABLE_SPECS:
            if key == "activity":
                steps, active = obs.get("steps"), obs.get("active_minutes")
                if steps is None and active is None:
                    factors.append(Factor(key, name, "wearable", None, 0.0, weight, direction, ref))
                    continue
                val = steps if steps is not None else active
                factors.append(Factor(key, name, "wearable", val, _activity_score(steps or 0.0, active), weight, direction, ref))
                continue

            raw = obs.get(key)
            if raw is None:
                factors.append(Factor(key, name, "wearable", None, 0.0, weight, direction, ref))
                continue

            if key == "sleep_hours":
                score = _u_shaped_sleep_hours(float(raw))
            elif key == "deep_sleep_pct":
                score = _u_shaped_deep_sleep(float(raw))
            else:
                score = _piecewise(_ANCHORS[key], float(raw))
            factors.append(Factor(key, name, "wearable", float(raw), round(score, 3), weight, direction, ref))
        return factors

    # ------------------------------------------------------------------ #
    # Labs
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate_labs(cls, labs: Optional[Mapping[str, float]]) -> List[Factor]:
        if not labs:
            return [Factor(k, n, "lab", None, 0.0, w, d, r) for k, n, w, d, r in cls.LAB_SPECS]
        analysis = LipidCalculator.analyze(LipidPanel.from_dict(dict(labs)))
        values = {
            "ldl": analysis.ldl,
            "non_hdl": analysis.non_hdl,
            "tc_hdl_ratio": analysis.tc_hdl_ratio,
            "triglycerides": labs.get("triglycerides"),
            "hdl": labs.get("hdl"),
        }
        factors: List[Factor] = []
        for key, name, weight, direction, ref in cls.LAB_SPECS:
            val = values.get(key)
            if val is None:
                factors.append(Factor(key, name, "lab", None, 0.0, weight, direction, ref))
                continue
            score = _piecewise(_ANCHORS[key], float(val))
            factors.append(Factor(key, name, "lab", float(val), round(score, 3), weight, direction, ref))
        return factors

    # ------------------------------------------------------------------ #
    # Demographics
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate_demographics(cls, profile: Mapping) -> List[Factor]:
        age = profile.get("age")
        height = profile.get("height_cm")
        weight = profile.get("weight_kg")
        bmi = None
        if height and weight:
            bmi = weight / (height / 100.0) ** 2

        prepared = {
            "age_years": age,
            "bmi": bmi,
            "smoking": 1.0 if profile.get("smoker") else 0.0,
            "diabetes": 1.0 if profile.get("diabetic") else 0.0,
            "family_history": 1.0 if profile.get("family_history") else 0.0,
        }
        binary_scores = {"smoking": 0.9, "diabetes": 0.85, "family_history": 0.55}  # fixed risk-point values when present
        factors: List[Factor] = []
        for key, name, weight, direction, ref in cls.DEMOGRAPHIC_SPECS:
            val = prepared.get(key)
            if val is None:
                factors.append(Factor(key, name, "demographic", None, 0.0, weight, direction, ref))
                continue
            if key in binary_scores:
                score = binary_scores[key] if val >= 0.5 else 0.0
            else:
                score = _piecewise(_ANCHORS[key], float(val))
            factors.append(Factor(key, name, "demographic", round(float(val), 2), round(score, 3), weight, direction, ref))
        return factors

    # ------------------------------------------------------------------ #
    # Composite
    # ------------------------------------------------------------------ #
    @staticmethod
    def composite(factors: Sequence[Factor]) -> CompositeResult:
        """Weighted mean over *available* factors; unavailable ones excluded."""
        available = [f for f in factors if f.value is not None]
        total_weight_expected = sum(f.weight for f in factors)
        if not available or total_weight_expected == 0:
            return CompositeResult(0.0, 0.0, 0, list(factors))
        total_weight = sum(f.weight for f in available)
        score = sum(f.weight * f.score for f in available) / total_weight
        breakdown = sorted(available, key=lambda f: f.weight * f.score, reverse=True)
        coverage = round(total_weight / total_weight_expected, 3)
        return CompositeResult(round(score, 3), coverage, len(available), breakdown)

    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate_all(
        cls,
        obs: Mapping[str, float],
        labs: Optional[Mapping[str, float]] = None,
        profile: Optional[Mapping] = None,
    ) -> CompositeResult:
        factors: List[Factor] = []
        if profile:
            factors += cls.evaluate_demographics(profile)
        factors += cls.evaluate_wearables(obs)
        factors += cls.evaluate_labs(labs)
        return cls.composite(factors)
