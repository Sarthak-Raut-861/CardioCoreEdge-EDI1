"""
lipid_calculator.py
===================
Lipid profile analysis for the CardioCore digital twin.

Provides:
    * ``LipidPanel``      – container for a standard lipid panel (mg/dL)
    * ``LipidCalculator`` – LDL-C estimation (Friedewald + Sampson/NIH eq. 2),
                            non-HDL-C, atherogenic ratios, guideline categories,
                            and a normalized 0-1 lipid risk score.

Units: all concentrations are in mg/dL.

Clinical notes
--------------
* Friedewald LDL:  LDL = TC - HDL - TG/5           (valid for TG < 400 mg/dL)
* Sampson (NIH eq.2):
      LDL = TC/0.948 - HDL/0.971 - (TG/8.56 + TG*(TC-HDL)/2140 - TG^2/16100) - 9.44
  (valid for TG < 800 mg/dL) — see CDC NHANES documentation.
* Directly measured LDL, when available, always takes precedence.

This module is for research/educational use and is NOT a diagnostic device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

__all__ = ["LipidPanel", "LipidAnalysis", "LipidCalculator"]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class LipidPanel:
    """A standard fasting lipid panel in mg/dL."""

    total_cholesterol: float
    hdl: float
    triglycerides: float
    ldl_direct: Optional[float] = None  # measured LDL if available

    def __post_init__(self) -> None:
        for name in ("total_cholesterol", "hdl", "triglycerides"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"LipidPanel field '{name}' must be a positive finite number, got {value!r}")
        if self.ldl_direct is not None and self.ldl_direct <= 0:
            raise ValueError("ldl_direct must be positive when provided")

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "LipidPanel":
        """Build a panel from a plain dict (keys match attribute names)."""
        allowed = {"total_cholesterol", "hdl", "triglycerides", "ldl_direct"}
        kwargs = {k: float(v) for k, v in data.items() if k in allowed}
        missing = {"total_cholesterol", "hdl", "triglycerides"} - kwargs.keys()
        if missing:
            raise ValueError(f"LipidPanel missing required fields: {sorted(missing)}")
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass
class LipidAnalysis:
    """Complete result of analysing one lipid panel."""

    panel: LipidPanel
    ldl: Optional[float] = None          # best available LDL (direct > Sampson > Friedewald)
    ldl_method: Optional[str] = None
    friedewald_ldl: Optional[float] = None
    sampson_ldl: Optional[float] = None
    non_hdl: Optional[float] = None
    tc_hdl_ratio: Optional[float] = None
    tg_hdl_ratio: Optional[float] = None
    atherogenic_index: Optional[float] = None  # log10(TG/HDL)
    ldl_category: Optional[str] = None
    non_hdl_category: Optional[str] = None
    risk_score: float = 0.0              # 0 (optimal) .. 1 (very high risk)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["panel"] = self.panel.to_dict()
        return out


# --------------------------------------------------------------------------- #
# Guideline thresholds / anchor tables
# --------------------------------------------------------------------------- #
# (upper-limit, category) – LDL-C mg/dL, aligned with ATP III / ESC bands
LDL_CATEGORIES: List[Tuple[float, str]] = [
    (70.0, "optimal (low risk)"),
    (100.0, "near/above optimal"),
    (130.0, "borderline high"),
    (160.0, "high"),
    (190.0, "very high"),
    (float("inf"), "very high"),
]

NON_HDL_CATEGORIES: List[Tuple[float, str]] = [
    (130.0, "optimal"),
    (160.0, "above optimal"),
    (190.0, "borderline high"),
    (220.0, "high"),
    (float("inf"), "very high"),
]

# Anchor tables for piecewise-linear normalization to 0..1 risk points.
# (value, risk_points); interpolation between anchors, clamped at the ends.
_LDL_ANCHORS: List[Tuple[float, float]] = [
    (50, 0.00), (70, 0.05), (100, 0.20), (130, 0.45), (160, 0.70), (190, 0.90), (250, 1.00),
]
_NONHDL_ANCHORS: List[Tuple[float, float]] = [
    (80, 0.00), (130, 0.15), (160, 0.40), (190, 0.65), (220, 0.85), (260, 1.00),
]
_TCHDL_ANCHORS: List[Tuple[float, float]] = [
    (2.5, 0.00), (3.5, 0.10), (5.0, 0.40), (6.5, 0.70), (8.0, 0.90), (10.0, 1.00),
]
_TG_ANCHORS: List[Tuple[float, float]] = [
    (75, 0.00), (150, 0.25), (200, 0.50), (300, 0.75), (500, 1.00),
]
_HDL_ANCHORS: List[Tuple[float, float]] = [  # low HDL is risky (inverted)
    (25, 1.00), (35, 0.75), (45, 0.45), (55, 0.15), (70, 0.00),
]


def _piecewise(anchors: List[Tuple[float, float]], value: float) -> float:
    """Piecewise-linear interpolation through (value, score) anchors, clamped."""
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            frac = (value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return anchors[-1][1]


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #
class LipidCalculator:
    """Static methods + one-shot ``analyze`` for a full lipid work-up."""

    MAX_FRIEDEWALD_TG = 400.0   # mg/dL
    MAX_SAMPSON_TG = 800.0      # mg/dL

    # ---- individual quantities ------------------------------------------- #
    @staticmethod
    def friedewald_ldl(panel: LipidPanel) -> Optional[float]:
        """LDL = TC - HDL - TG/5. Returns None when TG >= 400 (invalid)."""
        if panel.triglycerides >= LipidCalculator.MAX_FRIEDEWALD_TG:
            return None
        ldl = panel.total_cholesterol - panel.hdl - panel.triglycerides / 5.0
        return round(max(ldl, 0.0), 1)

    @staticmethod
    def sampson_ldl(panel: LipidPanel) -> Optional[float]:
        """Sampson / NIH equation 2 (valid TG < 800 mg/dL). Returns None otherwise."""
        tg = panel.triglycerides
        if tg >= LipidCalculator.MAX_SAMPSON_TG:
            return None
        tc, hdl = panel.total_cholesterol, panel.hdl
        non_hdl = tc - hdl
        ldl = tc / 0.948 - hdl / 0.971 - (tg / 8.56 + tg * non_hdl / 2140.0 - tg * tg / 16100.0) - 9.44
        return round(max(ldl, 0.0), 1)

    @staticmethod
    def non_hdl(panel: LipidPanel) -> float:
        return round(panel.total_cholesterol - panel.hdl, 1)

    @staticmethod
    def tc_hdl_ratio(panel: LipidPanel) -> float:
        return round(panel.total_cholesterol / panel.hdl, 2)

    @staticmethod
    def tg_hdl_ratio(panel: LipidPanel) -> float:
        return round(panel.triglycerides / panel.hdl, 2)

    @staticmethod
    def atherogenic_index(panel: LipidPanel) -> float:
        """AIP = log10(TG/HDL); > 0.48 is considered atherogenic."""
        return round(math.log10(panel.triglycerides / panel.hdl), 3)

    # ---- categories ------------------------------------------------------- #
    @staticmethod
    def ldl_category(ldl: Optional[float]) -> Optional[str]:
        if ldl is None:
            return None
        for limit, label in LDL_CATEGORIES:
            if ldl < limit:
                return label
        return LDL_CATEGORIES[-1][1]

    @staticmethod
    def non_hdl_category(non_hdl: float) -> str:
        for limit, label in NON_HDL_CATEGORIES:
            if non_hdl < limit:
                return label
        return NON_HDL_CATEGORIES[-1][1]

    # ---- aggregate -------------------------------------------------------- #
    @staticmethod
    def best_ldl(panel: LipidPanel) -> Tuple[Optional[float], Optional[str]]:
        """Preferred LDL: direct measurement > Sampson > Friedewald."""
        if panel.ldl_direct is not None:
            return round(panel.ldl_direct, 1), "direct"
        ldl = LipidCalculator.sampson_ldl(panel)
        if ldl is not None:
            return ldl, "sampson"
        ldl = LipidCalculator.friedewald_ldl(panel)
        if ldl is not None:
            return ldl, "friedewald"
        return None, None

    @staticmethod
    def lipid_risk_score(panel: LipidPanel) -> float:
        """Weighted 0..1 lipid risk score from the anchor tables."""
        ldl, method = LipidCalculator.best_ldl(panel)
        parts: List[Tuple[float, float]] = [  # (weight, score)
            (0.35, _piecewise(_LDL_ANCHORS, ldl) if ldl is not None else 0.3),
            (0.25, _piecewise(_NONHDL_ANCHORS, LipidCalculator.non_hdl(panel))),
            (0.20, _piecewise(_TCHDL_ANCHORS, LipidCalculator.tc_hdl_ratio(panel))),
            (0.10, _piecewise(_TG_ANCHORS, panel.triglycerides)),
            (0.10, _piecewise(_HDL_ANCHORS, panel.hdl)),
        ]
        total_w = sum(w for w, _ in parts)
        return round(sum(w * s for w, s in parts) / total_w, 3)

    # ---- one-shot --------------------------------------------------------- #
    @staticmethod
    def analyze(panel: LipidPanel) -> LipidAnalysis:
        """Full analysis of a panel (also accepts a dict via LipidPanel.from_dict)."""
        if isinstance(panel, dict):
            panel = LipidPanel.from_dict(panel)

        ldl, method = LipidCalculator.best_ldl(panel)
        non_hdl = LipidCalculator.non_hdl(panel)
        return LipidAnalysis(
            panel=panel,
            ldl=ldl,
            ldl_method=method,
            friedewald_ldl=LipidCalculator.friedewald_ldl(panel),
            sampson_ldl=LipidCalculator.sampson_ldl(panel),
            non_hdl=non_hdl,
            tc_hdl_ratio=LipidCalculator.tc_hdl_ratio(panel),
            tg_hdl_ratio=LipidCalculator.tg_hdl_ratio(panel),
            atherogenic_index=LipidCalculator.atherogenic_index(panel),
            ldl_category=LipidCalculator.ldl_category(ldl),
            non_hdl_category=LipidCalculator.non_hdl_category(non_hdl),
            risk_score=LipidCalculator.lipid_risk_score(panel),
        )
