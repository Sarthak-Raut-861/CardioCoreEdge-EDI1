"""
biomarker_engine.py
===================
Biomarker estimation from the 72-factor vector (doc Chapter 6).

Formulas (all constants per the project documentation §6.2-6.11):

    TC     = 150 + (Σ W_TC·F) × 0.486              range 150-320 mg/dL
    HDL    = 80  − (Σ W_HDL·F·d) × 0.45            range 20-80   mg/dL
    TG     = 80  + (Σ W_TG·F) × 1.4                range 80-500  mg/dL
    LDL    = TC − HDL − TG/5   [TG<400]            Friedewald
             TC − HDL − TG/6   [TG≥400]            Martin-Hopkins style
    CRP    = 0.2 + (Σ W_CRP·F) × 0.085             range 0.2-10  mg/L
    D-Dimer= 0.1 + (Σ W_DD·F) × 0.022 + CRP_norm×1.2   range 0.1-3.0 mg/L

Derived: VLDL = TG/5 · Non-HDL = TC−HDL · TC/HDL · LDL/HDL · AIP = log10(TG/HDL)

Interactions (§6.11):
    CRP multiplies TC risk context (1.0 / 1.2 / 1.5)
    HDL modifies CRP (×0.7 / ×1.0 / ×1.3)
    D-Dimer already includes CRP linkage (+CRP_norm × 1.2)
    CTR = 0.35·LDL_n + 0.40·DD_n + 0.25·CRP_n
    CVD = 0.20·TC_n + 0.20·LDL_n + 0.10·HDL_n + 0.10·TG_n + 0.20·CRP_n + 0.20·DD_n

Risk direction: protective factors carry direction −1 in the registry, so they
*reduce* the weighted sum (raising HDL, lowering CRP, ...).

RESEARCH PROTOTYPE — estimates, not measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from .factors72 import FACTORS, FACTOR_IDS, FactorSpec

__all__ = ["BiomarkerResult", "Contribution", "BiomarkerEngine", "CVD_CATEGORIES"]

# scaling factors (doc §6.3-6.9) — the documentation's constants apply with its
# illustrative top-weights; the full 72-factor weight matrix additionally needs
# a range-calibration factor so realistic profiles can actually span the
# documented biomarker ranges (TC 150-320, HDL 20-80, TG 80-500, CRP 0.2-10,
# DD 0.1-3). CALIBRATED_SCALING = doc_scaling x calibration, computed once from
# the weight matrix assuming a realistic worst-case profile (risk factors at
# 0.85, protective factors at 0.15) reaching ~92% of each range.
DOC_SCALING = {"TC": 0.486, "HDL": 0.45, "TG": 1.4, "CRP": 0.085, "DD": 0.022}
BASES = {"TC": 150.0, "HDL": 80.0, "TG": 80.0, "CRP": 0.2, "DD": 0.1}
RANGES = {  # (min, max) clipping per doc
    "TC": (150.0, 320.0), "HDL": (20.0, 80.0), "TG": (80.0, 500.0),
    "CRP": (0.2, 10.0), "DD": (0.1, 3.0),
}


def _calibrate(risk_hi: float = 0.85, risk_lo: float = 0.15, fill: float = 0.92) -> Dict[str, float]:
    """Per-biomarker calibration factor mapping the weight matrix to doc ranges."""
    calib: Dict[str, float] = {}
    for b in ("TC", "HDL", "TG", "CRP", "DD"):
        pos = neg = 0.0
        for spec in FACTORS.values():
            w, d = spec.weights.get(b, (0.0, 1))
            if w <= 0:
                continue
            if d > 0:
                pos += w
            else:
                neg += w
        net_max = risk_hi * pos - risk_lo * neg   # worst realistic net weighted sum
        span = RANGES[b][1] - RANGES[b][0]
        calib[b] = (fill * span / DOC_SCALING[b] / net_max) if net_max > 0 else 1.0
    return calib


SCALING = {b: round(DOC_SCALING[b] * c, 6) for b, c in _calibrate().items()}
DIRECT = ("TC", "TG", "CRP")          # computed directly from weighted sums
LDL_FRIEDEWALD_MAX_TG = 400.0


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Contribution:
    fid: str
    name: str
    value: float            # normalized factor value 0-1
    weight: float
    contribution: float     # value × weight × direction
    direction: str          # "risk-increasing" / "protective"

    def to_dict(self) -> Dict:
        return {"fid": self.fid, "name": self.name, "value": self.value,
                "weight": self.weight, "contribution": round(self.contribution, 3),
                "direction": self.direction}


@dataclass
class BiomarkerResult:
    day_index: int = 0
    values: Dict[str, float] = field(default_factory=dict)      # TC/HDL/TG/LDL/CRP/DD
    derived: Dict[str, float] = field(default_factory=dict)     # VLDL/non-HDL/ratios/AIP
    categories: Dict[str, str] = field(default_factory=dict)
    adjusted: Dict[str, float] = field(default_factory=dict)    # post-interaction values
    cvd_score: float = 0.0
    cvd_category: str = "Low"
    ctr: float = 0.0
    ctr_category: str = "Low"
    pathway_risk: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, List[Contribution]] = field(default_factory=dict)
    n_factors: int = 0

    def top(self, biomarker: str, n: int = 5) -> List[Contribution]:
        return self.contributions.get(biomarker, [])[:n]

    def to_dict(self) -> Dict:
        return {
            "day_index": self.day_index,
            "values": self.values,
            "derived": self.derived,
            "categories": self.categories,
            "adjusted": self.adjusted,
            "cvd_score": self.cvd_score,
            "cvd_category": self.cvd_category,
            "ctr": self.ctr,
            "ctr_category": self.ctr_category,
            "pathway_risk": self.pathway_risk,
            "n_factors": self.n_factors,
            "contributions": {b: [c.to_dict() for c in cs] for b, cs in self.contributions.items()},
        }


# --------------------------------------------------------------------------- #
# Clinical categories (doc §6.10)
# --------------------------------------------------------------------------- #
def _categorize(value: float, bands: Sequence) -> str:
    for limit, label in bands:
        if value < limit:
            return label
    return bands[-1][1]


TC_BANDS = [(200, "Desirable"), (240, "Borderline High"), (float("inf"), "High")]
HDL_BANDS = [(40, "Low (Risk)"), (60, "Normal"), (float("inf"), "Protective")]
LDL_BANDS = [(100, "Optimal"), (130, "Near Optimal"), (160, "Borderline High"),
             (190, "High"), (float("inf"), "Very High")]
TG_BANDS = [(150, "Normal"), (200, "Borderline High"), (500, "High"), (float("inf"), "Very High")]
CRP_BANDS = [(1.0, "Low CV Risk"), (3.0, "Moderate CV Risk"), (10.0, "High CV Risk"),
             (float("inf"), "Acute Infection")]
DD_BANDS = [(0.5, "Normal"), (1.0, "Mild Elevation"), (2.0, "Moderate"), (float("inf"), "High Risk")]

CVD_CATEGORIES = [(0.25, "Low"), (0.50, "Moderate"), (0.75, "High"), (1.01, "Very High")]
CTR_BANDS = [(0.33, "Low"), (0.66, "Moderate"), (1.01, "High")]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class BiomarkerEngine:
    """Weighted-linear biomarker estimation from a 72-factor vector."""

    BIOMARKERS = ("TC", "HDL", "TG", "CRP", "DD")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _weighted_sum(factors: Mapping[str, float], biomarker: str,
                      collect: Optional[List[Contribution]] = None) -> float:
        total = 0.0
        for fid in FACTOR_IDS:
            spec: FactorSpec = FACTORS[fid]
            if biomarker not in spec.weights or fid not in factors:
                continue
            w, d = spec.weights[biomarker]
            v = float(factors[fid])
            contrib = v * w * d
            total += contrib
            if collect is not None:
                collect.append(Contribution(
                    fid=fid, name=spec.name, value=round(v, 3), weight=w,
                    contribution=contrib,
                    direction="risk-increasing" if d > 0 else "protective",
                ))
        return total

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute(factors: Mapping[str, float], day_index: int = 0) -> BiomarkerResult:
        result = BiomarkerResult(day_index=day_index, n_factors=len(factors))
        contribs: Dict[str, List[Contribution]] = {}

        # 1) direct weighted biomarkers --------------------------------- #
        values: Dict[str, float] = {}
        for b in ("TC", "TG", "CRP"):
            cs: List[Contribution] = []
            ws = BiomarkerEngine._weighted_sum(factors, b, collect=cs)
            cs.sort(key=lambda c: abs(c.contribution), reverse=True)
            contribs[b] = cs
            lo, hi = RANGES[b]
            values[b] = round(min(max(BASES[b] + ws * SCALING[b], lo), hi), 1)

        # HDL (protective formula: 80 − net risk × 0.45) ------------------ #
        cs = []
        ws = BiomarkerEngine._weighted_sum(factors, "HDL", collect=cs)
        cs.sort(key=lambda c: abs(c.contribution), reverse=True)
        contribs["HDL"] = cs
        lo, hi = RANGES["HDL"]
        values["HDL"] = round(min(max(BASES["HDL"] - ws * SCALING["HDL"], lo), hi), 1)

        # 2) cross-biomarker interactions (§6.11) -------------------------- #
        crp_norm = (values["CRP"] - 0.2) / (10.0 - 0.2)
        hdl_protect = 0.7 if values["HDL"] >= 60 else (1.3 if values["HDL"] < 40 else 1.0)
        crp_adj = round(min(max(values["CRP"] * hdl_protect, 0.2), 10.0), 2)

        # D-Dimer with CRP linkage ---------------------------------------- #
        cs = []
        ws = BiomarkerEngine._weighted_sum(factors, "DD", collect=cs)
        cs.sort(key=lambda c: abs(c.contribution), reverse=True)
        contribs["DD"] = cs
        lo, hi = RANGES["DD"]
        dd = BASES["DD"] + ws * SCALING["DD"] + crp_norm * 1.2
        values["DD"] = round(min(max(dd, lo), hi), 2)

        # 3) LDL from TC/HDL/TG (Friedewald / M-H per §6.3) ---------------- #
        divisor = 5.0 if values["TG"] < LDL_FRIEDEWALD_MAX_TG else 6.0
        values["LDL"] = round(max(values["TC"] - values["HDL"] - values["TG"] / divisor, 30.0), 1)

        result.values = values

        # TC context multiplier from CRP (§6.11) --------------------------- #
        crp_mult = 1.0 if values["CRP"] < 1 else (1.2 if values["CRP"] <= 3 else 1.5)
        tc_adj = round(min(values["TC"] * crp_mult, 320.0), 1)
        result.adjusted = {"TC_adj": tc_adj, "CRP_adj": crp_adj, "CRP_mult": crp_mult,
                           "HDL_protection": hdl_protect}

        # 4) derived values (§6.7) ----------------------------------------- #
        vldl = values["TG"] / 5.0
        result.derived = {
            "VLDL": round(vldl, 1),
            "Non_HDL": round(values["TC"] - values["HDL"], 1),
            "TC_HDL_ratio": round(values["TC"] / values["HDL"], 2),
            "LDL_HDL_ratio": round(values["LDL"] / values["HDL"], 2),
            "AIP": round(math.log10(values["TG"] / values["HDL"]), 3),
        }

        # 5) categories (§6.10) -------------------------------------------- #
        result.categories = {
            "TC": _categorize(values["TC"], TC_BANDS),
            "HDL": _categorize(values["HDL"], HDL_BANDS),
            "LDL": _categorize(values["LDL"], LDL_BANDS),
            "TG": _categorize(values["TG"], TG_BANDS),
            "CRP": _categorize(crp_adj, CRP_BANDS),
            "DD": _categorize(values["DD"], DD_BANDS),
        }

        # 6) risk scores (§6.11) -------------------------------------------- #
        def n(v, lo, hi):
            return min(max((v - lo) / (hi - lo), 0.0), 1.0)

        tc_n = n(values["TC"], 150, 320)
        ldl_n = n(values["LDL"], 30, 300)
        hdl_n = n(values["HDL"], 20, 80)          # high normalized HDL = protective
        tg_n = n(values["TG"], 80, 500)
        crp_n = crp_norm
        dd_n = n(values["DD"], 0.1, 3.0)
        result.cvd_score = round(0.20 * tc_n + 0.20 * ldl_n + 0.10 * (1 - hdl_n)
                                 + 0.10 * tg_n + 0.20 * crp_n + 0.20 * dd_n, 3)
        result.cvd_category = _categorize(result.cvd_score, CVD_CATEGORIES)
        result.ctr = round(0.35 * ldl_n + 0.40 * dd_n + 0.25 * crp_n, 3)
        result.ctr_category = _categorize(result.ctr, CTR_BANDS)

        # 7) pathway risks (§5.1) ------------------------------------------- #
        def mean_norm(fids) -> float:
            vals = [float(factors[f]) for f in fids if f in factors]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        result.pathway_risk = {
            "lipid": round((tc_n + ldl_n + (1 - hdl_n)) / 3, 3),
            "inflammation": round(crp_n, 3),
            "thrombosis": round(dd_n, 3),
            "hemodynamic": mean_norm(("F16", "F18", "F19")),
            "autonomic": round(1 - mean_norm(("F21", "F22", "F24")), 3),   # low HRV = risky
            "metabolic": mean_norm(("F50", "F51", "F53", "F55")),
        }
        result.contributions = contribs
        return result
