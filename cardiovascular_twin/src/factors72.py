"""
factors72.py
============
The **72-factor framework** (Chapter 5 of the project documentation).

Every factor F1..F72 is declared once in the FACTORS registry with:
    * id, name, source (sensor), raw range (min, max) used for 0-1 normalization
    * biomarker weights: {biomarker: (weight, direction)} — direction +1 =
      risk-increasing, -1 = protective
    * category (NIR, PPG, BP, HRV, SpO2, ECG, Temp, GSR, BioZ, Activity,
      Sleep/Stress, Demographic, History, Diet, Smoking, Derived)

``build_factor_vector`` derives all 72 raw values from one day of wearable
observations + the user profile. Where a sensor stream is not yet available in
this prototype (NIR, PPG morphology, ECG morphology), values are synthesized
deterministically from physiologically-linked channels (documented inline) so
the full pipeline runs end-to-end. Real sensor drivers can replace each
derivation independently.

Normalization (doc §6.1):  F_norm = (raw - min) / (max - min), clipped [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = ["FactorSpec", "FACTORS", "FACTOR_IDS", "build_factor_vector", "normalize"]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorSpec:
    fid: str                 # "F1".."F72"
    name: str
    source: str              # sensor / profile / derived
    category: str
    raw_min: float
    raw_max: float
    # biomarker -> (weight, direction); direction: +1 risk, -1 protective
    weights: Dict[str, Tuple[float, int]] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """Overall reading direction of the *normalized* value vs risk."""
        if not self.weights:
            return "n/a"
        return "risk-increasing" if all(d > 0 for _, d in self.weights.values()) \
            else ("protective" if all(d < 0 for _, d in self.weights.values()) else "mixed")

    def normalize(self, raw: float) -> float:
        return normalize(raw, self.raw_min, self.raw_max)


def normalize(raw: float, lo: float, hi: float) -> float:
    """Doc §6.1 normalization, clipped to [0, 1]."""
    if hi <= lo:
        return 0.0
    return float(min(max((raw - lo) / (hi - lo), 0.0), 1.0))


def _f(fid: str, name: str, source: str, category: str, lo: float, hi: float,
       weights: Optional[Dict[str, Tuple[float, int]]] = None) -> Tuple[str, FactorSpec]:
    return fid, FactorSpec(fid, name, source, category, float(lo), float(hi),
                           dict(weights or {}))


# ---- The 72 factors (Chapter 5.2 reference table) -------------------------- #
FACTORS: Dict[str, FactorSpec] = dict([
    # NIR spectroscopy (F1-F8)
    _f("F1", "NIR 1040nm absorption", "NIR sensor", "NIR", 0.1, 1.0, {"TC": (18, 1), "LDL": (10, 1)}),
    _f("F2", "NIR 1680nm absorption", "NIR sensor", "NIR", 0.1, 1.0, {"TG": (15, 1)}),
    _f("F3", "NIR 930/1200nm ratio", "NIR calculated", "NIR", 0.1, 1.0, {"TC": (8, 1)}),
    _f("F4", "NIR spectral slope", "NIR calculated", "NIR", 0.1, 1.0, {"TC": (8, 1)}),
    _f("F5", "NIR 2nd deriv. 1042nm", "NIR calculated", "NIR", 0.1, 1.0, {"LDL": (9, 1), "TC": (6, 1)}),
    _f("F6", "NIR 2nd deriv. 1726nm", "NIR calculated", "NIR", 0.1, 1.0, {"TG": (9, 1)}),
    _f("F7", "NIR PCA Score 1", "PCA of spectrum", "NIR", 0.1, 1.0, {"TC": (16, 1)}),
    _f("F8", "NIR PCA Score 2", "PCA of spectrum", "NIR", 0.1, 1.0, {"HDL": (5, -1)}),
    # PPG morphology (F9-F15)
    _f("F9", "Pulse Wave Velocity (PWV)", "PPG sensor", "PPG", 4, 15, {"LDL": (9, 1)}),
    _f("F10", "Augmentation Index (AIx)", "PPG waveform", "PPG", 0, 60, {"LDL": (7, 1)}),
    _f("F11", "Stiffness Index (SI)", "PPG waveform", "PPG", 4, 15, {"LDL": (7, 1)}),
    _f("F12", "Reflection Index (RI)", "PPG waveform", "PPG", 0, 100, {"TC": (6, 1)}),
    _f("F13", "SDPPG b/a ratio", "PPG 2nd deriv.", "PPG", -1, 1, {"LDL": (6, 1)}),
    _f("F14", "SDPPG Aging Index", "PPG 2nd deriv.", "PPG", 0, 1, {"TC": (6, 1), "LDL": (5, 1)}),
    _f("F15", "Systolic/Diastolic Time Ratio", "PPG timing", "PPG", 0.3, 2.0, {"TC": (5, 1)}),
    # Blood pressure (F16-F20)
    _f("F16", "Systolic Blood Pressure", "BP sensor", "BP", 90, 180, {"TC": (9, 1), "LDL": (6, 1)}),
    _f("F17", "Diastolic Blood Pressure", "BP sensor", "BP", 60, 110, {"TC": (7, 1)}),
    _f("F18", "Pulse Pressure", "Calculated", "BP", 20, 80, {"TC": (14, 1), "LDL": (8, 1)}),
    _f("F19", "Mean Arterial Pressure", "Calculated", "BP", 70, 120, {"TC": (7, 1)}),
    _f("F20", "BP Variability (SD)", "SD of readings", "BP", 0, 1, {"CRP": (6, 1), "DD": (6, 1)}),
    # HRV (F21-F25)
    _f("F21", "SDNN", "ECG RR intervals", "HRV", 10, 120, {"TC": (7, -1), "CRP": (7, -1)}),
    _f("F22", "RMSSD", "ECG RR intervals", "HRV", 5, 80, {"CRP": (8, -1), "HDL": (4, -1)}),
    _f("F23", "LF/HF Ratio", "Freq domain HRV", "HRV", 1.0, 5.0, {"TG": (8, 1)}),
    _f("F24", "pNN50", "ECG RR intervals", "HRV", 0, 50, {"TC": (5, -1), "CRP": (5, -1)}),
    _f("F25", "VLF Power", "Freq domain HRV", "HRV", 0, 1, {"TG": (6, 1)}),
    # SpO2 (F26-F29)
    _f("F26", "Resting SpO2", "SpO2 sensor", "SpO2", 88, 100, {"TC": (6, -1)}),
    _f("F27", "Activity SpO2", "SpO2 sensor", "SpO2", 88, 100, {"TC": (4, -1)}),
    _f("F28", "SpO2 Recovery Rate", "SpO2 post-exercise", "SpO2", 0, 1, {"TC": (4, -1)}),
    _f("F29", "SpO2 Variability (SD)", "SD of SpO2", "SpO2", 0, 1, {"CRP": (5, 1)}),
    # ECG morphology (F30-F33)
    _f("F30", "AF Probability Score", "ECG rhythm", "ECG", 0, 1, {"DD": (18, 1)}),
    _f("F31", "RR Interval Irregularity", "ECG analysis", "ECG", 0, 1, {"DD": (8, 1)}),
    _f("F32", "ST Segment Deviation", "ECG morphology", "ECG", 0, 1, {"CRP": (7, 1), "DD": (7, 1)}),
    _f("F33", "QTc Interval", "ECG analysis", "ECG", 350, 480, {"CRP": (5, 1), "DD": (5, 1)}),
    # Temperature (F34-F35)
    _f("F34", "Skin Temperature Elevation", "Temp sensor", "Temp", 36.0, 38.0, {"CRP": (8, 1)}),
    _f("F35", "Temperature Variability", "SD temperature", "Temp", 0, 1, {"CRP": (5, 1)}),
    # GSR (F36)
    _f("F36", "GSR / EDA Level", "GSR sensor", "GSR", 0, 1, {"CRP": (7, 1), "TC": (5, 1)}),
    # Bioimpedance (F37-F38)
    _f("F37", "Dehydration Index", "Bioimpedance", "BioZ", 0, 1, {"DD": (6, 1)}),
    _f("F38", "Body Fat Percentage", "Bioimpedance", "BioZ", 0, 1, {"CRP": (8, 1), "TC": (6, 1)}),
    # Activity (F39-F42)
    _f("F39", "Daily Activity Score", "IMU steps", "Activity", 0, 15000, {"HDL": (8, -1), "CRP": (6, -1)}),
    _f("F40", "Sedentary Hours/Day", "IMU accelerometer", "Activity", 2, 16, {"DD": (10, 1)}),
    _f("F41", "Exercise Intensity (MET)", "IMU + HR", "Activity", 0, 1, {"TG": (8, -1), "CRP": (5, -1), "HDL": (6, -1)}),
    _f("F42", "Post-Exercise HR Recovery", "HR after activity", "Activity", 0, 1, {"TC": (5, -1)}),
    # Sleep / stress (F43-F47)
    _f("F43", "Sleep Duration Estimate", "HR + IMU", "Sleep/Stress", 4, 10, {"TG": (5, -1), "HDL": (3, -1)}),
    _f("F44", "Sleep Regularity Score", "HR + IMU patterns", "Sleep/Stress", 0, 1, {"TG": (5, -1)}),
    _f("F45", "Nocturnal SpO2 Pattern", "SpO2 during sleep", "Sleep/Stress", 0, 1, {"CRP": (6, 1), "DD": (6, 1)}),
    _f("F46", "Composite Stress Score", "HRV+HR+GSR+IMU", "Sleep/Stress", 0, 1, {"CRP": (8, 1), "TC": (6, 1)}),
    _f("F47", "Cortisol Proxy Index", "Nocturnal HRV", "Sleep/Stress", 0, 1, {"CRP": (7, 1), "TC": (5, 1)}),
    # Demographics & history (F48-F60)
    _f("F48", "Age", "User profile", "Demographic", 18, 80, {"DD": (13, 1), "TC": (12, 1)}),
    _f("F49", "Sex (M=1/F=0)", "User profile", "Demographic", 0, 1, {"HDL": (10, 1)}),
    _f("F50", "BMI", "User input", "Demographic", 15, 45, {"CRP": (12, 1), "TG": (12, 1)}),
    _f("F51", "Waist Circumference", "User input", "Demographic", 60, 130, {"CRP": (10, 1), "TG": (8, 1)}),
    _f("F52", "Ethnicity Risk Score", "User profile", "Demographic", 0, 1, {"TC": (6, 1), "LDL": (5, 1)}),
    _f("F53", "Waist-Hip Ratio", "Calculated", "Demographic", 0.7, 1.1, {"TG": (7, 1), "HDL": (9, 1)}),
    _f("F54", "Height", "User profile", "Demographic", 0, 1, {"TC": (3, -1)}),
    _f("F55", "Diabetes / Pre-diabetes", "User history", "History", 0, 1, {"CRP": (10, 1), "DD": (10, 1)}),
    _f("F56", "Known Cholesterol History", "User input", "History", 0, 1, {"TC": (14, 1)}),
    _f("F57", "Hypertension History", "User history", "History", 0, 1, {"CRP": (8, 1)}),
    _f("F58", "Previous Clot History", "User history", "History", 0, 1, {"DD": (15, 1)}),
    _f("F59", "Thyroid Disorder", "User history", "History", 0, 1, {"LDL": (6, 1), "TC": (6, 1)}),
    _f("F60", "Family History Heart Disease", "User history", "History", 0, 1, {"TC": (8, 1), "CRP": (6, 1)}),
    # Diet & smoking (F61-F67)
    _f("F61", "Saturated Fat Intake", "Diet questionnaire", "Diet", 0, 1, {"LDL": (10, 1), "TC": (8, 1)}),
    _f("F62", "Sugar / Refined Carbs", "Diet questionnaire", "Diet", 0, 1, {"TG": (10, 1)}),
    _f("F63", "Vegetable / Fruit Intake", "Diet questionnaire", "Diet", 0, 1, {"LDL": (6, -1)}),
    _f("F64", "Alcohol Consumption", "Diet questionnaire", "Diet", 0, 1, {"TG": (6, 1)}),
    _f("F65", "Omega-3 / Healthy Fat Intake", "Diet questionnaire", "Diet", 0, 1, {"CRP": (6, -1), "TG": (5, -1)}),
    _f("F66", "Smoking Intensity Score", "User questionnaire", "Smoking", 0, 1, {"HDL": (10, 1), "CRP": (10, 1)}),
    _f("F67", "Pack-Year History", "User questionnaire", "Smoking", 0, 1, {"CRP": (7, 1), "DD": (7, 1)}),
    # Derived vascular indices (F68-F72)
    _f("F68", "Vascular Age Gap", "PPG+BP+age calc", "Derived", 0, 1, {"TC": (7, 1), "LDL": (6, 1), "DD": (6, 1)}),
    _f("F69", "Arterial Compliance Index", "PPG+BP calc", "Derived", 0, 1, {"TC": (6, -1), "HDL": (3, -1)}),
    _f("F70", "Peripheral Perfusion Index", "PPG amplitude", "Derived", 0, 1, {"DD": (6, -1)}),
    _f("F71", "Estimated Atherogenic Index", "Derived", "Derived", 0, 1, {"LDL": (8, 1)}),
    _f("F72", "Endothelial Dysfunction Index", "Post-exercise PPG", "Derived", 0, 1, {"CRP": (10, 1), "DD": (10, 1)}),
])

FACTOR_IDS: List[str] = sorted(FACTORS.keys(), key=lambda s: int(s[1:]))

assert len(FACTORS) == 72, f"expected 72 factors, got {len(FACTORS)}"


# --------------------------------------------------------------------------- #
# Derivation of the daily 72-factor vector
# --------------------------------------------------------------------------- #
def _bmi(profile: Mapping) -> float:
    h, w = profile.get("height_cm"), profile.get("weight_kg")
    if not h or not w:
        return 25.0
    return float(w) / (float(h) / 100.0) ** 2


def _gauss(day_index: int, salt: int) -> float:
    """Deterministic standard normal per (day, salt)."""
    rng = np.random.default_rng((day_index + 1) * 9973 + salt)
    return float(rng.normal())


def build_factor_vector(obs: Mapping, profile: Mapping, day_index: int = 0,
                        bp_sd: float = 0.0, sleep_sd: float = 0.0) -> Dict[str, float]:
    """Derive all 72 *normalized* factors (0-1) for one day.

    ``obs``     – daily wearable aggregates (same keys as the simulator)
    ``profile`` – user profile dict (UserProfile.to_dict())
    ``bp_sd``   – rolling SD of systolic BP (state variability), if tracked
    ``sleep_sd``– rolling SD of sleep hours
    Returns {F1: 0.42, ...} normalized per doc §6.1.
    """
    g = lambda salt, mu=0.0, sd=1.0: mu + sd * _gauss(day_index, salt)  # noqa: E731
    age = float(profile.get("age", 45))
    sex_m = 1.0 if str(profile.get("sex", "male")).lower().startswith("m") else 0.0
    bmi = _bmi(profile)
    smoker = 1.0 if profile.get("smoker") else 0.0
    diabetic = 1.0 if profile.get("diabetic") else 0.0

    sbp = float(obs.get("systolic_bp", 120))
    dbp = float(obs.get("diastolic_bp", 80))
    rhr = float(obs.get("resting_heart_rate", 65))
    hrv = float(obs.get("hrv_rmssd", 40))
    spo2 = float(obs.get("nocturnal_spo2_pct", 96))
    dips = float(obs.get("spo2_dips", 1))
    sleep = float(obs.get("sleep_hours", 7))
    steps = float(obs.get("steps", 7000))
    active = float(obs.get("active_minutes", 30))
    deep = float(obs.get("deep_sleep_pct", 16))
    eff = float(obs.get("sleep_efficiency", 0.88))

    # ---- underlying lipid propensity (drives synthetic NIR + vascular) ---- #
    # anchored on profile lipid baselines (static, mg/dL) blended with a
    # *dynamic* physiological-deviation index: NIR measures tissue lipid
    # content, which in this model tracks the current cardiovascular
    # trajectory (e.g. a declining scenario -> rising NIR lipid signal).
    tc_prop_static = normalize(float(profile.get("baseline_total_cholesterol", 190)), 140, 280)
    hdl_prop_static = normalize(float(profile.get("baseline_hdl", 50)), 25, 85)
    tg_prop_static = normalize(float(profile.get("baseline_triglycerides", 130)), 60, 350)

    base_rhr = float(profile.get("baseline_resting_hr", rhr))
    base_hrv = float(profile.get("baseline_hrv_rmssd", hrv))
    base_sbp = float(profile.get("baseline_systolic", sbp))
    dyn = float(np.clip(
        0.5
        + 0.30 * (rhr - base_rhr) / 18.0
        + 0.30 * (base_hrv - hrv) / 20.0
        + 0.25 * (sbp - base_sbp) / 25.0
        + 0.10 * (7.3 - sleep) / 2.0
        + 0.05 * (8500.0 - steps) / 5000.0
        + g(50, 0, 0.04), 0.0, 1.0))
    tc_prop = float(np.clip(0.6 * tc_prop_static + 0.4 * dyn, 0, 1))
    tg_prop = float(np.clip(0.6 * tg_prop_static + 0.4 * dyn, 0, 1))
    hdl_prop = float(np.clip(0.6 * hdl_prop_static + 0.4 * (1.0 - dyn), 0, 1))

    pp = sbp - dbp                          # pulse pressure
    map_ = dbp + pp / 3.0                   # mean arterial pressure
    stress = np.clip(0.45 + (70 - rhr) / 90 + (45 - hrv) / 120 + g(1, 0, 0.08), 0.0, 1.0)
    fitness = np.clip(0.3 + hrv / 160 + steps / 30000 - bmi / 60, 0.0, 1.0)

    raw: Dict[str, float] = {
        # NIR (synthesized from lipid propensity + noise) ------------------ #
        "F1": 0.1 + 0.9 * np.clip(tc_prop + g(11, 0, 0.10), 0, 1),
        "F2": 0.1 + 0.9 * np.clip(tg_prop + g(12, 0, 0.10), 0, 1),
        "F3": 0.1 + 0.9 * np.clip(0.35 + 0.5 * (tc_prop - hdl_prop) + g(13, 0, 0.08), 0, 1),
        "F4": 0.1 + 0.9 * np.clip(0.3 + 0.6 * tc_prop - 0.3 * fitness + g(14, 0, 0.08), 0, 1),
        "F5": 0.1 + 0.9 * np.clip(0.7 * tc_prop + 0.2 * tg_prop + g(15, 0, 0.09), 0, 1),
        "F6": 0.1 + 0.9 * np.clip(tg_prop + g(16, 0, 0.09), 0, 1),
        "F7": 0.1 + 0.9 * np.clip(0.8 * tc_prop + 0.2 * (1 - fitness) + g(17, 0, 0.07), 0, 1),
        "F8": 0.1 + 0.9 * np.clip(hdl_prop + g(18, 0, 0.09), 0, 1),
        # PPG morphology (from BP/age/fitness) ----------------------------- #
        "F9": 4 + 11 * np.clip(0.45 + (age - 40) / 80 + (sbp - 115) / 90 - 0.3 * fitness + g(19, 0, 0.07), 0, 1),
        "F10": 60 * np.clip(0.35 + (age - 40) / 90 + (sbp - 120) / 80 + g(20, 0, 0.08), 0, 1),
        "F11": 4 + 11 * np.clip(0.5 + (age - 40) / 75 - 0.35 * fitness + g(21, 0, 0.07), 0, 1),
        "F12": 100 * np.clip(0.4 + 0.6 * (1 - fitness) + 0.3 * (sbp - 120) / 80 + g(22, 0, 0.08), 0, 1),
        "F13": -1 + 2 * np.clip(0.5 + (age - 45) / 90 + g(23, 0, 0.07), 0, 1),
        "F14": np.clip(0.4 + (age - 40) / 80 + 0.2 * (sbp - 120) / 80 + g(24, 0, 0.07), 0, 1),
        "F15": 0.3 + 1.7 * np.clip(0.5 + (sbp - 120) / 90 - 0.25 * fitness + g(25, 0, 0.07), 0, 1),
        # BP ----------------------------------------------------------------- #
        "F16": sbp,
        "F17": dbp,
        "F18": pp,
        "F19": map_,
        "F20": min(bp_sd / 10.0, 1.0) if bp_sd else float(np.clip(0.45 + abs(g(26, 0, 0.04)), 0, 1)),
        # HRV ---------------------------------------------------------------- #
        "F21": 1.35 * hrv + 6,                       # SDNN ≈ 1.35 × RMSSD
        "F22": hrv,
        "F23": np.clip(5.0 - 3.2 * (hrv / 70) - 0.4 * fitness + g(27, 0, 0.25), 1.0, 5.0),
        "F24": np.clip(50 * (hrv / 75) ** 2 + g(28, 0, 3), 0, 50),
        "F25": np.clip(0.55 - 0.5 * fitness + 0.4 * stress + g(29, 0, 0.06), 0, 1),
        # SpO2 --------------------------------------------------------------- #
        "F26": spo2,
        "F27": spo2 - 1.0 - 2.0 * min(active, 60) / 60,
        "F28": np.clip(0.35 + 0.5 * fitness + g(30, 0, 0.07), 0, 1),
        "F29": float(np.clip(0.10 + dips / 25 + abs(g(31, 0, 0.05)), 0, 1)),
        # ECG ---------------------------------------------------------------- #
        "F30": float(np.clip(0.05 + 0.55 * diabetic + (age - 60) / 150 + max(0, rhr - 85) / 40 + g(32, 0, 0.04), 0, 1)),
        "F31": float(np.clip(0.15 + max(0, 40 - hrv) / 60 + (age - 50) / 120 + g(33, 0, 0.05), 0, 1)),
        "F32": float(np.clip(0.08 + 0.4 * stress + 0.3 * diabetic + g(34, 0, 0.04), 0, 1)),
        "F33": 350 + 130 * np.clip(0.45 + (age - 45) / 90 + 0.15 * stress + g(35, 0, 0.06), 0, 1),
        # Temperature -------------------------------------------------------- #
        "F34": 36.0 + 2.0 * np.clip(0.18 + 0.55 * stress + 0.25 * tg_prop + g(36, 0, 0.07), 0, 1),
        "F35": float(np.clip(0.12 + 0.5 * stress + abs(g(37, 0, 0.05)), 0, 1)),
        # GSR ---------------------------------------------------------------- #
        "F36": float(np.clip(stress + g(38, 0, 0.07), 0, 1)),
        # Bioimpedance ------------------------------------------------------- #
        "F37": float(np.clip(0.25 + 0.4 * stress - 0.2 * fitness + g(39, 0, 0.07), 0, 1)),
        "F38": float(np.clip((bmi - 15) / 30 + g(40, 0, 0.05), 0, 1)),
        # Activity ----------------------------------------------------------- #
        "F39": steps,
        "F40": np.clip(16 - 14 * min(steps, 15000) / 15000, 2, 16),
        "F41": float(np.clip(active / 90 + g(41, 0, 0.05), 0, 1)),
        "F42": np.clip(0.3 + 0.6 * fitness + g(42, 0, 0.07), 0, 1),
        # Sleep / stress ----------------------------------------------------- #
        "F43": sleep,
        "F44": float(np.clip(1.0 - min(sleep_sd, 3) / 3 * 0.8 - abs(g(43, 0, 0.05)), 0, 1)) if sleep_sd \
            else float(np.clip(0.80 + g(43, 0, 0.05), 0, 1)),
        "F45": float(np.clip(dips / 12 + (97 - spo2) / 6, 0, 1)),
        "F46": float(stress),
        "F47": float(np.clip(0.5 * stress + 0.3 * (1 - fitness) + 0.2 * max(0, 7.5 - sleep) / 3 + g(44, 0, 0.05), 0, 1)),
        # Demographics & history --------------------------------------------- #
        "F48": age,
        "F49": sex_m,
        "F50": bmi,
        "F51": float(profile.get("waist_cm") or (75 + 0.9 * (bmi - 22) * 2.4 + 4 * sex_m)),
        "F52": float(profile.get("ethnicity_risk", 0.3)),
        "F53": float(profile.get("waist_hip_ratio") or np.clip(0.78 + 0.008 * (bmi - 22) + 0.04 * sex_m, 0.7, 1.1)),
        "F54": float(profile.get("height_cm", 170)) / 210,
        "F55": diabetic,
        "F56": float(profile.get("chol_history", 0.2)),
        "F57": 0.7 if sbp >= 140 or dbp >= 90 else float(profile.get("htn_history", 0.1)),
        "F58": float(profile.get("clot_history", 0.0)),
        "F59": float(profile.get("thyroid", 0.05)),
        "F60": 1.0 if profile.get("family_history") else 0.0,
        # Diet & smoking ----------------------------------------------------- #
        "F61": float(profile.get("sat_fat", 0.3)),
        "F62": float(profile.get("sugar", 0.3)),
        "F63": float(profile.get("vegetables", 0.6)),
        "F64": float(profile.get("alcohol", 0.2)),
        "F65": float(profile.get("omega3", 0.5)),
        "F66": smoker * float(profile.get("smoking_intensity", 0.6)) if smoker else 0.0,
        "F67": float(profile.get("pack_years", 12.0)) / 50 * smoker,
        # Derived vascular --------------------------------------------------- #
        "F68": float(np.clip((age - 40) / 50 * 0.5 + (sbp - 115) / 70 * 0.4 - 0.3 * fitness + g(45, 0, 0.06), 0, 1)),
        "F69": float(np.clip(0.6 - 0.5 * ((sbp - 110) / 80) - 0.2 * ((age - 40) / 60) + 0.2 * fitness + g(46, 0, 0.06), 0, 1)),
        "F70": float(np.clip(0.5 + 0.4 * fitness - 0.3 * stress + g(47, 0, 0.06), 0, 1)),
        "F71": float(np.clip(0.5 * tc_prop + 0.3 * (1 - hdl_prop) + 0.2 * ((sbp - 110) / 80) + g(48, 0, 0.06), 0, 1)),
        "F72": float(np.clip(0.3 + 0.4 * stress + 0.2 * diabetic + 0.2 * (1 - fitness) + g(49, 0, 0.05), 0, 1)),
    }

    # normalize everything through the registry
    return {fid: round(FACTORS[fid].normalize(raw[fid]), 4) for fid in FACTOR_IDS}
