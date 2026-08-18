"""
data_simulator.py
=================
Synthetic multimodal wearable data source for the CardioCore digital twin.

Design goals
------------
1. Produce *daily aggregated* physiological observations similar to what
   fitness-band / smartwatch APIs expose (resting HR, HRV, BP estimate,
   nocturnal SpO2, sleep architecture, activity).
2. Be **pluggable**: everything implements/uses the ``WearableDataSource``
   interface, so a real data source (CSV export, cloud API) can be dropped
   in later without touching the digital twin.
3. Support longitudinal *scenarios* ("stable", "improving", "declining")
   so risk-assessment logic can be exercised end-to-end.

Model: first-order mean-reverting random walk (Ornstein-Uhlenbeck style)
around a drifting baseline, with weekly rhythm (busier weekends -> slightly
worse sleep, higher HR) and bounded Gaussian noise.

This simulator is for research/education – values are synthetic, NOT clinical.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "UserProfile",
    "PRESET_PROFILES",
    "WearableDataSource",
    "SimulatedWearableSource",
    "simulate_history",
]


# --------------------------------------------------------------------------- #
# User profile
# --------------------------------------------------------------------------- #
@dataclass
class UserProfile:
    """Static demographic + clinical attributes of the twin's owner."""

    name: str = "User"
    age: int = 42
    sex: str = "male"                      # "male" | "female"
    height_cm: float = 172.0
    weight_kg: float = 74.0
    smoker: bool = False
    diabetic: bool = False
    family_history: bool = False           # premature CVD in 1st-degree relatives

    # physiological baselines (used as simulator anchors)
    baseline_resting_hr: float = 62.0      # bpm
    baseline_hrv_rmssd: float = 55.0       # ms
    baseline_systolic: float = 118.0       # mmHg
    baseline_diastolic: float = 76.0       # mmHg

    # lipid baselines (mg/dL)
    baseline_total_cholesterol: float = 185.0
    baseline_hdl: float = 55.0
    baseline_triglycerides: float = 120.0

    @property
    def bmi(self) -> float:
        return round(self.weight_kg / (self.height_cm / 100.0) ** 2, 1)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["bmi"] = self.bmi
        return d


PRESET_PROFILES: Dict[str, UserProfile] = {
    "healthy": UserProfile(
        name="Healthy User", age=38, smoker=False, diabetic=False, family_history=False,
        baseline_resting_hr=58, baseline_hrv_rmssd=65, baseline_systolic=114,
        baseline_diastolic=73, baseline_total_cholesterol=175, baseline_hdl=62,
        baseline_triglycerides=95,
    ),
    "typical": UserProfile(
        name="Typical User", age=45, smoker=False, diabetic=False, family_history=False,
        baseline_resting_hr=64, baseline_hrv_rmssd=48, baseline_systolic=122,
        baseline_diastolic=79, baseline_total_cholesterol=195, baseline_hdl=50,
        baseline_triglycerides=140,
    ),
    "at_risk": UserProfile(
        name="At-Risk User", age=56, smoker=True, diabetic=True, family_history=True,
        height_cm=170, weight_kg=92,
        baseline_resting_hr=74, baseline_hrv_rmssd=30, baseline_systolic=138,
        baseline_diastolic=88, baseline_total_cholesterol=235, baseline_hdl=37,
        baseline_triglycerides=210,
    ),
}


# --------------------------------------------------------------------------- #
# Source interface (real sources implement this later)
# --------------------------------------------------------------------------- #
class WearableDataSource(abc.ABC):
    """Interface every data source must satisfy.

    ``next_day()`` returns a dict of *daily aggregated* metrics plus the keys
    ``day_index`` (int) and, when a lab draw happened that day,
    ``labs`` (dict with total_cholesterol / hdl / triglycerides).
    """

    @abc.abstractmethod
    def next_day(self) -> Dict: ...

    def next_days(self, n: int) -> List[Dict]:
        return [self.next_day() for _ in range(n)]


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
# per-day multiplicative drift applied to baselines under each scenario
_SCENARIO_DRIFT = {
    #                 RHR      HRV      SBP      DBP     sleep   steps
    "stable":     (0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000),
    "improving":  (-0.0018,  0.0032, -0.0014, -0.0011,  0.0016,  0.0024),
    "declining":  (0.0022, -0.0036,  0.0017,  0.0013, -0.0019, -0.0025),
}


@dataclass
class _Channel:
    """One simulated physiological channel (OU process)."""

    value: float
    baseline: float
    theta: float          # mean-reversion strength (per day)
    sigma: float          # noise scale (absolute units)
    lo: float = -math.inf
    hi: float = math.inf

    def step(self, target: float, rng: np.random.Generator) -> float:
        noise = rng.normal(0.0, self.sigma)
        self.value += self.theta * (target - self.value) + noise
        self.value = min(max(self.value, self.lo), self.hi)
        return self.value


class SimulatedWearableSource(WearableDataSource):
    """Simulates daily wearable observations for one user.

    Parameters
    ----------
    profile   : UserProfile (see PRESET_PROFILES)
    scenario  : "stable" | "improving" | "declining" – long-run trajectory
    seed      : RNG seed for reproducibility
    lab_interval_days : days between lipid lab draws (None disables labs)
    """

    def __init__(
        self,
        profile: Optional[UserProfile] = None,
        scenario: str = "stable",
        seed: Optional[int] = None,
        start_date: Optional[date] = None,
        lab_interval_days: Optional[int] = 90,
    ) -> None:
        if scenario not in _SCENARIO_DRIFT:
            raise ValueError(f"scenario must be one of {list(_SCENARIO_DRIFT)}, got {scenario!r}")
        self.profile = profile or PRESET_PROFILES["typical"]
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.start_date = start_date or date.today()
        self.day_index = 0
        self.lab_interval_days = lab_interval_days

        p = self.profile
        self._channels: Dict[str, _Channel] = {
            "resting_heart_rate": _Channel(p.baseline_resting_hr, p.baseline_resting_hr, 0.25, 1.6, lo=35, hi=130),
            "hrv_rmssd":          _Channel(p.baseline_hrv_rmssd,  p.baseline_hrv_rmssd,  0.25, 4.0, lo=8,  hi=180),
            "systolic_bp":        _Channel(p.baseline_systolic,   p.baseline_systolic,   0.25, 4.5, lo=80, hi=220),
            "diastolic_bp":       _Channel(p.baseline_diastolic,  p.baseline_diastolic,  0.25, 3.2, lo=50, hi=140),
            "sleep_hours":        _Channel(7.2,  7.2,  0.35, 0.75, lo=0.0, hi=11.0),
            "deep_sleep_pct":     _Channel(18.0, 18.0, 0.30, 2.2, lo=2.0, hi=40.0),
            "steps":              _Channel(8500.0, 8500.0, 0.35, 1900.0, lo=0, hi=40000),
            "active_minutes":     _Channel(42.0, 42.0, 0.35, 12.0, lo=0, hi=300),
            "nocturnal_spo2_pct": _Channel(96.4, 96.4, 0.35, 0.55, lo=88.0, hi=99.5),
        }

    # ------------------------------------------------------------------ #
    def _day_targets(self) -> Dict[str, float]:
        """Baseline + scenario drift + weekend effect -> today's channel targets."""
        d_rhr, d_hrv, d_sbp, d_dbp, d_sleep, d_steps = _SCENARIO_DRIFT[self.scenario]
        t = self.day_index
        weekend = self.start_date.weekday() >= 5  # Sat/Sun

        def wk(scale: float) -> float:
            return scale if weekend else 0.0

        p = self.profile
        return {
            "resting_heart_rate": p.baseline_resting_hr * (1 + d_rhr * t) + wk(2.0),
            "hrv_rmssd": p.baseline_hrv_rmssd * (1 + d_hrv * t) - wk(3.0),
            "systolic_bp": p.baseline_systolic * (1 + d_sbp * t) + wk(1.5),
            "diastolic_bp": p.baseline_diastolic * (1 + d_dbp * t) + wk(1.0),
            "sleep_hours": 7.2 * (1 + d_sleep * t) - wk(0.5),
            "deep_sleep_pct": 18.0 * (1 + 0.6 * d_sleep * t) - wk(0.8),
            "steps": 8500.0 * (1 + d_steps * t) + wk(-800.0),
            "active_minutes": 42.0 * (1 + d_steps * t) + wk(-4.0),
            "nocturnal_spo2_pct": 96.4 - (0.8 if self.scenario == "declining" else 0.0) * min(t / 60.0, 1.0),
        }

    # ------------------------------------------------------------------ #
    def _maybe_labs(self) -> Optional[Dict[str, float]]:
        if self.lab_interval_days is None:
            return None
        if self.day_index == 0 or (self.day_index % self.lab_interval_days == 0):
            p = self.profile
            drift = 1.0 + (0.0008 * self.day_index if self.scenario == "declining" else 0.0)
            return {
                "total_cholesterol": round(max(p.baseline_total_cholesterol * drift + self.rng.normal(0, 7), 80), 0),
                "hdl": round(max(p.baseline_hdl + self.rng.normal(0, 3), 20), 0),
                "triglycerides": round(max(p.baseline_triglycerides * drift + self.rng.normal(0, 12), 40), 0),
            }
        return None

    # ------------------------------------------------------------------ #
    def next_day(self) -> Dict:
        targets = self._day_targets()
        obs: Dict = {
            "day_index": self.day_index,
            "date": (self.start_date + timedelta(days=self.day_index)).isoformat(),
        }
        for key, ch in self._channels.items():
            obs[key] = round(ch.step(targets[key], self.rng), 2)
        obs["spo2_dips"] = int(self.rng.poisson(max(0.0, (97.5 - obs["nocturnal_spo2_pct"]) * 0.8)))
        obs["sleep_efficiency"] = round(
            min(max(0.98 - max(0.0, 7.5 - obs["sleep_hours"]) * 0.07 + self.rng.normal(0, 0.02), 0.50), 1.0), 3,
        )
        labs = self._maybe_labs()
        if labs is not None:
            obs["labs"] = labs
        self.day_index += 1
        return obs


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #
def simulate_history(
    n_days: int,
    profile: Optional[UserProfile] = None,
    scenario: str = "stable",
    seed: Optional[int] = None,
    lab_interval_days: Optional[int] = 90,
) -> pd.DataFrame:
    """Convenience: simulate ``n_days`` and return a tidy DataFrame.

    Lab panels (drawn every ``lab_interval_days``) are forward-filled into
    columns ``total_cholesterol`` / ``hdl`` / ``triglycerides``.
    """
    src = SimulatedWearableSource(profile, scenario, seed=seed, lab_interval_days=lab_interval_days)
    rows = src.next_days(n_days)
    df = pd.DataFrame(rows)
    if "labs" in df.columns:
        labs = pd.json_normalize(df["labs"].apply(lambda d: d if isinstance(d, dict) else {}))
        for col in ("total_cholesterol", "hdl", "triglycerides"):
            if col in labs.columns:
                df[col] = labs[col].replace("", np.nan).astype(float).ffill()
        df = df.drop(columns=["labs"])
    return df
