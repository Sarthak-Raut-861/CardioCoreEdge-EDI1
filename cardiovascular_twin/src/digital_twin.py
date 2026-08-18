"""
digital_twin.py
===============
The personalized **cardiovascular digital twin**.

Responsibilities
----------------
1. **State assimilation** – maintain an exponentially-weighted moving average
   (EWMA) of each wearable channel, so noisy daily readings are smoothed into
   a stable "current physiology" estimate (loosely analogous to Kalman update).
2. **Risk assessment** – blend demographic prior (static risk) with the
   current multimodal factor composite (via ``FactorEngine``) into a 0..1
   risk score + category (Low / Moderate / High / Very High).
3. **Trajectory** – least-squares slope over the trailing risk history →
   improving / stable / declining.
4. **Alerts** – rule-based safety-net checks (hypertensive crisis, sustained
   RHR elevation, SpO2 desaturation, HRV collapse, risk-threshold crossings).
5. **Persistence** – ``snapshot()`` / ``from_dict`` round-trip.

DISCLAIMER: research/educational prototype. NOT a medical device and must
never be used for diagnosis or clinical decision-making.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .adaptive_engine import AdaptiveWeightLearner, ConfidenceModel, PersonalBaselineTracker
from .factor_engine import CompositeResult, Factor, FactorEngine
from .lipid_calculator import LipidCalculator

__all__ = ["DigitalTwin", "TwinUpdate", "RISK_CATEGORIES", "DISCLAIMER"]


DISCLAIMER = (
    "Research/educational prototype. Not a medical device; output must not be "
    "used for diagnosis or clinical decisions. Consult a physician for medical advice."
)

RISK_CATEGORIES: List[Tuple[float, str]] = [
    (0.25, "Low"),
    (0.45, "Moderate"),
    (0.65, "High"),
    (1.01, "Very High"),
]

# wearable channels the twin tracks in state
STATE_KEYS = (
    "resting_heart_rate", "hrv_rmssd", "systolic_bp", "diastolic_bp",
    "nocturnal_spo2_pct", "spo2_dips", "sleep_hours", "deep_sleep_pct",
    "sleep_efficiency", "steps", "active_minutes",
)


def _risk_category(score: float) -> str:
    for upper, label in RISK_CATEGORIES:
        if score < upper:
            return label
    return RISK_CATEGORIES[-1][1]


@dataclass
class TwinUpdate:
    """Result of one assimilation step."""

    day_index: int
    date: Optional[str]
    risk_score: float            # 0..1
    category: str
    composite: Optional[CompositeResult] = None
    alerts: List[Dict[str, str]] = field(default_factory=list)
    trend: Optional[str] = None
    ci95: Optional[Tuple[float, float]] = None
    multipliers: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        return {
            "day_index": self.day_index,
            "date": self.date,
            "risk_score": self.risk_score,
            "category": self.category,
            "alerts": self.alerts,
            "trend": self.trend,
            "ci95": self.ci95,
            "multipliers": self.multipliers,
            "top_factors": [f.to_dict() for f in (self.composite.top(5) if self.composite else [])],
        }


class DigitalTwin:
    """Per-user cardiovascular digital twin.

    Parameters
    ----------
    profile       : UserProfile or mapping with age/sex/smoker/diabetic/
                    family_history/height_cm/weight_kg (+ optional baselines)
    learning_rate : EWMA gain for state assimilation (higher = reacts faster,
                    lower = smoother). 0.15 ≈ ~4-5 day effective memory.
    history_len   : max days of risk history retained for trend analysis.
    """

    PRIOR_WEIGHT = 0.25   # demographic share of final score
    CURRENT_WEIGHT = 0.75

    def __init__(
        self,
        profile: Mapping,
        learning_rate: float = 0.15,
        history_len: int = 90,
        adaptive: bool = True,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.profile: Dict[str, Any] = dict(profile)
        self.learning_rate = learning_rate
        self.history_len = history_len

        self.state: Dict[str, Optional[float]] = {k: None for k in STATE_KEYS}
        self.labs: Optional[Dict[str, float]] = None
        self.days_seen = 0
        self.risk_history: Deque[Tuple[int, float]] = deque(maxlen=history_len)
        self.alerts_raised: List[Dict[str, str]] = []

        # --- self-learning layer ---------------------------------------- #
        self.adaptive = adaptive
        self.baselines = PersonalBaselineTracker()
        self.weight_learner = AdaptiveWeightLearner()
        self.confidence = ConfidenceModel()

        # optional ML model hook (see explainability.RiskExplainer)
        self.model = None
        self.model_features: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Assimilation
    # ------------------------------------------------------------------ #
    def _assimilate_state(self, obs: Mapping[str, float]) -> None:
        alpha = self.learning_rate
        for key in STATE_KEYS:
            if key not in obs or obs[key] is None or (isinstance(obs[key], float) and math.isnan(obs[key])):
                continue
            new = float(obs[key])
            old = self.state[key]
            self.state[key] = new if old is None else round(old + alpha * (new - old), 3)

    def assimilate(self, obs: Mapping[str, float], labs: Optional[Mapping[str, float]] = None) -> TwinUpdate:
        """Ingest one day of aggregated observations; returns the new twin update."""
        zscores = self.baselines.update(obs) if self.adaptive else {}
        self._assimilate_state(obs)
        if labs:
            self.labs = {k: float(v) for k, v in labs.items()}
        self.days_seen += 1

        composite = self._compute_risk()
        self.risk_history.append((self.days_seen - 1, composite.score))

        multipliers_used: Optional[Dict[str, float]] = None
        ci: Optional[Tuple[float, float]] = None
        if self.adaptive:
            factor_scores = {f.key: f.score for f in composite.breakdown if f.value is not None}
            multipliers_used = self.weight_learner.observe(factor_scores, composite.score)
            ci = self.confidence.update(composite.score)

        alerts = self._evaluate_alerts(obs)
        # personal-baseline anomaly detection (learned norms, not population)
        if self.adaptive:
            warm_anoms = [k for k in self.baselines.anomalies(zscores) if self.baselines.is_warm(k)]
            if warm_anoms:
                worst = max(warm_anoms, key=lambda k: abs(zscores[k]))
                alerts.append({
                    "level": "info",
                    "code": "anomaly_detected",
                    "message": (f"Personal-baseline anomaly: {worst} at {self.state.get(worst)} "
                                f"({zscores[worst]:+.1f} sigma vs your learned normal)."),
                    "day": str(obs.get("date", obs.get("day_index", ""))),
                })
        self.alerts_raised.extend(alerts)
        trend = self.risk_trend()

        return TwinUpdate(
            day_index=self.days_seen - 1,
            date=obs.get("date"),
            risk_score=composite.score,
            category=_risk_category(composite.score),
            composite=composite,
            alerts=alerts,
            trend=trend,
            ci95=ci,
            multipliers=multipliers_used,
        )

    # ------------------------------------------------------------------ #
    # Risk
    # ------------------------------------------------------------------ #
    def _compute_risk(self) -> CompositeResult:
        demo = FactorEngine.evaluate_demographics(self.profile)
        prior = FactorEngine.composite(demo)
        wear = FactorEngine.evaluate_wearables(self.state)
        lab = FactorEngine.evaluate_labs(self.labs)
        multipliers = self.weight_learner.active_multipliers() if self.adaptive else None
        current = FactorEngine.composite(wear + lab, multipliers=multipliers)

        if current.n_factors == 0:
            blended = prior.score
        else:
            blended = self.PRIOR_WEIGHT * prior.score + self.CURRENT_WEIGHT * current.score

        # combined breakdown for downstream explanation
        combined = sorted(demo + wear + lab, key=lambda f: f.weight * (f.score if f.value is not None else 0.0), reverse=True)
        return CompositeResult(round(blended, 3), current.coverage, current.n_factors, combined,
                               multipliers=current.multipliers)

    @property
    def risk_score(self) -> float:
        """Latest blended risk score (0..1)."""
        return self.risk_history[-1][1] if self.risk_history else 0.0

    @property
    def risk_category(self) -> str:
        return _risk_category(self.risk_score)

    # ------------------------------------------------------------------ #
    # Trend
    # ------------------------------------------------------------------ #
    def risk_trend(self, window: Optional[int] = None) -> Optional[str]:
        """Least-squares slope of the risk trajectory.

        ``window=None`` (default) uses the full monitored history – stable and
        meaningful for a monitoring twin. Pass an int (e.g. 21) to assess just
        the trailing days. The series is lightly smoothed (3-point moving
        average) so day-to-day OU noise does not flip the trend label.
        """
        pts = list(self.risk_history if window is None else list(self.risk_history)[-window:])
        if len(pts) < 5:
            return None
        y = np.array([p[1] for p in pts], dtype=float)
        if len(y) >= 7:
            y = np.convolve(y, np.ones(3) / 3.0, mode="valid")
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
        if slope <= -0.0008:
            return "improving"
        if slope >= 0.0008:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------ #
    # Alerts (safety net)
    # ------------------------------------------------------------------ #
    def _evaluate_alerts(self, obs: Mapping[str, float]) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        s = self.state
        p = self.profile

        def add(level: str, code: str, message: str) -> None:
            alerts.append({"level": level, "code": code, "message": message, "day": str(obs.get("date", obs.get("day_index", "")))})

        sbp, dbp = s.get("systolic_bp"), s.get("diastolic_bp")
        rhr = s.get("resting_heart_rate")
        spo2 = s.get("nocturnal_spo2_pct")
        hrv = s.get("hrv_rmssd")

        if sbp is not None and (sbp >= 180 or (dbp or 0) >= 120):
            add("critical", "hypertensive_crisis", f"Severely elevated blood pressure ({sbp:.0f}/{dbp:.0f} mmHg). Seek medical evaluation urgently.")
        elif sbp is not None and sbp >= 140 and (dbp or 0) >= 90:
            add("warning", "bp_stage2", f"Blood pressure consistently in stage-2 hypertension range ({sbp:.0f}/{dbp:.0f} mmHg).")

        if rhr is not None:
            base_rhr = float(p.get("baseline_resting_hr", rhr))
            if rhr >= 100:
                add("warning", "tachycardia", f"Resting heart rate {rhr:.0f} bpm is unusually high.")
            elif rhr - base_rhr >= 7:
                add("info", "rhr_elevated", f"Resting HR {rhr:.0f} bpm is ~{rhr - base_rhr:.0f} bpm above your baseline ({base_rhr:.0f} bpm) – watch for illness, stress or overtraining.")

        if spo2 is not None and spo2 < 90:
            add("warning", "desaturation", f"Nocturnal SpO2 averaging {spo2:.0f}% – possible sleep-disordered breathing; discuss with a clinician.")

        if hrv is not None:
            base_hrv = float(p.get("baseline_hrv_rmssd", hrv))
            if base_hrv > 0 and hrv <= 0.7 * base_hrv:
                add("info", "hrv_suppression", f"HRV {hrv:.0f} ms is ~{100 * (1 - hrv / base_hrv):.0f}% below your baseline ({base_hrv:.0f} ms) – recovery/stress warning.")

        score = self.risk_history[-1][1] if self.risk_history else 0.0
        if score >= 0.65:
            add("warning", "high_risk", f"Composite cardiovascular risk is High/Very High ({score:.2f}). Review contributing factors and consider a check-up.")
        return alerts

    # ------------------------------------------------------------------ #
    # ML hook
    # ------------------------------------------------------------------ #
    def attach_model(self, model: Any, feature_order: List[str]) -> None:
        """Attach a trained classifier exposing predict_proba (e.g. XGBoost)."""
        self.model = model
        self.model_features = list(feature_order)

    def ml_risk(self) -> Optional[float]:
        """Model-estimated risk probability for the current twin state."""
        if self.model is None or not self.model_features:
            return None
        x = []
        for f in self.model_features:
            v = self.state.get(f, self.labs.get(f) if self.labs else None)
            x.append(float(v) if v is not None else 0.0)
        proba = self.model.predict_proba(np.asarray([x]))[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict:
        return {
            "profile": self.profile,
            "learning_rate": self.learning_rate,
            "history_len": self.history_len,
            "adaptive": self.adaptive,
            "state": self.state,
            "labs": self.labs,
            "days_seen": self.days_seen,
            "risk_history": list(self.risk_history),
            "alerts_raised": self.alerts_raised[-50:],
            "adaptive_state": {
                "baselines": self.baselines.to_dict(),
                "weight_learner": self.weight_learner.to_dict(),
                "confidence": self.confidence.to_dict(),
            } if self.adaptive else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DigitalTwin":
        twin = cls(
            profile=data["profile"],
            learning_rate=data.get("learning_rate", 0.15),
            history_len=data.get("history_len", 90),
            adaptive=data.get("adaptive", True),
        )
        twin.state.update({k: v for k, v in data.get("state", {}).items() if k in twin.state})
        twin.labs = data.get("labs")
        twin.days_seen = int(data.get("days_seen", 0))
        twin.risk_history.extend((int(d), float(r)) for d, r in data.get("risk_history", []))
        twin.alerts_raised = list(data.get("alerts_raised", []))
        ad = data.get("adaptive_state")
        if ad and twin.adaptive:
            twin.baselines = PersonalBaselineTracker.from_dict(ad.get("baselines", {}))
            twin.weight_learner = AdaptiveWeightLearner.from_dict(ad.get("weight_learner", {}))
            twin.confidence = ConfidenceModel.from_dict(ad.get("confidence", {}))
        return twin

    # ------------------------------------------------------------------ #
    def latest_lipid_analysis(self) -> Optional[Any]:
        if not self.labs:
            return None
        return LipidCalculator.analyze(dict(self.labs))

    # ------------------------------------------------------------------ #
    # Learning summary
    # ------------------------------------------------------------------ #
    def learning_summary(self) -> Dict:
        """Snapshot of the self-learning layer for dashboards/reports."""
        if not self.adaptive:
            return {"enabled": False}
        return {
            "enabled": True,
            "days_seen": self.days_seen,
            "warm_baselines": self.baselines.warm_count(),
            "channels_tracked": len(self.baselines.stats),
            "ci95": self.confidence.ci95(self.risk_score),
            "ci_width": self.confidence.width,
            "precision_gain_pct": self.confidence.precision_pct(),
            "top_learned_weights": self.weight_learner.summary(top=5),
        }
