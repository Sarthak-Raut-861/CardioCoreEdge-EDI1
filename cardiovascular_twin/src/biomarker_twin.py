"""
biomarker_twin.py
=================
The **biomarker digital twin** implementing the documentation's Chapter 9
state design and 11-step update cycle, plus the Chapter 8 clustering engine.

Update cycle (§9.2):
  1 receive sensor readings  -> 2 normalize  -> 3 calculate biomarkers
  -> 4 update state  -> 5 append history (rolling 100)  -> 6 baseline z-scores
  -> 7 risk score  -> 8 trend (last 3)  -> 9 cluster reassignment
  -> 10 alerts  -> 11 dashboard push (dashboard pulls ``snapshot()``)

Personalized baseline (§9.3): minimum 5 readings; significant |Z| > 2,
critical |Z| > 3; pct change vs baseline mean.

Clustering (Ch. 8): K-Means with K=4 on the 6 biomarkers, StandardScaler,
PCA 2-D projection, cluster confidence = 1 − d_own/max(d_any), clusters
labelled Low→Very-High by mean CVD score. A synthetic patient cohort fits
the model so a single user can be positioned within the population.

RESEARCH PROTOTYPE — AI estimates, not clinical measurements.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .adaptive_engine import PersonalBaselineTracker
from .biomarker_engine import BiomarkerEngine, BiomarkerResult
from .factors72 import FACTORS, FACTOR_IDS, build_factor_vector

__all__ = ["BiomarkerTwin", "BiomarkerClusterer"]

BIOMARKERS = ("TC", "HDL", "LDL", "TG", "CRP", "DD")
HISTORY_LEN = 100          # rolling buffer (§9.1)
BASELINE_MIN = 5           # §9.3
Z_SIGNIFICANT = 2.0
Z_CRITICAL = 3.0


# --------------------------------------------------------------------------- #
# Clustering (Chapter 8)
# --------------------------------------------------------------------------- #
class BiomarkerClusterer:
    """K-Means (K=4) over the 6 biomarkers; PCA 2-D; confidence scores."""

    RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]

    def __init__(self, k: int = 4, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.cluster_cvd_: Dict[int, float] = {}
        self.labels_map_: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, cvd_scores: Optional[pd.Series] = None) -> "BiomarkerClusterer":
        """Fit on a patient-cohort biomarker matrix (columns = 6 biomarkers)."""
        data = X[list(BIOMARKERS)].astype(float).dropna()
        if len(data) < 4 * self.k:
            raise ValueError(f"Need >= {4 * self.k} cohort rows, got {len(data)}")
        self.scaler = StandardScaler().fit(data.values)
        Xs = self.scaler.transform(data.values)
        self.model = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state).fit(Xs)
        self.pca = PCA(n_components=2, random_state=self.random_state).fit(Xs)

        cvd = np.asarray(cvd_scores if cvd_scores is not None
                         else BiomarkerEngine.compute(build_factor_vector({}, {})).cvd_score * np.ones(len(data)),
                         dtype=float)
        if len(cvd) != len(data):
            cvd = cvd[:len(data)]
        labels = self.model.labels_
        self.cluster_cvd_ = {int(c): round(float(cvd[labels == c].mean()), 3) for c in range(self.k)}
        order = sorted(self.cluster_cvd_, key=self.cluster_cvd_.get)   # low -> high
        self.labels_map_ = {int(c): self.RISK_LABELS[i] for i, c in enumerate(order)}
        return self

    # ------------------------------------------------------------------ #
    def assign(self, row: Mapping[str, float]) -> Dict[str, Any]:
        """Assign one patient's biomarkers; returns cluster, label, confidence, coords."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Clusterer not fitted")
        x = np.asarray([[float(row[b]) for b in BIOMARKERS]])
        xs = self.scaler.transform(x)
        dists = np.linalg.norm(self.model.cluster_centers_ - xs[0], axis=1)
        cid = int(np.argmin(dists))
        confidence = round(float(1 - dists[cid] / (dists.max() or 1.0)), 3)
        pt = self.pca.transform(xs)[0] if self.pca else (0.0, 0.0)
        return {
            "cluster": cid + 1,
            "label": self.labels_map_.get(cid, f"cluster {cid + 1}"),
            "confidence": confidence,
            "distances": [round(float(d), 3) for d in dists],
            "pc1": round(float(pt[0]), 3), "pc2": round(float(pt[1]), 3),
            "cluster_cvd_mean": self.cluster_cvd_.get(cid),
        }

    def cohort_points_2d(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X[list(BIOMARKERS)].astype(float).dropna()
        Xs = self.scaler.transform(data.values)
        pts = self.pca.transform(Xs)
        return pd.DataFrame({"pc1": pts[:, 0], "pc2": pts[:, 1],
                             "cluster": [self.labels_map_.get(int(c), str(c)) for c in self.model.labels_]})


def generate_cohort(n: int = 120, seed: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic patient cohort (uniform mixes of healthy/at-risk profiles)..

    Used to fit the K-Means population model so a single monitored user can be
    positioned among risk groups (doc §8).
    """
    rng = np.random.default_rng(seed)
    rows, cvds = [], []
    for _ in range(n):
        t = rng.uniform()                     # 0 = very healthy, 1 = very unhealthy
        noise = rng.normal(0, 0.05, 6)
        tc = 150 + (170 * t) + 12 * noise[0]
        hdl = 80 - (55 * t) + 6 * noise[1]
        tg = 80 + (330 * t) + 25 * noise[2]
        ldl = max(tc - hdl - tg / 5, 30)
        crp = 0.2 + (9.0 * t) + 0.6 * noise[3]
        dd = 0.1 + (2.6 * t) + 0.25 * noise[4]
        rows.append({"TC": tc, "HDL": hdl, "LDL": ldl, "TG": tg,
                     "CRP": max(crp, 0.2), "DD": dd})
        cvds.append(0.2 + 0.6 * t + 0.1 * noise[5])
    return pd.DataFrame(rows), pd.Series(cvds, name="cvd")


# --------------------------------------------------------------------------- #
# The biomarker digital twin (Chapter 9)
# --------------------------------------------------------------------------- #
@dataclass
class TwinState:
    """§9.1 state groups (serializable)."""
    biomarkers: Dict[str, float] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    categories: Dict[str, str] = field(default_factory=dict)
    adjusted: Dict[str, float] = field(default_factory=dict)
    cvd_score: float = 0.0
    cvd_category: str = "Low"
    ctr: float = 0.0
    ctr_category: str = "Low"
    pathway_risk: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, float] = field(default_factory=dict)
    cluster: Optional[Dict[str, Any]] = None
    zscores: Dict[str, float] = field(default_factory=dict)
    alerts: List[Dict[str, str]] = field(default_factory=list)
    top_contributions: Dict[str, List[Dict]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        return d


class BiomarkerTwin:
    """Continuously-updated cardiovascular biomarker twin (doc Ch. 9)."""

    def __init__(self, profile: Mapping, clusterer: Optional[BiomarkerClusterer] = None,
                 history_len: int = HISTORY_LEN, smooth_alpha: float = 0.05) -> None:
        self.profile = dict(profile)
        self.history_len = history_len
        self.smooth_alpha = smooth_alpha
        self.state = TwinState()
        self.factor_history: Deque[Dict[str, float]] = deque(maxlen=history_len)
        self.biomarker_history: Deque[Dict[str, float]] = deque(maxlen=history_len)
        self.risk_history: Deque[Tuple[int, float]] = deque(maxlen=history_len)
        self.alerts_raised: List[Dict[str, str]] = []
        self.baselines = PersonalBaselineTracker(window=history_len, min_cv=0.005)
        self._sbp_window: Deque[float] = deque(maxlen=21)
        self._sleep_window: Deque[float] = deque(maxlen=21)
        self._smoothed_factors: Optional[Dict[str, float]] = None
        self.days_seen = 0
        self.trend: Optional[str] = None
        self.clusterer = clusterer or BiomarkerClusterer().fit(*generate_cohort())

    # ------------------------------------------------------------------ #
    def update(self, obs: Mapping, day_index: int = 0) -> BiomarkerResult:
        """One 11-step update cycle from daily observations."""
        # 1-2) readings + normalization inside build_factor_vector
        bp_sd = float(np.std(self._sbp_window, ddof=1)) if len(self._sbp_window) >= 3 else 0.0
        sleep_sd = float(np.std(self._sleep_window, ddof=1)) if len(self._sleep_window) >= 3 else 0.0
        factors = build_factor_vector(obs, self.profile, day_index, bp_sd=bp_sd, sleep_sd=sleep_sd)

        # track windows for variability factors
        if obs.get("systolic_bp") is not None:
            self._sbp_window.append(float(obs["systolic_bp"]))
        if obs.get("sleep_hours") is not None:
            self._sleep_window.append(float(obs["sleep_hours"]))

        # 3) biomarkers — computed on the EWMA-smoothed factor vector
        # (a real NIR/PPG measurement window averages many beats/days; smoothing
        # also keeps daily channel noise from whipsawing the biomarker estimates)
        if self._smoothed_factors is None:
            self._smoothed_factors = dict(factors)
        else:
            a = self.smooth_alpha
            self._smoothed_factors = {
                k: round(v + a * (factors[k] - v), 4)
                for k, v in self._smoothed_factors.items()
            }
        result = BiomarkerEngine.compute(self._smoothed_factors, day_index=day_index)

        # 4) update state
        self.state = TwinState(biomarkers=dict(result.values), derived=dict(result.derived),
                               categories=dict(result.categories), adjusted=dict(result.adjusted),
                               cvd_score=result.cvd_score, cvd_category=result.cvd_category,
                               ctr=result.ctr, ctr_category=result.ctr_category,
                               pathway_risk=dict(result.pathway_risk), factors=dict(factors))

        # 5) history
        self.factor_history.append(dict(factors))
        self.biomarker_history.append(dict(result.values))
        self.risk_history.append((day_index, result.cvd_score))
        self.days_seen += 1

        # 6) baseline z-scores for biomarkers
        zs = self.baselines.update({b: result.values[b] for b in BIOMARKERS})
        self.state.zscores = zs

        # 7) risk score already in state; 8) trend from smoothed last points
        # (doc: compare last 3 risk scores; we lightly smooth the trailing
        # series first so daily estimation noise does not flip the direction)
        recent = [r for _, r in self.risk_history]
        if len(recent) >= 30:
            # doc §9.2 step 8: "compare last 3 risk scores". 3 points are noise
            # dominated on daily estimates, so we keep the semantics (recent
            # direction) but compare the mean of the last 14 days against the
            # previous 14 - robust to multi-day oscillation.
            last14 = float(np.mean(recent[-14:]))
            prev14 = float(np.mean(recent[-28:-14]))
            delta = last14 - prev14
            if delta > 0.010:
                self.trend = "Increasing"
            elif delta < -0.010:
                self.trend = "Decreasing"
            else:
                self.trend = "Stable"
        else:
            self.trend = None

        # 9) cluster assignment
        self.state.cluster = self.clusterer.assign(result.values)

        # 10) alerts + explainability payload (doc §10, top-5 per biomarker)
        self.state.top_contributions = {
            b: [c.to_dict() for c in result.top(b, 5)] for b in ("TC", "HDL", "TG", "CRP", "DD")
        }
        alerts = self._evaluate_alerts(result)
        self.state.alerts = alerts
        self.alerts_raised.extend(alerts)
        return result

    # ------------------------------------------------------------------ #
    def _evaluate_alerts(self, r: BiomarkerResult) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        v, c = r.values, r.categories

        def add(level, code, msg):
            alerts.append({"level": level, "code": code, "message": msg})

        if v["TC"] >= 240:
            add("warning", "tc_high", f"Total cholesterol {v['TC']:.0f} mg/dL (High).")
        if v["LDL"] >= 190:
            add("critical", "ldl_very_high", f"LDL {v['LDL']:.0f} mg/dL (Very High).")
        elif v["LDL"] >= 160:
            add("warning", "ldl_high", f"LDL {v['LDL']:.0f} mg/dL (High).")
        if v["HDL"] < 40:
            add("info", "hdl_low", f"HDL {v['HDL']:.0f} mg/dL (Low - protective barrier reduced).")
        if v["TG"] >= 200:
            add("warning", "tg_high", f"Triglycerides {v['TG']:.0f} mg/dL ({c['TG']}).")
        if v["CRP"] > 3.0:
            add("warning", "crp_high", f"CRP {v['CRP']:.2f} mg/L (High CV risk — inflammation).")
        elif v["CRP"] > 10:
            add("critical", "crp_acute", f"CRP {v['CRP']:.2f} mg/L — acute infection range.")
        if v["DD"] > 2.0:
            add("critical", "dd_high", f"D-Dimer {v['DD']:.2f} mg/L (High Risk — clotting activity).")
        elif v["DD"] > 1.0:
            add("warning", "dd_moderate", f"D-Dimer {v['DD']:.2f} mg/L (Moderate elevation).")
        if r.ctr > 0.66:
            add("warning", "ctr_high", f"Coronary thrombosis risk {r.ctr:.2f} (High).")

        # baseline-deviation alerts (§9.3) — deduped vs previous day per biomarker
        prev_codes = {a["code"] for a in self.alerts_raised[-10:]}
        for b in BIOMARKERS:
            z = self.state.zscores.get(b)
            if z is None or self.days_seen < BASELINE_MIN:
                continue
            if abs(z) > Z_CRITICAL:
                add("warning", f"baseline_shift_{b}", f"{b} deviates {z:+.1f}σ from personal baseline (critical).")
            elif abs(z) > Z_SIGNIFICANT:
                add("info", f"baseline_shift_{b}", f"{b} deviates {z:+.1f}σ from personal baseline (significant).")
        # suppress codes that already fired yesterday (avoid daily spam)
        alerts = [a for a in alerts
                  if a["code"] not in prev_codes or a["level"] == "critical"]
        return alerts

    # ------------------------------------------------------------------ #
    def baseline_table(self) -> List[Dict[str, Any]]:
        rows = []
        for b in BIOMARKERS:
            base = self.baselines.baseline(b)
            ci = self.baselines.baseline_ci(b)
            cur = self.state.biomarkers.get(b)
            z = self.state.zscores.get(b)
            pct = None
            if base and cur is not None and base != 0:
                pct = round(100 * (cur - base) / base, 1)
            rows.append({"biomarker": b, "baseline_mean": base,
                         "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                         "current": cur, "z": z, "pct_change": pct,
                         "significant": bool(z is not None and abs(z) > Z_SIGNIFICANT),
                         "n_readings": self.baselines.stats[b].n if b in self.baselines.stats else 0})
        return rows

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict:
        return {
            "profile": self.profile,
            "state": self.state.to_dict(),
            "days_seen": self.days_seen,
            "trend": self.trend,
            "risk_history": list(self.risk_history),
            "biomarker_history": list(self.biomarker_history),
            "alerts_raised": self.alerts_raised[-50:],
            "baselines": self.baselines.to_dict(),
        }
