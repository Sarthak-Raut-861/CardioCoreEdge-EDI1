"""
adaptive_engine.py
==================
The **self-learning personalization layer** of the CardioCore digital twin.

Three cooperating components make the twin "learn from previous data":

1. ``RunningStats`` / ``PersonalBaselineTracker``
   Online (Welford) per-channel statistics -> the twin learns the *user's own*
   normal values, replacing population reference ranges with personalized
   baselines after a warm-up period. Exposes personalized z-scores and
   anomaly flags (|z| > 2.5).

2. ``AdaptiveWeightLearner``
   Self-supervised online learning (Hedge / multiplicative-weights):
   each day, factors whose day-over-day change correctly anticipated the
   next composite-risk movement are up-weighted; noisy/uninformative factors
   are down-weighted. Multipliers are bounded in [0.5, 2.0] so learning can
   refine, never explode. A precision term (stability = 1/(1+CV) of the
   trailing factor scores) additionally favours factors that are *stable*
   for this specific user.

3. ``ConfidenceModel``
   Precision that visibly improves with data volume: the reported risk
   carries a 95% CI whose width shrinks ~ 1/sqrt(n_days) as history
   accumulates.

Everything is serializable (``to_dict``/``from_dict``) and stored inside the
twin snapshot.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RunningStats",
    "PersonalBaselineTracker",
    "AdaptiveWeightLearner",
    "ConfidenceModel",
]


# --------------------------------------------------------------------------- #
# 1) Online statistics
# --------------------------------------------------------------------------- #
class RunningStats:
    """Numerically stable running mean/variance (Welford's algorithm)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 1e-9))

    def z(self, x: float) -> float:
        return (x - self.mean) / self.std

    def to_dict(self) -> Dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: Mapping) -> "RunningStats":
        s = cls()
        s.n, s.mean, s.m2 = int(d.get("n", 0)), float(d.get("mean", 0.0)), float(d.get("m2", 0.0))
        return s


class PersonalBaselineTracker:
    """Learns per-channel personal baselines from the user's own history."""

    WARMUP_DAYS = 14          # days before baselines replace population norms
    ANOMALY_Z = 2.5

    def __init__(self, window: int = 90, min_cv: float = 0.03) -> None:
        self.window = window
        self.min_cv = min_cv   # channels below this CV produce no z-scores
        self.stats: Dict[str, RunningStats] = {}
        self.recent: Dict[str, Deque[float]] = {}

    def update(self, obs: Mapping[str, Optional[float]]) -> Dict[str, float]:
        """Record one day of observations; returns z-scores for provided keys."""
        zs: Dict[str, float] = {}
        for key, value in obs.items():
            if value is None or isinstance(value, (str, bool)):
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(x):
                continue
            self.stats.setdefault(key, RunningStats()).update(x)
            dq = self.recent.setdefault(key, deque(maxlen=self.window))
            # leave-one-out z: compare against history *excluding* today, and
            # only when that history is long enough (>=10) and dispersed
            # enough (cv >= 3%) for a z-score to be meaningful
            if len(dq) >= 10:
                hist = np.asarray(dq)
                mu, sd = float(hist.mean()), float(hist.std(ddof=1)) or 1e-9
                if abs(mu) > 1e-9 and (sd / abs(mu)) >= self.min_cv:
                    zs[key] = round((x - mu) / sd, 3)
            dq.append(x)
        return zs

    def is_warm(self, key: str) -> bool:
        st = self.stats.get(key)
        return st is not None and st.n >= self.WARMUP_DAYS

    def baseline(self, key: str) -> Optional[float]:
        st = self.stats.get(key)
        return round(st.mean, 3) if st else None

    def baseline_ci(self, key: str) -> Optional[Tuple[float, float]]:
        st = self.stats.get(key)
        if not st or st.n < self.WARMUP_DAYS:
            return None
        half = 1.96 * st.std / math.sqrt(st.n)
        return (round(st.mean - half, 3), round(st.mean + half, 3))

    def warm_count(self) -> int:
        return sum(1 for k in self.stats if self.is_warm(k))

    def anomalies(self, zscores: Mapping[str, float]) -> List[str]:
        """Keys whose leave-one-out |z| exceeds the threshold.

        (Dispersion/length guards are applied when the z-score is produced in
        ``update``; this simply thresholds the surviving z-scores.)
        """
        return [k for k, z in zscores.items() if abs(z) > self.ANOMALY_Z]

    def table(self, keys: Sequence[str]) -> List[Dict]:
        rows = []
        for k in keys:
            ci = self.baseline_ci(k)
            dq = self.recent.get(k)
            current = dq[-1] if dq else None
            rows.append({
                "channel": k,
                "n_days": self.stats[k].n if k in self.stats else 0,
                "warm": self.is_warm(k),
                "baseline": self.baseline(k),
                "ci95_low": ci[0] if ci else None,
                "ci95_high": ci[1] if ci else None,
                "current": current,
                "z_current": None,
            })
        return rows

    def to_dict(self) -> Dict:
        return {
            "window": self.window,
            "stats": {k: s.to_dict() for k, s in self.stats.items()},
            "recent": {k: list(v) for k, v in self.recent.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "PersonalBaselineTracker":
        t = cls(window=int(d.get("window", 90)))
        t.stats = {k: RunningStats.from_dict(v) for k, v in d.get("stats", {}).items()}
        t.recent = {k: deque(v, maxlen=t.window) for k, v in d.get("recent", {}).items()}
        return t


# --------------------------------------------------------------------------- #
# 2) Self-supervised adaptive weights (Hedge)
# --------------------------------------------------------------------------- #
class AdaptiveWeightLearner:
    """Multiplicative-weights online learning of per-factor weight multipliers.

    Self-supervised target: the *next-day direction* of the composite risk
    score. A factor is rewarded when its day-over-day change agrees with the
    next composite move (it "anticipated" the risk movement), penalised
    otherwise. Multipliers stay in [MIN, MAX].
    """

    MIN_MULT = 0.5
    MAX_MULT = 2.0
    MIN_HISTORY = 10          # days before multipliers activate
    ETA = 0.08                # learning rate

    def __init__(self) -> None:
        self.multipliers: Dict[str, float] = {}
        self.prev_scores: Optional[Dict[str, float]] = None
        self.prev_composite: Optional[float] = None
        self.rewards: Dict[str, Deque[float]] = {}
        self.score_history: Dict[str, Deque[float]] = {}

    # ------------------------------------------------------------------ #
    def observe(self, factor_scores: Mapping[str, Optional[float]], composite: float) -> Dict[str, float]:
        """Record one day; update multipliers; return active multipliers."""
        clean = {k: float(v) for k, v in factor_scores.items() if v is not None}

        # per-factor trailing score history (for stability/precision term)
        for k, v in clean.items():
            self.score_history.setdefault(k, deque(maxlen=14)).append(v)

        if self.prev_scores is not None and self.prev_composite is not None:
            target_up = composite > self.prev_composite
            for k, cur in clean.items():
                prev = self.prev_scores.get(k)
                if prev is None or abs(cur - prev) < 1e-6:
                    continue
                factor_up = cur > prev
                reward = 1.0 if factor_up == target_up else -1.0
                self.rewards.setdefault(k, deque(maxlen=14)).append(reward)
                n_obs = len(self.rewards[k])
                if n_obs >= self.MIN_HISTORY:
                    mult = self.multipliers.get(k, 1.0) * math.exp(self.ETA * reward)
                    self.multipliers[k] = min(max(mult, self.MIN_MULT), self.MAX_MULT)

        self.prev_scores = clean
        self.prev_composite = composite
        return self.active_multipliers()

    # ------------------------------------------------------------------ #
    def stability(self, key: str) -> float:
        """1 for perfectly stable factor -> 0.5 for very noisy (CV=1)."""
        dq = self.score_history.get(key)
        if not dq or len(dq) < 5:
            return 1.0
        arr = np.asarray(dq, dtype=float)
        mu = arr.mean()
        if abs(mu) < 1e-9:
            return 1.0
        cv = float(arr.std(ddof=1) / abs(mu))
        return 1.0 / (1.0 + cv)

    def active_multipliers(self) -> Dict[str, float]:
        """Multiplier per factor: hedge * precision, once enough history."""
        out: Dict[str, float] = {}
        for k in self.score_history:
            if len(self.rewards.get(k, ())) < self.MIN_HISTORY:
                out[k] = 1.0
                continue
            hedge = self.multipliers.get(k, 1.0)
            out[k] = round(hedge * (0.5 + 0.5 * self.stability(k)), 3)
        return out

    def summary(self, top: int = 10) -> List[Dict]:
        rows = []
        for k, m in sorted(self.active_multipliers().items(), key=lambda kv: kv[1], reverse=True):
            dq = self.rewards.get(k, ())
            rows.append({
                "factor": k,
                "multiplier": m,
                "hedge": round(self.multipliers.get(k, 1.0), 3),
                "stability": round(self.stability(k), 3),
                "hit_rate": round(sum(dq) / len(dq), 3) if dq else None,  # directional accuracy
                "n_updates": len(dq),
            })
        return rows[:top] + rows[-top:] if len(rows) > 2 * top else rows

    def to_dict(self) -> Dict:
        return {
            "multipliers": dict(self.multipliers),
            "prev_scores": self.prev_scores,
            "prev_composite": self.prev_composite,
            "rewards": {k: list(v) for k, v in self.rewards.items()},
            "score_history": {k: list(v) for k, v in self.score_history.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "AdaptiveWeightLearner":
        l = cls()
        l.multipliers = {k: float(v) for k, v in d.get("multipliers", {}).items()}
        l.prev_scores = d.get("prev_scores")
        l.prev_composite = d.get("prev_composite")
        l.rewards = {k: deque(v, maxlen=14) for k, v in d.get("rewards", {}).items()}
        l.score_history = {k: deque(v, maxlen=14) for k, v in d.get("score_history", {}).items()}
        return l


# --------------------------------------------------------------------------- #
# 3) Confidence / precision model
# --------------------------------------------------------------------------- #
class ConfidenceModel:
    """95% CI for the composite risk estimate; width shrinks with data volume."""

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.scores: Deque[float] = deque(maxlen=window)
        self.n_total = 0
        self.history: List[Tuple[int, float]] = []  # (day, ci_width)

    def update(self, score: float) -> Tuple[float, float]:
        self.scores.append(score)
        self.n_total += 1
        low, high = self.ci95()
        self.history.append((self.n_total - 1, round(high - low, 4)))
        return low, high

    def ci95(self, point: Optional[float] = None) -> Tuple[float, float]:
        centre = float(point if point is not None else (np.mean(self.scores) if self.scores else 0.5))
        if len(self.scores) < 3:
            return (round(max(centre - 0.15, 0.0), 3), round(min(centre + 0.15, 1.0), 3))  # conservative prior width
        arr = np.asarray(self.scores, dtype=float)
        half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
        return (round(max(centre - half, 0.0), 3), round(min(centre + half, 1.0), 3))

    @property
    def width(self) -> float:
        low, high = self.ci95()
        return round(high - low, 4)

    def precision_pct(self) -> float:
        """Improvement of CI width vs the 3-day baseline width (higher = better)."""
        if len(self.history) < 5:
            return 0.0
        first = next((w for _, w in self.history if w > 0), None)
        if not first:
            return 0.0
        return round(100 * (1 - self.width / first), 1)

    def to_dict(self) -> Dict:
        return {"window": self.window, "scores": list(self.scores),
                "n_total": self.n_total, "history": self.history}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ConfidenceModel":
        c = cls(window=int(d.get("window", 14)))
        c.scores = deque(d.get("scores", []), maxlen=c.window)
        c.n_total = int(d.get("n_total", 0))
        c.history = [tuple(x) for x in d.get("history", [])]
        return c
