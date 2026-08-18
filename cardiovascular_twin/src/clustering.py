"""
clustering.py
=============
Unsupervised *phenotyping* of daily physiological states.

The digital twin feeds this component a matrix of daily aggregated wearable
features; it discovers recurring physiological states ("phenotypes") using
KMeans, selects k by silhouette score, projects to 2-D with PCA for
visualisation, and names each cluster with heuristics based on how the
cluster deviates from the population mean (e.g. "sympathetic-dominant",
"sedentary / poor-sleep", "hypertensive pattern", "balanced / recovered").

Useful for
----------
* summarising what "kinds of days" a user has,
* detecting whether unhealthy states dominate recently (state-share drift),
* labelling data for downstream supervised analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

__all__ = ["PhenotypeClusterer", "DEFAULT_FEATURES"]


DEFAULT_FEATURES = [
    "resting_heart_rate", "hrv_rmssd", "systolic_bp", "diastolic_bp",
    "sleep_hours", "deep_sleep_pct", "steps", "nocturnal_spo2_pct",
]


class PhenotypeClusterer:
    """KMeans + PCA phenotyper over daily wearable features.

    Parameters
    ----------
    k           : number of clusters; None -> searched over 2..k_max by silhouette
    k_max       : upper bound for the k search
    random_state: reproducibility
    """

    def __init__(self, k: Optional[int] = None, k_max: int = 5, random_state: int = 42) -> None:
        self.k = k
        self.k_max = max(2, k_max)
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.features: List[str] = []
        self.silhouette_: Optional[float] = None
        self.chosen_k_: Optional[int] = None
        self.profiles_: Optional[pd.DataFrame] = None
        self.labels_map_: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    def _prepare(self, df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        cols = list(features) or DEFAULT_FEATURES
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing feature columns: {missing}")
        data = df[cols].astype(float).dropna()
        if len(data) < 2 * self.k_max:
            raise ValueError(f"Need at least {2 * self.k_max} complete rows to cluster, got {len(data)}")
        return data

    def fit(self, df: pd.DataFrame, features: Optional[Sequence[str]] = None) -> "PhenotypeClusterer":
        data = self._prepare(df, features or DEFAULT_FEATURES)
        self.features = list(data.columns)
        X = self.scaler = StandardScaler().fit(data.values)
        Xs = X.transform(data.values)

        # --- choose k -------------------------------------------------- #
        best_k, best_score = 2, -1.0
        k_candidates = [self.k] if self.k else range(2, min(self.k_max, len(data) - 1) + 1)
        for k in k_candidates:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(Xs)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(Xs, labels)
            if s > best_score:
                best_k, best_score = k, s
        self.chosen_k_ = best_k
        self.silhouette_ = float(best_score) if best_score >= 0 else None

        self.model = KMeans(n_clusters=best_k, n_init=10, random_state=self.random_state).fit(Xs)
        self.pca = PCA(n_components=2, random_state=self.random_state).fit(Xs)
        labels = self.model.labels_

        # --- profiles in original units -------------------------------- #
        prof = data.groupby(labels).mean()
        prof.index.name = "cluster"
        prof["n_days"] = data.groupby(labels).size()
        prof["share"] = (prof["n_days"] / len(data)).round(3)
        # z-scores vs overall mean (population std from scaler)
        overall = self.scaler.mean_
        scale = self.scaler.scale_
        z = (prof[self.features].values - overall) / scale
        for j, f in enumerate(self.features):
            prof[f"z_{f}"] = z[:, j].round(2)
        self.profiles_ = prof
        self.labels_map_ = self._name_clusters(prof)
        return self

    # ------------------------------------------------------------------ #
    def _name_clusters(self, prof: pd.DataFrame) -> Dict[int, str]:
        """Heuristic naming from z-deviations of each cluster centroid."""
        names: Dict[int, str] = {}
        # rank clusters by a crude "strain" score to keep naming consistent
        strain = (
            prof.get("z_resting_heart_rate", 0) * 1.0
            - prof.get("z_hrv_rmssd", 0) * 1.0
            + prof.get("z_systolic_bp", 0) * 1.0
            + prof.get("z_diastolic_bp", 0) * 0.5
            - prof.get("z_sleep_hours", 0) * 0.5
            - prof.get("z_steps", 0) * 0.5
            - prof.get("z_hrv_rmssd", 0) * 0.0
        )
        order = strain.sort_values().index.tolist()  # least strained first

        for cid in prof.index:
            z = {f: prof.loc[cid, f"z_{f}"] for f in self.features}
            if cid == order[0] and strain[cid] < 0.3:
                names[cid] = "balanced / well-recovered"
                continue
            tags = []
            if z.get("resting_heart_rate", 0) > 0.5 and z.get("hrv_rmssd", 0) < -0.3:
                tags.append("sympathetic-dominant")
            if z.get("systolic_bp", 0) > 0.7 or z.get("diastolic_bp", 0) > 0.7:
                tags.append("hypertensive pattern")
            if z.get("steps", 0) < -0.5:
                tags.append("sedentary")
            if z.get("sleep_hours", 0) < -0.5 or z.get("deep_sleep_pct", 0) < -0.5:
                tags.append("poor sleep")
            if z.get("hrv_rmssd", 0) < -0.6:
                tags.append("low HRV")
            if z.get("nocturnal_spo2_pct", 0) < -0.6:
                tags.append("nocturnal desaturation")
            names[cid] = " + ".join(tags[:2]) if tags else "mixed pattern"
        return names

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Cluster labels for new daily rows (ints, aligned with df index)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Clusterer is not fitted; call fit() first.")
        data = df[self.features].astype(float)
        Xs = self.scaler.transform(data.fillna(data.mean()).values)
        return pd.Series(self.model.predict(Xs), index=data.index, name="cluster")

    def transform_2d(self, df: pd.DataFrame) -> pd.DataFrame:
        """2-D PCA coordinates for plotting."""
        if self.pca is None or self.scaler is None:
            raise RuntimeError("Clusterer is not fitted; call fit() first.")
        data = df[self.features].astype(float)
        Xs = self.scaler.transform(data.fillna(data.mean()).values)
        pts = self.pca.transform(Xs)
        return pd.DataFrame({"pc1": pts[:, 0], "pc2": pts[:, 1]}, index=data.index)

    # ------------------------------------------------------------------ #
    def state_shares(self, labels: pd.Series) -> Dict[str, float]:
        """Share of days spent in each named phenotype."""
        if not self.labels_map_:
            raise RuntimeError("Clusterer is not fitted; call fit() first.")
        counts = labels.value_counts(normalize=True)
        return {
            self.labels_map_.get(int(cid), f"cluster {cid}"): round(float(share), 3)
            for cid, share in counts.items()
        }

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict:
        if self.profiles_ is None:
            raise RuntimeError("Clusterer is not fitted; call fit() first.")
        return {
            "k": self.chosen_k_,
            "silhouette": round(float(self.silhouette_ or 0.0), 3),
            "features": self.features,
            "clusters": [
                {
                    "cluster": int(cid),
                    "label": self.labels_map_.get(int(cid), ""),
                    "n_days": int(row["n_days"]),
                    "share": float(row["share"]),
                    "means": {f: round(float(row[f]), 2) for f in self.features},
                }
                for cid, row in self.profiles_.iterrows()
            ],
        }
