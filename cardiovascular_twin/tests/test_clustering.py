"""Tests for src/clustering.py"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import DEFAULT_FEATURES, PhenotypeClusterer


def make_two_cluster_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Two well-separated day types: 'recovered' and 'strained'."""
    rng = np.random.default_rng(seed)
    good = {
        "resting_heart_rate": 55, "hrv_rmssd": 70, "systolic_bp": 112, "diastolic_bp": 72,
        "sleep_hours": 7.8, "deep_sleep_pct": 20, "steps": 10000, "nocturnal_spo2_pct": 97,
    }
    bad = {
        "resting_heart_rate": 85, "hrv_rmssd": 25, "systolic_bp": 148, "diastolic_bp": 95,
        "sleep_hours": 5.2, "deep_sleep_pct": 9, "steps": 2500, "nocturnal_spo2_pct": 92,
    }
    rows = []
    for _ in range(n // 2):
        rows.append({k: v + rng.normal(0, abs(v) * 0.02) for k, v in good.items()})
        rows.append({k: v + rng.normal(0, abs(v) * 0.02) for k, v in bad.items()})
    return pd.DataFrame(rows)


class TestFitting:
    def test_fits_and_selects_k(self):
        df = make_two_cluster_df(40)
        c = PhenotypeClusterer().fit(df)
        assert c.chosen_k_ is not None and c.chosen_k_ >= 2
        assert c.silhouette_ is not None and c.silhouette_ > 0.5  # well separated

    def test_fixed_k_respected(self):
        df = make_two_cluster_df(40)
        c = PhenotypeClusterer(k=2).fit(df)
        assert c.chosen_k_ == 2

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            PhenotypeClusterer().fit(pd.DataFrame({"resting_heart_rate": [60, 61, 62]}))

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            PhenotypeClusterer().fit(make_two_cluster_df(6))


class TestOutputs:
    def test_summary_structure(self):
        c = PhenotypeClusterer().fit(make_two_cluster_df(40))
        s = c.summary()
        assert s["k"] == c.chosen_k_
        assert len(s["clusters"]) == c.chosen_k_
        assert all(set(cl["means"]) == set(c.features) for cl in s["clusters"])
        shares = sum(cl["share"] for cl in s["clusters"])
        assert shares == pytest.approx(1.0, abs=0.01)

    def test_cluster_labels_assigned(self):
        c = PhenotypeClusterer().fit(make_two_cluster_df(40))
        assert len(c.labels_map_) == c.chosen_k_
        assert all(isinstance(v, str) and v for v in c.labels_map_.values())

    def test_predict_and_shares(self):
        df = make_two_cluster_df(40)
        c = PhenotypeClusterer(k=2).fit(df)
        labels = c.predict(df)
        assert len(labels) == len(df)
        shares = c.state_shares(labels)
        assert sum(shares.values()) == pytest.approx(1.0, abs=0.02)

    def test_transform_2d(self):
        df = make_two_cluster_df(40)
        c = PhenotypeClusterer(k=2).fit(df)
        pts = c.transform_2d(df)
        assert list(pts.columns) == ["pc1", "pc2"]
        assert len(pts) == len(df)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            PhenotypeClusterer().predict(make_two_cluster_df(20))
