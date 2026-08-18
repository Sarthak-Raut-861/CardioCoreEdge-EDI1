"""
explainability.py
=================
Explanation layer for the CardioCore digital twin.

Two complementary explanation paths:

1. **Rule-based / factor attribution** (always available)
   Decomposes the composite risk score into per-factor contributions
   (weight x risk points), ranks them, and produces a short natural-language
   summary of what is driving the current risk level.

2. **Model attribution via SHAP** (when a tree model is attached to the twin)
   Uses ``shap.TreeExplainer`` to attribute an XGBoost risk-model prediction
   to individual physiological features.

Educational/research use only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .factor_engine import CompositeResult, Factor

__all__ = ["RiskExplainer"]


class RiskExplainer:
    """Generates explanations from composites and/or SHAP-attributed models."""

    # ------------------------------------------------------------------ #
    # 1) Rule-based factor attribution
    # ------------------------------------------------------------------ #
    @staticmethod
    def explain_composite(composite: CompositeResult) -> pd.DataFrame:
        """Per-factor contribution table for one composite result.

        Columns: factor, modality, value, score, weight, contribution,
                 share, direction (risk-increasing / risk-decreasing / neutral)
        """
        rows: List[Dict] = []
        available = [f for f in composite.breakdown if f.value is not None]
        total_contribution = sum(f.weight * f.score for f in available) or 1.0
        for f in available:
            contribution = round(f.weight * f.score, 4)
            share = round(contribution / total_contribution, 3)
            if f.score >= 0.6:
                direction = "risk-increasing"
            elif f.score <= 0.2:
                direction = "risk-decreasing"
            else:
                direction = "neutral"
            rows.append({
                "factor": f.name,
                "modality": f.modality,
                "value": f.value,
                "score": f.score,
                "weight": f.weight,
                "contribution": contribution,
                "share": share,
                "assessment": direction,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def narrative(composite: CompositeResult, top_n: int = 3) -> str:
        """Short natural-language summary of the dominant risk drivers."""
        if composite.n_factors == 0:
            return "No data available yet to assess cardiovascular risk."
        drivers = [f for f in composite.top(top_n) if f.score >= 0.35]
        positives = [f for f in composite.breakdown if f.value is not None and f.score <= 0.15]

        parts: List[str] = []
        if drivers:
            binary = {"smoking", "diabetes", "family_history"}
            named = ", ".join(
                f.name if f.key in binary else f"{f.name} ({_fmt(f.value, f.key)})"
                for f in drivers
            )
            parts.append(f"Risk is driven mainly by {named}.")
        else:
            parts.append("No strongly elevated risk factors right now.")
        if positives:
            parts.append("Protective/healthy: " + ", ".join(f.name for f in positives[:3]) + ".")
        parts.append(f"Composite risk {composite.score:.2f} with {int(composite.coverage * 100)}% of expected data available.")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # 2) SHAP model attribution
    # ------------------------------------------------------------------ #
    @staticmethod
    def explain_model_prediction(
        model,
        X: pd.DataFrame,
        row_index: int = -1,
    ) -> Dict:
        """SHAP attribution for one row of ``X`` using a tree model.

        Returns dict with prediction probability, base value, per-feature SHAP
        contributions (sorted by |value|) and a plain-English driver list.
        Falls back with a clear error if shap is unavailable.
        """
        try:
            import shap  # lazy – keeps core importable without shap
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("shap is required for model explanations: pip install shap") from exc

        explainer = shap.TreeExplainer(model)
        Xs = X.astype(float)
        row = Xs.iloc[[row_index]]
        sv = explainer.shap_values(row)

        # binary xgboost returns array or list of arrays depending on version
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv).reshape(-1)

        proba = model.predict_proba(row.values)[0]
        prediction = float(proba[1]) if len(proba) > 1 else float(proba[0])

        contributions = sorted(
            (
                {"feature": col, "shap_value": round(float(v), 4)}
                for col, v in zip(Xs.columns, sv)
            ),
            key=lambda d: abs(d["shap_value"]),
            reverse=True,
        )
        drivers = [c["feature"] for c in contributions[:3] if c["shap_value"] > 0]
        return {
            "prediction": round(prediction, 3),
            "base_value": round(float(np.ravel(explainer.expected_value)[-1]), 3),
            "contributions": contributions,
            "top_risk_drivers": drivers,
            "n_features": len(contributions),
        }

    # ------------------------------------------------------------------ #
    def explain(self, composite: Optional[CompositeResult] = None, model=None,
                X: Optional[pd.DataFrame] = None) -> Dict:
        """Convenience: returns both explanation paths where possible."""
        out: Dict = {}
        if composite is not None:
            table = self.explain_composite(composite)
            out["factor_table"] = table
            out["narrative"] = self.narrative(composite)
        if model is not None and X is not None:
            out["shap"] = self.explain_model_prediction(model, X)
        return out


# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], key: str) -> str:
    if value is None:
        return "n/a"
    units = {
        "resting_heart_rate": "bpm", "hrv_rmssd": "ms", "systolic_bp": "mmHg",
        "diastolic_bp": "mmHg", "steps": "steps", "sleep_hours": "h",
        "ldl": "mg/dL", "non_hdl": "mg/dL", "hdl": "mg/dL",
        "triglycerides": "mg/dL", "nocturnal_spo2_pct": "%",
    }
    return f"{value:.0f}{units.get(key, '')}" if key in units else f"{value:.2f}"
