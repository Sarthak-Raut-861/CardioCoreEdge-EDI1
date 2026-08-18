#!/usr/bin/env python3
"""
main.py — CardioCore demo pipeline
==================================
End-to-end demonstration of the core engine:

    simulator -> digital twin (daily assimilation) -> risk trajectory
              -> phenotype clustering -> factor explanations -> (optional) ML + SHAP

Usage
-----
    python main.py                          # 60-day typical user, stable
    python main.py --profile at_risk --scenario declining --days 90
    python main.py --ml                     # + train XGBoost cohort model & SHAP

DISCLAIMER: research/educational prototype – not a medical device.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from src.clustering import DEFAULT_FEATURES, PhenotypeClusterer
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource, UserProfile, simulate_history
from src.digital_twin import DISCLAIMER, DigitalTwin
from src.explainability import RiskExplainer
from src.factor_engine import FactorEngine


# --------------------------------------------------------------------------- #
def run_pipeline(days: int, profile_name: str, scenario: str, seed: int, lr: float) -> DigitalTwin:
    """Simulate + assimilate one user; prints progress. Returns fitted twin."""
    profile: UserProfile = PRESET_PROFILES[profile_name]
    source = SimulatedWearableSource(profile, scenario=scenario, seed=seed, lab_interval_days=90)
    twin = DigitalTwin(profile.to_dict(), learning_rate=lr)

    print(f"\n=== CardioCore digital twin — {profile.name} "
          f"(age {profile.age}, sex {profile.sex}, BMI {profile.bmi}) | scenario: {scenario} ===")

    header_printed = False
    for obs in source.next_days(days):
        update = twin.assimilate(obs, obs.get("labs"))
        if obs["day_index"] % max(1, days // 10) == 0 or obs["day_index"] == days - 1:
            if not header_printed:
                print(f"{'day':>4}  {'date':>10}  {'risk':>5}  {'category':<9}  trend")
                header_printed = True
            top = update.composite.top(1)[0].name if update.composite and update.composite.breakdown else "-"
            print(f"{obs['day_index']:>4}  {obs['date']:>10}  {update.risk_score:>5.2f}  "
                  f"{update.category:<9}  {update.trend or '-':<10} top: {top}")
    return twin


def final_report(twin: DigitalTwin, history: pd.DataFrame) -> None:
    print("\n--- FINAL ASSESSMENT -------------------------------------------")
    print(f"Days monitored : {twin.days_seen}")
    print(f"Risk score     : {twin.risk_score:.2f}  ({twin.risk_category})")
    print(f"Trend          : {twin.risk_trend()}")

    composite = twin._compute_risk()
    explainer = RiskExplainer()
    print("\nTop risk drivers (factor attribution):")
    table = explainer.explain_composite(composite).head(8)
    for _, row in table.iterrows():
        print(f"  - {row['factor']:<26} value={row['value']!s:>8}  score={row['score']:.2f}  "
              f"share={row['share'] * 100:4.0f}%  [{row['modality']}]")
    print(f"\nNarrative: {explainer.narrative(composite)}")

    analysis = twin.latest_lipid_analysis()
    if analysis is not None:
        print(f"\nLipids: LDL {analysis.ldl} mg/dL ({analysis.ldl_method}, {analysis.ldl_category}) | "
              f"non-HDL {analysis.non_hdl} ({analysis.non_hdl_category}) | "
              f"TC/HDL {analysis.tc_hdl_ratio} | AIP {analysis.atherogenic_index}")

    if twin.alerts_raised:
        print(f"\nAlerts raised ({len(twin.alerts_raised)}):")
        for a in twin.alerts_raised[-6:]:
            print(f"  [{a['level'].upper():<8}] {a['day']}: {a['message']}")
    else:
        print("\nNo alerts raised.")

    # phenotype clustering over the monitored window
    try:
        clusterer = PhenotypeClusterer().fit(history)
        print(f"\nDaily phenotypes (k={clusterer.chosen_k_}, silhouette={clusterer.silhouette_:.2f}):")
        for c in clusterer.summary()["clusters"]:
            print(f"  - {c['label']:<34} {c['n_days']:>3} days ({c['share'] * 100:.0f}%)")
    except ValueError as exc:
        print(f"\n(phenotype clustering skipped: {exc})")

    print(f"\n{DISCLAIMER}")


# --------------------------------------------------------------------------- #
def ml_report(twin: DigitalTwin, model_path: str | None = None) -> None:
    """Load (or train) the real risk model and score the twin's state with SHAP."""
    from pathlib import Path

    from src.model_training import ensure_dataset, load_artifact, load_dataset, save_artifact, train_risk_model, twin_feature_vector

    print("\n--- ML LAYER (real-data risk model) -----------------------------")
    artifact = None
    path = Path(model_path or "models/risk_model.joblib")
    if path.exists():
        artifact = load_artifact(path)
        print(f"Loaded trained model from {path} ({artifact['model_name']}, "
              f"holdout AUC {artifact['holdout_calibrated']['roc_auc']})")
    else:
        print("No trained model found - training on the UCI-combined dataset ...")
        data_path = ensure_dataset()
        artifact = train_risk_model(load_dataset(data_path), tune=False)
        save_artifact(artifact, "models", training_df=load_dataset(data_path))

    # twin state -> model feature row
    ref_df = artifact.get("data_sample")
    if ref_df is None:
        ref_df = load_dataset(ensure_dataset())
    row = twin_feature_vector(twin, ref_df)
    prob = float(artifact["model"].predict_proba(row)[0, 1])
    print(f"Twin-state risk probability: {prob:.2f} "
          f"(threshold {artifact['threshold']:.2f} -> "
          f"{'HIGH' if prob >= artifact['threshold'] else 'low'} risk)")

    # SHAP attribution through the pipeline's preprocessing
    try:
        prep = artifact["raw_model"].named_steps["prep"]
        clf = artifact["raw_model"].named_steps["clf"]
        Xt = pd.DataFrame(prep.transform(row), columns=prep.get_feature_names_out())
        sh = RiskExplainer.explain_model_prediction(clf, Xt)
        print(f"SHAP base value {sh['base_value']:.2f}; top drivers: "
              f"{', '.join(sh['top_risk_drivers'] or ['none'])}")
        for c in sh["contributions"][:5]:
            print(f"    {c['feature']:<28} shap={c['shap_value']:+.3f}")
    except Exception as exc:  # SHAP optional in this report
        print(f"(SHAP attribution skipped: {exc})")


# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CardioCore cardiovascular digital twin demo")
    p.add_argument("--days", type=int, default=60, help="days to simulate (default 60)")
    p.add_argument("--profile", choices=list(PRESET_PROFILES), default="typical")
    p.add_argument("--scenario", choices=["stable", "improving", "declining"], default="stable")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=0.15, help="twin EWMA learning rate")
    p.add_argument("--ml", action="store_true", help="also train demo cohort model + SHAP explanation")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    twin = run_pipeline(args.days, args.profile, args.scenario, args.seed, args.lr)

    # rebuild identical history for clustering/ML frames
    history = simulate_history(args.days, PRESET_PROFILES[args.profile], scenario=args.scenario,
                               seed=args.seed, lab_interval_days=90)
    final_report(twin, history)
    if args.ml:
        ml_report(twin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
