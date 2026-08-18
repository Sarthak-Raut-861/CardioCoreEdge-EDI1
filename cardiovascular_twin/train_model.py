#!/usr/bin/env python3
"""
train_model.py — train the CardioCore cardiac-risk model
=========================================================
Usage
-----
    python train_model.py                      # UCI-combined (auto-download), tuned
    python train_model.py --data my.csv --target-col HeartDisease
    python train_model.py --no-tune            # fast bake-off without grid search

Outputs (models/):
    risk_model.joblib  – trained + calibrated model bundle
    metrics.json       – CV scores, holdout metrics, importances

DISCLAIMER: research/educational prototype – not a medical device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.model_training import (
    TARGET_COL,
    ensure_dataset,
    load_dataset,
    save_artifact,
    train_risk_model,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the CardioCore risk model")
    here = Path(__file__).parent
    p.add_argument("--data", default=str(here / "data/raw/heart_uci_combined.csv"),
                   help="CSV path (default: UCI-combined, auto-downloaded)")
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--out", default=str(here / "models"))
    p.add_argument("--no-tune", dest="tune", action="store_false",
                   help="skip XGBoost grid search (faster)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    path = ensure_dataset(args.data)
    df = load_dataset(path)
    if args.target_col != TARGET_COL:
        df = df.rename(columns={args.target_col: TARGET_COL})
    print(f"[data] {path} -> {df.shape[0]} rows, "
          f"{df[TARGET_COL].mean():.1%} positive class")

    artifact = train_risk_model(df, tune=args.tune, random_state=args.seed)

    print("\n================ TRAINING REPORT ================")
    print(f"Best model        : {artifact['model_name']}")
    if artifact["tuning"].get("grid_used"):
        print(f"Best params       : {artifact['tuning']['best_params']} "
              f"(CV AUC {artifact['tuning']['best_cv_auc']})")
    print(f"Decision threshold: {artifact['threshold']:.3f} (Youden-J, out-of-fold)")
    h = artifact["holdout_calibrated"]
    print(f"Holdout (calibrated): AUC {h['roc_auc']} | PR-AUC {h['pr_auc']} | "
          f"acc {h['accuracy']} | sens {h['sensitivity']} | spec {h['specificity']} | "
          f"F1 {h['f1']} | Brier {h['brier']}")
    print(f"Confusion (tn/fp/fn/tp): {h['confusion']}")
    print("\nTop features (permutation importance on holdout):")
    for f in artifact["importances"][:8]:
        print(f"  - {f['feature']:<16} {f['importance']:+.4f}")

    out = save_artifact(artifact, args.out, training_df=df)
    print(f"\nSaved model bundle -> {out}")
    print(f"Saved metrics      -> {Path(args.out) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
