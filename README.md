# CardioCoreEdge-EDI1
AI-Driven Personalized Cardiovascular Digital Twin for Continuous Cardiac Risk Assessment Using Multimodal Wearable Sensing

> 📘 **New to the project?** Read [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) —
> the complete guide to every concept, formula, algorithm, constant and
> technology used in this codebase (72-factor framework, biomarker equations,
> digital-twin mechanics, self-learning engine, clustering, ML pipeline, XAI).
Overview

CardioCore is an intelligent cardiovascular health monitoring system that combines wearable sensing, Artificial Intelligence (AI), and Digital Twin technology to provide continuous and personalised cardiac health assessment.

The system is designed to collect physiological data from a wearable device, analyse the information continuously, and maintain a real-time Digital Twin—a virtual representation of the user's cardiovascular health. By monitoring changes in health parameters over time, the system can identify abnormal patterns, estimate cardiovascular risk, and provide timely insights to support preventive healthcare.

Unlike traditional health monitoring systems that only display individual sensor readings, CardioCore focuses on understanding the overall cardiovascular condition of a person by analysing multiple health indicators together and updating the Digital Twin continuously.

## Architecture (core engine)

```
cardiovascular_twin/
├── main.py                    # end-to-end demo pipeline (CLI)
├── src/
│   ├── data_simulator.py      # synthetic wearable source + pluggable WearableDataSource interface
│   ├── lipid_calculator.py    # LDL (Friedewald / Sampson-NIH), non-HDL, ratios, categories
│   ├── factor_engine.py       # multimodal -> normalized weighted risk factors (0..1)
│   ├── digital_twin.py        # EWMA state assimilation, risk score, trend, alerts, persistence
│   ├── clustering.py          # KMeans + PCA daily-phenotype discovery & labelling
│   └── explainability.py      # factor attribution + SHAP model explanations
├── dashboard/                 # Streamlit dashboard (planned)
└── tests/                     # pytest suite (66 tests)
```

**Pipeline (documentation v1.0, Chapters 5-11):** daily multimodal observations +
profile → **72-factor framework** (`factors72.py`, doc Ch. 5) → **biomarker
estimation** TC/HDL/TG/LDL/CRP/D-Dimer with weighted formulas, derived values,
interactions, CTR & CVD scores (`biomarker_engine.py`, Ch. 6) → **biomarker
digital twin** with 11-step update cycle, personal baselines, trend, alerts
(`biomarker_twin.py`, Ch. 9) → **K=4 clustering** with PCA + confidence (Ch. 8) →
**dashboard** with gauge, radar, lipid table, 72-factor table, baselines,
cluster, trends, alerts and top-5 explanations per biomarker (Ch. 11).

Alongside: the wearable factor engine (`factor_engine.py`) with EWMA state
assimilation, adaptive weights and confidence intervals (`adaptive_engine.py`),
phenotype clustering and the real-data XGBoost model (holdout AUC 0.925).

## Patient portal (web app)

`web/` contains a production-style patient website — FastAPI backend + vanilla
SPA frontend (no build step):

- **Signup / login** — PBKDF2-hashed passwords, 7-day session tokens, SQLite storage
- **4-step onboarding wizard** — collects every patient-side input of the 72-factor
  framework (demographics, medical history, lifestyle, diet, smoking, optional known
  lab values) with live BMI, sliders and toggles
- **Results dashboard** — CVD risk gauge, thrombosis risk, 6 biomarker cards with
  clinical categories, pathway bars, radar, explainable top-5 drivers, graded alerts,
  60-day trajectory, searchable 72-factor table, what-if scenarios
- **Backend reuses the real engine** — the saved questionnaire builds a `UserProfile`,
  the simulator + `BiomarkerTwin` produce the estimates server-side

```bash
cd cardiovascular_twin/web && uvicorn server:app --port 8502   # http://localhost:8502
```

## Quick start

```bash
pip install -r cardiovascular_twin/requirements.txt

# demo: 60-day at-risk user on a declining trajectory (+ ML/SHAP layer)
cd cardiovascular_twin
python main.py --profile at_risk --scenario declining --ml

# interactive dashboard (gauge, radar, 72 factors, clusters, alerts, XAI)
streamlit run dashboard/app.py

# run the test suite (131 tests)
python -m pytest tests/ -q
```

Data source is abstracted behind `WearableDataSource` — a CSV/cloud-API
implementation can be plugged in later without touching the twin.

## Training the ML risk model (real data)

```bash
cd cardiovascular_twin
python train_model.py            # auto-downloads the UCI-combined dataset, trains + tunes + calibrates
python train_model.py --data data/raw/my_other_dataset.csv --target-col target
python main.py --profile at_risk --ml    # score a live twin state with the trained model (+ SHAP)
```

Pipeline: leakage-safe imputation (impossible zeros -> NaN -> fold-fitted
median) -> 80/20 stratified split -> LR / RF / XGBoost bake-off (5-fold CV
AUC) -> XGBoost grid search -> Youden-J threshold from out-of-fold
predictions -> isotonic probability calibration -> permutation importances.

**Results on the UCI-combined dataset (918 records):**
holdout ROC-AUC **0.925**, accuracy **0.90**, sensitivity **0.93**,
specificity **0.85** (see `models/metrics.json`).

### Where to get (more) training data

| Dataset | Records | Label | Access |
|---|---|---|---|
| UCI Heart Disease (combined: Cleveland+Hungarian+Switzerland+Statlog) | 918 | CAD diagnosis | auto-downloaded by `train_model.py` |
| Framingham Heart Study (Kaggle: `aasheesh200/framingham-heart-study-dataset`) | ~4,240 | 10-year CHD outcome | Kaggle (free account) |
| Cardiovascular Disease dataset (Kaggle: `sulianova/cardiovascular-disease-dataset`) | 70,000 | CVD presence | Kaggle (free account) |
| NHANES (CDC) | 100k+ | risk factors + mortality linkage | free download |
| MIMIC-IV / PhysioNet (WESAD, PPG-DaLiA) | varies | signals (ECG/PPG) for wearable features | PhysioNet credentialing |
| UK Biobank / All of Us | 100k+ | long-term outcomes + accelerometer | institutional application |

To use a Kaggle CSV: download it, drop it in `data/raw/`, then
`python train_model.py --data data/raw/<file>.csv --target-col <label column>`.
Column names differing from the UCI schema can be mapped in
`src/model_training.py` (`NUMERIC`/`CATEGORICAL`).

## Status

- [x] Core engine (`src/`) — simulator, lipids, factor engine, digital twin, clustering, explainability
- [x] Test suite (66 tests)
- [ ] Streamlit dashboard (`dashboard/app.py`)
- [ ] FastAPI service layer
- [x] ML risk model trained on real data (UCI-combined, 918 records, holdout AUC 0.925)
- [x] 72-factor framework + biomarker estimation engine (TC/HDL/TG/LDL/CRP/D-Dimer) per documentation v1.0
- [x] Biomarker digital twin (11-step cycle, baselines, K=4 clustering) + Streamlit dashboard (gauge/radar/tables/XAI)
- [ ] Real wearable-data ingestion (CSV export / cloud API)

## Disclaimer

Research/educational prototype. Not a medical device; output must not be used
for diagnosis or clinical decision-making.

