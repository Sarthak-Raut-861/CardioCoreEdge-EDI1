# CardioCoreEdge-EDI1
AI-Driven Personalized Cardiovascular Digital Twin for Continuous Cardiac Risk Assessment Using Multimodal Wearable Sensing
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

**Pipeline:** daily wearable aggregates + sparse labs + demographics → `FactorEngine`
(piecewise-linear normalization, weighted composite) → `DigitalTwin` (EWMA state
assimilation, 0..1 risk score & category, trend, safety-net alerts) →
`PhenotypeClusterer` (recurring physiological states) → `RiskExplainer`
(ranked drivers, narrative, SHAP when an XGBoost model is attached).

## Quick start

```bash
pip install -r cardiovascular_twin/requirements.txt

# demo: 60-day at-risk user on a declining trajectory (+ ML/SHAP layer)
cd cardiovascular_twin
python main.py --profile at_risk --scenario declining --ml

# other options
python main.py --profile healthy --scenario improving --days 90
python main.py --help

# run the test suite
python -m pytest tests/ -q
```

Data source is abstracted behind `WearableDataSource` — a CSV/cloud-API
implementation can be plugged in later without touching the twin.

## Status

- [x] Core engine (`src/`) — simulator, lipids, factor engine, digital twin, clustering, explainability
- [x] Test suite (66 tests)
- [ ] Streamlit dashboard (`dashboard/app.py`)
- [ ] FastAPI service layer
- [ ] Real wearable-data ingestion (CSV export / cloud API)

## Disclaimer

Research/educational prototype. Not a medical device; output must not be used
for diagnosis or clinical decision-making.

