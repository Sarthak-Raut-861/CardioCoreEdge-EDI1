# CardioCoreEdge-EDI1 — Complete Concepts, Formulas & Algorithms Guide

> **Purpose of this document:** a single, self-contained reference so that anyone
> (developer, reviewer, examiner, collaborator) can understand **every concept,
> every formula, every algorithm, every constant, and every technology** used in
> this codebase — without reading the source code.
>
> Status: matches implementation as of commit `e33b966` · 131 passing tests.
>
> ⚠️ **RESEARCH PROTOTYPE** — all biomarker values are AI estimates. Not a
> medical device. Not for clinical diagnosis. Confirm with lab blood tests.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Repository Map — what every file does](#2-repository-map)
3. [Layer 1 — Data Simulation (synthetic wearable)](#3-layer-1--data-simulation)
4. [Layer 2 — The 72-Factor Framework](#4-layer-2--the-72-factor-framework)
5. [Layer 3 — Biomarker Estimation Engine](#5-layer-3--biomarker-estimation-engine)
6. [Layer 4 — The Biomarker Digital Twin](#6-layer-4--the-biomarker-digital-twin)
7. [Layer 5 — Patient Clustering (K-Means + PCA)](#7-layer-5--patient-clustering)
8. [Layer 6 — Wearable Factor Engine (the first twin)](#8-layer-6--wearable-factor-engine)
9. [Layer 7 — Self-Learning Adaptive Engine](#9-layer-7--self-learning-adaptive-engine)
10. [Layer 8 — Classical Lipid Calculator](#10-layer-8--classical-lipid-calculator)
11. [Layer 9 — Machine-Learning Risk Model](#11-layer-9--machine-learning-risk-model)
12. [Layer 10 — Explainability (XAI)](#12-layer-10--explainability-xai)
13. [The Dashboard](#13-the-dashboard)
14. [Complete Formula Reference (one page)](#14-complete-formula-reference)
15. [Technology Stack](#15-technology-stack)
16. [How to Run Everything](#16-how-to-run-everything)
17. [Design Decisions & Deviations from the Spec](#17-design-decisions--deviations)
18. [Known Limitations](#18-known-limitations)

---

## 1. The Big Picture

**Concept.** A *digital twin* is a continuously-updated virtual representation
of a real system — here, a person's cardiovascular health. Physical activity
generates physiological signals → wearable sensors measure them → software
converts them into **72 normalized factors** → a **biomarker engine** estimates
blood values that normally require a lab (TC, HDL, LDL, TG, CRP, D-Dimer) →
the **twin** tracks these over time, compares them to the person's *own learned
baseline*, computes risk scores, assigns a **risk cluster**, raises graded
**alerts**, and **explains** which factors drove each value. A **dashboard**
visualizes everything. The loop repeats with every new day of data.

```
 Person → Sensors → 72 Factors → Biomarkers (TC/HDL/TG/LDL/CRP/DD)
        → Twin (state, baselines, trend, alerts) → Clustering (K=4)
        → Risk Scores (CTR, CVD) → XAI explanations → Dashboard ↺
```

**Two parallel engines.** The system contains two complementary twins:

| | Biomarker Twin (`biomarker_twin.py`) | Wearable Twin (`digital_twin.py`) |
|---|---|---|
| Purpose | Estimate blood biomarkers from 72 factors | Score daily wearable physiology |
| Time constant | Slow (α=0.05, ≈20-day memory) | Fast (α=0.15, ≈6-day memory) |
| Output | TC/HDL/LDL/TG/CRP/DD + CTR + CVD 0-1 | 0-1 risk + category + CI |
| Learning | Personal baselines (z-scores) | Baselines + adaptive weights + CIs |

---

## 2. Repository Map

```
cardiovascular_twin/
├── main.py                  # CLI demo: simulate → both twins → reports (+ --ml)
├── train_model.py           # Trains the XGBoost risk model on real data
├── requirements.txt         # Pinned dependencies (Python ≤ 3.11)
├── src/
│   ├── data_simulator.py    # Synthetic wearable source (OU processes)
│   ├── factors72.py         # The 72-factor registry + derivation
│   ├── biomarker_engine.py  # Weighted biomarker formulas + risk scores
│   ├── biomarker_twin.py    # 11-step twin cycle + K-Means clustering
│   ├── digital_twin.py      # Wearable twin (EWMA state, alerts, trend)
│   ├── factor_engine.py     # 20-factor wearable scoring + anchors
│   ├── adaptive_engine.py   # Self-learning: baselines, Hedge weights, CIs
│   ├── lipid_calculator.py  # Friedewald / Sampson LDL + ratios
│   ├── clustering.py        # Daily phenotype clustering (KMeans+PCA)
│   ├── explainability.py    # Factor attribution + SHAP
│   └── model_training.py    # Leakage-safe ML training pipeline
├── dashboard/app.py         # Streamlit dashboard (11 sections)
├── models/                  # Trained artifacts (risk_model.joblib + metrics)
├── data/raw/                # Datasets (git-ignored; auto-downloaded)
└── tests/                   # 131 pytest tests
```

---

## 3. Layer 1 — Data Simulation

**File:** `src/data_simulator.py` · **Model: Ornstein–Uhlenbeck (OU) mean-reverting random walk**

Real sensors are not available yet, so each physiological channel (resting HR,
HRV, systolic/diastolic BP, sleep, deep-sleep %, steps, active minutes, nocturnal
SpO₂) is simulated as an OU process:

```
x(t+1) = x(t) + θ · (target(t) − x(t)) + ε,      ε ~ N(0, σ)
```

| Parameter | Meaning | Values |
|---|---|---|
| θ | mean-reversion strength | 0.25 (HR/HRV/BP), 0.35 (sleep/activity) |
| σ | daily noise (daily *aggregates* average sensor noise out) | RHR 1.2 bpm · HRV 3.0 ms · SBP 2.8 mmHg · DBP 2.2 mmHg · sleep 0.6 h |
| target(t) | baseline × (1 + drift·t) + weekend effect | drifts below |

**Scenario drift** (multiplicative per day): applied to the target so long-run
trajectories are realistic:

| Scenario | RHR | HRV | SBP | DBP | sleep | steps |
|---|---|---|---|---|---|---|
| stable | 0 | 0 | 0 | 0 | 0 | 0 |
| improving | −0.27%/d | +0.48%/d | −0.21%/d | −0.17%/d | +0.24%/d | +0.36%/d |
| declining | +0.33%/d | −0.54%/d | +0.26%/d | +0.20%/d | −0.29%/d | −0.38%/d |

Weekend effect: weekends add +2 bpm RHR, −3 ms HRV, +1.5 mmHg SBP, −0.5 h sleep,
−800 steps to the target. **Labs** (TC/HDL/TG) are drawn every 90 days.
Spo₂ dips per night ~ Poisson(λ = 0.8·(97.5 − SpO₂)).

Three preset profiles (healthy / typical / at_risk) define demographics,
history, diet, smoking, and physiological baselines.

---

## 4. Layer 2 — The 72-Factor Framework

**File:** `src/factors72.py` · **Spec: project documentation v1.0, Chapter 5**

Every factor is declared once with: ID (F1…F72), name, source sensor, category,
raw range, and per-biomarker weights `{biomarker: (weight, direction)}` where
direction **+1 = risk-increasing** and **−1 = protective**.

**Normalization (doc §6.1):**

```
F_norm = (F_raw − F_min) / (F_max − F_min),  clipped to [0, 1]
```

**Factor groups and examples of documented weights:**

| Group | IDs | Examples (doc weights) |
|---|---|---|
| NIR spectroscopy | F1–F8 | F1 NIR-1040nm → TC w=18 · F7 PCA-1 → TC w=16 · F2 NIR-1680nm → TG w=15 |
| PPG morphology | F9–F15 | PWV, AIx, stiffness, reflection, SDPPG ratios |
| Blood pressure | F16–F20 | F16 SBP, F18 Pulse Pressure → TC w=14 (strong) |
| HRV | F21–F25 | F21 SDNN (protective), F22 RMSSD (protective), F23 LF/HF → TG |
| SpO₂ | F26–F29 | resting/activity/recovery/variability |
| ECG | F30–F33 | F30 AF-probability → DD w=18 · ST-deviation · QTc |
| Temp / GSR / BioZ | F34–F38 | skin-temp elevation → CRP · body-fat → CRP |
| Activity | F39–F42 | F39 steps → HDL protective w=10 · F40 sedentary → DD w=10 |
| Sleep / stress | F43–F47 | nocturnal SpO₂ pattern · composite stress · cortisol proxy |
| Demographics | F48–F54 | F48 age (DD w=13, TC 12) · F50 BMI (CRP 12, TG 12) · F51 waist 10 |
| Medical history | F55–F60 | F55 diabetes 10 · F56 chol-history TC 14 · F58 clot-history DD 15 |
| Diet / smoking | F61–F67 | F61 sat-fat 10 · F62 sugar TG 10 · F66 smoking 10 |
| Derived vascular | F68–F72 | vascular age gap · F72 endothelial dysfunction 10 |

**Derivation of unmeasured sensors (prototype).** Sensors we don't simulate yet
(NIR, PPG morphology, ECG morphology) are synthesized *deterministically* from
physiologically-linked channels so the full pipeline runs end-to-end:

- **NIR lipid signal** (F1–F8): a lipid propensity
  `prop = 0.6·static + 0.4·dynamic`, where *static* comes from the profile's
  lipid baselines and *dynamic* is a deviation index of current physiology:

```
dyn = clip( 0.5 + 0.30·(RHR−base_RHR)/18 + 0.30·(base_HRV−HRV)/20
          + 0.25·(SBP−base_SBP)/25 + 0.10·(7.3−sleep)/2
          + 0.05·(8500−steps)/5000 , 0, 1 )
```
  (A declining physiology therefore raises the NIR lipid signal, as real NIR
  would track tissue lipid content.)
- **PWV / stiffness** (F9–F14): functions of age, SBP and fitness
  (arteries stiffen with age + pressure, loosen with fitness).
- **SDNN ≈ 1.35 × RMSSD + 6** (F21); **pNN50 ≈ 50·(RMSSD/75)²** (F24) — standard
  HRV inter-relations.
- **F20 BP-variability** = rolling 21-day SD of SBP ÷ 10 (0–1).
- Deterministic per-day noise uses a seeded RNG keyed by `(day, salt)` so runs
  are exactly reproducible.

---

## 5. Layer 3 — Biomarker Estimation Engine

**File:** `src/biomarker_engine.py` · **Spec: doc Chapter 6**

### 5.1 Core formulas (doc §6.3–6.9)

```
Weighted_Sum(b) = Σᵢ  Fᵢ_norm · Wᵢ,b · Directionᵢ,b          (+1 risk / −1 protective)

TC      = 150 + Weighted_Sum(TC)  · 0.486        → clipped to [150, 320] mg/dL
HDL     =  80 − Weighted_Sum(HDL) · 0.45         → clipped to [20, 80]   mg/dL
TG      =  80 + Weighted_Sum(TG)  · 1.4          → clipped to [80, 500]  mg/dL
CRP     = 0.2 + Weighted_Sum(CRP) · 0.085        → clipped to [0.2, 10]  mg/L
D-Dimer = 0.1 + Weighted_Sum(DD) · 0.022 + 1.2·CRP_norm   → clipped to [0.1, 3.0] mg/L
LDL     = TC − HDL − TG/5   (TG < 400)                       # Friedewald
          TC − HDL − TG/6   (TG ≥ 400)                       # Martin-Hopkins style
```

### 5.2 Range calibration (implementation addition)

The documentation's constants assume its illustrative top-weights; the full
72-factor weight matrix alone cannot span the stated ranges. Each biomarker
therefore gets a one-time auto-calibration factor computed from the weight
matrix, assuming a realistic worst-case profile (risk factors at 0.85,
protective at 0.15) reaching 92 % of the range:

```
calib(b) = 0.92 · (max_b − min_b) / (doc_scaling_b · net_max_b)
net_max_b = 0.85·Σ(positive weights) − 0.15·Σ(negative weights)
```
Relative factor importance (including all documented top-weights and ratios)
is preserved exactly; only the overall gain is calibrated.

### 5.3 Derived lipid values (doc §6.7)

```
VLDL = TG/5        Non-HDL = TC − HDL
TC/HDL ratio       LDL/HDL ratio        AIP = log₁₀(TG/HDL)
```

### 5.4 Cross-biomarker interactions (doc §6.11)

| Interaction | Rule |
|---|---|
| CRP adjusts TC risk context | CRP < 1 → ×1.0 · 1–3 → ×1.2 · > 3 → ×1.5 |
| HDL modifies CRP | HDL ≥ 60 → ×0.7 · 40–60 → ×1.0 · < 40 → ×1.3 |
| D-Dimer rises with CRP | + 1.2 × CRP_norm (inflammation activates coagulation) |

### 5.5 Risk scores (doc §6.11)

All biomarkers are risk-normalized to 0–1 (e.g. `TC_norm=(TC−150)/170`,
`HDL_risk_norm = 1 − (HDL−20)/60`):

```
CTR (Coronary Thrombosis Risk) = 0.35·LDL_n + 0.40·DD_n + 0.25·CRP_n
      Low < 0.33 · Moderate 0.33–0.66 · High > 0.66

CVD (Overall risk) = 0.20·TC_n + 0.20·LDL_n + 0.10·(1−HDL_n)
                   + 0.10·TG_n + 0.20·CRP_n + 0.20·DD_n
      Low < 0.25 · Moderate 0.25–0.50 · High 0.50–0.75 · Very High > 0.75
```

**Six pathway risks** (doc §5.1): lipid = mean(TC_n, LDL_n, 1−HDL_n);
inflammation = CRP_n; thrombosis = DD_n; hemodynamic = mean(F16, F18, F19);
autonomic = 1 − mean(F21, F22, F24); metabolic = mean(F50, F51, F53, F55).

### 5.6 Clinical categories (doc §6.10)

| Biomarker | Categories |
|---|---|
| TC | Desirable <200 · Borderline 200–239 · High ≥240 mg/dL |
| HDL | Low <40 · Normal 40–59 · Protective ≥60 mg/dL |
| LDL | Optimal <100 · Near 100–129 · Borderline 130–159 · High 160–189 · Very High ≥190 |
| TG | Normal <150 · Borderline 150–199 · High 200–499 · Very High ≥500 |
| CRP | Low <1.0 · Moderate 1–3 · High 3–10 · Acute >10 mg/L |
| D-Dimer | Normal <0.5 · Mild 0.5–1 · Moderate 1–2 · High >2 mg/L |

---

## 6. Layer 4 — The Biomarker Digital Twin

**File:** `src/biomarker_twin.py` · **Spec: doc Chapter 9**

**11-step update cycle** (§9.2): ① receive readings → ② normalize → ③ compute
biomarkers → ④ update state → ⑤ append to rolling 100-reading history →
⑥ baseline z-scores → ⑦ risk score → ⑧ trend → ⑨ cluster reassignment →
⑩ alerts → ⑪ dashboard.

**Factor smoothing.** The 72-factor vector is EWMA-smoothed before computing
biomarkers (α = 0.05 ⇒ ≈ 20-day effective memory — lipids move slowly):

```
smoothed(k) = smoothed(k) + 0.05 · (raw(k) − smoothed(k))
```

**Personal baselines (§9.3).** Welford running mean/SD per biomarker; minimum
5 readings; deviations via z-score; significant |Z| > 2, critical |Z| > 3;
also % change vs baseline mean.

**Trend.** Doc's "compare last 3 risk scores" is noise-dominated on daily
estimates, so we keep the semantics but compare the mean of the last 14 days
vs the previous 14 (dead-band ±0.010) → Increasing / Stable / Decreasing.
Validated on 32/32 scenario×seed runs.

**Alerts (graded, deduplicated).** Threshold rules: TC≥240 · LDL≥160/190 ·
HDL<40 · TG≥200 · CRP>3 (warning) / >10 (critical) · DD>1 (warning) / >2
(critical) · CTR>0.66 · baseline shifts (per biomarker). Non-critical codes
are suppressed on consecutive days; **critical alerts intentionally repeat
daily until resolved**.

---

## 7. Layer 5 — Patient Clustering

**File:** `src/biomarker_twin.py` (class `BiomarkerClusterer`) · **Spec: doc Chapter 8**

- **Algorithm:** K-Means, K = 4 (doc: elbow method on K=2..8), `n_init=10`,
  on the 6 biomarkers, after `StandardScaler` (z-score) preprocessing.
- **Cohort:** 120–200 synthetic patients sampled along a health axis
  `t ~ U(0,1)` with the same biomarker formulas, to position the single
  monitored user within a population.
- **Labels:** clusters sorted by mean CVD score → Low / Moderate / High /
  Very High Risk.
- **PCA** to 2 components for visualization.
- **Cluster confidence (§8.5):** `confidence = 1 − d_own / max(d_any)`.
- **Validation metrics implemented:** Silhouette score (target > 0.4) and
  Davies–Bouldin index (target < 1.0) via the phenotype clusterer.

A second clusterer (`src/clustering.py`, `PhenotypeClusterer`) clusters
*daily wearable states* (K searched 2..5 by silhouette) and names phenotypes
heuristically from z-deviations (e.g. "sympathetic-dominant", "sedentary +
poor sleep").

---

## 8. Layer 6 — Wearable Factor Engine

**File:** `src/factor_engine.py` + `src/digital_twin.py`

**20 factors** (10 wearable + 5 lab + 5 demographic), each scored 0–1 by
**piecewise-linear anchor tables** (reference-range interpolation). Example
anchors (value → risk points):

```
Resting HR :  40→0.00  50→0.03  60→0.12  70→0.32  80→0.58  90→0.82 100→1.00
HRV RMSSD  :  15→1.00  20→0.92  30→0.75  40→0.52  55→0.28  70→0.12  90→0.02
Systolic BP:  95→0.00 110→0.04 120→0.14 130→0.36 140→0.56 160→0.80 180→1.00
```
U-shaped functions for sleep duration (7–9 h optimal), deep-sleep %,
BMI; blended activity score = 0.6·steps + 0.4·active-minutes.

**EWMA state assimilation** (α = 0.15 default):

```
state(k) = state(k) + α · (observation(k) − state(k))
```

**Composite risk** = weighted mean of factor scores (weights: SBP 1.6,
smoking 1.6, RHR 1.3, HRV 1.25, LDL 1.4 …), blended
`0.25·demographic_prior + 0.75·current_physiology`. Categories:
Low <0.25 · Moderate <0.45 · High <0.65 · Very High ≥0.65.

**Trend:** least-squares slope of the (3-point-smoothed) full risk history,
labels at |slope| ≥ 0.0008/day.

---

## 9. Layer 7 — Self-Learning Adaptive Engine

**File:** `src/adaptive_engine.py`

**(a) Personal baselines — Welford's online algorithm:**

```
n ← n+1;  δ = x − mean;  mean ← mean + δ/n;  M₂ ← M₂ + δ·(x − mean)
variance = M₂ / (n−1)
```
Warm-up 14 days (wearable channels). **Anomaly detection:** leave-one-out
z-score `z = (x − mean_hist)/SD_hist` with guards — ≥10 days of history AND
trailing coefficient of variation ≥ 3 % (avoids huge z-scores on
near-constant channels). Flag at |z| > 2.5.

**(b) Adaptive factor weights — Hedge / multiplicative-weights online learning:**

Self-supervised target = the *next-day direction* of the composite risk.

```
rewardᵢ = +1 if sign(Δscoreᵢ) == sign(Δcomposite_next) else −1
multᵢ ← clip( multᵢ · e^(η·rewardᵢ), 0.5, 2.0 ),   η = 0.08
```
Multipliers activate after 10 observations; a precision term favours stable
factors: `effective = multᵢ · (0.5 + 0.5·stability)`,
`stability = 1/(1 + CV of trailing 14 scores)`.

**(c) Confidence intervals:** the reported risk carries a 95 % CI
`± 1.96·SD/√n` over the trailing 14 daily scores — width shrinks ≈ 1/√n as
evidence accumulates ("precision gain %" vs early days is displayed).

---

## 10. Layer 8 — Classical Lipid Calculator

**File:** `src/lipid_calculator.py`

- **Friedewald LDL** = TC − HDL − TG/5 (invalid TG ≥ 400 mg/dL).
- **Sampson / NIH equation 2** (valid TG < 800; verified against CDC NHANES
  documentation):
```
LDL = TC/0.948 − HDL/0.971 − ( TG/8.56 + TG·(TC−HDL)/2140 − TG²/16100 ) − 9.44
```
- Preference order: direct LDL > Sampson > Friedewald.
- Non-HDL, TC/HDL, TG/HDL ratios, **AIP = log₁₀(TG/HDL)**.
- Guideline categories (ATP-III/ESC bands) and a 0–1 lipid risk score from
  piecewise anchor tables.

---

## 11. Layer 9 — Machine-Learning Risk Model

**Files:** `src/model_training.py`, `train_model.py`

**Dataset:** UCI-combined Heart Disease (Cleveland+Hungarian+Switzerland+
Statlog; 918 records, 11 clinical features, CAD diagnosis label; auto-downloaded
from a public mirror).

**Leakage-safe pipeline:**
1. Impossible zeros (RestingBP/Cholesterol = 0) → NaN → median imputer fitted
   inside the pipeline (train-fold only).
2. Stratified 80/20 split; 5-fold stratified CV.
3. Bake-off: Logistic Regression (class_weight=balanced) vs Random Forest vs
   XGBoost, ranked by CV ROC-AUC.
4. GridSearchCV on XGBoost (n_estimators, max_depth, learning_rate,
   subsample). Best: lr 0.03, depth 3, 200 trees, subsample 0.8.
5. **Decision threshold by Youden's J** (sensitivity + specificity − 1) on
   *out-of-fold training* predictions (never on the test set) → 0.524.
6. **Isotonic probability calibration** (CalibratedClassifierCV, 5-fold);
   Brier score reported before/after.
7. Metrics on the untouched holdout: **ROC-AUC 0.925 · accuracy 0.90 ·
   sensitivity 0.93 · specificity 0.85 · PR-AUC 0.93**.
8. Permutation importance on the holdout for global interpretation.

**Anomaly detection (doc §7.5):** the twin layer implements personal-baseline
anomaly scoring (statistical); Isolation Forest is the documented production
alternative.

---

## 12. Layer 10 — Explainability (XAI)

**File:** `src/explainability.py` + contribution payloads in both twins

**(a) Rule/weight attribution (always available):**
`contribution_i = F_i · W_i · direction_i`, ranked; shares normalized to 100 %;
natural-language narrative assembled from the top drivers.

**(b) SHAP (doc §10.2)** — `shap.TreeExplainer` on the XGBoost model:
per-feature φ-values for an individual prediction
`φ_i = Σ_S [|S|!(|F|−|S|−1)!/|F|!]·[f(S∪{i}) − f(S)]`, run through the
preprocessing pipeline's transformed feature space; positive φ raises the
risk estimate. Graceful fallback to weight-based contributions when no model
is attached.

The dashboard shows top-5 contributors for TC, HDL, TG, CRP, D-Dimer, plus
SHAP drivers for the ML model.

---

## 13. The Dashboard

**File:** `dashboard/app.py` (Streamlit) · **Spec: doc Chapter 11 — all 11 sections**

| # | Section | Implementation |
|---|---|---|
| 1 | Patient header | name, demographics, twin status, readings count |
| 2 | Overall risk | Plotly gauge 0–100 + category + trend + CTR + pathway bars |
| 3 | Lipid profile | full table with values, categories, normal ranges, colors |
| 4 | Inflammatory markers | CRP / D-Dimer / adjusted-CRP + interaction readouts |
| 5 | Radar | 6-axis Scatterpolar (each biomarker risk-normalized 0–1) |
| 6 | Cluster | label, confidence %, PCA scatter, patient starred |
| 7 | Baseline comparison | per-biomarker mean, SD, current, z, % change, significance |
| 8 | Trends | time series: lipids, CRP/DD, CVD risk with band shading |
| 9 | Alerts | color-graded INFO/WARNING/CRITICAL, deduplicated |
| 10 | AI explanation | top-5 factors per biomarker with signed contributions |
| 11 | Disclaimer | red banner, always visible |
| + | 72-factor table | every factor: ID, source, range, normalized value, direction, weights |
| + | Wearable twin | adaptive engine scores, learning stats, trajectory |

**Color coding (§11.2):** green normal · yellow borderline · orange high ·
red very-high/critical · gray insufficient.

---

## 14. Complete Formula Reference

```
# Simulation (OU)
x(t+1) = x(t) + θ(target − x(t)) + N(0, σ)

# Normalization
F_norm = (F − F_min)/(F_max − F_min)                      clipped [0,1]

# Biomarkers
TC   = 150 + Σ(F·W·d)|TC  ·0.486·calib                    ∈[150,320] mg/dL
HDL  =  80 − Σ(F·W·d)|HDL ·0.45 ·calib                    ∈[20,80]   mg/dL
TG   =  80 + Σ(F·W·d)|TG  ·1.4 ·calib                     ∈[80,500]  mg/dL
CRP  = 0.2 + Σ(F·W·d)|CRP ·0.085·calib                    ∈[0.2,10]  mg/L
DD   = 0.1 + Σ(F·W·d)|DD  ·0.022·calib + 1.2·CRP_norm     ∈[0.1,3]   mg/L
LDL  = TC − HDL − TG/5   (TG<400)   |   TC − HDL − TG/6 (TG≥400)
VLDL = TG/5    Non-HDL = TC−HDL    AIP = log₁₀(TG/HDL)

# Interactions
CRP_mult = 1.0/1.2/1.5 by CRP band        HDL_prot = 0.7/1.0/1.3 by HDL band

# Risk scores
CTR = 0.35·LDL_n + 0.40·DD_n + 0.25·CRP_n
CVD = 0.20·TC_n + 0.20·LDL_n + 0.10·(1−HDL_n) + 0.10·TG_n + 0.20·CRP_n + 0.20·DD_n

# Twin mechanics
EWMA:      s ← s + α(x − s)               (biomarker α=0.05, wearable α=0.15)
Baseline:  z = (x − mean)/SD  (Welford)   sig |z|>2, crit |z|>3 (bio) / 2.5 (wear)
Hedge:     mᵢ ← clip(mᵢ·e^(0.08·rᵢ), 0.5, 2.0)
Stability: 1/(1+CV₁₄)      effective weight = m·(0.5+0.5·stability)
CI95:      ±1.96·SD/√n
Trend:     mean(last14) − mean(prev14) vs ±0.010
Cluster:   confidence = 1 − d_own/max(d_any)

# Classical LDL equations
Friedewald: TC − HDL − TG/5
Sampson:    TC/0.948 − HDL/0.971 − (TG/8.56 + TG(TC−HDL)/2140 − TG²/16100) − 9.44

# ML layer
Youden J*: argmax_t (sens(t)+spec(t)−1)   (*out-of-fold train predictions)
Calibration: isotonic (Platt alternative)
```

---

## 15. Technology Stack

| Technology | Version (pinned) | Used for |
|---|---|---|
| Python | 3.9–3.11 | language |
| NumPy | 1.24.3 | numerics, RNG, regression |
| pandas | 2.0.2 | tables, history, styling |
| scikit-learn | 1.3.0 | KMeans, PCA, StandardScaler, calibration, metrics |
| XGBoost | 1.7.6 | gradient-boosted risk model (classifier) |
| SHAP | 0.42.1 | TreeExplainer model attribution |
| Streamlit | 1.24.0 | dashboard |
| Plotly | 5.15.0 | gauge, radar, scatter, time series |
| Matplotlib | 3.7.2 | pandas table gradients |
| SciPy / joblib | pinned | sklearn dependency, model persistence |
| pytest | — | 131 tests |

Documented-but-not-yet-wired (future): FastAPI service layer, paho-MQTT /
pyserial / Bleak sensor ingestion, NeuroKit2/biosppy biosignal processing,
SQLite/SQLAlchemy storage.

---

## 16. How to Run Everything

```bash
pip install -r cardiovascular_twin/requirements.txt      # Python ≤ 3.11
cd cardiovascular_twin

python -m pytest tests/ -q                               # 131 tests
python main.py --profile at_risk --scenario declining --ml   # CLI demo + ML
python train_model.py                                    # (re)train ML model
streamlit run dashboard/app.py                           # interactive dashboard
```

Useful flags: `--profile healthy|typical|at_risk`, `--scenario
stable|improving|declining`, `--days N`, `--seed N`, `--lr EWMA`,
`--no-tune` (train_model).

---

## 17. Design Decisions & Deviations

All deviations from the source documentation are deliberate, documented in
code comments, and made for statistical honesty:

1. **Biomarker weight calibration** (§5.2 above) — doc's constants kept,
   gain auto-calibrated so documented biomarker ranges are reachable.
2. **Trend via 14-vs-14-day means** instead of "last 3 scores" — 3 points
   are noise-dominated; validated 32/32 across scenarios and seeds.
3. **Factor EWMA smoothing α=0.05** — lipids move slowly; unsmoothed daily
   estimates whipsaw ±15 mg/dL.
4. **HDL weight rebalancing** (e.g. F39 10→8, F49 6→10) — the doc's literal
   HDL table cannot produce realistic mid-range HDL for typical males.
5. **Critical alerts repeat daily** (dedupe only applies to non-critical).
6. **Synthetic derivations** for unmeasured sensors (NIR/PPG/ECG) are
   documented inline and replaceable by real sensor drivers.

---

## 18. Known Limitations

- **Estimation ≠ measurement.** NIR-derived biomarkers carry ±15–25 mg/dL
  error margins in principle; this prototype's numbers are synthetic-model
  outputs, not tissue measurements.
- Skin tone, hydration and motion artifacts affect real NIR; not modeled.
- Single-user demo; no database persistence yet (snapshots serialize to JSON-
  compatible dicts).
- The ML risk model is trained on a *diagnosis* dataset (UCI), not on wearable
  trajectories; it anchors the ML layer demonstration.
- Educational/research use only — **not a medical device**.
