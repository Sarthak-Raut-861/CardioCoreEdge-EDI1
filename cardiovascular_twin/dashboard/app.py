"""
dashboard/app.py — CardioCore digital twin dashboard (Streamlit)
================================================================
Numbers-first dashboard for the self-learning cardiovascular twin.

Run:
    streamlit run dashboard/app.py

Tabs
----
1. Overview    – headline scores: risk, category, trend, 95% CI, precision gain
2. Factors     – every factor with value, 0-1 score, weight, learned multiplier,
                 effective weight and contribution share
3. Learning    – personalized baselines, adaptive weight evolution, CI shrinkage
4. Trajectory  – risk history + raw physiological channels
5. Phenotypes  – daily-state clustering (KMeans + PCA)
6. ML model    – trained risk-model probability for the current twin state

Research/educational prototype – NOT a medical device.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.clustering import PhenotypeClusterer  # noqa: E402
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource, simulate_history  # noqa: E402
from src.digital_twin import DISCLAIMER, DigitalTwin  # noqa: E402
from src.explainability import RiskExplainer  # noqa: E402

st.set_page_config(page_title="CardioCore Digital Twin", page_icon="❤️", layout="wide")


# --------------------------------------------------------------------------- #
# Simulation (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running digital-twin simulation ...")
def run_twin(profile_name: str, scenario: str, days: int, seed: int, lr: float):
    profile = PRESET_PROFILES[profile_name]
    source = SimulatedWearableSource(profile, scenario=scenario, seed=seed, lab_interval_days=90)
    twin = DigitalTwin(profile.to_dict(), learning_rate=lr)
    updates = [twin.assimilate(obs, obs.get("labs")) for obs in source.next_days(days)]
    history = simulate_history(days, profile, scenario=scenario, seed=seed, lab_interval_days=90)
    return twin.snapshot(), [u.to_dict() for u in updates], history


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = ROOT / "models" / "risk_model.joblib"
    if not path.exists():
        return None, None
    from src.model_training import load_artifact

    artifact = load_artifact(path)
    metrics = None
    mpath = ROOT / "models" / "metrics.json"
    if mpath.exists():
        import json

        metrics = json.loads(mpath.read_text())
    return artifact, metrics


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("❤️ CardioCore")
st.sidebar.caption("Self-learning cardiovascular digital twin")

st.sidebar.subheader("Simulation")
profile_name = st.sidebar.selectbox("User profile", list(PRESET_PROFILES), index=1)
scenario = st.sidebar.selectbox("Scenario", ["stable", "improving", "declining"], index=2)
days = st.sidebar.slider("Days monitored", 30, 180, 60, 10)
seed = st.sidebar.number_input("Random seed", 0, 999, 42)
lr = st.sidebar.slider("Twin learning rate (EWMA)", 0.05, 0.4, 0.15, 0.05)

snap, updates, history = run_twin(profile_name, scenario, int(days), int(seed), float(lr))
twin = DigitalTwin.from_dict(snap)

st.sidebar.success(f"{days} days simulated · {twin.days_seen} assimilated")
st.sidebar.markdown(f"<small>{DISCLAIMER}</small>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
profile = PRESET_PROFILES[profile_name]
st.title(f"Digital Twin — {profile.name}")
st.caption(f"age {profile.age} · {profile.sex} · BMI {profile.bmi} · smoker {profile.smoker} · "
           f"diabetic {profile.diabetic} · family history {profile.family_history} · scenario: {scenario}")

tab_overview, tab_factors, tab_learning, tab_traj, tab_pheno, tab_model = st.tabs(
    ["📊 Overview", "🧮 Factors", "🧠 Learning", "📈 Trajectory", "🧬 Phenotypes", "🤖 ML model"]
)

# --------------------------------------------------------------------------- #
# 1) Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    learning = twin.learning_summary()
    last = updates[-1]
    ci = last.get("ci95") or (None, None)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk score (0-100)", f"{twin.risk_score * 100:.1f}",
                f"95% CI {ci[0] * 100:.0f}–{ci[1] * 100:.0f}" if ci[0] is not None else None)
    col2.metric("Category", twin.risk_category, twin.risk_trend())
    col3.metric("Precision gain (CI shrinkage)", f"{learning.get('precision_gain_pct', 0):+.1f}%",
                f"CI width ±{learning.get('ci_width', 0) / 2 * 100:.1f} pts")
    col4.metric("Personalized baselines", f"{learning.get('warm_baselines', 0)}/{learning.get('channels_tracked', 0)}",
                f"after {twin.days_seen} days")

    st.subheader("Final numbers")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Days monitored", twin.days_seen)
    m2.metric("Factors evaluated", twin._compute_risk().n_factors)
    m3.metric("Data coverage", f"{twin._compute_risk().coverage * 100:.0f}%")
    m4.metric("Alerts raised", len(twin.alerts_raised))

    # top drivers table
    st.subheader("Top risk drivers (weighted contribution)")
    comp = twin._compute_risk()
    table = RiskExplainer.explain_composite(comp)
    mults = comp.multipliers or {}
    table["multiplier"] = table["factor"].map(lambda n: None)
    name_to_mult = {f.name: mults.get(f.key) for f in comp.breakdown}
    table["multiplier"] = table["factor"].map(name_to_mult)
    st.dataframe(table.head(12), use_container_width=True, height=420)

    st.subheader("Alerts")
    if twin.alerts_raised:
        alerts_df = pd.DataFrame(twin.alerts_raised[-15:])
        st.dataframe(alerts_df[["day", "level", "code", "message"]], use_container_width=True)
    else:
        st.success("No alerts raised during the monitored period.")

    st.info(RiskExplainer.narrative(comp))

# --------------------------------------------------------------------------- #
# 2) Factors
# --------------------------------------------------------------------------- #
with tab_factors:
    st.subheader("All factors — full calculation table")
    comp = twin._compute_risk()
    mults = comp.multipliers or {}
    rows = []
    total_contrib = sum(f.weight * mults.get(f.key, 1.0) * f.score for f in comp.breakdown if f.value is not None) or 1.0
    for f in comp.breakdown:
        m = mults.get(f.key, 1.0)
        eff_w = f.weight * m
        contrib = eff_w * f.score
        rows.append({
            "factor": f.name, "key": f.key, "modality": f.modality,
            "value": f.value, "score (0-1)": f.score, "base weight": f.weight,
            "learned multiplier": m, "effective weight": round(eff_w, 3),
            "contribution": round(contrib, 4), "share %": round(100 * contrib / total_contrib, 1),
            "reference": f.reference,
        })
    fdf = pd.DataFrame(rows)
    modality_filter = st.multiselect("Filter by modality", ["wearable", "lab", "demographic"],
                                     default=["wearable", "lab", "demographic"])
    fdf_view = fdf[fdf["modality"].isin(modality_filter)] if modality_filter else fdf
    st.caption(f"Showing {len(fdf_view)} factors · composite risk = Σ(effective weight × score) / Σ(effective weight) "
               f"= **{comp.score:.3f}** (wearable/lab part), blended with demographic prior → **{twin.risk_score:.3f}**")
    st.dataframe(fdf_view, use_container_width=True, height=520)

    st.subheader("Contribution shares (top 15)")
    top = fdf.nlargest(15, "contribution").sort_values("contribution")
    fig = go.Figure(go.Bar(
        x=top["contribution"], y=top["factor"], orientation="h",
        text=top["share %"].map(lambda v: f"{v:.1f}%"), textposition="outside",
        marker_color=np.where(top["score (0-1)"] >= 0.5, "#d62728", "#2ca02c"),
    ))
    fig.update_layout(height=460, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="weighted contribution")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 3) Learning
# --------------------------------------------------------------------------- #
with tab_learning:
    st.subheader("🧠 How the twin teaches itself")
    st.markdown(
        "1. **Personal baselines** — after a 14-day warm-up, population norms are replaced by *your own* "
        "learned normal values (Welford online statistics), with 95% CIs.\n"
        "2. **Adaptive weights (Hedge online learning)** — every day, factors that correctly anticipated the "
        "next risk movement are up-weighted; noisy ones down-weighted (bounded 0.5–2.0×).\n"
        "3. **Precision growth** — the risk estimate's confidence interval shrinks as evidence accumulates."
    )
    learning = twin.learning_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Days learned from", twin.days_seen)
    c2.metric("Warm personal baselines", f"{learning.get('warm_baselines', 0)}/{learning.get('channels_tracked', 0)}")
    c3.metric("CI width now", f"±{learning.get('ci_width', 0) / 2 * 100:.1f} pts",
              f"{learning.get('precision_gain_pct', 0):+.1f}% vs early days")

    st.subheader("Personalized baselines (learned norms)")
    base_rows = twin.baselines.table(list(twin.state.keys()))
    for r in base_rows:  # annotate z of current state value
        cur = twin.state.get(r["channel"])
        r["current (EWMA state)"] = cur
        dq = twin.baselines.recent.get(r["channel"])
        if dq and len(dq) >= 2 and cur is not None:
            arr = np.asarray(dq)
            r["z_current"] = round((cur - arr.mean()) / (arr.std(ddof=1) or 1e-9), 2)
    bdf = pd.DataFrame(base_rows)
    st.dataframe(bdf, use_container_width=True, height=420)

    st.subheader("Adaptive factor weights (top learned)")
    wdf = pd.DataFrame(twin.weight_learner.summary(top=8))
    if not wdf.empty:
        st.dataframe(wdf, use_container_width=True)
        st.caption("`multiplier` = hedge × stability · `hit_rate` = how often the factor's daily move "
                   "anticipated the composite's next move · multiplier activates after 10 observations")

    st.subheader("Precision growth — CI width over days")
    if twin.confidence.history:
        ch = twin.confidence.history
        fig = go.Figure(go.Scatter(x=[d for d, _ in ch], y=[w * 100 for _, w in ch],
                                   mode="lines", fill="tozeroy", line=dict(color="#1f77b4")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="day", yaxis_title="95% CI width (risk points ×100)")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 4) Trajectory
# --------------------------------------------------------------------------- #
with tab_traj:
    st.subheader("Composite risk trajectory")
    rh = pd.DataFrame(twin.risk_history, columns=["day", "risk"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rh["day"], y=rh["risk"] * 100, mode="lines", name="risk",
                             line=dict(color="#d62728", width=2)))
    if twin.confidence.history:
        ch = pd.DataFrame(twin.confidence.history, columns=["day", "width"])
        merged = rh.merge(ch, on="day", how="left").ffill()
        half = merged["width"] / 2 * 100
        fig.add_trace(go.Scatter(x=merged["day"], y=(merged["risk"] * 100 + half), mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=merged["day"], y=(merged["risk"] * 100 - half), mode="lines",
                                 fill="tonexty", line=dict(width=0), name="95% CI", hoverinfo="skip"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="risk (0-100)", xaxis_title="day")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Physiological channels")
    channels = ["resting_heart_rate", "hrv_rmssd", "systolic_bp", "diastolic_bp",
                "sleep_hours", "steps", "nocturnal_spo2_pct"]
    sel = st.multiselect("Channels", channels, default=["resting_heart_rate", "hrv_rmssd", "systolic_bp"])
    if sel:
        fig2 = go.Figure()
        for c in sel:
            fig2.add_trace(go.Scatter(x=history["day_index"], y=history[c], mode="lines", name=c))
        fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="day")
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Raw daily observations (points) — the twin scores their EWMA-smoothed state.")

# --------------------------------------------------------------------------- #
# 5) Phenotypes
# --------------------------------------------------------------------------- #
with tab_pheno:
    st.subheader("Daily physiological phenotypes")
    try:
        cl = PhenotypeClusterer().fit(history)
        summary = cl.summary()
        st.caption(f"k = {summary['k']} (silhouette {summary['silhouette']})")
        pdf = pd.DataFrame([{**c["means"], "phenotype": c["label"], "days": c["n_days"], "share %": c["share"] * 100}
                            for c in summary["clusters"]])
        st.dataframe(pdf, use_container_width=True)

        pts = cl.transform_2d(history)
        labels = cl.predict(history)
        fig = go.Figure()
        for cid in sorted(labels.unique()):
            mask = labels == cid
            fig.add_trace(go.Scatter(
                x=pts.loc[mask, "pc1"], y=pts.loc[mask, "pc2"], mode="markers",
                name=cl.labels_map_.get(int(cid), f"cluster {cid}"),
            ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig, use_container_width=True)
    except ValueError as exc:
        st.warning(f"Phenotype clustering unavailable: {exc}")

# --------------------------------------------------------------------------- #
# 6) ML model
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Trained risk model (population-level)")
    artifact, metrics = load_model_bundle()
    if artifact is None:
        st.warning("No trained model found. Run `python train_model.py` first.")
    else:
        from src.model_training import twin_feature_vector

        ref_df = artifact.get("data_sample")
        if ref_df is None:
            ref_df = history
        row = twin_feature_vector(twin, ref_df)
        prob = float(artifact["model"].predict_proba(row)[0, 1])
        thr = artifact["threshold"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Model risk probability", f"{prob:.2f}",
                  "HIGH" if prob >= thr else "low", delta_color="inverse")
        c2.metric("Decision threshold", f"{thr:.2f}", "Youden-J")
        c3.metric("Holdout ROC-AUC", f"{metrics['holdout_calibrated']['roc_auc']}" if metrics else "n/a")
        st.caption("Model input mapped from twin state: Age, Sex, RestingBP←systolic EWMA, "
                   "Cholesterol←latest labs, FastingBS←diabetes. Exercise-test features use training-set defaults.")
        st.dataframe(row.T.rename(columns={0: "value"}).astype(str), use_container_width=True)
        if metrics:
            hm = metrics["holdout_calibrated"]
            st.markdown(f"**Holdout performance:** accuracy {hm['accuracy']} · sensitivity {hm['sensitivity']} · "
                        f"specificity {hm['specificity']} · PR-AUC {hm['pr_auc']} · Brier {hm['brier']}")
            imp = pd.DataFrame(metrics["importances"][:8])
            fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
