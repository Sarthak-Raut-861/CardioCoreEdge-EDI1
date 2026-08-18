"""
dashboard/app.py — CardioCore Digital Twin dashboard (Streamlit)
================================================================
Implements the documentation's Chapter 11 dashboard design:

 1. Patient header            7. Baseline comparison (z-scores)
 2. Overall risk gauge + CTR  8. Time-series trends
 3. Lipid profile table       9. Graded alerts
 4. Inflammatory markers     10. AI explanation (top-5 factors per biomarker)
 5. 6-axis radar chart       11. Disclaimer
 6. Cluster assignment (K=4, PCA, confidence)
 + full 72-factor table (all normalized values + weights + directions)
 + wearable twin tab (EWMA factor engine + adaptive learning)

Run:  streamlit run dashboard/app.py

RESEARCH PROTOTYPE — all biomarker values are AI estimates. Not for clinical
diagnosis. Confirm findings with laboratory blood tests.
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

from src.biomarker_twin import BiomarkerClusterer, BiomarkerTwin, generate_cohort  # noqa: E402
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource, simulate_history  # noqa: E402
from src.digital_twin import DISCLAIMER, DigitalTwin  # noqa: E402
from src.explainability import RiskExplainer  # noqa: E402
from src.factors72 import FACTORS, FACTOR_IDS  # noqa: E402

st.set_page_config(page_title="CardioCore Digital Twin", page_icon="❤️", layout="wide")

# §11.2 color coding
GREEN, YELLOW, ORANGE, RED, GRAY = "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6"
CATEGORY_COLORS = {
    "Desirable": GREEN, "Borderline High": YELLOW, "High": ORANGE, "Very High": RED,
    "Optimal": GREEN, "Near Optimal": GREEN, "Borderline": YELLOW,
    "Protective": GREEN, "Normal": GREEN, "Low (Risk)": RED,
    "Low CV Risk": GREEN, "Moderate CV Risk": YELLOW, "High CV Risk": ORANGE, "Acute Infection": RED,
    "Mild Elevation": YELLOW, "Mild": YELLOW, "Moderate": ORANGE, "High Risk": RED,
    "Low": GREEN, "Moderate ": YELLOW, "Very High": RED,
}


def color_for(category: str) -> str:
    if not category:
        return GRAY
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    if category.startswith("High"):
        return ORANGE
    if category.startswith("Very High"):
        return RED
    if category.startswith("Moderate"):
        return YELLOW
    return GRAY


BIOMARKER_INFO = {  # unit + normal range text (doc §6.10)
    "TC": ("mg/dL", "Desirable < 200"),
    "HDL": ("mg/dL", "Protective ≥ 60"),
    "LDL": ("mg/dL", "Optimal < 100"),
    "TG": ("mg/dL", "Normal < 150"),
    "CRP": ("mg/L", "Low risk < 1.0"),
    "DD": ("mg/L", "Normal < 0.5"),
}
NAMES = {"TC": "Total Cholesterol", "HDL": "HDL Cholesterol", "LDL": "LDL Cholesterol",
         "TG": "Triglycerides", "CRP": "C-Reactive Protein", "DD": "D-Dimer"}


# --------------------------------------------------------------------------- #
# Simulation (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running 72-factor digital-twin simulation ...")
def run_twin(profile_name: str, scenario: str, days: int, seed: int):
    profile = PRESET_PROFILES[profile_name]
    source = SimulatedWearableSource(profile, scenario=scenario, seed=seed, lab_interval_days=90)
    twin = BiomarkerTwin(profile.to_dict())
    wearable = DigitalTwin(profile.to_dict(), learning_rate=0.15)
    for i, obs in enumerate(source.next_days(days)):
        twin.update(obs, day_index=i)
        wearable.assimilate(obs, obs.get("labs"))
    history = simulate_history(days, profile, scenario=scenario, seed=seed, lab_interval_days=90)
    bio_df = pd.DataFrame(list(twin.biomarker_history))
    bio_df["CVD"] = [r for _, r in twin.risk_history]
    factor_df = pd.DataFrame(list(twin.factor_history))
    return twin.snapshot(), wearable.snapshot(), bio_df, factor_df, history


# --------------------------------------------------------------------------- #
# Sidebar (patient header + controls)
# --------------------------------------------------------------------------- #
st.sidebar.title("❤️ CardioCore Digital Twin")
st.sidebar.caption("Continuous · Personalized · Explainable · Non-Invasive")

profile_name = st.sidebar.selectbox("User profile", list(PRESET_PROFILES), index=2)
scenario = st.sidebar.selectbox("Scenario", ["stable", "improving", "declining"], index=2)
days = st.sidebar.slider("Days monitored", 30, 180, 60, 10)
seed = st.sidebar.number_input("Random seed", 0, 999, 42)

snap, wear_snap, bio_df, factor_df, history = run_twin(profile_name, scenario, int(days), int(seed))
twin = BiomarkerTwin.__new__(BiomarkerTwin)   # lightweight view over snapshot
state = snap["state"]
profile = PRESET_PROFILES[profile_name]

st.sidebar.success(f"{days} days simulated · {snap['days_seen']} readings assimilated")
st.sidebar.warning("■ RESEARCH PROTOTYPE — biomarker values are AI estimates. Confirm with lab blood tests.")

# ---- 1) Patient header ------------------------------------------------------ #
st.title(f"❤️ {profile.name} — Cardiovascular Digital Twin")
st.caption(f"Age {profile.age} · {profile.sex.upper()} · BMI {profile.bmi} · smoker {profile.smoker} · "
           f"diabetic {profile.diabetic} · family history {profile.family_history} · scenario: **{scenario}** · "
           f"twin status: **ACTIVE** · {snap['days_seen']} readings")

tabs = st.tabs(["📊 Overview", "🧬 Lipids & Biomarkers", "🔢 72 Factors", "🧠 Baselines",
                "🧭 Cluster", "📈 Trends", "🚨 Alerts", "💬 AI Explanation", "⌚ Wearable Twin"])

# --------------------------------------------------------------------------- #
# 2) Overview: gauge + CTR + radar
# --------------------------------------------------------------------------- #
with tabs[0]:
    b = state["biomarkers"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CVD Risk Score", f"{state['cvd_score'] * 100:.0f} / 100", state["cvd_category"])
    c2.metric("Trend (14d vs prior 14d)", snap["trend"] or "—")
    c3.metric("Coronary Thrombosis Risk", f"{state['ctr']:.2f}", state["ctr_category"])
    c4.metric("Risk Cluster", state["cluster"]["label"], f"conf {state['cluster']['confidence'] * 100:.0f}%")

    gauge_col, radar_col = st.columns([1, 1])
    with gauge_col:
        st.subheader("Overall CVD Risk (0-100)")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=state["cvd_score"] * 100,
            number={"suffix": " / 100"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color_for(state["cvd_category"])},
                   "steps": [
                       {"range": [0, 25], "color": "#d5f5e3"},
                       {"range": [25, 50], "color": "#fdebd0"},
                       {"range": [50, 75], "color": "#fad7a0"},
                       {"range": [75, 100], "color": "#f5b7b1"}],
                   "threshold": {"line": {"color": RED, "width": 3},
                                 "thickness": 0.9, "value": state["cvd_score"] * 100}}))
        fig.update_layout(height=280, margin=dict(l=25, r=25, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pathway Risk Scores")
        pw = state["pathway_risk"]
        pw_df = pd.DataFrame({"pathway": list(pw.keys()),
                              "risk (0-1)": list(pw.values())})
        pw_df["color"] = pw_df["risk (0-1)"].map(lambda v: RED if v > .66 else (ORANGE if v > .33 else GREEN))
        figp = go.Figure(go.Bar(x=pw_df["risk (0-1)"], y=pw_df["pathway"], orientation="h",
                                marker_color=pw_df["color"], text=pw_df["risk (0-1)"], textposition="outside"))
        figp.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=10), xaxis_range=[0, 1])
        st.plotly_chart(figp, use_container_width=True)

    with radar_col:
        st.subheader("6-Axis Biomarker Risk Radar")
        def risk_norm(k):
            v = b[k]
            return {"TC": (v - 150) / 170, "LDL": (v - 30) / 270, "HDL": 1 - (v - 20) / 60,
                    "TG": (v - 80) / 420, "CRP": (v - 0.2) / 9.8, "DD": (v - 0.1) / 2.9}[k]
        axes = ["TC", "LDL", "HDL", "TG", "CRP", "DD"]
        vals = [max(0.0, min(1.0, risk_norm(k))) for k in axes]
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=axes + [axes[0]], fill="toself",
            line=dict(color="#e74c3c"), name="risk"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True)),
                          height=380, margin=dict(l=60, r=60, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each axis = biomarker risk-normalized to 0 (optimal) → 1 (worst). Larger area = higher risk.")

# --------------------------------------------------------------------------- #
# 3) + 4) Lipid profile & inflammatory markers tables
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Lipid Profile (AI-estimated)")
    rows = []
    for k in ("TC", "HDL", "LDL", "TG"):
        unit, normal = BIOMARKER_INFO[k]
        rows.append({"Biomarker": NAMES[k], f"Value ({unit})": b[k], "Category": state["categories"][k],
                     "Normal range": normal, "Direction": "↓ better" if k == "HDL" else "↑ worse"})
    for label, key, fmt in (("VLDL", "VLDL", "{:.1f}"), ("Non-HDL", "Non_HDL", "{:.1f}"),
                            ("TC/HDL ratio", "TC_HDL_ratio", "{:.2f}"), ("LDL/HDL ratio", "LDL_HDL_ratio", "{:.2f}"),
                            ("AIP (log₁₀ TG/HDL)", "AIP", "{:.3f}")):
        rows.append({"Biomarker": label, f"Value ({'mg/dL' if key in ('VLDL', 'Non_HDL') else 'ratio'})":
                     fmt.format(state["derived"][key]), "Category": "—",
                     "Normal range": {"VLDL": "2-30", "Non_HDL": "< 130", "TC_HDL_ratio": "< 4.5",
                                      "LDL_HDL_ratio": "< 2.0", "AIP": "< 0.21"}[key], "Direction": "↑ worse"})
    lip = pd.DataFrame(rows)
    st.dataframe(lip.style.applymap(lambda c: f"color: {color_for(c)}; font-weight: bold", subset=["Category"]),
                 use_container_width=True, hide_index=True)

    st.subheader("Inflammatory & Thrombosis Markers")
    i1, i2, i3 = st.columns(3)
    i1.metric("CRP (inflammation)", f"{b['CRP']:.2f} mg/L", state["categories"]["CRP"],
              delta_color="inverse" if "Low" in state["categories"]["CRP"] else "normal")
    i2.metric("D-Dimer (thrombosis)", f"{b['DD']:.2f} mg/L", state["categories"]["DD"],
              delta_color="inverse" if state["categories"]["DD"] == "Normal" else "normal")
    i3.metric("Adjusted CRP (HDL-modified)", f"{state['adjusted']['CRP_adj']:.2f} mg/L",
              f"HDL protection ×{state['adjusted']['HDL_protection']}")
    st.caption(f"Interactions (§6.11): TC context multiplier ×{state['adjusted']['CRP_mult']} from CRP; "
               f"adjusted TC {state['adjusted']['TC_adj']:.0f} mg/dL · D-Dimer includes CRP linkage (+1.2 × CRP_norm).")

# --------------------------------------------------------------------------- #
# 72-factor table
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("The 72 Factors — all normalized values (doc Chapter 5)")
    fac = state["factors"]
    rows = []
    for fid in FACTOR_IDS:
        spec = FACTORS[fid]
        links = ", ".join(f"{bm} (w={w}, {'+' if d > 0 else '−'})"
                          for bm, (w, d) in sorted(spec.weights.items()))
        rows.append({"ID": fid, "Factor": spec.name, "Source": spec.source, "Category": spec.category,
                     "Raw range": f"{spec.raw_min:g} – {spec.raw_max:g}",
                     "Normalized (0-1)": fac.get(fid), "Reading": spec.direction,
                     "Biomarker links (weight, dir)": links or "—"})
    fdf = pd.DataFrame(rows)
    cat_filter = st.multiselect("Filter by category", sorted(fdf["Category"].unique()))
    view = fdf[fdf["Category"].isin(cat_filter)] if cat_filter else fdf
    st.caption(f"Showing {len(view)} / 72 factors · normalization F_norm = (raw − min) / (max − min), clipped [0,1]")
    st.dataframe(view.style.format({"Normalized (0-1)": "{:.3f}"}).background_gradient(
        subset=["Normalized (0-1)"], cmap="RdYlGn_r", vmin=0, vmax=1), use_container_width=True, height=600)

    st.subheader("Factor categories coverage")
    cc = fdf.groupby("Category")["Normalized (0-1)"].mean().sort_values(ascending=False)
    st.bar_chart(cc)

# --------------------------------------------------------------------------- #
# 7) Baseline comparison
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Personalized Baseline Comparison (doc §9.3)")
    twin_baselines = snap.get("baselines", {})
    stats = twin_baselines.get("stats", {})
    recent = twin_baselines.get("recent", {})
    rows = []
    for k in ("TC", "HDL", "LDL", "TG", "CRP", "DD"):
        st_ = stats.get(k, {})
        n = st_.get("n", 0)
        mean = st_.get("mean")
        dq = recent.get(k, [])
        cur = state["biomarkers"].get(k)
        z = state["zscores"].get(k)
        import math as _m
        sd = None
        if len(dq) >= 2:
            arr = np.asarray(dq)
            sd = float(arr.std(ddof=1))
        pct = round(100 * (cur - mean) / mean, 1) if (mean and cur is not None) else None
        sig = bool(z is not None and abs(z) > 2.0)
        rows.append({"Biomarker": k, "n readings": n, "Baseline mean": round(mean, 2) if mean else None,
                     "Baseline SD": round(sd, 3) if sd else None, "Current": cur,
                     "Z-score": z, "% change": pct,
                     "Significant (|Z|>2)": "⚠️ YES" if sig else "no"})
    bdf = pd.DataFrame(rows)
    st.dataframe(bdf.style.applymap(lambda v: "color: #e74c3c; font-weight: bold" if v == "⚠️ YES" else "",
                               subset=["Significant (|Z|>2)"]), use_container_width=True, hide_index=True)
    st.caption("Baseline established after 5 readings; significant deviation |Z| > 2, critical |Z| > 3 (§9.3).")

# --------------------------------------------------------------------------- #
# 6) Cluster assignment
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Risk Cluster Assignment (K-Means, K=4 — doc Chapter 8)")
    cl = state["cluster"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster", cl["label"])
    c2.metric("Confidence", f"{cl['confidence'] * 100:.0f}%",
              "clear assignment" if cl["confidence"] > 0.6 else "borderline patient")
    c3.metric("Cluster mean CVD (cohort)", f"{cl['cluster_cvd_mean'] or '—'}")

    cohort_X, cohort_cvd = generate_cohort(160, seed=7)
    clusterer = BiomarkerClusterer().fit(cohort_X, cohort_cvd)
    pts = clusterer.cohort_points_2d(cohort_X)
    fig = go.Figure()
    for label in sorted(pts["cluster"].unique()):
        sub = pts[pts["cluster"] == label]
        fig.add_trace(go.Scatter(x=sub["pc1"], y=sub["pc2"], mode="markers", name=label,
                                 marker=dict(size=6, opacity=0.55)))
    fig.add_trace(go.Scatter(x=[cl["pc1"]], y=[cl["pc2"]], mode="markers",
                             marker=dict(size=18, color="black", line=dict(width=2, color="white")),
                             name="⭐ THIS PATIENT"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Patient distances to cluster centers: {cl['distances']} · "
               "confidence = 1 − d_own/max(d_any) (§8.5)")

# --------------------------------------------------------------------------- #
# 8) Trends
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Biomarker Trends")
    fig = go.Figure()
    for k, color in (("TC", "#e74c3c"), ("HDL", "#2ecc71"), ("LDL", "#e67e22"), ("TG", "#8e44ad")):
        fig.add_trace(go.Scatter(x=bio_df.index, y=bio_df[k], mode="lines", name=f"{k} (mg/dL)",
                                 line=dict(color=color)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="mg/dL", xaxis_title="day")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bio_df.index, y=bio_df["CRP"], mode="lines", name="CRP (mg/L)",
                              line=dict(color="#c0392b")))
    fig2.add_trace(go.Scatter(x=bio_df.index, y=bio_df["DD"], mode="lines", name="D-Dimer (mg/L)",
                              line=dict(color="#16a085")))
    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="mg/L")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bio_df.index, y=bio_df["CVD"], mode="lines", name="CVD risk (0-1)",
                              line=dict(color="#2c3e50", width=3)))
    fig3.add_hrect(y0=0.25, y1=0.5, line_width=0, fillcolor="#f1c40f", opacity=0.08)
    fig3.add_hrect(y0=0.5, y1=0.75, line_width=0, fillcolor="#e67e22", opacity=0.08)
    fig3.add_hrect(y0=0.75, y1=1.0, line_width=0, fillcolor="#e74c3c", opacity=0.08)
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1])
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------- #
# 9) Alerts
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Alerts (INFO / WARNING / CRITICAL)")
    alerts = snap["alerts_raised"]
    if not alerts:
        st.success("No alerts raised during the monitored period.")
    else:
        level_color = {"critical": "error", "warning": "warning", "info": "info"}
        for a in alerts[-15:][::-1]:
            getattr(st, level_color.get(a["level"], "info"))(f"**[{a['level'].upper()}]** `{a['code']}` — {a['message']}")
        st.caption(f"{len(alerts)} alerts total (showing latest 15, deduplicated daily)")

# --------------------------------------------------------------------------- #
# 10) AI explanation
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Explainable AI — WHY these values? (doc Chapter 10)")
    tcps = state["top_contributions"]
    for bm in ("TC", "HDL", "TG", "CRP", "DD"):
        contribs = tcps.get(bm, [])
        if not contribs:
            continue
        val = b[bm]
        unit = BIOMARKER_INFO[bm][0]
        st.markdown(f"**{NAMES[bm]} = {val} {unit} ({state['categories'][bm]})** is mainly influenced by:")
        for i, c in enumerate(contribs[:5], 1):
            arrow = "↑" if c["contribution"] > 0 else "↓"
            st.markdown(f"{i}. {arrow} **{c['name']}** ({c['fid']}) — contribution {c['contribution']:+.1f} "
                        f"(value {c['value']:.2f} × weight {c['weight']:.0f}, {c['direction']})")
        st.markdown("")

# --------------------------------------------------------------------------- #
# Wearable twin (previous engine)
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Wearable Digital Twin (adaptive factor engine)")
    wearable = DigitalTwin.from_dict(wear_snap)
    learning = wearable.learning_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wearable-twin risk", f"{wearable.risk_score * 100:.0f}/100", wearable.risk_category)
    c2.metric("Trend", wearable.risk_trend() or "—")
    c3.metric("Warm personal baselines", f"{learning.get('warm_baselines', 0)}/{learning.get('channels_tracked', 0)}")
    c4.metric("CI precision gain", f"{learning.get('precision_gain_pct', 0):+.1f}%")

    comp = wearable._compute_risk()
    table = RiskExplainer.explain_composite(comp).head(10)
    st.dataframe(table, use_container_width=True)
    rh = pd.DataFrame(wearable.risk_history, columns=["day", "risk"])
    fig = go.Figure(go.Scatter(x=rh["day"], y=rh["risk"] * 100, mode="lines",
                               line=dict(color="#d62728", width=2)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="risk (0-100)", xaxis_title="day")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This engine scores daily wearable aggregates with EWMA state + adaptive weights; "
               "the biomarker twin (other tabs) estimates blood biomarkers from the 72 factors.")

# 11) Disclaimer
st.markdown("---")
st.error("■ RESEARCH PROTOTYPE — All biomarker values are AI estimates derived from sensor data. "
         "This system is NOT a medical device and is NOT approved for clinical diagnosis. "
         "Always confirm findings with laboratory blood tests and consult a qualified healthcare professional.")
