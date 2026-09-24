/* ============================================================
   CardioCore — Personal Cardiovascular Command Center
   Frontend Architecture & Reactive SPA Application
   ============================================================ */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const app = $("#app");

/* Safe local storage helper */
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  del(k) { try { localStorage.removeItem(k); } catch (e) {} },
};

/* Application State */
const state = {
  token: store.get("cc_token") || null,
  user: JSON.parse(store.get("cc_user") || "null"),
  profile: null,
  result: null,
  activeView: "overview", // "overview" | "twin" | "trends" | "factors" | "alerts" | "profile"
  scenario: "stable",    // "stable" | "improving" | "declining"
  trendMetric: "cvd",     // "cvd" | "TC" | "HDL" | "LDL" | "TG" | "CRP" | "DD"
  trendHorizon: "60D",    // "7D" | "30D" | "60D" | "all"
  factorQuery: "",
  factorCategory: "all",
  dismissedAlerts: new Set(),
  _step: 0,
};

/* ---------------- Utilities ---------------- */
function toast(msg, isErr = false) {
  const t = $("#toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.hidden = false;
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.hidden = true; }, 3400);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
      ...(opts.headers || {}),
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401 && state.token && !path.startsWith("/api/auth/")) {
      doLogout(true);
      toast(data.detail || "Session expired — please sign in again", true);
    }
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[c]));

const fmt = (v, d = 0) => Number(v ?? 0).toFixed(d);

function nav(hash) { location.hash = hash; }
function navOrRoute(hash) { if (location.hash === hash) route(); else nav(hash); }

function doLogout(silent = false) {
  if (state.token) api("/api/auth/logout", { method: "POST" }).catch(() => {});
  store.del("cc_token");
  store.del("cc_user");
  state.token = null;
  state.user = null;
  state.profile = null;
  state.result = null;
  if (!silent) toast("Logged out successfully");
  navOrRoute("#/auth");
}

/* ---------------- Clinical Color System & Metadata ---------------- */
const CAT_COLORS = {
  Desirable: "#059669",
  Optimal: "#059669",
  Protective: "#059669",
  Normal: "#059669",
  Low: "#059669",
  "Low CV Risk": "#059669",
  "Near Optimal": "#10B981",
  "Borderline High": "#D97706",
  Borderline: "#D97706",
  "Moderate CV Risk": "#D97706",
  Moderate: "#D97706",
  "Mild Elevation": "#D97706",
  Mild: "#D97706",
  High: "#EA580C",
  "High CV Risk": "#EA580C",
  "High Risk": "#EA580C",
  "Low (Risk)": "#EA580C",
  "Very High": "#DC2626",
  "Very High Risk": "#DC2626",
  "Acute Infection": "#DC2626",
};

const catColor = (c) => CAT_COLORS[c] || (
  String(c).startsWith("Very High") ? "#DC2626" :
  String(c).startsWith("High") ? "#EA580C" :
  String(c).startsWith("Moderate") ? "#D97706" :
  String(c).startsWith("Borderline") ? "#D97706" : "#059669"
);

const levelName = (v) => (v < 0.33 ? "Low" : v < 0.66 ? "Moderate" : "High");
const levelColor = (v) => (v < 0.33 ? "#059669" : v < 0.66 ? "#D97706" : "#DC2626");
const levelBadgeClass = (v) => (v < 0.33 ? "badge-low" : v < 0.66 ? "badge-mod" : "badge-high");

const PATHWAY_INFO = {
  lipid: { icon: "🫀", name: "Lipid Pathway", desc: "Atherogenic particle synthesis & transport", weight: "30%" },
  inflammation: { icon: "🔥", name: "Inflammation", desc: "Systemic vascular & arterial inflammatory tone", weight: "20%" },
  thrombosis: { icon: "🩸", name: "Thrombosis", desc: "Coagulation, fibrinolysis & clot susceptibility", weight: "20%" },
  hemodynamic: { icon: "💓", name: "Hemodynamic", desc: "Arterial stiffness, wave reflections & BP load", weight: "12%" },
  autonomic: { icon: "🧠", name: "Autonomic", desc: "Sympathovagal balance & heart rate variability", weight: "10%" },
  metabolic: { icon: "⚙️", name: "Metabolic", desc: "Glycemic stress, adiposity & lipid ratios", weight: "8%" },
};

const SENSOR_CATALOG = {
  ECG: { icon: "📈", name: "ECG Sensor", stream: "Lead-I Biopotential", desc: "Measures electrical cardiac activation, RR intervals, HRV (SDNN/RMSSD), and repolarization dynamics." },
  PPG: { icon: "🩸", name: "PPG Morphology", stream: "Optical Plethysmography", desc: "Captures pulse wave velocity (PWV), augmentation index (AIx), stiffness, and vascular transit times." },
  SpO2: { icon: "🫁", name: "Pulse Oximetry", stream: "Dual-wavelength Red/IR", desc: "Monitors continuous peripheral oxygen saturation, desaturation nadirs, and respiratory stability." },
  "Blood Pressure": { icon: "🩺", name: "Blood Pressure", stream: "Continuous Cuffless BP", desc: "Estimates real-time systolic, diastolic, pulse pressure, and beat-to-beat pressure variability." },
  NIR: { icon: "🔆", name: "NIR Spectroscopy", stream: "Multi-band Spectral (900-1700nm)", desc: "Optical resonance absorption for non-invasive lipid, glucose, and tissue water attenuation." },
  Temperature: { icon: "🌡️", name: "Thermal Sensor", stream: "High-precision Skin Temp", desc: "Tracks circadian core-to-shell temperature gradients and inflammatory fever variations." },
  GSR: { icon: "⚡", name: "Electrodermal (GSR)", stream: "Skin Conductance Response", desc: "Quantifies sympathetic nervous arousal, stress response intensity, and autonomic tone." },
  Bioimpedance: { icon: "📏", name: "Bioimpedance (BioZ)", stream: "Multi-frequency Impedance", desc: "Assesses body composition, extracellular fluid shifts, thoracic fluid content, and hydration." },
  IMU: { icon: "🏃", name: "Inertial (IMU)", stream: "6-Axis Accelerometer + Gyro", desc: "Analyzes physical exertion, activity energy expenditure, sleep posture, and sedentary periods." },
};

/* Educational Glossary Database for "What does this mean?" */
const GLOSSARY = {
  cvd: {
    title: "Cardiovascular Risk Score",
    text: "A continuous composite index (0–100) estimated by your digital twin. It integrates estimated lipid fractions, systemic inflammation (CRP), thrombotic risk (D-Dimer), arterial stiffness (PWV), and lifestyle factors to predict overall cardiovascular load.",
    clinical: "Score < 30 indicates Low Risk; 30–60 indicates Moderate Risk; > 60 indicates Elevated Risk requiring preventive focus."
  },
  ctr: {
    title: "Coronary Thrombosis Risk (CTR)",
    text: "An AI-modeled score indicating the susceptibility to intravascular clot formation inside coronary vessels. It specifically combines LDL cholesterol (35%), D-Dimer (40%), and CRP (25%).",
    clinical: "Lower scores indicate balanced coagulative and inflammatory homeostasis."
  },
  TC: {
    title: "Total Cholesterol",
    text: "The estimated total amount of cholesterol in your blood, comprising HDL, LDL, and VLDL particles.",
    clinical: "Desirable: < 200 mg/dL. Borderline high: 200–239 mg/dL. High: ≥ 240 mg/dL."
  },
  HDL: {
    title: "HDL Cholesterol ('Protective')",
    text: "High-Density Lipoprotein absorbs excess cholesterol in the bloodstream and carries it back to the liver for excretion (reverse cholesterol transport).",
    clinical: "Protective: ≥ 60 mg/dL. Acceptable: 40–59 mg/dL. Low (risk factor): < 40 mg/dL."
  },
  LDL: {
    title: "LDL Cholesterol ('Atherogenic')",
    text: "Low-Density Lipoprotein transports cholesterol to tissues. Excess circulating LDL can penetrate and oxidize within the arterial intima, initiating atherosclerotic plaque.",
    clinical: "Optimal: < 100 mg/dL. Near optimal: 100–129 mg/dL. Borderline high: 130–159 mg/dL. High: ≥ 160 mg/dL."
  },
  TG: {
    title: "Triglycerides",
    text: "The primary chemical form of fat in food and the body. Elevated triglycerides often accompany metabolic syndrome and insulin resistance.",
    clinical: "Normal: < 150 mg/dL. Borderline high: 150–199 mg/dL. High: 200–499 mg/dL."
  },
  CRP: {
    title: "C-Reactive Protein (CRP)",
    text: "A liver-synthesized acute-phase reactant protein whose blood levels rise in response to systemic inflammation and arterial wall stress.",
    clinical: "Low CV risk: < 1.0 mg/L. Moderate risk: 1.0–3.0 mg/L. High risk: > 3.0 mg/L."
  },
  DD: {
    title: "D-Dimer",
    text: "A fibrin degradation protein fragment generated when a blood clot dissolves. Monitored to detect abnormal thrombotic activity or micro-clotting.",
    clinical: "Normal reference level: < 0.50 mg/L."
  },
  AIP: {
    title: "Atherogenic Index of Plasma (AIP)",
    text: "Calculated as log₁₀(TG / HDL). Reflects the ratio between atherogenic and anti-atherogenic lipoproteins and correlates strongly with small, dense LDL particle size.",
    clinical: "Low atherogenic risk: < 0.11. Moderate: 0.11–0.21. High: > 0.21."
  },
  baseline: {
    title: "Personal Baseline (You vs. You)",
    text: "Unlike static population charts, your digital twin continuously learns YOUR individual baseline distribution using an Exponentially Weighted Moving Average (EWMA). Deviations are evaluated using Z-scores based on your own physiological variance.",
    clinical: "Identifies early health trajectory shifts before population thresholds are crossed."
  },
};

/* ---------------- Modal & Drawer System ---------------- */
function openModal(title, contentHtml) {
  const backdrop = $("#modal-backdrop");
  const tEl = $("#modal-title");
  const bEl = $("#modal-body");
  if (!backdrop || !tEl || !bEl) return;
  tEl.innerHTML = title;
  bEl.innerHTML = contentHtml;
  backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  $("#modal-close").onclick = closeModal;
  backdrop.onclick = (e) => { if (e.target === backdrop) closeModal(); };
}

function closeModal() {
  const backdrop = $("#modal-backdrop");
  if (!backdrop) return;
  backdrop.hidden = true;
  document.body.style.overflow = "";
}

function showWhatDoesThisMean(key) {
  const item = GLOSSARY[key];
  if (!item) return;
  openModal(`<span>ℹ️</span> ${item.title}`, `
    <div style="display:flex;flex-direction:column;gap:14px">
      <p style="font-size:14.5px;color:var(--ink-primary);line-height:1.6">${item.text}</p>
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:4px solid var(--brand);border-radius:var(--radius-sm);padding:14px 16px">
        <b style="color:var(--brand-dark);font-size:13px;display:block;margin-bottom:4px">Clinical Interpretation:</b>
        <span style="font-size:13px;color:var(--ink-secondary)">${item.clinical}</span>
      </div>
      <div class="badge badge-ai" style="width:fit-content">
        ✨ AI Estimate — Not a substitute for laboratory testing
      </div>
    </div>
  `);
}

function showSensorDetails(sensorName) {
  const s = SENSOR_CATALOG[sensorName] || { icon: "📡", name: sensorName, stream: "Wearable Channel", desc: "Multimodal sensor streaming continuous physiological telemetry to the digital twin." };
  openModal(`<span>${s.icon}</span> ${s.name}`, `
    <div style="display:flex;flex-direction:column;gap:16px">
      <div style="display:flex;align-items:center;justify-content:space-between;background:#F0FDF4;border:1px solid #BBF7D0;padding:12px 16px;border-radius:var(--radius-sm)">
        <div style="display:flex;align-items:center;gap:10px">
          <span class="pulse-dot"></span>
          <b style="color:#166534;font-size:13.5px">Active & Synchronized</b>
        </div>
        <span style="font-size:12px;color:#15803D;font-weight:600">Stream: ${esc(s.stream)}</span>
      </div>
      <p style="font-size:14px;color:var(--ink-primary);line-height:1.6">${s.desc}</p>
      <div>
        <h4 style="font-size:13px;font-weight:700;color:var(--ink-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">Fed into 72-Factor Digital Twin:</h4>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <span class="badge badge-neutral">Feature Extraction: 100 Hz</span>
          <span class="badge badge-neutral">Artifact Rejection: Active</span>
          <span class="badge badge-neutral">EWMA Smoothing: α = 0.15</span>
        </div>
      </div>
    </div>
  `);
}

function showPathwayDetails(pathwayKey) {
  if (!state.result) return;
  const p = PATHWAY_INFO[pathwayKey];
  const score = state.result.pathways[pathwayKey] ?? 0;
  const col = levelColor(score);
  const lvl = levelName(score);

  // Find factors belonging to this pathway category
  const relatedFactors = (state.result.factors || []).filter((f) => {
    const cat = (f.category || "").toLowerCase();
    if (pathwayKey === "lipid") return cat === "nir" || cat === "diet";
    if (pathwayKey === "inflammation") return cat === "temp" || cat === "history" || cat === "smoking";
    if (pathwayKey === "thrombosis") return cat === "bioz" || cat === "history";
    if (pathwayKey === "hemodynamic") return cat === "bp" || cat === "ppg";
    if (pathwayKey === "autonomic") return cat === "hrv" || cat === "gsr" || cat === "sleep/stress";
    if (pathwayKey === "metabolic") return cat === "demographic" || cat === "activity";
    return true;
  }).slice(0, 6);

  openModal(`<span>${p.icon}</span> ${p.name}`, `
    <div style="display:flex;flex-direction:column;gap:18px">
      <div style="display:flex;align-items:center;justify-content:space-between;background:var(--surface-subtle);padding:16px 20px;border-radius:var(--radius-md)">
        <div>
          <div style="font-size:12px;color:var(--ink-muted);font-weight:600">Current Pathway Activity</div>
          <div style="font-size:28px;font-weight:800;color:${col}">${score.toFixed(2)} <span style="font-size:14px;font-weight:600;color:var(--ink-muted)">/ 1.00</span></div>
        </div>
        <span class="badge ${levelBadgeClass(score)}" style="font-size:14px;padding:6px 14px">
          <span class="badge-dot"></span>${lvl} Risk
        </span>
      </div>
      <p style="font-size:14px;color:var(--ink-secondary);line-height:1.6">${p.desc}. Contributing ${p.weight} to overall cardiovascular scoring.</p>
      
      <div>
        <h4 style="font-size:13px;font-weight:700;color:var(--ink-primary);margin-bottom:10px">Primary Contributing Factors in this Pathway:</h4>
        <div style="display:flex;flex-direction:column;gap:8px">
          ${relatedFactors.map((f) => `
            <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#FBFCFD;border:1px solid var(--line-subtle);border-radius:var(--radius-xs)">
              <span style="font-size:13px;font-weight:600;color:var(--ink-primary)">${esc(f.name)}</span>
              <span class="badge ${f.reading === "protective" ? "badge-low" : "badge-high"}">${(f.value ?? 0).toFixed(2)} · ${esc(f.reading)}</span>
            </div>
          `).join("")}
        </div>
      </div>
    </div>
  `);
}

/* ============================================================
   AUTH VIEW
   ============================================================ */
function renderAuth() {
  document.title = "CardioCore · Sign in";
  const mode = state._authMode || "login";
  app.innerHTML = `
  <div class="auth-wrap fade-in">
    <section class="auth-brand">
      <div>
        <div class="logo"><span class="mark">❤️</span> CardioCore</div>
        <svg class="pulse-lines" width="360" height="70" viewBox="0 0 360 70" fill="none" style="margin: 32px 0 16px; opacity: 0.95">
          <path d="M0 35 H70 l12-24 16 44 14-32 10 12 H190 l12-24 16 44 14-32 10 12 H360"
                stroke="#FDA4AF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <h1>Your heart, <em>continuously</em> understood.</h1>
        <p class="sub">A personalized cardiovascular digital twin built from 72 physiological factors —
        estimating lipids, inflammation, and thrombosis risk continuously from wearable sensing.</p>
      </div>
      <div style="display:grid;gap:12px;margin:28px 0">
        <div style="display:flex;gap:10px;align-items:center;color:#FFE4E6;font-size:13.5px">✦ &nbsp;<b>72-factor continuous framework</b> · multimodal wearable fusion</div>
        <div style="display:flex;gap:10px;align-items:center;color:#FFE4E6;font-size:13.5px">✦ &nbsp;<b>Biomarker estimation</b> · TC · HDL · LDL · TG · CRP · D-Dimer</div>
        <div style="display:flex;gap:10px;align-items:center;color:#FFE4E6;font-size:13.5px">✦ &nbsp;<b>Explainable AI</b> · transparent attribution for every score</div>
        <div style="display:flex;gap:10px;align-items:center;color:#FFE4E6;font-size:13.5px">✦ &nbsp;<b>Personal baseline</b> · compares you with YOUR normal</div>
      </div>
      <div style="color:#FDA4AF;font-size:12px">Research prototype — AI estimates, not a medical device.</div>
    </section>

    <section class="auth-panel">
      <div class="auth-card">
        <div class="nav-links" style="margin-bottom:24px;width:100%">
          <button class="nav-item ${mode === "login" ? "active" : ""}" style="flex:1" data-m="login">Sign in</button>
          <button class="nav-item ${mode === "signup" ? "active" : ""}" style="flex:1" data-m="signup">Create account</button>
        </div>
        <h2>${mode === "login" ? "Welcome back" : "Create your account"}</h2>
        <p class="lead">${mode === "login" ? "Sign in to access your cardiovascular command center." : "Build your personalized cardiovascular digital twin in 4 quick steps."}</p>
        <form id="auth-form" novalidate>
          ${mode === "signup" ? `
          <div class="field" data-f="name">
            <label>Full name</label>
            <input name="name" autocomplete="name" placeholder="Sarthak Raut" />
            <div class="err" hidden></div>
          </div>` : ""}
          <div class="field" data-f="email">
            <label>Email address</label>
            <input name="email" type="email" autocomplete="email" placeholder="sarthak@example.com" />
            <div class="err" hidden></div>
          </div>
          <div class="field" data-f="password">
            <label>Password ${mode === "signup" ? '<span style="font-weight:400;color:var(--ink-faint);font-size:12px">(min 6 characters)</span>' : ""}</label>
            <input name="password" type="password" autocomplete="${mode === "signup" ? "new-password" : "current-password"}" placeholder="••••••••" />
            <div class="err" hidden></div>
          </div>
          <button class="btn btn-primary" style="width:100%;margin-top:10px;padding:12px" type="submit">
            ${mode === "login" ? "Sign in →" : "Create account & begin →"}
          </button>
        </form>
      </div>
    </section>
  </div>`;

  $$(".nav-links button").forEach((b) => b.onclick = () => { state._authMode = b.dataset.m; renderAuth(); });

  $("#auth-form").onsubmit = async (e) => {
    e.preventDefault();
    const payload = Object.fromEntries(new FormData(e.target).entries());
    let ok = true;
    const setErr = (f, msg) => {
      const wrap = $(`.field[data-f="${f}"]`);
      if (!wrap) return;
      wrap.classList.toggle("invalid", !!msg);
      const errEl = $(".err", wrap);
      if (errEl) { errEl.hidden = !msg; errEl.textContent = msg || ""; }
      if (msg) ok = false;
    };
    setErr("email", !payload.email ? "Email is required" : !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(payload.email) ? "Enter a valid email" : "");
    setErr("password", !payload.password ? "Password is required" : payload.password.length < 6 ? "Minimum 6 characters required" : "");
    if (mode === "signup") setErr("name", !payload.name?.trim() ? "Full name is required" : "");
    if (!ok) return;

    try {
      const res = await api(mode === "login" ? "/api/auth/login" : "/api/auth/signup", { method: "POST", body: payload });
      state.token = res.token;
      state.user = { name: res.name, email: res.email };
      store.set("cc_token", res.token);
      store.set("cc_user", JSON.stringify(state.user));
      toast(mode === "login" ? `Welcome back, ${res.name.split(" ")[0]}!` : "Account initialized — let's configure your twin");
      navOrRoute(res.has_profile ? "#/dashboard" : "#/onboarding");
    } catch (err) { toast(err.message, true); }
  };
}

/* ============================================================
   FASCINATING & DELIGHTFUL QUESTIONNAIRE WIZARD
   ============================================================ */
const WIZ_STEPS = [
  { title: "Demographics", sub: "Body & Biometrics", icon: "👤" },
  { title: "Clinical History", sub: "Medical Prior Factors", icon: "🩺" },
  { title: "Diet & Lifestyle", sub: "Nutritional Patterns", icon: "🥗" },
  { title: "Smoking & Labs", sub: "Baselines & Telemetry", icon: "🧬" },
];

const FREQ_OPTIONS = [
  { val: "never", label: "Never", emoji: "⚪" },
  { val: "rarely", label: "Rarely", emoji: "🟡" },
  { val: "sometimes", label: "Moderate", emoji: "🟠" },
  { val: "often", label: "Frequent", emoji: "🔴" },
  { val: "very_often", label: "Daily", emoji: "🟣" },
];

const ETHNICITIES = [
  ["other", "Other / Prefer not to specify", "Global average baseline priors"],
  ["european", "European Ancestry", "Standard baseline calibration"],
  ["south_asian", "South Asian Ancestry", "Calibrated for atherogenic particle variance (F52)"],
  ["east_asian", "East Asian Ancestry", "Calibrated for microvascular sensitivity"],
  ["african", "African Ancestry", "Calibrated for arterial hypertension sensitivity"],
  ["hispanic", "Hispanic / Latino", "Calibrated for metabolic lipid spectrum"],
  ["middle_eastern", "Middle Eastern Ancestry", "Calibrated for lipid & glycemic dynamics"],
];

const CLINICAL_CONDITIONS = [
  { key: "diabetic", title: "Diabetes / Pre-diabetes", sub: "Elevated fasting blood sugar or HbA1c", tag: "Tunes F55 + HRV RMSSD (F22)", icon: "🩺" },
  { key: "high_chol", title: "Diagnosed High Cholesterol", sub: "History of elevated LDL or total cholesterol", tag: "Tunes F56 + Lipid Synthesis Rate", icon: "🫀" },
  { key: "hypertension", title: "Essential Hypertension", sub: "Blood pressure consistently > 130/80 mmHg", tag: "Tunes F57 + Arterial Stiffness (PWV)", icon: "💓" },
  { key: "clot", title: "Thromboembolism / Blood Clot", sub: "History of DVT, pulmonary embolism, or clotting", tag: "Tunes F58 + D-Dimer Baseline", icon: "🩸" },
  { key: "thyroid", title: "Thyroid Dysregulation", sub: "Hypo- or hyper-thyroidism condition", tag: "Tunes F59 + Basal Metabolic Load", icon: "🦋" },
  { key: "family_history", title: "Family History of Heart Disease", sub: "First-degree relative with premature cardiovascular event", tag: "Tunes F60 + Genetic Prior Weight", icon: "🧬" },
];

const DIET_ITEMS = [
  { key: "sat_fat_freq", title: "Saturated Fats", desc: "Fried foods, fatty red meat, butter, palm oil, processed meats", icon: "🍔" },
  { key: "sugar_freq", title: "Refined Sugars & Fast Carbs", desc: "Sweets, sodas, pastries, white bread, ultra-processed snacks", icon: "🍰" },
  { key: "veg_freq", title: "Fresh Vegetables & High-Fiber", desc: "Leafy greens, broccoli, legumes, berries, whole grains", icon: "🥦", protective: true },
  { key: "alcohol_freq", title: "Alcohol Intake", desc: "Beer, wine, spirits, cocktails", icon: "🍷" },
  { key: "omega3_freq", title: "Omega-3 Rich Nutrients", desc: "Fatty fish (salmon, mackerel), flaxseeds, chia, walnuts, olive oil", icon: "🐟", protective: true },
];

function renderWizard() {
  if (!state.token) return nav("#/auth");
  document.title = "CardioCore · Digital Twin Calibration";
  const step = state._step ?? 0;
  const p = state.profile || {};
  const F = (n, d) => p[n] ?? d;
  const smokeStatus = p.smoke_status || "never";
  const progressPct = ((step + 1) / 4) * 100;

  const stepBody = [
    /* ---------------- Step 1: Body & Demographics ---------------- */
    `
    <div class="fade-in">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <h3 style="font-size:19px;font-weight:800;color:var(--ink-primary)">👤 Body Composition & Demographics</h3>
          <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Calibrates your baseline arterial volume, hemodynamic impedance, and metabolic rate.</p>
        </div>
        <span class="badge badge-ai">✨ F48–F54 Factors</span>
      </div>

      <!-- Biological Sex Card Selector -->
      <div style="margin-bottom:20px">
        <label style="font-size:13px;font-weight:700;color:var(--ink-primary);display:block;margin-bottom:8px">Biological Sex assigned at birth</label>
        <div class="visual-selector-grid">
          <div class="visual-tile ${F("sex", "male") === "male" ? "selected" : ""}" data-set-sex="male">
            <div class="tile-icon">👨</div>
            <div class="tile-title">Male</div>
            <div class="tile-desc">Calibrates androgenic lipid transport and coronary baseline curves.</div>
            <input type="radio" name="sex" value="male" ${F("sex", "male") === "male" ? "checked" : ""}/>
          </div>
          <div class="visual-tile ${F("sex", "male") === "female" ? "selected" : ""}" data-set-sex="female">
            <div class="tile-icon">👩</div>
            <div class="tile-title">Female</div>
            <div class="tile-desc">Calibrates estrogenic cardioprotective baseline priors and HDL kinetics.</div>
            <input type="radio" name="sex" value="female" ${F("sex", "male") === "female" ? "checked" : ""}/>
          </div>
        </div>
      </div>

      <!-- Measurements Grid -->
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px">
        <div class="field" data-f="age">
          <label>Chronological Age (years)</label>
          <input name="age" type="number" min="18" max="100" value="${F("age", 40)}" required />
          <div class="err" hidden></div>
        </div>

        <div class="field" data-f="height_cm">
          <label>Height (cm)</label>
          <input name="height_cm" type="number" min="100" max="230" step="1" value="${F("height_cm", 175)}" required />
          <div class="err" hidden></div>
        </div>

        <div class="field" data-f="weight_kg">
          <label>Weight (kg)</label>
          <input name="weight_kg" type="number" min="30" max="300" step="0.5" value="${F("weight_kg", 72)}" required />
          <div class="err" hidden></div>
        </div>

        <div class="field" data-f="waist_cm">
          <label>Waist Circumference (cm) <span style="font-weight:400;color:var(--ink-muted)">(Optional)</span></label>
          <input name="waist_cm" type="number" min="50" max="200" step="1" value="${F("waist_cm", "")}" placeholder="e.g. 86" />
          <div class="err" hidden></div>
        </div>

        <div class="field" data-f="hip_cm">
          <label>Hip Circumference (cm) <span style="font-weight:400;color:var(--ink-muted)">(Optional)</span></label>
          <input name="hip_cm" type="number" min="50" max="220" step="1" value="${F("hip_cm", "")}" placeholder="e.g. 96" />
          <div class="err" hidden></div>
        </div>

        <div class="field" data-f="ethnicity">
          <label>Ancestral Background</label>
          <select name="ethnicity">
            ${ETHNICITIES.map(([v, l]) => `<option value="${v}" ${F("ethnicity", "other") === v ? "selected" : ""}>${l}</option>`).join("")}
          </select>
        </div>
      </div>

      <!-- Real-Time Hologram Biometrics Preview Box -->
      <div class="live-biometrics-preview-box">
        <div class="live-bio-metric-item">
          <span class="live-bio-label">Calculated BMI</span>
          <span class="live-bio-value" id="wiz-live-bmi">—</span>
        </div>
        <div class="live-bio-metric-item">
          <span class="live-bio-label">Classification</span>
          <span id="wiz-live-bmi-badge" class="badge badge-neutral">—</span>
        </div>
        <div class="live-bio-metric-item">
          <span class="live-bio-label">Waist-to-Hip Ratio</span>
          <span class="live-bio-value" id="wiz-live-whr">—</span>
        </div>
        <div class="live-bio-metric-item">
          <span class="live-bio-label">Cardiovascular Risk Status</span>
          <span id="wiz-live-whr-badge" class="badge badge-neutral">—</span>
        </div>
      </div>
    </div>`,

    /* ---------------- Step 2: Medical History ---------------- */
    `
    <div class="fade-in">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <h3 style="font-size:19px;font-weight:800;color:var(--ink-primary)">🩺 Clinical & Medical History</h3>
          <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Click any condition card to toggle its presence. These tune your twin's clinical baseline priors.</p>
        </div>
        <span class="badge badge-ai">✨ F55–F60 Factors</span>
      </div>

      <div class="clinical-matrix-grid">
        ${CLINICAL_CONDITIONS.map((cond) => {
          const isActive = !!F(cond.key, false);
          return `
          <div class="clinical-condition-tile ${isActive ? "active" : ""}" data-toggle-cond="${cond.key}">
            <div class="cond-info-side">
              <span class="cond-icon">${cond.icon}</span>
              <div>
                <div class="cond-title">${cond.title}</div>
                <div class="cond-sub">${cond.sub}</div>
                <span class="cond-equation-tag">${cond.tag}</span>
              </div>
            </div>
            <div class="cond-switch"></div>
            <input type="checkbox" name="${cond.key}" ${isActive ? "checked" : ""} style="display:none" />
          </div>`;
        }).join("")}
      </div>
    </div>`,

    /* ---------------- Step 3: Diet & Lifestyle ---------------- */
    `
    <div class="fade-in">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <h3 style="font-size:19px;font-weight:800;color:var(--ink-primary)">🥗 Dietary & Lifestyle Spectrum</h3>
          <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Select your average weekly consumption frequency for each nutritional category.</p>
        </div>
        <span class="badge badge-ai">✨ F61–F65 Factors</span>
      </div>

      <div style="display:flex;flex-direction:column;gap:12px">
        ${DIET_ITEMS.map((item) => {
          const currentVal = F(item.key, "sometimes");
          return `
          <div class="diet-scale-card" data-diet-key="${item.key}">
            <div class="diet-card-head">
              <div>
                <div class="diet-card-name"><span>${item.icon}</span> ${item.title}</div>
                <div class="diet-card-desc">${item.desc}</div>
              </div>
              <span class="badge ${item.protective ? "badge-low" : "badge-neutral"}">${item.protective ? "Protective Factor" : "Risk Factor"}</span>
            </div>
            <div class="freq-segment-bar">
              ${FREQ_OPTIONS.map((opt) => `
                <div class="freq-seg-opt ${currentVal === opt.val ? "selected" : ""}" data-val="${opt.val}">
                  ${opt.emoji} ${opt.label}
                </div>
              `).join("")}
            </div>
            <input type="hidden" name="${item.key}" value="${currentVal}" />
          </div>`;
        }).join("")}
      </div>
    </div>`,

    /* ---------------- Step 4: Smoking & Optional Lab Baselines ---------------- */
    `
    <div class="fade-in">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <h3 style="font-size:19px;font-weight:800;color:var(--ink-primary)">🧬 Smoking Status & Lab Baselines</h3>
          <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Smoking calibrates vascular oxidative factors. Lab numbers are strictly optional.</p>
        </div>
        <span class="badge badge-ai">✨ F66–F72 Factors</span>
      </div>

      <!-- Smoking Visual Tile Selector -->
      <label style="font-size:13px;font-weight:700;color:var(--ink-primary);display:block;margin-bottom:8px">Smoking History Status</label>
      <div class="visual-selector-grid" style="margin-bottom:16px">
        <div class="visual-tile ${smokeStatus === "never" ? "selected" : ""}" data-set-smoke="never">
          <div class="tile-icon">🚭</div>
          <div class="tile-title">Non-Smoker</div>
          <div class="tile-desc">No active smoking history. Zero pack-year multiplier.</div>
          <input type="radio" name="smoke_status" value="never" ${smokeStatus === "never" ? "checked" : ""}/>
        </div>
        <div class="visual-tile ${smokeStatus === "current" ? "selected" : ""}" data-set-smoke="current">
          <div class="tile-icon">🚬</div>
          <div class="tile-title">Active Smoker</div>
          <div class="tile-desc">Currently smoking. Daily intensity & pack-years calculated.</div>
          <input type="radio" name="smoke_status" value="current" ${smokeStatus === "current" ? "checked" : ""}/>
        </div>
        <div class="visual-tile ${smokeStatus === "former" ? "selected" : ""}" data-set-smoke="former">
          <div class="tile-icon">⏳</div>
          <div class="tile-title">Former Smoker</div>
          <div class="tile-desc">Previous history retained as historical baseline risk.</div>
          <input type="radio" name="smoke_status" value="former" ${smokeStatus === "former" ? "checked" : ""}/>
        </div>
      </div>

      <div id="wiz-smoke-inputs" style="display:${smokeStatus === "never" ? "none" : "grid"};grid-template-columns:1fr 1fr;gap:16px;background:#FFF7ED;border:1px solid #FED7AA;border-radius:var(--radius-md);padding:16px;margin-bottom:20px">
        <div class="field" data-f="cigarettes_per_day">
          <label>Average Cigarettes / Day</label>
          <input name="cigarettes_per_day" type="number" min="0" max="80" step="1" value="${F("cigarettes_per_day", "")}" placeholder="e.g. 10" />
          <div class="err" hidden></div>
        </div>
        <div class="field" data-f="years_smoked">
          <label>Total Years Smoked</label>
          <input name="years_smoked" type="number" min="0" max="70" step="1" value="${F("years_smoked", "")}" placeholder="e.g. 12" />
          <div class="err" hidden></div>
        </div>
        <div id="wiz-pack-summary" style="grid-column:span 2;font-size:12.5px;font-weight:700;color:#C2410C"></div>
      </div>

      <!-- Known Lab Vault -->
      <div style="background:#F8FAFC;border:1px solid var(--line-subtle);border-radius:var(--radius-md);padding:20px;margin-top:12px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div>
            <h4 style="font-size:15px;font-weight:800;color:var(--ink-primary)">🧪 Known Clinical Lab Baselines <span style="font-size:12px;font-weight:500;color:var(--ink-muted)">(Optional)</span></h4>
            <div style="font-size:12.5px;color:var(--ink-muted)">If you have recent blood test or wearable numbers, enter them to accelerate baseline convergence.</div>
          </div>
          <span class="badge badge-neutral">Skip if unknown</span>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px">
          <div class="field"><label>Resting HR (bpm)</label><input name="baseline_resting_hr" type="number" value="${F("baseline_resting_hr", "")}" placeholder="e.g. 62" /></div>
          <div class="field"><label>HRV RMSSD (ms)</label><input name="baseline_hrv_rmssd" type="number" value="${F("baseline_hrv_rmssd", "")}" placeholder="e.g. 48" /></div>
          <div class="field"><label>Systolic BP (mmHg)</label><input name="baseline_systolic" type="number" value="${F("baseline_systolic", "")}" placeholder="e.g. 120" /></div>
          <div class="field"><label>Diastolic BP (mmHg)</label><input name="baseline_diastolic" type="number" value="${F("baseline_diastolic", "")}" placeholder="e.g. 80" /></div>
          <div class="field"><label>Total Cholesterol (mg/dL)</label><input name="baseline_total_cholesterol" type="number" value="${F("baseline_total_cholesterol", "")}" placeholder="e.g. 195" /></div>
          <div class="field"><label>HDL Cholesterol (mg/dL)</label><input name="baseline_hdl" type="number" value="${F("baseline_hdl", "")}" placeholder="e.g. 55" /></div>
          <div class="field"><label>Triglycerides (mg/dL)</label><input name="baseline_triglycerides" type="number" value="${F("baseline_triglycerides", "")}" placeholder="e.g. 140" /></div>
        </div>
      </div>
    </div>`,
  ][step];

  app.innerHTML = `
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand-wrap">
        <div class="brand-icon">❤️</div>
        <div class="brand-text">
          <div class="brand-name">CardioCore</div>
          <div class="brand-sub">Digital Twin Calibration</div>
        </div>
      </div>
      <button class="btn btn-ghost" id="wiz-logout">Sign out</button>
    </div>
  </header>

  <div class="container">
    <div class="wiz-container fade-in">
      <!-- Wizard Progress & Calibration Header -->
      <div class="wiz-hero-header">
        <div class="wiz-hero-title">
          <h2><span>🧬</span> ${state.profile ? "Personalize Digital Twin Parameters" : "Calibrate Your Cardiovascular Twin"}</h2>
          <p>Assimilating 19 patient-side physiological parameters into your 72-factor digital twin.</p>
        </div>

        <div class="wiz-calib-pill">
          <span style="font-size:12px;font-weight:700;color:var(--brand)">Step ${step + 1} / 4</span>
          <div class="wiz-calib-progress-bar">
            <div class="wiz-calib-fill" style="width:${progressPct}%"></div>
          </div>
          <span style="font-size:12px;font-weight:700;color:var(--ink-muted)">${progressPct.toFixed(0)}%</span>
        </div>
      </div>

      <!-- Interactive Steps Navigation Bar -->
      <div class="wiz-steps-bar">
        ${WIZ_STEPS.map((s, i) => `
          <div class="wiz-step-btn ${i === step ? "active" : i < step ? "done" : ""}" data-goto-step="${i}">
            <div class="wiz-step-icon">${i < step ? "✓" : s.icon}</div>
            <div>
              <div class="wiz-step-label">${s.title}</div>
              <div class="wiz-step-sub">${s.sub}</div>
            </div>
          </div>
        `).join("")}
      </div>

      <!-- Main Form Card -->
      <div class="card card-panel">
        <form id="wiz-form" novalidate>${stepBody}</form>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:32px;padding-top:20px;border-top:1px solid var(--line-subtle)">
          <button class="btn btn-ghost" id="wiz-back" ${step === 0 ? "disabled" : ""}>← Previous Step</button>
          <button class="btn btn-primary" id="wiz-next">
            ${step === 3 ? (state.profile ? "Save & Re-synthesize Twin ✨" : "Launch My Digital Twin ✨") : "Continue to Next Step →"}
          </button>
        </div>
      </div>
    </div>
  </div>`;

  $("#wiz-logout").onclick = () => doLogout();

  // Direct step bar clicks if steps are visited
  $$("[data-goto-step]").forEach((btn) => {
    btn.onclick = () => {
      const targetStep = +btn.dataset.gotoStep;
      if (targetStep < step || validateStep(step)) {
        collectForm();
        state._step = targetStep;
        renderWizard();
        window.scrollTo(0, 0);
      }
    };
  });

  // Dynamic Sex Tile Selector
  $$("[data-set-sex]").forEach((tile) => {
    tile.onclick = () => {
      const sexVal = tile.dataset.setSex;
      $$("[data-set-sex]").forEach((t) => t.classList.toggle("selected", t === tile));
      const radio = tile.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
      updLiveBiometrics();
    };
  });

  // Dynamic Medical Condition Tiles Toggle
  $$("[data-toggle-cond]").forEach((tile) => {
    tile.onclick = () => {
      const chk = tile.querySelector('input[type="checkbox"]');
      if (chk) {
        chk.checked = !chk.checked;
        tile.classList.toggle("active", chk.checked);
      }
    };
  });

  // Dynamic Dietary Segment Bar Clicks
  $$(".diet-scale-card").forEach((card) => {
    const hiddenInput = card.querySelector('input[type="hidden"]');
    const opts = card.querySelectorAll(".freq-seg-opt");
    opts.forEach((opt) => {
      opt.onclick = () => {
        opts.forEach((o) => o.classList.toggle("selected", o === opt));
        if (hiddenInput) hiddenInput.value = opt.dataset.val;
      };
    });
  });

  // Dynamic Smoking Selector
  $$("[data-set-smoke]").forEach((tile) => {
    tile.onclick = () => {
      const smokeVal = tile.dataset.setSmoke;
      $$("[data-set-smoke]").forEach((t) => t.classList.toggle("selected", t === tile));
      const radio = tile.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
      const smokeInputs = $("#wiz-smoke-inputs");
      if (smokeInputs) smokeInputs.style.display = smokeVal === "never" ? "none" : "grid";
      updSmokeSummary();
    };
  });

  // Live BMI & WHR Hologram Calculation
  function updLiveBiometrics() {
    const h = +$('[name="height_cm"]')?.value, w = +$('[name="weight_kg"]')?.value;
    const wa = +$('[name="waist_cm"]')?.value, hi = +$('[name="hip_cm"]')?.value;
    const bmiEl = $("#wiz-live-bmi"), bmiBadge = $("#wiz-live-bmi-badge");
    const whrEl = $("#wiz-live-whr"), whrBadge = $("#wiz-live-whr-badge");

    if (bmiEl && bmiBadge) {
      if (h > 100 && w > 30) {
        const bmi = w / ((h / 100) ** 2);
        bmiEl.textContent = bmi.toFixed(1);
        const cls = bmi < 18.5 ? "Underweight" : bmi < 25 ? "Optimal" : bmi < 30 ? "Overweight" : "Obese";
        const badgeCol = bmi < 25 ? "badge-low" : bmi < 30 ? "badge-mod" : "badge-high";
        bmiBadge.className = `badge ${badgeCol}`;
        bmiBadge.innerHTML = `<span class="badge-dot"></span>${cls}`;
      } else {
        bmiEl.textContent = "—";
        bmiBadge.textContent = "Awaiting inputs";
      }
    }

    if (whrEl && whrBadge) {
      if (wa > 50 && hi > 50) {
        const whr = wa / hi;
        whrEl.textContent = whr.toFixed(2);
        const risk = whr <= 0.85 ? "Optimal" : whr <= 0.95 ? "Moderate" : "Elevated";
        const badgeCol = whr <= 0.85 ? "badge-low" : whr <= 0.95 ? "badge-mod" : "badge-high";
        whrBadge.className = `badge ${badgeCol}`;
        whrBadge.innerHTML = `<span class="badge-dot"></span>${risk} Ratio`;
      } else {
        whrEl.textContent = "—";
        whrBadge.textContent = "Optional";
      }
    }
  }

  // Smoking summary calculation
  function updSmokeSummary() {
    const st = ($('[name="smoke_status"]:checked') || {}).value || "never";
    const summ = $("#wiz-pack-summary");
    if (!summ) return;
    if (st !== "never") {
      const cigs = +$('[name="cigarettes_per_day"]')?.value || 0;
      const yrs = +$('[name="years_smoked"]')?.value || 0;
      if (cigs > 0 && yrs > 0) {
        const py = (cigs * yrs) / 20;
        summ.innerHTML = `🚭 Cumulative Exposure: <b>${py.toFixed(1)} pack-years</b> (${st === "current" ? "Active arterial oxidative factor" : "Retained prior risk in digital twin"})`;
      } else {
        summ.innerHTML = `Enter daily cigarettes and years to calculate cumulative pack-years.`;
      }
    }
  }

  $('[name="height_cm"]')?.addEventListener("input", updLiveBiometrics);
  $('[name="weight_kg"]')?.addEventListener("input", updLiveBiometrics);
  $('[name="waist_cm"]')?.addEventListener("input", updLiveBiometrics);
  $('[name="hip_cm"]')?.addEventListener("input", updLiveBiometrics);
  $('[name="cigarettes_per_day"]')?.addEventListener("input", updSmokeSummary);
  $('[name="years_smoked"]')?.addEventListener("input", updSmokeSummary);
  if (step === 0) updLiveBiometrics();
  if (step === 3) updSmokeSummary();

  // Navigation handlers
  $("#wiz-back").onclick = () => {
    collectForm();
    state._step = step - 1;
    renderWizard();
    window.scrollTo(0, 0);
  };

  $("#wiz-next").onclick = async () => {
    if (!validateStep(step)) return;
    collectForm();
    if (step < 3) {
      state._step = step + 1;
      renderWizard();
      window.scrollTo(0, 0);
      return;
    }

    // Launch Synthesis Holographic Overlay
    showSynthesisAnimation();

    try {
      await api("/api/profile", { method: "PUT", body: state.profile });
      setTimeout(() => {
        state._step = 0;
        state.result = null; // force fresh simulation
        hideSynthesisAnimation();
        toast("Digital Twin successfully synthesized!");
        navOrRoute("#/dashboard");
      }, 1600);
    } catch (err) {
      hideSynthesisAnimation();
      toast(err.message, true);
    }
  };

  function collectForm() {
    const fd = new FormData($("#wiz-form"));
    const numeric = new Set([
      "age", "height_cm", "weight_kg", "waist_cm", "hip_cm",
      "cigarettes_per_day", "years_smoked", "baseline_resting_hr", "baseline_hrv_rmssd",
      "baseline_systolic", "baseline_diastolic", "baseline_total_cholesterol",
      "baseline_hdl", "baseline_triglycerides"
    ]);
    const boolQ = new Set(["diabetic", "high_chol", "hypertension", "clot", "thyroid", "family_history"]);
    const out = { ...(state.profile || {}) };

    for (const [k, v] of fd.entries()) {
      if (numeric.has(k)) { if (v !== "") out[k] = +v; }
      else if (boolQ.has(k)) out[k] = v === "on" || v === "yes" || v === "true";
      else out[k] = v;
    }

    // Ensure un-checked checkboxes are false
    boolQ.forEach((k) => {
      const el = $(`[name="${k}"]`);
      if (el) out[k] = el.checked;
    });

    for (const k of ["waist_cm", "hip_cm", "baseline_resting_hr", "baseline_hrv_rmssd",
      "baseline_systolic", "baseline_diastolic", "baseline_total_cholesterol",
      "baseline_hdl", "baseline_triglycerides", "cigarettes_per_day", "years_smoked"]) {
      if (!(k in out) || out[k] === "" || out[k] === undefined || isNaN(out[k])) delete out[k];
    }
    state.profile = out;
  }

  function validateStep(s) {
    $$('.field[data-f]').forEach((w) => { w.classList.remove("invalid"); const e = $(".err", w); if (e) e.hidden = true; });
    let ok = true;
    const bad = (f, msg) => {
      const el = $(`[name="${f}"]`); if (!el) return;
      const w = el.closest(".field");
      if (w) { w.classList.add("invalid"); const e = $(".err", w); if (e) { e.hidden = false; e.textContent = msg; } }
      ok = false;
    };
    if (s === 0) {
      const age = +$('[name="age"]')?.value, h = +$('[name="height_cm"]')?.value, w = +$('[name="weight_kg"]')?.value;
      if (!(age >= 18 && age <= 100)) bad("age", "Please enter an age between 18 and 100");
      if (!(h > 100 && h <= 230)) bad("height_cm", "Enter valid height (100–230 cm)");
      if (!(w > 30 && w <= 300)) bad("weight_kg", "Enter valid weight (30–300 kg)");
    }
    return ok;
  }
}

function showSynthesisAnimation() {
  const overlay = document.createElement("div");
  overlay.id = "synthesis-loading-overlay";
  overlay.className = "synthesis-overlay";
  overlay.innerHTML = `
    <div class="synthesis-heart">🫀</div>
    <div class="synthesis-title">Calibrating Cardiovascular Twin</div>
    <div class="synthesis-sub">Assimilating 72 physiological channels, initializing EWMA state vectors, and tuning personal baseline curves…</div>
    <div class="synthesis-log-line">
      <span class="pulse-dot"></span>
      <span id="synthesis-step-text">Synthesizing multimodal feature vectors…</span>
    </div>
  `;
  document.body.appendChild(overlay);

  setTimeout(() => {
    const textEl = $("#synthesis-step-text");
    if (textEl) textEl.textContent = "Aligning EWMA baseline distributions (95% CI)...";
  }, 700);

  setTimeout(() => {
    const textEl = $("#synthesis-step-text");
    if (textEl) textEl.textContent = "Ready: Initializing Personal Command Center...";
  }, 1300);
}

function hideSynthesisAnimation() {
  const overlay = $("#synthesis-loading-overlay");
  if (overlay) overlay.remove();
}

/* ============================================================
   SVG VISUALIZATION BUILDERS
   ============================================================ */

/* 1. Circular Animated Risk Score Ring */
function circularScoreSVG(score, col) {
  const r = 88;
  const c = 2 * Math.PI * r;
  const val = Math.max(0, Math.min(1, score));
  const offset = c * (1 - val);

  return `
  <div class="circular-gauge-box">
    <svg viewBox="0 0 210 210">
      <circle class="gauge-bg-ring" cx="105" cy="105" r="${r}" />
      <circle class="gauge-val-ring" cx="105" cy="105" r="${r}"
              stroke="${col}"
              stroke-dasharray="${c.toFixed(1)}"
              stroke-dashoffset="${offset.toFixed(1)}" />
    </svg>
    <div class="gauge-inner-content">
      <div class="gauge-number" style="color:${col}">${(val * 100).toFixed(0)}</div>
      <div class="gauge-total">/ 100</div>
    </div>
  </div>`;
}

/* 2. Interactive Multi-Metric Trend Chart */
function lineChartSVG(series, metricKey, color) {
  const w = 720, h = 200, padX = 24, padY = 24;
  if (!series || !series.length) return `<div style="text-align:center;padding:40px;color:var(--ink-muted)">No trend telemetry available</div>`;

  const ys = series.map((p) => p[1]);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minY === maxY) { minY *= 0.9; maxY *= 1.1; }
  const rangeY = (maxY - minY) || 1;

  const X = (i) => padX + (i / Math.max(1, series.length - 1)) * (w - 2 * padX);
  const Y = (v) => h - padY - ((v - minY) / rangeY) * (h - 2 * padY);

  const points = series.map((p, i) => `${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
  const areaPoints = `${padX},${h - padY} ${points} ${w - padX},${h - padY}`;

  // Grid lines
  const midVal = (minY + maxY) / 2;

  return `
  <svg viewBox="0 0 ${w} ${h}" width="100%" style="overflow:visible;display:block">
    <defs>
      <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${color}" stop-opacity="0.2" />
        <stop offset="100%" stop-color="${color}" stop-opacity="0.0" />
      </linearGradient>
    </defs>

    <!-- Grid lines -->
    <line x1="${padX}" y1="${Y(maxY)}" x2="${w - padX}" y2="${Y(maxY)}" stroke="#E2E8F0" stroke-dasharray="4 4" />
    <line x1="${padX}" y1="${Y(midVal)}" x2="${w - padX}" y2="${Y(midVal)}" stroke="#F1F5F9" />
    <line x1="${padX}" y1="${Y(minY)}" x2="${w - padX}" y2="${Y(minY)}" stroke="#E2E8F0" />

    <!-- Area & Line -->
    <polygon points="${areaPoints}" fill="url(#areaGrad)" />
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />

    <!-- End point dot -->
    <circle cx="${X(series.length - 1)}" cy="${Y(ys[ys.length - 1])}" r="5" fill="${color}" stroke="#FFF" stroke-width="2" />

    <!-- Min/Max labels -->
    <text x="${padX}" y="${h - 6}" font-size="11" font-weight="600" fill="#94A3B8">Day 1</text>
    <text x="${w - padX}" y="${h - 6}" text-anchor="end" font-size="11" font-weight="600" fill="#94A3B8">Day ${series.length} (Latest)</text>
    <text x="${w - padX + 8}" y="${Y(ys[ys.length - 1]) + 4}" font-size="12" font-weight="800" fill="${color}">${ys[ys.length - 1].toFixed(metricKey === "cvd" ? 2 : 0)}</text>
  </svg>`;
}

/* 3. Biomarker Risk Radar Chart */
function radarSVG(bio) {
  const axes = ["TC", "LDL", "HDL", "TG", "CRP", "DD"];
  const norm = (k) => {
    const v = bio[k] ?? 0;
    const m = {
      TC: (v - 140) / 160,
      LDL: (v - 40) / 200,
      HDL: 1 - (v - 20) / 60,
      TG: (v - 70) / 380,
      CRP: (v - 0.2) / 8.0,
      DD: (v - 0.1) / 2.5
    }[k] || 0.5;
    return Math.max(0.08, Math.min(1, m));
  };
  const cx = 130, cy = 120, R = 86;
  const pt = (i, r) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  };
  const ring = (r) => axes.map((_, i) => pt(i, r).map((x) => x.toFixed(1)).join(",")).join(" ");
  const poly = axes.map((k, i) => pt(i, R * norm(k)).map((x) => x.toFixed(1)).join(",")).join(" ");
  const labels = axes.map((k, i) => {
    const [x, y] = pt(i, R + 18);
    return `<text x="${x}" y="${y + 4}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#475569">${k}</text>`;
  }).join("");

  return `
  <svg viewBox="0 0 260 240" width="100%" style="max-width:280px;display:block;margin:0 auto">
    ${[0.33, 0.66, 1].map((f) => `<polygon points="${ring(R * f)}" fill="none" stroke="#E2E8F0" stroke-width="1"/>`).join("")}
    ${axes.map((_, i) => { const [x, y] = pt(i, R); return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#E2E8F0"/>`; }).join("")}
    <polygon points="${poly}" fill="rgba(225,29,72,0.18)" stroke="#E11D48" stroke-width="2.2" stroke-linejoin="round"/>
    ${labels}
  </svg>`;
}

/* 4. PCA Cohort Scatter Chart */
function scatterSVG(scatter, patient, colors) {
  const pts = scatter || [];
  if (!pts.length) return "";
  const xs = pts.map((p) => p.x).concat(patient ? [patient.pc1] : []);
  const ys = pts.map((p) => p.y).concat(patient ? [patient.pc2] : []);
  const x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.3;
  const y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
  const W = 400, H = 260, P = 20;
  const X = (v) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P);
  const Y = (v) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P);

  const dots = pts.map((p) => `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="4" fill="${colors[p.cluster] || "#94A3B8"}" opacity="0.45"/>`).join("");
  const youMarker = patient ? `
    <circle cx="${X(patient.pc1)}" cy="${Y(patient.pc2)}" r="12" fill="none" stroke="#0F172A" stroke-width="2.5" />
    <circle cx="${X(patient.pc1)}" cy="${Y(patient.pc2)}" r="5" fill="#E11D48" />
  ` : "";

  return `
  <svg viewBox="0 0 ${W} ${H}" width="100%">
    <rect width="${W}" height="${H}" fill="#FAFBFC" rx="12" />
    ${dots}
    ${youMarker}
    <text x="${W / 2}" y="${H - 4}" text-anchor="middle" font-size="11" font-weight="600" fill="#94A3B8">PCA Component 1 (Atherogenic Variance) →</text>
  </svg>`;
}

/* ============================================================
   MAIN DASHBOARD RENDERER & COMMAND CENTER
   ============================================================ */
async function renderDashboard() {
  if (!state.token) return nav("#/auth");
  document.title = "CardioCore · Personal Cardiovascular Command Center";
  app.innerHTML = `
    <div class="loading-state-box fade-in">
      <div class="spin"></div>
      <div style="font-weight:700;color:var(--ink-primary)">Synthesizing 72-Factor Digital Twin…</div>
      <div style="font-size:13px;color:var(--ink-muted)">Running continuous multimodal assimilation models</div>
    </div>`;

  try {
    if (!state.result) {
      state.result = await api("/api/twin/simulate", {
        method: "POST",
        body: { scenario: state.scenario }
      });
    }
  } catch (err) {
    toast(err.message, true);
    nav("#/onboarding");
    return;
  }

  drawCommandCenter();
}

function drawCommandCenter() {
  const r = state.result;
  const b = r.biomarkers, cats = r.categories, pat = r.patient || {};
  const cvdCol = catColor(r.cvd_category);
  const ctrCol = catColor(r.ctr_category);
  const trendIcon = { Increasing: "▲", Decreasing: "▼", Stable: "▬" }[r.trend] || "▬";

  /* Baseline Deviation Statistics */
  const zs = r.baselines.filter((x) => x.z != null).map((x) => Math.abs(x.z));
  const meanZ = zs.length ? zs.reduce((a, c) => a + c, 0) / zs.length : 0;
  const devStatus = meanZ < 0.5 ? "Within normal variation" : meanZ < 1.2 ? "Mild baseline deviation" : "Meaningful baseline deviation";
  const devBadge = meanZ < 0.5 ? "badge-low" : meanZ < 1.2 ? "badge-mod" : "badge-high";

  /* Dynamically Calculate Top 3 "What Changed Today?" highlights */
  const changes = [];
  const baseTg = r.baselines.find((x) => x.biomarker === "TG");
  if (baseTg && baseTg.pct_change != null) {
    changes.push({
      metric: "Triglycerides",
      icon: "⚡",
      delta: `${baseTg.pct_change > 0 ? "↑" : "↓"} ${Math.abs(baseTg.pct_change).toFixed(1)}%`,
      type: baseTg.pct_change > 3 ? "warning" : "favorable",
      desc: baseTg.pct_change > 0 ? "Elevated vs. learned baseline" : "Favorable reduction",
      note: `Current estimate: ${fmt(b.TG)} mg/dL (${cats.TG})`
    });
  }
  const baseCrp = r.baselines.find((x) => x.biomarker === "CRP");
  if (baseCrp && baseCrp.pct_change != null) {
    changes.push({
      metric: "C-Reactive Protein",
      icon: "🔥",
      delta: `${baseCrp.pct_change > 0 ? "↑" : "↓"} ${Math.abs(baseCrp.pct_change).toFixed(1)}%`,
      type: Math.abs(baseCrp.pct_change) > 5 ? (baseCrp.pct_change > 0 ? "warning" : "favorable") : "neutral",
      desc: baseCrp.pct_change > 0 ? "Slight increase from baseline" : "Stable inflammatory tone",
      note: `Current estimate: ${fmt(b.CRP, 2)} mg/L (${cats.CRP})`
    });
  }
  const baseLdl = r.baselines.find((x) => x.biomarker === "LDL");
  if (baseLdl && baseLdl.pct_change != null) {
    changes.push({
      metric: "LDL Cholesterol",
      icon: "🫀",
      delta: `${baseLdl.pct_change > 0 ? "↑" : "↓"} ${Math.abs(baseLdl.pct_change).toFixed(1)}%`,
      type: baseLdl.pct_change < 0 ? "favorable" : "warning",
      desc: baseLdl.pct_change < 0 ? "Moving in favorable direction" : "Mild upward shift",
      note: `Current estimate: ${fmt(b.LDL)} mg/dL (${cats.LDL})`
    });
  }

  /* Trend line series */
  const METRIC_TABS = [
    ["cvd", "Risk Score"], ["TC", "TC"], ["HDL", "HDL"],
    ["LDL", "LDL"], ["TG", "TG"], ["CRP", "CRP"], ["DD", "D-Dimer"]
  ];
  const mk = state.trendMetric;
  const series = r.trajectory.map((p) => [p.day, p[mk] ?? p.cvd]);
  const activeSeriesCol = mk === "cvd" ? cvdCol : mk === "HDL" ? "#059669" : mk === "CRP" || mk === "DD" ? "#E11D48" : "#2563EB";

  /* Cluster colors */
  const clusterColors = {
    "Low Risk": "#059669",
    "Moderate Risk": "#D97706",
    "High Risk": "#EA580C",
    "Very High Risk": "#DC2626"
  };

  /* XAI top drivers */
  const drivers = (r.top_contributions?.TC || []).slice(0, 5);

  app.innerHTML = `
  <!-- Navigation Topbar -->
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand-wrap" id="nav-brand">
        <div class="brand-icon">❤️</div>
        <div class="brand-text">
          <div class="brand-name">CardioCore</div>
          <div class="brand-sub">Cardiovascular Command Center</div>
        </div>
      </div>

      <nav class="nav-links" role="navigation" aria-label="Main Navigation">
        <button class="nav-item ${state.activeView === "overview" ? "active" : ""}" data-tab="overview">Overview</button>
        <button class="nav-item ${state.activeView === "twin" ? "active" : ""}" data-tab="twin">My Twin</button>
        <button class="nav-item ${state.activeView === "trends" ? "active" : ""}" data-tab="trends">Trends</button>
        <button class="nav-item ${state.activeView === "factors" ? "active" : ""}" data-tab="factors">Factors (72)</button>
        <button class="nav-item ${state.activeView === "alerts" ? "active" : ""}" data-tab="alerts">
          Alerts ${r.alerts.length ? `<span class="badge badge-high" style="padding:2px 6px;font-size:10px">${r.alerts.length}</span>` : ""}
        </button>
        <button class="nav-item ${state.activeView === "profile" ? "active" : ""}" data-tab="profile">Profile</button>
      </nav>

      <div class="topbar-actions">
        <div class="sensor-status-chip" id="sensor-chip-btn" title="View connected sensors">
          <span class="pulse-dot"></span>
          <span>9/9 Connected</span>
          <span class="sync-text" style="color:var(--ink-muted);font-weight:400">· 2m ago</span>
        </div>

        <button class="user-avatar-btn" id="profile-avatar-btn">
          <div class="avatar-circle">${esc((state.user?.name || "P")[0].toUpperCase())}</div>
          <span class="avatar-name">${esc((state.user?.name || "Patient").split(" ")[0])}</span>
        </button>

        <button class="btn btn-ghost" id="top-logout-btn" style="padding:7px 12px;font-size:12.5px">Sign out</button>
      </div>
    </div>
  </header>

  <!-- Mobile Bottom Navigation -->
  <div class="mobile-nav">
    <button class="mob-nav-btn ${state.activeView === "overview" ? "active" : ""}" data-tab="overview">
      <svg fill="currentColor" viewBox="0 0 24 24"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>Overview
    </button>
    <button class="mob-nav-btn ${state.activeView === "twin" ? "active" : ""}" data-tab="twin">
      <svg fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>Twin
    </button>
    <button class="mob-nav-btn ${state.activeView === "trends" ? "active" : ""}" data-tab="trends">
      <svg fill="currentColor" viewBox="0 0 24 24"><path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/></svg>Trends
    </button>
    <button class="mob-nav-btn ${state.activeView === "factors" ? "active" : ""}" data-tab="factors">
      <svg fill="currentColor" viewBox="0 0 24 24"><path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/></svg>Factors
    </button>
    <button class="mob-nav-btn ${state.activeView === "alerts" ? "active" : ""}" data-tab="alerts">
      <svg fill="currentColor" viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6v-5c0-3.07-1.63-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>Alerts
    </button>
  </div>

  <main class="container fade-in">

    <!-- 5. PERSONALIZED GREETING HERO -->
    <div class="card hero-greeting">
      <div class="greeting-main">
        <h1>Good ${new Date().getHours() < 12 ? "morning" : new Date().getHours() < 18 ? "afternoon" : "evening"}, ${esc((state.user?.name || "Patient").split(" ")[0])} 👋</h1>
        <div class="greeting-sub">Your cardiovascular twin is <b>stable today</b>. Continuous wearable telemetry assimilated.</div>
        <div class="meta-info">
          <span>🕒 Last synchronized 2 mins ago</span>
          <span>👤 ${pat.age ?? 40} yrs · ${esc(pat.sex || "Male").toUpperCase()} · BMI ${pat.bmi ?? 23.5}</span>
          <span>🧬 60-day baseline memory</span>
        </div>
      </div>
      <div class="greeting-chips">
        <span class="badge badge-low"><span class="badge-dot"></span>Wearable connected</span>
        <span class="badge badge-low"><span class="badge-dot"></span>Digital Twin active</span>
      </div>
    </div>

    <!-- 6. MAIN CARDIOVASCULAR RISK CENTERPIECE -->
    <div class="hero-grid">
      <!-- Central Circular Risk Score -->
      <div class="card hero-main-card card-hover">
        <div class="card-heading">
          <h3>❤️ Cardiovascular Risk Score</h3>
          <button class="btn-icon" data-help="cvd" title="What does this mean?">ℹ️</button>
        </div>

        ${circularScoreSVG(r.cvd_score, cvdCol)}

        <div class="gauge-status-badge">
          <span class="badge ${levelBadgeClass(r.cvd_score)}" style="font-size:13px;padding:6px 14px">
            <span class="badge-dot"></span>${esc(r.cvd_category).toUpperCase()} RISK
          </span>
        </div>

        <div class="risk-insight-text">
          Stable compared with your personal baseline
        </div>
        <div class="risk-delta-chip">
          <span>${trendIcon} ${esc(r.trend || "Stable")}</span>
          <span>·</span>
          <span>Within normal expected variation</span>
        </div>
      </div>

      <!-- Secondary: Coronary Thrombosis Risk -->
      <div class="card hero-secondary-card card-hover">
        <div class="sec-card-top">
          <div class="metric-label">
            <span>🩸 Coronary Thrombosis Risk (CTR)</span>
            <button class="btn-icon" data-help="ctr">ℹ️</button>
          </div>
          <div class="metric-value" style="color:${ctrCol}">
            ${(r.ctr * 100).toFixed(0)}<span class="metric-unit"> / 100</span>
          </div>
          <div><span class="badge ${levelBadgeClass(r.ctr)}">${esc(r.ctr_category)}</span></div>
        </div>

        <div class="sec-card-middle">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink-muted)">
            <span>Thrombotic Susceptibility</span>
            <span>${(r.ctr * 100).toFixed(0)}%</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width:${r.ctr * 100}%;background:${ctrCol}"></div>
          </div>
          <div style="font-size:11.5px;color:var(--ink-muted);margin-top:4px">Model: LDL (35%) + D-Dimer (40%) + CRP (25%)</div>
        </div>

        <div class="sec-card-bottom">
          <div>🔥 CRP: <b>${fmt(b.CRP, 2)} mg/L</b> (${esc(cats.CRP)})</div>
          <div>🩸 D-Dimer: <b>${fmt(b.DD, 2)} mg/L</b> (${esc(cats.DD)})</div>
        </div>
      </div>

      <!-- Secondary: Digital Twin Vitality & Baseline Comparison -->
      <div class="card hero-secondary-card card-hover">
        <div class="sec-card-top">
          <div class="metric-label">
            <span>🎯 Personal Baseline Status</span>
            <button class="btn-icon" data-help="baseline">ℹ️</button>
          </div>
          <div class="metric-value" style="color:var(--ink-primary)">
            ${fmt(b.TC)}<span class="metric-unit"> mg/dL TC</span>
          </div>
          <div><span class="badge ${devBadge}"><span class="badge-dot"></span>${devStatus}</span></div>
        </div>

        <div class="sec-card-middle">
          <div style="font-size:12.5px;color:var(--ink-secondary);line-height:1.5">
            Your twin compares today's measurements against <b>your own normal</b>, not generalized population curves.
          </div>
        </div>

        <div class="sec-card-bottom">
          <div>🧭 Population Cluster: <b>${esc(r.cluster.label)}</b></div>
          <div>✨ Model Confidence: <b>${(r.cluster.confidence * 100).toFixed(0)}%</b></div>
        </div>
      </div>
    </div>

    <!-- 7. "WHAT CHANGED TODAY?" SECTION -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>🔍 What changed today?</h2>
          <p>The 3 most clinically meaningful physiological shifts compared with your learned baseline.</p>
        </div>
      </div>

      <div class="what-changed-grid">
        ${changes.map((c) => `
          <div class="card change-card ${c.type} card-hover">
            <div class="change-top">
              <div class="change-metric"><span>${c.icon}</span> ${esc(c.metric)}</div>
              <div class="change-delta ${c.type === "warning" ? "up-warn" : c.type === "favorable" ? "down-fav" : "neutral"}">${c.delta}</div>
            </div>
            <div class="change-desc">${esc(c.desc)}</div>
            <div class="change-clinical-note">${esc(c.note)}</div>
          </div>
        `).join("")}
      </div>
    </section>

    <!-- 8. DIGITAL TWIN INTERACTIVE VISUALIZATION -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>🧬 Your Cardiovascular Digital Twin</h2>
          <p>Continuous multimodal state assimilation across six distinct cardiovascular biological pathways.</p>
        </div>
        <button class="btn btn-ghost" data-tab="twin">Deep Dive into Twin →</button>
      </div>

      <div class="twin-visual-layout">
        <!-- Interactive Organ & Pathway Network Map -->
        <div class="card twin-canvas-card">
          <div class="twin-node-network">
            <!-- Center Twin Core -->
            <div class="twin-center-node" id="twin-core-node" title="CardioCore Digital Twin Core">
              <div class="icon">🫀</div>
              <div class="title">Digital Twin</div>
              <span class="badge badge-low" style="padding:2px 6px;font-size:9.5px;margin-top:2px">Active</span>
            </div>

            <!-- 6 Orbiting Pathway Satellite Nodes -->
            <div class="pathway-satellite-node" style="top:12px;left:28px" data-pathway="lipid">
              <span>🫀</span> Lipid <span class="badge ${levelBadgeClass(r.pathways.lipid)}">${r.pathways.lipid.toFixed(2)}</span>
            </div>
            <div class="pathway-satellite-node" style="top:12px;right:28px" data-pathway="inflammation">
              <span>🔥</span> Inflam <span class="badge ${levelBadgeClass(r.pathways.inflammation)}">${r.pathways.inflammation.toFixed(2)}</span>
            </div>
            <div class="pathway-satellite-node" style="top:150px;right:8px" data-pathway="thrombosis">
              <span>🩸</span> Thromb <span class="badge ${levelBadgeClass(r.pathways.thrombosis)}">${r.pathways.thrombosis.toFixed(2)}</span>
            </div>
            <div class="pathway-satellite-node" style="bottom:16px;right:34px" data-pathway="hemodynamic">
              <span>💓</span> Hemodyn <span class="badge ${levelBadgeClass(r.pathways.hemodynamic)}">${r.pathways.hemodynamic.toFixed(2)}</span>
            </div>
            <div class="pathway-satellite-node" style="bottom:16px;left:34px" data-pathway="autonomic">
              <span>🧠</span> Autonomic <span class="badge ${levelBadgeClass(r.pathways.autonomic)}">${r.pathways.autonomic.toFixed(2)}</span>
            </div>
            <div class="pathway-satellite-node" style="top:150px;left:8px" data-pathway="metabolic">
              <span>⚙️</span> Metabolic <span class="badge ${levelBadgeClass(r.pathways.metabolic)}">${r.pathways.metabolic.toFixed(2)}</span>
            </div>
          </div>
          <div style="font-size:12px;color:var(--ink-muted);margin-top:8px">💡 Click any pathway satellite to inspect physiological factors and clinical attributions</div>
        </div>

        <!-- 6-Axis Biomarker Radar -->
        <div class="card card-panel" style="display:flex;flex-direction:column;justify-content:space-between">
          <div>
            <div style="display:flex;align-items:center;justify-content:space-between">
              <h3 style="font-size:15px;font-weight:700;color:var(--ink-primary)">Biomarker Risk Radar</h3>
              <span class="badge badge-ai">6-Axis Profile</span>
            </div>
            <div style="font-size:12.5px;color:var(--ink-muted);margin-top:2px">TC · LDL · HDL · TG · CRP · D-Dimer (normalized risk surface)</div>
          </div>
          ${radarSVG(b)}
          <div style="font-size:11.5px;color:var(--ink-muted);text-align:center">Surface area indicates multi-pathway cardiovascular risk loading</div>
        </div>
      </div>
    </section>

    <!-- 9. SIX RISK PATHWAYS CARDS -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>📊 Six Cardiovascular Risk Pathways</h2>
          <p>Modular pathway breakdown identifying which biological mechanism requires attention.</p>
        </div>
      </div>

      <div class="pathways-grid">
        ${Object.entries(r.pathways).map(([k, v]) => {
          const info = PATHWAY_INFO[k] || { icon: "•", name: k, desc: "" };
          const col = levelColor(v);
          const lvl = levelName(v);
          return `
          <div class="card pathway-card card-hover" data-pathway="${k}">
            <div class="path-card-head">
              <div class="path-info">
                <div class="path-icon-box">${info.icon}</div>
                <div class="path-name-wrap">
                  <h4>${info.name}</h4>
                  <div class="path-meta-desc">${info.desc}</div>
                </div>
              </div>
              <span class="badge ${levelBadgeClass(v)}">${lvl}</span>
            </div>

            <div>
              <div class="path-card-score-row">
                <div class="path-score-num" style="color:${col}">${v.toFixed(2)}<small> / 1.00</small></div>
                <div style="font-size:12px;color:var(--ink-muted)">Weight: ${info.weight}</div>
              </div>
              <div class="progress-bar-wrap">
                <div class="progress-bar-fill" style="width:${v * 100}%;background:${col}"></div>
              </div>
            </div>

            <div class="path-card-foot">
              <span>View contributing factors</span>
              <span>→</span>
            </div>
          </div>`;
        }).join("")}
      </div>
    </section>

    <!-- 10. AI-ESTIMATED BIOMARKERS SECTION -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>🧪 AI-Estimated Biomarkers</h2>
          <p>Continuous biomarker synthesis derived from wearable photoplethysmography, spectroscopy, and physiological history.</p>
        </div>
        <button class="btn btn-ghost" id="view-full-lipids-btn">View Complete Lipid Profile →</button>
      </div>

      <!-- 11. MANDATORY AI ESTIMATION DISCLAIMER BANNER -->
      <div class="ai-disclaimer-banner">
        <div class="ai-disclaimer-text">
          <span>✨</span>
          <span><b>AI-Estimated Values:</b> These values are computational estimates derived from multimodal sensor signals and questionnaire data. They are not a substitute for clinical laboratory blood tests.</span>
        </div>
        <button class="btn btn-ghost" style="padding:5px 10px;font-size:11.5px;background:#FFF" data-help="baseline">How it works</button>
      </div>

      <!-- Primary 4 Biomarker Cards: TC, HDL, LDL, TG -->
      <div class="biomarkers-grid">
        <!-- Total Cholesterol -->
        <div class="card biomarker-card card-hover" data-help="TC">
          <div class="bio-head">
            <span class="bio-name">Total Cholesterol</span>
            <button class="btn-icon" data-help="TC">ℹ️</button>
          </div>
          <div class="bio-val-box">
            <span class="bio-main-val">${fmt(b.TC)}</span>
            <span class="bio-unit">mg/dL</span>
          </div>
          <div><span class="badge ${cats.TC === "Desirable" ? "badge-low" : "badge-mod"}"><span class="badge-dot"></span>${esc(cats.TC)}</span></div>
          <div class="bio-ref-row">
            <span>Reference: &lt; 200 mg/dL</span>
            <span style="color:var(--brand);font-weight:600">Details →</span>
          </div>
        </div>

        <!-- HDL -->
        <div class="card biomarker-card card-hover" data-help="HDL">
          <div class="bio-head">
            <span class="bio-name">HDL ('Protective')</span>
            <button class="btn-icon" data-help="HDL">ℹ️</button>
          </div>
          <div class="bio-val-box">
            <span class="bio-main-val">${fmt(b.HDL)}</span>
            <span class="bio-unit">mg/dL</span>
          </div>
          <div><span class="badge ${cats.HDL === "Protective" ? "badge-low" : "badge-mod"}"><span class="badge-dot"></span>${esc(cats.HDL)}</span></div>
          <div class="bio-ref-row">
            <span>Reference: ≥ 60 mg/dL</span>
            <span style="color:var(--brand);font-weight:600">Details →</span>
          </div>
        </div>

        <!-- LDL -->
        <div class="card biomarker-card card-hover" data-help="LDL">
          <div class="bio-head">
            <span class="bio-name">LDL ('Atherogenic')</span>
            <button class="btn-icon" data-help="LDL">ℹ️</button>
          </div>
          <div class="bio-val-box">
            <span class="bio-main-val">${fmt(b.LDL)}</span>
            <span class="bio-unit">mg/dL</span>
          </div>
          <div><span class="badge ${cats.LDL === "Optimal" ? "badge-low" : cats.LDL === "Near Optimal" ? "badge-low" : "badge-mod"}"><span class="badge-dot"></span>${esc(cats.LDL)}</span></div>
          <div class="bio-ref-row">
            <span>Reference: &lt; 100 mg/dL</span>
            <span style="color:var(--brand);font-weight:600">Details →</span>
          </div>
        </div>

        <!-- Triglycerides -->
        <div class="card biomarker-card card-hover" data-help="TG">
          <div class="bio-head">
            <span class="bio-name">Triglycerides</span>
            <button class="btn-icon" data-help="TG">ℹ️</button>
          </div>
          <div class="bio-val-box">
            <span class="bio-main-val">${fmt(b.TG)}</span>
            <span class="bio-unit">mg/dL</span>
          </div>
          <div><span class="badge ${cats.TG === "Normal" ? "badge-low" : "badge-mod"}"><span class="badge-dot"></span>${esc(cats.TG)}</span></div>
          <div class="bio-ref-row">
            <span>Reference: &lt; 150 mg/dL</span>
            <span style="color:var(--brand);font-weight:600">Details →</span>
          </div>
        </div>
      </div>

      <!-- Secondary Biomarkers: CRP & D-Dimer -->
      <div class="secondary-bio-row">
        <div class="card card-panel card-hover" style="display:flex;align-items:center;justify-content:space-between" data-help="CRP">
          <div>
            <div style="display:flex;align-items:center;gap:6px">
              <span style="font-size:18px">🔥</span>
              <h4 style="font-size:14px;font-weight:700">Inflammation — C-Reactive Protein (CRP)</h4>
            </div>
            <div style="font-size:28px;font-weight:800;color:var(--ink-primary);margin:6px 0 4px">
              ${fmt(b.CRP, 2)} <span style="font-size:14px;color:var(--ink-muted);font-weight:600">mg/L</span>
            </div>
            <div style="font-size:12px;color:var(--ink-muted)">Reference: &lt; 1.0 mg/L (Low CV Risk)</div>
          </div>
          <span class="badge ${cats.CRP === "Low CV Risk" ? "badge-low" : "badge-mod"}">${esc(cats.CRP)}</span>
        </div>

        <div class="card card-panel card-hover" style="display:flex;align-items:center;justify-content:space-between" data-help="DD">
          <div>
            <div style="display:flex;align-items:center;gap:6px">
              <span style="font-size:18px">🩸</span>
              <h4 style="font-size:14px;font-weight:700">Thrombosis — D-Dimer</h4>
            </div>
            <div style="font-size:28px;font-weight:800;color:var(--ink-primary);margin:6px 0 4px">
              ${fmt(b.DD, 2)} <span style="font-size:14px;color:var(--ink-muted);font-weight:600">mg/L</span>
            </div>
            <div style="font-size:12px;color:var(--ink-muted)">Reference: &lt; 0.50 mg/L (Normal Fibrinolysis)</div>
          </div>
          <span class="badge ${cats.DD === "Normal" ? "badge-low" : "badge-mod"}">${esc(cats.DD)}</span>
        </div>
      </div>
    </section>

    <!-- 12. PERSONAL BASELINE ("YOU VS. YOU") -->
    <section>
      <div class="card baseline-banner-card">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
          <div>
            <h3 style="font-size:17px;font-weight:800;color:var(--ink-primary);display:flex;align-items:center;gap:8px">
              🎯 Your Normal vs. Today
            </h3>
            <p style="font-size:13px;color:var(--ink-secondary);margin-top:2px">
              Your digital twin continuously learns your personal baseline distribution over ${r.days} days and compares changes against <b>YOU</b> — not population averages.
            </p>
          </div>
          <span class="badge ${devBadge}" style="font-size:13px;padding:6px 14px">
            <span class="badge-dot"></span>${devStatus}
          </span>
        </div>

        <div class="baseline-cards-grid">
          ${r.baselines.slice(0, 3).map((row) => {
            const zc = row.z != null && Math.abs(row.z) > 1.8 ? "var(--risk-mod)" : "var(--risk-low)";
            return `
            <div class="baseline-param-card">
              <div class="base-head">
                <span>${esc(row.biomarker)}</span>
                <span class="badge ${Math.abs(row.z || 0) > 1.8 ? "badge-mod" : "badge-low"}">Z: ${row.z != null ? (row.z > 0 ? "+" : "") + row.z.toFixed(2) : "—"}</span>
              </div>
              <div class="base-compare-row">
                <div class="base-val-group">
                  <span class="base-label">Your Baseline</span>
                  <span class="base-num" style="color:var(--ink-muted)">${row.baseline_mean != null ? fmt(row.baseline_mean, row.biomarker === "CRP" || row.biomarker === "DD" ? 2 : 0) : "Learning…"}</span>
                </div>
                <div style="font-size:18px;color:var(--ink-faint)">→</div>
                <div class="base-val-group">
                  <span class="base-label">Today's State</span>
                  <span class="base-num">${fmt(row.current, row.biomarker === "CRP" || row.biomarker === "DD" ? 2 : 0)}</span>
                </div>
              </div>
              <div style="font-size:12px;color:${row.pct_change > 3 ? "var(--risk-mod)" : row.pct_change < -3 ? "var(--risk-low)" : "var(--ink-muted)"};font-weight:600">
                ${row.pct_change != null ? (row.pct_change > 0 ? "↑ +" : "↓ ") + Math.abs(row.pct_change).toFixed(1) + "% shift from normal" : "Establishing baseline"}
              </div>
            </div>`;
          }).join("")}
        </div>
      </div>
    </section>

    <!-- 13. HEALTH TRENDS -->
    <section>
      <div class="card trends-card">
        <div class="trends-controls">
          <div>
            <h3 style="font-size:17px;font-weight:800;color:var(--ink-primary)">📈 Longitudinal Health Trends</h3>
            <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Continuous ${r.days}-day trajectory assimilated from daily wearable feature vectors.</p>
          </div>

          <!-- Metric Selectors -->
          <div class="filter-pills">
            ${METRIC_TABS.map(([k, l]) => `
              <button class="filter-btn ${mk === k ? "active" : ""}" data-metric="${k}">${l}</button>
            `).join("")}
          </div>
        </div>

        <div class="chart-wrapper">
          ${lineChartSVG(series, mk, activeSeriesCol)}
        </div>

        <div class="scenario-switcher-bar">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-weight:600;color:var(--ink-primary)">Simulation Scenario:</span>
            <div class="filter-pills">
              ${["stable", "improving", "declining"].map((s) => `
                <button class="filter-btn ${state.scenario === s ? "active" : ""}" data-scen="${s}">${s.toUpperCase()}</button>
              `).join("")}
            </div>
          </div>
          <span>Active scenario: <b>${esc(state.scenario)} trajectory</b></span>
        </div>
      </div>
    </section>

    <!-- 14. EXPLAINABLE AI (XAI) -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>🤖 Why is my risk score ${(r.cvd_score * 100).toFixed(0)}?</h2>
          <p>Explainable AI factor attribution showing exact mathematical weight and physiological contribution.</p>
        </div>
      </div>

      <div class="xai-grid">
        <!-- Factor Attributions -->
        <div class="card xai-attribution-card">
          <h4 style="font-size:14px;font-weight:700;color:var(--ink-primary);margin-bottom:12px">Primary Risk Contributors</h4>
          ${drivers.map((c) => {
            const pct = Math.min(100, Math.abs(c.contribution) * 12);
            return `
            <div class="attribution-bar-row">
              <div class="attr-name" title="${esc(c.name)}">${esc(c.name)}</div>
              <div class="attr-bar-bg">
                <div class="attr-bar-fill" style="width:${pct}%;background:${c.contribution > 0 ? "var(--risk-high)" : "var(--risk-low)"}"></div>
              </div>
              <div class="attr-val" style="color:${c.contribution > 0 ? "var(--risk-high)" : "var(--risk-low)"}">
                ${c.contribution > 0 ? "+" : "-"}${Math.abs(c.contribution).toFixed(1)}
              </div>
            </div>`;
          }).join("")}
        </div>

        <!-- AI Clinical Interpretation -->
        <div class="card xai-narrative-card">
          <div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
              <h4 style="font-size:14px;font-weight:700;color:var(--ink-primary)">AI Clinical Interpretation</h4>
              <span class="badge badge-ai">✨ 85% Confidence</span>
            </div>
            <div class="xai-quote-box">
              "Your current overall cardiovascular risk remains <b>${esc(r.cvd_category.toLowerCase())}</b>.
              The primary contributors to today's estimate are your <b>${PATHWAY_INFO.lipid.name.toLowerCase()}</b> and <b>${PATHWAY_INFO.hemodynamic.name.toLowerCase()}</b>, while your protective HDL parameters remain favorable."
            </div>
          </div>
          <div style="font-size:11.5px;color:var(--ink-muted);line-height:1.4">
            Confidence represents mathematical model calibration on multimodal signal-to-noise ratios, not medical certainty.
          </div>
        </div>
      </div>
    </section>

    <!-- 16. CONNECTED SENSORS MATRIX -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>⌚ Connected Wearable Sensors</h2>
          <p>9 multimodal wearable streams active — continuous sensor fusion feeding the digital twin.</p>
        </div>
        <span class="badge badge-low"><span class="badge-dot"></span>9 / 9 Active</span>
      </div>

      <div class="sensors-grid">
        ${(r.wearable_status || []).map((s) => {
          const cat = SENSOR_CATALOG[s.sensor] || { icon: "📡", name: s.sensor, stream: "Active stream" };
          return `
          <div class="sensor-card card-hover" data-sensor="${s.sensor}">
            <div class="sensor-left">
              <span class="sensor-icon">${cat.icon}</span>
              <div>
                <div class="sensor-name">${esc(s.sensor)}</div>
                <div class="sensor-stream-tag">${esc(cat.stream)}</div>
              </div>
            </div>
            <span class="pulse-dot" title="Streaming telemetry"></span>
          </div>`;
        }).join("")}
      </div>
    </section>

    <!-- 17. HEALTH ALERTS & INSIGHTS FEED -->
    <section>
      <div class="section-header">
        <div class="section-title-wrap">
          <h2>🔔 Health Insights & Alerts</h2>
          <p>Intelligent notifications triggered when continuous readings meaningfully deviate from your learned baseline.</p>
        </div>
        <span class="badge badge-neutral">${r.alerts.length} Active Alerts</span>
      </div>

      <div class="alerts-container">
        ${r.alerts.length ? r.alerts.slice(-4).reverse().map((a, i) => {
          const lvl = (a.level || "info").toLowerCase();
          const badgeClass = lvl === "critical" ? "badge-crit" : lvl === "warning" ? "badge-mod" : "badge-info";
          return `
          <div class="alert-card ${lvl}">
            <div class="alert-icon-box" style="background:${lvl === "critical" ? "#FEE2E2" : lvl === "warning" ? "#FEF3C7" : "#DBEAFE"}">
              ${lvl === "critical" ? "🚨" : lvl === "warning" ? "⚠️" : "ℹ️"}
            </div>
            <div class="alert-body">
              <div class="alert-title">
                <span>${esc(a.message)}</span>
                <span class="badge ${badgeClass}">${lvl.toUpperCase()}</span>
              </div>
              <div class="alert-desc">
                Observed on Day ${a.day || r.days}. This insight reflects a statistical deviation from your personalized resting distribution.
              </div>
              <div class="alert-actions">
                <button class="btn btn-ghost" style="padding:4px 10px;font-size:11.5px" data-help="baseline">View Baseline Details</button>
              </div>
            </div>
          </div>`;
        }).join("") : `
          <div class="card card-panel" style="text-align:center;padding:32px;color:var(--risk-low)">
            <span style="font-size:24px">✅</span>
            <div style="font-weight:700;margin-top:6px">All systems in balance</div>
            <div style="font-size:13px;color:var(--ink-muted)">No active clinical alerts — all biomarker estimates are within your normal variance.</div>
          </div>
        `}
      </div>
    </section>

    <!-- 15. EXPLORE ALL 72 FACTORS -->
    <section>
      <div class="card factors-explorer-card">
        <div class="factors-toolbar">
          <div>
            <h3 style="font-size:17px;font-weight:800;color:var(--ink-primary)">🔬 Explore All 72 Physiological Factors</h3>
            <p style="font-size:13px;color:var(--ink-muted);margin-top:2px">Normalized (0.00–1.00) multivariables derived from continuous sensing and clinical questionnaires.</p>
          </div>

          <div class="search-input-wrap">
            <span class="search-icon-pos">🔍</span>
            <input type="text" id="factor-search" placeholder="Search factors… (e.g. PWV, NIR, blood pressure, HRV)" value="${esc(state.factorQuery)}" />
          </div>
        </div>

        <!-- Category filter pills -->
        <div class="category-scroll-pills" style="margin-bottom:14px">
          ${["all", "NIR", "PPG", "BP", "HRV", "SpO2", "ECG", "Temp", "GSR", "BioZ", "Activity", "Diet", "Smoking", "History"].map((c) => `
            <button class="cat-pill-btn ${state.factorCategory === c.toLowerCase() ? "active" : ""}" data-cat="${c.toLowerCase()}">${c}</button>
          `).join("")}
        </div>

        <div class="table-responsive-box">
          <table class="modern-table" id="factors-table">
            <thead>
              <tr>
                <th style="width:70px">ID</th>
                <th>Factor Name</th>
                <th>Category</th>
                <th>Current Value</th>
                <th>Risk Direction</th>
              </tr>
            </thead>
            <tbody>
              ${r.factors.map((f) => `
                <tr data-fid="${esc(f.id)}" data-cat="${esc(f.category || "").toLowerCase()}">
                  <td><span class="badge badge-neutral">${esc(f.id)}</span></td>
                  <td style="font-weight:700;color:var(--ink-primary)">${esc(f.name)}</td>
                  <td style="color:var(--ink-secondary)">${esc(f.category)}</td>
                  <td><b>${(f.value ?? 0).toFixed(2)}</b></td>
                  <td>
                    <span class="badge ${f.reading === "protective" ? "badge-low" : f.reading === "risk-increasing" ? "badge-high" : "badge-neutral"}">
                      ${esc(f.reading)}
                    </span>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- RESEARCH PROTOTYPE DISCLAIMER -->
    <div class="research-footer-disclaimer">
      <span style="font-size:22px">⚠️</span>
      <div>
        <b>Research Prototype Notice:</b> CardioCore is an educational and clinical research demonstration.
        All biomarker estimations and cardiovascular risk ratings are computational model outputs derived from wearable sensor features and questionnaire responses.
        CardioCore is <b>not a regulated medical device</b> and must not be used for emergency, diagnostic, or prescription decisions. Always consult a licensed healthcare professional and confirm with certified laboratory tests.
      </div>
    </div>

  </main>`;

  /* ---------------- Wire Events & Microinteractions ---------------- */
  // Topbar and navigation handlers
  $$("[data-tab]").forEach((btn) => {
    btn.onclick = () => {
      const tab = btn.dataset.tab;
      state.activeView = tab;
      if (tab === "overview") drawCommandCenter();
      else if (tab === "twin") renderTwinView();
      else if (tab === "trends") renderTrendsView();
      else if (tab === "factors") renderFactorsView();
      else if (tab === "alerts") renderAlertsView();
      else if (tab === "profile") renderProfileView();
      window.scrollTo(0, 0);
    };
  });

  $("#nav-brand").onclick = () => { state.activeView = "overview"; drawCommandCenter(); };
  $("#sensor-chip-btn").onclick = () => showSensorDetails("PPG");
  $("#profile-avatar-btn").onclick = () => { state.activeView = "profile"; renderProfileView(); };
  $("#top-logout-btn").onclick = () => doLogout();

  // "What does this mean?" Educational buttons
  $$("[data-help]").forEach((el) => {
    el.onclick = (e) => {
      e.stopPropagation();
      showWhatDoesThisMean(el.dataset.help);
    };
  });

  // Pathway card and node clicks
  $$("[data-pathway]").forEach((el) => {
    el.onclick = () => showPathwayDetails(el.dataset.pathway);
  });

  // Sensor card clicks
  $$("[data-sensor]").forEach((el) => {
    el.onclick = () => showSensorDetails(el.dataset.sensor);
  });

  // Full Lipid Profile Modal
  $("#view-full-lipids-btn").onclick = () => {
    const lipidRows = [
      ["Total Cholesterol", fmt(b.TC), "mg/dL", cats.TC, "< 200 mg/dL"],
      ["HDL Cholesterol", fmt(b.HDL), "mg/dL", cats.HDL, "≥ 60 mg/dL"],
      ["LDL Cholesterol", fmt(b.LDL), "mg/dL", cats.LDL, "< 100 mg/dL"],
      ["Triglycerides", fmt(b.TG), "mg/dL", cats.TG, "< 150 mg/dL"],
      ["VLDL Cholesterol", fmt(r.derived.VLDL), "mg/dL", r.derived.VLDL < 30 ? "Normal" : "Borderline High", "< 30 mg/dL"],
      ["Non-HDL Cholesterol", fmt(r.derived.Non_HDL), "mg/dL", r.derived.Non_HDL < 130 ? "Desirable" : "High", "< 130 mg/dL"],
      ["TC / HDL Ratio", fmt(r.derived.TC_HDL_ratio, 2), "", r.derived.TC_HDL_ratio < 4.5 ? "Desirable" : "High", "< 4.5"],
      ["LDL / HDL Ratio", fmt(r.derived.LDL_HDL_ratio, 2), "", r.derived.LDL_HDL_ratio < 2.0 ? "Optimal" : "High", "< 2.0"],
      ["AIP (Atherogenic Index)", fmt(r.derived.AIP, 2), "", r.derived.AIP < 0.11 ? "Low Risk" : "High Risk", "< 0.11"],
    ];

    openModal("🧪 Complete AI-Estimated Lipid & Biomarker Profile", `
      <div style="overflow-x:auto">
        <table class="modern-table">
          <thead>
            <tr><th>Biomarker</th><th>AI Estimate</th><th>Clinical Status</th><th>Reference Target</th></tr>
          </thead>
          <tbody>
            ${lipidRows.map(([name, val, unit, cat, ref]) => `
              <tr>
                <td style="font-weight:700">${name}</td>
                <td><b>${val}</b> <span style="font-size:12px;color:var(--ink-muted)">${unit}</span></td>
                <td><span class="badge ${catColor(cat) === "#059669" ? "badge-low" : "badge-mod"}">${esc(cat)}</span></td>
                <td style="color:var(--ink-muted)">${ref}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
      <div class="badge badge-ai" style="margin-top:16px">
        ✨ AI estimates synthesized from 72 physiological factors
      </div>
    `);
  };

  // Trend Metric Switcher
  $$("[data-metric]").forEach((btn) => {
    btn.onclick = () => {
      state.trendMetric = btn.dataset.metric;
      drawCommandCenter();
    };
  });

  // Scenario Switcher
  $$("[data-scen]").forEach((btn) => {
    btn.onclick = () => {
      if (state.scenario === btn.dataset.scen) return;
      state.scenario = btn.dataset.scen;
      state.result = null;
      renderDashboard();
    };
  });

  // Factor Search & Category Filters
  const searchInput = $("#factor-search");
  const filterTable = () => {
    const q = (searchInput?.value || "").toLowerCase();
    const cat = state.factorCategory;
    $$("#factors-table tbody tr").forEach((tr) => {
      const matchText = tr.textContent.toLowerCase().includes(q);
      const matchCat = cat === "all" || tr.dataset.cat === cat || tr.dataset.cat.includes(cat);
      tr.style.display = (matchText && matchCat) ? "" : "none";
    });
  };

  if (searchInput) searchInput.oninput = filterTable;

  $$(".cat-pill-btn").forEach((btn) => {
    btn.onclick = () => {
      state.factorCategory = btn.dataset.cat;
      $$(".cat-pill-btn").forEach((b) => b.classList.toggle("active", b === btn));
      filterTable();
    };
  });
}

/* ============================================================
   FOCUSED VIEWS (MY TWIN, TRENDS, FACTORS, ALERTS, PROFILE)
   ============================================================ */

/* View: My Twin Deep-Dive */
function renderTwinView() {
  drawCommandCenter();
  const twinSection = $("section:nth-of-type(2)");
  if (twinSection) twinSection.scrollIntoView({ behavior: "smooth" });
}

/* View: Health Trends */
function renderTrendsView() {
  drawCommandCenter();
  const trendsCard = $(".trends-card");
  if (trendsCard) trendsCard.scrollIntoView({ behavior: "smooth" });
}

/* View: 72 Factors */
function renderFactorsView() {
  drawCommandCenter();
  const factorsCard = $(".factors-explorer-card");
  if (factorsCard) factorsCard.scrollIntoView({ behavior: "smooth" });
}

/* View: Alerts */
function renderAlertsView() {
  drawCommandCenter();
  const alertsContainer = $(".alerts-container");
  if (alertsContainer) alertsContainer.scrollIntoView({ behavior: "smooth" });
}

/* View: Profile & Settings */
function renderProfileView() {
  const p = state.profile || {};
  const pat = state.result?.patient || {};
  
  openModal("👤 Cardiovascular Profile & Settings", `
    <div style="display:flex;flex-direction:column;gap:18px">
      <div style="display:flex;align-items:center;gap:14px;background:var(--surface-subtle);padding:16px;border-radius:var(--radius-md)">
        <div class="avatar-circle" style="width:48px;height:48px;font-size:18px">
          ${esc((state.user?.name || "P")[0].toUpperCase())}
        </div>
        <div>
          <h3 style="font-size:16px;font-weight:800;color:var(--ink-primary)">${esc(state.user?.name || "Patient")}</h3>
          <div style="font-size:13px;color:var(--ink-muted)">${esc(state.user?.email || "")}</div>
        </div>
      </div>

      <div>
        <h4 style="font-size:13px;font-weight:700;color:var(--ink-primary);margin-bottom:8px">Personal Parameters</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:13px">
          <div class="card" style="padding:10px 14px">Age: <b>${pat.age ?? 40} yrs</b></div>
          <div class="card" style="padding:10px 14px">Sex: <b>${esc(pat.sex || "Male").toUpperCase()}</b></div>
          <div class="card" style="padding:10px 14px">BMI: <b>${pat.bmi ?? 23.5}</b></div>
          <div class="card" style="padding:10px 14px">Smoking: <b>${esc(pat.smoke_status || "Never")}</b></div>
        </div>
      </div>

      <div>
        <h4 style="font-size:13px;font-weight:700;color:var(--ink-primary);margin-bottom:8px">Digital Twin Diagnostics</h4>
        <div style="font-size:13px;color:var(--ink-secondary);line-height:1.6;background:#F8FAFC;padding:12px;border-radius:var(--radius-xs);border:1px solid var(--line-subtle)">
          <div>• Monitored days in memory: <b>${state.result?.days || 60} days</b></div>
          <div>• Active physiological factors: <b>72 / 72 active</b></div>
          <div>• Baseline convergence status: <b>Converged (95% CI)</b></div>
        </div>
      </div>

      <div style="display:flex;justify-content:space-between;gap:10px;margin-top:10px;padding-top:16px;border-top:1px solid var(--line-subtle)">
        <button class="btn btn-primary" id="edit-profile-btn">Edit Health Questionnaire →</button>
        <button class="btn btn-ghost" id="modal-logout-btn">Sign Out</button>
      </div>
    </div>
  `);

  $("#edit-profile-btn").onclick = async () => {
    closeModal();
    try { state.profile = (await api("/api/me")).profile; } catch (e) {}
    state._step = 0;
    navOrRoute("#/onboarding");
  };
  $("#modal-logout-btn").onclick = () => { closeModal(); doLogout(); };
}

/* ============================================================
   ROUTER & APPLICATION LIFECYCLE
   ============================================================ */
async function route() {
  const h = location.hash || "#/auth";
  if (!state.token) {
    if (h !== "#/auth") return nav("#/auth");
    return renderAuth();
  }
  if (h === "#/auth") {
    try {
      const me = await api("/api/me");
      state.profile = me.profile;
      return navOrRoute(me.has_profile ? "#/dashboard" : "#/onboarding");
    } catch { return renderAuth(); }
  }
  if (h === "#/onboarding") {
    if (!state.profile) {
      try { state.profile = (await api("/api/me")).profile; } catch {}
    }
    return renderWizard();
  }
  if (h === "#/dashboard") return renderDashboard();
  nav("#/auth");
}

window.addEventListener("hashchange", route);
route();
