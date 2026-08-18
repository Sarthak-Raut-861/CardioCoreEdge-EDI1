/* ============================================================
   CardioCore patient portal — SPA logic
   Onboarding: patient-friendly questionnaire (19 user factors)
   Dashboard: 10 sections per design spec
   ============================================================ */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const app = $("#app");

/* storage safe against sandboxed iframes / blocked third-party storage */
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* in-memory only */ } },
  del(k) { try { localStorage.removeItem(k); } catch (e) {} },
};

const state = {
  token: store.get("cc_token") || null,
  user: JSON.parse(store.get("cc_user") || "null"),
  profile: null,
  result: null,
  scenario: "stable",
  trendMetric: "cvd",
  smokeShown: null,
  factorQuery: "",
};

/* ---------------- utilities ---------------- */
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.hidden = false;
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.hidden = true), 3200);
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

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 0) => Number(v ?? 0).toFixed(d);

function nav(hash) { location.hash = hash; }
function navOrRoute(hash) { if (location.hash === hash) route(); else nav(hash); }
function doLogout(silent) {
  if (state.token) api("/api/auth/logout", { method: "POST" }).catch(() => {});
  store.del("cc_token");
  store.del("cc_user");
  state.token = null; state.user = null; state.profile = null; state.result = null;
  if (!silent) toast("Logged out");
  navOrRoute("#/auth");
}

const CAT_COLORS = {
  Desirable: "#12876F", "Borderline High": "#C77700", High: "#D7263D", "Very High": "#A3122A",
  Optimal: "#12876F", "Near Optimal": "#4E9B7F", Borderline: "#C77700",
  Protective: "#12876F", Normal: "#12876F", "Low (Risk)": "#D7263D",
  "Low CV Risk": "#12876F", "Moderate CV Risk": "#C77700", "High CV Risk": "#D7263D", "Acute Infection": "#A3122A",
  "Mild Elevation": "#C77700", Mild: "#C77700", Moderate: "#E06C00", "High Risk": "#D7263D",
  Low: "#12876F",
};
const catColor = (c) => CAT_COLORS[c] || (String(c).startsWith("Very High") ? "#A3122A" : String(c).startsWith("High") ? "#D7263D" : String(c).startsWith("Moderate") ? "#C77700" : "#64748B");
const PATH_META = {
  lipid: ["🫀", "Lipid"], inflammation: ["🔥", "Inflammation"], thrombosis: ["🩸", "Thrombosis"],
  hemodynamic: ["💓", "Hemodynamic"], autonomic: ["🧠", "Autonomic"], metabolic: ["⚙️", "Metabolic"],
};
const level = (v) => (v < 0.33 ? "Low" : v < 0.66 ? "Moderate" : "High");
const levelColor = (v) => (v < 0.33 ? "#12876F" : v < 0.66 ? "#C77700" : "#D7263D");

/* ============================================================
   AUTH VIEW (unchanged behavior)
   ============================================================ */
function renderAuth() {
  document.title = "CardioCore · Sign in";
  const mode = state._authMode || "login";
  app.innerHTML = `
  <div class="auth-wrap fade-in">
    <section class="auth-brand">
      <div>
        <div class="logo"><span class="mark">❤</span> CardioCore</div>
        <svg class="pulse-lines" width="360" height="70" viewBox="0 0 360 70" fill="none">
          <path d="M0 40 H70 l12-24 16 44 14-32 10 12 H190 l12-24 16 44 14-32 10 12 H360"
                stroke="#FF8DA0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity=".95"/>
        </svg>
        <h1>Your heart, <em>continuously</em> understood.</h1>
        <p class="sub">A personalized cardiovascular digital twin built from 72 physiological factors —
        estimating lipids, inflammation and thrombosis markers without a needle.</p>
      </div>
      <div class="brand-points">
        <div class="pt">✦ &nbsp;<b>72-factor analysis</b> · multimodal sensing fused into one picture</div>
        <div class="pt">✦ &nbsp;<b>Biomarker estimation</b> · TC · HDL · LDL · TG · CRP · D-Dimer</div>
        <div class="pt">✦ &nbsp;<b>Explainable AI</b> · see exactly which factors drive each value</div>
        <div class="pt">✦ &nbsp;<b>Personalized baselines</b> · compared against your own normal</div>
      </div>
      <div class="brand-foot">Research prototype — AI estimates, not a medical device.</div>
    </section>
    <section class="auth-panel">
      <div class="auth-card">
        <div class="tabs">
          <button class="${mode === "login" ? "active" : ""}" data-m="login">Sign in</button>
          <button class="${mode === "signup" ? "active" : ""}" data-m="signup">Create account</button>
        </div>
        <h2>${mode === "login" ? "Welcome back" : "Create your account"}</h2>
        <p class="lead">${mode === "login" ? "Sign in to view your digital twin." : "Start your cardiovascular digital twin in 4 short steps."}</p>
        <form id="auth-form" novalidate>
          ${mode === "signup" ? `
          <div class="field" data-f="name">
            <label>Full name</label>
            <input name="name" autocomplete="name" placeholder="Aarav Sharma" />
            <div class="err" hidden></div>
          </div>` : ""}
          <div class="field" data-f="email">
            <label>Email</label>
            <input name="email" type="email" autocomplete="email" placeholder="you@example.com" />
            <div class="err" hidden></div>
          </div>
          <div class="field" data-f="password">
            <label>Password ${mode === "signup" ? '<span class="hint">min 6 characters</span>' : ""}</label>
            <input name="password" type="password" autocomplete="${mode === "signup" ? "new-password" : "current-password"}" placeholder="••••••••" />
            <div class="err" hidden></div>
          </div>
          <button class="btn btn-primary" style="width:100%;margin-top:8px" type="submit">
            ${mode === "login" ? "Sign in" : "Create account →"}
          </button>
        </form>
      </div>
    </section>
  </div>`;

  $$(".tabs button").forEach((b) => b.onclick = () => { state._authMode = b.dataset.m; renderAuth(); });

  $("#auth-form").onsubmit = async (e) => {
    e.preventDefault();
    const payload = Object.fromEntries(new FormData(e.target).entries());
    let ok = true;
    const setErr = (f, msg) => {
      const wrap = $(`.field[data-f="${f}"]`);
      wrap.classList.toggle("invalid", !!msg);
      $(".err", wrap).hidden = !msg;
      $(".err", wrap).textContent = msg || "";
      if (msg) ok = false;
    };
    setErr("email", !payload.email ? "Email is required" : !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(payload.email) ? "Enter a valid email" : "");
    setErr("password", !payload.password ? "Password is required" : payload.password.length < 6 ? "At least 6 characters" : "");
    if (mode === "signup") setErr("name", !payload.name?.trim() ? "Name is required" : "");
    if (!ok) return;
    try {
      const res = await api(mode === "login" ? "/api/auth/login" : "/api/auth/signup", { method: "POST", body: payload });
      state.token = res.token;
      state.user = { name: res.name, email: res.email };
      store.set("cc_token", res.token);
      store.set("cc_user", JSON.stringify(state.user));
      toast(mode === "login" ? `Welcome back, ${res.name.split(" ")[0]}!` : "Account created — let's set up your twin");
      navOrRoute(res.has_profile ? "#/dashboard" : "#/onboarding");
    } catch (err) { toast(err.message, true); }
  };
}

/* ============================================================
   ONBOARDING — patient-friendly questionnaire
   ============================================================ */
const STEPS = ["Basics", "History", "Lifestyle", "Smoking"];
const FREQS = [
  ["never", "Never"], ["rarely", "Rarely"], ["sometimes", "Sometimes"],
  ["often", "Often"], ["very_often", "Very often"],
];
const ETHNICITIES = [
  ["other", "Other / prefer not to say"], ["european", "European"], ["east_asian", "East Asian"],
  ["south_asian", "South Asian"], ["african", "African"], ["hispanic", "Hispanic"], ["middle_eastern", "Middle Eastern"],
];

function inputField(name, label, val, opts = "") {
  return `<div class="field" data-f="${name}"><label>${label}</label>
    <input name="${name}" value="${val ?? ""}" ${opts} /><div class="err" hidden></div></div>`;
}

function yesNo(name, question, sub, checked) {
  return `
  <div class="yesno" data-f="${name}">
    <div><div class="q">${question}</div>${sub ? `<div class="qsub">${sub}</div>` : ""}</div>
    <div class="seg">
      <label><input type="radio" name="${name}" value="no" ${!checked ? "checked" : ""}/><span>No</span></label>
      <label><input type="radio" name="${name}" value="yes" ${checked ? "checked" : ""}/><span>Yes</span></label>
    </div>
  </div>`;
}

function freqQ(name, question, current) {
  return `
  <div class="freqq" data-f="${name}">
    <div class="q">${question}</div>
    <div class="seg wide">
      ${FREQS.map(([v, l]) => `<label><input type="radio" name="${name}" value="${v}" ${current === v ? "checked" : ""}/><span>${l}</span></label>`).join("")}
    </div>
  </div>`;
}

function renderWizard() {
  if (!state.token) return nav("#/auth");
  document.title = "CardioCore · Health profile";
  const step = state._step ?? 0;
  const p = state.profile || {};
  const F = (n, d) => p[n] ?? d;

  const smokeStatus = p.smoke_status || "never";
  const stepBody = [
    /* ---------- 1 · Basics ---------- */
    `
    <h3>The basics</h3>
    <p class="desc">A few measurements get your twin started. BMI and waist-hip ratio are calculated automatically.</p>
    <div class="grid3">
      ${inputField("age", "Age (years)", F("age", 40), 'type="number" min="18" max="100" required')}
      <div class="field" data-f="sex"><label>Sex</label>
        <select name="sex"><option value="male" ${F("sex", "male") === "male" ? "selected" : ""}>Male</option>
        <option value="female" ${F("sex") === "female" ? "selected" : ""}>Female</option></select></div>
      <div class="field" data-f="ethnicity"><label>Ethnic background</label>
        <select name="ethnicity">${ETHNICITIES.map(([v, l]) => `<option value="${v}" ${F("ethnicity", "other") === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      ${inputField("height_cm", "Height (cm)", F("height_cm", 170), 'type="number" step="1" required')}
      ${inputField("weight_kg", "Weight (kg)", F("weight_kg", 70), 'type="number" step="0.5" required')}
      <div></div>
      ${inputField("waist_cm", "Waist circumference (cm)", F("waist_cm", ""), 'type="number" step="1" placeholder="e.g. 92"')}
      ${inputField("hip_cm", "Hip circumference (cm)", F("hip_cm", ""), 'type="number" step="1" placeholder="e.g. 100"')}
    </div>
    <div class="calc-chips">
      <div class="chip-calc" id="chip-bmi" hidden></div>
      <div class="chip-calc" id="chip-whr" hidden></div>
    </div>`,
    /* ---------- 2 · Medical history ---------- */
    `
    <h3>Medical history</h3>
    <p class="desc">Six quick questions — these personalize your cardiovascular history profile.</p>
    ${yesNo("diabetic", "Do you have diabetes or pre-diabetes?", "Blood sugar conditions affect inflammation and clotting estimates", F("diabetic", false))}
    ${yesNo("high_chol", "Have you ever been diagnosed with high cholesterol?", "", F("high_chol", false))}
    ${yesNo("hypertension", "Do you have high blood pressure (hypertension)?", "", F("hypertension", false))}
    ${yesNo("clot", "Have you ever had a blood clot?", "e.g. DVT, pulmonary embolism", F("clot", false))}
    ${yesNo("thyroid", "Do you have a thyroid disorder?", "", F("thyroid", false))}
    ${yesNo("family_history", "Does heart disease run in your family?", "Parent or sibling with early heart disease", F("family_history", false))}`,
    /* ---------- 3 · Lifestyle & diet ---------- */
    `
    <h3>Lifestyle &amp; diet</h3>
    <p class="desc">How often do you typically… (be honest — this directly tunes your twin)</p>
    ${freqQ("sat_fat_freq", "🍔 Eat high-saturated-fat foods? (fried food, red meat, butter, cheese)", F("sat_fat_freq", "sometimes"))}
    ${freqQ("sugar_freq", "🍰 Have sugary foods or refined carbs? (sweets, soft drinks, white bread)", F("sugar_freq", "sometimes"))}
    ${freqQ("veg_freq", "🥦 Eat fruits and vegetables?", F("veg_freq", "sometimes"))}
    ${freqQ("alcohol_freq", "🍷 Drink alcohol?", F("alcohol_freq", "rarely"))}
    ${freqQ("omega3_freq", "🐟 Eat omega-3 rich foods? (fatty fish, nuts, seeds, olive oil)", F("omega3_freq", "sometimes"))}`,
    /* ---------- 4 · Smoking + optional numbers ---------- */
    `
    <h3>Smoking</h3>
    <p class="desc">Your smoking history sets two risk factors automatically.</p>
    <div class="seg wide" id="smoke-status" style="max-width:420px">
      ${[["never", "Never smoked"], ["current", "I smoke now"], ["former", "I used to smoke"]].map(([v, l]) =>
        `<label><input type="radio" name="smoke_status" value="${v}" ${smokeStatus === v ? "checked" : ""}/><span>${l}</span></label>`).join("")}
    </div>
    <div class="grid2" id="smoke-detail" style="margin-top:16px;${smokeStatus === "never" ? "display:none" : ""}">
      ${inputField("cigarettes_per_day", "Cigarettes per day (average)", F("cigarettes_per_day", ""), 'type="number" min="0" max="80" step="1" placeholder="e.g. 10"')}
      ${inputField("years_smoked", "Years you smoked", F("years_smoked", ""), 'type="number" min="0" max="70" step="1" placeholder="e.g. 15"')}
    </div>
    <div class="bmi-note" id="pack-note" style="margin-top:12px" hidden></div>

    <h3 style="margin-top:26px">Know your numbers? <span class="muted" style="font-weight:500">(optional)</span></h3>
    <p class="desc">Recent readings or lab values make your twin's simulation more personal. Skip anything you don't know.</p>
    <div class="grid3">
      ${inputField("baseline_resting_hr", "Resting heart rate (bpm)", F("baseline_resting_hr", ""), 'type="number" placeholder="64"')}
      ${inputField("baseline_hrv_rmssd", "HRV — RMSSD (ms)", F("baseline_hrv_rmssd", ""), 'type="number" placeholder="48"')}
      ${inputField("baseline_systolic", "Systolic BP (top number)", F("baseline_systolic", ""), 'type="number" placeholder="122"')}
      ${inputField("baseline_diastolic", "Diastolic BP (bottom number)", F("baseline_diastolic", ""), 'type="number" placeholder="79"')}
      ${inputField("baseline_total_cholesterol", "Total cholesterol (mg/dL)", F("baseline_total_cholesterol", ""), 'type="number" placeholder="195"')}
      ${inputField("baseline_hdl", "HDL cholesterol (mg/dL)", F("baseline_hdl", ""), 'type="number" placeholder="50"')}
      ${inputField("baseline_triglycerides", "Triglycerides (mg/dL)", F("baseline_triglycerides", ""), 'type="number" placeholder="140"')}
    </div>`,
  ][step];

  app.innerHTML = `
  <div class="topbar">
    <div class="who"><span style="width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center;color:#fff">❤</span>
      <b>CardioCore</b></div>
    <button class="btn btn-link" id="wiz-logout">Sign out</button>
  </div>
  <div class="container">
    <div class="wizard fade-in">
      <div class="wiz-head">
        <h2>${state.profile ? "Edit your health profile" : "Let's build your digital twin"}</h2>
        <p>Step ${step + 1} of 4 — everything stays private to your account.</p>
      </div>
      <div class="wiz-steps">
        ${STEPS.map((s, i) => `<div class="st ${i === step ? "active" : i < step ? "done" : ""}">${i < step ? "✓ " : ""}${s}</div>`).join("")}
      </div>
      <div class="card wiz-card">
        <form id="wiz-form" novalidate>${stepBody}</form>
        <div class="wiz-nav">
          <button class="btn btn-ghost" id="wiz-back" ${step === 0 ? "disabled" : ""}>← Back</button>
          <button class="btn btn-primary" id="wiz-next">${step === 3 ? (state.profile ? "Save & recalculate" : "Create my twin →") : "Continue →"}</button>
        </div>
      </div>
    </div>
  </div>`;

  $("#wiz-logout").onclick = () => doLogout();

  /* live BMI + WHR */
  const updBMI = () => {
    const h = +$('[name="height_cm"]')?.value, w = +$('[name="weight_kg"]')?.value;
    const wa = +$('[name="waist_cm"]')?.value, hi = +$('[name="hip_cm"]')?.value;
    const bmiChip = $("#chip-bmi"), whrChip = $("#chip-whr");
    if (bmiChip) {
      if (h > 100 && w > 30) {
        const bmi = w / (h / 100) ** 2;
        const cls = bmi < 18.5 ? "Underweight" : bmi < 25 ? "Healthy range" : bmi < 30 ? "Overweight" : "Obese range";
        const col = bmi < 18.5 ? "#2563EB" : bmi < 25 ? "#12876F" : bmi < 30 ? "#C77700" : "#D7263D";
        bmiChip.hidden = false;
        bmiChip.innerHTML = `<b style="color:${col}">BMI ${bmi.toFixed(1)}</b> · ${cls}`;
      } else bmiChip.hidden = true;
    }
    if (whrChip) {
      if (wa > 50 && hi > 50) {
        const whr = wa / hi;
        const risk = whr > 1.0 ? "elevated" : whr > 0.9 ? "moderate" : "low";
        const col = whr > 1.0 ? "#D7263D" : whr > 0.9 ? "#C77700" : "#12876F";
        whrChip.hidden = false;
        whrChip.innerHTML = `<b style="color:${col}">Waist-hip ratio ${whr.toFixed(2)}</b> · ${risk} risk`;
      } else whrChip.hidden = true;
    }
  };
  $('[name="height_cm"]')?.addEventListener("input", updBMI);
  $('[name="weight_kg"]')?.addEventListener("input", updBMI);
  $('[name="waist_cm"]')?.addEventListener("input", updBMI);
  $('[name="hip_cm"]')?.addEventListener("input", updBMI);
  if (step === 0) updBMI();

  /* smoking reveal + pack-years */
  const updSmoke = () => {
    const st = ($('[name="smoke_status"]:checked') || {}).value || "never";
    const detail = $("#smoke-detail");
    if (detail) detail.style.display = st === "never" ? "none" : "";
    const note = $("#pack-note");
    if (note && st !== "never") {
      const cigs = +$('[name="cigarettes_per_day"]')?.value || 0;
      const yrs = +$('[name="years_smoked"]')?.value || 0;
      if (cigs > 0 && yrs > 0) {
        const py = cigs * yrs / 20;
        note.hidden = false;
        note.innerHTML = `🚭 &nbsp;Pack-year history: <b>${py.toFixed(1)}</b> ${st === "current" ? "(smoking intensity factor also active)" : "(retained as historical risk)"}`;
      } else note.hidden = true;
    } else if (note) note.hidden = true;
  };
  $$('[name="smoke_status"]').forEach((r) => (r.onchange = updSmoke));
  $('[name="cigarettes_per_day"]')?.addEventListener("input", updSmoke);
  $('[name="years_smoked"]')?.addEventListener("input", updSmoke);
  if (step === 3) updSmoke();

  $("#wiz-back").onclick = () => { collectForm(); state._step = step - 1; renderWizard(); };
  $("#wiz-next").onclick = async () => {
    if (!validateStep(step)) return;
    collectForm();
    if (step < 3) { state._step = step + 1; renderWizard(); window.scrollTo(0, 0); return; }
    const btn = $("#wiz-next");
    btn.disabled = true; btn.innerHTML = '<span class="spin" style="width:16px;height:16px;border-width:2.5px"></span> Building twin…';
    try {
      await api("/api/profile", { method: "PUT", body: state.profile });
      toast("Profile saved — running your twin");
      state._step = 0;
      navOrRoute("#/dashboard");
    } catch (err) {
      toast(err.message, true);
      btn.disabled = false; btn.textContent = "Create my twin →";
    }
  };

  function collectForm() {
    const fd = new FormData($("#wiz-form"));
    const numeric = new Set(["age", "height_cm", "weight_kg", "waist_cm", "hip_cm",
      "cigarettes_per_day", "years_smoked", "baseline_resting_hr", "baseline_hrv_rmssd",
      "baseline_systolic", "baseline_diastolic", "baseline_total_cholesterol",
      "baseline_hdl", "baseline_triglycerides"]);
    const boolQ = new Set(["diabetic", "high_chol", "hypertension", "clot", "thyroid", "family_history"]);
    const out = { ...(state.profile || {}) };
    for (const [k, v] of fd.entries()) {
      if (numeric.has(k)) { if (v !== "") out[k] = +v; }
      else if (boolQ.has(k)) out[k] = v === "yes";
      else out[k] = v;
    }
    for (const k of ["waist_cm", "hip_cm", "baseline_resting_hr", "baseline_hrv_rmssd",
      "baseline_systolic", "baseline_diastolic", "baseline_total_cholesterol",
      "baseline_hdl", "baseline_triglycerides", "cigarettes_per_day", "years_smoked"])
      if (!(k in out) || out[k] === "" || out[k] === undefined) delete out[k];
    state.profile = out;
  }

  function validateStep(s) {
    $$('.field[data-f]').forEach((w) => { w.classList.remove("invalid"); const e = $(".err", w); if (e) e.hidden = true; });
    let ok = true;
    const bad = (f, msg) => {
      const el = $(`[name="${f}"]`); if (!el) return;
      const w = el.closest(".field"); if (w) { w.classList.add("invalid"); const e = $(".err", w); if (e) { e.hidden = false; e.textContent = msg; } }
      ok = false;
    };
    if (s === 0) {
      const age = +$('[name="age"]')?.value, h = +$('[name="height_cm"]')?.value, w = +$('[name="weight_kg"]')?.value;
      if (!(age >= 18 && age <= 100)) bad("age", "Age must be 18–100");
      if (!(h > 100 && h <= 230)) bad("height_cm", "Enter a valid height");
      if (!(w > 30 && w <= 300)) bad("weight_kg", "Enter a valid weight");
    }
    if (s === 3) {
      const st = ($('[name="smoke_status"]:checked') || {}).value;
      if (st && st !== "never") {
        const c = +$('[name="cigarettes_per_day"]')?.value, y = +$('[name="years_smoked"]')?.value;
        if (!(c > 0)) bad("cigarettes_per_day", "Please enter your average cigarettes per day");
        if (!(y > 0)) bad("years_smoked", "Please enter how many years you smoked");
      }
    }
    return ok;
  }
}

/* ============================================================
   DASHBOARD — 10 sections
   ============================================================ */
async function renderDashboard() {
  if (!state.token) return nav("#/auth");
  document.title = "CardioCore · Your twin";
  app.innerHTML = `<div class="loading fade-in"><div class="spin"></div><div>Running your 72-factor digital twin…</div></div>`;
  try {
    if (!state.result) {
      state.result = await api("/api/twin/simulate", { method: "POST", body: { scenario: state.scenario } });
    }
  } catch (err) { toast(err.message, true); nav("#/onboarding"); return; }
  drawDashboard();
}

/* ---------- SVG chart builders ---------- */
function gaugeSVG(score, color) {
  const frac = Math.max(0, Math.min(1, score));
  const arc = "M 18 105 A 87 87 0 0 1 192 105";
  const len = Math.PI * 87;
  return `
  <svg viewBox="0 0 210 118" width="200">
    <path d="${arc}" stroke="#EFF2F6" stroke-width="15" fill="none" stroke-linecap="round"/>
    <path d="${arc}" stroke="${color}" stroke-width="15" fill="none" stroke-linecap="round"
      stroke-dasharray="${(frac * len).toFixed(1)} ${len}" style="transition: stroke-dasharray .8s ease"/>
  </svg>`;
}

function radarSVG(bio) {
  const axes = ["TC", "LDL", "HDL", "TG", "CRP", "DD"];
  const norm = (k) => {
    const v = bio[k];
    const m = { TC: (v - 150) / 170, LDL: (v - 30) / 270, HDL: 1 - (v - 20) / 60,
      TG: (v - 80) / 420, CRP: (v - 0.2) / 9.8, DD: (v - 0.1) / 2.9 }[k];
    return Math.max(0.02, Math.min(1, m));
  };
  const cx = 130, cy = 118, R = 88;
  const pt = (i, r) => { const a = -Math.PI / 2 + (i * 2 * Math.PI) / 6; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; };
  const ring = (r) => axes.map((_, i) => pt(i, r).map((x) => x.toFixed(1)).join(",")).join(" ");
  const poly = axes.map((k, i) => pt(i, R * norm(k)).map((x) => x.toFixed(1)).join(",")).join(" ");
  const labels = axes.map((k, i) => { const [x, y] = pt(i, R + 18); return `<text x="${x}" y="${y + 4}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748B">${k}</text>`; }).join("");
  return `
  <svg viewBox="0 0 260 240" width="100%" style="max-width:300px;display:block;margin:0 auto">
    ${[0.33, 0.66, 1].map((f) => `<polygon points="${ring(R * f)}" fill="none" stroke="#E7EAF0" stroke-width="1"/>`).join("")}
    ${axes.map((_, i) => { const [x, y] = pt(i, R); return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#E7EAF0"/>`; }).join("")}
    <polygon points="${poly}" fill="rgba(215,38,61,.16)" stroke="#D7263D" stroke-width="2.2" stroke-linejoin="round"/>
    ${labels}
  </svg>`;
}

function lineSVG(series, color, unit = "") {
  const w = 680, h = 130, pad = 10;
  const ys = series.map((p) => p[1]);
  const maxY = Math.max(...ys, 0.001), minY = Math.min(...ys, 0);
  const X = (i) => pad + (i / Math.max(1, series.length - 1)) * (w - 2 * pad);
  const Y = (v) => h - pad - ((v - minY) / (maxY - minY || 1)) * (h - 2 * pad);
  const line = series.map((p, i) => `${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
  return `
  <svg viewBox="0 0 ${w} ${h}" width="100%">
    <polyline points="${pad},${h - pad} ${line} ${w - pad},${h - pad}" fill="rgba(215,38,61,.06)" stroke="none"/>
    <polyline points="${line}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${X(series.length - 1)}" cy="${Y(ys[ys.length - 1])}" r="4" fill="${color}"/>
    <text x="${pad}" y="14" font-size="11" fill="#94A3B8">${minY.toFixed(minY < 10 ? 1 : 0)}${unit}</text>
    <text x="${w - pad}" y="14" text-anchor="end" font-size="11" fill="#94A3B8">${maxY.toFixed(maxY < 10 ? 1 : 0)}${unit}</text>
  </svg>`;
}

function scatterSVG(scatter, patient, colors) {
  const pts = scatter || [];
  if (!pts.length) return "";
  const xs = pts.map((p) => p.x).concat(patient ? [patient.pc1] : []);
  const ys = pts.map((p) => p.y).concat(patient ? [patient.pc2] : []);
  const x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.3, y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
  const W = 420, H = 300, P = 18;
  const X = (v) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P);
  const Y = (v) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P);
  const dots = pts.map((p) => `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="4" fill="${colors[p.cluster] || "#94A3B8"}" opacity=".55"/>`).join("");
  const star = patient ? `<circle cx="${X(patient.pc1)}" cy="${Y(patient.pc2)}" r="10" fill="none" stroke="#0F172A" stroke-width="2.5"/><circle cx="${X(patient.pc1)}" cy="${Y(patient.pc2)}" r="4.5" fill="#0F172A"/>` : "";
  return `
  <svg viewBox="0 0 ${W} ${H}" width="100%">
    ${dots}${star}
    <text x="${W / 2}" y="${H - 2}" text-anchor="middle" font-size="11" fill="#94A3B8">PCA component 1 →</text>
  </svg>`;
}

function drawDashboard() {
  const r = state.result;
  const b = r.biomarkers, cats = r.categories, pat = r.patient || {};
  const cvdColor = catColor(r.cvd_category);
  const trendIcon = { Increasing: "▲", Decreasing: "▼", Stable: "▬" }[r.trend] || "▬";

  /* trajectory trend helper for CRP / DD cards */
  const metricTrend = (k) => {
    const t = r.trajectory; if (t.length < 20) return ["Stable", "▬"];
    const third = Math.floor(t.length / 3);
    const first = t.slice(0, third).reduce((s, p) => s + p[k], 0) / third;
    const last = t.slice(-third).reduce((s, p) => s + p[k], 0) / third;
    const pct = first ? (last - first) / first : 0;
    if (pct > 0.05) return ["Increasing", "▲"];
    if (pct < -0.05) return ["Decreasing", "▼"];
    return ["Stable", "▬"];
  };

  /* baseline deviation summary */
  const zs = r.baselines.filter((x) => x.z != null).map((x) => Math.abs(x.z));
  const meanZ = zs.length ? zs.reduce((a, c) => a + c, 0) / zs.length : 0;
  const devLevel = meanZ < 0.5 ? "Low" : meanZ < 1.0 ? "Moderate" : "High";
  const devColor = meanZ < 0.5 ? "#12876F" : meanZ < 1.0 ? "#C77700" : "#D7263D";

  /* narratives */
  const pathwaysSorted = Object.entries(r.pathways).sort((a, c) => c[1] - a[1]);
  const [crpT, crpI] = metricTrend("CRP"), [ddT, ddI] = metricTrend("DD");
  const lipidRows = [
    ["Total Cholesterol", fmt(b.TC), "mg/dL", cats.TC],
    ["HDL Cholesterol", fmt(b.HDL), "mg/dL", cats.HDL],
    ["LDL Cholesterol", fmt(b.LDL), "mg/dL", cats.LDL],
    ["Triglycerides", fmt(b.TG), "mg/dL", cats.TG],
    ["VLDL", fmt(r.derived.VLDL), "mg/dL", r.derived.VLDL < 30 ? "Normal" : "Borderline High"],
    ["Non-HDL", fmt(r.derived.Non_HDL), "mg/dL", r.derived.Non_HDL < 130 ? "Desirable" : r.derived.Non_HDL < 160 ? "Borderline High" : "High"],
    ["TC / HDL ratio", fmt(r.derived.TC_HDL_ratio, 2), "", r.derived.TC_HDL_ratio < 4.5 ? "Desirable" : r.derived.TC_HDL_ratio < 5 ? "Borderline High" : "High"],
    ["LDL / HDL ratio", fmt(r.derived.LDL_HDL_ratio, 2), "", r.derived.LDL_HDL_ratio < 2 ? "Optimal" : r.derived.LDL_HDL_ratio < 3.5 ? "Borderline High" : "High"],
    ["AIP (atherogenic index)", fmt(r.derived.AIP, 2), "", r.derived.AIP < 0.11 ? "Desirable" : r.derived.AIP <= 0.21 ? "Borderline High" : "High"],
  ];
  const clusterColors = { "Low Risk": "#12876F", "Moderate Risk": "#C77700", "High Risk": "#E06C00", "Very High Risk": "#D7263D" };
  const drivers = (r.top_contributions.TC || []).slice(0, 5);

  /* trend chart metric */
  const METRICS = [["cvd", "Risk score"], ["TC", "TC"], ["HDL", "HDL"], ["LDL", "LDL"], ["TG", "TG"], ["CRP", "CRP"], ["DD", "D-Dimer"]];
  const mk = state.trendMetric;
  const series = r.trajectory.map((p) => [p.day, p[mk] ?? p.cvd]);

  app.innerHTML = `
  <div class="topbar">
    <div class="who"><span style="width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center;color:#fff">❤</span>
      <b>CardioCore</b></div>
    <div class="who">
      <button class="btn btn-ghost" id="edit-profile">Edit profile</button>
      <div class="avatar">${esc((state.user?.name || "P")[0].toUpperCase())}</div>
      <button class="btn btn-link" id="logout">Sign out</button>
    </div>
  </div>

  <div class="container fade-in">

    <!-- header -->
    <div class="dash-header card">
      <div>
        <h2>Welcome, ${esc((state.user?.name || "there").split(" ")[0])} 🫀</h2>
        <p class="muted small">${pat.age ?? "—"} yrs · ${esc((pat.sex || "").toUpperCase())} · BMI ${pat.bmi ?? "—"}${pat.waist_hip_ratio ? ` · WHR ${pat.waist_hip_ratio}` : ""} · Updated ${new Date(r.updated_at || Date.now()).toLocaleString()}</p>
      </div>
      <div class="status-chips">
        <span class="chip" style="background:var(--ok-tint);color:var(--ok)"><span class="dot"></span>Wearable connected</span>
        <span class="chip" style="background:var(--ok-tint);color:var(--ok)"><span class="dot"></span>Digital twin active</span>
      </div>
    </div>

    <!-- 1 · overall risk -->
    <div class="dash-hero">
      <div class="card gauge-card">
        <h4>Cardiovascular Risk</h4>
        <div class="gauge-wrap">${gaugeSVG(r.cvd_score, cvdColor)}
          <div class="gauge-center"><div class="num" style="color:${cvdColor}">${(r.cvd_score * 100).toFixed(0)}</div><div class="of">/ 100</div></div>
        </div>
        <span class="chip" style="background:${cvdColor}1a;color:${cvdColor}"><span class="dot"></span>${esc(r.cvd_category)} risk</span>
        <span class="chip" style="background:#EFF2F6;color:var(--muted);margin-left:6px">${trendIcon} ${esc(r.trend || "—")}</span>
      </div>
      <div class="card stat-card">
        <div class="label">🩸 Coronary Thrombosis Risk</div>
        <div class="big" style="color:${levelColor(r.ctr)}">${(r.ctr * 100).toFixed(0)}<small style="font-size:15px;color:var(--faint)"> / 100</small></div>
        <div class="sub">${esc(r.ctr_category)} · LDL 35% + D-Dimer 40% + CRP 25%</div>
        <div class="path-bar" style="margin-top:12px"><i style="width:${r.ctr * 100}%;background:${levelColor(r.ctr)}"></i></div>
        <div class="sub" style="margin-top:auto;padding-top:12px">🔥 CRP ${fmt(b.CRP, 2)} mg/L · ${esc(cats.CRP)}<br>🩸 D-Dimer ${fmt(b.DD, 2)} mg/L · ${esc(cats.DD)}</div>
      </div>
      <div class="card stat-card">
        <div class="label">Your twin at a glance</div>
        <div class="big">${fmt(b.TC)}<small style="font-size:14px;color:var(--faint)"> mg/dL TC</small></div>
        <div class="sub">HDL ${fmt(b.HDL)} · LDL ${fmt(b.LDL)} · TG ${fmt(b.TG)}</div>
        <div class="sub" style="margin-top:8px">🧭 Cluster: <b>${esc(r.cluster.label)}</b> (${(r.cluster.confidence * 100).toFixed(0)}% confidence)</div>
        <div class="sub">📈 Deviation from your baseline: <b style="color:${devColor}">${devLevel}</b></div>
      </div>
    </div>

    <!-- 2 · six pathways + radar -->
    <div class="two-col">
      <div class="card panel">
        <div class="sec-title">Six risk pathways</div>
        <div class="sec-sub">Which disease pathway is most active for you right now</div>
        <div class="path-grid">
          ${Object.entries(r.pathways).map(([k, v]) => `
          <div class="path-card" style="border-color:${levelColor(v)}33">
            <div class="p-emoji">${PATH_META[k]?.[0] || "•"}</div>
            <div class="p-body">
              <div class="p-name">${PATH_META[k]?.[1] || esc(k)}</div>
              <div class="path-bar" style="margin:6px 0 4px"><i style="width:${(v * 100).toFixed(0)}%;background:${levelColor(v)}"></i></div>
              <div class="p-status" style="color:${levelColor(v)}">${level(v)} · ${v.toFixed(2)}</div>
            </div>
          </div>`).join("")}
        </div>
      </div>
      <div class="card panel">
        <div class="sec-title">Biomarker risk radar</div>
        <div class="sec-sub">6-axis profile — TC · LDL · HDL · TG · CRP · D-Dimer (larger area = higher risk)</div>
        ${radarSVG(b)}
      </div>
    </div>

    <!-- 3 · lipid profile -->
    <div class="card panel" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
        <div><div class="sec-title">Estimated lipid profile</div>
        <div class="sec-sub" style="margin-bottom:0">Derived from your 72-factor model</div></div>
        <span class="chip" style="background:var(--warn-tint);color:var(--warn)">⚡ AI-estimated — not a lab measurement</span>
      </div>
      <div style="overflow:auto;margin-top:16px">
      <table class="lipids">
        <thead><tr><th>Biomarker</th><th>Estimated value</th><th>Status</th><th>Normal range</th></tr></thead>
        <tbody>
        ${lipidRows.map(([name, val, unit, cat]) => `
          <tr><td style="font-weight:600">${name}</td>
              <td><b>${val}</b> <span class="muted small">${unit}</span></td>
              <td><b style="color:${catColor(cat)}">●</b> <span style="color:${catColor(cat)};font-weight:600">${esc(cat)}</span></td>
              <td class="muted small">${esc(NORMAL_RANGES[name] || "")}</td></tr>`).join("")}
        </tbody>
      </table></div>
    </div>

    <!-- 4 · inflammation & thrombosis -->
    <div class="two-col">
      <div class="card panel infl-card">
        <div class="sec-title">🔥 Inflammation — CRP</div>
        <div class="infl-val" style="color:${catColor(cats.CRP)}">${fmt(b.CRP, 2)} <small>mg/L</small></div>
        <div class="chip" style="background:${catColor(cats.CRP)}1a;color:${catColor(cats.CRP)}"><span class="dot"></span>${esc(cats.CRP)}</div>
        <div class="sub" style="margin-top:8px">Trend: <b>${crpI} ${crpT}</b> · low risk &lt; 1.0 mg/L</div>
      </div>
      <div class="card panel infl-card">
        <div class="sec-title">🩸 Thrombosis — D-Dimer</div>
        <div class="infl-val" style="color:${catColor(cats.DD)}">${fmt(b.DD, 2)} <small>mg/L</small></div>
        <div class="chip" style="background:${catColor(cats.DD)}1a;color:${catColor(cats.DD)}"><span class="dot"></span>${esc(cats.DD)}</div>
        <div class="sub" style="margin-top:8px">Trend: <b>${ddI} ${ddT}</b> · normal &lt; 0.5 mg/L</div>
      </div>
    </div>

    <!-- 5 · personalized baseline -->
    <div class="card panel" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
        <div><div class="sec-title">Your current state vs your personal baseline</div>
        <div class="sec-sub" style="margin-bottom:0">Your twin learned your own normal over ${r.days} days — this compares you to <b>you</b>, not population averages</div></div>
        <div>Baseline deviation: <b style="color:${devColor}">${devLevel}</b></div>
      </div>
      <div style="overflow:auto;margin-top:14px">
      <table class="lipids">
        <thead><tr><th>Parameter</th><th>Current</th><th>Your baseline</th><th>Change</th><th>Z-score</th></tr></thead>
        <tbody>
        ${r.baselines.map((row) => {
          const zc = row.z != null && Math.abs(row.z) > 2 ? catColor("High") : "#64748B";
          return `<tr><td style="font-weight:600">${row.biomarker}</td>
            <td><b>${fmt(row.current, row.biomarker === "CRP" || row.biomarker === "DD" ? 2 : 0)}</b></td>
            <td class="muted">${row.baseline_mean != null ? fmt(row.baseline_mean, 2) : "learning…"}</td>
            <td style="color:${row.pct_change > 3 ? "#D7263D" : row.pct_change < -3 ? "#12876F" : "#64748B"}">${row.pct_change != null ? (row.pct_change > 0 ? "↑ " : row.pct_change < 0 ? "↓ " : "") + Math.abs(row.pct_change).toFixed(1) + "%" : "—"}</td>
            <td style="color:${zc};font-weight:${Math.abs(row.z || 0) > 2 ? 700 : 400}">${row.z != null ? (row.z > 0 ? "+" : "") + row.z.toFixed(2) + (Math.abs(row.z) > 2 ? " ⚠️" : "") : "—"}</td></tr>`;
        }).join("")}
        </tbody>
      </table></div>
    </div>

    <!-- 6 · health trends -->
    <div class="card panel" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <div><div class="sec-title">Health trends</div>
        <div class="sec-sub" style="margin-bottom:0">${r.days}-day continuous estimates · scenario: <b>${esc(r.scenario)}</b></div></div>
        <div class="whatif">
          ${METRICS.map(([k, l]) => `<button class="btn ${mk === k ? "btn-primary" : "btn-ghost"}" data-metric="${k}">${l}</button>`).join("")}
        </div>
      </div>
      <div style="margin-top:12px">${lineSVG(series, mk === "cvd" ? cvdColor : "#2563EB")}</div>
      <div class="whatif" style="margin-top:10px">
        <span class="small muted">Simulation scenario:</span>
        ${["stable", "improving", "declining"].map((s) => `<button class="btn ${state.scenario === s ? "btn-primary" : "btn-ghost"}" data-scen="${s}">${s}</button>`).join("")}
      </div>
    </div>

    <!-- 7 · AI explainability -->
    <div class="card panel" style="margin-bottom:18px">
      <div class="sec-title">Why is my risk ${esc(r.cvd_category.toLowerCase())}?</div>
      <div class="sec-sub">Top contributing factors to your cholesterol estimate — from explainable AI</div>
      <div class="grid2">
        <div>
        ${drivers.map((c, i) => `
          <div class="driver"><span class="rank">${i + 1}</span>
            <span class="dname">${esc(c.name)}</span>
            <span class="dcontrib ${c.contribution > 0 ? "up" : "down"}">${c.contribution > 0 ? "▲" : "▼"} ${Math.abs(c.contribution).toFixed(1)}</span>
          </div>`).join("")}
        </div>
        <div class="ai-narrative">
          <b>🤖 AI explanation</b>
          <p>"Your estimated cardiovascular risk is <b>${esc(r.cvd_category.toLowerCase())}</b>, driven mainly by your
          <b>${PATH_META[pathwaysSorted[0][0]]?.[1].toLowerCase() || "lipid"}</b> and
          <b>${PATH_META[pathwaysSorted[1][0]]?.[1].toLowerCase() || "metabolic"}</b> pathways.
          Compared with your personal baseline, ${zs.filter((z) => z > 2).length || "no"} biomarker${zs.filter((z) => z > 2).length === 1 ? " shows" : "s show"} meaningful deviation."</p>
        </div>
      </div>
    </div>

    <!-- 8 · risk cluster -->
    <div class="card panel" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <div><div class="sec-title">Your cardiovascular profile</div>
        <div class="sec-sub" style="margin-bottom:0">K-Means clustering of the six biomarkers across the patient population</div></div>
        <div style="text-align:right">
          <div style="font-size:19px;font-weight:800;color:${clusterColors[r.cluster.label] || "#64748B"}">${esc(r.cluster.label)}</div>
          <div class="small muted">Confidence: <b>${(r.cluster.confidence * 100).toFixed(0)}%</b></div>
        </div>
      </div>
      <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-top:10px">
        <div style="flex:1;min-width:300px">${scatterSVG(r.cluster.scatter, r.cluster, clusterColors)}</div>
        <div class="cluster-legend">
          ${Object.entries(clusterColors).map(([l, c]) => `
          <div class="leg ${l === r.cluster.label ? "me" : ""}"><span class="dot" style="background:${c}"></span>${l}${l === r.cluster.label ? " · YOU" : ""}</div>`).join("")}
        </div>
      </div>
    </div>

    <!-- 9 · alerts -->
    <div class="card panel" style="margin-bottom:18px">
      <div class="sec-title">🔔 Health alerts <span class="fpill" style="margin-left:6px">${r.n_alerts}</span></div>
      <div class="sec-sub">Graded INFO / WARNING / CRITICAL — deduplicated daily, critical repeat until resolved</div>
      ${r.alerts.length ? r.alerts.slice(-5).reverse().map((a) => `
        <div class="alert-item ${a.level}"><span class="lvl">${a.level.toUpperCase()}</span><span>${esc(a.message)}</span></div>`).join("")
        : `<div class="alert-item info"><span class="lvl">OK</span><span>No alerts — all biomarkers within safe ranges.</span></div>`}
    </div>

    <!-- 10 · wearable status -->
    <div class="card panel" style="margin-bottom:18px">
      <div class="sec-title">Connected sensors</div>
      <div class="sec-sub">Simulated multimodal stream feeding your 72-factor engine${r.factors.length ? ` · ${r.factors.length} factors live` : ""}</div>
      <div class="sensor-grid">
        ${(r.wearable_status || []).map((s) => `
        <div class="sensor"><span class="s-ico">${SENSOR_ICONS[s.sensor] || "📡"}</span>
          <span class="s-name">${esc(s.sensor)}</span>
          <span class="s-dot" style="background:#12876F"></span></div>`).join("")}
      </div>
    </div>

    <!-- factors table -->
    <div class="card panel factors-panel">
      <div class="sec-title">All 72 factors — current values</div>
      <div class="sec-sub">Normalized 0–1 · sensor-derived + questionnaire factors combined</div>
      <input class="search" id="factor-search" placeholder="Search factors… e.g. NIR, blood pressure, smoking" value="${esc(state.factorQuery)}" />
      <div style="overflow:auto;max-height:400px">
      <table class="factors">
        <thead><tr><th>ID</th><th>Factor</th><th>Category</th><th>Value</th><th>Reading</th></tr></thead>
        <tbody>
        ${r.factors.map((f) => `
          <tr><td><span class="fpill">${esc(f.id)}</span></td>
              <td style="font-weight:600">${esc(f.name)}</td>
              <td class="muted">${esc(f.category)}</td>
              <td><b>${(f.value ?? 0).toFixed(2)}</b></td>
              <td><span class="fpill ${f.reading === "protective" ? "prot" : f.reading === "risk-increasing" ? "risk" : ""}">${esc(f.reading)}</span></td></tr>`).join("")}
        </tbody>
      </table></div>
    </div>

    <div class="disclaimer">
      <b>Research prototype.</b> All biomarker values are AI estimates derived from physiological and
      questionnaire data — not measurements. This system is not a medical device and must not be used for
      diagnosis. Always confirm with laboratory blood tests and consult a qualified healthcare professional.
    </div>
  </div>`;

  /* wiring */
  $("#logout").onclick = () => doLogout();
  $("#edit-profile").onclick = async () => {
    try { state.profile = (await api("/api/me")).profile; } catch (e) {}
    state._step = 0;
    navOrRoute("#/onboarding");
  };
  $$("[data-scen]").forEach((btn) => btn.onclick = () => {
    if (state.scenario === btn.dataset.scen) return;
    state.scenario = btn.dataset.scen;
    state.result = null;
    renderDashboard();
  });
  $$("[data-metric]").forEach((btn) => btn.onclick = () => {
    state.trendMetric = btn.dataset.metric;
    drawDashboard();
  });
  const search = $("#factor-search");
  search.oninput = () => {
    state.factorQuery = search.value;
    const q = search.value.toLowerCase();
    $$("table.factors tbody tr").forEach((tr) => {
      tr.style.display = tr.textContent.toLowerCase().includes(q) ? "" : "none";
    });
  };
}

const NORMAL_RANGES = {
  "Total Cholesterol": "< 200 mg/dL", "HDL Cholesterol": "≥ 60 mg/dL", "LDL Cholesterol": "< 100 mg/dL",
  "Triglycerides": "< 150 mg/dL", "VLDL": "2–30 mg/dL", "Non-HDL": "< 130 mg/dL",
  "TC / HDL ratio": "< 4.5", "LDL / HDL ratio": "< 2.0", "AIP (atherogenic index)": "< 0.11",
};
const SENSOR_ICONS = {
  "ECG": "心电图", "PPG": "🩸", "SpO2": "🫁", "Blood Pressure": " cuff ", "NIR": "🔆",
  "Temperature": "🌡️", "GSR": "⚡", "Bioimpedance": "📏", "IMU": "🏃",
};
SENSOR_ICONS["ECG"] = "📈";
SENSOR_ICONS["Blood Pressure"] = "🧷";
SENSOR_ICONS["NIR"] = "🔆";

/* ============================================================
   ROUTER
   ============================================================ */
async function route() {
  const h = location.hash || "#/auth";
  if (!state.token) { if (h !== "#/auth") return nav("#/auth"); return renderAuth(); }
  if (h === "#/auth") {
    try { const me = await api("/api/me"); state.profile = me.profile; return navOrRoute(me.has_profile ? "#/dashboard" : "#/onboarding"); }
    catch { return renderAuth(); }
  }
  if (h === "#/onboarding") {
    if (!state.profile) { try { state.profile = (await api("/api/me")).profile; } catch {} }
    return renderWizard();
  }
  if (h === "#/dashboard") return renderDashboard();
  nav("#/auth");
}

window.addEventListener("hashchange", route);
route();
