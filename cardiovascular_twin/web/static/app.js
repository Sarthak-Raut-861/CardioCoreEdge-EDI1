/* ============================================================
   CardioCore patient portal — SPA logic
   ============================================================ */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const app = $("#app");

const state = {
  token: localStorage.getItem("cc_token") || null,
  user: JSON.parse(localStorage.getItem("cc_user") || "null"),
  profile: null,
  result: null,
  scenario: "stable",
  driverTab: "TC",
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
    if (res.status === 401) doLogout(true);
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function nav(hash) { location.hash = hash; }
function doLogout(silent) {
  if (state.token) api("/api/auth/logout", { method: "POST" }).catch(() => {});
  localStorage.removeItem("cc_token");
  localStorage.removeItem("cc_user");
  state.token = null; state.user = null; state.profile = null; state.result = null;
  if (!silent) toast("Logged out");
  nav("#/auth");
}

const CAT_COLORS = {
  Desirable: "#12876F", "Borderline High": "#C77700", High: "#D7263D", "Very High": "#A3122A",
  Optimal: "#12876F", "Near Optimal": "#4E9B7F", Borderline: "#C77700",
  Protective: "#12876F", Normal: "#12876F", "Low (Risk)": "#D7263D",
  "Low CV Risk": "#12876F", "Moderate CV Risk": "#C77700", "High CV Risk": "#D7263D", "Acute Infection": "#A3122A",
  "Mild Elevation": "#C77700", Mild: "#C77700", Moderate: "#E06C00", "High Risk": "#D7263D",
  Low: "#12876F", "Very High ": "#A3122A",
};
const catColor = (c) => CAT_COLORS[c] || (c.startsWith("Very High") ? "#A3122A" : c.startsWith("High") ? "#D7263D" : c.startsWith("Moderate") ? "#C77700" : "#64748B");
const fmt = (v, d = 0) => Number(v ?? 0).toFixed(d);

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
    const fd = new FormData(e.target);
    const payload = Object.fromEntries(fd.entries());
    // validation
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
      localStorage.setItem("cc_token", res.token);
      localStorage.setItem("cc_user", JSON.stringify(state.user));
      toast(mode === "login" ? `Welcome back, ${res.name.split(" ")[0]}!` : "Account created — let's set up your twin");
      nav(res.has_profile ? "#/dashboard" : "#/onboarding");
    } catch (err) { toast(err.message, true); }
  };
}

/* ============================================================
   ONBOARDING WIZARD
   ============================================================ */
const STEPS = ["Basics", "History", "Lifestyle", "Optional"];

const SLIDERS = {
  chol_history: { label: "Known cholesterol history", left: "None", right: "Strong", def: 0.2 },
  htn_history: { label: "Hypertension history", left: "None", right: "Severe", def: 0.1 },
  clot_history: { label: "Previous clot history", left: "None", right: "Recurrent", def: 0.0 },
  thyroid: { label: "Thyroid disorder", left: "None", right: "Active", def: 0.05 },
  sat_fat: { label: "Saturated fat intake", left: "Very low", right: "Very high", def: 0.3 },
  sugar: { label: "Sugar / refined carbs", left: "Very low", right: "Very high", def: 0.3 },
  vegetables: { label: "Vegetables & fruits", left: "Rarely", right: "Daily & varied", def: 0.6 },
  alcohol: { label: "Alcohol consumption", left: "Never", right: "Heavy", def: 0.2 },
  omega3: { label: "Omega-3 / healthy fats", left: "Rarely", right: "Daily", def: 0.5 },
  smoking_intensity: { label: "Smoking intensity", left: "Light", right: "Heavy", def: 0.6 },
  ethnicity_risk: { label: "Population risk background", left: "Low", right: "High", def: 0.3 },
};

function sliderField(name, def) {
  const s = SLIDERS[name];
  return `
  <div class="field slider-row" data-f="${name}">
    <div class="s-top"><label>${s.label}</label><span class="s-val" data-val>${Math.round(def * 100)}%</span></div>
    <input type="range" name="${name}" min="0" max="1" step="0.05" value="${def}" />
    <div class="s-ends"><span>${s.left}</span><span>${s.right}</span></div>
  </div>`;
}

function toggleField(name, label, sub, checked) {
  return `
  <label class="toggle" data-f="${name}">
    <span><span class="tlabel">${label}</span><span class="tsub">${sub}</span></span>
    <input type="checkbox" name="${name}" ${checked ? "checked" : ""} /><span class="switch"></span>
  </label>`;
}

function inputField(name, label, val, opts = "") {
  return `
  <div class="field" data-f="${name}">
    <label>${label}</label>
    <input name="${name}" value="${val ?? ""}" ${opts} />
    <div class="err" hidden></div>
  </div>`;
}

function renderWizard() {
  if (!state.token) return nav("#/auth");
  document.title = "CardioCore · Health profile";
  const step = state._step ?? 0;
  const p = state.profile || {};
  const F = (n, d) => p[n] ?? d;

  const stepBody = [
    // ---- 1 Basics ----
    `
    <h3>The basics</h3>
    <p class="desc">Used for factors F48–F54 (age, sex, BMI, waist, hip ratio, background).</p>
    <div class="grid3">
      ${inputField("age", "Age (years)", F("age", 40), 'type="number" min="18" max="100" required')}
      <div class="field" data-f="sex"><label>Sex</label>
        <select name="sex"><option value="male" ${F("sex") === "male" ? "selected" : ""}>Male</option>
        <option value="female" ${F("sex") === "female" ? "selected" : ""}>Female</option></select></div>
      ${inputField("height_cm", "Height (cm)", F("height_cm", 170), 'type="number" step="1" required')}
      ${inputField("weight_kg", "Weight (kg)", F("weight_kg", 70), 'type="number" step="0.5" required')}
      ${inputField("waist_cm", "Waist circumference (cm) <span class='hint'>optional</span>", F("waist_cm", ""), 'type="number" step="1"')}
      ${inputField("hip_cm", "Hip circumference (cm) <span class='hint'>optional</span>", F("hip_cm", ""), 'type="number" step="1"')}
    </div>
    ${sliderField("ethnicity_risk", F("ethnicity_risk", 0.3)).replace('class="field', 'style="margin-top:18px" class="field')}
    <div class="bmi-note" id="bmi-note" hidden></div>
    `,
    // ---- 2 History ----
    `
    <h3>Medical history</h3>
    <p class="desc">Used for factors F55–F60 (diabetes, cholesterol, BP, clots, thyroid, family history).</p>
    <div class="grid2" style="margin-bottom:18px">
      ${toggleField("diabetic", "Diabetes / pre-diabetes", "F55 — raises CRP & D-Dimer estimation", F("diabetic", false))}
      ${toggleField("family_history", "Family history of heart disease", "F60 — premature CVD in close relatives", F("family_history", false))}
    </div>
    <div class="grid2">
      ${sliderField("chol_history", F("chol_history", 0.2))}
      ${sliderField("htn_history", F("htn_history", 0.1))}
      ${sliderField("clot_history", F("clot_history", 0.0))}
      ${sliderField("thyroid", F("thyroid", 0.05))}
    </div>
    `,
    // ---- 3 Lifestyle ----
    `
    <h3>Lifestyle &amp; diet</h3>
    <p class="desc">Used for factors F61–F67 (diet patterns, smoking).</p>
    ${toggleField("smoker", "Do you smoke?", "F66–F67 — intensity & pack-years asked below", F("smoker", false))}
    <div class="grid2" id="smoke-extra" style="margin:16px 0 18px;${F("smoker", false) ? "" : "display:none"}">
      ${sliderField("smoking_intensity", F("smoking_intensity", 0.6))}
      ${inputField("pack_years", "Pack-years smoked", F("pack_years", 12), 'type="number" min="0" max="80" step="1"')}
    </div>
    <div class="grid2">
      ${sliderField("sat_fat", F("sat_fat", 0.3))}
      ${sliderField("sugar", F("sugar", 0.3))}
      ${sliderField("vegetables", F("vegetables", 0.6))}
      ${sliderField("alcohol", F("alcohol", 0.2))}
      ${sliderField("omega3", F("omega3", 0.5))}
    </div>
    `,
    // ---- 4 Optional baselines ----
    `
    <h3>Know your numbers? <span class="muted" style="font-weight:500">(optional)</span></h3>
    <p class="desc">Recent readings or lab values personalize the simulation baselines. Skip anything you don't know — sensible defaults are used.</p>
    <div class="grid3">
      ${inputField("baseline_resting_hr", "Resting heart rate (bpm)", F("baseline_resting_hr", ""), 'type="number" placeholder="64"')}
      ${inputField("baseline_hrv_rmssd", "HRV RMSSD (ms)", F("baseline_hrv_rmssd", ""), 'type="number" placeholder="48"')}
      ${inputField("baseline_systolic", "Systolic BP (mmHg)", F("baseline_systolic", ""), 'type="number" placeholder="122"')}
      ${inputField("baseline_diastolic", "Diastolic BP (mmHg)", F("baseline_diastolic", ""), 'type="number" placeholder="79"')}
      ${inputField("baseline_total_cholesterol", "Total cholesterol (mg/dL)", F("baseline_total_cholesterol", ""), 'type="number" placeholder="195"')}
      ${inputField("baseline_hdl", "HDL cholesterol (mg/dL)", F("baseline_hdl", ""), 'type="number" placeholder="50"')}
      ${inputField("baseline_triglycerides", "Triglycerides (mg/dL)", F("baseline_triglycerides", ""), 'type="number" placeholder="140"')}
    </div>
    <div class="bmi-note">Your answers personalize <b>&nbsp;30+ of the 72 factors&nbsp;</b> — the rest are measured continuously from wearable sensors.</div>
    `,
  ][step];

  app.innerHTML = `
  <div class="topbar">
    <div class="who"><span class="mark" style="width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center;color:#fff">❤</span>
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

  // live slider bubbles
  $$('.slider-row input[type="range"]').forEach((r) => {
    const upd = () => {
      $(('[data-val]'), r.closest(".slider-row")).textContent = `${Math.round(r.value * 100)}%`;
      r.style.setProperty("--fill", `${r.value * 100}%`);
    };
    r.oninput = upd; upd();
  });

  // live BMI
  const bmiNote = $("#bmi-note");
  const updBMI = () => {
    const h = +$('[name="height_cm"]')?.value, w = +$('[name="weight_kg"]')?.value;
    if (h > 100 && w > 30) {
      const bmi = w / (h / 100) ** 2;
      const cls = bmi < 18.5 ? "Underweight" : bmi < 25 ? "Healthy range" : bmi < 30 ? "Overweight" : "Obese range";
      bmiNote.hidden = false;
      bmiNote.innerHTML = `⚖️ &nbsp;BMI ${bmi.toFixed(1)} — ${cls}`;
    } else bmiNote.hidden = true;
  };
  $('[name="height_cm"]')?.addEventListener("input", updBMI);
  $('[name="weight_kg"]')?.addEventListener("input", updBMI);
  if (step === 0) updBMI();

  // smoker toggle reveals intensity
  const smTog = $('[name="smoker"]');
  if (smTog) smTog.onchange = () => ($("#smoke-extra").style.display = smTog.checked ? "" : "none");

  $("#wiz-back").onclick = () => { collectForm(); state._step = step - 1; renderWizard(); };
  $("#wiz-next").onclick = async () => {
    if (!validateStep(step)) return;
    collectForm();
    if (step < 3) { state._step = step + 1; renderWizard(); window.scrollTo(0, 0); return; }
    // submit
    const btn = $("#wiz-next");
    btn.disabled = true; btn.innerHTML = '<span class="spin" style="width:16px;height:16px;border-width:2.5px"></span> Building twin…';
    try {
      await api("/api/profile", { method: "PUT", body: state.profile });
      state.user = { ...state.user, has_profile: true };
      toast("Profile saved — running your twin");
      state._step = 0;
      nav("#/dashboard");
    } catch (err) {
      toast(err.message, true);
      btn.disabled = false; btn.textContent = "Create my twin →";
    }
  };

  function collectForm() {
    const fd = new FormData($("#wiz-form"));
    const numeric = new Set(["age", "height_cm", "weight_kg", "waist_cm", "hip_cm", "pack_years",
      "baseline_resting_hr", "baseline_hrv_rmssd", "baseline_systolic", "baseline_diastolic",
      "baseline_total_cholesterol", "baseline_hdl", "baseline_triglycerides",
      ...Object.keys(SLIDERS)]);
    const out = { ...(state.profile || {}) };
    for (const [k, v] of fd.entries()) out[k] = numeric.has(k) ? (v === "" ? null : +v) : v;
    // defaults for blank optionals
    for (const k of ["waist_cm", "hip_cm", "baseline_resting_hr", "baseline_hrv_rmssd", "baseline_systolic",
      "baseline_diastolic", "baseline_total_cholesterol", "baseline_hdl", "baseline_triglycerides"])
      if (out[k] === null || out[k] === undefined || Number.isNaN(out[k])) delete out[k];
    state.profile = out;
  }

  function validateStep(s) {
    let ok = true;
    const need = [
      [["age"], (v) => v >= 18 && v <= 100, "Age must be 18–100"],
      [["height_cm"], (v) => v > 100 && v <= 230, "Enter a valid height"],
      [["weight_kg"], (v) => v > 30 && v <= 300, "Enter a valid weight"],
      [["pack_years"], (v) => v === "" || (v >= 0 && v <= 80), "0–80"],
    ];
    const checks = need.filter((_, i) => [0, 0, 0, 2][i] === s || (s === 2));
    $$('.field[data-f]').forEach((w) => { w.classList.remove("invalid"); $(".err", w) && ($(".err", w).hidden = true); });
    for (const [fields, test, msg] of checks) {
      const f = fields[0];
      const el = $(`[name="${f}"]`);
      if (!el) continue;
      if (!test(el.value)) {
        const w = el.closest(".field");
        if (w) { w.classList.add("invalid"); const e = $(".err", w); if (e) { e.hidden = false; e.textContent = msg; } }
        ok = false;
      }
    }
    return ok;
  }
}

/* ============================================================
   DASHBOARD VIEW
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

function gaugeSVG(score, color) {
  const frac = Math.max(0, Math.min(1, score));
  const arc = "M 18 105 A 87 87 0 0 1 192 105";
  const len = Math.PI * 87;
  return `
  <svg viewBox="0 0 210 118" width="210">
    <path d="${arc}" stroke="#EFF2F6" stroke-width="15" fill="none" stroke-linecap="round"/>
    <path d="${arc}" stroke="${color}" stroke-width="15" fill="none" stroke-linecap="round"
      stroke-dasharray="${(frac * len).toFixed(1)} ${len}" style="transition: stroke-dasharray .8s ease"/>
  </svg>`;
}

function radarSVG(bio) {
  const axes = ["TC", "LDL", "HDL", "TG", "CRP", "DD"];
  const norm = (k) => {
    const v = bio[k];
    const m = { TC: [(v - 150) / 170], LDL: [(v - 30) / 270], HDL: [1 - (v - 20) / 60],
      TG: [(v - 80) / 420], CRP: [(v - 0.2) / 9.8], DD: [(v - 0.1) / 2.9] }[k][0];
    return Math.max(0.02, Math.min(1, m));
  };
  const cx = 130, cy = 118, R = 88;
  const pt = (i, r) => { const a = -Math.PI / 2 + (i * 2 * Math.PI) / 6; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; };
  const ring = (r) => axes.map((_, i) => pt(i, r).map((x) => x.toFixed(1)).join(",")).join(" ");
  const poly = axes.map((k, i) => pt(i, R * norm(k)).map((x) => x.toFixed(1)).join(",")).join(" ");
  const labels = axes.map((k, i) => { const [x, y] = pt(i, R + 18); return `<text x="${x}" y="${y + 4}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748B">${k}</text>`; }).join("");
  return `
  <svg viewBox="0 0 260 240" width="100%" style="max-width:290px;display:block;margin:0 auto">
    ${[0.33, 0.66, 1].map((f) => `<polygon points="${ring(R * f)}" fill="none" stroke="#E7EAF0" stroke-width="1"/>`).join("")}
    ${axes.map((_, i) => { const [x, y] = pt(i, R); return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#E7EAF0"/>`; }).join("")}
    <polygon points="${poly}" fill="rgba(215,38,61,.16)" stroke="#D7263D" stroke-width="2.2" stroke-linejoin="round"/>
    ${labels}
  </svg>`;
}

function sparklineSVG(points, color) {
  const w = 640, h = 120, pad = 8;
  const xs = points.map((p) => p.day), ys = points.map((p) => p.cvd);
  const maxX = Math.max(...xs, 1), maxY = Math.max(...ys, 0.001);
  const X = (d) => pad + (d / maxX) * (w - 2 * pad);
  const Y = (v) => h - pad - (v / (maxY * 1.15)) * (h - 2 * pad);
  const line = points.map((p) => `${X(p.day).toFixed(1)},${Y(p.cvd).toFixed(1)}`).join(" ");
  return `
  <svg viewBox="0 0 ${w} ${h}" width="100%">
    <polyline points="${pad},${h - pad} ${line} ${w - pad},${h - pad}" fill="rgba(215,38,61,.07)" stroke="none"/>
    <polyline points="${line}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
    ${points.length ? `<circle cx="${X(xs[xs.length - 1])}" cy="${Y(ys[ys.length - 1])}" r="4" fill="${color}"/>` : ""}
  </svg>`;
}

function drawDashboard() {
  const r = state.result, u = state.user;
  const b = r.biomarkers, cats = r.categories;
  const cvdColor = catColor(r.cvd_category);
  const trendIcon = { Increasing: "▲", Decreasing: "▼", Stable: "▬" }[r.trend] || "▬";
  const bios = [
    ["TC", "Total cholesterol", "mg/dL", 0], ["HDL", "HDL cholesterol", "mg/dL", 0],
    ["LDL", "LDL cholesterol", "mg/dL", 0], ["TG", "Triglycerides", "mg/dL", 0],
    ["CRP", "C-Reactive protein", "mg/L", 2], ["DD", "D-Dimer", "mg/L", 2],
  ];
  const drivers = (r.top_contributions[state.driverTab] || []).slice(0, 5);

  app.innerHTML = `
  <div class="topbar">
    <div class="who"><span style="width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center;color:#fff">❤</span>
      <b>CardioCore</b>
      <span class="chip" style="background:#EFF2F6;color:var(--muted);margin-left:8px"><span class="dot"></span>Twin active</span></div>
    <div class="who">
      <button class="btn btn-ghost" id="edit-profile">Edit profile</button>
      <div class="avatar">${esc((u.name || "P")[0].toUpperCase())}</div>
      <button class="btn btn-link" id="logout">Sign out</button>
    </div>
  </div>

  <div class="container fade-in">
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:20px;flex-wrap:wrap;gap:10px">
      <div>
        <h2 style="font-size:24px;font-weight:800;letter-spacing:-.4px">Hi ${esc((u.name || "there").split(" ")[0])} — here's your twin</h2>
        <p class="muted small">${r.days}-day continuous estimate · ${r.factors.length} factors · cluster: <b>${esc(r.cluster.label)}</b> (${(r.cluster.confidence * 100).toFixed(0)}% confidence)</p>
      </div>
      <div class="whatif">
        <span class="small muted">Simulation:</span>
        ${["stable", "improving", "declining"].map((s) => `<button class="btn ${state.scenario === s ? "btn-primary" : "btn-ghost"}" data-scen="${s}">${s}</button>`).join("")}
      </div>
    </div>

    <div class="dash-hero">
      <div class="card gauge-card">
        <h4>Overall CVD risk</h4>
        <div class="gauge-wrap">${gaugeSVG(r.cvd_score, cvdColor)}
          <div class="gauge-center"><div class="num" style="color:${cvdColor}">${(r.cvd_score * 100).toFixed(0)}</div><div class="of">/ 100 · ${esc(r.cvd_category)}</div></div>
        </div>
        <span class="chip" style="background:${cvdColor}1a;color:${cvdColor}"><span class="dot"></span>${trendIcon} ${esc(r.trend || "—")}</span>
      </div>
      <div class="card stat-card">
        <div class="label">Coronary thrombosis risk</div>
        <div class="big" style="color:${catColor(r.ctr_category + " ")}">${r.ctr.toFixed(2)}</div>
        <div class="sub">${esc(r.ctr_category)} · LDL 35% + D-Dimer 40% + CRP 25%</div>
        <div class="path-bar" style="margin-top:10px"><i style="width:${r.ctr * 100}%;background:${catColor(r.ctr_category + " ")}"></i></div>
      </div>
      <div class="card stat-card">
        <div class="label">Estimated biomarkers</div>
        <div class="big">${cats.TC === "Desirable" ? "✓" : "!"} TC ${fmt(b.TC)} <small style="font-size:14px;color:var(--faint)">mg/dL</small></div>
        <div class="sub">HDL ${fmt(b.HDL)} · LDL ${fmt(b.LDL)} · TG ${fmt(b.TG)} · CRP ${fmt(b.CRP, 2)} · DD ${fmt(b.DD, 2)}</div>
        <div class="sub" style="margin-top:auto;padding-top:10px">Derived: non-HDL ${fmt(r.derived.Non_HDL)} · TC/HDL ${fmt(r.derived.TC_HDL_ratio, 2)} · AIP ${fmt(r.derived.AIP, 2)}</div>
      </div>
    </div>

    <div class="bio-grid">
      ${bios.map(([k, name, unit, dec]) => `
      <div class="card bio-card" style="border-top:3px solid ${catColor(cats[k])}">
        <div class="b-name"><span>${name}</span><span class="fpill ${["HDL"].includes(k) && cats[k] === "Protective" ? "prot" : ""}">${k}</span></div>
        <div class="b-val">${fmt(b[k], dec)} <small>${unit}</small></div>
        <div class="b-cat" style="color:${catColor(cats[k])}">● ${esc(cats[k])}</div>
      </div>`).join("")}
    </div>

    <div class="two-col">
      <div class="card panel">
        <div class="sec-title">Pathway risks</div>
        <div class="sec-sub">Which disease pathway is most active for you right now</div>
        ${Object.entries(r.pathways).map(([k, v]) => {
          const c = v > 0.66 ? "#D7263D" : v > 0.33 ? "#E06C00" : "#12876F";
          return `<div class="path-row"><span class="pname">${esc(k)}</span>
            <div class="path-bar"><i style="width:${(v * 100).toFixed(1)}%;background:${c}"></i></div>
            <span class="pval" style="color:${c}">${v.toFixed(2)}</span></div>`;
        }).join("")}
      </div>
      <div class="card panel">
        <div class="sec-title">Biomarker risk radar</div>
        <div class="sec-sub">Larger area = higher combined risk (each axis 0 → 1)</div>
        ${radarSVG(b)}
      </div>
    </div>

    <div class="two-col">
      <div class="card panel">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div><div class="sec-title">What's driving your ${state.driverTab === "DD" ? "D-Dimer" : state.driverTab === "CRP" ? "CRP" : "cholesterol"}</div>
          <div class="sec-sub" style="margin-bottom:0">Top 5 contributing factors (explainable AI)</div></div>
          <div class="whatif">${["TC", "CRP", "DD"].map((t) =>
            `<button class="btn ${state.driverTab === t ? "btn-primary" : "btn-ghost"}" data-tab="${t}">${t === "DD" ? "D-Dimer" : t}</button>`).join("")}</div>
        </div>
        <div style="margin-top:14px">
        ${drivers.length ? drivers.map((c, i) => `
          <div class="driver"><span class="rank">${i + 1}</span>
            <span class="dname">${esc(c.name)} <span class="muted small">(${esc(c.fid)})</span></span>
            <span class="dcontrib ${c.contribution > 0 ? "up" : "down"}">${c.contribution > 0 ? "▲" : "▼"} ${Math.abs(c.contribution).toFixed(1)}</span>
          </div>`).join("") : '<p class="muted small">No data</p>'}
        </div>
      </div>
      <div class="card panel">
        <div class="sec-title">Alerts <span class="fpill" style="margin-left:6px">${r.n_alerts}</span></div>
        <div class="sec-sub">Graded, deduplicated — critical alerts repeat until resolved</div>
        ${r.alerts.length ? r.alerts.slice(-5).reverse().map((a) => `
          <div class="alert-item ${a.level}">
            <span class="lvl">${a.level.toUpperCase()}</span>
            <span>${esc(a.message)}</span>
          </div>`).join("") : `<div class="alert-item info"><span class="lvl">OK</span><span>No alerts — all biomarkers within safe ranges.</span></div>`}
      </div>
    </div>

    <div class="card panel" style="margin-bottom:18px">
      <div class="sec-title">CVD risk trajectory</div>
      <div class="sec-sub">${r.days}-day continuous estimate · scenario: <b>${esc(r.scenario)}</b> · final trend ${trendIcon} ${esc(r.trend || "—")}</div>
      ${sparklineSVG(r.trajectory, cvdColor)}
    </div>

    <div class="card panel factors-panel">
      <div class="sec-title">All 72 factors — your current values</div>
      <div class="sec-sub">Normalized 0–1 (F = (raw − min)/(max − min)) · your questionnaire drives 30+ of them</div>
      <input class="search" id="factor-search" placeholder="Search factors… e.g. NIR, blood pressure, smoking" value="${esc(state.factorQuery)}" />
      <div style="overflow:auto;max-height:420px">
      <table class="factors">
        <thead><tr><th>ID</th><th>Factor</th><th>Category</th><th>Value</th><th>Reading</th></tr></thead>
        <tbody>
        ${r.factors.filter((f) => !state.factorQuery ||
            (f.name + f.category + f.id).toLowerCase().includes(state.factorQuery.toLowerCase()))
          .map((f) => `
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

  $("#logout").onclick = () => doLogout();
  $("#edit-profile").onclick = async () => {
    const me = await api("/api/me");
    state.profile = me.profile;
    state._step = 0;
    nav("#/onboarding");
  };
  $$("[data-scen]").forEach((btn) => btn.onclick = async () => {
    if (state.scenario === btn.dataset.scen) return;
    state.scenario = btn.dataset.scen;
    state.result = null;
    renderDashboard();
  });
  $$("[data-tab]").forEach((btn) => btn.onclick = () => { state.driverTab = btn.dataset.tab; drawDashboard(); });
  const search = $("#factor-search");
  search.oninput = () => {
    state.factorQuery = search.value;
    const q = search.value.toLowerCase();
    $$("table.factors tbody tr").forEach((tr) => {
      tr.style.display = tr.textContent.toLowerCase().includes(q) ? "" : "none";
    });
  };
}

/* ============================================================
   ROUTER
   ============================================================ */
async function route() {
  const h = location.hash || "#/auth";
  if (!state.token) { if (h !== "#/auth") return nav("#/auth"); return renderAuth(); }
  if (h === "#/auth") {
    try { const me = await api("/api/me"); state.profile = me.profile; return nav(me.has_profile ? "#/dashboard" : "#/onboarding"); }
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
