"""
server.py — CardioCore patient portal backend (FastAPI)
=======================================================
Serves the patient-side web app (static SPA in ``static/``) and a JSON API:

    POST /api/auth/signup    {email, name, password}        -> session token
    POST /api/auth/login     {email, password}              -> session token
    POST /api/auth/logout
    GET  /api/me                                           -> user + profile status
    PUT  /api/profile        {profile fields}               -> saves questionnaire
    POST /api/twin/simulate  {days?, scenario?}             -> full twin estimate

Auth: PBKDF2-hashed passwords (120k iterations), random session tokens
(7-day expiry) sent as ``Authorization: Bearer <token>``.
Storage: SQLite (``app.db`` next to this file; git-ignored).

The scoring endpoint reuses the real engine: the saved questionnaire builds a
``UserProfile``, the OU simulator generates the daily wearable stream, and the
``BiomarkerTwin`` runs the full 72-factor → biomarker → risk pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biomarker_twin import BiomarkerTwin  # noqa: E402
from src.data_simulator import PRESET_PROFILES, SimulatedWearableSource, UserProfile  # noqa: E402
from src.factors72 import FACTORS, FACTOR_IDS  # noqa: E402

DB_PATH = Path(__file__).parent / "app.db"
TOKEN_TTL = 7 * 24 * 3600
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = FastAPI(title="CardioCore Patient Portal", version="1.0")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tokens(
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
    expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS profiles(
    user_id INTEGER PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at REAL NOT NULL);
"""


def db() -> sqlite3.Connection:
    """Open a connection; the schema check is idempotent and *self-healing*
    (even if the DB file is deleted at runtime, the next request rebuilds it)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def init_db() -> None:
    with db() as conn:
        pass


@app.exception_handler(Exception)
async def unhandled_error(request, exc: Exception):
    """Surface unexpected errors to the client instead of an opaque 500."""
    import traceback
    traceback.print_exc()
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"Server error: {exc}"})


@app.get("/api/health")
def health():
    with db() as conn:
        n_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    return {"ok": True, "users": int(n_users)}


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def current_user(authorization: Optional[str]) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    with db() as conn:
        row = conn.execute(
            "SELECT u.*, t.expires_at FROM tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token = ?", (token,)).fetchone()
        if not row or row["expires_at"] < time.time():
            raise HTTPException(401, "Session expired – please log in again")
        return row


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SignupIn(BaseModel):
    email: str
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: str
    password: str


class ProfileIn(BaseModel):
    # identity / basics (F48, F49, F50, F51, F52, F53, F54)
    age: int = Field(ge=18, le=100)
    sex: str = Field(pattern="^(male|female)$")
    height_cm: float = Field(gt=100, le=230)
    weight_kg: float = Field(gt=30, le=300)
    waist_cm: Optional[float] = Field(default=None, ge=50, le=200)
    hip_cm: Optional[float] = Field(default=None, ge=50, le=220)
    ethnicity: str = Field(default="other")          # category, mapped to F52 score
    # medical history (F55-F60) — simple yes/no questions
    diabetic: bool = False                            # diabetes / pre-diabetes
    high_chol: bool = False                           # diagnosed high cholesterol
    hypertension: bool = False                        # hypertension history
    clot: bool = False                                # previous blood clot
    thyroid: bool = False                             # thyroid disorder
    family_history: bool = False                      # family history of heart disease
    # diet questionnaire (F61-F65) — frequency words, mapped to 0-1 server-side
    sat_fat_freq: str = Field(default="sometimes", pattern="^(never|rarely|sometimes|often|very_often)$")
    sugar_freq: str = Field(default="sometimes", pattern="^(never|rarely|sometimes|often|very_often)$")
    veg_freq: str = Field(default="sometimes", pattern="^(never|rarely|sometimes|often|very_often)$")
    alcohol_freq: str = Field(default="rarely", pattern="^(never|rarely|sometimes|often|very_often)$")
    omega3_freq: str = Field(default="sometimes", pattern="^(never|rarely|sometimes|often|very_often)$")
    # smoking (F66, F67) — computed from status + consumption
    smoke_status: str = Field(default="never", pattern="^(never|current|former)$")
    cigarettes_per_day: float = Field(default=0, ge=0, le=80)
    years_smoked: float = Field(default=0, ge=0, le=70)
    # optional known baselines (personalize the simulation)
    baseline_resting_hr: Optional[float] = Field(default=None, ge=35, le=130)
    baseline_hrv_rmssd: Optional[float] = Field(default=None, ge=8, le=180)
    baseline_systolic: Optional[float] = Field(default=None, ge=80, le=220)
    baseline_diastolic: Optional[float] = Field(default=None, ge=50, le=140)
    baseline_total_cholesterol: Optional[float] = Field(default=None, ge=100, le=400)
    baseline_hdl: Optional[float] = Field(default=None, ge=20, le=120)
    baseline_triglycerides: Optional[float] = Field(default=None, ge=40, le=600)


# frequency words -> normalized 0-1 factor values
FREQ_MAP = {"never": 0.05, "rarely": 0.25, "sometimes": 0.5, "often": 0.75, "very_often": 0.95}
# ethnicity categories -> population risk background (F52)
ETHNICITY_MAP = {
    "european": 0.30, "east_asian": 0.30, "south_asian": 0.55,
    "african": 0.45, "hispanic": 0.35, "middle_eastern": 0.40, "other": 0.30,
}


def smoking_scores(status: str, cigs_per_day: float, years: float) -> tuple[float, float]:
    """Return (F66 intensity 0-1, F67 pack-years). Pack-years = cigs×years/20."""
    cigs = max(0.0, float(cigs_per_day))
    years = max(0.0, float(years))
    pack_years = min(cigs * years / 20.0, 80.0)
    if status != "current":
        return 0.0, pack_years          # former smokers keep pack-year risk
    if cigs <= 5:
        intensity = 0.3
    elif cigs <= 10:
        intensity = 0.5
    elif cigs <= 20:
        intensity = 0.75
    else:
        intensity = 1.0
    return intensity, pack_years


# --------------------------------------------------------------------------- #
# Profile -> engine mapping
# --------------------------------------------------------------------------- #
def profile_to_user_profile(p: Dict[str, Any]) -> UserProfile:
    whr = None
    if p.get("waist_cm") and p.get("hip_cm"):
        whr = round(p["waist_cm"] / p["hip_cm"], 3)
    intensity, pack_years = smoking_scores(
        p.get("smoke_status", "never"), p.get("cigarettes_per_day", 0), p.get("years_smoked", 0))
    preset = PRESET_PROFILES["typical"]

    def base(name):
        v = p.get(name)
        return float(v) if v not in (None, "") else float(getattr(preset, name))

    return UserProfile(
        name=p.get("name", "Patient"),
        age=int(p["age"]), sex=p["sex"],
        height_cm=float(p["height_cm"]), weight_kg=float(p["weight_kg"]),
        waist_cm=p.get("waist_cm"), waist_hip_ratio=whr,
        ethnicity_risk=ETHNICITY_MAP.get(p.get("ethnicity", "other"), 0.3),
        smoker=(p.get("smoke_status") == "current"),
        diabetic=bool(p.get("diabetic")),
        family_history=bool(p.get("family_history")),
        # yes/no -> graded history factors
        chol_history=0.85 if p.get("high_chol") else 0.05,
        htn_history=0.85 if p.get("hypertension") else 0.05,
        clot_history=0.90 if p.get("clot") else 0.0,
        thyroid=0.80 if p.get("thyroid") else 0.03,
        # frequency words -> normalized factors (protective ones inverted)
        smoking_intensity=intensity,
        pack_years=pack_years if pack_years > 0 else (12.0 if p.get("smoke_status") == "current" else 0.0),
        sat_fat=FREQ_MAP[p.get("sat_fat_freq", "sometimes")],
        sugar=FREQ_MAP[p.get("sugar_freq", "sometimes")],
        vegetables=1 - FREQ_MAP[p.get("veg_freq", "sometimes")],
        alcohol=FREQ_MAP[p.get("alcohol_freq", "rarely")],
        omega3=1 - FREQ_MAP[p.get("omega3_freq", "sometimes")],
        baseline_resting_hr=base("baseline_resting_hr"),
        baseline_hrv_rmssd=base("baseline_hrv_rmssd"),
        baseline_systolic=base("baseline_systolic"),
        baseline_diastolic=base("baseline_diastolic"),
        baseline_total_cholesterol=base("baseline_total_cholesterol"),
        baseline_hdl=base("baseline_hdl"),
        baseline_triglycerides=base("baseline_triglycerides"),
    )


def run_twin_estimate(profile: Dict[str, Any], days: int = 60,
                      scenario: str = "stable", seed: int = 42) -> Dict[str, Any]:
    """Full engine run for the patient's profile -> JSON-safe result dict."""
    import datetime as _dt

    user = profile_to_user_profile(profile)
    twin = BiomarkerTwin(user.to_dict())
    source = SimulatedWearableSource(user, scenario=scenario, seed=seed, lab_interval_days=None)
    for i, obs in enumerate(source.next_days(days)):
        twin.update(obs, day_index=i)
    s = twin.state

    # cluster population scatter (same deterministic fit as the twin's assigner)
    from src.biomarker_twin import BiomarkerClusterer, generate_cohort
    cohort_X, cohort_cvd = generate_cohort(160, seed=7)
    clusterer = BiomarkerClusterer().fit(cohort_X, cohort_cvd)
    pts = clusterer.cohort_points_2d(cohort_X)
    step = max(1, len(pts) // 90)  # subsample for payload size
    scatter = [{"x": round(float(pts.iloc[i]["pc1"]), 2),
                "y": round(float(pts.iloc[i]["pc2"]), 2),
                "cluster": pts.iloc[i]["cluster"]} for i in range(0, len(pts), step)]

    trajectory = []
    for i, (_, cvd), row in zip(range(len(twin.risk_history)), twin.risk_history, twin.biomarker_history):
        trajectory.append({"day": i, "cvd": round(float(cvd), 4),
                           **{b: round(float(row[b]), 2) for b in ("TC", "HDL", "LDL", "TG", "CRP", "DD")}})
    factors = [
        {"id": fid, "name": FACTORS[fid].name, "category": FACTORS[fid].category,
         "value": s.factors.get(fid), "reading": FACTORS[fid].direction}
        for fid in FACTOR_IDS
    ]
    intensity, pack_years = smoking_scores(
        profile.get("smoke_status", "never"),
        profile.get("cigarettes_per_day", 0), profile.get("years_smoked", 0))
    return {
        "patient": {
            "name": user.name, "age": user.age, "sex": user.sex, "bmi": user.bmi,
            "waist_hip_ratio": user.waist_hip_ratio,
            "smoke_status": profile.get("smoke_status", "never"),
            "pack_years": round(pack_years, 1),
        },
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="minutes"),
        "biomarkers": s.biomarkers,
        "derived": s.derived,
        "categories": s.categories,
        "adjusted": s.adjusted,
        "cvd_score": s.cvd_score, "cvd_category": s.cvd_category,
        "ctr": s.ctr, "ctr_category": s.ctr_category,
        "pathways": s.pathway_risk,
        "cluster": {"label": s.cluster["label"], "confidence": s.cluster["confidence"],
                    "pc1": s.cluster.get("pc1"), "pc2": s.cluster.get("pc2"),
                    "scatter": scatter},
        "trend": twin.trend,
        "baselines": twin.baseline_table(),
        "top_contributions": s.top_contributions,
        "alerts": twin.alerts_raised[-12:],
        "n_alerts": len(twin.alerts_raised),
        "trajectory": trajectory,
        "factors": factors,
        "bmi": user.bmi,
        "scenario": scenario, "days": days,
        # simulated demo stream (no physical sensors attached in this prototype)
        "wearable_status": [
            {"sensor": name, "status": "connected"}
            for name in ("ECG", "PPG", "SpO2", "Blood Pressure", "NIR",
                         "Temperature", "GSR", "Bioimpedance", "IMU")
        ],
    }


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.post("/api/auth/signup")
def signup(body: SignupIn):
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "Please enter a valid email address")
    salt = secrets.token_hex(16)
    token = secrets.token_urlsafe(32)
    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users(email, name, pw_hash, salt, created_at) VALUES (?,?,?,?,?)",
                (email, body.name.strip(), hash_password(body.password, salt), salt, time.time()))
            uid = cur.lastrowid
            conn.execute("INSERT INTO tokens VALUES (?,?,?)",
                         (token, uid, time.time() + TOKEN_TTL))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists")
    return {"token": token, "name": body.name.strip(), "email": email, "has_profile": False}


@app.post("/api/auth/login")
def login(body: LoginIn):
    email = body.email.strip().lower()
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user or hash_password(body.password, user["salt"]) != user["pw_hash"]:
            raise HTTPException(401, "Incorrect email or password")
        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO tokens VALUES (?,?,?)",
                     (token, user["id"], time.time() + TOKEN_TTL))
        prof = conn.execute("SELECT 1 FROM profiles WHERE user_id = ?", (user["id"],)).fetchone()
        return {"token": token, "name": user["name"], "email": email, "has_profile": bool(prof)}


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        with db() as conn:
            conn.execute("DELETE FROM tokens WHERE token = ?",
                         (authorization.split(" ", 1)[1],))
    return {"ok": True}


@app.get("/api/me")
def me(authorization: Optional[str] = Header(None)):
    user = current_user(authorization)
    with db() as conn:
        prof = conn.execute("SELECT data FROM profiles WHERE user_id = ?",
                            (user["id"],)).fetchone()
    return {"name": user["name"], "email": user["email"],
            "has_profile": bool(prof),
            "profile": json.loads(prof["data"]) if prof else None}


# --------------------------------------------------------------------------- #
# Profile + twin
# --------------------------------------------------------------------------- #
@app.put("/api/profile")
def save_profile(body: ProfileIn, authorization: Optional[str] = Header(None)):
    user = current_user(authorization)
    data = body.model_dump()
    data["name"] = user["name"]
    with db() as conn:
        conn.execute(
            "INSERT INTO profiles(user_id, data, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (user["id"], json.dumps(data), time.time()))
    return {"ok": True, "has_profile": True}


@app.post("/api/twin/simulate")
def simulate(body: Dict[str, Any], authorization: Optional[str] = Header(None)):
    user = current_user(authorization)
    with db() as conn:
        prof = conn.execute("SELECT data FROM profiles WHERE user_id = ?",
                            (user["id"],)).fetchone()
    if not prof:
        raise HTTPException(400, "Please complete your health profile first")
    days = int(body.get("days", 60))
    scenario = str(body.get("scenario", "stable"))
    if scenario not in ("stable", "improving", "declining"):
        raise HTTPException(422, "scenario must be stable|improving|declining")
    days = min(max(days, 30), 180)
    seed = int(body.get("seed", 42))
    return run_twin_estimate(json.loads(prof["data"]), days=days, scenario=scenario, seed=seed)


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


init_db()
