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
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
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
        """)


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
    ethnicity_risk: float = Field(default=0.3, ge=0, le=1)
    # medical history (F55-F60)
    diabetic: bool = False
    chol_history: float = Field(default=0.2, ge=0, le=1)
    htn_history: float = Field(default=0.1, ge=0, le=1)
    clot_history: float = Field(default=0.0, ge=0, le=1)
    thyroid: float = Field(default=0.05, ge=0, le=1)
    family_history: bool = False
    # lifestyle & diet (F61-F67)
    smoker: bool = False
    smoking_intensity: float = Field(default=0.6, ge=0, le=1)
    pack_years: float = Field(default=12.0, ge=0, le=80)
    sat_fat: float = Field(default=0.3, ge=0, le=1)
    sugar: float = Field(default=0.3, ge=0, le=1)
    vegetables: float = Field(default=0.6, ge=0, le=1)
    alcohol: float = Field(default=0.2, ge=0, le=1)
    omega3: float = Field(default=0.5, ge=0, le=1)
    # optional known baselines
    baseline_resting_hr: float = Field(default=64, ge=35, le=130)
    baseline_hrv_rmssd: float = Field(default=48, ge=8, le=180)
    baseline_systolic: float = Field(default=122, ge=80, le=220)
    baseline_diastolic: float = Field(default=79, ge=50, le=140)
    baseline_total_cholesterol: float = Field(default=195, ge=100, le=400)
    baseline_hdl: float = Field(default=50, ge=20, le=120)
    baseline_triglycerides: float = Field(default=140, ge=40, le=600)


# --------------------------------------------------------------------------- #
# Profile -> engine mapping
# --------------------------------------------------------------------------- #
def profile_to_user_profile(p: Dict[str, Any]) -> UserProfile:
    whr = None
    if p.get("waist_cm") and p.get("hip_cm"):
        whr = round(p["waist_cm"] / p["hip_cm"], 3)
    preset = PRESET_PROFILES["typical"]
    return UserProfile(
        name=p.get("name", "Patient"),
        age=int(p["age"]), sex=p["sex"],
        height_cm=float(p["height_cm"]), weight_kg=float(p["weight_kg"]),
        waist_cm=p.get("waist_cm"), waist_hip_ratio=whr,
        ethnicity_risk=float(p.get("ethnicity_risk", 0.3)),
        smoker=bool(p.get("smoker")), diabetic=bool(p.get("diabetic")),
        family_history=bool(p.get("family_history")),
        chol_history=float(p.get("chol_history", 0.2)),
        htn_history=float(p.get("htn_history", 0.1)),
        clot_history=float(p.get("clot_history", 0.0)),
        thyroid=float(p.get("thyroid", 0.05)),
        smoking_intensity=float(p.get("smoking_intensity", 0.6)),
        pack_years=float(p.get("pack_years", 12.0)),
        sat_fat=float(p.get("sat_fat", 0.3)), sugar=float(p.get("sugar", 0.3)),
        vegetables=float(p.get("vegetables", 0.6)), alcohol=float(p.get("alcohol", 0.2)),
        omega3=float(p.get("omega3", 0.5)),
        baseline_resting_hr=float(p.get("baseline_resting_hr", preset.baseline_resting_hr)),
        baseline_hrv_rmssd=float(p.get("baseline_hrv_rmssd", preset.baseline_hrv_rmssd)),
        baseline_systolic=float(p.get("baseline_systolic", preset.baseline_systolic)),
        baseline_diastolic=float(p.get("baseline_diastolic", preset.baseline_diastolic)),
        baseline_total_cholesterol=float(p.get("baseline_total_cholesterol", preset.baseline_total_cholesterol)),
        baseline_hdl=float(p.get("baseline_hdl", preset.baseline_hdl)),
        baseline_triglycerides=float(p.get("baseline_triglycerides", preset.baseline_triglycerides)),
    )


def run_twin_estimate(profile: Dict[str, Any], days: int = 60,
                      scenario: str = "stable", seed: int = 42) -> Dict[str, Any]:
    """Full engine run for the patient's profile -> JSON-safe result dict."""
    user = profile_to_user_profile(profile)
    twin = BiomarkerTwin(user.to_dict())
    source = SimulatedWearableSource(user, scenario=scenario, seed=seed, lab_interval_days=None)
    for i, obs in enumerate(source.next_days(days)):
        twin.update(obs, day_index=i)
    s = twin.state
    trajectory = []
    for i, (_, cvd), row in zip(range(len(twin.risk_history)), twin.risk_history, twin.biomarker_history):
        trajectory.append({"day": i, "cvd": round(float(cvd), 4),
                           **{b: round(float(row[b]), 2) for b in ("TC", "HDL", "LDL", "TG", "CRP", "DD")}})
    factors = [
        {"id": fid, "name": FACTORS[fid].name, "category": FACTORS[fid].category,
         "value": s.factors.get(fid), "reading": FACTORS[fid].direction}
        for fid in FACTOR_IDS
    ]
    return {
        "biomarkers": s.biomarkers,
        "derived": s.derived,
        "categories": s.categories,
        "adjusted": s.adjusted,
        "cvd_score": s.cvd_score, "cvd_category": s.cvd_category,
        "ctr": s.ctr, "ctr_category": s.ctr_category,
        "pathways": s.pathway_risk,
        "cluster": {"label": s.cluster["label"], "confidence": s.cluster["confidence"]},
        "trend": twin.trend,
        "baselines": twin.baseline_table(),
        "top_contributions": s.top_contributions,
        "alerts": twin.alerts_raised[-12:],
        "n_alerts": len(twin.alerts_raised),
        "trajectory": trajectory,
        "factors": factors,
        "bmi": user.bmi,
        "scenario": scenario, "days": days,
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
