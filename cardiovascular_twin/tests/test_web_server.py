"""Tests for the patient-portal backend (web/server.py) — logic layer."""

import json
import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

import server  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "test.db")
    server.init_db()
    return server


class TestAuthStore:
    def test_signup_login_flow(self, temp_db):
        body = server.SignupIn(email="A@B.IO", name="  Alice  ", password="secret123")
        res = temp_db.signup(body)
        assert res["token"] and res["name"] == "Alice" and res["has_profile"] is False

        login = temp_db.login(server.LoginIn(email="a@b.io", password="secret123"))
        assert login["token"] and login["name"] == "Alice"

    def test_wrong_password_rejected(self, temp_db):
        temp_db.signup(server.SignupIn(email="x@y.z", name="X", password="secret123"))
        with pytest.raises(Exception):
            temp_db.login(server.LoginIn(email="x@y.z", password="nope"))

    def test_duplicate_email_rejected(self, temp_db):
        temp_db.signup(server.SignupIn(email="x@y.z", name="X", password="secret123"))
        with pytest.raises(Exception):
            temp_db.signup(server.SignupIn(email="x@y.z", name="X2", password="secret123"))

    def test_password_hashing_is_salted(self):
        h1 = server.hash_password("pw123456", "aa" * 16)
        h2 = server.hash_password("pw123456", "bb" * 16)
        assert h1 != h2 and len(h1) == 64

    def test_token_required(self, temp_db):
        with pytest.raises(Exception):
            temp_db.current_user(None)
        with pytest.raises(Exception):
            temp_db.current_user("Bearer invalidtoken")


class TestProfileFlow:
    def test_profile_save_and_me(self, temp_db):
        res = temp_db.signup(server.SignupIn(email="p@q.r", name="Pat", password="secret123"))
        hdr = {"Authorization": f"Bearer {res['token']}"}
        prof = server.ProfileIn(age=44, sex="female", height_cm=165, weight_kg=68,
                                waist_cm=82, hip_cm=98, smoker=False, diabetic=False,
                                family_history=True)
        out = temp_db.save_profile(prof, hdr["Authorization"])
        assert out["has_profile"] is True

        me = temp_db.me(hdr["Authorization"])
        assert me["has_profile"] is True
        assert me["profile"]["age"] == 44 and me["profile"]["family_history"] is True

    def test_profile_validation(self):
        with pytest.raises(Exception):
            server.ProfileIn(age=10, sex="male", height_cm=180, weight_kg=80)  # under 18
        with pytest.raises(Exception):
            server.ProfileIn(age=40, sex="alien", height_cm=180, weight_kg=80)


class TestEngineBridge:
    PROFILE = {"age": 56, "sex": "male", "height_cm": 172, "weight_kg": 92,
               "waist_cm": 104, "hip_cm": 106, "ethnicity_risk": 0.5,
               "diabetic": True, "family_history": True, "chol_history": 0.8,
               "smoker": True, "pack_years": 28, "sat_fat": 0.8, "sugar": 0.75,
               "vegetables": 0.2, "alcohol": 0.6, "omega3": 0.15}

    def test_profile_mapping(self):
        up = server.profile_to_user_profile(self.PROFILE)
        assert up.age == 56 and up.bmi == pytest.approx(92 / 1.72 ** 2, abs=0.05)
        assert up.waist_hip_ratio == pytest.approx(104 / 106, abs=0.01)
        assert up.smoker and up.diabetic

    def test_run_twin_estimate_shape(self):
        r = server.run_twin_estimate(self.PROFILE, days=45, scenario="declining")
        assert set(r["biomarkers"]) == {"TC", "HDL", "LDL", "TG", "CRP", "DD"}
        assert 0 <= r["cvd_score"] <= 1
        assert len(r["factors"]) == 72
        assert len(r["trajectory"]) == 45
        assert r["cluster"]["label"] in ("Low Risk", "Moderate Risk", "High Risk", "Very High Risk")
        assert json.dumps(r)  # fully JSON-serializable

    def test_scenarios_differentiate(self):
        dec = server.run_twin_estimate(self.PROFILE, scenario="declining")["cvd_score"]
        imp = server.run_twin_estimate(self.PROFILE, scenario="improving")["cvd_score"]
        assert dec > imp

    def test_at_risk_higher_than_healthy(self):
        healthy = dict(self.PROFILE, age=32, weight_kg=62, height_cm=175,
                       diabetic=False, smoker=False, family_history=False,
                       waist_cm=80, hip_cm=96, chol_history=0.1,
                       sat_fat=0.2, sugar=0.2, vegetables=0.85, alcohol=0.05, omega3=0.8)
        h = server.run_twin_estimate(healthy, scenario="stable")
        a = server.run_twin_estimate(self.PROFILE, scenario="stable")
        assert a["cvd_score"] > h["cvd_score"] + 0.1
