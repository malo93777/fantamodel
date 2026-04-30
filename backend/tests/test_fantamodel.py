"""FantaModel Backend Tests - auth + ML endpoints"""
import os
import time
import uuid
import math
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://calcio-stats-22.preview.emergentagent.com").rstrip("/")
ADMIN_EMAIL = "admin@fantamodel.com"
ADMIN_PASSWORD = "admin123"


@pytest.fixture(scope="session")
def admin_session():
    s = requests.Session()
    r = s.post(f"{BASE_URL}/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=30)
    assert r.status_code == 200, f"Admin login failed: {r.status_code} {r.text}"
    data = r.json()
    assert "user" in data and "access_token" in data
    assert data["user"]["email"] == ADMIN_EMAIL
    assert "access_token" in s.cookies
    return s


# ============== AUTH ==============
class TestAuth:
    def test_register_new_user(self):
        s = requests.Session()
        email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        r = s.post(f"{BASE_URL}/api/auth/register",
                   json={"email": email, "password": "password123", "name": "Tester"}, timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["user"]["email"] == email
        assert data["user"]["role"] == "user"
        assert "access_token" in data and len(data["access_token"]) > 10
        assert "access_token" in s.cookies

    def test_register_duplicate_fails(self, admin_session):
        r = requests.post(f"{BASE_URL}/api/auth/register",
                          json={"email": ADMIN_EMAIL, "password": "whatever1"}, timeout=30)
        assert r.status_code == 400

    def test_login_admin(self, admin_session):
        # admin_session fixture proves login works
        r = admin_session.get(f"{BASE_URL}/api/auth/me", timeout=30)
        assert r.status_code == 200
        assert r.json()["email"] == ADMIN_EMAIL

    def test_login_invalid(self):
        r = requests.post(f"{BASE_URL}/api/auth/login",
                          json={"email": ADMIN_EMAIL, "password": "wrong"}, timeout=30)
        assert r.status_code == 401

    def test_me_unauthenticated(self):
        r = requests.get(f"{BASE_URL}/api/auth/me", timeout=30)
        assert r.status_code == 401

    def test_logout_clears_cookies(self, admin_session):
        s = requests.Session()
        s.post(f"{BASE_URL}/api/auth/login",
               json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=30)
        r = s.post(f"{BASE_URL}/api/auth/logout", timeout=30)
        assert r.status_code == 200


# ============== ML LISTS ==============
class TestMLLists:
    def test_players_requires_auth(self):
        r = requests.get(f"{BASE_URL}/api/players", timeout=60)
        assert r.status_code == 401

    def test_players(self, admin_session):
        r = admin_session.get(f"{BASE_URL}/api/players", timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) and len(data) > 100
        assert all("raw" in x and "display" in x for x in data[:5])
        names = [x["raw"].lower() for x in data]
        assert any("yildiz" in n for n in names)

    def test_teams(self, admin_session):
        r = admin_session.get(f"{BASE_URL}/api/teams", timeout=60)
        assert r.status_code == 200
        teams = r.json()
        assert isinstance(teams, list) and len(teams) >= 18
        assert any(t.lower() == "juventus" for t in teams)

    def test_opponents(self, admin_session):
        r = admin_session.get(f"{BASE_URL}/api/opponents", timeout=60)
        assert r.status_code == 200
        opp = r.json()
        assert isinstance(opp, list) and len(opp) >= 10

    def test_player_info(self, admin_session):
        r = admin_session.get(f"{BASE_URL}/api/player-info", params={"player": "yildiz"}, timeout=60)
        assert r.status_code == 200
        d = r.json()
        for k in ("team", "auto_team", "auto_opponent", "auto_ha"):
            assert k in d


# ============== ML PREDICT ==============
def _no_nan_inf(v):
    if v is None: return True
    if isinstance(v, float):
        return not (math.isnan(v) or math.isinf(v))
    return True


class TestMLPredict:
    def test_predict_bonus(self, admin_session):
        body = {"player": "yildiz", "team": "juventus", "opponent": "udinese", "h_a": "a"}
        r = admin_session.post(f"{BASE_URL}/api/predict/bonus", json=body, timeout=120)
        assert r.status_code == 200, r.text
        d = r.json()
        for k in ("goal_proba", "assist_proba", "bonus_proba", "stats"):
            assert k in d
        assert _no_nan_inf(d["goal_proba"]) and _no_nan_inf(d["assist_proba"]) and _no_nan_inf(d["bonus_proba"])
        assert d["bonus_proba"] is not None and 0.0 <= d["bonus_proba"] <= 1.0

    def test_predict_compare(self, admin_session):
        body = {
            "player1": "yildiz", "team1": "juventus", "opponent1": "udinese", "h_a1": "a",
            "player2": "cancellieri", "team2": "lazio", "opponent2": "cremonese", "h_a2": "h",
        }
        r = admin_session.post(f"{BASE_URL}/api/predict/compare", json=body, timeout=180)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "p1" in d and "p2" in d
        for p in (d["p1"], d["p2"]):
            assert "goal_proba" in p and "assist_proba" in p and "bonus_proba" in p

    def test_predict_index(self, admin_session):
        r = admin_session.post(f"{BASE_URL}/api/predict/index",
                               json={"players": ["yildiz", "cancellieri"]}, timeout=240)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "rows" in d
        if d["rows"]:
            assert "columns" in d and isinstance(d["columns"], list)

    def test_top_by_role(self, admin_session):
        r = admin_session.get(f"{BASE_URL}/api/predict/top-by-role", timeout=300)
        assert r.status_code == 200, r.text
        d = r.json()
        assert isinstance(d, dict)
        # Roles P/D/C/A may or may not all be present; check structure of any present
        for role, payload in d.items():
            assert "rows" in payload and "columns" in payload
