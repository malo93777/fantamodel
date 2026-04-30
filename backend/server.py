from dotenv import load_dotenv
load_dotenv()

import os
import sys
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

import bcrypt
import jwt
import pandas as pd
import numpy as np
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Request, Response, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient

# Add fantamodel src to path
ROOT = Path(__file__).resolve().parent
FANTA_DIR = ROOT / "fantamodel"
sys.path.insert(0, str(FANTA_DIR / "src"))

# Stub streamlit to avoid heavy import (utils.py imports it but doesn't call it on API paths)
import types as _types
_st_stub = _types.ModuleType("streamlit")
def _noop(*a, **kw):
    return None
def _passthrough(func=None, **_):
    if func is None:
        return lambda f: f
    return func
_st_stub.cache_resource = _passthrough
_st_stub.cache_data = _passthrough
_st_stub.cache = _passthrough
for _n in ["write","markdown","title","header","subheader","metric","progress","warning","error","info","success","caption","dataframe","plotly_chart","button","selectbox","multiselect","radio","text_input","columns","spinner","set_page_config","switch_page"]:
    setattr(_st_stub, _n, _noop)
sys.modules["streamlit"] = _st_stub

import config as fm_config  # type: ignore
import utils as fm_utils  # type: ignore
import model_predict_fantavoto  # type: ignore

JWT_ALGORITHM = "HS256"
JWT_SECRET = os.environ["JWT_SECRET"]
MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]

# ============== ML CACHE ==============
_ML_CACHE = {}

def get_ml():
    if not _ML_CACHE:
        _ML_CACHE["models_goal"] = fm_utils.load_models()
        _ML_CACHE["models_assist"] = fm_utils.load_models_assist()
        _ML_CACHE["model_xg"] = fm_utils.load_xg_model()
        _ML_CACHE["model_fv"] = fm_utils.load_fv_model()
        _ML_CACHE["model_fv_gk"] = fm_utils.load_fv_model_gk()
        _ML_CACHE["df_goal"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.PROD_DATA_FILE_GOALS)
        _ML_CACHE["df_assist"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.PROD_DATA_FILE_ASSIST)
        _ML_CACHE["df_voti"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.PROD_DATA_FILE_VOTI)
        _ML_CACHE["df_teams"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.TEAMS_DATA_FILE)
        _ML_CACHE["df_teams_curr"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.CURRENT_SEASON_TEAMS_FILE)
        _ML_CACHE["df_next"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.NEXT_GAMES_FILE)
        _ML_CACHE["df_inf"] = pd.read_csv(fm_config.DATASET_DATA_DIR / fm_config.INFORTUNATI_FILE)
    return _ML_CACHE

# ============== AUTH HELPERS ==============
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(user_id: str, email: str) -> str:
    payload = {"sub": user_id, "email": email, "exp": datetime.now(timezone.utc) + timedelta(minutes=60), "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7), "type": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def set_auth_cookies(response: Response, access: str, refresh: str):
    response.set_cookie("access_token", access, httponly=True, secure=True, samesite="none", max_age=3600, path="/")
    response.set_cookie("refresh_token", refresh, httponly=True, secure=True, samesite="none", max_age=604800, path="/")

def serialize_user(u: dict) -> dict:
    return {
        "id": str(u["_id"]),
        "email": u["email"],
        "name": u.get("name", ""),
        "role": u.get("role", "user"),
        "subscription": u.get("subscription", "free"),
        "created_at": u.get("created_at").isoformat() if u.get("created_at") else None,
    }

# ============== APP ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo_client = AsyncIOMotorClient(MONGO_URL)
    app.state.db = app.state.mongo_client[DB_NAME]
    db = app.state.db
    await db.users.create_index("email", unique=True)
    await db.login_attempts.create_index("identifier")
    # seed admin
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@fantamodel.com").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one({
            "email": admin_email,
            "password_hash": hash_password(admin_password),
            "name": "Admin",
            "role": "admin",
            "subscription": "premium",
            "created_at": datetime.now(timezone.utc),
        })
    elif not verify_password(admin_password, existing["password_hash"]):
        await db.users.update_one({"email": admin_email}, {"$set": {"password_hash": hash_password(admin_password)}})
    yield
    app.state.mongo_client.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")

# ============== AUTH DEPENDENCY ==============
async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        db = app.state.db
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============== AUTH MODELS ==============
class RegisterReq(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: Optional[str] = ""

class LoginReq(BaseModel):
    email: EmailStr
    password: str

# ============== AUTH ENDPOINTS ==============
@api.post("/auth/register")
async def register(req: RegisterReq, response: Response):
    db = app.state.db
    email = req.email.lower()
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email già registrata")
    doc = {
        "email": email,
        "password_hash": hash_password(req.password),
        "name": req.name or "",
        "role": "user",
        "subscription": "free",
        "created_at": datetime.now(timezone.utc),
    }
    res = await db.users.insert_one(doc)
    doc["_id"] = res.inserted_id
    access = create_access_token(str(res.inserted_id), email)
    refresh = create_refresh_token(str(res.inserted_id))
    set_auth_cookies(response, access, refresh)
    return {"user": serialize_user(doc), "access_token": access}

@api.post("/auth/login")
async def login(req: LoginReq, response: Response):
    db = app.state.db
    email = req.email.lower()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email o password non validi")
    access = create_access_token(str(user["_id"]), email)
    refresh = create_refresh_token(str(user["_id"]))
    set_auth_cookies(response, access, refresh)
    return {"user": serialize_user(user), "access_token": access}

@api.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return {"ok": True}

@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return serialize_user(user)

# ============== ML HELPERS ==============
def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def player_history_stats(player: str, ml: dict):
    df_p = ml["df_goal"][ml["df_goal"]["player"].str.contains(player, case=False, na=False)]
    df_pa = ml["df_assist"][ml["df_assist"]["player"].str.contains(player, case=False, na=False)]
    if df_p.empty:
        return None
    curr = df_p[df_p["season"] == fm_config.CURRENT_SEASON]
    curr_a = df_pa[df_pa["season"] == fm_config.CURRENT_SEASON]
    canonical = df_p["player"].iloc[0]
    return {
        "player": canonical,
        "appearances": int(curr.shape[0]),
        "goals": int(curr["goals"].sum() or 0),
        "assists": int(curr_a["assists"].sum() or 0) if not curr_a.empty else 0,
        "xg_mean": safe_float(curr["sum_xG"].mean()),
        "xg_last5": safe_float(curr["sum_xG"].tail(5).mean()),
        "xa_mean": safe_float(curr_a["sum_xA"].mean()) if not curr_a.empty else None,
        "xa_last5": safe_float(curr_a["sum_xA"].tail(5).mean()) if not curr_a.empty else None,
        "shots_per_match": safe_float(curr["shots_perMatch"].mean()) if "shots_perMatch" in curr.columns else None,
        "shots_last5": safe_float(curr["shots_perMatch"].tail(5).mean()) if "shots_perMatch" in curr.columns else None,
        "history_xg": [
            {"date": str(r["date"]), "xG": safe_float(r["sum_xG"]), "goal": safe_float(r["goals"])}
            for _, r in curr.sort_values("date").tail(10).iterrows()
        ],
        "history_xa": [
            {"date": str(r["date"]), "xA": safe_float(r["sum_xA"]), "assist": safe_float(r["assists"])}
            for _, r in curr_a.sort_values("date").tail(10).iterrows()
        ] if not curr_a.empty else [],
    }

def predict_goal_assist(player: str, team: str, opponent: str, h_a: Optional[str], ml: dict):
    feats_g = list(ml["models_goal"]["poiss_reg"].feature_names_)
    if "finishing_form_resid" in feats_g:
        feats_g.remove("finishing_form_resid")
    goal_p = fm_utils.get_goal_prob(
        ml["model_xg"]["catboost_regressor_xg"], ml["models_goal"]["poiss_reg"], feats_g,
        player, team, opponent, ml["df_goal"], ml["df_teams"], ml["df_teams_curr"],
        ml["models_goal"]["lin"], fm_config.ROLE_STATS, h_a,
    )
    feats_a = ml["models_assist"]["poisson_reg_assist"].feature_names_
    assist_p = fm_utils.get_assist_prob(
        ml["models_assist"]["poisson_reg_assist"], feats_a,
        player, team, opponent, ml["df_assist"], ml["df_teams"], ml["df_teams_curr"], h_a,
    )
    return safe_float(goal_p), safe_float(assist_p)

# ============== ML ENDPOINTS ==============
@api.get("/players")
async def players(_: dict = Depends(get_current_user)):
    ml = get_ml()
    plist = sorted(ml["df_goal"]["player"].dropna().unique().tolist())
    return [{"raw": p, "display": p.title()} for p in plist]

@api.get("/teams")
async def teams(_: dict = Depends(get_current_user)):
    ml = get_ml()
    tl = sorted(ml["df_teams"][ml["df_teams"]["season"] == fm_config.CURRENT_SEASON]["Team"].dropna().unique().tolist())
    return tl

@api.get("/opponents")
async def opponents(_: dict = Depends(get_current_user)):
    ml = get_ml()
    ol = sorted(ml["df_goal"][ml["df_goal"]["season"] == fm_config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
    return ol

@api.get("/player-info")
async def player_info(player: str, _: dict = Depends(get_current_user)):
    ml = get_ml()
    team = fm_utils.get_latest_team(ml["df_goal"], player, "player_team") or ""
    try:
        squadra, avversario, ha = fm_utils.get_team_opponent_ha(player, ml["df_voti"], ml["df_next"])
    except Exception:
        squadra, avversario, ha = team, "", None
    return {
        "team": team,
        "auto_team": squadra or "",
        "auto_opponent": (avversario or "").title() if avversario else "",
        "auto_ha": ha,
    }

class BonusReq(BaseModel):
    player: str
    team: str
    opponent: str
    h_a: Optional[str] = None  # 'h' / 'a' / None

@api.post("/predict/bonus")
async def predict_bonus(req: BonusReq, _: dict = Depends(get_current_user)):
    ml = get_ml()
    try:
        goal_p, assist_p = predict_goal_assist(req.player, req.team, req.opponent, req.h_a, ml)
        if goal_p is None and assist_p is None:
            raise HTTPException(status_code=400, detail="Nessuna previsione disponibile per questo giocatore.")
        gp = goal_p or 0.0
        ap = assist_p or 0.0
        bonus = gp + ap - (gp * ap)
        history = player_history_stats(req.player, ml) or {}
        return {
            "player": req.player.title(),
            "team": req.team,
            "opponent": req.opponent,
            "goal_proba": goal_p,
            "assist_proba": assist_p,
            "bonus_proba": bonus,
            "stats": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore di predizione: {e}")

class CompareReq(BaseModel):
    player1: str
    team1: str
    opponent1: str
    h_a1: Optional[str] = None
    player2: str
    team2: str
    opponent2: str
    h_a2: Optional[str] = None

@api.post("/predict/compare")
async def predict_compare(req: CompareReq, _: dict = Depends(get_current_user)):
    ml = get_ml()
    def one(p, t, o, ha):
        gp, ap = predict_goal_assist(p, t, o, ha, ml)
        gp_v = gp or 0.0
        ap_v = ap or 0.0
        bonus = gp_v + ap_v - (gp_v * ap_v)
        h = player_history_stats(p, ml) or {}
        return {
            "player": p.title(),
            "team": t,
            "opponent": o,
            "goal_proba": gp,
            "assist_proba": ap,
            "bonus_proba": bonus,
            "stats": h,
        }
    try:
        p1 = one(req.player1, req.team1, req.opponent1, req.h_a1)
        p2 = one(req.player2, req.team2, req.opponent2, req.h_a2)
        return {"p1": p1, "p2": p2}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {e}")

class IndexReq(BaseModel):
    players: List[str]

@api.post("/predict/index")
async def predict_index(req: IndexReq, _: dict = Depends(get_current_user)):
    ml = get_ml()
    if not req.players:
        raise HTTPException(status_code=400, detail="Seleziona almeno un giocatore")
    df_voti_raw = ml["df_voti"]
    try:
        # Build input_data list as in streamlit page
        input_data = []
        for p in req.players:
            try:
                squadra, avversario, ha = fm_utils.get_team_opponent_ha(p, df_voti_raw, ml["df_next"])
            except Exception:
                squadra, avversario, ha = "", "", None
            input_data.append((p, squadra or "", avversario or "", ha))

        df_voti = fm_utils.prepare_voto_dataframe(df_voti_raw)
        roles = {}
        for p, *_ in input_data:
            pv = df_voti[df_voti["player_norm"] == p] if "player_norm" in df_voti.columns else pd.DataFrame()
            if not pv.empty:
                roles[p] = fm_utils.get_main_position_weighted(pv["fanta_role"], window=10, decay=0.8)
            else:
                roles[p] = None

        players_, teams_, opponents_, h_a_ = zip(*input_data)
        gk_idx = [i for i, p in enumerate(players_) if roles.get(p) == "P"]
        oth_idx = [i for i, p in enumerate(players_) if roles.get(p) != "P"]
        results = []
        if gk_idx:
            df_gk = model_predict_fantavoto.pred_voto_prod_gk(
                [players_[i] for i in gk_idx], [teams_[i] for i in gk_idx],
                [opponents_[i] for i in gk_idx], [h_a_[i] for i in gk_idx],
                df_voti, ml["df_teams"], ml["df_teams_curr"],
                ml["model_fv_gk"]["fantavoto_model_gk"], False
            )
            results.append(df_gk)
        if oth_idx:
            df_o = model_predict_fantavoto.pred_voto_prod(
                [players_[i] for i in oth_idx], [teams_[i] for i in oth_idx],
                [opponents_[i] for i in oth_idx], [h_a_[i] for i in oth_idx],
                df_voti, ml["df_goal"], ml["df_assist"], ml["df_teams"], ml["df_teams_curr"],
                ml["models_goal"], ml["models_assist"], ml["model_xg"],
                ml["model_fv"]["fantavoto_model"], False
            )
            results.append(df_o)
        if not results:
            return {"rows": []}
        df_pred = pd.concat(results, ignore_index=True)
        df_pred = fm_utils.prepare_df_for_display(df_pred).copy()
        rows = []
        for _, r in df_pred.iterrows():
            row = {}
            for c in df_pred.columns:
                v = r[c]
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[c] = safe_float(v)
                else:
                    row[c] = str(v) if v is not None else ""
            rows.append(row)
        return {"rows": rows, "columns": list(df_pred.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {e}")

@api.get("/predict/top-by-role")
async def top_by_role(_: dict = Depends(get_current_user)):
    ml = get_ml()
    try:
        df_voti = fm_utils.prepare_voto_dataframe(ml["df_voti"])
        results = model_predict_fantavoto.predizioni_per_ruolo(
            df_voti, ml["df_next"], ml["df_inf"],
            pipeline=ml["model_fv"]["fantavoto_model"],
            pipeline_gk=ml["model_fv_gk"]["fantavoto_model_gk"],
            top_n=10, debug=False,
        )
        out = {}
        for role, df in (results or {}).items():
            try:
                df = fm_utils.prepare_df_for_display(df)
            except Exception:
                pass
            rows = []
            for _, r in df.iterrows():
                row = {}
                for c in df.columns:
                    v = r[c]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        row[c] = safe_float(v)
                    else:
                        row[c] = str(v) if v is not None else ""
                rows.append(row)
            out[role] = {"rows": rows, "columns": list(df.columns)}
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {e}")

@api.get("/")
async def root():
    return {"name": "FantaModel API", "status": "ok"}

app.include_router(api)
