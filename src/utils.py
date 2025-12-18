import joblib
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from unidecode import unidecode
import os
from pathlib import Path
import config
from collections import Counter
from collections import defaultdict
from scipy.stats import skew, kurtosis
import streamlit as st
from datetime import datetime
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, f1_score
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import trange  # opzionale, per barra di progresso
# =====================================================
# 🔹 Caricamento modelli e scaler
# =====================================================
#@st.cache_resource
def load_models():
    return {
        "scaler_sumxg": joblib.load(config.SCALER_DIR / config.SCALER_XG),
        "lin_poly": joblib.load(config.MODEL_DIR / config.LIN_POLY),
        "scaler_features": joblib.load(config.SCALER_DIR / config.SCALER),
        "poiss_reg":joblib.load(config.MODEL_DIR / config.POISS_MODEL),
        "lin": joblib.load(config.MODEL_DIR / config.LIN)
    }

def load_models_assist():
    return {
        "scaler_features_assist": joblib.load(config.SCALER_DIR_ASSIST / config.SCALER),
        "poisson_reg_assist":  joblib.load(config.MODEL_DIR_ASSIST / config.POISS_MODEL_ASSIST)
    }

def get_latest_team(df, player_name):
    if player_name == "":
        return ""

    rows = df[df["player"] == player_name]
    if rows.empty:
        return ""

    return rows.sort_values("date").iloc[-1]["player_team"]


def get_assist_prob(model, features_names, player, team, opponent, df_orig, df_teams,df_teams_curr, h_a_player):

    """Prepara le feature per la predizione goal probability con modello CatBoost Reg."""

     # 1️⃣ Filtra storico giocatore
    df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
    if df.empty:
        print(f"⚠️ Nessun dato disponibile per {player}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= datetime.now()].reset_index(drop=True)


    if df.empty:
        print(f"⚠️ Nessuna partita valida (tutte future) per {player}")
        return None
    
    season = config.CURRENT_SEASON

    numeric_features, categorical_features = split_features_by_type(df, features_names)

    df = fill_missing_values_player_df(df, numeric_features, season_ref=season)

    # 3️⃣ Riempi i NaN
    df[features_names] = df[features_names].fillna(0)

    main_role = get_main_position_weighted(df["position"], window=10, decay=0.8)

    # 4️⃣ Recupera dati della squadra e avversario

    num_giornate = count_matchdays(df_teams_curr)

        #se ho un numero sufficiente di giornate, applico discriminante home/away
    if num_giornate >= 10: 
            h_a = get_h_a_opponent(h_a_player)
            #OPPONENT TEAM DATA home/away 
            opponent_xGA_90min_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
            xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

            #PLAYER TEAM DATA home/away
            team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
            xG_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)
    else:
            #OPPONENT TEAM DATA
            opponent_xGA_90min_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
            xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent,"", df_teams)

            #PLAYER TEAM DATA
            team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, "", df_teams)
            xG_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, "", df_teams)
        
    # 5️⃣ Calcola statistiche base del giocatore. Media delle ultime 12, che poi viene pesata esponenzialmente
    sum_xA = df["sum_xA"].tail(12).to_list()

    sum_xA_weighted = progressive_weighted_mean(sum_xA, alpha=0.2)

    sum_xA_weighted = weighted_xg_vs_opponent_mixed(sum_xA_weighted, df, opponent_xGA_90min_last5_per90, xGA_last5_opp, GA_last5_opp)

    sum_xA_weighted = weighted_xg_team_mixed(sum_xA_weighted,df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team)

    # Calcolo goals_last5 per la riga da prevedere
    if len(df) >= 5:
        # Prendi le ultime 5 partite, includendo l'ultima
        xA_last5 = df["sum_xA"].iloc[-5:].mean()
    else:
         # Se ci sono meno di 5 partite, usa tutte le partite disponibili
        xA_last5 = df["sum_xA"].mean()

        # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xA": sum_xA_weighted,
        "xA_last5": xA_last5    
    }])

    player_pos = df[categorical_features]

    # Aggiungi le dummy di posizione
    X_new_df = pd.concat([X_new_df.reset_index(drop=True), player_pos.tail(1).reset_index(drop=True)], axis=1)

    probs = predict_probabilities_poisson(
        model=model,
        X_new_df=X_new_df,
        main_role=main_role,
        alpha_fn=get_alpha_for_role,
        poisson_fn=poisson_goal_probs
        )

    return probs["p_any"]

def get_goal_prob(model_poiss, features_names, player, team, opponent, df_orig, df_teams, df_teams_curr_season, lin_model, ROLE_STATS,h_a_player):
    """
    Prepara le feature per la predizione goal probability con modello CatBoost Reg.
    """
    # 1️⃣ Filtra storico giocatore
    df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
    if df.empty:
        print(f"⚠️ Nessun dato disponibile per {player}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= datetime.now()].reset_index(drop=True)

    if df["season"].min() == config.CURRENT_SEASON:
        df = add_other_leagues_data(
            df, player,
            config.DATASET_DATA_DIR, config.GOALS_DATA_FILE_ALL_LEAGUES,
            config.CURRENT_SEASON
        )

    numeric_features, categorical_features = split_features_by_type(df, features_names)

    df = add_overperformance_features(df, ROLE_STATS, player_col="player", prod=True)
    df = compute_shot_quality_index_per_shot(df,prod=True)
    df = reduce_penalty_xg(df)

    df_teams_curr = compute_defensive_overperf_stats(df_teams_curr_season, team_col="team_name", ga_col="missed", xga_col="xGA", window=5)
    
    df["position"] = df["position"].apply(clean_position)
    # Rimuovo i "None"
    df = df[df["position"] != "None"]

    df = fill_missing_values_player_df(df, numeric_features, season_ref=config.CURRENT_SEASON)

    df[features_names] = df[features_names].fillna(0)

     # Calcolo residuo  per finishing_form
    df["xg_mean_12"] = (
        df.groupby("player")["sum_xG"]
        .apply(lambda x: x.rolling(window=12, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    df["xg_mean_12"] = df["xg_mean_12"].fillna(0)

    df = compute_finishing_form(df, window=12, use_rank=True, prod=True)

    # 2️⃣ Calcolo residuo lineare della finishing_form
    pred_lin = lin_model.predict(df[["xg_mean_12"]])
    df["finishing_form_resid"] = df["finishing_form"] - pred_lin

    #numero giornate già giocate nella stagione corrente
    num_giornate = count_matchdays(df_teams_curr)

    #se ho un numero sufficiente di giornate, applico discriminante home/away
    if num_giornate >= 10: 
        h_a = get_h_a_opponent(h_a_player)
        #OPPONENT TEAM DATA home/away 
        opponent_xGA_90min_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
        xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

        #PLAYER TEAM DATA home/away
        team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
        xG_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)
    else:
        #OPPONENT TEAM DATA
        opponent_xGA_90min_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
        xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent,"", df_teams)

        #PLAYER TEAM DATA
        team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, "", df_teams)
        G_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, "", df_teams)
    
    #prendo xg base player
    sum_xG_new = df["sum_xG"].tail(12).tolist()

    sum_xG_new = progressive_weighted_mean(sum_xG_new, alpha=0.3)

    # 5️⃣ Calcolo sum_xG corretto in base all’avversario e alla produzione offensiva della squadra
    sum_xG_new = weighted_xg_vs_opponent_mixed(sum_xG_new, df, opponent_xGA_90min_last5_per90, xGA_last5_opp, GA_last5_opp)

    sum_xG_new = weighted_xg_team_mixed(sum_xG_new, df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team)
        
    main_role = get_main_position_weighted(df["position"], window=10, decay=0.8)

    if player == "nico paz" or player == "odgaard":
        main_role = "FM" 

    sum_xG_new = adjust_xg_by_minutes(sum_xG_new,df["minutes_played"].rolling(window=5, min_periods=1).mean())

    # 6️⃣ Ultimo residuo disponibile
    finishing_form_resid = df["finishing_form_resid"].iloc[-1]
    overperf_value = df["overperf_role_resid"].iloc[-1]
    shot_quality_index = df["shot_quality_index"].iloc[-1]

    # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xG": sum_xG_new,
        "overperf_role_resid": overperf_value,  
        "shot_quality_index": shot_quality_index,
        "finishing_form_resid": finishing_form_resid
    }])

     # 8️⃣ Applica boost
     #for feature, factor in config.BOOST_FACTORS_XGB.items():
        #X_new_df[feature] = X_new_df[feature] * factor

    player_pos = df[categorical_features]

    # Aggiungi le dummy di posizione
    X_new_df = pd.concat([X_new_df.reset_index(drop=True), player_pos.tail(1).reset_index(drop=True)], axis=1)

    probs = predict_probabilities_poisson(
        model=model_poiss,
        X_new_df=X_new_df,
        main_role=main_role,
        alpha_fn=get_alpha_for_role,
        poisson_fn=poisson_goal_probs
    )

    return probs["p_any"]

def save_models(model, scaler_xg, scaler, poly, lin_poly, lin, is_baseline=False):
    """
    Salva il modello e lo scaler, chiedendo conferma se i file esistono già.
    """

    # Percorsi completi dei file
    if is_baseline:
         model_path = config.MODEL_DIR / config.POISS_MODEL_ASSIST if model is not None else None
    else:        
        model_path = config.MODEL_DIR / config.POISS_MODEL if model is not None else None
    
    scaler_path = config.SCALER_DIR / config.SCALER if scaler is not None else None
    scaler_xg_path = config.SCALER_DIR / config.SCALER_XG if scaler_xg is not None else None
    poly_path = config.MODEL_DIR / config.POLY_TRANSFORMER if poly is not None else None
    lin_poly_path = config.MODEL_DIR / config.LIN_POLY if lin_poly is not None else None
    lin_path = config.MODEL_DIR / config.LIN if lin is not None else None

    # Crea le cartelle se non esistono
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if scaler_path:
        os.makedirs(config.SCALER_DIR, exist_ok=True)
    if scaler_xg_path:
        os.makedirs(config.SCALER_DIR, exist_ok=True)
    if poly_path:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if lin_poly_path:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if lin_path:
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    # --- Salvataggio modello ---
    if model_path.exists():
        overwrite = input(f"⚠️ Il file '{model_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")
        else:
            joblib.dump(model, model_path)
            print(f"✅ Modello sovrascritto in: {model_path}")
    else:
        joblib.dump(model, model_path)
        print(f"✅ Modello salvato in: {model_path}")

    # --- Salvataggio scaler ---
    if scaler_path:
        if scaler_path.exists():
            overwrite = input(f"⚠️ Il file '{scaler_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio scaler annullato.")
                return
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler salvato in: {scaler_path}")
    # --- Salvataggio scaler_xg ---
    if scaler_xg_path:
        if scaler_xg_path.exists():
            overwrite = input(f"⚠️ Il file '{scaler_xg_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio scaler_xg annullato.")
                return
        joblib.dump(scaler_xg, scaler_xg_path)
        print(f"✅ Scaler XG salvato in: {scaler_xg_path}")
    # --- Salvataggio poly transformer ---
    if poly_path:
        if poly_path.exists():
            overwrite = input(f"⚠️ Il file '{poly_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio poly transformer annullato.")
                return
        joblib.dump(poly, poly_path)
        print(f"✅ Poly transformer salvato in: {poly_path}")
    # --- Salvataggio lin_poly model ---
    if lin_poly_path:
        if lin_poly_path.exists():
            overwrite = input(f"⚠️ Il file '{lin_poly_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio lin_poly model annullato.")
                return
        joblib.dump(lin_poly, lin_poly_path)
        print(f"✅ Lin_poly model salvato in: {lin_poly_path}")

    # --- Salvataggio linear model ---
    if lin_path:
        if lin_path.exists():
            overwrite = input(f"⚠️ Il file '{lin_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio lin model annullato.")
                return
        joblib.dump(lin, lin_path)
        print(f"✅ Lin model salvato in: {lin_path}")

def save_models_assist(model):

    #scaler_path = config.SCALER_DIR_ASSIST / config.SCALER if scaler is not None else None
    model_path = config.MODEL_DIR_ASSIST / config.POISS_MODEL_ASSIST if model is not None else None    

    # Crea le cartelle se non esistono
    os.makedirs(config.MODEL_DIR_ASSIST, exist_ok=True)
    #if scaler_path:
        #os.makedirs(config.SCALER_DIR_ASSIST, exist_ok=True)
    if model_path:
        os.makedirs(config.MODEL_DIR_ASSIST, exist_ok=True)

     # --- Salvataggio modello ---
    if model_path.exists():
        overwrite = input(f"⚠️ Il file '{model_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")
        else:
            joblib.dump(model, model_path)
            print(f"✅ Modello sovrascritto in: {model_path}")
    else:
        joblib.dump(model, model_path)
        print(f"✅ Modello salvato in: {model_path}")

    # --- Salvataggio scaler ---
    #if scaler_path:
        #if scaler_path.exists():
            #overwrite = input(f"⚠️ Il file '{scaler_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio scaler annullato.")
            return
        #joblib.dump(scaler, scaler_path)
        #print(f"✅ Scaler salvato in: {scaler_path}")



# trasformazione che moltiplica (usata dopo lo StandardScaler)
def multiply_by_factor(X, factor=2.0):
    return X * factor

def get_player_data(df: pd.DataFrame, player_name: str):
    """
    Cerca i dati di un giocatore nel dataframe, gestendo:
    - accenti (Martínez -> Martinez)
    - case-insensitivity
    - match esatto o parola intera
    - ambiguità se esistono più giocatori con lo stesso nome
    """

    # Normalizza i nomi
    df = df.copy()
    df["player_norm"] = df["player"].apply(lambda x: unidecode(str(x)).lower())
    player_norm = unidecode(player_name).lower()

    # 1️⃣ Match esatto
    player_df = df[df["player_norm"] == player_norm]

    # 2️⃣ Se non trovato, prova con parola intera (regex)
    if player_df.empty:
        player_df = df[df["player_norm"].str.contains(rf"\b{re.escape(player_norm)}\b", case=False, na=False)]

    # 3️⃣ Se ancora vuoto
    if player_df.empty:
        print(f"⚠️ Nessun giocatore trovato per '{player_name}'.")
        return pd.DataFrame()

    # 4️⃣ Se più giocatori hanno lo stesso nome
    matching_players = player_df["player"].unique()
    if len(matching_players) > 1:
        print(f"⚠️ Trovati più giocatori con nome simile a '{player_name}':")
        for i, p in enumerate(matching_players, 1):
            teams = ", ".join(df[df["player"] == p]["player_team"].dropna().unique())
            print(f"   {i}. {p} ({teams})")

        # Chiede all'utente quale scegliere
        try:
            choice = int(input("👉 Inserisci il numero del giocatore desiderato: ")) - 1
            chosen_player = matching_players[choice]
            player_df = df[df["player"] == chosen_player]
        except (ValueError, IndexError):
            print("❌ Scelta non valida, interrotto.")
            return pd.DataFrame()

    return player_df.sort_values("date").reset_index(drop=True)


def weighted_xg_vs_opponent(base_xG, player_df, opponent_xGA_90min):
    """
    Calcola uno xG medio del giocatore pesato per la forza dell'avversario (xGA_90min).
    """
    # forza media degli avversari affrontati nelle ultime 12 partite
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(12).mean()

    # se mancano valori, fallback alla media
    if pd.isna(base_xG) or pd.isna(avg_opponent_xGA):
        return base_xG

    # calcola fattore di correzione
    # se l’avversario concede più del normale → boost
    # se concede meno → penalità
    factor = opponent_xGA_90min / avg_opponent_xGA

    # limitiamo il fattore per non esplodere
    factor = np.clip(factor, 0.5, 1.5)

    # xG pesato
    weighted_xG = base_xG * factor
    return weighted_xG

def weighted_xA_vs_opponent(base_xA, player_df, opponent_xGA_90min):
    """
    Calcola uno xG medio del giocatore pesato per la forza dell'avversario (xGA_90min).
    """
    # forza media degli avversari affrontati nelle ultime 18 partite (quanto sta concedendo l'avversario)
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(12).mean()

    # se mancano valori, fallback alla media
    if pd.isna(base_xA) or pd.isna(avg_opponent_xGA):
        return base_xA

    # calcola fattore di correzione
    # se l’avversario concede più del normale → boost
    # se concede meno → penalità
    factor = opponent_xGA_90min / avg_opponent_xGA

    # limitiamo il fattore per non esplodere
    factor = np.clip(factor, 0.75, 1.25)

    # xG pesato
    weighted_xA = base_xA * factor
    return weighted_xA


def get_Xga_90min_opp_team(team: str, season: str, teams_df: pd.DataFrame) -> float:
    row = teams_df[(teams_df["Team"].str.lower() == team.lower()) & (teams_df["season"] == season)]
    if not row.empty:
        return row["XGA_90min"].values[0]
    else:
        return np.nan
    

def normalize_team_name(name: str) -> str:
    """Normalizza il nome della squadra per confronti più robusti."""
    name = name.lower()
    # Rimuovi prefissi e parole comuni
    name = re.sub(r'\b(fc|ac|ss|us|as|cf|sc|calcio|club|sporting)\b', '', name)
    # Rimuovi spazi e punteggiatura
    name = re.sub(r'[^a-z]', '', name)
    return name.strip()

def get_Xg_90min_team(team: str, season: str, teams_df: pd.DataFrame) -> float:
    """Restituisce lo XG_90min della squadra, gestendo variazioni nel nome."""
    team_norm = normalize_team_name(team)
    
    # Normalizza anche la colonna Team del dataset
    teams_df = teams_df.copy()
    teams_df["Team"] = teams_df["Team"].apply(normalize_team_name)
    
    row = teams_df[
        (teams_df["Team"] == team_norm) &
        (teams_df["season"] == season)
    ]
    
    if not row.empty:
        return row["XG_90min"].values[0]
    else:
        return np.nan

def get_xGA_last5_team_h_a_mean(team: str, h_a: str, teams_df: pd.DataFrame) -> float:
    """
    Restituisce l'xGA medio delle ultime 5 partite,
    con filtro opzionale per casa (h) o trasferta (a).
    """
    team_norm = normalize_team_name(team)

    df = teams_df.copy()
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    team_rows = df[df["team_name"] == team_norm].sort_values("date")

    if team_rows.empty:
        return np.nan

    # filtro casa/trasferta solo se richiesto
    if h_a in ["h", "a"]:
        team_rows = team_rows[team_rows["h_a"] == h_a]

    if team_rows.empty:
        return np.nan

    recent = team_rows.tail(5)
    return recent["xGA"].mean()

def get_xG_last5_team_h_a_mean(team: str, h_a: str, teams_df: pd.DataFrame) -> float:
    """
    Restituisce l'xG medio delle ultime 5 partite.
    Se h_a ∈ {h, a} filtra casa/trasferta.
    Altrimenti usa tutte le partite.
    """
    team_norm = normalize_team_name(team)

    df = teams_df.copy()
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    team_rows = df[df["team_name"] == team_norm]

    if team_rows.empty:
        return np.nan

    # Filtra per home/away solo se richiesto
    if h_a in ["h", "a"]:
        team_rows = team_rows[team_rows["h_a"] == h_a]

    if team_rows.empty:
        return np.nan

    recent = team_rows.tail(5)

    return recent["xG"].mean()

def get_xGA_last5_team_h_a(team: str, h_a: str, teams_df: pd.DataFrame) -> float:
    """
    Restituisce l'xGA medio delle ultime 5 partite,
    filtrando per casa/trasferta solo se h_a è valido.
    """
    team_norm = normalize_team_name(team)

    df = teams_df.copy()
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    team_rows = df[df["team_name"] == team_norm]

    if team_rows.empty:
        return np.nan

    # ───────────────────────────────────────────────
    # Se h_a è 'h' o 'a', filtra. Altrimenti usa tutto.
    # ───────────────────────────────────────────────
    if h_a in ["h", "a"]:
        team_rows = team_rows[team_rows["h_a"] == h_a]

    if team_rows.empty:
        return np.nan

    recent = team_rows.tail(5)
    return recent["xGA"].mean()


def get_def_data_last5_team_h_a(team: str, h_a: str, teams_df: pd.DataFrame) -> tuple:
    """
    Restituisce (xGA_last5, GA_last5) filtrando per h/a solo se richiesto.
    """
    team_norm = normalize_team_name(team)

    df = teams_df.copy()
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    team_rows = df[df["team_name"] == team_norm]

    if team_rows.empty:
        return np.nan, np.nan

    # ───────────────────────────────────────────────
    # Se h_a è 'h' o 'a', filtra. Altrimenti usa tutto.
    # ───────────────────────────────────────────────
    if h_a in ["h", "a"]:
        team_rows = team_rows[team_rows["h_a"] == h_a]

    if team_rows.empty:
        return np.nan, np.nan

    recent = team_rows.tail(5)

    xGA_last5 = recent["xGA"].sum()
    GA_last5  = recent["missed"].sum()

    return xGA_last5, GA_last5

def get_att_data_last5_team_h_a(team: str, h_a: str, teams_df: pd.DataFrame) -> tuple:
    """
    Restituisce (xGA_last5, GA_last5) filtrando solo le partite
    giocate in casa (h) o fuori (a).
    """
    team_norm = normalize_team_name(team)

    df = teams_df.copy()
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    team_rows = df[df["team_name"] == team_norm]

    if team_rows.empty:
        return np.nan, np.nan

    # filtro casa/trasferta
    side_rows = team_rows[team_rows["h_a"] == h_a].sort_values("date")

    if side_rows.empty:
        return np.nan, np.nan

    recent = side_rows.tail(5)

    xG_last5 = recent["xG"].sum()
    Goal_last5  = recent["scored"].sum()

    return xG_last5, Goal_last5  


    
def clean_position(pos):
    """
    Pulisce e classifica la posizione di un calciatore:
    - GK -> None
    - 3+ ruoli -> None
    - D+M -> 'DM' (terzino/mediano)
    - F+M -> 'FM' (trequartista)
    - D+F -> 'DF' (esterno)
    - Singola lettera -> 'D', 'M' o 'F'
    """
    text = str(pos).upper()
    
    # Escludi portieri
    #if "GK" in text:
       # return None
    
    # Estrai ruoli principali
    roles = re.findall(r"[DMF]", text)
    roles = list(dict.fromkeys(roles))  # rimuovi duplicati preservando ordine
    
    # Troppi ruoli => ambigua
    if len(roles) >= 3:
        return "None"
    
    # Mappa combinazioni specifiche
    if len(roles) == 2:
        combo = "".join(sorted(roles))
        if combo == "DM":
            return "DM"   # terzino o mediano
        elif combo == "FM":
            return "FM"   # trequartista
        elif combo == "DF":
            return "DF"   # esterno difensivo
        else:
            return "None"
    
    # Ruolo singolo
    if len(roles) == 1:
        return roles[0]
    
    return "None"

def get_positions(player_df, pos_dummies_index):
    # 1. Prendo l'ultima posizione nota del giocatore
    player_pos = player_df["position"].iloc[-1]

    # 2. Creo le dummies con le stesse colonne usate nel training
    player_pos = clean_position(player_df["position"].iloc[-1])
    pos_dummies = pd.get_dummies([f"pos_{player_pos}"], dtype=int)

    # Se nel training c'erano più colonne, devo riallinearle:
    pos_feature_names = pos_dummies_index  # <- quelle usate nel train
    for col in pos_feature_names:
        if col not in pos_dummies:
            pos_dummies[col] = 0

    # Riordino le colonne come nel training
    pos_dummies = pos_dummies[pos_feature_names]

    return pos_dummies

def multicoll_check(X, features):

    X = X[features].dropna()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)

def weighted_xg_by_team_strength(sum_xg,player_df,team_xg_90min, df_teams):
    """
    Calcola uno xG del giocatore pesato sulla forza offensiva della propria squadra.
    
    Parametri
    ----------
    player_df : pd.DataFrame
        Storico del giocatore, deve contenere almeno 'sum_xG', 'team' e 'season'.
    df_teams : pd.DataFrame
        Dataset con statistiche per squadra e stagione, deve contenere 'team', 'season', 'xG_90min' 
        (xG prodotti medi per 90 min dalla squadra).
    
    Ritorna
    -------
    weighted_xG : float
        xG medio del giocatore corretto per la forza offensiva della squadra.
    """
    if player_df.empty:
        return np.nan

    # ottieni la squadra e stagione correnti
    season = player_df["season"].iloc[-1]

    # forza offensiva media del campionato (baseline)
    league_avg_xG = df_teams.groupby("season")["XG_90min"].mean().loc[season]
    std_dev = df_teams[df_teams["season"] == season]["XG_90min"].std()

    z_score = (team_xg_90min - league_avg_xG) / std_dev

    factor = 1 + 0.2 * np.clip(z_score, -1, 1) 

    # applica il moltiplicatore
    weighted_xG = sum_xg * factor

    return weighted_xG

# calcolo media stagionale per giocatore
def get_shot_conversion_mean(df):

    df["n_shots"] = df["n_shots"].replace(0, np.nan)  # evitiamo div/0
    df["shot_conversion_rate"] = df["goals"] / df["n_shots"]

    conversion_mean = (
        df.groupby(["player", "season"], as_index=False)["shot_conversion_rate"]
        .mean()
        .rename(columns={"shot_conversion_rate": "shot_conversion_rate_mean"})
    )

    # uniamo al df principale
    df = df.merge(conversion_mean, on=["player", "season"], how="left")

    # eventuali NaN (es. giocatori senza tiri in una stagione) → 0
    df["shot_conversion_rate_mean"] = df["shot_conversion_rate_mean"].fillna(0)

    df = df.drop(columns=["shot_conversion_rate"])

    return df 

def get_stat_desc(df, features, target_col):
    """
    Analisi descrittiva del dataset:
    - Distribuzione del target
    - Statistiche descrittive
    - Boxplot delle features
    - Matrice di correlazione e correlazioni col target

    Parametri:
    ----------
    df : pd.DataFrame
        Dataset completo
    features : list
        Lista delle colonne di feature numeriche
    target_col : str
        Nome della colonna target (es. 'assists')
    """

    # --- Distribuzione generale del target ---
    plt.figure(figsize=(5,4))
    sns.countplot(x=df[target_col])
    plt.title(f"Distribuzione del target ({target_col})")
    plt.xlabel(f"{target_col} (0 = no, 1 = sì)")
    plt.ylabel("Conteggio")
    plt.show()

    # --- Statistiche di base ---
    print("\n📈 Statistiche descrittive delle features numeriche:")
    print(df[features + [target_col]].describe().T)

    # --- Distribuzione delle feature principali ---
    df_melted = df[features].melt(value_vars=features)
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_melted, x="variable", y="value")
    plt.title("Distribuzione delle features (Boxplot)")
    plt.xticks(rotation=45)
    plt.show()

    # --- Correlazione numerica ---
    corr = df[features + [target_col]].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Matrice di correlazione con il target '{target_col}'")
    plt.show()

    # --- Correlazione diretta con il target ---
    corr_target = corr[target_col].sort_values(ascending=False)
    print(f"\n🔥 Correlazioni con il target '{target_col}':")
    print(corr_target)


def analyze_feature_skewness(df, feature_cols, plot=True):
    """
    Analizza l'asimmetria (skewness) e la curtosi delle features numeriche.
    Suggerisce trasformazioni log/sqrt per ridurre la skewness.
    """

    results = []

    for col in feature_cols:
        data = df[col].dropna()

        sk = skew(data)
        kt = kurtosis(data)

        # Suggerimento di trasformazione
        if sk > 1:
            suggestion = "log1p"
        elif 0.5 < sk <= 1:
            suggestion = "sqrt"
        else:
            suggestion = "none"

        results.append({
            "feature": col,
            "skewness": sk,
            "kurtosis": kt,
            "suggested_transform": suggestion
        })

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(data, bins=30, ax=axes[0], kde=True)
            axes[0].set_title(f"Distribuzione originale: {col}")

            if suggestion != "none":
                if suggestion == "log1p":
                    transformed = np.log1p(data)
                else:
                    transformed = np.sqrt(data)
                sns.histplot(transformed, bins=30, ax=axes[1], kde=True)
                axes[1].set_title(f"Distribuzione {suggestion}: {col}")
            else:
                axes[1].set_visible(False)

            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results).sort_values(by="skewness", ascending=False)
    print("\n📊 Analisi asimmetria delle feature numeriche:")
    print(results_df)
    return results_df

def process_positions(df, position_col="position"):
    """
    Pulisce e trasforma la colonna 'position' in un DataFrame, applicando il one-hot encoding.

    Parametri:
    ----------
    df : pd.DataFrame
        Il DataFrame contenente la colonna delle posizioni.
    position_col : str, default="position"
        Il nome della colonna che contiene le posizioni.

    Ritorna:
    --------
    pd.DataFrame
        Il DataFrame con le posizioni trasformate in colonne one-hot encoded.
    """
    # Stampa i valori unici iniziali (debug)
    print(f"Valori unici iniziali in '{position_col}':", df[position_col].unique())

    # Pulisci la colonna delle posizioni
    df[position_col] = df[position_col].apply(clean_position)

    # Stampa i valori unici dopo la pulizia (debug)
    print(f"Valori unici dopo la pulizia in '{position_col}':", df[position_col].unique())

    # Rimuovi valori NaN
    df = df.dropna(subset=[position_col])

    # Conta le occorrenze (debug)
    counts = df[position_col].value_counts(dropna=False)
    print(f"Occorrenze delle posizioni:\n{counts}")

    # Rimuovi i valori "None"
    df = df[df[position_col] != "None"]

    # Applica il one-hot encoding
    pos_dummies = pd.get_dummies(df[position_col], prefix="pos", dtype=int)

    # Rimuovi la colonna originale
    #df = df.drop(columns=[position_col])

    # Aggiungi le colonne one-hot encoded al DataFrame
    df = pd.concat([df, pos_dummies], axis=1)

    return df

def add_other_leagues_data(player_df, player, DATASET_DATA_DIR, GOALS_DATA_FILE_ALL_LEAGUES, CURRENT_SEASON):
    # se è la prima stagione in Serie A → aggiungi dati da altri campionati
    if player_df["season"].min() == CURRENT_SEASON:
        all_leagues_df = pd.read_csv(DATASET_DATA_DIR / GOALS_DATA_FILE_ALL_LEAGUES)

        # Filtra solo i dati del giocatore
        player_all_leagues = all_leagues_df[all_leagues_df["player"].str.contains(player, case=False, na=False)]

        # Converte le date in formato datetime (per entrambi)
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
        player_all_leagues["date"] = pd.to_datetime(player_all_leagues["date"], errors="coerce")

        # Escludi match già presenti nel dataset principale
        player_all_leagues = player_all_leagues[
            ~player_all_leagues["match_id"].isin(player_df["match_id"])
        ]

        # Concatena i dati
        player_df = pd.concat(
            [player_df.reset_index(drop=True), player_all_leagues.reset_index(drop=True)],
            ignore_index=True
        )

        # 🔽 Riordina per data
        player_df = player_df.sort_values("date").reset_index(drop=True)

        if player_all_leagues.empty:
            print(f"Nessun dato aggiuntivo trovato per {player} in all_leagues_df.")
        else:
            print(f"Dati trovati per {player}: {len(player_all_leagues)} righe in all_leagues_df.")

    return player_df

def fill_missing_values_player_df(player_df: pd.DataFrame, cols_to_check: list, season_ref: int = 2025) -> pd.DataFrame:
    """
    Riempie i valori NaN in player_df:
      - Tutte le colonne in cols_to_check con 0, tranne 'opponent_xGA_90min'
      - 'opponent_xGA_90min' con la media della stagione season_ref (default 2025)
        oppure, se non disponibile, con la media generale.
    """
    col_excluded = "opponent_xGA_90min"

    # 1️⃣ Fillna(0) su tutte le colonne tranne quella esclusa
    cols_fill_zero = [c for c in cols_to_check if c != col_excluded]
    player_df[cols_fill_zero] = player_df[cols_fill_zero].fillna(0)

    # 2️⃣ Calcola media per la stagione di riferimento
    mean_xga_season = (
        player_df.loc[player_df["season"] == season_ref, col_excluded]
        .mean()
    )

    # 3️⃣ Fallback: media generale se non ci sono dati per la stagione_ref
    if pd.isna(mean_xga_season):
        mean_xga_season = player_df[col_excluded].mean()

    # 4️⃣ Riempie solo quella colonna con la media calcolata
    player_df[col_excluded] = player_df[col_excluded].fillna(mean_xga_season)

    return player_df

def get_latest_cold_penalty(player_df):
    df = player_df.copy().sort_values("date")

    df["no_goal_streak"] = (
        df.groupby("player")["goals"]
        .apply(lambda g: g.eq(0).astype(int)
                .groupby(g.ne(0).cumsum()).cumsum())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    streak = float(df["no_goal_streak"].iloc[-1])

    # Parametri della logistica scalata
    low = 0.25   # minimo asintotico
    high = 1.00  # massimo esatto quando streak=0
    k = 0.45     # ripidità
    c = 6        # centro

    base_sigmoid = 1 / (1 + np.exp(k * (streak - c)))

    # Scaling 0–1 to low–high
    penalty = low + (high - low) * base_sigmoid
    return float(penalty)

def tune_catboost_regressor(X, y, cat_features, n_iter=20, random_seed=42):
    """
    Random Search per CatBoostRegressor (Poisson) con cross-validation.
    Aggiunti: leaf_estimation_backtracking e feature_weights opzionale.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    param_grid = {
        "depth": [4, 5, 6, 7, 8, 9],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "l2_leaf_reg": [2, 3, 4, 5, 6, 8, 10, 20],
        "bagging_temperature": [0, 0.25, 0.5, 0.75, 1.0],
        "iterations": [600, 800, 1000, 1200],
        "random_strength": [0, 0.5, 1.0, 1.5, 2.0],
        "min_data_in_leaf": [10, 20, 30, 50]
    }

    best_rmse = np.inf
    best_params = None

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for i in trange(n_iter, desc="Tuning CatBoostRegressor"):
        params = {
            "depth": random.choice(param_grid["depth"]),
            "learning_rate": random.choice(param_grid["learning_rate"]),
            "l2_leaf_reg": random.choice(param_grid["l2_leaf_reg"]),
            "bagging_temperature": random.choice(param_grid["bagging_temperature"]),
            "iterations": random.choice(param_grid["iterations"]),
            "random_strength": random.choice(param_grid["random_strength"]),
            "min_data_in_leaf": random.choice(param_grid["min_data_in_leaf"]),
          

            "bootstrap_type": "Bayesian",
            "loss_function": "Poisson",
            "verbose": False,
            "random_seed": random_seed,
        }

        rmses, r2s = [], []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, cat_features=cat_features)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)
            r2s.append(r2_score(y_val, preds))

        mean_rmse = np.mean(rmses)

        print(f"[{i+1}/{n_iter}] RMSE={mean_rmse:.4f} | Params={params}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    # Train final model
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X, y, cat_features=cat_features, verbose=False)

    print("\n🏆 **Migliori iperparametri trovati:**")
    print(best_params)
    print(f"📉 Best RMSE: {best_rmse:.4f}")

    return best_model, best_params



from sklearn.linear_model import LinearRegression

def compute_cold_penalty_res(
    df: pd.DataFrame,
    xg_col: str = "sum_xG",
    penalty_col: str = "cold_penalty",
    prod: bool = False,
    penalty_model: LinearRegression = None,
    cold_penalty_prod: float = None
):
    """
    Applica una penalizzazione ai valori di xG in base alla cold_penalty
    tramite regressione lineare e residui.

    Formula:
        xg_resid = xG - pred(xG | cold_penalty)
        xG_penalized = xg_resid * cold_penalty

    Parametri:
        df (pd.DataFrame): dati contenenti le colonne [xg_col, penalty_col]
        xg_col (str): nome della colonna xG
        penalty_col (str): nome della colonna penalty
        prod (bool): se True, usa un modello già addestrato (inference)
        penalty_model (LinearRegression): modello salvato da fase di training

    Ritorna:
        - df (pd.DataFrame): con colonne aggiunte ['xg_resid', 'xG_penalized']
        - reg (LinearRegression | None): modello addestrato (solo in training)
    """

    df = df.copy()

    # --- Controlli di sicurezza ---
    if xg_col not in df.columns or penalty_col not in df.columns:
        raise ValueError(f"Le colonne '{xg_col}' e/o '{penalty_col}' non sono presenti nel DataFrame.")

    if prod:
        if penalty_model is None:
            raise ValueError("In modalità 'prod', devi passare un modello di regressione già addestrato (penalty_model).")
        reg = penalty_model
        temp_df = pd.DataFrame({"cold_penalty":[cold_penalty_prod]})
        
        df["cold_penalty_res"] = df[xg_col].iloc[-1] - reg.predict(temp_df)[0]
    else:
        reg = LinearRegression()
        reg.fit(df[[penalty_col]], df[xg_col])

        # --- Calcolo residuo e penalizzazione ---
        df["cold_penalty_res"] = df[xg_col].iloc[-1] - reg.predict(df[[penalty_col]])
        #df["xG_penalized"] = df["xg_resid"] * df[penalty_col]

    if prod:
        return df  # solo dati trasformati
    else:
        return df, reg  # dati + modello per riuso in prod


def penalize_xg_with_cold_penalty(
    sum_xG: float,
    cold_penalty_value: float,
    position: str = None,
    alpha: float = 2.0,
    floor_base: float = 0.5
    ):
    """
    Penalizza gli xG in base alla 'cold_penalty' e al ruolo del giocatore.

    Formula:
        factor = floor + (1 - floor) * (cold_penalty ** alpha)
        xG_penalized = sum_xG * factor

    Dove:
      - 'cold_penalty' ∈ [0,1]: misura quanto il giocatore è "freddo" (più basso = meno gol recentemente)
      - 'floor': livello minimo di riduzione, diverso per ruolo
      - 'alpha': controlla la curvatura della penalità (1 = lineare, >1 = più aggressiva)
    
    Parametri:
        sum_xG (float): xG del giocatore
        cold_penalty_value (float): penalità calcolata (0–1)
        position (str): ruolo del giocatore ("F", "M", "D", "P", o None)
        alpha (float): intensità della penalità
        floor_base (float): valore minimo di riduzione per un attaccante

    Ritorna:
        float: xG penalizzato
    """

   # 🔹 Imposta floor in base al ruolo (inclusi ruoli ibridi)
    role_floor_map = {
        "F":  floor_base - 0.35,    # Attaccanti → penalità più forte
        "FM": floor_base + 0.15,          # Trequartisti / esterni offensivi → leggera riduzione
        "M":  floor_base + 0.3,     # Centrocampisti → più soft
        "DM": floor_base + 0.35,    # Centrocampisti difensivi → penalità più leggera
        "D":  floor_base + 0.4,     # Difensori → penalità molto leggera
        "DF": floor_base + 0.45     # Difensori puri o terzini difensivi → penalità minima
    }

    floor = role_floor_map.get(position, floor_base + 0.1)

    # 🔹 Calcola fattore di penalità (bounded tra floor e 1)
    factor = floor + (1 - floor) * (cold_penalty_value ** alpha)
    factor = np.clip(factor, floor, 1.0)

    # 🔹 Applica la penalità agli xG
    xG_penalized = sum_xG * factor

    return xG_penalized

def print_metrics(y_true, y_pred, y_proba, split=""):
    print(f"\n**************  METRICHE {split.upper()}  ***************")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"Brier Score: {brier_score_loss(y_true, y_proba):.4f}")
    print(f"Log Loss: {log_loss(y_true, y_proba):.4f}")

def adjust_sumxg_by_position(df, pos_factors):
    df = df.copy()
    df["sum_xG"] = df.apply(
        lambda row: row["sum_xG"] * pos_factors.get(row["position"], 1.0),
        axis=1
    )
    return df

import numpy as np
from sklearn.metrics import brier_score_loss

def find_best_alpha_per_role(model, X_val, y_val, role_col="position", alphas=None):
    """
    Trova il miglior α per ciascun ruolo (F, M, D) minimizzando il Brier score.
    Restituisce un dizionario: {"F": α_f, "M": α_m, "D": α_d}
    """

    if alphas is None:
        alphas = np.linspace(0.3, 1.2, 40)

    role_alphas = {}
    lambda_val = model.predict(X_val)

    roles = X_val[role_col].unique()

    for role in roles:
        mask = X_val[role_col] == role
        if mask.sum() < 10:
            continue  # ignora ruoli troppo rari

        y_role = y_val[mask]
        λ_role = lambda_val[mask]

        best_a, best_brier = None, 1e9
        for a in alphas:
            p = 1 - np.exp(-np.clip(a * λ_role, 0, None))
            b = brier_score_loss(y_role, p)
            if b < best_brier:
                best_brier = b
                best_a = a

        role_alphas[role] = best_a
        print(f"Ruolo {role}: best α = {best_a:.3f}, Brier = {best_brier:.5f}")

    return role_alphas

def get_alpha_for_role(role: str) -> float:
    """
    Restituisce il valore di alpha ottimale per un determinato ruolo.
    Se il ruolo non è presente o è None, usa il valore medio generale.
    """
    role_alphas = {
       
        "F": 0.65,     #0.659
        "FM": 0.603,    #0.551
        "M": 0.500,     #0.42
        "DM": 0.476,    #0.30
        "D": 0.3,     #0.33
        "DF": 0.300
    
    }

    # Se il ruolo non è noto, calcola un fallback come media pesata o media semplice
    default_alpha = np.mean(list(role_alphas.values()))

    # Normalizza input (maiuscolo, nessuno spazio)
    if isinstance(role, str):
        role = role.strip().upper()
    else:
        role = None

    return role_alphas.get(role, default_alpha)


def get_main_position_weighted(pos_series: pd.Series, window: int = 10, decay: float = 0.85) -> str:
    """
    Restituisce la posizione principale di un giocatore pesando maggiormente le partite più recenti.
    
    Args:
        pos_series (pd.Series): Colonna 'position' del giocatore (ordinata cronologicamente).
        window (int): Numero di partite recenti da considerare (default 10).
        decay (float): Fattore di decadimento (0 < decay ≤ 1). 
                       Più è basso → più le partite recenti contano di più (default 0.85).
    
    Returns:
        str: Posizione principale (più frequente, pesata per recency).
    """
    if pos_series.empty:
        return None

    # Prendi le ultime N posizioni (escludendo NaN)
    last_positions = pos_series.dropna().astype(str).tail(window).tolist()
    n = len(last_positions)
    if n == 0:
        return None

    # Calcola pesi decrescenti (es. [1.0, 0.85, 0.72, ...])
    weights = np.array([decay ** (n - i - 1) for i in range(n)])

    # Somma pesi per ciascuna posizione
    pos_weights = defaultdict(float)
    for pos, w in zip(last_positions, weights):
        pos_weights[pos] += w

    # Trova la posizione con peso massimo
    main_pos = max(pos_weights, key=pos_weights.get)
    return main_pos

def split_features_by_type(df: pd.DataFrame, feature_names: list):
    """
    Divide una lista di feature in numeriche e categoriche in base ai tipi del DataFrame.

    Parametri
    ----------
    df : pd.DataFrame
        Il DataFrame che contiene i dati.
    feature_names : list
        Lista delle colonne da analizzare.

    Ritorna
    -------
    numeric_features : list
        Colonne numeriche.
    categorical_features : list
        Colonne categoriche (object, string, category o bool).
    """
    numeric_features = []
    categorical_features = []

    for col in feature_names:
        if col not in df.columns:
            print(f"⚠️ Attenzione: '{col}' non trovato nel DataFrame, salto.")
            continue

        dtype = df[col].dtype

        # Numeric: int, float, np.number
        if np.issubdtype(dtype, np.number):
            numeric_features.append(col)
        # Categorical: string, object, category, boolean
        elif dtype == "object" or dtype.name == "category" or np.issubdtype(dtype, np.bool_):
            categorical_features.append(col)
        else:
            # fallback: se non riconosciuto, consideriamo categorico
            categorical_features.append(col)

    return numeric_features, categorical_features

def predict_goal_probability(model, X_goal, player, role, get_alpha_for_role_fn):
    """
    Calcola la probabilità che un giocatore segni almeno un gol,
    usando un modello Poisson (che predice λ = expected goals)
    e un fattore di calibrazione α specifico per ruolo.

    Parameters
    ----------
    model : object
        Modello Poisson o regressore (es. CatBoostRegressor).
    X_goal : pd.DataFrame
        Feature del giocatore per la predizione (una sola riga).
    player : str
        Nome del giocatore (solo per debug/log).
    role : str
        Ruolo principale del giocatore (es. 'F', 'M', 'D', 'FM', 'DM', ecc.).
    get_alpha_for_role_fn : callable
        Funzione che dato un ruolo restituisce il miglior alpha calibrato.

    Returns
    -------
    float
        Probabilità stimata di segnare almeno un gol.
    """

    # 1️⃣ Predici λ (expected goals)
    lambda_pred = model.predict(X_goal)[0]

    print(f"Ruolo principale (pesato) di {player}: {role}")

    # 2️⃣ Recupera α ottimale per il ruolo
    best_a = get_alpha_for_role_fn(role)

    # 3️⃣ Converti λ → probabilità P(goal ≥ 1)
    goal_proba = 1 - np.exp(-best_a * np.clip(lambda_pred, 0, None))

    # 4️⃣ Estrai scalare se è un array
    if isinstance(goal_proba, np.ndarray):
        goal_proba = goal_proba.item()

    return goal_proba

def predict_probabilities_poisson(
    model,
    X_new_df,
    main_role,
    alpha_fn,
    poisson_fn
):
    """
    Predice la distribuzione di probabilità dei gol usando un modello
    che stima lambda (xG atteso) e una correzione per ruolo.

    Parameters
    ----------
    model : fitted model
        Modello che predice lambda (es. regressione Poisson o simile)
    X_new_df : pd.DataFrame
        Feature del giocatore per la partita da prevedere
    main_role : str
        Ruolo principale del giocatore (es. 'FW', 'MF', ...)
    alpha_fn : callable
        Funzione che ritorna alpha dato un ruolo (es. utils.get_alpha_for_role)
    poisson_fn : callable
        Funzione che calcola le probabilità Poisson (es. utils.poisson_goal_probs)

    Returns
    -------
    dict or np.ndarray
        Distribuzione di probabilità dei gol
    """

    # 1️⃣ Predizione lambda grezzo
    lambda_pred = model.predict(X_new_df)

    # 2️⃣ Correzione per ruolo
    best_alpha = alpha_fn(main_role)
    lambda_adj = np.clip(best_alpha * lambda_pred, 0, None)

    # 3️⃣ Distribuzione Poisson
    probs = poisson_fn(lambda_adj)

    return probs


def compute_role_overperf_stats(global_df):
    """
    Calcola statistiche globali dell'intero dataset (training).
    Queste sono usate sia in training che in produzione.
    """

    # overperf_log calcolato su TUTTI i giocatori
    all_overperf = (
        np.log1p(global_df["npgoals_perMatch"]) -
        np.log1p(global_df["npxG_perMatch"] + 1e-6)
    ).clip(-1.2, 1.2)

    # Mediana globale (più robusta della media)
    global_median = float(all_overperf.median())

    # Mediana per ruolo (molto robusta)
    role_medians = (
        global_df
        .assign(overperf_log=all_overperf)
        .groupby("position")["overperf_log"]
        .median()
        .fillna(0)
        .to_dict()
    )

    # fallback per sicurezza
    default_role_median = np.median(list(role_medians.values()))

    # ----------------------------
    # 3) divisore per normalizzare gli shot
    #    robusto tramite quantili globali
    # ----------------------------
    shots_series = global_df["shots_perMatch"].fillna(0)

    Q2 = shots_series.rolling(20, min_periods=5).sum().median()
    Q3 = shots_series.rolling(20, min_periods=5).sum().quantile(0.75)

    shots_divisor = max(5, (Q2 + Q3) / 2)

    return {
        "global_overperf_median": global_median,
        "role_overperf_medians": role_medians,
        "default_role_median": default_role_median,
        "shots_divisor": shots_divisor
    }



def add_overperformance_features_old(
    df: pd.DataFrame,
    stats: dict,
    player_col: str = "player",
    prod: bool = False
):

    df = df.copy()
    df = df.sort_values([player_col, "date"])

    # ============================================================
    # 1️⃣ OVERPERFORMANCE LOG DI BASE
    # ============================================================
    df["overperf_log"] = (
        np.log1p(df["npgoals_perMatch"]) -
        np.log1p(df["npxG_perMatch"] + 1e-6)
    ).clip(-1.2, 1.2)

    # ============================================================
    # 2️⃣ FORMA RECENTE (5 PARTITE)
    # ============================================================
    goals5 = (
        df.groupby(player_col)["npgoals_perMatch"]
          .rolling(5, min_periods=1).sum()
          .reset_index(level=0, drop=True)
    )
    xg5 = (
        df.groupby(player_col)["npxG_perMatch"]
          .rolling(5, min_periods=1).sum()
          .reset_index(level=0, drop=True)
    )

    # anti-leak
    if not prod:
        goals5 = goals5.shift()
        xg5 = xg5.shift()

    df["overperf_last5"] = (
        np.log1p(goals5) - np.log1p(xg5 + 1e-6)
    ).clip(-1.0, 1.0)

    # ============================================================
    # 3️⃣ PESO BASATO SUI TIRI (ULTIME 20)
    # ============================================================
    shots20 = (
        df.groupby(player_col)["shots_perMatch"]
          .rolling(20, min_periods=1).sum()
          .reset_index(level=0, drop=True)
    )

    if not prod:
        shots20 = shots20.shift()

    df["shots_last20"] = shots20.fillna(0)

    # normalizzazione usando divisore globale
    div = stats["shots_divisor"]

    df["weight"] = 0.2 + 0.65 * (1 - np.exp(-df["shots_last20"] / div))
    df["weight"] = df["weight"].clip(0.2, 1.0)


    # 4️⃣ BLEND STORICO ↔ MEDIANA DEL RUOLO
    role_medians = stats["role_overperf_medians"]
    default_median = stats["default_role_median"]

    #print("Unique POS:", df["position"].unique())
    #print("Role medians keys:", list(role_medians.keys()))

     #clean position column
    df["position"] = df["position"].apply(clean_position)
    #ora prendo, usando la chiave che è la posizione del giocatore, la mediana corrispondente
    df["gmedian_role"] = df["position"].map(role_medians).fillna(default_median)

    df["overperf_blend"] = (
        df["weight"] * df["overperf_log"] +
        (1 - df["weight"]) * df["gmedian_role"]
    ).clip(-1.0, 1.0)

    # ============================================================
    # 5️⃣ COMBINAZIONE: 27% storico + 73% forma recente
    # ============================================================
    df["overperf_combined"] = (
        0.37 * df["overperf_blend"] +
        0.63 * df["overperf_last5"]
    ).clip(-1.5, 1.5)


    df["overperf_role_resid"] = (
        df["overperf_combined"] - df["gmedian_role"]
    ).clip(-1.5, 1.5)

    df.drop(columns=["gmedian_role"], inplace=True)
    return df

def add_overperformance_features(
    df: pd.DataFrame,
    stats: dict,
    player_col: str = "player",
    prod: bool = False
):
    df = df.copy()

    # ORDINAMENTO + RESET INDEX (FONDAMENTALE)
    df = df.sort_values([player_col, "date"]).reset_index(drop=True)

    # ============================================================
    # 1️⃣ OVERPERFORMANCE LOG BASE
    # ============================================================
    df["overperf_log"] = (
        np.log1p(df["npgoals_perMatch"]) -
        np.log1p(df["npxG_perMatch"].clip(lower=0) + 1e-6)
    ).clip(-1.2, 1.2)

    # ============================================================
    # 2️⃣ FORMA RECENTE (ULTIME 5 PARTITE) — SAFE
    # ============================================================
    goals5 = (
        df.groupby(player_col)["npgoals_perMatch"]
          .transform(lambda s: s.rolling(5, min_periods=1).sum())
    )

    xg5 = (
        df.groupby(player_col)["npxG_perMatch"]
          .transform(lambda s: s.rolling(5, min_periods=1).sum())
    )

    if not prod:
        goals5 = goals5.groupby(df[player_col]).shift()
        xg5 = xg5.groupby(df[player_col]).shift()

    df["overperf_last5"] = (
        np.log1p(goals5.fillna(0)) -
        np.log1p(xg5.fillna(0) + 1e-6)
    ).clip(-1.0, 1.0)

    # ============================================================
    # 3️⃣ PESO BASATO SUI TIRI (ULTIME 20) — SAFE
    # ============================================================
    shots20 = (
        df.groupby(player_col)["shots_perMatch"]
          .transform(lambda s: s.rolling(20, min_periods=1).sum())
    )

    if not prod:
        shots20 = shots20.groupby(df[player_col]).shift()

    shots20 = shots20.fillna(0)

    div = stats["shots_divisor"]

    df["weight"] = (
        0.2 + 0.65 * (1 - np.exp(-shots20 / div))
    ).clip(0.2, 1.0)

    # ============================================================
    # 4️⃣ MEDIANA DI RUOLO
    # ============================================================
    role_medians = stats["role_overperf_medians"]
    default_median = stats["default_role_median"]

    df["position"] = df["position"].apply(clean_position)
    df["gmedian_role"] = (
        df["position"].map(role_medians).fillna(default_median)
    )

    df["overperf_blend"] = (
        df["weight"] * df["overperf_log"] +
        (1 - df["weight"]) * df["gmedian_role"]
    ).clip(-1.0, 1.0)

    # ============================================================
    # 5️⃣ COMBINAZIONE FINALE
    # ============================================================
    df["overperf_combined"] = (
        0.37 * df["overperf_blend"] +
        0.63 * df["overperf_last5"]
    ).clip(-1.5, 1.5)

    df["overperf_role_resid"] = (
        df["overperf_combined"] - df["gmedian_role"]
    ).clip(-1.5, 1.5)

    df.drop(columns=["gmedian_role"], inplace=True)

    return df

def calibrate_by_role(df_val: pd.DataFrame, lambda_val: np.ndarray, y_val: np.ndarray, role_col: str = "position"):
    """
    Calibra α (shrink factor) e isotonic regression per ciascun ruolo.
    Restituisce:
      - dict con best α per ruolo
      - dict con modelli di calibrazione isotonic per ruolo
      - dataframe riassuntivo delle performance per ruolo
    """

    alphas = np.linspace(0.3, 1.0, 40)
    role_results = []
    alpha_map = {}
    iso_models = {}

    roles = df_val[role_col].unique()

    for role in roles:
        if pd.isna(role):
            continue

        mask = df_val[role_col] == role
        lam_r = lambda_val[mask]
        y_r = y_val[mask]

        # Trova best α minimizzando il Brier score
        best_a, best_brier = None, 1e9
        for a in alphas:
            p = 1 - np.exp(-np.clip(a * lam_r, 0, None))
            b = brier_score_loss(y_r, p)
            if b < best_brier:
                best_brier = b
                best_a = a

        # Calibrazione isotonic (su prob. grezze con best α)
        raw_p = 1 - np.exp(-np.clip(best_a * lam_r, 0, None))
        raw_p = np.clip(raw_p, 0.01, 0.85)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_p, y_r)

        # Salva risultati
        alpha_map[role] = best_a
        iso_models[role] = iso
        role_results.append({
            "role": role,
            "best_alpha": round(best_a, 3),
            "brier_pre_iso": round(best_brier, 5),
        })

    df_results = pd.DataFrame(role_results).sort_values("best_alpha", ascending=False)

    print("📊 RISULTATI CALIBRAZIONE PER RUOLO:")
    print(df_results)
    return alpha_map, iso_models, df_results

def adjust_prob_final(
    prob_base: float,
    overperf_value: float,
    finishing_resid: float,
    role: str,
    strength_overperf: float = 0.5,
    strength_finishing: float = 0.3
):
    """
    Calibra la probabilità di gol usando:
      - overperformance stabile
      - residui di finishing
      - fattori per ruolo
      - correzione in log-odds (molto stabile)

    Parametri:
        prob_base: float
        overperf_value: float
        finishing_resid: float
        role: str
        strength_overperf: float
        strength_finishing: float
    
    Return:
        probabilità finale aggiustata (float)
    """

    # -----------------------------------------------------------
    # 0️⃣ Protezione contro valori estremi
    # -----------------------------------------------------------
    #prob = np.clip(prob_base, 1e-6, 1 - 1e-6)

    # -----------------------------------------------------------
    # 1️⃣ Converti in log-odds
    # -----------------------------------------------------------
    logodds = np.log(prob_base / (1 - prob_base))

    # -----------------------------------------------------------
    # 2️⃣ Scaling dei segnali (tanh)
    # -----------------------------------------------------------
    overperf_scaled = np.tanh(overperf_value)
    finish_scaled   = np.tanh(finishing_resid)

    # -----------------------------------------------------------
    # 3️⃣ Pesi dinamici per ruolo
    # -----------------------------------------------------------
    role_factor = {
        "F":  0.7,
        "FM": 0.8,
        "M":  1.1,
        "DM": 1.3,
        "D":  1.3,
        "DF": 1.3,
        None: 1.0
    }.get(role, 1.0)

    # -----------------------------------------------------------
    # 4️⃣ Shift nei log-odds
    # -----------------------------------------------------------
    logodds_adj = (
          logodds
        + role_factor * strength_overperf  * overperf_scaled
        + role_factor * strength_finishing * finish_scaled
    )

    # -----------------------------------------------------------
    # 5️⃣ Torna a probabilità
    # -----------------------------------------------------------
    prob_final = 1 / (1 + np.exp(-logodds_adj))

    return float(np.clip(prob_final, 0, 1))


def reduce_penalty_xg(df, penalty_weight=0.5):
    """
    Riduce il peso degli xG su rigore mantenendo consistenza metrica.
    penalty_weight:
        1.0 = usa tutto il rigore (default xG)
        0.3 = usa il 30% del rigore
        0.0 = ignora completamente il rigore
    """

    df = df.copy()
    df["sum_xG"] = df["sum_xG"].fillna(0)
    df["npxG_perMatch"] = df["npxG_perMatch"].fillna(0)

    # xG su rigore (vero, non log)
    df["penalty_xG"] = df["sum_xG"] - df["npxG_perMatch"]

    # Riduzione
    df["penalty_xG_reduced"] = df["penalty_xG"] * penalty_weight

    # xG finale
    df["sum_xG"] = df["npxG_perMatch"] + df["penalty_xG_reduced"]

    return df

def add_finishing_efficiency_hist(df, window=20, prod=False):
        """
        Calcola una metrica storica di efficienza di finalizzazione ('finishing_efficiency_hist')
        per ciascun giocatore sulle ultime `window` partite.

        Formula:
            finishing_eff = (rolling_goals / rolling_xG) * weight(shots)

        Dove:
        - rolling_* sono somme mobili sulle ultime `window` partite (shiftate per escludere la partita corrente)
    

        Parametri:
            df (pd.DataFrame): dataframe contenente almeno ['player', 'date', 'goals', 'sum_xG', 'shots']
            window (int): numero di partite considerate nella media mobile
            prod (bool): se True non applica shift (usa anche gli ultimi valori disponibili)

        Ritorna:
            pd.DataFrame: con nuova colonna 'finishing_efficiency_hist'
        """
        df = df.sort_values(["player", "date"]).copy()
        eps = 1e-5

        # Calcolo cumulativo goals/xG
        df["finishing_efficiency"] = df["npgoals_perMatch"] / (df["npxG_perMatch"] + eps)

        # EMA per ogni giocatore
        if prod:
            # NO SHIFT in produzione → usa anche gli ultimi valori
            df["finishing_efficiency_hist"] = (
                df.groupby("player")["finishing_efficiency"]
                .apply(lambda x: x.ewm(span=window, min_periods=3).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            # VERSIONE TRAINING → SHIFT per evitare leakage
            df["finishing_efficiency_hist"] = (
                df.groupby("player")["finishing_efficiency"]
                .apply(lambda x: x.shift().ewm(span=window, min_periods=3).mean())
                .reset_index(level=0, drop=True)
            )

        # Clipping per outlier
        max_clip = df["finishing_efficiency_hist"].quantile(0.99)
        df["finishing_efficiency_hist"] = df["finishing_efficiency_hist"].clip(0, max_clip)

        # Fill iniziali
        df["finishing_efficiency_hist"] = df["finishing_efficiency_hist"].fillna(
            df["finishing_efficiency_hist"].median()
        )

        return df


def weight_efficiency_shots_old(df, prod=False):
    """
        Aggiunge una colonna 'finishing_eff_weighted' che combina
        l'efficienza di finalizzazione con l'esperienza (numero totale di tiri storici).

        Formula:
            finishing_eff_weighted = finishing_efficiency_hist * weight(shots_hist)

        Dove:
        - shots_hist è il cumulativo di tiri fino alla partita precedente
        - weight(shots_hist) è una funzione logaritmica che cresce lentamente con i tiri

        Parametri:
            df (pd.DataFrame): dataframe con colonne 'player', 'shots', 'finishing_efficiency_hist'
            prod (bool): se True non applica shift sulla storia dei tiri

        Ritorna:
            pd.DataFrame: con colonne 'shots_hist' e 'finishing_eff_weighted'
    """

    df = df.copy()

    if prod:
        # NO SHIFT → usa anche l’ultimo match
        df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum()
    else:
        # VERSIONE TRAINING → SHIFT per evitare leakage
        df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum().shift(1)

    df["shots_hist"] = df["shots_hist"].fillna(0)

    # Peso logaritmico più realistico
    weight = np.log1p(df["shots_hist"]) / np.log1p(20)
    weight = np.clip(weight, 0, 1)

    df["finishing_eff_weighted"] = df["finishing_efficiency_hist"] * weight
    return df

def weight_efficiency_shots(df, prod=False):
    df = df.copy()

    # Calcolo dei tiri storici
    if prod:
        df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum()
    else:
        df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum().shift(1)

    df["shots_hist"] = df["shots_hist"].fillna(0)

    # Sigmoid weighting
    k = 0.02        # slope
    midpoint = 80   # saturazione intorno ai 70 tiri
    weight = 1 / (1 + np.exp(-k * (df["shots_hist"] - midpoint)))

    df["finishing_eff_weighted"] = df["finishing_efficiency_hist"] * weight

    return df

    
def combine_sumxg_efficiency(df, use_rank=False):
        """
        Combina la pericolosità (xG generato) e l'efficienza (finishing_eff_weighted)
        in un unico indice 'finishing_form'.

        Due opzioni di normalizzazione:
            - use_rank=True → usa rank percentuali (0-1), robusti a outlier ma perdono scala metrica
            - use_rank=False → usa z-score (StandardScaler), più informativi per modelli lineari

        Formula:
            finishing_form = 0.5 * norm(sum_xG) + 0.5 * norm(finishing_eff_weighted)

        Parametri:
            df (pd.DataFrame)
            use_rank (bool): se True usa rank percentuali, altrimenti z-score

        Ritorna:
            pd.DataFrame: con nuova colonna 'finishing_form'
        """
        df = df.copy()

        if use_rank:
            # Versione rank percentuale
            df["finishing_form"] = (
                0.5 * df["sum_xG"].rank(pct=True) +
                0.5 * df["finishing_eff_weighted"].rank(pct=True)
            )
        else:
            # Versione z-score (mantiene informazione metrica)
            scaler = StandardScaler()
            z_sumxg = scaler.fit_transform(df[["sum_xG"]])
            z_eff = scaler.fit_transform(df[["finishing_eff_weighted"]])
            df["finishing_form"] = 0.5 * z_sumxg.flatten() + 0.5 * z_eff.flatten()

        return df

# ---------------------------------------------------------------

def compute_finishing_form(df, window=20, use_rank=True, prod=False):
    """
        Esegue in sequenza:
        1) add_finishing_efficiency_hist
        2) weight_efficiency_shots
        3) combine_sumxg_efficiency

        Se prod=True:
            - NON applica shift nei calcoli storici (usa tutti i dati disponibili).
    """

    merged_df = df.copy()

    # Calcolo finishing efficiency
    merged_df = add_finishing_efficiency_hist(
            merged_df, window=window, prod=prod
    )

    # Calcolo finishing_eff_weighted
    merged_df = weight_efficiency_shots(
            merged_df, prod=prod
    )

    # Calcolo finishing_form
    merged_df = combine_sumxg_efficiency(
            merged_df, use_rank=use_rank
    )

    return merged_df

def adjust_xg_by_minutes(sum_xg, minutes_last5, decay=0.75):
    """
    Aggiusta sum_xG in base ai minuti giocati nelle ultime partite.

    Migliorie:
        - Le partite più recenti pesano di più (decadimento esponenziale)
        - Penalizza chi gioca molto meno della media
        - Non aumenta MAI sum_xG
        - Curva concava + clipping di sicurezza
    """

    # ===============================
    # 1️⃣ Estrazione minuti (weighted)
    # ===============================
    if isinstance(minutes_last5, pd.Series):
        if minutes_last5.empty:
            return sum_xg

        mins = minutes_last5.dropna().values
        if len(mins) == 0:
            return sum_xg

        # Pesi esponenziali (ultima partita pesa di più)
        weights = np.array([decay ** i for i in range(len(mins)-1, -1, -1)])
        weights = weights / weights.sum()

        minutes_val = np.sum(mins * weights)

    else:
        try:
            minutes_val = float(minutes_last5)
        except:
            return sum_xg

    if minutes_val <= 0 or np.isnan(minutes_val):
        return sum_xg

    # ===============================
    # 2️⃣ Penalizzazione concava
    # ===============================
    mean_minutes_overall = 68.0
    ratio = minutes_val / mean_minutes_overall

    # Curva concava (robusta)
    weight = ratio ** 0.6

    # Mai boost
    weight = min(weight, 1.0)

    # Clipping inferiore
    weight = max(weight, 0.20)

    return sum_xg * weight

def compute_shot_quality_index(df, window=20, player_col="player", prod=False):
    """
    Calcola un indice di qualità tiro normalizzato 0–1 SENZA SCALER.
    
    - prod=False → training (usa shift per evitare leakage)
    - prod=True  → produzione (non usa shift, usa tutto lo storico)
    """

    df = df.copy()
    eps = 1e-6

    # --------------------------------------------
    # 1) Efficienza di tiro logaritmica
    # --------------------------------------------
    df["shot_eff_log"] = np.log1p(df["npgoals_perMatch"]) - np.log1p(df["npxG_perMatch"] + eps)

    # --------------------------------------------
    # 2) Difficoltà del tiro (premia gol difficili)
    # --------------------------------------------
    df["shot_difficulty"] = df["npgoals_perMatch"] * (1 - df["npxG_perMatch"].clip(0, 1))

    # --------------------------------------------
    # 3) Indice grezzo
    # --------------------------------------------
    df["shot_quality_raw"] = (
        0.2 * df["shot_eff_log"] +
        0.8 * df["shot_difficulty"]
    )

    # --------------------------------------------
    # 4) Rolling window di stabilizzazione
    # --------------------------------------------
    df = df.sort_values([player_col, "date"])

    if prod:
        # 🚀 Produzione → usa anche la riga corrente
        df["shot_quality_roll"] = (
            df.groupby(player_col)["shot_quality_raw"]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    else:
        # 🎓 Training → shift per evitare leakage
        df["shot_quality_shifted"] = df.groupby(player_col)["shot_quality_raw"].shift(1)

        df["shot_quality_roll"] = (
            df.groupby(player_col)["shot_quality_shifted"]
              .rolling(window=window, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )

        # fallback iniziale
        df["shot_quality_roll"] = df["shot_quality_roll"].fillna(df["shot_quality_shifted"])

    # --------------------------------------------
    # 5) Normalizzazione 0–1 senza scaler
    #    usando quantile clipping robusto (evita outlier)
    # --------------------------------------------

    # quantili robusti
    q01 = df["shot_quality_roll"].quantile(0.01)
    q99 = df["shot_quality_roll"].quantile(0.99)

    # protezione
    if q99 - q01 < 1e-6:
        df["shot_quality_index"] = 0.5
        return df

    # normalizzazione
    df["shot_quality_index"] = (df["shot_quality_roll"] - q01) / (q99 - q01)

    # clipping finale
    df["shot_quality_index"] = df["shot_quality_index"].clip(0, 1)

    return df

def compute_shot_quality_index_per_shot(
    df,
    player_col="player",
    xg_col="sum_xG",
    goals_col="npgoals_perMatch",
    shots_col="shots_perMatch",
    date_col="date",
    window=20,
    prod=False
):
    """
    Shot Quality Index 0–1 (semplificato)
    - qualità PER TIRO
    - rolling mean
    - shrink su campione
    - anti-leakage
    """

    df = df.copy()
    eps = 1e-6

    df = df.sort_values([player_col, date_col])

    # ---------------------------
    # 1) Residuo per tiro
    # ---------------------------
    df["xg_per_shot"] = df[xg_col] / (df[shots_col] + eps)
    df["goals_per_shot"] = df[goals_col] / (df[shots_col] + eps)

    df["shot_quality_raw"] = (
        df["goals_per_shot"] - df["xg_per_shot"]
    ).clip(-1, 1)

    # ---------------------------
    # 2) Rolling mean (anti-leak)
    # ---------------------------
    grouped = df.groupby(player_col)["shot_quality_raw"]

    if prod:
        roll = grouped.rolling(window, min_periods=3).mean()
    else:
        roll = grouped.shift(1).rolling(window, min_periods=3).mean()

    df["shot_quality_roll"] = roll.reset_index(level=0, drop=True)
    df["shot_quality_roll"] = df["shot_quality_roll"].fillna(df["shot_quality_raw"])

    # ---------------------------
    # 3) Shrink semplice su volume
    # ---------------------------
    if prod:
        shot_count = df.groupby(player_col)[shots_col].rolling(window).sum()
    else:
        shot_count = df.groupby(player_col)[shots_col].shift(1).rolling(window).sum()

    shot_count = shot_count.reset_index(level=0, drop=True).fillna(0)

    # peso 0–1: pochi tiri → poco peso
    weight = (shot_count / (shot_count + 10)).clip(0, 1)

    global_median = df["shot_quality_roll"].median()

    df["shot_quality_shrunk"] = (
        weight * df["shot_quality_roll"] +
        (1 - weight) * global_median
    )

    # ---------------------------
    # 4) Normalizzazione robusta
    # ---------------------------
    q_lo = df["shot_quality_shrunk"].quantile(0.05)
    q_hi = df["shot_quality_shrunk"].quantile(0.95)

    if q_hi - q_lo < 1e-6:
        df["shot_quality_index"] = 0.5
    else:
        df["shot_quality_index"] = (
            (df["shot_quality_shrunk"] - q_lo) / (q_hi - q_lo)
        ).clip(0, 1)

    return df


def compute_shot_quality_index_v2(df, window=30, player_col="player", prod=False):
    """
    Shot Quality Index 0–1 indipendente dal volume di tiri.

    - Usa residui di conversione (goals - xG)
    - Premia gol con xG bassi
    - Smoothing tramite rolling mean
    - Quantile normalization 0–1 senza scaler
    - prod=True → NO SHIFT (usa tutto lo storico)
    """

    df = df.copy()
    eps = 1e-6

    # ----------------------------------------------------
    # 1) Residuo di conversione
    # ----------------------------------------------------
    df["conv_residual"] = df["npgoals_perMatch"] - df["npxG_perMatch"]

    # normalizzazione logaritmica robusta
    df["conv_residual_log"] = np.sign(df["conv_residual"]) * np.log1p(np.abs(df["conv_residual"]))

    # ----------------------------------------------------
    # 2) Difficoltà del tiro (gol da xG bassi)
    # ----------------------------------------------------
    df["shot_difficulty"] = df["npgoals_perMatch"] * (1 - df["npxG_perMatch"].clip(0, 1))

    # ----------------------------------------------------
    # 3) Shot quality grezzo
    # ----------------------------------------------------
    df["shot_quality_raw"] = (
        0.7 * df["conv_residual_log"] +
        0.3 * df["shot_difficulty"]
    )

    # ----------------------------------------------------
    # 4) Rolling smoothing per giocatore
    # ----------------------------------------------------
    df = df.sort_values([player_col, "date"])

    if prod:
        # produzione → usa la riga corrente
        df["sq_roll"] = (
            df.groupby(player_col)["shot_quality_raw"]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    else:
        # training → shift per evitare leakage
        df["sq_shifted"] = df.groupby(player_col)["shot_quality_raw"].shift(1)

        df["sq_roll"] = (
            df.groupby(player_col)["sq_shifted"]
              .rolling(window=window, min_periods=3)
              .mean()
              .reset_index(level=0, drop=True)
        )

        df["sq_roll"] = df["sq_roll"].fillna(df["sq_shifted"])

    # ----------------------------------------------------
    # 5) Normalizzazione robusta 0–1 (senza scaler)
    # ----------------------------------------------------
    q01 = df["sq_roll"].quantile(0.01)
    q99 = df["sq_roll"].quantile(0.99)

    if q99 - q01 < 1e-6:
        df["shot_quality_index"] = 0.5
        return df

    df["shot_quality_index"] = (df["sq_roll"] - q01) / (q99 - q01)
    df["shot_quality_index"] = df["shot_quality_index"].clip(0, 1)

    return df

def compute_defensive_overperf_stats(df, team_col="team_name", ga_col="missed", xga_col="xGA", window=5):
    """
    Calcola la metrica di overperformance difensiva per ogni squadra:
    
        xGA_missed_last5 = xGA_last5 - GA_last5

    e la trasforma in un fattore correttivo 0.8–1.2 tramite funzione logistica,
    utile per pesare la qualità dei tiri o il sum_xG dell’avversario.

    Parametri:
        df        : DataFrame contenente almeno [team, GA, xGA]
        team_col  : colonna con nome squadra
        ga_col    : gol subiti
        xga_col   : expected goals against
        window    : rolling window (default 5)

    Output:
        df con nuove colonne:
            - xGA_last5
            - GA_last5
            - xGA_missed_last5
            - defensive_adjust_factor (0.8 – 1.2)
    """

    df = df.copy()

    # 1) Calcolo rolling last5 per ogni squadra
    df = df.sort_values([team_col, "date"])  # Assicurazione ordine temporale

    df["xGA_last5"] = (
        df.groupby(team_col)[xga_col]
          .rolling(window=window, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    df["GA_last5"] = (
        df.groupby(team_col)[ga_col]
          .rolling(window=window, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    return df


def adjust_sumxg_by_defensive_factor(sum_xg, def_factor):
    """
    Applica il fattore difensivo alla probabilità gol del giocatore.
    """
    return sum_xg * def_factor


def weighted_xg_vs_opponent_mixed(
        base_xG,
        player_df,
        opponent_xGA_90min,
        opponent_xGA_last5,
        opponent_GA_last5,
        w_xga=0.4,
        w_overperf=0.6
    ):
    """
    Pesa il base_xG del giocatore con:
        1) forza difensiva attesa (xGA/90min)
        2) performance reale difesa (GA90_last5 / xGA90_last5)
    """

    # ------------------------------
    # 1) Fattore atteso (xGA/90)
    # ------------------------------
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(12).mean()

    if pd.isna(base_xG) or pd.isna(avg_opponent_xGA):
        return base_xG

    factor_xGA = opponent_xGA_90min / avg_opponent_xGA
    factor_xGA = np.clip(factor_xGA, 0.5, 1.5)

    # ------------------------------
    # 2) Fattore reale (GA_last5 / xGA_last5)
    #    → tutto riportato su base 90 minuti
    # ------------------------------
    MINUTES_5_MATCHES = 5 * 90

    if opponent_xGA_last5 <= 0:
        factor_overperf = 1.0
    else:
        xGA90_last5 = opponent_xGA_last5 / MINUTES_5_MATCHES * 90
        GA90_last5 = opponent_GA_last5 / MINUTES_5_MATCHES * 90

        # rapporto coerente
        ratio = GA90_last5 / xGA90_last5
        factor_overperf = np.clip(ratio, 0.5, 1.5)

    # ------------------------------
    # 3) Mix finale
    # ------------------------------
    final_factor = (
        w_xga * factor_xGA +
        w_overperf * factor_overperf
    )
    final_factor = np.clip(final_factor, 0.6, 1.4)

    return base_xG * final_factor

import numpy as np

def weighted_xg_team_mixed(
    sum_xG_player,
    df_teams,
    team_xG_90_min_last5,
    xG_last5_team,
    goals_last5_team,
    max_boost=0.25  # ±25% massimo
):
    """
    Applica un bonus/malus moderato all'xG del giocatore
    in base al contesto offensivo della squadra.
    """

    # =========================
    # 1️⃣ Media campionato
    # =========================
    league_avg_xG = df_teams[df_teams["season"]==config.CURRENT_SEASON]["XG_90min"].mean()

    # =========================
    # 2️⃣ Quanto la squadra produce (vs media)
    # =========================
    prod_ratio = team_xG_90_min_last5 / league_avg_xG
    prod_ratio = np.clip(prod_ratio, 0.85, 1.15)

    # =========================
    # 3️⃣ Quanto la squadra converte (gol vs xG)
    # =========================
    finishing_ratio = goals_last5_team / max(xG_last5_team, 0.01)
    finishing_ratio = np.clip(finishing_ratio, 0.85, 1.15)

    # =========================
    # 4️⃣ Peso ridotto (non aggressivo)
    # =========================
    team_factor = (
        0.6 * prod_ratio +
        0.4 * finishing_ratio
    )

    # =========================
    # 5️⃣ Normalizzazione finale
    # =========================
    team_factor = np.clip(team_factor, 1 - max_boost, 1 + max_boost)

    # =========================
    # 6️⃣ Applica al giocatore
    # =========================
    sum_xG_adjusted = sum_xG_player * team_factor

    return sum_xG_adjusted


def tune_feature_weights(
    X, y, cat_features, 
    n_iter=20, 
    random_seed=42
):
    """
    Ottimizza i feature_weights per CatBoostRegressor usando Random Search + CV.
    Ritorna:
        - best_weights
        - migliori metriche
        - modello addestrato con i pesi migliori
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    weight_space = {
        "sum_xG": (0.6, 2.0),                # deve rimanere dominante
        "shot_quality_index": (0.5, 2.5),
        "finishing_form_resid": (0.2, 1.5),
        "overperf_combined": (0.05, 1.0),
        "position_weighted": (0.2, 1.5)
    }

    best_rmse = np.inf
    best_w = None

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for i in range(n_iter):
        # Genera pesi casuali nel range scelto
        w = {k: np.random.uniform(*rng) for k, rng in weight_space.items()}

        rmses = []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostRegressor(
                depth=6,
                iterations=800,
                learning_rate=0.03,
                min_data_in_leaf=5,
                bagging_temperature=0.7,
                l2_leaf_reg=10,
                loss_function="Poisson",
                bootstrap_type="Bayesian",
                random_seed=random_seed,
                verbose=False,
                feature_weights=w
            )

            model.fit(X_train, y_train, cat_features=cat_features)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)

        mean_rmse = np.mean(rmses)

        print(f"[{i+1}/{n_iter}] RMSE={mean_rmse:.4f} | weights={w}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_w = w

    # Addestra modello finale
    final_model = CatBoostRegressor(
                depth=6,
                iterations=800,
                learning_rate=0.03,
                min_data_in_leaf=5,
                bagging_temperature=0.7,
                l2_leaf_reg=10,
                loss_function="Poisson",
                bootstrap_type="Bayesian",
                random_seed=random_seed,
                verbose=False,
                feature_weights=best_w
            )

    final_model.fit(X, y, cat_features=cat_features)

    print("\n🏆 MIGLIORI PESI TROVATI:")
    for k,v in best_w.items():
        print(f"  {k}: {v:.3f}")
    print(f"📉 RMSE migliore: {best_rmse:.4f}")

    return final_model, best_w, best_rmse

def count_matchdays(teams_df: pd.DataFrame) -> int:
    """
    Restituisce quante giornate sono state giocate nella stagione corrente,
    contando quante partite ha disputato la prima squadra trovata nel df.
    """

    # Prende la prima squadra presente nel df
    first_team = teams_df["team_name"].iloc[0]

    # Filtra tutte le sue partite (home o away)
    team_matches = teams_df[teams_df["team_name"] == first_team]

    # Il numero di partite è il numero di giornate
    return len(team_matches)

def get_h_a_opponent(h_a_player):
    if h_a_player == "h":
        h_a = "a"
    elif h_a_player == "a":
        h_a = "h"
    else:
        h_a = ""
    return h_a


def progressive_weighted_mean(values, alpha=0.15):
    """
    Calcola una media pesata progressiva (EWMA) su una serie di valori.
    Le partite più recenti pesano di più.

    Args:
        values (list/array): valori tipo xG/xA ordinati cronologicamente.
        alpha (float): fattore di decadimento. 
                       Più alto = più peso al recente (0.1–0.3 consigliato).
    Returns:
        float: media pesata progressiva.
    """
    if len(values) == 0:
        return 0.0
    
    # EWMA: Exponentially Weighted Moving Average
    weighted = 0.0
    weight_sum = 0.0
    weight = 1.0
    
    for v in reversed(values):  # parte dalla più recente
        weighted += weight * v
        weight_sum += weight
        weight *= (1 - alpha)   # peso decresce progressivamente
    
    return weighted / weight_sum if weight_sum > 0 else 0.0

def poisson_goal_probs(lam):
    """
    Calcola le probabilità Poisson per 0, 1, 2, 3+ gol e almeno 1 gol dato λ.
    Restituisce un dizionario con chiavi: lambda, p0, p1, p2, p3plus, p_any.
    """
    from scipy.stats import poisson
    # Estrai scalare se necessario
    def scalar(x): return float(x.item()) if isinstance(x, np.ndarray) else float(x)
    p0 = poisson.pmf(0, lam)
    p1 = poisson.pmf(1, lam)
    p2 = poisson.pmf(2, lam)
    p3plus = 1 - (p0 + p1 + p2)
    p_any = 1 - p0
    return {
        "lambda": scalar(lam),
        "p0": scalar(p0),
        "p1": scalar(p1),
        "p2": scalar(p2),
        "p3plus": scalar(p3plus),
        "p_any": scalar(p_any)
    }

