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
import unicodedata
from collections import defaultdict
from scipy.stats import skew, kurtosis
import streamlit as st
from datetime import datetime
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from catboost import CatBoostRegressor
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

def load_xg_model():
    return {
        "catboost_regressor_xg": joblib.load(config.MODEL_DIR_XG / config.CAT_MODEL_XG)
    }
def load_fv_model():
    return {
        "fantavoto_model": joblib.load(config.MODEL_DIR_FV / config.FV_MODEL)
    }
def load_fv_model_gk():
    return {
        "fantavoto_model_gk": joblib.load(config.MODEL_DIR_FV / config.FV_MODEL_GK)
    }
def load_voto_model(ruolo):
    return {
        "voto_model": joblib.load(config.MODEL_DIR_FV / (ruolo + "_" + config.VOTO_MODEL))
    }
def get_latest_team(df_orig, player_name, team_col):
    """ Fun per prendere squadra per cui un giocatore ha giocato utima partita"""

    df = df_orig.copy()

    if df.empty or team_col not in df.columns:
        return None
    
    # 1️⃣ Filtra storico giocatore
    df = df[df["player"].str.contains(player_name, case=False, na=False)].sort_values("date")
    if df.empty:
        print(f"⚠️ Nessun dato disponibile per {player_name}")
        return None

    # prendi ultimo valore non nullo
    latest_team = (
        df[team_col]
        .dropna()
        .iloc[-1] if not df[team_col].dropna().empty else None
    )

    return latest_team

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

    df["position"] = df["position"].apply(clean_position)

    #sostituisco none con position piu frequente
    most_freq = df.loc[df["position"] != "None", "position"].mode()
    if not most_freq.empty:
        most_freq_value = most_freq.iloc[0]
        df.loc[df["position"] == "None", "position"] = most_freq_value

    numeric_features, categorical_features = split_features_by_type(df, features_names)

    df = fill_missing_values_player_df(df, numeric_features, season_ref=season)

    # 3️⃣ Riempi i NaN
    df[features_names] = df[features_names].fillna(0)

    main_role = get_main_position_weighted(df["position"], window=10, decay=0.8)

    # 4️⃣ Recupera dati della squadra e avversario

    num_giornate = count_matchdays(df_teams_curr)

        #se ho un numero sufficiente di giornate, applico discriminante home/away
    if num_giornate >= 15: 
            h_a = get_h_a_opponent(h_a_player)

            # ==========================
            # 🔹 OPPONENT
            # ==========================

            # Split casa/trasferta
            opponent_xGA_split = get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
            xGA_split_opp, GA_split_opp = get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

            # Overall (forma pura)
            opponent_xGA_overall = get_xGA_last5_team_h_a_mean(opponent, "", df_teams_curr)
            xGA_overall_opp, GA_overall_opp = get_def_data_last5_team_h_a(opponent, "", df_teams_curr)

            # Media pesata 70 / 30
            opponent_xGA_last5_per90 = (
                0.7 * opponent_xGA_overall +
                0.3 * opponent_xGA_split
            )

            xGA_last5_opp = 0.7 * xGA_overall_opp + 0.3 * xGA_split_opp
            GA_last5_opp  = 0.7 * GA_overall_opp  + 0.3 * GA_split_opp

            # ==========================
            # 🔹 PLAYER TEAM
            # ==========================

            # Split casa/trasferta
            team_xG_split = get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
            xG_split_team, Goal_split_team = get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)

            # Overall
            team_xG_overall = get_xG_last5_team_h_a_mean(team, "", df_teams_curr)
            xG_overall_team, Goal_overall_team = get_att_data_last5_team_h_a(team, "", df_teams_curr)

            # Media pesata 70 / 30
            team_xG_90_min_last5 = (
                0.7 * team_xG_overall +
                0.3 * team_xG_split
            )

            xG_last5_team     = 0.7 * xG_overall_team   + 0.3 * xG_split_team
            Goal_last5_team   = 0.7 * Goal_overall_team + 0.3 * Goal_split_team
    else:
            #OPPONENT TEAM DATA
            opponent_xGA_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
            xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent,"", df_teams)

            #PLAYER TEAM DATA
            team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, "", df_teams)
            xG_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, "", df_teams)
        
    # 5️⃣ Calcola statistiche base del giocatore. Media delle ultime 12, che poi viene pesata esponenzialmente
    sum_xA = df["sum_xA"].tail(12).to_list()

    sum_xA_weighted = progressive_weighted_mean(sum_xA, alpha=0.2)

    sum_xA_weighted = weighted_xg_vs_opponent_mixed(sum_xA_weighted, df, opponent_xGA_last5_per90, xGA_last5_opp, GA_last5_opp)

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
        poisson_fn=poisson_goal_probs,
        target="assist"
        )

    return probs["p_any"]

def get_goal_prob(model_xg, model, features_names, player, team, opponent, df_orig, df_teams, df_teams_curr_season, lin_model, ROLE_STATS,h_a_player):
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
    df = add_goal_scoring_features(df, player_col="player", prod=True)
    df = reduce_penalty_xg(df)

    df_teams_curr = compute_defensive_overperf_stats(df_teams_curr_season, team_col="team_name", ga_col="missed", xga_col="xGA", window=5)
    
    df["position"] = df["position"].apply(clean_position)
    # Sostituisco "None" con il valore più frequente (escluso None) nella colonna position
    most_freq = df.loc[df["position"] != "None", "position"].mode()
    if not most_freq.empty:
        most_freq_value = most_freq.iloc[0]
        df.loc[df["position"] == "None", "position"] = most_freq_value

    df = fill_missing_values_player_df(df, numeric_features, season_ref=config.CURRENT_SEASON)

    df[features_names] = df[features_names].fillna(0)

    # Calcolo residuo  per finishing_form
    df["xg_mean_12"] = df.groupby("player")["sum_xG"].transform(lambda x: progressive_weighted_rolling(x, alpha=0.2)).fillna(0)

    df = compute_finishing_form(df, window=12, use_rank=True, prod=True)

    if df.empty:
        print(f"⚠️ Nessun dato valido dopo preprocessing per {player}")
        return None

    # 2️⃣ Calcolo residuo lineare della finishing_form
    pred_lin = lin_model.predict(df[["xg_mean_12"]])
    df["finishing_form_resid"] = df["finishing_form"] - pred_lin

    #numero giornate già giocate nella stagione corrente
    num_giornate = count_matchdays(df_teams_curr)

    #se ho un numero sufficiente di giornate, applico discriminante home/away
    if num_giornate >= 15: 
        h_a = get_h_a_opponent(h_a_player)

        # ==========================
        # 🔹 OPPONENT
        # ==========================

        # Split casa/trasferta
        opponent_xGA_split = get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
        xGA_split_opp, GA_split_opp = get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

        # Overall (forma pura)
        opponent_xGA_overall = get_xGA_last5_team_h_a_mean(opponent, "", df_teams_curr)
        xGA_overall_opp, GA_overall_opp = get_def_data_last5_team_h_a(opponent, "", df_teams_curr)

        # Media pesata 70 / 30
        opponent_xGA_last5_per90 = (
            0.7 * opponent_xGA_overall +
            0.3 * opponent_xGA_split
        )

        xGA_last5_opp = 0.7 * xGA_overall_opp + 0.3 * xGA_split_opp
        GA_last5_opp  = 0.7 * GA_overall_opp  + 0.3 * GA_split_opp

        # ==========================
        # 🔹 PLAYER TEAM
        # ==========================

        # Split casa/trasferta
        team_xG_split = get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
        xG_split_team, Goal_split_team = get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)

        # Overall
        team_xG_overall = get_xG_last5_team_h_a_mean(team, "", df_teams_curr)
        xG_overall_team, Goal_overall_team = get_att_data_last5_team_h_a(team, "", df_teams_curr)

        # Media pesata 70 / 30
        team_xG_90_min_last5 = (
            0.7 * team_xG_overall +
            0.3 * team_xG_split
        )

        xG_last5_team     = 0.7 * xG_overall_team   + 0.3 * xG_split_team
        Goal_last5_team   = 0.7 * Goal_overall_team + 0.3 * Goal_split_team
    else:
        #OPPONENT TEAM DATA
        opponent_xGA_last5_per90 = get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
        xGA_last5_opp, GA_last5_opp = get_def_data_last5_team_h_a(opponent,"", df_teams)

        #PLAYER TEAM DATA
        team_xG_90_min_last5 = get_xG_last5_team_h_a_mean(team, "", df_teams)
        xG_last5_team, Goal_last5_team = get_att_data_last5_team_h_a(team, "", df_teams)
    
    if player == "nico paz" or player == "odgaard":
        main_role = "FM"
        
    main_role = get_main_position_weighted(df["position"], window=10, decay=0.8)
    opponent_strength = map_strength(opponent)
    #prendo xg base player
    sum_xG_new = predict_xg_next_match(model_xg, df, main_role, opponent_strength)

    # 5️⃣ Calcolo sum_xG corretto in base all’avversario e alla produzione offensiva della squadra
    sum_xG_new = weighted_xg_vs_opponent_mixed(sum_xG_new, df, opponent_xGA_last5_per90, xGA_last5_opp, GA_last5_opp)

    sum_xG_new = weighted_xg_team_mixed(sum_xG_new, df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team) 

    sum_xG_new = adjust_xg_by_minutes(sum_xG_new,df["minutes_played"].rolling(window=5, min_periods=1).mean())

    opponent = normalize_team(opponent)
    opponent_strength = map_strength(opponent) 

    xg_adj_pct = compute_player_vs_strength_xg_adjustment(
            df,
            opponent_strength
    )

    sum_xG_new = sum_xG_new * (1 + xg_adj_pct)
    
    # 6️⃣ Ultimo residuo disponibile
    finishing_form_resid = df["finishing_form_resid"].iloc[-1]
    overperf_value = df["overperf_role_resid"].iloc[-1]
    shot_quality_index = df["shot_quality_index"].iloc[-1]
    goal_signal = df["goal_signal"].iloc[-1]

    # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xG": sum_xG_new,
        "overperf_role_resid": overperf_value,  
        "shot_quality_index": shot_quality_index,
        "finishing_form_resid": finishing_form_resid,
        "goal_signal": goal_signal
    }])

     # 8️⃣ Applica boost
     #for feature, factor in config.BOOST_FACTORS_XGB.items():
        #X_new_df[feature] = X_new_df[feature] * factor

    player_pos = df[categorical_features]

    # Aggiungi le dummy di posizione
    X_new_df = pd.concat([X_new_df.reset_index(drop=True), player_pos.tail(1).reset_index(drop=True)], axis=1)

    probs = predict_probabilities_poisson(
        model=model,
        X_new_df=X_new_df,
        main_role=main_role,
        poisson_fn=poisson_goal_probs,
        target="goal"
    )

    return probs["p_any"]

def preprocess_data(df: pd.DataFrame):
    """
    Seleziona le feature e il target.
    Rimuove le righe con NaN nelle colonne usate.
    """
    #df = add_team_strength_column(df, 'opponent_team', 'opponent_team_strength')
    #df = add_team_strength_column(df, 'player_team', 'player_team_strength')

    features = [
        'voto_gds',
        'goals',
        'assists',
        'ammonizioni',
        'position_clean'
    ]

    target = 'fantavoto'

    #rimuovo tutti i Senza voto
    df = df[df['voto_gds'].notna()]

    #per ora teniamo solo le colonne con season 2025
    df = df[df['season'] == config.CURRENT_SEASON]

    # Applica la pulizia della posizione
    if 'position' in df.columns:
        df['position_clean'] = df['position'].apply(clean_position)
        # Esempio: media fantavoto per posizione pulita
        print("\nMedia fantavoto per posizione pulita:")
        print(df.groupby('position_clean')['fantavoto'].mean().sort_values(ascending=False))

    
        # Tieni solo le colonne necessarie
    df_model = df[features + [target]].dropna()

      # Aggiungi il DataFrame delle squadre se disponibile

    X = df_model[features]
    y = df_model[target]

    return X, y

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

def normalize_fn(name):
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    #Sostituisco - con ""
    name = name.replace("-", "")
    special_map = {
                'ø':'o','æ':'ae','œ':'oe','ß':'ss','þ':'th',
                'č':'c','ć':'c','š':'s','ž':'z','đ':'d','ğ':'g',
                'ł':'l','ń':'n','ř':'r','ě':'e','ť':'t','ď':'d',
                'á':'a','à':'a','ä':'a','â':'a','é':'e','è':'e','ë':'e','ê':'e',
                'í':'i','ì':'i','ï':'i','î':'i','ó':'o','ò':'o','ö':'o','ô':'o',
                'ú':'u','ù':'u','ü':'u','û':'u','ñ':'n'
    }
    for k,v in special_map.items():
        name = name.replace(k,v)
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_team(name):
    if pd.isna(name):
        return None
    # Remove accents and normalize
    name = unidecode(str(name)).strip().lower()
    return name

def map_strength(team):
    if team in config.TOP_TEAMS:
        return 'top'
    elif team in config.MID_TEAMS:
        return 'mid'
    else:
        return 'weak'

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
    
def normalize_match_string(s):
    """
    Normalizza una stringa di match (es. 'Milan - Parma' -> 'milan - parma'),
    rimuovendo accenti, apostrofi e portando tutto in minuscolo.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", " ").replace("’", " ")
    s = " ".join(s.split())  # rimuove spazi multipli
    return s

def normalize_team_name(name: str) -> str:
    """Normalizza il nome della squadra per confronti più robusti."""
    if name is None or pd.isna(name) or name.strip() == "":
        return ""
    
    name = name.lower()
    # Rimuovi prefissi e parole comuni
    name = re.sub(r'\b(fc|ac|ss|us|as|cf|sc|calcio|club|sporting|hellas)\b', '', name)
                
    # Rimuovi spazi e punteggiatura NON virgola, che serve quando i giocatori cambiano squadra
    name = re.sub(r'[^a-z,]', '', name)
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

    # ───────────────────────────────────────────────
    # Se h_a è 'h' o 'a', filtra. Altrimenti usa tutto.
    # ───────────────────────────────────────────────
    if h_a in ["h", "a"]:
        # filtro casa/trasferta
        team_rows = team_rows[team_rows["h_a"] == h_a].sort_values("date")

    if team_rows.empty:
        return np.nan, np.nan

    recent = team_rows.tail(5)

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

     # Gestione SUB
    if "SUB" in text:
        return "SUB"
    
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

def prepare_voto_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara il dataframe voti:
    - filtra stagione corrente
    - converte date
    - rimuove senza voto
    - pulisce la posizione
    """
    df = df_raw.copy()

    df = df[df['season'] == config.CURRENT_SEASON]
    df['date'] = pd.to_datetime(df['date'])

    # rimuovo senza voto
    df = df[df['voto_gds'].notna()]

    # pulizia posizione
    df['position_clean'] = df['position'].apply(clean_position)

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

def tune_catboost_regressor(X, y, cat_features, n_iter=15, random_seed=42):
    """
    Random Search per CatBoostRegressor (Poisson) con cross-validation.
    Aggiunti: leaf_estimation_backtracking e feature_weights opzionale.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    param_grid = {
        "depth": [ 7, 8, 9, 10],
        "learning_rate": [ 0.01, 0.02, 0.03, 0.05],
        "l2_leaf_reg": [ 4, 5, 6, 8, 10, 20],
        "bagging_temperature": [ 0.5, 0.75, 1.0],
        "iterations": [800, 1000, 1200],
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
            "loss_function": "RMSE",
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

def get_alpha_for_role(role: str, target) -> float:
    """
    Restituisce il valore di alpha ottimale per un determinato ruolo.
    Se il ruolo non è presente o è None, usa il valore medio generale.
    """
    if target == "goal":
        role_alphas = {
        
            "F": 0.6,     #0.60
            "FM": 0.53,    #0.6
            "M": 0.4,     #0.45
            "DM": 0.45,    #0.45
            "D": 0.35,     #0.35
            "DF": 0.300
            #{'D': np.float64(0.5204081632653061), 
            # 'FM': np.float64(0.6122448979591837),
            #  'F': np.float64(0.7224489795918367),
            #  'M': np.float64(0.5387755102040817),
            #  'None': np.float64(0.6489795918367347),
            #  'DM': np.float64(0.42857142857142855),
            #  'DF': np.float64(0.8877551020408163)}
        }
    elif target == "assist":
        role_alphas = {

        'M': 0.31,
        'D': 0.25,
        'None': 0.3,
        'FM': 0.38,
        'F': 0.32,
        'DM': 0.32,
        'DF': 0.3
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
            #print(f"⚠️ Attenzione: '{col}' non trovato nel DataFrame, salto.")
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
    poisson_fn,
    target
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
    best_alpha = get_alpha_for_role(main_role, target)
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

def add_goal_scoring_features(
    df: pd.DataFrame,
    player_col: str = "player",
    prod: bool = False
):
    df = df.copy()
    df = df.sort_values([player_col, "date"]).reset_index(drop=True)

    # ============================================================
    # 1️⃣ xG LEVEL (BASELINE PRINCIPALE)
    # ============================================================
    xg_ewm = (
        df.groupby(player_col)["npxG_perMatch"]
        .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    )

    df["xg_level"] = xg_ewm.fillna(0)

    # ============================================================
    # 2️⃣ GOAL RATE (tendenza al gol reale)
    # ============================================================
    goal_rate = df["npgoals_perMatch"]

    goal_rate_ewm = (
        df.groupby(player_col)[goal_rate.name]
        .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    )

    df["goal_rate_level"] = goal_rate_ewm.fillna(0)

    # ============================================================
    # 3️⃣ VOLUME (stabilità → evita 1 gol random)
    # ============================================================
    shots_ewm = (
        df.groupby(player_col)["shots_perMatch"]
        .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    )

    df["shots_level"] = shots_ewm.fillna(0)

    # confidence → penalizza low sample
    df["confidence"] = (
        1 - np.exp(-df["shots_level"] / 2)
    ).clip(0.0, 1.0)

    # ============================================================
    # 4️⃣ ROLE ADJUSTMENT (QUI STA IL FIX PER DIMARCO)
    # ============================================================
    # goal rate medio per ruolo (baseline)
    role_goal_rate = (
        df.groupby("position")["npgoals_perMatch"]
        .transform("median")
    )

    df["goal_vs_role"] = (
        df["goal_rate_level"] - role_goal_rate
    )

    # ============================================================
    # 5️⃣ FINISHING SKILL (soft, NON dominante)
    # ============================================================
    df["finishing"] = (
        df["goal_rate_level"] - df["xg_level"]
    ).clip(-0.5, 0.5)

    # ridimensionato da confidence
    df["finishing"] *= df["confidence"]

    # ============================================================
    # 6️⃣ FINAL SIGNAL (OTTIMIZZATO PER GOAL PROB)
    # ============================================================
    df["goal_signal"] = (
        0.55 * df["xg_level"] +          # baseline forte
        0.25 * df["goal_rate_level"] +  # chi segna davvero
        0.15 * df["goal_vs_role"] +     # bonus ruolo (CRUCIALE)
        0.05 * df["finishing"]          # piccolo extra
    )

    return df

def add_overperformance_features(
    df: pd.DataFrame,
    stats: dict,
    player_col: str = "player",
    prod: bool = False
):
    df = df.copy()
    df = df.sort_values([player_col, "date"]).reset_index(drop=True)

    # ============================================================
    # 1️⃣ OVERPERFORMANCE LOG BASE
    # ============================================================
    df["overperf_log"] = (
        np.log1p(df["npgoals_perMatch"]) -
        np.log1p(df["npxG_perMatch"].clip(lower=0) + 1e-6)
    ).clip(-1.2, 1.2)

    # ============================================================
    # 2️⃣ FORMA RECENTE (ULTIME 5) — SAFE
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
    # 3️⃣ VOLUME + CONFIDENCE (FIX PRINCIPALE)
    # ============================================================
    shots20 = (
        df.groupby(player_col)["shots_perMatch"]
        .transform(lambda s: s.rolling(20, min_periods=1).sum())
    )

    if not prod:
        shots20 = shots20.groupby(df[player_col]).shift()

    shots20 = shots20.fillna(0)

    div = stats["shots_divisor"]

    # 🔥 peso più conservativo
    df["weight"] = (
        0.1 + 0.75 * (1 - np.exp(-shots20 / div))
    ).clip(0.1, 1.0)

    # 🔥 confidence → penalizza low sample
    df["confidence"] = (
        1 - np.exp(-shots20 / (div * 1.5))
    ).clip(0.0, 1.0)

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
        0.35 * df["overperf_blend"] +
        0.65 * df["overperf_last5"]
    ).clip(-1.5, 1.5)

    # 🔥 applico confidence (FIX CRUCIALE)
    df["overperf_combined"] *= df["confidence"]

    df["overperf_role_resid"] = (
        df["overperf_combined"] - df["gmedian_role"]
    ).clip(-1.5, 1.5)

    # ============================================================
    # 6️⃣ LIVELLO ASSOLUTO xG (FONDAMENTALE)
    # ============================================================
    xg_ewm = (
        df.groupby(player_col)["npxG_perMatch"]
        .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    )

    df["xg_level"] = xg_ewm.fillna(0)

    # 👉 feature finale per modello (IMPORTANTISSIMA)
    df["goal_signal"] = (
        0.7 * df["xg_level"] +
        0.3 * df["overperf_role_resid"]
    )

    df.drop(columns=["gmedian_role"], inplace=True)

    return df

def add_overperformance_features_old(
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


def reduce_penalty_xg(df, penalty_weight=0.7):
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
    window=12,
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

    df["shot_quality_index_generic"] = (df["sq_roll"] - q01) / (q99 - q01)
    df["shot_quality_index_generic"] = df["shot_quality_index_generic"].clip(0, 1)

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

def progressive_weighted_rolling(df, alpha=0.3):
    """
        Calcola media pesata progressiva delle ultime 12 partite per una serie x.
        x: pd.Series ordinata cronologicamente (vecchio → nuovo)
        alpha: peso decrescente (0 < alpha < 1)
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    result = []
        
    for i in range(n):
            weighted_sum = 0.0
            weight_total = 0.0
            weight = 1.0
            # consideriamo al massimo ultime 12 partite
            for j in range(max(0, i-11), i+1):
                weighted_sum += weight * df.iloc[j]
                weight_total += weight
                weight *= (1 - alpha)
            result.append(weighted_sum / weight_total if weight_total > 0 else 0.0)
        
    return pd.Series(result, index=df.index)

def add_ewma_features(df, span=10, prod = False):

    #calcola xG cumulativo, shots per partita e minuti giocati cumulativi
    #se prod = true non uso shift(1) per evitare di perdere l'ultimo dato disponibile,
    #SE ho solo una riga per giocatore, faccio fillna(0) per evitare NaN iniziali

    cols = ["sum_xG", "shots_perMatch", "minutes_played"]

    if prod == False:
        df = df.sort_values(["player", "date"]).copy()
            
        for col in cols:
            df[f"{col}_ewm_{span}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
            )
    else:
        df = df.sort_values(["player", "date"]).copy()
        
        for col in cols:
            df[f"{col}_ewm_{span}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
    #fillna SOLO alla fine
    ewm_cols = [f"{col}_ewm_{span}" for col in cols]
    df[ewm_cols] = df[ewm_cols].fillna(0)
    return df

def predict_xg_next_match(
    model,
    df: pd.DataFrame,
    main_role: str,
    opponent_strength: str
):
    """
    Predice l'xG per la prossima partita di un giocatore usando il modello CatBoost.
    
    Parametri:
        model: modello CatBoostRegressor addestrato
        player_features: DataFrame con le feature del giocatore per la prossima partita
        cat_features: lista delle colonne categoriche
    
    Ritorna:
        float: xG predetto per la prossima partita
    """

    df = add_ewma_features(df, span=7, prod=True)

    xg_last = df["sum_xG_ewm_7"].iloc[-1]
    shots_last = df["shots_perMatch_ewm_7"].iloc[-1]
    minutes_played_last = df["minutes_played_ewm_7"].iloc[-1]

    X_modelxg = [[xg_last,                                                                                                                
                  shots_last,
                  minutes_played_last,
                  main_role,
                  opponent_strength                                               
                  ]]
    
    xg_forecast_df = pd.DataFrame(X_modelxg, columns=["sum_xG_ewm_7", "shots_perMatch_ewm_7", "minutes_played_ewm_7", "position", "opponent_team_strength"])
    xg_forecast = model.predict(xg_forecast_df)

    return float(xg_forecast[0])

def adjust_fantavoto_by_opp(fv_pred, opponent_strength, home_away):
    """
    Calibrazione finale del fantavoto
    """
    adj = 0.0

    if opponent_strength == 'top':
        adj -= 0.5
    elif opponent_strength == 'weak':
        adj += 0.5


    return round(fv_pred + adj, 2)

def compute_player_vs_strength_adjustment(
    player_df,
    target_opponent_strength,
    min_matches=5,
    neutral_value=0.0,
    ema_span=10,
    max_adjustment=0.2
):
    """
    Calcola un adjustment basato sulle performance del giocatore
    contro squadre di una certa forza, usando media esponenziale.

    Parameters
    ----------
    player_df : pd.DataFrame
        Storico partite del giocatore (ordinato temporalmente)
    target_opponent_strength : str
        'top', 'mid', 'weak'
    min_matches : int
        Numero minimo di partite per considerare valido il dato
    neutral_value : float
        Valore di fallback se pochi dati
    ema_span : int
        Span della media esponenziale (più basso = più peso al recente)
    max_adjustment : float
        Clamp massimo assoluto dell'adjustment

    Returns
    -------
    float : adjustment da applicare al fantavoto
    """

    # sicurezza
    required_cols = {'voto_gds', 'opponent_team_strength'}
    if not required_cols.issubset(player_df.columns):
        return neutral_value

    # Ultime 15 partite del giocatore
    last_matches = player_df.tail(15)

    # Media esponenziale generale
    player_mean = (
        last_matches['voto_gds']
        .ewm(span=ema_span, adjust=False)
        .mean()
        .iloc[-1]
    )

    # Filtra per forza avversaria
    subset = player_df[
        player_df['opponent_team_strength'] == target_opponent_strength
    ].tail(15)

    if len(subset) < min_matches:
        print("meno di 5 partite contro avversari di forza, ritorno 0", target_opponent_strength)
        return neutral_value

    # Media esponenziale vs quella forza
    vs_strength_mean = (
        subset['voto_gds']
        .ewm(span=ema_span, adjust=False)
        .mean()
        .iloc[-1]
    )

    # Delta performance
    adjustment = vs_strength_mean - player_mean

    # Clamp di sicurezza
    adjustment = max(min(adjustment, max_adjustment), -max_adjustment)

    return adjustment

def compute_player_home_away_adjustment(
    player_df,
    target_ha,
    min_matches=5,
    neutral_value=0.0,
    halflife=10,
    max_adjustment=0.2
):
    """
    Adjustment home/away usando media esponenziale (EWM)
    """

    required_cols = {'voto_gds', 'home_away', 'date'}
    if not required_cols.issubset(player_df.columns):
        return neutral_value

    # Ordina per data (fondamentale)
    player_df = player_df.sort_values('date')

    subset = player_df[player_df['home_away'] == target_ha]

    if len(subset) < min_matches:
        print("col home_away missing")
        return neutral_value

    # Media esponenziale globale
    player_mean = (
        player_df['voto_gds']
        .ewm(halflife=halflife, adjust=False)
        .mean()
        .iloc[-1]
    )

    # Media esponenziale home/away
    ha_mean = (
        subset['voto_gds']
        .ewm(halflife=halflife, adjust=False)
        .mean()
        .iloc[-1]
    )

    adjustment = ha_mean - player_mean

    # clamp di sicurezza
    adjustment = max(min(adjustment, max_adjustment), -max_adjustment)

    return adjustment

def add_home_away_column(df):
    """
    Aggiunge la colonna home_away ('H' / 'A')
    usando h_team, a_team e player_team
    """

    def compute_ha(row):
        if pd.isna(row['player_team']):
            return pd.NA
        if row['player_team'] == row['h_team']:
            return 'h'
        elif row['player_team'] == row['a_team']:
            return 'a'
        else:
            return pd.NA
    df["player_team"] = df["player_team"].apply(normalize_team_name)
    df['home_away'] = df.apply(compute_ha, axis=1)

    return df

import numpy as np

def exponential_mean(series, alpha=0.3):
    """
    Media esponenziale dando più peso alle partite recenti.
    """
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def baseline_attenuation(player_mean, ref=6.5):
    return 0.6 if player_mean >= ref else 1.0

def adjusted_form_delta(
    recent_mean,
    baseline_mean,
    baseline_std,
    tolerance=0.3,
    eps=0.25
):
    raw_delta = recent_mean - baseline_mean

    # zona di tolleranza
    if raw_delta > -tolerance:
        return 0.0

    if baseline_std < eps:
        z = 0.0
    else:
        z = raw_delta / baseline_std

    return z * baseline_attenuation(baseline_mean)

def compute_player_team_strength_adjustment(
    player_df,
    target_team_strength,
    min_matches=5,
    neutral_value=0.0,
    alpha=0.3,
    clamp=0.4
    ):
    """
    Adjustment basato su come il giocatore performa
    quando gioca IN una squadra di una certa forza.

    Parameters
    ----------
    player_df : pd.DataFrame
        Storico partite del giocatore
    target_team_strength : str
        'top', 'mid', 'weak'
    min_matches : int
        Numero minimo di partite richieste
    neutral_value : float
        Fallback se pochi dati
    alpha : float
        Peso esponenziale (recency)
    clamp : float
        Limite massimo adjustment

    Returns
    -------
    float : adjustment da sommare al voto_gds
    """

    required_cols = {'voto_gds', 'player_team_strength'}
    if not required_cols.issubset(player_df.columns):
        return neutral_value

    # Ordine cronologico (importantissimo)
    player_df = player_df.sort_values('date')

    # Subset per forza squadra
    subset = player_df[
        player_df['player_team_strength'] == target_team_strength
    ]

    if len(subset) < min_matches:
        return neutral_value

    # Media pesata globale
    player_mean = exponential_mean(
        player_df['voto_gds'], alpha=alpha
    )

    # Media pesata in quel contesto
    team_strength_mean = exponential_mean(
        subset['voto_gds'], alpha=alpha
    )

    adjustment = team_strength_mean - player_mean

    # Clamp di sicurezza
    adjustment = np.clip(adjustment, -clamp, clamp)

    return float(adjustment)

def compute_form_index(
    player_df,
    recent_n=5,
    season_n=15,
    clamp=1.0
):
    df = player_df.sort_values("date")

    if len(df) < recent_n + 3:
        return 0.0

    recent = df.tail(recent_n)
    previous = df.iloc[:-recent_n].tail(season_n)

    if len(previous) < 3:
        return 0.0

    components = []

    # ---- voto_gds ----
    components.append(
        adjusted_form_delta(
            recent['voto_gds'].mean(),
            previous['voto_gds'].mean(),
            previous['voto_gds'].std()
        )
    )

    # ---- involvement offensivo ----
    off_recent = recent['goals'].mean() + recent['assists'].mean()
    off_prev = previous['goals'].mean() + previous['assists'].mean()
    off_std = (previous['goals'] + previous['assists']).std()

    components.append(
        adjusted_form_delta(
            off_recent,
            off_prev,
            off_std
        )
    )

    form_index = np.mean(components)
    return float(np.clip(form_index, -clamp, clamp))

def compute_player_vs_strength_xg_adjustment(
    player_df,
    target_opponent_strength,
    min_matches=10,
    max_pct=0.25,
    neutral_value=0.0
):
    """
    Calcola un adjustment percentuale dell'xG
    in base a come il giocatore performa contro
    avversari di una certa forza.

    Parameters
    ----------
    player_df : pd.DataFrame
        Storico partite del giocatore
    target_opponent_strength : str
        'top', 'mid', 'weak'
    min_matches : int
        Minimo numero di partite richieste
    max_pct : float
        Limite massimo percentuale (+/-)
    neutral_value : float
        Fallback se pochi dati

    Returns
    -------
    float
        Adjustment percentuale da applicare all'xG
        (es. +0.15 = +15%)
    """

    required_cols = {'sum_xG', 'opponent_team_strength'}
    if not required_cols.issubset(player_df.columns):
        return neutral_value

    df = player_df.dropna(subset=['sum_xG'])

    subset = df[
        df['opponent_team_strength'] == target_opponent_strength
    ]

    if len(subset) < min_matches:
        return neutral_value

    player_mean_xg = df['sum_xG'].mean()
    vs_strength_xg = subset['sum_xG'].mean()

    if player_mean_xg <= 0:
        return neutral_value

    # delta percentuale
    adjustment_pct = (vs_strength_xg - player_mean_xg) / player_mean_xg

    # clamp di sicurezza
    adjustment_pct = max(
        min(adjustment_pct, max_pct),
        -max_pct
    )

    return adjustment_pct

def compute_base_voto(
    player_df,
    recent_n=5,
    season_n=15,
    recent_weight=0.75,
    feature_col='voto_gds',
    ewma_span_recent=10,
    ewma_span_season=10
):
    """
    Calcola il fantavoto base come media ponderata
    usando medie esponenziali (EWMA) per dare più peso
    alle partite recenti.

    Returns
    -------
    float
    """

    if feature_col not in player_df.columns or player_df.empty:
        return 0.0

    df = (
        player_df
        .dropna(subset=[feature_col])
        .sort_values('date')
    )

    if len(df) == 0:
        return 0.0

    # Usa al massimo le ultime season_n partite
    df = df.tail(season_n)

    # Se pochissimi dati → fallback semplice
    if len(df) <= recent_n:
        return float(
            df[feature_col]
            .ewm(span=min(len(df), ewma_span_recent), adjust=False)
            .mean()
            .iloc[-1]
        )

    recent = df.tail(recent_n)
    previous = df.iloc[:-recent_n]

    # ---- EWMA recenti ----
    recent_ewma = (
        recent[feature_col]
        .ewm(span=min(len(recent), ewma_span_recent), adjust=False)
        .mean()
        .iloc[-1]
    )

    # ---- EWMA precedenti ----
    if len(previous) > 0:
        prev_ewma = (
            previous[feature_col]
            .ewm(span=min(len(previous), ewma_span_season), adjust=False)
            .mean()
            .iloc[-1]
        )
    else:
        prev_ewma = recent_ewma

    prev_weight = 1 - recent_weight

    base_voto = (
        recent_ewma * recent_weight +
        prev_ewma * prev_weight
    )

    return float(base_voto)

def compute_base_voto_by_role(player_df, role):

    recent_weight = config.ROLE_WEIGHTS_VOTO.get(role, 0.75)
    return compute_base_voto(
        player_df,
        recent_weight=recent_weight
    )

def compute_weighted_feature_mean(
    player_df,
    last5_weight=0.75,
    max_matches=15,
    feature_col='goals'
):
    df = player_df.sort_values("date")

    last5 = df.tail(5)
    prev = df.tail(max_matches).head(max(0, len(df) - 5))

    #MEDIA esponenziale aumenta il peso delle partite recenti
    mean_5 = last5[feature_col].mean() if not last5.empty else 0.0
    mean_prev = prev[feature_col].mean() if not prev.empty else mean_5

    return last5_weight * mean_5 + (1 - last5_weight) * mean_prev

def compute_feature_role_impact(
    player_df,
    position,
    role_feature_stats,
    alpha=0.35,
    min_impact=0.6,
    max_impact=1.4,
    feature_col='goals'
):
    role_mean = role_feature_stats[position]["mean"]
    role_std = role_feature_stats[position]["std"]

    if role_std == 0:
        return 1.0

    player_mean = player_df[feature_col].mean()

    z = (player_mean - role_mean) / role_std
    impact = 1.0 + alpha * z

    return max(min(impact, max_impact), min_impact)


def compute_assists_role_weighted(
    player_df,
    role_assist_stats,
    last_n_recent=5,
    last_n_season=20,
    w_recent=0.7,
    w_season=0.3,
    fallback=0.0
):
    """
    Crea una feature 'assist_role_weighted' che rappresenta
    quanto il giocatore fa assist rispetto al suo ruolo.

    Returns
    -------
    float
    """
    required_cols = {"assist", "position_clean", "date"}
    if not required_cols.issubset(player_df.columns):
        return fallback

    player_df = player_df.sort_values("date")

    position = player_df["position_clean"].iloc[-1]
    if position not in role_assist_stats:
        return fallback

    # ---- last N recent ----
    recent = player_df.tail(last_n_recent)
    recent_mean = recent["assist"].mean() if not recent.empty else 0.0

    # ---- last N season (esclude recent per evitare overlap eccessivo) ----
    season = player_df.tail(last_n_season)
    season_mean = season["assist"].mean() if not season.empty else recent_mean
    
    # ---- weighted goal estimate ----
    assist_est = w_recent * recent_mean + w_season * season_mean

    # ---- role normalization ----
    role_mean = role_assist_stats[position]["mean"]
    role_std = role_assist_stats[position]["std"]

    if role_std <= 0:
        return fallback

    # z-score rispetto al ruolo
    assist_role_weighted = (assist_est - role_mean) / role_std

    # ---- clamp di sicurezza ----
    assist_role_weighted = np.clip(assist_role_weighted, -1.0, 1.0)

    return assist_role_weighted

def z_to_index_asymmetric_soft(
    z,
    role=None,
    scale_pos=1.5,
    scale_neg=0.8
):
    """
    Curva indulgente sotto media.
    """
    base = 60
    if role is not None:
        base = config.ROLE_FANTAVOTO_STATS[role]['mean'] * 10    
    else:
        base = 60

    if z < 0:
        z = z * scale_neg
    else:
        z = z * scale_pos

    index = base + 45 * (1 / (1 + np.exp(-z)) - 0.5) * 2
    return round(index, 1)

BASELINE_INDEX = 60  # neutro

def fantavoto_to_schierability_index(
    fv_pred,
    position,
    role_fv_stats
):
    if position not in role_fv_stats:
        return BASELINE_INDEX

    role_mean = role_fv_stats[position]['mean']
    role_std = role_fv_stats[position]['std']

    if role_std <= 0:
        return BASELINE_INDEX

    # z-score per ruolo
    z = (fv_pred - role_mean) / role_std

    # mappa su scala morbida (centrata su 60)
    raw_index = z_to_index_asymmetric_soft(z, position)

    # peso ruolo
    role_weight = config.ROLE_WEIGHTS_INDEX.get(position, 0.6)

    # distanza dalla neutralità
    delta = raw_index - BASELINE_INDEX

    # scaling per ruolo
    index = BASELINE_INDEX + delta * role_weight

    return round(float(np.clip(index, 30.0, 100.0)),1)

def compute_consistency_adjustment(
    player_df,
    recent_n=5,
    season_n=10,
    weight_recent=0.7,
    max_adjustment=0.1,
    neutral_value=0.0
):
    """
    Calcola un adjustment che premia la costanza di rendimento del giocatore.
    Usa la variabilità del avoto, dando più peso alle ultime partite.
    Premia SOLO se la media delle ultime 10 almeno è >= 6.
    """

    required_cols = {"fantavoto", "date", "voto_gds"}
    if not required_cols.issubset(player_df.columns):
        return neutral_value

    df = player_df.sort_values("date")

    if len(df) < recent_n + 3:
        return neutral_value

    # ---- GUARDIA CHIAVE: media minima ----
    season_mean = df["voto_gds"].tail(season_n).mean()
    if pd.isna(season_mean) or season_mean < 6.0:
        return neutral_value

    recent = df.tail(recent_n)
    previous = df.iloc[:-recent_n].tail(season_n)

    if len(previous) < 3:
        return neutral_value

    # ---- statistiche ----
    def safe_consistency(subset):
        mean = subset["voto_gds"].mean()
        std = subset["voto_gds"].std()
        if mean <= 0 or pd.isna(std):
            return 0.0
        return 1.0 / (1.0 + std / mean)

    recent_cons = safe_consistency(recent)
    prev_cons = safe_consistency(previous)

    # ---- combinazione pesata ----
    consistency_score = (
        weight_recent * recent_cons +
        (1 - weight_recent) * prev_cons
    )

    # ---- baseline neutra ----
    baseline = 0.6

    raw_adj = consistency_score - baseline

    # ---- clamp finale ----
    adjustment = np.clip(raw_adj, -max_adjustment, max_adjustment)

    return float(adjustment)

def compute_opponent_offense_bonus(
    role,
    opponent_xg_last5,
    opponent_goal_last5,
    matchday,
    df_teams_curr_season,
    league_baseline_xg=1.4,
    league_baseline_goal=1.17,
    multiplier=0.3,
    clamp_min=-0.35,
    clamp_max=0.35,
):
    """
    Calcola BONUS/MALUS offensivo se la squadra avversaria
    produce poco offensivamente (xG).

    Parameters
    ----------
    opponent_xg_last5 : float
        xG medio dell'avversario nelle ultime 5 giornate
    matchday : int
        Numero di giornate giocate
    df_teams_curr_season : pd.DataFrame
        Deve contenere la colonna 'xG_last5_mean'
    league_baseline_xg : float
        Baseline iniziale di campionato
    multiplier : float
        Peso del delta
    clamp_min, clamp_max : float
        Limiti di sicurezza (solo bonus)

    Returns
    -------
    float
        Bonus <>= 0
    """
    if role != "P":  #SE NON è PORTIERI USO XG

        # ---- sicurezza ----
        if opponent_xg_last5 is None or pd.isna(opponent_xg_last5):
            return 0.0

        # ---- STEP 1: media campionato ----
        if matchday >= 15 and 'xG_last5_mean' in df_teams_curr_season.columns:
            league_avg_xg = df_teams_curr_season['xG_last5_mean'].mean()
        else:
            league_avg_xg = league_baseline_xg

        # ---- STEP 2: confronto avversario vs campionato ----
        delta = league_avg_xg - opponent_xg_last5

        bonus = delta * multiplier
    else:
        # ---- sicurezza ----
        if opponent_goal_last5 is None or pd.isna(opponent_goal_last5):
            return 0.0
        
        # ---- STEP 1: media campionato ----
        if matchday >= 15 and 'scored' in df_teams_curr_season.columns:
            league_avg_goal = df_teams_curr_season['scored'].mean()
        else:
            league_avg_goal = league_baseline_goal
        # ---- STEP 2: confronto avversario vs campionato ----
        delta = league_avg_goal - opponent_goal_last5

        bonus = delta * multiplier

    # ---- STEP 4: clamp ----
    bonus = max(min(bonus, clamp_max), clamp_min)

    return float(bonus)

def compute_defensive_xga_bonus(
    role,
    team_xga_last5,
    team_goal_against_last5,
    matchday,
    df_teams_curr_season,
    league_baseline_xga=1.4,
    league_baseline_ga=1.17,
    multiplier=0.3,
    clamp_min=-0.35,
    clamp_max=0.35
):
    """
    Calcola bonus/malus difensivo per difensori (e GK)
    basato su xGA della squadra.

    Parameters
    ----------
    team_xga_last5 : float
        xGA medio della squadra nelle ultime 5 giornate
    matchday : int
        Numero di giornate giocate
    df_teams_curr_season : pd.DataFrame
        Deve contenere la colonna 'xGA_last5_mean'
    league_baseline_xga : float
        Baseline iniziale di campionato
    multiplier : float
        Peso del delta
    clamp_min, clamp_max : float
        Limiti di sicurezza

    Returns
    -------
    float
        Bonus (>0) o malus (<0)
    """
    if role != "P":  #SE NON è PORTIERI USO XG

        # ---- sicurezza ----
        if team_xga_last5 is None or pd.isna(team_xga_last5):
            return 0.0

        # ---- STEP 1: media campionato ----
        if matchday >= 15 and 'xGA_last5_mean' in df_teams_curr_season.columns:
            league_avg_xga = df_teams_curr_season['xGA_last5_mean'].mean()
        else:
            league_avg_xga = league_baseline_xga

        # ---- STEP 2: confronto squadra vs campionato ----
        delta = league_avg_xga - team_xga_last5
    else:
        # ---- sicurezza ----
        if team_goal_against_last5 is None or pd.isna(team_goal_against_last5):
            return 0.0
        
        # ---- STEP 1: media campionato ----
        if matchday >= 15 and 'missed' in df_teams_curr_season.columns:
            league_avg_ga = df_teams_curr_season['missed'].mean()
        else:
            league_avg_ga = league_baseline_ga

        # ---- STEP 2: confronto squadra vs campionato ----
        delta = league_avg_ga - team_goal_against_last5

    # ---- STEP 3: traduzione in bonus/malus ----
    bonus = delta * multiplier

    # ---- STEP 4: clamp ----
    bonus = max(min(bonus, clamp_max), clamp_min)

    return float(bonus)

def compute_clean_sheet(
    df_team,
    opponent_xg_last5=None,
    opponent_goal_last5=None,
    baseline_ga=1.17,
    baseline_xg=1.4,
    weight_def=0.6,
    weight_opp=0.4,
    multiplier=0.3,
    clamp_min=-0.3,
    clamp_max=0.3,
):
    """
    Bonus/malus difensivo (GK / difesa) che combina:
    - forma difensiva squadra (gol subiti)
    - forza offensiva avversaria (xG o gol)

    Returns
    -------
    dict
        {
            mean_ga_last5,
            clean_sheet_prob,
            bonus
        }
    """

    if df_team is None or df_team.empty or "missed" not in df_team.columns:
        return {
            "mean_ga_last5": None,
            "clean_sheet_prob": None,
            "bonus": 0.0
        }

    # --- ultime 5 ---
    last5 = df_team.tail(5)
    if len(last5) < 3:
        return {
            "mean_ga_last5": None,
            "clean_sheet_prob": None,
            "bonus": 0.0
        }

    # ===============================
    # 1️⃣ DIFESA SQUADRA
    # ===============================
    mean_ga = last5["missed"].mean()
    clean_sheet_prob = (last5["missed"] == 0).sum() / len(last5)

    delta_def = baseline_ga - mean_ga
    score_def = delta_def * weight_def

    # ===============================
    # 2️⃣ ATTACCO AVVERSARIO
    # ===============================
    score_opp = 0.0

    if opponent_xg_last5 is not None and not pd.isna(opponent_xg_last5):
        delta_opp = baseline_xg - opponent_xg_last5
        score_opp = delta_opp * weight_opp

    elif opponent_goal_last5 is not None and not pd.isna(opponent_goal_last5):
        delta_opp = baseline_ga - opponent_goal_last5
        score_opp = delta_opp * weight_opp

    # ===============================
    # 3️⃣ BONUS FINALE
    # ===============================
    bonus = (score_def + score_opp) * multiplier

    bonus = max(min(bonus, clamp_max), clamp_min)

    return {
        "mean_ga_last5": float(mean_ga),
        "clean_sheet_prob": float(clean_sheet_prob),
        "bonus": float(bonus),
    }

def compute_ammonizioni_adjustment(player_df, role, max_malus=0.25, span=7):
    """
    Calcola un malus leggero in base alla media esponenziale dei cartellini 
    rispetto alla media e std del ruolo.
    
    Parametri:
        player_df : pd.DataFrame -> contiene almeno la colonna 'ammonizioni'
        role : str -> ruolo del giocatore ['A','C','D','P','SUB']
        max_adjustment : float -> massimo bonus/malus da applicare
        span : int -> span per media esponenziale (ultimo N match)
        
    Ritorna:
        float -> aggiustamento da sommare al voto
    """
    if player_df.empty or role not in config.AMMONIZIONI_MEAN_FANTAROLE:
        return 0.0
    
    # consideriamo le ultime 15 partite
    df_last = player_df.tail(15)
    
    # media esponenziale dei cartellini
    amms_ewm = df_last['ammonizioni'].ewm(span=span, adjust=False).mean().iloc[-1]
    
    role_stats = config.AMMONIZIONI_MEAN_FANTAROLE[role]
    mean_role = role_stats['mean']
    std_role = role_stats['std']
    
    # differenza normalizzata rispetto al ruolo
    z_score = (amms_ewm - mean_role) / std_role if std_role > 0 else 0.0
    
    # scalare lo z-score in range [-max_adjustment, +max_adjustment]
    # se più cartellini della media -> malus, meno cartellini -> bonus
    # malus scalato e clamped

    malus = -z_score * max_malus

    #clamp: non deve diventare positivo, massimo 0
    malus = min(0.0, malus)
    malus = max(-max_malus, malus)
    return malus

def apply_fantarole_boost(index, fantarole, cap_max=99.9):
    """
    Applica un boost percentuale all'indice finale
    in base al fantarole e clippa il risultato.

    P : +10%
    D : +4%
    A,C : +5,4%
    """

    if index is None or np.isnan(index):
        return None

    boost_map = {
        "P": 0.10,
        "D": 0.06,
        "A": 0.07,
        "C": 0.06,
    }

    boost = boost_map.get(fantarole, 0.0)

    # ---- BOOST ESPLICITO ----
    increment = index * boost
    boosted_index = index + increment

    # ---- CLIP DI SICUREZZA ----
    boosted_index = np.clip(boosted_index, 0.0, cap_max)

    return float(round(boosted_index, 1))

def prepare_df_for_display(df):
    df = df.copy()

    # 🔠 Metti iniziali maiuscole in tutte le colonne stringa
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.title()

    # Index che parte da 1
    df = df.reset_index(drop=True)
    df.index = df.index + 1

    return df

def normalize_name_short(name: str):
    """
    Converte 'Rossi A.' -> ('rossi', 'a')
    """
    name = name.lower().replace(".", "").strip()
    parts = name.split()

    if len(parts) == 1:
        return parts[0], ""

    return parts[0], parts[1]


def remove_unavailable_players(
    df_pred: pd.DataFrame,
    df_unavailable: pd.DataFrame,
    suspended: tuple,
    col_pred: str = "Giocatore",
    debug: bool = True
) -> pd.DataFrame:

    PREFIXES = {'de','da','di','del','do','van','von','der','le','la','el','al','du','ze'}

    def normalize_player_name_squalificati(full_name: str) -> str:
        """
        Normalizza il nome giocatore per ottenere il cognome corretto.
        """

        name = full_name.lower().strip()
        parts = name.split()

        # 🔹 Caso 1: iniziale + cognome → inverti
        # es: "s. esposito"
        if len(parts) == 2 and parts[0].endswith('.'):
            return f"{parts[1]} {parts[0]}"

        # 🔹 Caso 2: cognome con prefisso → tienilo intero
        # es: "da cunha"
        if len(parts) >= 2 and parts[0] in config.PREFIXES:
            return " ".join(parts[:2])

        # 🔹 Caso 3: nome + cognome normale → tieni solo cognome
        # es: "nico paz"
        if len(parts) >= 2:
            return parts[-1]

        # 🔹 Caso fallback (nome singolo)
        return name

    df_pred = df_pred.copy()

    # ---------- 1️⃣ Estrai cognome + iniziale (gestione cognomi composti) ----------
    pred_info = []
    compound_particles = config.PREFIXES  # es: ["de", "di", "da", "van", "del", ...]

    for name in df_pred[col_pred]:
        parts = str(name).strip().split()

        if not parts:
            pred_info.append(("", ""))
            continue

        parts_lower = [p.lower() for p in parts]

        # ---- COGNOME COMPOSTO (Nome De Cognome) ----
        if len(parts_lower) >= 3 and parts_lower[-2] in compound_particles:
            surname = parts_lower[-2] + " " + parts_lower[-1]
            first_name = " ".join(parts_lower[:-2])
            initial = first_name[0] if first_name else ""

        # ---- CASO NORMALE ----
        elif len(parts_lower) >= 2:
            surname = parts_lower[-1]
            first_name = " ".join(parts_lower[:-1])
            initial = first_name[0] if first_name else ""

        # ---- SOLO COGNOME ----
        else:
            surname = parts_lower[0]
            initial = ""

        pred_info.append((surname, initial))

    surname_counts = pd.Series([s for s, _ in pred_info]).value_counts()

    # ---------- 2️⃣ Costruisci set indisponibili (Cognome N.) ----------
    unavailable_keys = set()

    compound_particles = config.PREFIXES

    # AGGIUNGO GLI SQUALIFICATI NORMALIZZANDO

    suspended_players_clean = [
        normalize_player_name_squalificati(player)
        for player, _ in suspended
    ]

    # crea un DataFrame solo con i nuovi giocatori
    df_suspended_players = pd.DataFrame({"Giocatore": suspended_players_clean})

    # concatena al df_unavailable esistente
    df_unavailable = pd.concat([df_unavailable, df_suspended_players], ignore_index=True)

    # opzionale: rimuovere duplicati
    #df_unavailable = df_unavailable.drop_duplicates(subset="Giocatore").reset_index(drop=True)

    for name in df_unavailable["Giocatore"]:

        parts = str(name).replace(".", "").strip().split()

        if not parts:
            continue

        parts_lower = [p.lower() for p in parts]

        # ---- CASO COGNOME COMPOSTO ----
        if len(parts_lower) >= 3 and parts_lower[0] in compound_particles:
            surname = parts_lower[0] + " " + parts_lower[1]
            initial = parts_lower[2]

        # ---- CASO NORMALE ----
        elif len(parts_lower) >= 2:
            surname = parts_lower[0]
            initial = parts_lower[1]

        # ---- CASO SOLO COGNOME ----
        else:
            surname = parts_lower[0]
            initial = ""

        # Logica rimozione
        if surname_counts.get(surname, 0) > 1:
            unavailable_keys.add((surname, initial))
        else:
            unavailable_keys.add((surname, None))

    # ---------- 3️⃣ Filtro + DEBUG ----------
    keep_mask = []
    excluded_players = []

    for (surname, initial), original_name in zip(pred_info, df_pred[col_pred]):
        
        if surname_counts.get(surname, 0) > 1:
            key = (surname.lower(), initial.lower())
            reason = "cognome duplicato → match su iniziale"
        else:
            key = (surname.lower(), None)
            reason = "cognome unico → match diretto"

        if key in unavailable_keys:
            keep_mask.append(False)
            excluded_players.append((original_name, reason))
        else:
            keep_mask.append(True)

    df_filtered = df_pred[keep_mask].copy()

    # ---------- DEBUG ----------
    if debug:
        print("\n=== DEBUG INFORTUNATI ===")
        print(f"Totale esclusi: {len(excluded_players)}\n")

        for name, reason in excluded_players:
            print(f"Escluso: {name}  |  Motivo: {reason}")

        print("=========================\n")

    return df_filtered

def get_suspended_players(
        
    path_squalificati: str
) -> tuple:
    """
    Legge il file 'squalificati' che contiene una riga tipo:
    'Nome (Team), Nome (Team), ...'
    Se player + team matchano, lo rimuove dal df.
    """

    # 🔹 Leggi file come testo puro
    with open(path_squalificati, "r", encoding="utf-8") as f:
        content = f.read()

    # 🔍 Estrae coppie (Nome, Team)
    matches = re.findall(r"([^,]+?)\s*\(([^)]+)\)", content)

    # 🔹 Normalizza elenco squalificati
    suspended = [
        (name.lower().strip(), squad.lower().strip())
        for name, squad in matches
    ]

    return suspended

def get_team_opponent_ha(player, df_voti, next_games_df):
    
    next_games_df['home'] = next_games_df['home'].apply(normalize_team_name)
    next_games_df['away'] = next_games_df['away'].apply(normalize_team_name)
    player = normalize_fn(player)

    team = df_voti.loc[df_voti['player_norm'] == player, 'player_team'].iloc[-1] if not df_voti[df_voti['player_norm'] == player].empty else None
    
    if team is None or pd.isna(team):
        team = "squadra sconosciuta"
    if "," in str(team):
        team = team.split(",")[-1]
    
    team = normalize_team_name(team)
    next_game = next_games_df[(next_games_df['home'] == team) | (next_games_df['away'] == team)]
    
    if not next_game.empty:
        row = next_game.iloc[0]
        if team in row['home']:
            h_a = 'h'
            opponent = row['away']
        else:
            h_a = 'a'
            opponent = row['home']
    else:
        h_a = ""
        opponent = ""

    return team, opponent, h_a

def calculate_inactivity_malus(date_col, reference_date=None,
                               start_weeks=2,
                               base_malus=0.05,
                               weekly_increment=0.02,
                               max_malus=0.15):
    """
    Calcola un malus basato sull'inattività del giocatore.

    - start_weeks: settimane di tolleranza senza penalità
    - base_malus: malus iniziale dopo start_weeks
    - weekly_increment: incremento settimanale del malus
    - max_malus: malus massimo
    """

    if reference_date is None:
        reference_date = datetime.now()
    else:
        reference_date = pd.to_datetime(reference_date)

    last_played = pd.to_datetime(date_col).max()

    days_since = (reference_date - last_played).days
    weeks_since = days_since // 7

    if weeks_since <= start_weeks:
        return 0.0

    extra_weeks = weeks_since - start_weeks

    malus = base_malus + (extra_weeks - 1) * weekly_increment

    return min(malus, max_malus)

def build_player_features_weighted(df: pd.DataFrame, stats, short_window=5, long_window=15, short_weight=0.7, long_weight=0.3, prod=False) -> pd.DataFrame:
    """
    Costruisce feature sintetiche per il modello di voto.

    Combina le statistiche recenti (short_window) e quelle più lunghe (long_window)
    pesandole (short_weight, long_weight) e normalizzando per 90 minuti.
    
    Richiede colonne base:
    player, giornata, shots, key_passes, xGBuildup, xGChain, time,
    team_strength, opponent_strength
    """

    df = df.sort_values(["player", "date"]).copy()

    #postprocessing categoriche team_strength e opponent_strength
    #INSERISCO mid e wak in una sola casisistica "not top"
    #top : 1, not top (mid+weak) : 0
      # default fallback
    if not prod:
        df["player_team_strength"] = df["player_team_strength"].apply(set_top_notop)
        df["opponent_team_strength"] = df["opponent_team_strength"].apply(set_top_notop)
        df["player_team_strength"] = df["player_team_strength"].map({"top": 1, "no_top": 0})  
        df["opponent_team_strength"] = df["opponent_team_strength"].map({"top": 1, "no_top": 0})  

        # contesto partita
        df["strength_diff"] = df["player_team_strength"] - df["opponent_team_strength"]

    for stat in stats:
        # media rolling short e long
        roll_short = (
            df.groupby("player")[stat]
            .rolling(short_window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        roll_long = (
            df.groupby("player")[stat]
            .rolling(long_window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        # rolling time medio per calcolo per90
        time_short = (
            df.groupby("player")["time"]
            .rolling(short_window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        time_long = (
            df.groupby("player")["time"]
            .rolling(long_window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        # combinazione pesata + per90
        df[f"{stat}_per90_weighted"] = (
            short_weight * (roll_short / time_short * 90)
            + long_weight * (roll_long / time_long * 90)
        )

        # trend sintetico: breve - lungo
        df[f"{stat}_trend_per90"] = (roll_short / time_short * 90) - (roll_long / time_long * 90)

    # minuti medi ultimi short_window per90
    df["minutes_per90_last5"] = (
        df.groupby("player")["time"]
        .rolling(short_window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    ) / 90



    # features finali: shots_per90_weighted, key_passes_per90_weighted, xGBuildup_per90_weighted, xGChain_per90_weighted,
    # trend per90, minutes_per90_last5, strength_diff
    new_features = []
    for stat in stats:
        new_features.append(f"{stat}_per90_weighted")
        new_features.append(f"{stat}_trend_per90")
    # aggiungi strength diff
    new_features.append("strength_diff")
    return df, new_features

def set_top_notop(strength):
    if strength in ["mid", "weak"]:
        return "no_top"
    elif strength == "top":
        return "top"
    else:
        return "no_top"

def compute_impact_feature(df, weights=None):
    """
    Crea una feature sintetica di impatto offensivo sulla partita per90.
    
    Parametri:
    - df: pd.DataFrame con colonne per90 come xGBuildup_per90_weighted, xGChain_per90_weighted, key_passes_per90_weighted
    - weights: dizionario con pesi delle singole componenti, default [0.5,0.3,0.2]
    
    Restituisce:
    - pd.Series con la feature 'impact_per90'
    """
    if weights is None:
        weights = {
            'xGBuildup_per90_weighted': 0.5,
            'xGChain_per90_weighted': 0.3,
            'key_passes_per90_weighted': 0.2
        }
    
    # Controllo colonne
    missing_cols = [col for col in weights.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataframe: {missing_cols}")
    
    # Calcolo feature
    impact = sum(df[col] * w for col, w in weights.items())
    
    return impact

def compute_prod_features(player_df, stats,
                          short_window=5,
                          long_window=15,
                          short_weight=0.7,
                          long_weight=0.3):
    
    #funzione per calcolare le feature di produzione recenti e stagionali pesate, 
    # da usare nel modello di voto

    features = {}

    for stat in stats:

        roll_short = player_df[stat].rolling(short_window, min_periods=1).mean()
        roll_long = player_df[stat].rolling(long_window, min_periods=1).mean()

        time_short = player_df["time"].rolling(short_window, min_periods=1).mean()
        time_long = player_df["time"].rolling(long_window, min_periods=1).mean()

        short_per90 = roll_short / time_short * 90
        long_per90 = roll_long / time_long * 90

        features[f"{stat}_per90_weighted"] = (
            short_weight * short_per90.iloc[-1]
            + long_weight * long_per90.iloc[-1]
        )

        features[f"{stat}_trend_per90"] = (
            short_per90.iloc[-1] - long_per90.iloc[-1]
        )

    # minutes form (feature molto utile per il voto)
    features["time"] = (
        short_weight * player_df["time"].tail(short_window).mean()
        + long_weight * player_df["time"].tail(long_window).mean()
    )

    return features