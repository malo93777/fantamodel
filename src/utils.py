import joblib
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import config
from scipy.stats import skew, kurtosis
import streamlit as st
from datetime import datetime
from sklearn.metrics import brier_score_loss

# =====================================================
# 🔹 Caricamento modelli e scaler
# =====================================================
#@st.cache_resource
def load_models():
    return {
        "scaler_sumxg": joblib.load(config.SCALER_DIR / config.SCALER_XG),
        "poly": joblib.load(config.MODEL_DIR / config.POLY_TRANSFORMER),
        "lin_poly": joblib.load(config.MODEL_DIR / config.LIN_POLY),
        "scaler_features": joblib.load(config.SCALER_DIR / config.SCALER),
        "clf": joblib.load(config.MODEL_DIR / config.CALIB_LOGISTIC_REG),
        "xgbclass":joblib.load(config.MODEL_DIR / config.XGB_MODEL),
        "lin": joblib.load(config.MODEL_DIR / config.LIN)
    }

def load_models_assist():
    return {
        "scaler_features_assist": joblib.load(config.SCALER_DIR_ASSIST / config.SCALER),
        "log_reg_assist":  joblib.load(config.MODEL_DIR_ASSIST / config.CALIB_LOGISTIC_REG)
    }

# =====================================================
# 🔹 Prepara feature giocatore per BASELINA LOG REGRESSION GOAL PRED
# =====================================================
def prepare_player_features(player, team, opponent, df_orig, df_teams, models):
    df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
    if df.empty:
        st.warning(f"Nessun dato disponibile per {player}.")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= datetime.now()].reset_index(drop=True)

    # Log-transform
    df["sum_xG"] = np.log1p(df["sum_xG"])
    clip_val = df["sum_xG"].quantile(0.99)
    df["sum_xG"] = df["sum_xG"].clip(upper=clip_val)

    # Residuo polinomiale
    sumxg_scaled = models["scaler_sumxg"].transform(df[["sum_xG"]])
    pred_poly = models["lin_poly"].predict(models["poly"].transform(sumxg_scaled))

    #boost del residuo della capacità di finalizzazione
    df["finishing_form_resid"] = 2 * (df["finishing_form"] - pred_poly)

    # Feature principali
    # Calcolo goals_last5 per la riga da prevedere
    if len(df) >= 5:
        # Prendi le ultime 5 partite, includendo l'ultima
        goals_last5 = df["goals"].iloc[-5:].mean()
    else:
        # Se ci sono meno di 5 partite, usa tutte le partite disponibili
        goals_last5 = df["goals"].mean()

    # Calcolo xG_last5 per partita da prevedere
    if len(df) >= 5:
        xG_last5 = df["sum_xG"].iloc[-5:].mean()
    else:
        xG_last5 = df["sum_xG"].mean()

    #**** Ricalcolo sum_Xg attesi dal giocatore in base alla sua forma nel medio periodo 
    # e xgAgainst per partita avversario ***
    
    # 1. Ottieni info squadre
    season = df["season"].iloc[-1]
    opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)

    # 2. pesa l'xg in base all'avversario
    sum_xG_new = weighted_xg_vs_opponent(df, opponent_xGA_90min)
    finishing_form_resid = df["finishing_form_resid"].iloc[-1]

    #df con features e valori
    X_new = [[sum_xG_new, xG_last5, goals_last5, finishing_form_resid]]
    feature_names = config.FEATURES_LR
    X_new_df = pd.DataFrame(X_new, columns=feature_names)

    # 3. Applica boost 

    boosts = config.BOOST_FACTORS
    for feature, factor in boosts.items():
        X_new_df[feature] = X_new_df[feature] * factor

    X_scaled = models["scaler_features"].transform(X_new_df[feature_names])

    return X_scaled

def prepare_features_assist(features_names, player, team, opponent, df_orig, df_teams, scaler):

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

    df = fill_missing_values_player_df(df, features_names, season_ref=season)

    # 3️⃣ Riempi i NaN
    df[features_names] = df[features_names].fillna(0)

    # 4️⃣ Recupera dati della squadra e avversario
    
    opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)
    #team_xG_90min = get_Xg_90min_team(team, season, df_teams)

    # trasformazione logaritmica
    df["sum_xA"] = np.log1p(df["sum_xA"])

    # 5️⃣ Calcola statistiche base del giocatore
    sum_xA = df["sum_xA"].tail(12).mean()

    sum_xA_new = weighted_xA_vs_opponent(sum_xA, df, opponent_xGA_90min)

    # Calcolo goals_last5 per la riga da prevedere
    if len(df) >= 5:
        # Prendi le ultime 5 partite, includendo l'ultima
        xA_last5 = df["sum_xA"].iloc[-5:].mean()
    else:
         # Se ci sono meno di 5 partite, usa tutte le partite disponibili
        xA_last5 = df["sum_xA"].mean()

        # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xA": sum_xA_new,
        "xA_last5": xA_last5    
    }])

    X_new_df = scaler.transform(X_new_df)

    return X_new_df

def prepare_features_xgb(features_names, player, team, opponent, df_orig, df_teams, lin_model):
    """
    Prepara le feature per la predizione goal probability con modello XGBoost.
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

    if "position" in df.columns:
        player_position = clean_position(df["position"].iloc[-1])
    else:
        player_position = None

    df =fill_missing_values_player_df(df, features_names, season_ref=config.CURRENT_SEASON)

    df[features_names] = df[features_names].fillna(0)

    df["sum_xG"] = np.log1p(df["sum_xG"])

    # 2️⃣ Calcolo residuo lineare della finishing_form
    pred_lin = lin_model.predict(df[["sum_xG"]])
    df["finishing_form_resid"] = 1 * (df["finishing_form"] - pred_lin) 

    # 3️⃣ Calcolo xG_last5 e goals_last5 (media ultime 5 partite)
    if len(df) >= 5:
        xG_last5 = df["sum_xG"].iloc[-5:].mean()
        goals_last5 = df["goals"].iloc[-5:].mean()
    else:
        xG_last5 = df["sum_xG"].mean()
        goals_last5 = df["goals"].mean()

    # 4️⃣ Ottieni opponent xGA
    season = df["season"].iloc[-1]
    opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)
    team_xG_90_min = get_Xg_90min_team(opponent, season, df_teams)

    sum_xG_new = (df["sum_xG"].tail(12).mean())

    resid = df["finishing_form_resid"].iloc[-1]

    sum_xG_new = sum_xG_new * (1.0 + config.BOOST_RESID * resid)  #2.0

    # 5️⃣ Calcolo sum_xG corretto in base all’avversario e alla produzione offensiva della squadra
    sum_xG_new = weighted_xg_vs_opponent(sum_xG_new, df, opponent_xGA_90min)
    sum_xG_new = weighted_xg_by_team_strength(sum_xG_new, df, team_xG_90_min, df_teams)

    # 6️⃣ Ultimo residuo disponibile
    finishing_form_resid = df["finishing_form_resid"].iloc[-1]

    # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xG": sum_xG_new,
        "xG_last5": xG_last5,
        "goals_last5": goals_last5,
        "opponent_xGA_90min": opponent_xGA_90min,    
        "finishing_form_resid": finishing_form_resid
    }])

     # 8️⃣ Applica boost
    for feature, factor in config.BOOST_FACTORS_XGB.items():
        X_new_df[feature] = X_new_df[feature] * factor

    return X_new_df

def save_models(model, scaler_xg, scaler, poly, lin_poly, lin, is_baseline=False):
    """
    Salva il modello e lo scaler, chiedendo conferma se i file esistono già.
    """

    # Percorsi completi dei file
    if is_baseline:
         model_path = config.MODEL_DIR / config.CALIB_LOGISTIC_REG if model is not None else None
    else:        
        model_path = config.MODEL_DIR / config.XGB_MODEL if model is not None else None
    
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

def save_models_assist(model, scaler, is_baseline=False):

    scaler_path = config.SCALER_DIR_ASSIST / config.SCALER if scaler is not None else None
    model_path = config.MODEL_DIR_ASSIST / config.CALIB_LOGISTIC_REG if model is not None else None    

    # Crea le cartelle se non esistono
    os.makedirs(config.MODEL_DIR_ASSIST, exist_ok=True)
    if scaler_path:
        os.makedirs(config.SCALER_DIR_ASSIST, exist_ok=True)
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
    if scaler_path:
        if scaler_path.exists():
            overwrite = input(f"⚠️ Il file '{scaler_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio scaler annullato.")
                return
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler salvato in: {scaler_path}")



# trasformazione che moltiplica (usata dopo lo StandardScaler)
def multiply_by_factor(X, factor=2.0):
    return X * factor

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
    factor = np.clip(factor, 0.75, 1.25)

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
    
def get_xG_last5_team(team: str, teams_df: pd.DataFrame) -> float:
    """
    Restituisce l'ultimo valore di xG_last5 per una squadra dal DataFrame partite.

    Args:
        team (str): nome della squadra (non necessariamente perfettamente uguale)
        teams_df (pd.DataFrame): DataFrame prodotto da build_team_dataframe()

    Returns:
        float: ultimo valore di xG_last5, oppure np.nan se non trovato
    """
    team_norm = normalize_team_name(team)
    teams_df = teams_df.copy()
    teams_df["team_name"] = teams_df["team_name"].apply(normalize_team_name)

    team_rows = teams_df[teams_df["team_name"] == team_norm].sort_values("date")

    if team_rows.empty:
        return np.nan

    # Prende le ultime 5 partite, includendo l'ultima giocata
    recent_rows = team_rows.tail(5)

    # Usa la colonna xG "grezza" e non xG_last5 per evitare feedback loop
    return recent_rows["xG"].mean()


def get_xGA_last5_team(team: str, teams_df: pd.DataFrame) -> float:
    """
    Restituisce l'ultimo valore di xGA_last5 per una squadra dal DataFrame partite.

    Args:
        team (str): nome della squadra
        teams_df (pd.DataFrame): DataFrame prodotto da build_team_dataframe()

    Returns:
        float: ultimo valore di xGA_last5, oppure np.nan se non trovato
    """
    team_norm = normalize_team_name(team)
    teams_df = teams_df.copy()
    teams_df["team_name"] = teams_df["team_name"].apply(normalize_team_name)

    team_rows = teams_df[teams_df["team_name"] == team_norm].sort_values("date")

    if team_rows.empty:
        return np.nan

    # Prende le ultime 5 partite, includendo l'ultima giocata
    recent_rows = team_rows.tail(5)

    return recent_rows["xGA"].mean()
      
def clean_position(pos):
    # Restituisce "D", "M" o "F" in base al ruolo principale, altrimenti "None"
    roles = re.findall(r"[DMF]", str(pos).upper())
    if "F" in roles:
        return "F"
    elif "M" in roles:
        return "M"
    elif "D" in roles:
        return "D"
    else:
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

import numpy as np
import pandas as pd

def get_latest_cold_penalty(player_df):
    """
    Calcola la penalità 'cold_penalty' aggiornata per un singolo giocatore,
    includendo anche l'ultima partita (quindi adatta per la predizione in prod).

    Parametri:
        player_df (pd.DataFrame): dati del giocatore con almeno la colonna 'goals'
        cold_weight (float): coefficiente di quanto velocemente cresce la penalità
        cold_center (int): numero di partite dopo cui la penalità diventa significativa

    Ritorna:
        float: valore cold_penalty più aggiornato (ultima partita)
    """
    if "goals" not in player_df.columns:
        raise ValueError("La colonna 'goals' deve essere presente in player_df")

    df = player_df.copy().sort_values("date")
    df["no_goal_streak"] = (
        df.groupby("player")["goals"]
        .apply(lambda g: g.eq(0).astype(int)
                .groupby(g.ne(0).cumsum()).cumsum())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Penalità logistica (più morbida e scalata tra ~0.2 e ~1)
    df["cold_penalty"] = 1 / (1 + np.exp(0.25 * (df["no_goal_streak"] - 8)))

    # Ritorna il valore più recente
    return float(df["cold_penalty"].iloc[-1])

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
import random

def tune_catboost(X, y, n_iter=20, random_seed=42):
    """
    Esegue una ricerca casuale di iperparametri per CatBoostClassifier
    e restituisce il miglior modello addestrato e le metriche medie.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    param_grid = {
        "depth": [3, 4, 5, 6, 7, 8, 9],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 8, 10],
        "bagging_temperature": [0, 0.25, 0.5, 0.75, 1.0],
        "iterations": [500, 800, 1000, 1200],
        "random_strength": [0, 0.5, 1.0, 1.5, 2.0],
    }
    
    best_logloss = np.inf
    best_params = None
    best_model = None
    
    for i in range(n_iter):
        params = {
            "depth": random.choice(param_grid["depth"]),
            "learning_rate": random.choice(param_grid["learning_rate"]),
            "l2_leaf_reg": random.choice(param_grid["l2_leaf_reg"]),
            "bagging_temperature": random.choice(param_grid["bagging_temperature"]),
            "iterations": random.choice(param_grid["iterations"]),
            "random_strength": random.choice(param_grid["random_strength"]),
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            "random_seed": random_seed,
        }

        model = CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        losses = []
        f1s = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]
            pred_labels = (preds > 0.5).astype(int)
            
            losses.append(log_loss(y_val, preds))
            f1s.append(f1_score(y_val, pred_labels))
        
        mean_loss = np.mean(losses)
        mean_f1 = np.mean(f1s)

        print(f"[{i+1}/{n_iter}] LogLoss={mean_loss:.4f} | F1={mean_f1:.4f} | Params={params}")

        if mean_loss < best_logloss:
            best_logloss = mean_loss
            best_params = params
            best_model = model

    print("\n🏆 Miglior combinazione trovata:")
    print(best_params)
    print(f"📉 Best LogLoss: {best_logloss:.4f}")
    
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

    # 🔹 Imposta floor in base al ruolo
    role_floor_map = {
        "F": floor_base + 0.1,          # Attaccanti → penalità più forte
        "M": floor_base + 0.3,    # Centrocampisti → un po’ più soft
        "D": floor_base + 0.4,    # Difensori → penalità molto leggera     
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