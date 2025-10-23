import streamlit as st
import pandas as pd
import numpy as np
import joblib
import config
import utils
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# =====================================================
# 🔹 Caricamento modelli e scaler
# =====================================================
@st.cache_resource
def load_models():
    return {
        "scaler_sumxg": joblib.load(config.SCALER_DIR / config.SCALER_XG),
        "poly": joblib.load(config.MODEL_DIR / config.POLY_TRANSFORMER),
        "lin_poly": joblib.load(config.MODEL_DIR / config.LIN_POLY),
        "scaler_features": joblib.load(config.SCALER_DIR / config.SCALER),
        "clf": joblib.load(config.MODEL_DIR / config.CALIB_LOGISTIC_REG),
    }

# =====================================================
# 🔹 Prepara feature giocatore
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
    opponent_xGA_90min = utils.get_Xga_90min_opp_team(opponent, season, df_teams)

    # 2. pesa l'xg in base all'avversario
    sum_xG_new = utils.weighted_xg_vs_opponent(df, opponent_xGA_90min)
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

# =====================================================
# 🔹 Interfaccia Streamlit
# =====================================================
def main():

    # ==========================
    # ARGOMENTI DA LINEA DI COMANDO
    # ==========================
    parser = argparse.ArgumentParser(description="FantaModel")
    parser.add_argument("--gol", action="store_true", help="Scraping e Prepocessing per il modello dei gol")

    st.set_page_config(page_title="Goal Predictor ⚽", page_icon="⚽", layout="centered")
    st.title("🎯 Goal Probability Predictor")
    st.markdown("Inserisci o seleziona i dati del giocatore per stimare la probabilità che segni nella prossima partita.")

    # Carica dataset e modelli
    models = load_models()
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

    # --- Dropdown dinamici
    players = sorted(df_orig_goal["player"].dropna().unique().tolist())
    teams= sorted(df_teams[df_teams['season'] == config.CURRENT_SEASON]['Team'].dropna().unique().tolist())
    opponents = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            player = st.selectbox("👤 Giocatore", options=[""] + players)
        with col2:
            team = st.selectbox("🏟️ Squadra", options=[""] + teams)
        opponent = st.selectbox("⚔️ Avversario", options=[""] + opponents)

        submitted = st.form_submit_button("Prevedi")

        if submitted:
            if not player or not team or not opponent:
                st.warning("⚠️ Seleziona tutti i campi prima di procedere.")
            else:
                X_scaled = prepare_player_features(player, team, opponent, df_orig_goal, df_teams, models)
                if X_scaled is not None:
                    proba = models["clf"].predict_proba(X_scaled)[0, 1]
                    st.success(f"📈 **Probabilità di goal per {player} ({team} vs {opponent})**: {proba*100:.2f}%")
                    st.progress(float(proba))
                    
                    st.markdown("---")
                    st.markdown("🧠 Basato su xG, forma recente e qualità finalizzazione")

# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
