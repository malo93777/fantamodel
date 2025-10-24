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

    # 3️⃣ Riempi i NaN
    df[features_names] = df[features_names].fillna(0)

    # 4️⃣ Recupera dati della squadra e avversario
    season = config.CURRENT_SEASON
    opponent_xGA_90min = utils.get_Xga_90min_opp_team(opponent, season, df_teams)
    #team_xG_90min = get_Xg_90min_team(team, season, df_teams)

    # 5️⃣ Calcola statistiche base del giocatore
    sum_xA = df["sum_xA"].tail(12).mean()

    sum_xA_new = utils.weighted_xA_vs_opponent(sum_xA, df, opponent_xGA_90min)

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

    df[features_names] = df[features_names].fillna(0)

    # 2️⃣ Calcolo residuo lineare della finishing_form
    pred_lin = lin_model.predict(df[["sum_xG"]])
    df["finishing_form_resid"] = 1 * (df["finishing_form"] - pred_lin)  # boost residuo ×1

    # 3️⃣ Calcolo xG_last5 e goals_last5 (media ultime 5 partite)
    if len(df) >= 5:
        xG_last5 = df["sum_xG"].iloc[-5:].mean()
        goals_last5 = df["goals"].iloc[-5:].mean()
    else:
        xG_last5 = df["sum_xG"].mean()
        goals_last5 = df["goals"].mean()

    # 4️⃣ Ottieni opponent xGA
    season = df["season"].iloc[-1]
    opponent_xGA_90min = utils.get_Xga_90min_opp_team(opponent, season, df_teams)

    sum_xG_new = (df["sum_xG"].tail(12).mean())

    resid = df["finishing_form_resid"].iloc[-1]
    resid_pos = max(0.0, resid)
    sum_xG_new = sum_xG_new * (1.0 + 1 * resid_pos)

    # 5️⃣ Calcolo sum_xG corretto in base all’avversario
    sum_xG_new = utils.weighted_xg_vs_opponent(sum_xG_new, df, opponent_xGA_90min)

    # 6️⃣ Ultimo residuo disponibile
    finishing_form_resid = df["finishing_form_resid"].iloc[-1]

    # 7️⃣ Crea dataframe con le feature finali
    X_new_df = pd.DataFrame([{
        "sum_xG": sum_xG_new,
        "xG_last5": xG_last5,
        "opponent_xGA_90min": opponent_xGA_90min,    
        "finishing_form_resid": finishing_form_resid
    }])

    return X_new_df


# =====================================================
# 🔹 Interfaccia Streamlit
# =====================================================
def main():
    st.set_page_config(page_title="Bonus Predictor ⚽", page_icon="✨", layout="centered")
    st.title("🎯 Bonus Predictor")
    st.markdown("Prevedi la probabilità che un giocatore **segni o faccia assist** nella prossima partita.")

    # --- Carica dataset e modelli
    models = load_models()  # modelli goal
    models_assist = load_models_assist()  # modelli assist
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

    # --- Dropdown dinamici
    players = sorted(df_orig_goal["player"].dropna().unique().tolist())
    teams = sorted(df_teams[df_teams['season'] == config.CURRENT_SEASON]['Team'].dropna().unique().tolist())
    opponents = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            player = st.selectbox("👤 Giocatore", options=[""] + players)
        with col2:
            team = st.selectbox("🏟️ Squadra", options=[""] + teams)
        opponent = st.selectbox("⚔️ Avversario", options=[""] + opponents)

        submitted = st.form_submit_button("Prevedi Bonus")

    # --- Logica di predizione
    if submitted:
        if not player or not team or not opponent:
            st.warning("⚠️ Seleziona tutti i campi prima di procedere.")
        else:
            # === PREDIZIONE GOAL ===
            #tolgo finishing_form_resid perchè va ancora calcolata
            features_names_goal = list(models["xgbclass"].feature_names_in_)
            if "finishing_form_resid" in features_names_goal:
                features_names_goal.remove("finishing_form_resid")

            X_goal = prepare_features_xgb(features_names_goal, player, team, opponent, df_orig_goal, df_teams, models["lin"])
            goal_proba = None
            if X_goal is not None:
                try:
                    goal_proba = models["xgbclass"].predict_proba(X_goal)[0, 1]
                except Exception as e:
                    st.error(f"Errore nel modello goal: {e}")

            # === PREDIZIONE ASSIST ===
            features_names_assist = models_assist["log_reg_assist"].feature_names_in_
            X_assist = prepare_features_assist(features_names_assist, player, team, opponent, df_orig_assist, df_teams, models_assist["scaler_features_assist"])
            assist_proba = None
            if X_assist is not None:
                try:
                    assist_proba = models_assist["log_reg_assist"].predict_proba(X_assist)[0, 1]
                except Exception as e:
                    st.error(f"Errore nel modello assist: {e}")

            # --- Output finale
            st.markdown("---")
            st.subheader(f"📊 {player} ({team} vs {opponent})")

            if goal_proba is not None:
                st.metric("⚽ Probabilità Goal", f"{goal_proba * 100:.1f}%")
                st.progress(float(goal_proba))

            if assist_proba is not None:
                st.metric("🎯 Probabilità Assist", f"{assist_proba * 100:.1f}%")
                st.progress(float(assist_proba))

            if goal_proba is None and assist_proba is None:
                st.warning("Nessuna previsione disponibile per questo giocatore.")

            st.markdown("---")
            st.caption("🧠 Basato su xG, forma recente, efficienza di finalizzazione e forza difensiva avversaria.")

# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
