import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import utils
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
# Aggiunge la cartella "fantamodel" al percorso dei moduli
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================
# 🔹 Interfaccia Streamlit
# =====================================================
def main():
    st.set_page_config(page_title="Bonus Predictor ⚽", page_icon="✨", layout="centered")
    st.title("🎯 Bonus Predictor")
    st.markdown("Prevedi la probabilità che un giocatore **segni o faccia assist** nella prossima partita.")

    # --- Carica dataset e modelli
    models_goal = utils.load_models()  # modelli goal
    models_assist = utils.load_models_assist()  # modelli assist
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

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
        #submitted = True
        #player = 'orban'
        #team = "inter"
        #opponent = "cagliari"

    # --- Logica di predizione
    if submitted:
        if not player or not team or not opponent:
            st.warning("⚠️ Seleziona tutti i campi prima di procedere.")
        else:
            # === PREDIZIONE GOAL ===
            #tolgo finishing_form_resid perchè va ancora calcolata
            features_names_goal = list(models_goal["poiss_reg"].feature_names_)
            if "finishing_form_resid" in features_names_goal:
                features_names_goal.remove("finishing_form_resid")         

            X_goal, role = utils.prepare_features_xgb(features_names_goal, player, team, opponent, df_orig_goal, df_teams, df_teams_curr_season, models_goal["lin"])
            goal_proba = None
            if X_goal is not None:
                try:
                    #probabilità base dal modello
                    goal_proba = utils.predict_goal_probability(
                        model=models_goal["poiss_reg"],
                        X_goal=X_goal,
                        player=player,
                        role=role,
                        get_alpha_for_role_fn=utils.get_alpha_for_role
                    )              

                except Exception as e:
                    st.error(f"Errore nel modello goal: {e}")

            # === PREDIZIONE ASSIST ===
            features_names_assist = models_assist["log_reg_assist"].feature_names_in_
            X_assist = utils.prepare_features_assist(features_names_assist, player, team, opponent, df_orig_assist, df_teams, models_assist["scaler_features_assist"])
            assist_proba = None
            if X_assist is not None:
                try:
                    assist_proba = models_assist["log_reg_assist"].predict_proba(X_assist)[0, 1]
                except Exception as e:
                    st.error(f"Errore nel modello assist: {e}")

            # Probabilità combinate — Goal O Assist
            prob_any = goal_proba + assist_proba - (goal_proba * assist_proba)

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

            st.markdown("### ⚡ Probabilità Bonus Totale (Goal o Assist)")
            st.metric(label="Probabilità complessiva", value=f"{prob_any*100:.1f}%")

            # Barra di progressione
            st.progress(float(prob_any))   

            st.markdown("---")
            st.caption("🧠 Basato su xG, forma recente, efficienza di finalizzazione e forza difensiva avversaria.")

# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
