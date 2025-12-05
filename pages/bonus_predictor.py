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
    
    st.markdown("""
        <style>

        /* Sfondo blu chiaro */
        .stApp {
            background: linear-gradient(135deg, #60a5fa 0%, #93c5fd 100%) !important;
        }

        /* Contenitori trasparenti */
        .main, .stAppViewContainer, .block-container {
            background: transparent !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: transparent !important;
        }

        /* 🔥 TESTO ULTRA LEGGIBILE */
        html, body, [class*="css"], .stMarkdown, .stText, .stSelectbox label, .stRadio label,
        .stMetric, .stMetric label, .stRadio, .stSelectbox, .stButton, .stAlert {
            color: #ffffff !important;       /* Testo bianco purissimo */
            font-weight: 600 !important;     /* Leggermente più marcato */
            text-shadow: 1px 1px 2px #00000066;  /* Leggera ombra per super contrasto */
        }

        /* Titolo */
        h1, h2, h3, h4, h5 {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px #00000055;
        }

        </style>
        """, unsafe_allow_html=True)

    st.title("🎯 Bonus Predictor")
    st.markdown("Prevedi la probabilità che un giocatore **segni o faccia assist** nella prossima partita.")
    # 🔙 Pulsante torna alla Home
    if st.button("🏠 Torna alla Home"):
        st.switch_page("app.py")

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
    num_giornate = utils.count_matchdays(df_teams_curr_season)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            player = st.selectbox("👤 Giocatore", options=[""] + players)
        with col2:
            team = st.selectbox("🏟️ Squadra", options=[""] + teams)

        opponent = st.selectbox("⚔️ Avversario", options=[""] + opponents)

        if num_giornate >= 10:

            is_home = False
            is_away = False

            st.markdown("### ⚑ Il giocatore gioca in:")

            place = st.radio(
                "",
                ["🏠 Casa", "✈️ Trasferta"],
                horizontal=True
            )
            if place == "🏠 Casa":
                is_home = True
                is_away = False
            elif place == "✈️ Trasferta":
                is_home = False
                is_away = True
            else:
                is_home = False
                is_away = False

            submitted = st.form_submit_button("Prevedi Bonus")


        #*** per test in locale ***
        submitted = True
        player = 'nico paz'
        team = "como"
        opponent = "cagliari"
        is_home = True
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

            if is_home and not is_away:
                h_a_player = 'h'   
            elif is_away and not is_home:
                h_a_player = 'a'
            else:
                h_a_player = None

            X_goal, role = utils.prepare_features_xgb(features_names_goal,
                                                       player, 
                                                       team, 
                                                       opponent, 
                                                       df_orig_goal, 
                                                       df_teams, 
                                                       df_teams_curr_season, 
                                                       models_goal["lin"],
                                                       config.ROLE_STATS,
                                                       h_a_player       
                                   )
            
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

            df_p = df_orig_goal[df_orig_goal["player"].str.contains(player, case=False, na=False)]
            df_p_assist = df_orig_assist[df_orig_assist["player"].str.contains(player, case=False, na=False)]

            col1, col2 = st.columns(2)

            with col1:
                curr_season_df = df_p[df_p['season'] == config.CURRENT_SEASON]
                curr_season_df_assist = df_p_assist[df_p_assist['season'] == config.CURRENT_SEASON]

                st.markdown(
                    f"""
                    <div style='
                        background-color:#1f2937;
                        padding:18px;
                        border-radius:12px;
                        text-align:center;
                        margin-bottom:12px;
                    '>
                        <h2 style='color:white; margin:0;'>{df_p["player"].iloc[0]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Metriche in griglia
                colA, colB = st.columns(2)
                with colA:
                    st.metric("📅 Presenze", f"{curr_season_df.shape[0]}")
                    st.metric("⚽ Gol segnati", f"{int(curr_season_df['goals'].sum())}")
                    st.metric("🎯 Assist forniti", f"{int(curr_season_df_assist['assists'].sum())}")

                with colB:
                    st.metric("📊 xG medio stagione", f"{curr_season_df['sum_xG'].mean():.2f}")
                    st.metric("🔥 xG medio ultime 5", f"{curr_season_df['xG_last5'].mean():.2f}")
                    st.metric("📈 xA medio stagione", f"{curr_season_df_assist['sum_xA'].mean():.2f}")
                    st.metric("✨ xA medio ultime 5", f"{curr_season_df_assist['xA_last5'].mean():.2f}")

            st.markdown("---")
            st.caption("🧠 Basato su xG, xA, forma recente, qualità di tiro, forza offensiva della squadra e forza difensiva avversaria.")

# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
