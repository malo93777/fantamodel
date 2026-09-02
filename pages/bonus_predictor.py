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
import plotly.graph_objects as go  # Import necessario per i grafici

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
            color: #ffffff !important;
            font-weight: 600 !important;
            text-shadow: 1px 1px 2px #00000066;
        }

        /* Titolo */
        h1,h2,h3,h4,h5 {
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
    models_goal = utils.load_models()
    models_assist = utils.load_models_assist()
    model_xg = utils.load_xg_model()
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

    #carico dataset per le prossime partite
    next_games_df = pd.read_csv(config.DATASET_DATA_DIR / config.NEXT_GAMES_FILE)
    df_voti = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI)

    # --- Dropdown dinamici
    players = sorted(df_orig_goal["player"].dropna().unique().tolist())
    teams = sorted(df_teams[df_teams['season'] == config.CURRENT_SEASON]['Team'].dropna().unique().tolist())
    opponents = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
    num_giornate = utils.count_matchdays(df_teams_curr_season)

    #mett0 le maiuscole nelle iniziali dei nomi dei giocatori 
    players = [p.title() for p in players]

    col1, col2 = st.columns(2)

    with col1:
        player = st.selectbox("👤 Giocatore", options=[""] + players)

        # Calcolo del team **fuori dal with**
    team = utils.get_latest_team(df_orig_goal, player, "player_team") if player else ""

    with col2:
            st.text_input("🏟️ Squadra", value=team, disabled=False)

            squadra, avversario, ha = utils.get_team_opponent_ha(player, df_voti, next_games_df)
            opponent = st.text_input("⚔️ Avversario", value=avversario.title(), disabled=False)

    # Valori di default, sempre validi anche se num_giornate < 10
    is_home = False
    is_away = False

    if num_giornate >= 10:
        is_home = ha == 'h'
        is_away = ha == 'a'
        if ha == 'h':
            st.markdown("### ⚑ Il giocatore gioca in: 🏠 Casa")
        elif ha == 'a':
            st.markdown("### ⚑ Il giocatore gioca in: ✈️ Trasferta")
        else:
            st.markdown("### ⚑ Il giocatore: luogo non determinato")

    submitted = st.button("⚡ Prevedi Bonus ")

    #DEBUG
    #submitted = True
    #player = "Cancellieri"
    #team = "Lazio"
    #opponent = "Cremonese"
    #is_home = True
    #is_away = False
    
    # --- Logica di predizione
    if submitted:
        if not player or not team or not opponent:
            st.warning("⚠️ Seleziona tutti i campi prima di procedere.")
        else:
            # === PREDIZIONE GOAL ===
            features_names_goal = list(models_goal["poiss_reg"].feature_names_)
            if "finishing_form_resid" in features_names_goal:
                features_names_goal.remove("finishing_form_resid")

            if is_home and not is_away:
                h_a_player = 'h'
            elif is_away and not is_home:
                h_a_player = 'a'
            else:
                h_a_player = None

            goal_proba = utils.get_goal_prob(
                model_xg["catboost_regressor_xg"],
                models_goal["poiss_reg"],
                features_names_goal,
                player, team, opponent, df_orig_goal, df_teams,
                df_teams_curr_season, models_goal["lin"], config.ROLE_STATS,
                h_a_player
            )

            # === PREDIZIONE ASSIST ===
            features_names_assist = models_assist["poisson_reg_assist"].feature_names_
            assist_proba = utils.get_assist_prob(
                models_assist["poisson_reg_assist"], features_names_assist,
                player, team, opponent, df_orig_assist, df_teams,
                df_teams_curr_season, h_a_player
            )

            # Probabilità combinate — Goal O Assist
            prob_bonus = goal_proba + assist_proba - (goal_proba * assist_proba)

            # --- Output finale
            st.markdown("---")
            st.subheader(f"📊 {player} ({team} vs {opponent})")

            if goal_proba is not None:
                st.metric("⚽ Probabilità Goal", f"{goal_proba*100:.1f}%")
                st.progress(float(goal_proba))

            if assist_proba is not None:
                st.metric("👟 Probabilità Assist", f"{assist_proba*100:.1f}%")
                st.progress(float(assist_proba))

            if goal_proba is None and assist_proba is None:
                st.warning("Nessuna previsione disponibile per questo giocatore.")

            st.markdown("### ⚡ Probabilità Bonus Totale (Goal o Assist)")
            st.metric(label="Probabilità complessiva", value=f"{prob_bonus*100:.1f}%")
            st.progress(float(prob_bonus))

            df_p = df_orig_goal[df_orig_goal["player"].str.contains(player, case=False, na=False)]
            df_p_assist = df_orig_assist[df_orig_assist["player"].str.contains(player, case=False, na=False)]

            col1, col2 = st.columns(2)
            with col1:
                curr_season_df = df_p[df_p['season'] == config.CURRENT_SEASON]
                curr_season_df_assist = df_p_assist[df_p_assist['season'] == config.CURRENT_SEASON]

                st.markdown(f"""
                    <div style='background-color:#1f2937; padding:18px; border-radius:12px; text-align:center; margin-bottom:12px;'>
                        <h2 style='color:white; margin:0;'>{df_p["player"].iloc[0]}</h2>
                    </div>
                """, unsafe_allow_html=True)

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("📅 Presenze", f"{curr_season_df.shape[0]}")
                    st.metric("⚽ Gol segnati", f"{int(curr_season_df['goals'].sum())}")
                    st.metric("🎯 Assist forniti", f"{int(curr_season_df_assist['assists'].sum())}")
                with colB:
                    st.metric("📊 xG medio stagione", f"{curr_season_df['sum_xG'].mean():.2f}")
                    st.metric("🔥 xG medio ultime 5", f"{curr_season_df['sum_xG'].tail(5).mean():.2f}")
                    st.metric("📈 xA medio stagione", f"{curr_season_df_assist['sum_xA'].mean():.2f}")
                    st.metric("✨ xA medio ultime 5", f"{curr_season_df_assist['sum_xA'].tail(5).mean():.2f}")
                    st.metric("👟  Media tiri a partita", f"{curr_season_df['shots_perMatch'].mean():.1f}")
                    st.metric("📉  Media tiri ultime 5", f"{curr_season_df['shots_perMatch'].tail(5).mean():.1f}")

            with col2:
                # GRAFICI xG e xA
                st.markdown("### 📉 Andamento xG / Goal nelle ultime 10 partite giocate")
                recente_df_goals = curr_season_df.sort_values("date").tail(10)
                recente_df_assist = curr_season_df_assist.sort_values("date").tail(10)

                plot_df = pd.merge(
                    recente_df_goals[["date","sum_xG"]],
                    recente_df_goals[["date","goals"]],
                    on="date", how="outer"
                ).sort_values("date")
                plot_df.rename(columns={"sum_xG":"xG","goals":"Goal"}, inplace=True)

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["xG"], mode="lines+markers", name="xG",
                                         line=dict(color="#3b82f6" 
                                                   #width=3
                                                   ), 
                                         marker=dict(size=8)))
                
                fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["Goal"], mode="lines+markers", name="Goals",
                                         line=dict(color="#10b981"
                                                    #width=3
                                                    ),
                                                    marker=dict(size=8)))
                
                fig.update_layout(height=330,
                                   #width=520,
                                   margin=dict(l=10,r=10,t=10,b=10),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📉 Andamento xA / Assist nelle ultime 10 partite giocate")
                plot_df2 = pd.merge(
                    recente_df_assist[["date","sum_xA"]],
                    recente_df_assist[["date","assists"]],
                    on="date", how="outer"
                ).sort_values("date")
                plot_df2.rename(columns={"sum_xA":"xAssist","assists":"Assist"}, inplace=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=plot_df2["date"], y=plot_df2["xAssist"], mode="lines+markers", name="xA",
                                          line=dict(color="#10b981", width=3), marker=dict(size=8)))
                fig2.add_trace(go.Scatter(x=plot_df2["date"], y=plot_df2["Assist"], mode="lines+markers", name="Assist",
                                          line=dict(color="#fbbf24", width=3), marker=dict(size=8)))
                fig2.update_layout(height=330,
                                    #width=520,
                                    margin=dict(l=10,r=10,t=10,b=10),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.caption("🧠 Basato su xG, xA, forma recente, qualità di tiro, forza offensiva della squadra e forza difensiva avversaria.")


# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
