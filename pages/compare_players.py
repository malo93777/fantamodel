import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils

DATASET_DATA_DIR = config.DATASET_DATA_DIR
PROD_DATA_FILE_GOALS = config.PROD_DATA_FILE_GOALS
PROD_DATA_FILE_ASSIST = config.PROD_DATA_FILE_ASSIST
TEAMS_DATA_FILE = config.TEAMS_DATA_FILE
CURRENT_SEASON = config.CURRENT_SEASON

# ======================================================
# ⚔️ FUNZIONE PRINCIPALE
# ======================================================
def main():

    is_home1 = False
    is_away1 = False
    is_home2 = False
    is_away2 = False    

    st.set_page_config(page_title="Confronto Giocatori ⚔️", layout="centered")

     # 🔵 Sfondo blu rilassante
    st.markdown("""
        <style>

        /* Sfondo generale */
        .stApp {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        }

        /* Sfondo container contenuti */
        .main, .stAppViewContainer, .block-container {
            background: transparent !important;
        }

        /* Rimuovi eventuali layer bianchi */
        [data-testid="stAppViewContainer"] {
            background-color: transparent !important;
        }

        </style>
        """, unsafe_allow_html=True)

    st.title("⚔️ Confronta Giocatori")
    st.markdown("Confronta due giocatori su **forma**, **xG**, e probabilità di **goal, assist o bonus totale.**")

     # 🔙 Pulsante torna alla Home
    if st.button("🏠 Torna alla Home"):
        st.switch_page("app.py")

    # Carica dati e modelli
    models_goal = utils.load_models()
    models_assist = utils.load_models_assist()
    df_orig_goal = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

    players = sorted(df_orig_goal["player"].dropna().unique().tolist())
    teams = sorted(df_teams[df_teams['season'] == CURRENT_SEASON]['Team'].dropna().unique().tolist())
    opponents = sorted(df_orig_goal[df_orig_goal['season'] == CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
    num_giornate = utils.count_matchdays(df_teams_curr_season)
    # ======================================================
    # 🔗 Gestione query parameters
    # ======================================================
    query_params = st.query_params

    default_player1 = query_params.get("player1", [""])[0]
    default_team1 = query_params.get("team1", [""])[0]
    default_opponent1 = query_params.get("opponent1", [""])[0]

    default_player2 = query_params.get("player2", [""])[0]
    default_team2 = query_params.get("team2", [""])[0]
    default_opponent2 = query_params.get("opponent2", [""])[0]

    # ======================================================
    # 🎯 INPUT UTENTE
    # ======================================================
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("👤 Giocatore 1", [""] + players,
                               index=players.index(default_player1) + 1 if default_player1 in players else 0)
        team1 = st.selectbox("🏟️ Squadra 1", [""] + teams,
                             index=teams.index(default_team1) + 1 if default_team1 in teams else 0)
        opponent1 = st.selectbox("⚔️ Avversario 1", [""] + opponents,
                                 index=opponents.index(default_opponent1) + 1 if default_opponent1 in opponents else 0)     
        if num_giornate >= 10:

            place = st.radio(
                f"**Dove gioca {player1}?👤**",
                ["🏠 Casa", "✈️ Trasferta"],
                horizontal=True
            )
            if place == "🏠 Casa":
                is_home1 = True
                is_away1 = False
            elif place == "✈️ Trasferta":
                is_home1 = False
                is_away1 = True
            else:
                is_home1 = False
                is_away1 = False

    with col2:
        player2 = st.selectbox("👤 Giocatore 2", [""] + players,
                               index=players.index(default_player2) + 1 if default_player2 in players else 0)
        team2 = st.selectbox("🏟️ Squadra 2", [""] + teams,
                             index=teams.index(default_team2) + 1 if default_team2 in teams else 0)
        opponent2 = st.selectbox("⚔️ Avversario 2", [""] + opponents,
                                 index=opponents.index(default_opponent2) + 1 if default_opponent2 in opponents else 0)
        if num_giornate >= 10:

            place = st.radio(
                 f"**👤 Dove gioca {player2}?**",
                ["🏠 Casa", "✈️ Trasferta"],
                horizontal=True
            )
            if place == "🏠 Casa":
                is_home2 = True
                is_away2 = False
            elif place == "✈️ Trasferta":
                is_home2 = False
                is_away2 = True
            else:
                is_home2 = False
                is_away2 = False


    compare_btn = st.button("🔍 Confronta")

    # ======================================================
    # 📊 FUNZIONE PER CALCOLARE FEATURE E PROBABILITÀ
    # ======================================================
    def get_player_data(player, team, opponent,h_a_player=False):
        features_names_goal = list(models_goal["poiss_reg"].feature_names_)
        if "finishing_form_resid" in features_names_goal:
            features_names_goal.remove("finishing_form_resid")   

        X_goal, role = utils.prepare_features_xgb(
            features_names=features_names_goal,
            player=player,
            team=team,
            opponent=opponent,
            df_orig=df_orig_goal,
            df_teams=df_teams,
            df_teams_curr_season=df_teams_curr_season,
            lin_model=models_goal["lin"],
            ROLE_STATS=config.ROLE_STATS,
            h_a_player=h_a_player
        )

        X_assist = utils.prepare_features_assist(
            features_names=models_assist["log_reg_assist"].feature_names_in_,
            player=player,
            team=team,
            opponent=opponent,
            df_orig=df_orig_assist,
            df_teams=df_teams,
            scaler=models_assist["scaler_features_assist"]
        )

        if X_goal is None or X_assist is None:
            return None

        proba_goal = utils.predict_goal_probability(
                    model=models_goal["poiss_reg"],
                    X_goal=X_goal,
                    player=player,
                    role=role,
                    get_alpha_for_role_fn=utils.get_alpha_for_role
                    )    
        proba_assist = models_assist["log_reg_assist"].predict_proba(X_assist)[0, 1]
        proba_bonus = 1 - (1 - proba_goal) * (1 - proba_assist)

        df_p = df_orig_goal[df_orig_goal["player"].str.contains(player, case=False, na=False)]
        df_p_assist = df_orig_assist[df_orig_assist["player"].str.contains(player, case=False, na=False)]
        if df_p.empty or df_p_assist.empty:
            return None

        return {
            "player": player,
            "sum_xG": df_p["sum_xG"].mean(),
            "xG_last5": df_p["sum_xG"].rolling(5).mean().iloc[-1],
            "sum_xA": df_p_assist["sum_xA"].mean(),
            "xA_last5": df_p_assist["sum_xA"].rolling(5).mean().iloc[-1],
            "prob_goal": proba_goal,
            "prob_assist": proba_assist,
            "prob_bonus": proba_bonus
        }

    # ======================================================
    # ⚔️ MOSTRA CONFRONTO
    # ======================================================
    if compare_btn:
        if not player1 or not player2:
            st.warning("⚠️ Seleziona entrambi i giocatori per procedere.")
        else:
            st.query_params.update({
                "player1": player1,
                "team1": team1,
                "opponent1": opponent1,
                "player2": player2,
                "team2": team2,
                "opponent2": opponent2
            })

            if is_home1 and not is_away1:
                h_a_player1 = 'h'   
            elif is_away1 and not is_home1:
                h_a_player1 = 'a'
            else:
                h_a_player1 = None

            if is_home2 and not is_away2:
                h_a_player2 = 'h'   
            elif is_away2 and not is_home2:
                h_a_player2 = 'a'
            else:
                h_a_player2 = None

            p1 = get_player_data(player1, team1, opponent1, h_a_player=h_a_player1)
            p2 = get_player_data(player2, team2, opponent2, h_a_player=h_a_player2)

            if p1 and p2:

                st.markdown("### 📊 Statistiche a confronto (Serie A)")

                col1, col2 = st.columns(2)

                # --- PLAYER 1 ---
                with col1:
                    st.markdown(
                        f"""
                        <div style='
                            background-color:#1f2937;
                            padding:16px;
                            border-radius:12px;
                            text-align:center;
                            margin-bottom:12px;
                        '>
                            <h3 style='color:white; margin:0;'>{p1["player"]}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    colA, colB = st.columns(2)
                    with colA:
                        st.metric("📊 xG medio SA", f"{p1['sum_xG']:.2f}")
                        st.metric("🔥 xG ultime 5", f"{p1['xG_last5']:.2f}")
                        st.metric("🎯 xA ultime 5", f"{p1['xA_last5']:.2f}")

                    with colB:
                        st.metric("📈 xA medio SA", f"{p1['sum_xA']:.2f}")
                        st.metric("⚽ Prob. Goal", f"{p1['prob_goal']*100:.1f}%")
                        st.metric("✨ Prob. Assist", f"{p1['prob_assist']*100:.1f}%")

                    st.metric("💎 Prob. Bonus Totale", f"{p1['prob_bonus']*100:.1f}%")

                # --- PLAYER 2 ---
                with col2:
                    st.markdown(
                        f"""
                        <div style='
                            background-color:#1f2937;
                            padding:16px;
                            border-radius:12px;
                            text-align:center;
                            margin-bottom:12px;
                        '>
                            <h3 style='color:white; margin:0;'>{p2["player"]}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    colA2, colB2 = st.columns(2)
                    with colA2:
                        st.metric("📊 xG medio", f"{p2['sum_xG']:.2f}")
                        st.metric("🔥 xG ultime 5", f"{p2['xG_last5']:.2f}")
                        st.metric("🎯 xA ultime 5", f"{p2['xA_last5']:.2f}")

                    with colB2:
                        st.metric("📈 xA medio", f"{p2['sum_xA']:.2f}")
                        st.metric("⚽ Prob. Goal", f"{p2['prob_goal']*100:.1f}%")
                        st.metric("✨ Prob. Assist", f"{p2['prob_assist']*100:.1f}%")

                    st.metric("💎 Prob. Bonus Totale", f"{p2['prob_bonus']*100:.1f}%")

                # --- RADAR PLOT ---
                radar = go.Figure()
                categories = ["xG medio in Serie A", "xG ultime 5", "xA ultime 5", "Prob. Goal", "Prob. Assist", "Prob. Bonus"]

                p1_values = [
                    p1["sum_xG"], p1["xG_last5"], p1["xA_last5"],
                    p1["prob_goal"], p1["prob_assist"], p1["prob_bonus"]
                ]

                p2_values = [
                    p2["sum_xG"], p2["xG_last5"], p2["xA_last5"],
                    p2["prob_goal"], p2["prob_assist"], p2["prob_bonus"]
                ]

                radar.add_trace(go.Scatterpolar(r=p1_values, theta=categories, fill='toself', name=p1["player"], line_color="dodgerblue"))
                radar.add_trace(go.Scatterpolar(r=p2_values, theta=categories, fill='toself', name=p2["player"], line_color="orange"))

                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    title="Radar Statistico (Goal + Assist + Bonus Totale)",
                    height=600
                )
                st.plotly_chart(radar, use_container_width=True)

                st.caption("🧠 Basato su xG, xA, forma recente, qualità di tiro, forza offensiva della squadra e forza difensiva avversaria.")

                # Link di condivisione
                #st.markdown("### 🔗 Condividi questo confronto")
                #link = f"{st.link_button()}?player1={player1}&team1={team1}&opponent1={opponent1}&player2={player2}&team2={team2}&opponent2={opponent2}"
                #st.code(link, language="text")
            else:
                st.error("Impossibile calcolare i dati per uno dei giocatori.")

# ======================================================
# ⚔️ EXEC MAIN
# ======================================================
if __name__ == "__main__":
    main()
