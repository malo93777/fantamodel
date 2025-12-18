import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import utils
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="Bonus Predictor ⚽", page_icon="✨", layout="centered")

    # ----------------------------
    # Stile generale
    # ----------------------------
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #60a5fa 0%, #93c5fd 100%) !important; }
        .main, .stAppViewContainer, .block-container { background: transparent !important; }
        html, body, [class*="css"], .stMarkdown, .stText, .stSelectbox label, .stRadio label,
        .stMetric, .stMetric label, .stRadio, .stSelectbox, .stButton, .stAlert { 
            color: #ffffff !important; 
            font-weight: 600 !important; 
            text-shadow: 1px 1px 2px #00000066; 
        }
        h1,h2,h3,h4,h5 { color: #ffffff !important; text-shadow: 2px 2px 4px #00000055; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Bonus Predictor")
    st.markdown("Prevedi la probabilità che un giocatore **segni o faccia assist** nella prossima partita.")

    # ----------------------------
    # Caricamento dati e modelli
    # ----------------------------
    models_goal = utils.load_models()
    models_assist = utils.load_models_assist()
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

    players = sorted(df_orig_goal["player"].dropna().unique().tolist())
    opponents = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
    num_giornate = utils.count_matchdays(df_teams_curr_season)

    # ----------------------------
    # FORM input giocatore
    # ----------------------------
    def update_team():
        st.session_state['team'] = utils.get_latest_team(df_orig_goal, st.session_state['player'], "player_team")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            player = st.selectbox("👤 Giocatore", options=[""] + players, key="player", on_change=update_team)
        with col2:
            team_input = st.text_input("🏟️ Squadra", value="", key="team")

        opponent = st.selectbox("⚔️ Avversario", options=[""] + opponents)

        is_home, is_away = False, False
        if num_giornate >= 10:
            place = st.radio("🏠 Casa / ✈️ Trasferta", ["🏠 Casa", "✈️ Trasferta"], horizontal=True)
            if place == "🏠 Casa":
                is_home, is_away = True, False
            elif place == "✈️ Trasferta":
                is_home, is_away = False, True

        submitted = st.form_submit_button("Prevedi Bonus")

    # ----------------------------
    # Output predizione
    # ----------------------------
    if submitted:
        if not player or not team or not opponent:
            st.warning("⚠️ Seleziona tutti i campi prima di procedere.")
            return

        # --- Predizione Goal ---
        features_goal = list(models_goal["poiss_reg"].feature_names_)
        if "finishing_form_resid" in features_goal:
            features_goal.remove("finishing_form_resid")

        h_a_player = 'h' if is_home else 'a' if is_away else None

        goal_proba = utils.get_goal_prob(
            models_goal["poiss_reg"], features_goal,
            player, team, opponent, df_orig_goal, df_teams, df_teams_curr_season,
            models_goal["lin"], config.ROLE_STATS, h_a_player
        )

        # --- Predizione Assist ---
        features_assist = models_assist["poisson_reg_assist"].feature_names_
        assist_proba = utils.get_assist_prob(
            models_assist["poisson_reg_assist"], features_assist,
            player, team, opponent, df_orig_assist, df_teams, df_teams_curr_season, h_a_player
        )

        # --- Probabilità combinata ---
        prob_bonus = goal_proba + assist_proba - (goal_proba * assist_proba)

        # --- Visualizzazione principali metriche ---
        st.markdown("---")
        st.subheader(f"📊 {player} ({team} vs {opponent})")
        st.metric("⚽ Goal", f"{goal_proba*100:.1f}%")
        st.metric("👟 Assist", f"{assist_proba*100:.1f}%")
        st.metric("💎 Bonus Totale", f"{prob_bonus*100:.1f}%")
        st.progress(float(prob_bonus))

<<<<<<< HEAD
        # --- Dati storici ---
        df_p_goal = df_orig_goal[df_orig_goal["player"].str.contains(player, case=False, na=False)]
        df_p_assist = df_orig_assist[df_orig_assist["player"].str.contains(player, case=False, na=False)]
        curr_goal = df_p_goal[df_p_goal['season'] == config.CURRENT_SEASON]
        curr_assist = df_p_assist[df_p_assist['season'] == config.CURRENT_SEASON]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='background-color:#1f2937;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;'><h2 style='color:white;margin:0;'>{player}</h2></div>", unsafe_allow_html=True)
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("📅 Presenze", curr_goal.shape[0])
                st.metric("⚽ Gol", int(curr_goal['goals'].sum()))
                st.metric("🎯 Assist", int(curr_assist['assists'].sum()))
            with colB:
                st.metric("📊 xG medio stagione", f"{curr_goal['sum_xG'].mean():.2f}")
                st.metric("🔥 xG ultime 5", f"{curr_goal['xG_last5'].mean():.2f}")
                st.metric("📈 xA medio stagione", f"{curr_assist['sum_xA'].mean():.2f}")
                st.metric("✨ xA ultime 5", f"{curr_assist['xA_last5'].mean():.2f}")
=======
            if assist_proba is not None:
                st.metric("🎯 Probabilità Assist", f"{assist_proba * 100:.1f}%")
                st.progress(float(assist_proba))
>>>>>>> parent of 6624070 (fix layout)

        with col2:
            # --- Grafico xG / Goal ---
            recente_goal = curr_goal.sort_values("date").tail(10)
            plot_goal = pd.merge(
                recente_goal[["date","sum_xG"]],
                recente_goal[["date","goals"]],
                on="date", how="outer"
            ).sort_values("date")
            plot_goal.rename(columns={"sum_xG":"xG","goals":"Goal"}, inplace=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_goal["date"], y=plot_goal["xG"], mode="lines+markers", name="xG", line=dict(color="#3b82f6", width=3), marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=plot_goal["date"], y=plot_goal["Goal"], mode="lines+markers", name="Goal", line=dict(color="#10b981", width=3), marker=dict(size=8)))
            fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
            st.plotly_chart(fig, use_container_width=True)

            # --- Grafico xA / Assist ---
            recente_assist = curr_assist.sort_values("date").tail(10)
            plot_assist = pd.merge(
                recente_assist[["date","sum_xA"]],
                recente_assist[["date","assists"]],
                on="date", how="outer"
            ).sort_values("date")
            plot_assist.rename(columns={"sum_xA":"xAssist","assists":"Assist"}, inplace=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=plot_assist["date"], y=plot_assist["xAssist"], mode="lines+markers", name="xA", line=dict(color="#10b981", width=3), marker=dict(size=8)))
            fig2.add_trace(go.Scatter(x=plot_assist["date"], y=plot_assist["Assist"], mode="lines+markers", name="Assist", line=dict(color="#fbbf24", width=3), marker=dict(size=8)))
            fig2.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.caption("🧠 Basato su xG, xA, forma recente, qualità di tiro, forza offensiva della squadra e forza difensiva avversaria.")

<<<<<<< HEAD
=======
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
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("📅 Presenze", f"{curr_season_df.shape[0]}")
                    st.metric("⚽ Gol segnati", f"{int(curr_season_df['goals'].sum())}")
                    st.metric("🎯 Assist forniti", f"{int(curr_season_df_assist['assists'].sum())}")

                with colB:
                    st.metric("📊 xG medio stagione", f"{curr_season_df['sum_xG'].mean():.2f}")
                    st.metric("🔥 xG medio ultime 5", f"{curr_season_df['xG_last5'].mean():.2f}")
                    st.metric("📈 xA medio stagione", f"{curr_season_df_assist['sum_xA'].mean():.2f}")
                    st.metric("✨ xA medio ultime 5", f"{curr_season_df_assist['xA_last5'].mean():.2f}")
            with col2:
                    # ===============================
                    # 📈 GRAFICO ANDAMENTO ULTIME 5 PARTITE (xG & xA)
                    # ===============================

                    st.markdown("### 📉 Andamento xG / Goal nelle ultime 10 partite giocate")

                    # Prendiamo solo le ultime 10 partite del giocatore
                    recente_df_goals = curr_season_df.sort_values("date").tail(10)
                    recente_df_assist = curr_season_df_assist.sort_values("date").tail(10)

                    plot_df = pd.merge(
                        recente_df_goals[["date", "sum_xG"]],                  
                        recente_df_goals[["date", "goals"]],
                        on="date",
                        how="outer"
                    ).sort_values("date")

                    plot_df.rename(columns={"sum_xG": "xG", "goals": "Goal"}, inplace=True)

                    # ---- Plotly ----
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=plot_df["date"],
                        y=plot_df["xG"],
                        mode="lines+markers",
                        name="xG",
                        line=dict(color="#3b82f6", width=3),
                        marker=dict(size=8)
                    ))

                    fig.add_trace(go.Scatter(
                        x=plot_df["date"],
                        y=plot_df["Goal"],
                        mode="lines+markers",
                        name="Goals",
                        line=dict(color="#10b981", width=3),
                        marker=dict(size=8)
                    ))

                    fig.update_layout(
                        height=330,
                        width=520,   # <--- LARGHEZZA MAGGIORE
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
                    )

                    st.plotly_chart(fig, use_container_width=False)

                                        # ===============================
                    # 📈 GRAFICO ANDAMENTO ULTIME 10 PARTITE (xA & Assist)
                    # ===============================

                    st.markdown("### 📉 Andamento xA / Assist nelle ultime 10 partite giocate")

                    plot_df2 = pd.merge(
                        recente_df_assist[["date", "sum_xA"]],
                        recente_df_assist[["date", "assists"]],
                        on="date",
                        how="outer"
                    ).sort_values("date")

                    plot_df2.rename(columns={"sum_xA": "xAssist", "assists": "Assist"}, inplace=True)

                    fig2 = go.Figure()

                    fig2.add_trace(go.Scatter(
                        x=plot_df2["date"],
                        y=plot_df2["xAssist"],
                        mode="lines+markers",
                        name="xA",
                        line=dict(color="#10b981", width=3),
                        marker=dict(size=8)
                    ))

                    fig2.add_trace(go.Scatter(
                        x=plot_df2["date"],
                        y=plot_df2["Assist"],
                        mode="lines+markers",
                        name="Assist",
                        line=dict(color="#fbbf24", width=3),
                        marker=dict(size=8)
                    ))

                    fig2.update_layout(
                        height=330,
                        width=520,
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
                    )

                    st.plotly_chart(fig2, use_container_width=False)

            st.markdown("---")
            st.caption("🧠 Basato su xG, xA, forma recente, qualità di tiro, forza offensiva della squadra e forza difensiva avversaria.")

# =====================================================
# 🔹 Run app
# =====================================================
>>>>>>> parent of 6624070 (fix layout)
if __name__ == "__main__":
    main()
