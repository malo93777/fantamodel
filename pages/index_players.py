import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import streamlit as st
import pandas as pd
import model_predict_voto
import config
import utils

def main():
    # =====================================================
    # 🔹 Config pagina e CSS
    # =====================================================
    st.set_page_config(page_title="Indice Schierabilità ⚽", page_icon="✨", layout="centered")
    
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

    /* 🔥 TESTO NERO GENERALE */
    html, body, [class*="css"], .stMarkdown, .stText,
    .stSelectbox label, .stRadio label,
    .stMetric, .stMetric label,
    .stRadio, .stSelectbox, .stAlert {
        color: #111 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }

    /* Titoli */
    h1,h2,h3,h4,h5 {
        color: #111 !important;
        text-shadow: none !important;
    }

    /* ⚪ TESTO BIANCO NEI BOTTONI */
    .stButton > button {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Sfondo bottoni */
    .stButton > button {
        background-color: #1e40af !important;
        border: none !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # 🔹 Titolo e descrizione
    # =====================================================
    st.title("🎯 Indice di Schierabilità")
    st.markdown("Calcola l'indice di schierabilità per uno o più giocatori nella prossima partita inserendo i giocatori e gli avversari.")
    st.markdown("Seleziona Top Indici per Ruolo per vedere i migliori giocatori per la prossima partita per ogni reparto.")

    # Pulsante torna alla Home
    if st.button("🏠 Torna alla Home"):
        st.switch_page("app.py")

    # =====================================================
    # 🔹 Caricamento dati
    # =====================================================
    try:
        df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
        df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
        df_voti = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI)
        df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
        df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)
        next_games_df = pd.read_csv(config.DATASET_DATA_DIR / config.NEXT_GAMES_FILE)  # Assumendo il path corretto
        df_infortunati = pd.read_csv(config.DATASET_DATA_DIR / config.INFORTUNATI_FILE)

        # Liste per dropdown
        players_list = sorted(df_orig_goal["player"].dropna().unique().tolist())
        teams_list = sorted(df_teams[df_teams['season'] == config.CURRENT_SEASON]['Team'].dropna().unique().tolist())
        opponents_list = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
        num_giornate = utils.count_matchdays(df_teams_curr_season)

    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        players_list = []
        teams_list = []
        opponents_list = []
        next_games_df = pd.DataFrame()

    # =====================================================
    # 🔹 Selezione giocatori e input
    # =====================================================
    players_list_display = [p.lower().title() for p in players_list]
    player_display_to_raw = dict(zip(players_list_display, players_list))

    giocatori_display = st.multiselect(
        "Seleziona giocatori",
        players_list_display
    )

    giocatori = [player_display_to_raw[p] for p in giocatori_display]

    input_data = []

    for player in giocatori_display:
        col1, _, _ = st.columns([3,3,2])
        
        with col1:
            default_team = utils.get_latest_team(df_orig_goal, player, "player_team") if player else ""
            squadra = st.selectbox(
                f"Squadra di {player}",
                teams_list,
                index=teams_list.index(default_team) if default_team in teams_list else 0,
                key=f"team_{player}"
            )
        
        # Calcola avversario e ha automaticamente
        squadra, avversario, ha = utils.get_team_opponent_ha(player, df_voti, next_games_df, utils)
        
        input_data.append((player, squadra, avversario, ha))

    # =====================================================
    # 🔹 Pulsanti affiancati
    # =====================================================
    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        submitted = st.button("Calcola Indice")
    with col_btn2:
        top_ruolo = st.button("Top Indici per Ruolo")

    # =====================================================
    # 🔹 Logica Calcola Indice
    # =====================================================
    #submitted=True
    #giocatori = "francesco esposito"
    #input_data.append((giocatori, "sassuolo","inter","h"))
    if submitted:
        if not giocatori:
            st.warning("Seleziona almeno un giocatore.")
        else:         
            
            # --- Carica dataset e modelli 
            model_goal = utils.load_models() 
            model_assist = utils.load_models_assist() 
            model_xg = utils.load_xg_model()
            model = model_predict_voto.utils.load_fv_model()
            model_gk = model_predict_voto.utils.load_fv_model_gk()

            players, teams, opponents, h_a = zip(*input_data)
            #preprocesso df voti
            df_voti = utils.prepare_voto_dataframe(df_voti)

            df_pred = model_predict_voto.pred_voto_prod(
                players, teams, opponents, h_a, 
                df_voti, df_orig_goal,df_orig_assist, df_teams, df_teams_curr_season,
                model_goal, model_assist, model_xg,model['fantavoto_model']
            )
            df_pred = df_pred.reset_index(drop=True)
            df_pred = utils.prepare_df_for_display(df_pred).copy()

            # 🔢 Assicuriamoci che siano numerici e arrotondiamo
            if 'fantavoto_pred' in df_pred.columns:
                df_pred['fantavoto_pred'] = pd.to_numeric(
                    df_pred['fantavoto_pred'], errors='coerce'
                ).round(1)

            if 'Index' in df_pred.columns:
                df_pred['Index'] = pd.to_numeric(
                    df_pred['Index'], errors='coerce'
                ).round(1)

            # 🎨 Formattazione
            def format_player(val):
                return f"<b>{str(val).title()}</b>"

            def format_number(val):
                if pd.isna(val):
                    return ""
                return f"<b>{val:.1f}</b>"

            format_dict = {
                'player': format_player,
                'fantavoto_pred': format_number,
            }

            if 'Index' in df_pred.columns:
                format_dict['Index'] = format_number

            styled = (
                df_pred.style
                .format(format_dict)
                .hide(axis='index')
            )

            html = styled.to_html(escape=False)
            html = html.replace('<th ', '<th style="font-weight:bold;" ')

            st.write("## Risultati Predizione")
            st.write(html, unsafe_allow_html=True)

    # =====================================================
    # 🔹 Logica Top Indici per Ruolo
    # =====================================================
    # =====================================================
    #DEBUG
    #top_ruolo = True 
    if top_ruolo:
        if df_voti.empty or next_games_df.empty:
            st.warning("Dati insufficienti per calcolare i top indici.")
        else:
            st.write("## 🔝 Top Indici per Ruolo")
            model = model_predict_voto.utils.load_fv_model()
            model_gk = model_predict_voto.utils.load_fv_model_gk()

            with st.spinner("⏳ Calcolo dei Top Indici in corso..."):
                    #preprocesso df voti
                    df_voti = utils.prepare_voto_dataframe(df_voti)

                    results = model_predict_voto.predizioni_per_ruolo(
                    df_voti,
                    next_games_df,
                    df_infortunati,
                    pipeline=model['fantavoto_model'],
                    pipeline_gk=model_gk['fantavoto_model_gk'],
                    top_n=10
                )

            if results:
                if 'P' in results:
                    st.subheader("🧤 Portieri")
                    st.dataframe(
                        utils.prepare_df_for_display(results['P']),
                        use_container_width=True
                    )

                if 'D' in results:
                    st.subheader("🛡 Difensori")
                    st.dataframe(
                        utils.prepare_df_for_display(results['D']),
                        use_container_width=True
                    )

                if 'C' in results:
                    st.subheader("👟 Centrocampisti")
                    st.dataframe(
                        utils.prepare_df_for_display(results['C']),
                        use_container_width=True
                    )

                if 'A' in results:
                    st.subheader("⚽ Attaccanti")
                    st.dataframe(
                        utils.prepare_df_for_display(results['A']),
                        use_container_width=True
                    )


# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
