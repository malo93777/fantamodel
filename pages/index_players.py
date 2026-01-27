import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import streamlit as st
import pandas as pd
import model_predict_voto
import config
import utils
import streamlit as st
import pandas as pd

def main():
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

    st.title("🎯 Indice di Schierabilità")
    st.markdown("Calcola l'indice di schierabilità per uno o più giocatori nella prossima partita.")

    # 🔙 Pulsante torna alla Home
    if st.button("🏠 Torna alla Home"):
        st.switch_page("app.py")

    st.title("Indice di Schierabilità")

    # Carica dati giocatori disponibili (puoi personalizzare questa parte)
    # Qui si assume che tu abbia un DataFrame con i giocatori e le squadre
    # Sostituisci con il tuo caricamento dati reale
    try:
        # --- Carica dataset e modelli
        df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
        df_voti = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI)
        df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
        df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

        # --- Dropdown dinamici
        players_list = sorted(df_orig_goal["player"].dropna().unique().tolist())
        teams_list = sorted(df_teams[df_teams['season'] == config.CURRENT_SEASON]['Team'].dropna().unique().tolist())
        opponents_list = sorted(df_orig_goal[df_orig_goal['season'] == config.CURRENT_SEASON]["opponent_team"].dropna().unique().tolist())
        num_giornate = utils.count_matchdays(df_teams_curr_season)

    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        players_list = []
        teams_list = []
        opponents_list = []

    # Selezione multipla giocatori
    giocatori = st.multiselect("Seleziona giocatori", players_list)
    input_data = []
    for player in giocatori:
        col1, col2, col3 = st.columns(3)
        with col1:
            default_team = utils.get_latest_team(df_orig_goal, player, "player_team") if player else ""
            squadra = st.selectbox(
                f"Squadra di {player}",
                teams_list,
                index=teams_list.index(default_team) if default_team in teams_list else 0,
                key=f"team_{player}"
            )
        with col2:
            avversario = st.selectbox(f"Avversario di {player}", opponents_list, key=f"opp_{player}")
        if num_giornate > 15:
            with col3:
                ha_label = st.selectbox(f"Casa/Trasferta {player}", ["Casa", "Trasferta"], key=f"ha_{player}")
                ha = "h" if ha_label == "Casa" else "a"
        else:
            ha = None
        input_data.append((player, squadra, avversario, ha))

    submitted = st.button("Calcola Indice")

    if submitted:
        if not giocatori:
            st.warning("Seleziona almeno un giocatore.")
        else:
            # Carica modello
            model = model_predict_voto.utils.load_fv_model()
            # Prepara input per la funzione di predizione
            players, teams, opponents, h_a = zip(*input_data)
            df_pred = model_predict_voto.pred_voto_prod(
                players, teams, opponents, h_a, df_voti, model['fantavoto_model']
            )
            # Rimuovi la colonna degli indici se presente
            df_pred = df_pred.reset_index(drop=True)
            # Formatta i nomi dei giocatori e l'indice di schierabilità
            def format_player(val):
                return f"<b>{str(val).title()}</b>"
            def format_index(val):
                return f"<b>{val}</b>"
            styled = df_pred.style.format({
                'player': format_player,
                'fantavoto_pred': format_index,
                'index': format_index if 'index' in df_pred.columns else lambda x: x
            }).hide(axis='index')
            # Applica grassetto anche ai nomi delle colonne
            html = styled.to_html(escape=False)
            html = html.replace('<th ', '<th style="font-weight:bold;" ')
            st.write("## Risultati Predizione")
            st.write(html, unsafe_allow_html=True)

# =====================================================
# 🔹 Run app
# =====================================================
if __name__ == "__main__":
    main()
