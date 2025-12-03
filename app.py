import streamlit as st

st.set_page_config(page_title="FantaModel", layout="wide")

# --- Homepage / Landing Page ---
st.markdown(
    """
    <style>
        .centered-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh;
            text-align: center;
        }

        .title {
            font-size: 3rem;
            font-weight: 800;
            color: #1e3a8a; /* Blu scuro */
            margin-bottom: 0.5rem;
        }

        .subtitle {
            font-size: 1.2rem;
            color: #1e40af;
            margin-bottom: 2.5rem;
        }

        .card-container {
            display: flex;
            gap: 2rem;
        }

        .card {
            background-color: #dbeafe; /* Blu chiaro */
            padding: 25px 35px;
            border-radius: 16px;
            width: 250px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            transition: 0.2s ease-in-out;
            cursor: pointer;
        }

        .card:hover {
            background-color: #bfdbfe;
            transform: translateY(-4px);
        }

        .card-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }

        .card-text {
            font-size: 1rem;
            color: #1e40af;
        }
    </style>

    <div class="centered-container">
        <div class="title">Fantamodel App</div>
        <div class="subtitle">Seleziona un’opzione per iniziare</div>

        <div class="card-container">
            <a href="/bonus_predictor" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">🔮 Bonus Predictor</div>
                    <div class="card-text">Stima goal, assist e bonus previsti.</div>
                </div>
            </a>

            <a href="/compare_players" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">⚔️ Compare Players</div>
                    <div class="card-text">Confronta due giocatori con metriche avanzate.</div>
                </div>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

