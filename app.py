import streamlit as st
import streamlit.components.v1 as components
from style import hide_streamlit_ui

st.set_page_config(page_title="FantaModel", layout="wide")
hide_streamlit_ui()

# 🔵 Nascondi completamente sidebar + barra superiore Streamlit
HIDE_ALL = """
<style>
    /* Nasconde sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Nasconde barra superiore (hamburger + "Made with Streamlit") */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Rimuove spazio superiore */
    .block-container {
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Rimuove pallino menu laterale mobile */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
"""

st.markdown(HIDE_ALL, unsafe_allow_html=True)

# HTML completo
html_code = """
<style>

    /* ---- SFONDO SFUMATO ---- */
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ---- CONTAINER CENTRALE ---- */
    .wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 92vh;
        text-align: center;
        padding: 20px;
    }

    /* ---- LOGO (opzionale: se lo aggiungi) ---- */
    .logo {
        margin-bottom: 20px;
        width: 140px;
        animation: fadeIn 1.5s ease-in-out;
    }

    /* ---- TITOLO ---- */
    .title {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
        text-shadow: 0px 3px 8px rgba(0,0,0,0.25);
        animation: fadeInDown 1s ease;
    }

    /* ---- SOTTOTITOLO ---- */
    .subtitle {
        font-size: 1.25rem;
        color: #e0e7ff;
        margin-bottom: 2.8rem;
        animation: fadeInDown 1.4s ease;
    }

    /* ---- CONTENITORE CARD ---- */
    .cards {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        animation: fadeInUp 1.7s ease;
    }

    /* ---- CARD ---- */
    .card {
        background: white;
        padding: 28px 40px;
        border-radius: 18px;
        width: 260px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
        transition: 0.25s ease-in-out;
        transform: translateY(0px);
    }

    /* HOVER DELLA CARD */
    .card:hover {
        transform: translateY(-8px) scale(1.03);
        background:#f8fafc;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
        cursor: pointer;
    }

    .card-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.6rem;
    }

    .card-text {
        font-size: 1rem;
        color: #475569;
    }

    /* ---- ANIMAZIONI ---- */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }

</style>

<div class="wrapper">
    <div>

        <!-- LOGO (puoi aggiungerlo qui se vuoi) -->
        <!-- <img src="LOGO_URL" class="logo"> -->

        <div class="title">FantaModel App</div>
        <div class="subtitle">Seleziona una funzionalità per iniziare</div>

        <div class="cards">

            <a href="/index_players" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">📊 Indice Schierabilità</div>
                    <div class="card-text">
                        Calcola l'indice di schierabilità per uno o più giocatori.
                    </div>
                </div>
            </a>

            <a href="/bonus_predictor" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">🔮 Bonus Predictor</div>
                    <div class="card-text">
                        Stima goal, assist, xG/xA e probabilità di bonus.
                    </div>
                </div>
            </a>

            <a href="/compare_players" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">⚔️ Compare Players</div>
                    <div class="card-text">
                        Confronta statistiche avanzate tra due giocatori.
                    </div>
                </div>
            </a>

        </div>

    </div>
</div>
"""

components.html(html_code, height=900)