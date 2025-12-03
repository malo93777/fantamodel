import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Fantamodel", layout="wide")

# HTML completo – Streamlit NON lo modifica
html_code = """
<style>
    body {
        margin: 0;
        padding: 0;
    }

    .wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 90vh;
        text-align: center;
        font-family: 'Arial';
    }

    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.7rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #1e40af;
        margin-bottom: 2.5rem;
    }

    .cards {
        display: flex;
        justify-content: center;
        gap: 2rem;
    }

    .card {
        background-color: #dbeafe;
        padding: 25px 35px;
        border-radius: 16px;
        width: 250px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        transition: 0.2s ease-in-out;
    }

    .card:hover {
        background-color: #bfdbfe;
        transform: translateY(-4px);
        cursor: pointer;
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

<div class="wrapper">
    <div>
        <div class="title">Fantamodel App</div>
        <div class="subtitle">Seleziona un’opzione per iniziare</div>

        <div class="cards">
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
</div>
"""

components.html(html_code, height=800)
