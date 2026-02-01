import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="FantaModel",
    layout="wide"
)

# Nascondi UI Streamlit (SAFE)
st.markdown("""
<style>
[data-testid="stSidebar"],
header[data-testid="stHeader"],
[data-testid="collapsedControl"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# HTML HOME (CONFINATO)
components.html(
"""
<style>
html, body {
    margin: 0;
    padding: 0;
}

.wrapper {
    height: 100vh;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: 'Segoe UI', sans-serif;
}

.container {
    text-align: center;
}

.title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #e0e7ff;
    margin-bottom: 3rem;
}

.cards {
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
}

.card {
    background: white;
    padding: 28px 40px;
    border-radius: 18px;
    width: 260px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    transition: 0.25s;
}

.card:hover {
    transform: translateY(-8px);
    cursor: pointer;
}

.card-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e3a8a;
}
</style>

<div class="wrapper">
    <div class="container">
        <div class="title">FantaModel App</div>
        <div class="subtitle">Seleziona una funzionalità</div>

        <div class="cards">
            <a href="/index_players" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">📊 Indice Schierabilità</div>
                </div>
            </a>

            <a href="/bonus_predictor" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">🔮 Bonus Predictor</div>
                </div>
            </a>

            <a href="/compare_players" style="text-decoration:none;">
                <div class="card">
                    <div class="card-title">⚔️ Compare Players</div>
                </div>
            </a>
        </div>
    </div>
</div>
""",
height=900,
scrolling=False
)