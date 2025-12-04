import streamlit as st

def hide_streamlit_ui():
    st.markdown("""
    <style>

        /* 🔥 Nasconde footer Streamlit su desktop + mobile */
        footer, #stFooter, [data-testid="stFooter"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* 🔥 Nasconde icona GitHub / link / badge in basso a destra */
        .stApp a, .stApp button[title], .stDeployButton, 
        [data-testid="stAnchorButton"], 
        [data-testid="stActionButton"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* 🔥 Nasconde toolbar (3 puntini) ovunque */
        [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Nasconde status widget */
        [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Nasconde overlay mobile (quello che appare in basso) */
        div[data-testid="stBottomBlock"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Nasconde il menu laterale su mobile */
        [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
        }

    </style>
    """, unsafe_allow_html=True)
