import streamlit as st

def hide_streamlit_ui():
    st.markdown("""
    <style>

        /* 🔥 Nasconde footer Streamlit */
        footer, #stFooter, [data-testid="stFooter"] {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
        }

        /* 🔥 Nasconde icone/badge/link in basso a destra (desktop + mobile) */
        .stApp a, .stApp button[title], [data-testid="stActionButton"], [data-testid="stAnchorButton"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* 🔥 Nasconde completamente il COMMAND BAR (icona rossa su mobile) */
        [data-testid="stCommandBarButton"], 
        [data-testid="stBottomBar"], 
        [data-testid="stBottomBlock"], 
        div[class*="st-emotion-cache"][style*="bottom"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* 🔥 Nasconde il pulsante MENU MOBILE (quello verde) */
        [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
        }

        /* Altri container bottom su mobile */
        div[style*="position: fixed"][style*="bottom"] {
            display: none !important;
            visibility: hidden !important;
        }

    </style>
    """, unsafe_allow_html=True)
