import streamlit as st

def hide_streamlit_ui():
    st.markdown("""
    <style>

        /* Nasconde footer "Made with Streamlit" */
        footer, #stFooter {
            visibility: hidden !important;
            height: 0 !important;
        }

        /* Nasconde icone/badge/link in basso a destra */
        .stApp a {
            visibility: hidden !important;
        }

        /* Rimuove completamente widget di stato */
        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        /* Nasconde menu 3 puntini */
        [data-testid="stToolbar"] {
            display: none !important;
        }

    </style>
    """, unsafe_allow_html=True)
