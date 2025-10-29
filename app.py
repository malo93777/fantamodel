import streamlit as st
import sys
from pathlib import Path

# Imposta il percorso al modulo src/ per poter importare i tuoi file
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import diretto della pagina principale
from src.pages import compare_players

# Ora, se compare_players.py contiene codice Streamlit,
# viene eseguito automaticamente all'import