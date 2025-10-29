import streamlit as st
import sys
from pathlib import Path

# Aggiunge la cartella "src" al percorso di ricerca dei moduli
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Percorso al file principale della tua app Streamlit
APP_PATH = SRC_DIR / "pages" / "compare_players.py"

# Esegue lo script principale
with open(APP_PATH, encoding="utf-8") as f:
    code = compile(f.read(), str(APP_PATH), "exec")
    exec(code, globals())
