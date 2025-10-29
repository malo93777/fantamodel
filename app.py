import streamlit as st
import sys
from pathlib import Path

# Imposta il percorso al modulo src/
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Importa e avvia la pagina principale
APP_PATH = SRC_DIR / "pages" / "compare_players.py"

with open(APP_PATH, encoding="utf-8") as f:
    code = compile(f.read(), str(APP_PATH), "exec")
    exec(code, globals())
