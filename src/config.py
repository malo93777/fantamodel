from pathlib import Path

# Base directory del progetto (cartella principale)
BASE_DIR = Path(__file__).resolve().parent.parent

# Percorsi principali
DATASET_DATA_DIR = BASE_DIR / "dataset"
#MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# File
MATCH_DATA_FILE = "matches_df.csv"
PROD_DATA_FILE = "PROD_shots_2025_preproc_Serie_A.csv"
PLAYERS_ALL_SEASON_FILE = "players_all_seasons.csv"
SHOTS_DATA_FILE = "shots_2025.csv"
TEAMS_DATA_FILE = "teams_2014_2025.csv"

# Costanti varie
CURRENT_SEASON = 2025
BOOST_FACTORS = {
    "sum_xG": 2.5,
    "n_shots": 2.5
}

INPUT = {
    "players": ["kuhn","sanabria","Castellanos","Lautaro", "Christian Pulisic", "Pavlovic", "Orsolini", "barella", "acerbi", "Martín", "Pinamonti", "Dimarco"],
    "teams" : ["como","cremonese","lazio", "Inter", "AC Milan", "ac Milan", "Bologna", "inter", "inter", "genoa", "Sassuolo", "inter"],
    "opponents" : ["juventus","udinese","atalanta","Verona", "Torino", "Fiorentina", "Juventus", "Sassuolo", "Fiorentina", "como", "Lecce", "Bologna"]
}

top_teams = ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Atalanta", "Lazio"]
mid_teams = ["Fiorentina", "Torino", "Bologna", "Sassuolo", "Udinese"]
weak_teams = ["Empoli", "Verona", "Cagliari", "Lecce", "Salernitana", "Frosinone", "Monza", "Genoa", "Sampdoria", "Spezia", "Pisa", "Cremonese", "Benevento"]

map_strength_dict = {
    'top': 3,
    'mid': 2,
    'weak': 1
}
