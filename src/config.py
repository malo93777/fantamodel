from pathlib import Path

# Base directory del progetto (cartella principale)
BASE_DIR = Path(__file__).resolve().parent.parent

# Percorsi principali
DATASET_DATA_DIR = BASE_DIR / "dataset"
SRC_DIR = BASE_DIR / "src"
MODEL_DIR =  BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scaler"

# File
MATCH_DATA_FILE = "matches_df.csv"
PROD_DATA_FILE = "PROD_shots_2025_preproc_Serie_A.csv"
PLAYERS_ALL_SEASON_FILE = "players_all_seasons.csv"
SHOTS_DATA_FILE = "shots_2025.csv"
TEAMS_DATA_FILE = "teams_2014_2025.csv"
ASSIST_DATA_FILE = "assists_2025.csv"
PROD_DATA_FILE_ASSIST = "PROD_assists_2025_preproc_Serie_A.csv"

# Modelli
CALIB_LOGISTIC_REG = "calibrated_lr.pkl"
SCALER = "scaler.pkl"

# Costanti varie
CURRENT_SEASON = 2025
BOOST_FACTORS = {
    "sum_xG": 1.0,
    "xG_last5": 1.0,
    #"shots_last5": 1.0,
    "goals_last5": 1.0
}

INPUT = {
    "players": ["paz","paz","sanabria","Castellanos","Lautaro", "Leao", "Pavlovic", "Orsolini", "barella", "acerbi", "Martin", "Berardi", "Dimarco", "guendouzi", "loftus", "giovane", "giovane"],
    "teams" : ["como","como","cremonese","lazio", "Inter", "AC Milan", "ac Milan", "Bologna", "inter", "inter", "genoa", "Sassuolo", "inter", "lazio", "ac milan", "verona", "verona"],
    "opponents" : ["juventus","lecce","udinese","atalanta","Verona", "Torino", "Fiorentina", "Juventus", "Sassuolo", "Fiorentina", "como", "Lecce", "Bologna", "atalanta", "fiorentina", "pisa", "ac milan"]
}

top_teams = ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Atalanta", "Lazio"]
mid_teams = ["Fiorentina", "Torino", "Bologna", "Sassuolo", "Udinese"]
weak_teams = ["Empoli", "Verona", "Cagliari", "Lecce", "Salernitana", "Frosinone", "Monza", "Genoa", "Sampdoria", "Spezia", "Pisa", "Cremonese", "Benevento"]

map_strength_dict = {
    'top': 3,
    'mid': 2,
    'weak': 1
}

SERIE_A_TEAMS = ["spal", "pescara","crotone","brescia","cesena", "benevento", "carpi", "venezia", "pisa","palermo", "Parma","Como", "Milan", "Inter", "Juventus", "Roma", "Napoli", "Lazio", "Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Empoli", "Genoa", "Verona", "Lecce", "Udinese", "Monza", "Cagliari", "Frosinone", "Salernitana", "Chievo", "Spezia"]