from pathlib import Path

# Base directory del progetto (cartella principale)
BASE_DIR = Path(__file__).resolve().parent.parent

# Percorsi principali
DATASET_DATA_DIR = BASE_DIR / "dataset"
SRC_DIR = BASE_DIR / "src"
MODEL_DIR =  BASE_DIR / "models/goal"
SCALER_DIR = BASE_DIR / "scaler/goal"
MODEL_DIR_ASSIST =  BASE_DIR / "models/assist"
SCALER_DIR_ASSIST = BASE_DIR / "scaler/assist"


# File
MATCH_DATA_FILE = "matches_df.csv"
PROD_DATA_FILE_GOALS = "PROD_goals_2025_preproc_Serie_A.csv"
PLAYERS_ALL_SEASON_FILE = "players_all_seasons.csv"
GOALS_DATA_FILE = "goals_2025.csv"
TEAMS_DATA_FILE = "teams_2014_2025.csv"
ASSIST_DATA_FILE = "assists_2025.csv"
PROD_DATA_FILE_ASSIST = "PROD_assists_2025_preproc_Serie_A.csv"
RAW_DATA_FILE = "raw_data.csv"
CURRENT_SEASON_TEAMS_FILE = "teams_current_season.csv"
GOALS_DATA_FILE_ALL_LEAGUES = "all_leagues_goals.csv"

# Modelli
CALIB_LOGISTIC_REG = "calibrated_lr.pkl"
SCALER = "scaler.pkl"
SCALER_XG = "scaler_xg.pkl"
POLY_TRANSFORMER = "poly.pkl"
LIN_POLY = "lin_poly.pkl"
LIN = "lin.pkl"

POISS_MODEL = "poisson_regressor.pkl"

# Costanti varie
CURRENT_SEASON = 2025
BOOST_FACTORS = {
    "sum_xG": 1.0,
    "xG_last5": 1.0,
    "goals_last5": 1.5,
}

BOOST_FACTORS_XGB = {
    "sum_xG": 1.0,
    #"xG_last5": 1.0,
    #"goals_last5": 1.0,
}

BOOST_RESID = 1.0

INPUT = {
    "players": ["gudmundsson", "kean","buksa","buksa","orban" , "yildiz", "yildiz","di lorenzo", "conceicao","simeone","vlahovic","paz","paz","sanabria","Castellanos","Lautaro", "Leao", "Pavlovic", "Orsolini", "barella", "acerbi", "Martin", "Berardi", "Dimarco", "guendouzi", "loftus", "giovane", "giovane", "soule", "pinamonti", "gimenez", "bonny", "doig", "krstovic"],
    "teams" : ["fiorentina","fiorentina", "udinese","udinese","verona", "juventus","juventus","napoli", "juventus","torino","juventus","como","como","cremonese","lazio", "Inter", "AC Milan", "ac Milan", "Bologna", "inter", "inter", "genoa", "Sassuolo", "inter", "lazio", "ac milan", "verona", "verona", "roma", "sassuolo", "milan", "inter", "sassuolo", "atalanta"],
    "opponents" : ["cagliari", "atalanta","ac milan","cagliari", "cagliari", "sassuolo","ac milan","cagliari","juventus","pisa","como", "juventus","lecce","udinese","atalanta","Sassuolo", "Torino", "udinese", "Juventus", "Sassuolo", "Fiorentina", "Bologna", "Lecce", "Bologna", "atalanta", "fiorentina", "pisa", "ac milan", "lazio","roma","pisa", "napoli", "roma","sassuolo"]
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

FEATURES_LR =  ["sum_xG",  
                "xG_last5",  
                "goals_last5",                  
                "finishing_form_resid"                                
               ]

IS_SERIEA = True