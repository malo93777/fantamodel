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
MODEL_DIR_XG = BASE_DIR / "models/xg"


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
SCALER = "scaler.pkl"
SCALER_XG = "scaler_xg.pkl"
LIN_POLY = "lin_poly.pkl"
LIN = "lin.pkl"
POISS_MODEL = "poisson_regressor.pkl"
POISS_MODEL_ASSIST = "poisson_regressor_assist.pkl"
POISS_MODEL_XG = "poisson_regressor_xg.pkl"

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
    "players": [
        "cancellieri","david","odgaard","scamacca","colombo","gudmundsson", "kean","buksa","buksa","orban",
        "yildiz", "yildiz","di lorenzo", "conceicao","simeone","vlahovic",
        "paz","paz","sanabria","Castellanos","Lautaro", "Leao", "Pavlovic",
        "Orsolini", "barella", "acerbi", "Martin", "Berardi", "Dimarco",
        "guendouzi", "loftus", "giovane", "giovane", "soule", "pinamonti",
        "gimenez", "bonny", "doig", "krstovic"
    ],
    "teams" : [
        "lazio","juventus","bologna","atalanta","genoa","fiorentina","fiorentina", "udinese","udinese","verona",
        "juventus","juventus","napoli", "juventus","torino","juventus","como",
        "como","cremonese","lazio", "Inter", "AC Milan", "ac Milan", "Bologna",
        "inter", "inter", "genoa", "Sassuolo", "inter", "lazio", "ac milan",
        "verona", "verona", "roma", "sassuolo", "milan", "inter", "sassuolo", "atalanta"
    ],
    "opponents" : [
        "cremonese","torino","lazio","fiorentina","juventus","atalanta", "milan","sassuolo","cagliari", "napoli",
        "udinese","ac milan","cagliari","juventus","pisa","cagliari", "juventus",
        "lecce","udinese","atalanta","AC MILAN", "Lazio", "Juventus", "genoa",
        "Sassuolo", "Fiorentina", "Bologna", "Fiorentina", "Bologna", "atalanta",
        "fiorentina", "pisa", "ac milan", "pisa","roma","pisa", "napoli", "roma","roma"
    ],
    "h_a": [
        "a","h","a","h","a","h","a","a",
        "h","a","h","a","h","h",
        "a","h","a","a","h","h","a",
        "a","h","a","h","a","h",
        "a","h","h","a","h","a",
        "h","a","a","h"
    ]
}


ROLE_STATS = {
    "global_overperf_median": -0.000001,
    
    "role_overperf_medians": {
        "D":    -0.000001,
        "DF":   -0.000001,
        "DM":   -0.000001,
        "F":    -0.03560790906571769,
        "FM":   -0.028906161706381748,
        "M":    -0.000001,
        "None": -0.000001,
    },

    "default_role_median": -0.000001,

    "shots_divisor": 21.5
}

SERIE_A_TEAMS = ["cremonese","spal", "pescara","crotone","brescia","cesena", "benevento", "carpi", "venezia", "pisa","palermo", "Parma","Como", "Milan", "Inter", "Juventus", "Roma", "Napoli", "Lazio", "Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Empoli", "Genoa", "Verona", "Lecce", "Udinese", "Monza", "Cagliari", "Frosinone", "Salernitana", "Chievo", "Spezia"]

FEATURES_LR =  ["sum_xG",  
                "xG_last5",  
                "goals_last5",                  
                "finishing_form_resid"                                
               ]

IS_SERIEA = True