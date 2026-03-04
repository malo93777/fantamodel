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
MODEL_DIR_FV = BASE_DIR / "models/fantavoto"


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
VOTI_DATA_FILE = "voti_fantagiaveno_raw.csv"
PROD_DATA_FILE_VOTI = "PROD_voti_2025_preproc_Serie_A.csv"
FANTA_RUOLI_FILE = "ruoli_fanta_25.csv"
NEXT_GAMES_FILE = "next_games.csv"
INFORTUNATI_FILE = "infortunati.csv"
SQUALIFICATI_FILE = "squalificati.csv"

# Modelli
SCALER = "scaler.pkl"
SCALER_XG = "scaler_xg.pkl"
LIN_POLY = "lin_poly.pkl"
LIN = "lin.pkl"
POISS_MODEL = "poisson_regressor.pkl"
POISS_MODEL_ASSIST = "poisson_regressor_assist.pkl"
POISS_MODEL_XG = "poisson_regressor_xg.pkl"
FV_MODEL = "fantavoto_model.pkl"
FV_MODEL_GK = "fantavoto_model_gk.pkl"

# Costanti varie
CURRENT_SEASON = 2025

#

NEXT_GIORNATA = 28

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
    "players": ["gutierrez","pavlovic","nkunku","berardi","berardi","scamacca",'lautaro','doig','miranda','pavlovic',
        "cancellieri","david","odgaard","scamacca","colombo","gudmundsson", "kean","buksa","buksa","orban",
        "yildiz", "yildiz","di lorenzo", "conceicao","simeone","vlahovic",
        "paz","paz","sanabria","Castellanos","Lautaro", "Leao", "Pavlovic",
        "Orsolini", "barella", "acerbi", "Martin", "Berardi", "Dimarco",
        "guendouzi", "loftus", "giovane", "giovane", "soule", "pinamonti",
        "gimenez", "bonny", "doig", "krstovic"
    ],
    "teams" : [
        "milan","napoli","milan","sassuolo","sassuolo","atalanta","inter","sassuolo","bologna","milan",
        "lazio","juventus","bologna","atalanta","genoa","fiorentina","fiorentina", "udinese","udinese","verona",
        "juventus","juventus","napoli", "juventus","torino","juventus","como",
        "como","cremonese","lazio", "Inter", "AC Milan", "ac Milan", "Bologna",
        "inter", "inter", "genoa", "Sassuolo", "inter", "lazio", "ac milan",
        "verona", "verona", "roma", "sassuolo", "milan", "inter", "sassuolo", "atalanta"
    ],
    "opponents" : [
        "cremonese","verona","como","lecce","lecce","cremonese","pisa","cremonese","genoa","roma","cremonese","torino","lazio","fiorentina","juventus","atalanta", "milan","sassuolo","cagliari", "napoli",
        "udinese","ac milan","juventus","juventus","pisa","cagliari", "juventus",
        "lecce","udinese","atalanta","AC MILAN", "Lazio", "Juventus", "genoa",
        "Sassuolo", "Fiorentina", "Bologna", "Fiorentina", "Bologna", "atalanta",
        "fiorentina", "pisa", "ac milan", "pisa","roma","pisa", "napoli", "roma","roma"
    ],
    "h_a": [
        "a","a","h","a","h","h","a","a","a","h","a","h","a","h","a","a",
        "h","a","h","a","h","h",
        "a","h","a","a","h","h","a",
        "a","h","a","h","a","h",
        "a","h","h","a","h","a",
        "h","a","a","h"
    ]
}

TOP_TEAMS = {
        'inter', 'juventus', 'milan', 'ac milan', 'napoli', 'atalanta'
    }

MID_TEAMS = {
        'roma', 'lazio', 'fiorentina', 'bologna', 'sassuolo', 'como'
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

# ************   FANTAVOTO MODEL CONFIG   ***************

ROLE_FANTAVOTO_STATS = {
    'A': {"mean":6.48, "std":2.08},
    'C': {"mean":6.27, "std":1.49},
    'D': {"mean":6.00, "std":1.16},
    'P': {"mean":5.45, "std":1.91},
    'SUB': {"mean":6.09, "std":1.38}
}

ROLE_WEIGHTS_GOAL = {
        'A': {"mean":0.18, "std":0.45},
        'C': {"mean":0.09, "std":0.31},
        'D': {"mean":0.04, "std":0.19},
        'P': {"mean":0.00, "std":0.08},
        'SUB': {"mean":0.04, "std":0.2}
}

ROLE_WEIGHTS_ASSIST = {
        'A': {"mean":0.09, "std":0.32},
        'C': {"mean":0.07, "std":0.27},
        'D': {"mean":0.04, "std":0.21},
        'P': {"mean":0.00, "std":0.00},
        'SUB': {"mean":0.03, "std":0.18}
}

ROLE_WEIGHTS_INDEX = {
    'A': 0.85,
    'C': 0.75,
    'D': 0.65,
    'P': 0.56,
    'SUB': 0.40
}

#quantitativiper peso forma recente (last 5)
ROLE_WEIGHTS_VOTO = {
    'A': 0.8,
    'C': 0.7,
    'D': 0.7,
    'P': 0.7,
    'SUB': 0.7
}

AMMONIZIONI_MEAN_FANTAROLE = {
    "A":     {"mean": 0.10, "std":0.30} ,
    "C" :     {"mean": 0.15, "std":0.35},
    "D" :     {"mean": 0.16, "std":0.36},
    "P":      {"mean": 0.06, "std":0.24},
    "SUB":    {"mean": 0.09, "std":0.3},
}

# Lista dei nomi e ruoli che mi hai dato
manual_roles = {
    "keinan davis": "A",
    "nico paz": "C",
    "gift orban": "A",
    "santiago castro": "A",
    "mateo pellegrino": "A",
    "franck zambo": "C",
    "albert gudmundsson": "C",
    "che adams": "A",
    "sebastiano esposito": "A",
    "ismael kone": "C",
    "lassana coulibaly": "C",
    "matteo guendouzi": "C",
    "ruslan malinovskiy": "C",
    "kamaldeen sulemana": "A",
    "medon berisha": "C",
    "vitinha": "A",
    "pio esposito": "A",
    "alberto moreno": "A",
    "pedro": "A",
    "duvan zapata": "A",
    "enrico del prato": "D",
    "luca ranieri": "D",
    "youssouf fofana": "C",
    "juan miranda": "D",
    "samuele ricci": "C",
    "lloyd kelly": "D",
    "nikola moro": "A",
    "iker bravo": "A",
    "thomas thiesson kristensen": "D",
    "riyad idrissi": "C",
    "idrissa toure": "C",
    "oliver sorensen": "C",
    "petar sucic": "C",
    "giuseppe pezzella": "D",
    "federico ravaglia": "P",
    "berat gjimshiti": "D",
    "benjamin pavard": "D",
    "hassane kamara": "D",
    "luca pellegrini": "D",
    "marius marin": "C",
    "hernani": "C",
    "marco silvestri": "P",
    "antonio caracciolo": "A",
    "vanja milinkovicsavic": "P",
    "samuel chukwueze": "C",
    "pablo mari": "D",
    "nuno tavares": "D",
    "armel bella kotchap": "D",
    "mattia viti": "D",
    "jeremy sarmiento": "C",
    "romano floriani": "C",
    "zito": "A",
    "ederson": "C",
    "jesus santiago": "C",
    "alex sala": "C",
    "sulemana": "A",
    "ali dembele": "A",
    "amin sarr": "A",
    "alejandro jimenez": "D",
    "matteo palma": "D",
    "woyo coulibaly": "D",
    "kialonda gaspar": "D",
    "jesus rodriguez": "A",
    "benjamin dominguez": "A",
    "mandela keita": "C",
    "lautaro martinez": "A",
    "david neres": "C",
    "jean butez": "P",
    "valentin castellanos": "A",
    "matteo tramoni": "C",
    "lorenzo pellegrini": "C",
    "evan ferguson": "A",
    "franco vazquez": "C",
    "davide bartesaghi": "D",
    "kouadio kone":"C",
    "kephren thuram": "C",
    "david de gea": "P",
    "francesco coppola": "D",
    "juan rodriguez": "D",
    "robinio vaz": "A",
    "danilo veiga": "D",
    "al musrati": "C",
    "ivan smolcic": "D",
    "matias perez": "D",
    "luca moro": "A",
    "gabriele piccinini": "C",
    "christian ordonez": "C",
    "vakoun bayo": "A",
    "lennon miller": "C",
    "kenneth taylor": "C",
    "justin bijlow": "P",
    "pierre kyatengwa": "D",
    "hans caviglia": "C",
    "thomas kristensen": "D",
    "romano floriani": "C",
    "romano floriani mussolini": "C",
    "josep martinez":"P",
    "filippo terracciano":"D",
    "pietro terracciano": "P",
    "alberto moreno": "D",
    "davide zappacosta": "D",
    "davide frattesi":"C",
    "alisson santos":"C"
}

PREFIXES = {'de','da','di','del','do','van','von','der','le','la','el','al','du','ze'}

GONE_PLAYERS_NOT_TOP5_LEAGUES = {"noa lang", "edin dzeko", "matteo guendouzi","dennis johnsen", "albert gronbaek"}

POSTPONED_TEAMS_2025 = {
    16: {
        "teams": {
            "como", "milan", "inter", "lecce",
            "napoli", "parma", "verona", "bologna"
        },
        "match_order": 20.5
    },
    24: {
        "teams": {"milan", "como"},
        "match_order": 25.5
    }
}

POSTPONED_MATCHES = {
        20.5: {
            "como":     ("milan",   "como - milan"),
            "milan":    ("como",    "como - milan"),
            "inter":    ("lecce",   "inter - lecce"),
            "lecce":    ("inter",   "inter - lecce"),
            "napoli":   ("parma",   "napoli - parma"),
            "parma":    ("napoli",  "napoli - parma"),
            "verona":   ("bologna", "verona - bologna"),
            "bologna":  ("verona",  "verona - bologna"),
        },
        25.5: {
            "como":  ("milan", "milan - como"),
            "milan": ("como",  "milan - como"),
        }
}