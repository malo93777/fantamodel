import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config
import utils
import first_preproc as preproc

# ===================== TEAM STRENGTH MAPPING =====================
SERIE_A_TOP = ["Juventus", "Inter", "Milan", "Napoli", "Roma", "Lazio"]
SERIE_A_MID = ["Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Udinese", "Genoa", "Como"]
SERIE_A_WEAK = [s for s in config.SERIE_A_TEAMS if s not in SERIE_A_TOP + SERIE_A_MID]

def get_team_strength(team_name):
    if team_name in SERIE_A_TOP:
        return "top"
    elif team_name in SERIE_A_MID:
        return "mid"
    else:
        return "weak"

def add_opponent_strength_feature(df, opponent_col="opponent_team"):
    df = df.copy()
    df["opponent_strength"] = df[opponent_col].apply(get_team_strength)
    return df

def cat_training(X, y, cat_features):

    # ===================== TRAIN / TEST SPLIT =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    #model, best_params = utils.tune_catboost_regressor(X, y, cat_features)

    # ===================== MODEL =====================
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=8,
        min_data_in_leaf=10,

        loss_function="Poisson",
        random_seed=SEED,
        cat_features=cat_features,
        early_stopping_rounds=100,

        verbose=100
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    # ===================== FIT =====================
    model.fit(train_pool, eval_set=test_pool)

    print("\nBest iteration:", model.get_best_iteration())
    importance = model.get_feature_importance(prettified=True)
    print(importance.head(10))

    # ===================== TRAIN vs TEST METRICS =====================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
        f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
        f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")

    #spearman correlation
    from scipy.stats import spearmanr

    print(
        "Spearman TEST:",
        spearmanr(y_test, y_test_pred).correlation
    )

    # ===================== LEARNING CURVE =====================
    evals = model.get_evals_result()

    # ===================== ERROR ANALYSIS =====================
    compare_df = X_test.copy()
    compare_df["y_true"] = y_test
    compare_df["y_pred"] = y_test_pred
    compare_df["error"] = compare_df["y_pred"] - compare_df["y_true"]

    print("\n=== ERRORE PER RUOLO ===")
    for role, g in compare_df.groupby("position"):
        mae = np.abs(g["error"]).mean()
        mse = mean_squared_error(g["y_true"], g["y_pred"])
        print(f"{role:>8}: MAE={mae:.4f}  MSE={mse:.4f}  N={len(g)}")

    return model

def ridge_training(X, y, numeric_features, cat_features):

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop='first'), cat_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=SEED)),
    ])

    # ===================== TRAIN / TEST SPLIT =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)


    # Feature importance: ottieni nomi numerici e categorici
    feature_importance = pipeline.named_steps['regressor'].coef_
    preprocessor = pipeline.named_steps['preprocessor']
    num_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    feature_names = list(num_names) + list(cat_names)

    print("\n=== FEATURE IMPORTANCE ===")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")

    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
        f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
        f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
     #spearman correlation
    from scipy.stats import spearmanr

    print(
        "Spearman TEST:",
        spearmanr(y_test, y_test_pred).correlation
    )
    # ===================== ERROR ANALYSIS =====================
    compare_df = X_test.copy()
    compare_df["y_true"] = y_test
    compare_df["y_pred"] = y_test_pred
    compare_df["error"] = compare_df["y_pred"] - compare_df["y_true"]

    print("\n=== ERRORE PER RUOLO ===")

    for role, g in compare_df.groupby("position"):
        mae = np.abs(g["error"]).mean()
        mse = mean_squared_error(g["y_true"], g["y_pred"])
        print(f"{role:>8}: MAE={mae:.4f}  MSE={mse:.4f}  N={len(g)}")

    return pipeline

# ===================== CONFIG =====================
SEED = 42
TARGET = "sum_xG"

numeric_features = ["sum_xG", 
            "shots_perMatch", "minutes_played", 
            #"xg_per_shot",
              "goals", "finishing_form", "npgoals_perMatch", "xGChain_perMatch"]

cat_features = [
    "position",
    "opponent_team_strength",
]

preprocessor = preproc.Preprocessor(
    serie_a_teams=config.SERIE_A_TEAMS
)

# ===================== LOAD & PREPROCESS =====================
df = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
df = df[~df["position"].isin(["GK", "GKS"])]

stats = utils.compute_role_overperf_stats(df)
df = utils.add_overperformance_features(df, stats, player_col="player", prod=False)
df, ewm_cols = utils.add_ewma_features(df, span=7, numeric_features=numeric_features, prod=False)

#df["xg_per90"] = df["sum_xG_ewm_7"] / (df["minutes_played_ewm_7"] + 1e-6) * 90
#df["shots_sqrt"] = np.sqrt(df["shots_perMatch_ewm_7"])

df = df.sort_values(["player", "date"])
df["position"] = df["position"].apply(utils.clean_position)

df = df.dropna(subset=["position"])

# ===================== FEATURES / TARGET =====================
X = df[ewm_cols + cat_features]
y = df[TARGET]

model =ridge_training(X, y, ewm_cols, cat_features)
#model = cat_training(X, y, cat_features)

MODEL_PATH = config.MODEL_DIR_XG / "catboost_regressor_xg.pkl"
#chiedi all'utente se vuole salvare il modello
if MODEL_PATH.exists():
        overwrite = input(f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")
        else:
            joblib.dump(model, MODEL_PATH)
            print(f"✅ Modello sovrascritto in: {MODEL_PATH}")
else:
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modello salvato in: {MODEL_PATH}")