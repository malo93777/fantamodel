from statistics import LinearRegression
from catboost import CatBoostRegressor
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import config
import numpy as np
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from unidecode import unidecode
import warnings
import model_predict_fantavoto
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")

ROLE_FEATURES = {
        "dif": [
            "gol",
            "xGChain_trend_per90",
            "xGBuildup_per90_weighted",
            "xGBuildup_trend_per90",
            "assists",
            "xA_per90_weighted",
            "time",
        ],

        "cc": [
        "gol",
        "assists",
        "xGBuildup_per90_weighted",
        "xA_per90_weighted",
        "xA_trend_per90",
        "xGChain_per90_weighted",
        "key_passes_per90_weighted",
        "xGBuildup_trend_per90",
        "key_passes_trend_per90",   
        "time",
        ],
        
        "att": [
        "gol",
        "assists",
        "xGChain_trend_per90",
        "xGBuildup_trend_per90",
        "xGBuildup_per90_weighted",
        "xG_per90_weighted",
        "time"
        ]
    }

def build_X_pred(
    player_df: pd.DataFrame,
    fanta_role: str,
    team: str,
    opponent: str,
    model_xg: dict,
    model_goal: dict,
    model_assist: dict,
    selected_features: dict,
    df_orig_goal: pd.DataFrame,
    df_orig_assist: pd.DataFrame,
    df_teams: pd.DataFrame,
    df_teams_curr_season: pd.DataFrame,
    h_a: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    Costruisce X pre-match per un singolo giocatore, usando le feature selezionate
    dal modello per il ruolo e le probabilità pesate di gol e assist.

    - player_df: dataframe storico giocatore (ultime partite)
    - fanta_role: ruolo fantacalcio ("D", "C", "A")
    - team, opponent: squadre
    - model_goal: dizionario con modello gol
    - model_assist: dizionario con modello assist
    - selected_features: dict {ruolo: list(features da usare nel modello)}
    - df_orig_assist, df_teams, df_teams_curr_season: dati di contesto
    - h_a: casa/trasferta
    """
    norm_name = player_df['player_norm'].iloc[0]

     # === PREDIZIONE GOAL ===
    
    features_names_goal = list(model_goal["poiss_reg"].feature_names_)
    if "finishing_form_resid" in features_names_goal:
        features_names_goal.remove("finishing_form_resid")
        

    goal_proba = utils.get_goal_prob(
                model_xg["poisson_regressor_xg"],
                model_goal["poiss_reg"],
                features_names_goal,
                norm_name, team, opponent, df_orig_goal, df_teams,
                df_teams_curr_season, model_goal["lin"], config.ROLE_STATS,
                h_a
        )
    
    if goal_proba is None:
        print(f"⚠️ Impossibile calcolare la probabilità di goal per {norm_name}. Impostata a 0.05")
        goal_proba = 0.05

    # --- probabilità gol pesata per ruolo ---
    goal_impact = utils.compute_feature_role_impact(
        player_df,
        fanta_role,
        config.ROLE_WEIGHTS_GOAL,
        feature_col='goals'
    )

    goal_feature = goal_proba * goal_impact

    # === PREDIZIONE ASSIST ===
    features_names_assist = model_assist["poisson_reg_assist"].feature_names_
    assist_proba = utils.get_assist_prob(
                model_assist["poisson_reg_assist"], features_names_assist,
                norm_name, team, opponent, df_orig_assist, df_teams,
                df_teams_curr_season, h_a
    )
    if assist_proba is None:
        print(f"⚠️ Impossibile calcolare la probabilità di assist per {norm_name}. Impostata a 0.05")
        assist_proba = 0.05

    assist_impact = utils.compute_feature_role_impact(
        player_df,
        fanta_role,
        config.ROLE_WEIGHTS_ASSIST,
        feature_col='assists'
    )
    assist_feature = assist_proba * assist_impact

    # --- inizializza dict con le feature selezionate per ruolo ---
    X_dict = {}

    # calcola tutte le feature production
    prod_features = utils.compute_prod_features(player_df, config.STATS)

    for feat in selected_features:

        if feat == "gol":
            X_dict["gol"] = goal_feature

        elif feat == "assists":
            X_dict["assists"] = assist_feature

        else:
            if feat in prod_features:
                X_dict[feat] = prod_features[feat]
            else:
                X_dict[feat] = 0.0

        X_pred = pd.DataFrame([X_dict])

    if debug:
        print(f"X_pred per {player_df['player_norm'].iloc[0]} ({fanta_role}):")
        print(X_pred)

    return X_pred

def plot_distributions(df_model: pd.DataFrame, y: pd.Series,
                       numeric_features=None,
                       categorical_features=None,
                       ruolo=None):

    # =========================
    # DISTRIBUZIONE TARGET
    # =========================
    plt.figure(figsize=(8, 5))
    sns.histplot(y, bins=20, kde=True)
    plt.title(f"Distribuzione del Target (Voto) - {ruolo}")
    plt.xlabel("Voto")
    plt.ylabel("Frequenza")
    #plt.show()

    # =========================
    # PREPARAZIONE DATI CORR
    # =========================

    df_corr = df_model.copy()

    if numeric_features is None:
        numeric_features = []

    if categorical_features:
        df_cat = pd.get_dummies(df_corr[categorical_features], drop_first=False)
        df_corr = pd.concat([df_corr[numeric_features], df_cat], axis=1)
    else:
        df_corr = df_corr[numeric_features]

    # aggiungo il target per vedere correlazione con il voto
    df_corr["target"] = y

    # =========================
    # MATRICE CORRELAZIONE
    # =========================
    plt.figure(figsize=(10, 8))
    corr_matrix = df_corr.corr()

    print(f"\n=== MATRICE DI CORRELAZIONE - {ruolo} ===")
    print(corr_matrix["target"].sort_values(ascending=False))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0
    )

    plt.title(f"Matrice di Correlazione - {ruolo}")
    #plt.show()
    
def get_features(df: pd.DataFrame, features: list, split_by_role: bool = True):
    """
    Seleziona le feature e il target.
    Rimuove le righe con NaN nelle colonne usate.
    """
    #df = add_team_strength_column(df, 'opponent_team', 'opponent_team_strength')
    #df = add_team_strength_column(df, 'player_team', 'player_team_strength')

    target = 'voto_gds'

    df = df[df['fanta_role'] != 'P']

    #rimuovo tutti i Senza voto
    df = df[df['voto_gds'].notna()]

    #per ora teniamo solo le colonne con season 2025
    #df = df[df['season'] == config.CURRENT_SEASON] 

    if split_by_role:

        df_dif = df[df['fanta_role'] == 'D']
        df_cc = df[df['fanta_role'] == 'C']
        df_att = df[df['fanta_role'] == 'A']

        df_dif = df_dif[features + [target]].dropna()
        df_cc = df_cc[features + [target]].dropna() 
        df_att = df_att[features + [target]].dropna()

        X_dif = df_dif[features]
        X_cc = df_cc[features]
        X_att = df_att[features]
        y_dif = df_dif[target]
        y_cc = df_cc[target]
        y_att = df_att[target]

        return X_dif, X_cc, X_att, y_dif, y_cc, y_att
    else:
        df = df[features + [target]].dropna()
        X = df[features]
        y = df[target]

        return X, y

def train_catboost_regression(X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
    """Allena un modello di regressione CatBoost e stampa MAE e MSE"""

    # Suddividi in train e test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.02,
        depth=10,
        eval_metric='MAE',
        random_seed=42,
        l2_leaf_reg=10,
        verbose=False ,
        loss_function="RMSE"
    )

    summary = model.select_features(
        X_train,
        y_train,
        features_for_select=X_train.columns.tolist(),
        num_features_to_select=8,
        algorithm="RecursiveByPredictionValuesChange",
        train_final_model=True
    )

    selected_features = summary["selected_features_names"]
    print(selected_features)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # EVALUATION
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
          f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
          f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
    print("REAL")
    print(y_test.describe())

    print("PRED")
    print(pd.Series(y_test_pred).describe())

    print("\n=== FEATURE IMPORTANCE (CATBOOST) ===")
    feature_importance = model.get_feature_importance()
    for name, importance in zip(X_test.columns, feature_importance):
        print(f"{name}: {importance:.4f}")

    return model
def preproc_and_train_log_regression(X: pd.DataFrame, y: pd.Series, numeric_features: list) -> LinearRegression:
    """Allena un modello di regressione lineare e stampa MAE e MSE"""

    # Suddividi in train e test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # PREPROCESSING E MODELLO
    # ===============================

    # ---- preprocessors ----
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )   

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)    
        ]
    )
    # ---- model ----
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=2.0))
        ]
    )
    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
          f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
          f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
    print("REAL")
    print(y_test.describe())

    print("PRED")
    print(pd.Series(y_test_pred).describe())
    # recupera nomi delle feature dopo il preprocessing
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # recupera coefficienti della Ridge
    coefs = model.named_steps["regressor"].coef_

    # dataframe ordinato per importanza assoluta
    feat_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs)
        })
        .sort_values("abs_coefficient", ascending=False)
    )

    print("\n=== FEATURE IMPORTANCE (RIDGE) ===")
    print(feat_importance.head(20))
    
    return model

def pred_voto_prod(
        players,
        teams,
        opponents,
        h_a_players,
        player_df,                
        df_orig_goal,
        df_orig_assist,
        df_teams,
        df_teams_curr_season,
        model_goal,
        model_assist,
        model_xg,
        pipeline,
        #ROLE_FEATURES,
        debug=True
    ):

    predictions = []

    for player, team, opponent, h_a in zip(players, teams, opponents, h_a_players):

        #player_df, player_full_name = model_predict_fantavoto.get_player_data(df_voti, player)
        #if player_df.empty:
            #continue
        
        #caso in cui giocatore ha cambiato squadra durante la stagione
        if isinstance(team, str) and "," in team:
            team = team.split(",")[-1].strip()

        player_df = player_df.sort_values('date')

        fanta_role = utils.get_main_position_weighted(player_df["fanta_role"], window=10, decay=0.8)
        
        # ---- rolling stats ultime 15 ----
        rolling_15 = player_df.tail(15)

        # aggiustamento in base alla forza dell'avversario
        opponent = utils.normalize_team_name(opponent)
        if fanta_role == "D":
            features = ROLE_FEATURES["dif"]
            model = pipeline["dif"]
        elif fanta_role == "C":
            features = ROLE_FEATURES["cc"]
            model = pipeline["cc"]
        elif fanta_role == "A":
            features = ROLE_FEATURES["att"]
            model = pipeline["att"]
            
        X_pred = build_X_pred(
            player_df=rolling_15, fanta_role=fanta_role, team=team, opponent=opponent,
            model_xg=model_xg, model_goal=model_goal, model_assist=model_assist,
            selected_features=features, #model.feature_names_ = features prendo le feature usate nel modello, che sono quelle selezionate per ruolo
            df_orig_goal=df_orig_goal,
            df_orig_assist=df_orig_assist,
            df_teams=df_teams,
            df_teams_curr_season=df_teams_curr_season,
            h_a=h_a, 
            debug=False
        )

        # --- Gestione categoriche e standardizzazione come nel training ---
        if model is not None:
           
            voto_pred = model.predict(X_pred)[0]
        if debug:
            print(f"Predicted voto for {player_df['player'].iloc[0]}, role: {fanta_role}, ({team} vs {opponent}, {h_a}): {voto_pred:.2f}")    

        ha_to_print = "casa" if h_a == "h" else "trasf."

        predictions.append({
        'Giocatore': player_df['player'].iloc[0],
        'Voto': round(voto_pred, 2),
        'Squadra': team,
        'Avversario': opponent,
        'Campo': ha_to_print
        }) 

    return pd.DataFrame(predictions).reset_index(drop=True)

def main():

    train = True

    test = False

    split_by_role = True

    csv_path = config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI

    # carico tutti i df per le probabilità gol/assist e dati squadre
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

    # Carica dataset e modelli 
    model_goal = utils.load_models() 
    model_assist = utils.load_models_assist() 
    model_xg = utils.load_xg_model()

    df_voti = pd.read_csv(csv_path)

    numeric_features = [
        'gol',
        'xG',
        'assists',
        'xA',
        'shots',
        'key_passes',
        'xGBuildup',
        #'ammonizioni',
        #'finishing_form',
        'time'
    ]

    #qua devo fare anche postprocessing per categrocihe team_strength
    df_voti, new_features_per90 = utils.build_player_features_weighted(df_voti, config.STATS)

    # Poi puoi usarla nelle feature del modello
    model_features = new_features_per90 + ['gol', 'assists', 'time']

    if split_by_role:
        X_dif, X_cc, X_att, y_dif, y_cc, y_att = get_features(df_voti, model_features) #aggiungo anche ammonizioni e time che sono importanti per il voto
    else:
        X, y = get_features(df_voti, model_features, split_by_role=split_by_role)
    #plot_distributions(X_dif,y_dif,numeric_features=new_features_per90+ ['gol', 'assists', 'time'],
                       #categorical_features=None,ruolo="Difensori"
    #)
    #plot_distributions(X_cc,y_cc,numeric_features=new_features_per90+ ['gol', 'assists','time'],
                       #categorical_features=None,ruolo="Centrocampisti"
    #)
    #plot_distributions(X_att,y_att,numeric_features=new_features_per90+ ['gol', 'assists','time'],
                       #categorical_features=None,ruolo="Attaccanti"
    #)
    
    if train and split_by_role:

        models = {
            "dif": (X_dif, y_dif),
            "cc": (X_cc, y_cc),
            "att": (X_att, y_att)
        }

        for ruolo, (X_role, y_role) in models.items():

            print(f"\n🚀 Training modello {ruolo.upper()}")

            if ruolo == "dif":
                features_role = ROLE_FEATURES["dif"]
            elif ruolo == "cc":
                features_role = ROLE_FEATURES["cc"]
            elif ruolo == "att":
                features_role = ROLE_FEATURES["att"]

            pipeline = preproc_and_train_log_regression(X_role, y_role,features_role)
            #pipeline = train_catboost_regression(X_role, y_role)

            MODEL_PATH = config.MODEL_DIR_FV / f"{ruolo}_{config.VOTO_MODEL}"

            if MODEL_PATH.exists():
                overwrite = input(
                    f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): "
                ).strip().lower()

                if overwrite != "y":
                    print("❌ Salvataggio modello annullato.")
                    continue

            joblib.dump(pipeline, MODEL_PATH)
            print(f"✅ Modello salvato in: {MODEL_PATH}")
    elif not train and split_by_role:
        #carica lista modelli per ruolo

        pipeline_dif = utils.load_voto_model("dif")
        pipeline_cc = utils.load_voto_model("cc")
        pipeline_att = utils.load_voto_model("att")

        pipeline = {
            "dif": pipeline_dif["voto_model"],
            "cc": pipeline_cc["voto_model"],
            "att": pipeline_att["voto_model"]
        }
    elif train and not split_by_role:
        pipeline = preproc_and_train_log_regression(X, y,model_features)
            #pipeline = train_catboost_regression(X_role, y_role) SERVONO PIU DATI

        MODEL_PATH = config.MODEL_DIR_FV / f"{ruolo}_{config.VOTO_MODEL}"

        if MODEL_PATH.exists():
                overwrite = input(
                    f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): "
                ).strip().lower()

        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")         

        joblib.dump(pipeline, MODEL_PATH)
        print(f"✅ Modello salvato in: {MODEL_PATH}")

    if test:

        pred_df = pred_voto_prod(config.INPUT["players"],config.INPUT["teams"],config.INPUT["opponents"],config.INPUT["h_a"],
                                 df_voti,df_orig_goal, df_orig_assist, df_teams, df_teams_curr_season,
                                 model_goal, model_assist, model_xg, pipeline, ROLE_FEATURES)
        
        print("\n=== PREDIZIONI VOTO ===")
        print(pred_df)


  
if __name__ == "__main__":
    main()