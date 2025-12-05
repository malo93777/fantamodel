import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score, brier_score_loss
)

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import CURRENT_SEASON_TEAMS_FILE, GOALS_DATA_FILE_ALL_LEAGUES, DATASET_DATA_DIR, PROD_DATA_FILE_GOALS, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_RESID, BOOST_FACTORS_XGB, INPUT, MODEL_DIR, SCALER_DIR, CALIB_LOGISTIC_REG, SCALER, SERIE_A_TEAMS
from first_preproc import Preprocessor
from unidecode import unidecode
# ============================================================
# 1️⃣ FEATURE ENGINEERING PULITO
# ============================================================
def prepare_features(df_orig):

    df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
    df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)

    df = df_orig.copy()
    df = df.sort_values(["player", "date"])

    df = df[df["position"] != "GK"]
    df = df[df["position"] != "GKS"]

    # applica al dataset
    df["position"] = df["position"].apply(utils.clean_position)

    # controlla i valori unici
    print(df["position"].unique())
    df["position"]= df["position"].dropna()
    # Conta le occorrenze

    print(df.shape)
    #df = df[df["minutes_played"] >= 5]
    #print(df.shape)

    stats = utils.compute_role_overperf_stats(df)
    df = utils.add_overperformance_features(df, stats, player_col="player", prod=False)

    df = utils.compute_shot_quality_index(df,prod=False)

    print(df[["player", "overperf_log", "overperf_last5", "overperf_combined"]].tail())

    numeric_features = [
        "sum_xG", 
        "finishing_form",
        "overperf_role_resid",
        "shot_quality_index"
    ]

    #df = df.dropna(subset=cols_to_check)
    df[numeric_features] = df[numeric_features].fillna(0)

    #****** CONTROLLI STATISTICI *******
    #utils.analyze_feature_skewness(df, cols_to_check)
    utils.multicoll_check(df, numeric_features)
 
    #trasf log sum_xG per ridurre skewness
    df["sum_xG"] = np.log1p(df["sum_xG"])
    df["xG_last5"] = np.log1p(df["xG_last5"])
    # Seleziona le features (X) e target (y)

    # 1) Media mobile xG
    df["xg_mean_12"] = (
        df.groupby("player")["sum_xG"]
        .rolling(10, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    # 2) Residuo finishing
    lin_reg = LinearRegression()

    # Fit su tutto il dataset
    lin_reg.fit(df[["xg_mean_12"]], df["finishing_form"])

    #Predizione lienare
    pred_lin = lin_reg.predict(df[["xg_mean_12"]])

    # Calcolo residuo tramite Predizione lineare
    df["finishing_form_resid"] = df["finishing_form"] - pred_lin

    print("Media finishing_form:", df["finishing_form"].mean())
    print("Media pred:", pred_lin.mean())
    print("Media residuo:", df["finishing_form_resid"].mean())

    #plt.scatter(pred_lin, X["finishing_form_resid"], alpha=0.6)
    #plt.axhline(0, color='r')
    #plt.xlabel("Predizione regressione")
    #plt.ylabel("Residuo (finishing_form - pred_poly)")
    #plt.show()

    numeric_features.remove("finishing_form")
    numeric_features.append("finishing_form_resid")
    categorical_features = ["position"]

    #categorical_features = list(pos_dummies.columns)

    # Costruisci X finale
    # Aggiungi le dummy di posizione
    #X = pd.concat([X[numeric_features].reset_index(drop=True), df[categorical_features].reset_index(drop=True)], axis=1)

    vif_df = pd.DataFrame({
        "feature": numeric_features,
        "VIF": [variance_inflation_factor(df[numeric_features].values, i) for i in range(len(numeric_features))]
    })
    print(vif_df)

    return df, stats, lin_reg, numeric_features, categorical_features

def get_player_data(df: pd.DataFrame, player_name: str):
    """
    Cerca i dati di un giocatore nel dataframe, gestendo:
    - accenti (Martínez -> Martinez)
    - case-insensitivity
    - match esatto o parola intera
    - ambiguità se esistono più giocatori con lo stesso nome
    """

    # Normalizza i nomi
    df = df.copy()
    df["player_norm"] = df["player"].apply(lambda x: unidecode(str(x)).lower())
    player_norm = unidecode(player_name).lower()

    # 1️⃣ Match esatto
    player_df = df[df["player_norm"] == player_norm]

    # 2️⃣ Se non trovato, prova con parola intera (regex)
    if player_df.empty:
        player_df = df[df["player_norm"].str.contains(rf"\b{re.escape(player_norm)}\b", case=False, na=False)]

    # 3️⃣ Se ancora vuoto
    if player_df.empty:
        print(f"⚠️ Nessun giocatore trovato per '{player_name}'.")
        return pd.DataFrame()

    # 4️⃣ Se più giocatori hanno lo stesso nome
    matching_players = player_df["player"].unique()
    if len(matching_players) > 1:
        print(f"⚠️ Trovati più giocatori con nome simile a '{player_name}':")
        for i, p in enumerate(matching_players, 1):
            teams = ", ".join(df[df["player"] == p]["player_team"].dropna().unique())
            print(f"   {i}. {p} ({teams})")

        # Chiede all'utente quale scegliere
        try:
            choice = int(input("👉 Inserisci il numero del giocatore desiderato: ")) - 1
            chosen_player = matching_players[choice]
            player_df = df[df["player"] == chosen_player]
        except (ValueError, IndexError):
            print("❌ Scelta non valida, interrotto.")
            return pd.DataFrame()

    return player_df.sort_values("date").reset_index(drop=True)

# ============================================================
# 2️⃣ TRAINING CATBOOST POISSON
# ============================================================
def train_poisson_model(X_train, y_train, cat_features):

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weights = np.where(y_train == 1, pos_weight, 1)

    model = CatBoostRegressor(
        depth=9,
        iterations=1000,
        learning_rate=0.01,
        l2_leaf_reg=10,
        random_strength=2.0,
        bagging_temperature=0,
        min_data_in_leaf=15,
        bootstrap_type="Bayesian",
        loss_function="Poisson",
        verbose=False,
        random_seed=42,
        cat_features=cat_features
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


# ============================================================
# 3️⃣ CALIBRAZIONE α PER RUOLO
# ============================================================
def calibrate_role_alpha(model, X_val, y_val):

    role_alphas = {}

    for role in X_val["position"].unique():
        mask = (X_val["position"] == role)

        lam = model.predict(X_val[mask])
        y = y_val[mask]

        best_a, best_brier = None, 1e9

        for a in np.linspace(0.3, 1.0, 40):
            p = 1 - np.exp(-np.clip(a * lam, 0, None))
            b = brier_score_loss(y, p)

            if b < best_brier:
                best_brier = b
                best_a = a

        role_alphas[role] = best_a

    return role_alphas


# ============================================================
# 4️⃣ CONVERSIONE LAMBDA → PROBABILITÀ
# ============================================================
def predict_probability(model, X, role_alphas):

    lam = model.predict(X)

    a = X["position"].map(role_alphas).fillna(0.7)

    p = 1 - np.exp(-np.clip(a * lam, 0, None))

    return p


# ============================================================
# 5️⃣ METRICHE + THRESHOLD
# ============================================================
def evaluate_model(y_true, y_prob, threshold=0.5):

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier
    }


# ============================================================
# 6️⃣ PIPELINE COMPLETA
# ============================================================
def full_training_pipeline(df):

    df,stats, lin_reg, numeric_features, categorical_features = prepare_features(df)

    # Target multiclasse (0,1,2,3+)
    df["goal_class"] = df["is_goals"].clip(0, 3)

    # Binarizzazione per il modello Poisson
    df["binary_goal"] = (df["goal_class"] > 0).astype(int)

    features = numeric_features + categorical_features

    X = df[features]
    y = df["binary_goal"]

    cat_features = ["position"]

    # Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.3, random_state=42, stratify=y_train_full
    )

    # Train
    model = train_poisson_model(X_train, y_train, cat_features)

    # α per ruolo
    role_alphas = calibrate_role_alpha(model, X_val, y_val)

    # Predict
    y_test_prob = predict_probability(model, X_test, role_alphas)

    # Trova threshold ottimale
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Metriche
    metrics = evaluate_model(y_test, y_test_prob, threshold=best_threshold)

    print(best_threshold)

    # Aggiungi y_true, pred prob e pred label
    X_test["true_goal"] = y_test.values
    X_test["pred_prob"] = y_test_prob
    X_test["pred_label"] = (y_test_prob >= best_threshold).astype(int)

    # Ordina per probabilità discendente
    X_test_sorted = X_test.sort_values(by="pred_prob", ascending=False)

    # Stampa un riepilogo
    print("\n🔍 TOP 30 GIOCATORI CON PROBABILITÀ PIÙ ALTA DI GOL (TEST):")
    print(X_test_sorted.head(30))

    print("\n⚠️ 30 CASI PIÙ SBAGLIATI (Falsi positivi o negativi):")
    wrong_preds = X_test_sorted[X_test_sorted["true_goal"] != X_test_sorted["pred_label"]]
    print(wrong_preds.head(30))

    print("\n📊 30 CASI PIù SBAGLIATI falsi negativi")
    false_negatives = X_test_sorted[(X_test_sorted["pred_label"] == 0) & (X_test_sorted["true_goal"] == 1)]
    print(false_negatives.head(30))

    print("\n📊 30 CASI PIù SBAGLIATI falsi positivi")
    false_positives = X_test_sorted[(X_test_sorted["pred_label"] == 1) & (X_test_sorted["true_goal"] == 0)]
    print(false_positives.head(30))

    # Analisi media per ruolo
    print("\n📊 Probabilità media di gol per ruolo:")
    print(X_test_sorted.groupby("position")["pred_prob"].mean().sort_values(ascending=False))

    return {
        "model": model,
        "lin_reg": lin_reg,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "role_alphas": role_alphas,
        "test_prob": y_test_prob,
        "test_true": y_test,
        "best_threshold": best_threshold,
        "metrics": metrics,
        "stats": stats
    }

def predict_goal_probabilities(players, teams, opponents, df_orig, df_teams, df_teams_curr, model, lin, numeric_features, categorical_features, stats, h_a_player):
    results = []

    df_records = pd.DataFrame()
    preproc = Preprocessor(serie_a_teams=SERIE_A_TEAMS)
    for player, team, opponent, h_a_player in zip(players, teams, opponents, h_a_player):
        print(f"\n➡️ {player} ({team} vs {opponent})")
        
        # 1️⃣ Filtra storico del giocatore
        player_df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
        if player_df.empty:
            print(f"⚠️ Nessun dato per {player}")
            continue
        
        player_df = get_player_data(df_orig, player)

        if player_df["season"].min() == CURRENT_SEASON:
            player_df = utils.add_other_leagues_data(
                player_df, player,
                DATASET_DATA_DIR, GOALS_DATA_FILE_ALL_LEAGUES,
                CURRENT_SEASON
            )

        # applica al dataset
        player_df["position"] = player_df["position"].apply(utils.clean_position)

        # controlla i valori unici
        print(player_df["position"].unique())
        player_df["position"]= player_df["position"].dropna()
        # Conta le occorrenze
        counts = player_df["position"].value_counts(dropna=False)

        # Rimuovo i "None"
        player_df = player_df[player_df["position"] != "None"]

        main_role = utils.get_main_position_weighted( player_df["position"], window=10, decay=0.8)

        player_df = utils.add_overperformance_features(player_df, stats, player_col="player", prod=True)

        player_df = utils.compute_shot_quality_index(player_df,prod=True)
        player_df = utils.reduce_penalty_xg(player_df)

        df_teams_curr = utils.compute_defensive_overperf_stats(df_teams_curr, team_col="team_name", ga_col="missed", xga_col="xGA", window=5)

        # 3️⃣ Fill NaN con 0
        cols_to_check = ["sum_xG",  
                         #"xG_last5",
                         "finishing_form", #viene tolta e sostituita dal residuo  
                         "overperf_role_resid",
                         "shot_quality_index", 
                         #"position_weighted"       
                         #"cold_penalty",                       
                         #"opponent_xGA_90min",  
                         #"team_xG_90min"
                         ]
        
        
        player_df = utils.fill_missing_values_player_df(player_df, cols_to_check, season_ref=CURRENT_SEASON)

        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)

        #peso xg per ruolo
        #player_df = utils.adjust_sumxg_by_position(player_df, pos_factors)

        player_df["sum_xG"] = np.log1p(player_df["sum_xG"])

        # Calcolo residuo  per finishing_form
        player_df["xg_mean_12"] = (
            player_df.groupby("player")["sum_xG"]
            .apply(lambda x: x.rolling(window=12, min_periods=3).mean())
            .reset_index(level=0, drop=True)
        )
        player_df["xg_mean_12"] = player_df["xg_mean_12"].fillna(0)

        player_df = preproc.compute_shot_quality(player_df, window=12, use_rank=True, prod=True)

        player_df["finishing_form_resid"] = player_df["finishing_form"] - lin.predict(player_df[["xg_mean_12"]])

        cols_to_check.remove("finishing_form")
        cols_to_check.append("finishing_form_resid")
       
        sum_xG_new = (player_df["sum_xG"].tail(12).mean())

        # 4️⃣ Ottieni info squadre. se le partite del giocatore della corrente stagione sono superiori a 5 uso quelle
        num_giornate = utils.count_matchdays(df_teams_curr)

        #se ho un numero sufficiente di giornate, applico discriminante home/away
        if num_giornate >= 10: 
            h_a = utils.get_h_a_opponent(h_a_player)
            #OPPONENT TEAM DATA home/away 
            opponent_xGA_90min_last5_per90 = utils.get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
            xGA_last5_opp, GA_last5_opp = utils.get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

            #PLAYER TEAM DATA home/away
            team_xG_90_min_last5 = utils.get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
            xG_last5_team, Goal_last5_team = utils.get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)
        else:
            #OPPONENT TEAM DATA
            opponent_xGA_90min_last5_per90 = utils.get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
            xGA_last5_opp, GA_last5_opp = utils.get_def_data_last5_team_h_a(opponent,"", df_teams)

            #PLAYER TEAM DATA
            team_xG_90_min_last5 = utils.get_xG_last5_team_h_a_mean(team, "", df_teams)
            xG_last5_team, Goal_last5_team = utils.get_att_data_last5_team_h_a(team, "", df_teams)
        

        sum_xG_new = utils.weighted_xg_vs_opponent_mixed(sum_xG_new, player_df, opponent_xGA_90min_last5_per90, xGA_last5_opp, GA_last5_opp)

        sum_xG_new = utils.weighted_xg_team_mixed(sum_xG_new, player_df, df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team)

        #sum_xG_new = utils.adjust_sumxg_by_defensive_factor(sum_xG_new, df_teams_curr["defensive_adjust_factor_last5"].iloc[-1])

        # Streak senza gol
        cold_penalty = utils.get_latest_cold_penalty(player_df)     

        sum_xG_new = utils.penalize_xg_with_cold_penalty(sum_xG_new,cold_penalty, main_role)

        sum_xG_new = utils.adjust_xg_by_minutes(sum_xG_new, player_df["minutes_played"].rolling(window=5, min_periods=1).mean())
    
        # 6️⃣ Posizioni (dummy)
        #pos_dummy_df = get_positions(player_df, pos_dummies.columns)

        # 7️⃣ Costruisci feature row
        X_new = [[sum_xG_new,   
                  #xG_last5,                                                                                                               
                  player_df["overperf_combined"].iloc[-1],
                  player_df["shot_quality_index"].iloc[-1],
                  #player_df["position_weighted"].iloc[-1],
                  player_df["finishing_form_resid"].iloc[-1]
                                            
                  ]]

        feature_names = cols_to_check
        X_new_df = pd.DataFrame(X_new, columns=feature_names)

        df_records = pd.concat([df_records, X_new_df], axis=0)

        for col, val in X_new_df.iloc[0].items():
            print(f"  {col}: {val:.4f}")

        player_pos = player_df[categorical_features]

        # Aggiungi le dummy di posizione
        X_new_df = pd.concat([X_new_df.reset_index(drop=True), player_pos.tail(1).reset_index(drop=True)], axis=1)

        # Il modello Poisson predice λ = expected goals
        lambda_pred = model.predict(X_new_df)
        
        print(f"Ruolo principale (pesato) di {player}: {main_role}")
        best_a = utils.get_alpha_for_role(main_role)

        # Converti λ → probabilità di segnare almeno un gol applicando alfa per evitarae overconfidence
        prob_goal = 1 - np.exp(-best_a * np.clip(lambda_pred, 0, None))

        # Estrai lo scalare se è array di forma (1,)
        if isinstance(prob_goal, np.ndarray):
            prob_goal = prob_goal.item()
        '''
        # Applica la funzione
        prob_goal = utils.adjust_prob_final(
            prob_base = prob_goal_base,
            overperf_value = player_df["overperf_combined"].iloc[-1],
            finishing_resid = player_df["finishing_form_resid"].iloc[-1],
            role = role
        )
        '''
        
        print(f"✅ Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}. XGA avversaria last5:{opponent_xGA_90min_last5_per90:.2f}, GA avversaria last5:{GA_last5_opp:.2f} giocando in:{h_a_player} ")

        # (Facoltativo) per debugging:
        # print(f"λ previsto (expected goals): {lambda_pred:.4f}")
                #utils.shap_explanation(model, X_new_df, numeric_features)

        results.append({
            "player": player,
            "team": team,
            "opponent": opponent,
            "prob_goal": prob_goal
        })  

        results_df = pd.DataFrame(results) 

        #aggiungo a result anche il df x new conl suo contenuto
        df_records = pd.concat([results_df, X_new_df], axis=0)

        #df_records.to_csv("records_temp.csv")

    return pd.DataFrame(results), model, lin, numeric_features, categorical_features

def main():

    players = INPUT["players"]
    teams = INPUT["teams"]
    opponents = INPUT["opponents"]
    h_a = INPUT["h_a"]

    df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
    df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)

    results = full_training_pipeline(df_orig)

    model = results["model"]
    lin_reg = results["lin_reg"]
    numeric_features = results["numeric_features"]
    categorical_features = results["categorical_features"]
    stats = results["stats"]

    print("Best Threshold:", results["best_threshold"])
    print("Metrics:", results["metrics"])

    pred_df = predict_goal_probabilities(players, teams, opponents,
                                     df_orig, df_teams, df_teams_curr_season,
                                     model, lin_reg,
                                     numeric_features, categorical_features, stats, h_a)
    
    utils.save_models(model=model, scaler_xg=None,scaler=None,poly=None, lin_poly=None, lin=lin_reg, is_baseline=False) 

if __name__ == "__main__":
    main()