import argparse
from ast import arg
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
from config import CAT_MODEL_XG,SERIE_A_TEAMS, MODEL_DIR_XG, CURRENT_SEASON_TEAMS_FILE, GOALS_DATA_FILE_ALL_LEAGUES, DATASET_DATA_DIR, PROD_DATA_FILE_GOALS, TEAMS_DATA_FILE, CURRENT_SEASON, INPUT, SERIE_A_TEAMS
from first_preproc import Preprocessor
from unidecode import unidecode
from scipy.stats import poisson

SERIE_A_TOP = ["Juventus", "Inter", "Milan", "Napoli", "Roma", "Lazio"]
SERIE_A_MID = ["Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Udinese", "Genoa", "Como"]
SERIE_A_WEAK = [s for s in SERIE_A_TEAMS if s not in SERIE_A_TOP + SERIE_A_MID]

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
# ============================================================
# 1️⃣ FEATURE ENGINEERING PULITO
# ============================================================
def load_xg_model(model_path= MODEL_DIR_XG / CAT_MODEL_XG):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


def add_xg_pred_feature(df, model, numeric_features, cat_features):
    """
    Aggiunge la feature xg_pred al dataframe usando il modello xG
    """
    df = df.copy()

    # Feature engineering coerente col training
    df = add_opponent_strength_feature(df)

    df = df.sort_values(["player", "date"], kind="mergesort")
    df = df[~df["position"].isin(["GK", "GKS"])]
    df["position"] = df["position"].apply(utils.clean_position)

    df = df.dropna(subset=["position"])
    df[numeric_features] = df[numeric_features].fillna(0)

    # Controllo colonne richieste
    required_cols = numeric_features + cat_features
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[required_cols]
    df["xg_pred"] = model.predict(X)

    return df


def prepare_features(df_orig):

    df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
    df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)

    df = df_orig.copy()
    df = df.sort_values(["player", "date"])

    stats = utils.compute_role_overperf_stats(df)
    df = utils.add_overperformance_features(df, stats, player_col="player", prod=False)
    df = utils.add_goal_scoring_features(df, player_col="player", prod=False)

    #df = df[df["position"] != "GK"]
    #df = df[df["position"] != "GKS"]

    # applica al dataset
    #df["position"] = df["position"].apply(utils.clean_position)

    # controlla i valori unici
    print(df["position"].unique())
    df["position"]= df["position"].dropna()
    # Conta le occorrenze

    df = utils.compute_shot_quality_index_per_shot(df,prod=False)

    print(df[["player", "overperf_log", "overperf_last5", "overperf_combined"]].tail())

    numeric_features = [
        "sum_xG", 
        "finishing_form",
        "overperf_role_resid",
        "shot_quality_index",
        "goal_signal"
    ]

    #df = df.dropna(subset=cols_to_check)
    df[numeric_features] = df[numeric_features].fillna(0)

    #****** CONTROLLI STATISTICI *******
    #utils.analyze_feature_skewness(df, cols_to_check)
    utils.multicoll_check(df, numeric_features)
    
    # Seleziona le features (X) e target (y)

    # 1) Media mobile xG
    # Creazione colonna xG media pesata sulle ultime 12 partite
    df = df.reset_index(drop=True)  # importante: indice unico

    df["xg_mean_12"] = df.groupby("player")["sum_xG"].transform(lambda x: utils.progressive_weighted_rolling(x, alpha=0.1)).fillna(0)

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
    chosen_player = player_norm
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

    return player_df.sort_values("date").reset_index(drop=True), chosen_player

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
        l2_leaf_reg=15,
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
    alphas = {}

    for role in X_val["position"].unique():
        mask = X_val["position"] == role
        lam = model.predict(X_val[mask])
        y = y_val[mask]
        y_bin = (y > 0).astype(int)  # <--- AGGIUNTA: target binario

        best_a, best_score = None, 9e9

        for a in np.linspace(0.3, 1.2, 50):
            p = 1 - np.exp(-np.clip(a * lam, 0, None))
            score = brier_score_loss(y_bin, p)  # <--- USA y_bin

            if score < best_score:
                best_score = score
                best_a = a

        alphas[role] = best_a

    return alphas


# ============================================================
# 4️⃣ CONVERSIONE LAMBDA → PROBABILITÀ
# ============================================================
def predict_probability(model, X, role_alphas):
    """
    Restituisce un DataFrame con le probabilità di 0, 1, 2, 3+ gol per ogni riga di X.
    """ 

    lam = model.predict(X)
    a = X["position"].map(role_alphas).fillna(0.7)
    lam_adj = np.clip(a * lam, 0, None)

    # Calcola le probabilità Poisson per 0, 1, 2 gol
    p0 = poisson.pmf(0, lam_adj)
    p1 = poisson.pmf(1, lam_adj)
    p2 = poisson.pmf(2, lam_adj)
    # Probabilità di 3 o più gol: 1 - (p0 + p1 + p2)
    p3plus = 1 - (p0 + p1 + p2)
    # Probabilità di almeno 1 gol
    p_any = 1 - p0

    # Restituisci DataFrame con tutte le probabilità
    return pd.DataFrame({
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3plus": p3plus,
        "p_any": p_any,
        "lambda": lam_adj
    }, index=X.index)


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
def full_training_pipeline(df, args):

    df,stats, lin_reg, numeric_features, categorical_features = prepare_features(df)

    # Target multiclasse (0,1,2,3+)
    #df["goal_class"] = df["is_goals"].clip(0, 3)

    # Binarizzazione per il modello Poisson
    #df["binary_goal"] = (df["goal_class"] > 0).astype(int)
    if args.fit == True:

        features = numeric_features + categorical_features

        X = df[features]
        y = df["goals"]

        cat_features = ["position"]

        # Split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.3, random_state=42, stratify=y_train_full
        )

        model = None
        # Train
        #if no_train = false: 
        model = train_poisson_model(X_train, y_train, cat_features)

        importance = model.get_feature_importance(prettified=True)
        print(importance.head(10))

        # α per ruolo
        role_alphas = calibrate_role_alpha(model, X_val, y_val)

        print(role_alphas)

        # Predict
        y_test_prob = predict_probability(model, X_test, role_alphas)

        # Trova threshold ottimale
        y_test_bin = (y_test > 0).astype(int)
        precisions, recalls, thresholds = precision_recall_curve(y_test_bin, y_test_prob["p_any"])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        # Metriche
        metrics = evaluate_model(y_test_bin, y_test_prob["p_any"], threshold=best_threshold)

        print(best_threshold)

        # Aggiungi y_true, pred prob e pred label
        X_test["true_goal"] = y_test_bin.values
        X_test["pred_prob"] = y_test_prob["p_any"]
        X_test["pred_label"] = (y_test_prob["p_any"] >= best_threshold).astype(int)

        # Ordina per probabilità discendente
        X_test_sorted = X_test.sort_values(by="pred_prob", ascending=False)

        # Analisi media per ruolo
        print("\n📊 Probabilità media di gol per ruolo:")
        print(X_test_sorted.groupby("position")["pred_prob"].mean().sort_values(ascending=False))

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

        #Stamp di numero di errori per ruolo, aggiungendo anche su falsi positivi e negativi
        print("\n📊 Errori di classificazione per ruolo:")
        print(X_test_sorted[X_test_sorted["true_goal"] != X_test_sorted["pred_label"]].groupby("position").size())
        print("\n📊 Falsi Negativi per ruolo:")
        print(false_negatives.groupby("position").size())
        print("\n📊 Falsi Positivi per ruolo:")
        print(false_positives.groupby("position").size())

        return {
            "model": model,
            "lin_reg": lin_reg,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "role_alphas": role_alphas,
            "test_prob": y_test_prob["p_any"],
            "test_true": y_test_bin,
            "best_threshold": best_threshold,
            "metrics": metrics,
            "stats": stats
        }, df
    else:
        return {"lin_reg": lin_reg,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "stats": stats
            }, df

def predict_goal_probabilities(model_xg, players, teams, opponents, df_orig, df_teams, df_teams_curr, model, lin, numeric_features, categorical_features, stats, h_a_player):
    results = []
    
    df_records = pd.DataFrame()

    for player, team, opponent, h_a_player in zip(players, teams, opponents, h_a_player):
        print(f"\n➡️ {player} ({team} vs {opponent})")

        others_leagues_data = False
        
        # 1️⃣ Filtra storico del giocatore
        player_df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
        if player_df.empty:
            print(f"⚠️ Nessun dato per {player}")
            continue
        
        player_df, player_full_name = get_player_data(df_orig, player)

        if player_df["season"].min() == CURRENT_SEASON:
            others_leagues_data = True
            player_df = utils.add_other_leagues_data(
                player_df, player_full_name,
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

        # Sostituisco "None" con il valore più frequente (escluso None) nella colonna position
        if others_leagues_data == False:
            most_freq = player_df.loc[player_df["position"] != "None", "position"].mode()
            if not most_freq.empty:
                most_freq_value = most_freq.iloc[0]
                player_df.loc[player_df["position"] == "None", "position"] = most_freq_value

        main_role = utils.get_main_position_weighted( player_df["position"], window=10, decay=0.8)

        if player == "nico paz" or player == "odgaard":
            main_role = "FM"

        player_df = utils.add_overperformance_features(player_df, stats, player_col="player", prod=True)

        player_df = utils.compute_shot_quality_index_per_shot(player_df,prod=True)
        #player_df = utils.compute_shot_quality_index_v2(player_df, prod=False)
        player_df = utils.reduce_penalty_xg(player_df)

        #DEBUGGGG DA TOGLIERE
        #player_df = utils.compute_finishing_form(player_df, prod=True)

        df_teams_curr = utils.compute_defensive_overperf_stats(df_teams_curr, team_col="team_name", ga_col="missed", xga_col="xGA", window=5)

        # 3️⃣ Fill NaN con 0
        cols_to_check = ["sum_xG",  
                         #"xG_last5",
                         "finishing_form", #viene tolta e sostituita dal residuo  
                         "overperf_role_resid",
                         "shot_quality_index", 
                         "goal_signal"
                         ]
        
        
        player_df = utils.fill_missing_values_player_df(player_df, cols_to_check, season_ref=CURRENT_SEASON)

        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)
   
        # Creazione colonna xG media pesata sulle ultime 12 partite
        player_df = player_df.reset_index(drop=True)  # importante: indice unico
        player_df["xg_mean_12"] = player_df.groupby("player")["sum_xG"].transform(lambda x: utils.progressive_weighted_rolling(x, alpha=0.1)).fillna(0)

        # Calcolo residuo  per finishing_form
        player_df["finishing_form_resid"] = player_df["finishing_form"] - lin.predict(player_df[["xg_mean_12"]])

        cols_to_check.remove("finishing_form")
        cols_to_check.append("finishing_form_resid")
       
        #sum_xG_new = (player_df["sum_xG"].tail(12).mean())
        #last_xG = player_df["sum_xG"].tail(12).tolist()

        #sum_xG_new = utils.progressive_weighted_mean(last_xG, alpha=0.1)

        # 4️⃣ Ottieni info squadre. se le partite del giocatore della corrente stagione sono superiori a 5 uso quelle
        num_giornate = utils.count_matchdays(df_teams_curr)

        #se ho un numero sufficiente di giornate, applico discriminante home/away
        if num_giornate >= 15: 

            h_a = utils.get_h_a_opponent(h_a_player)

            # ==========================
            # 🔹 OPPONENT
            # ==========================

            # Split casa/trasferta
            opponent_xGA_split = utils.get_xGA_last5_team_h_a_mean(opponent, h_a, df_teams_curr)
            xGA_split_opp, GA_split_opp = utils.get_def_data_last5_team_h_a(opponent, h_a, df_teams_curr)

            # Overall (forma pura)
            opponent_xGA_overall = utils.get_xGA_last5_team_h_a_mean(opponent, "", df_teams_curr)
            xGA_overall_opp, GA_overall_opp = utils.get_def_data_last5_team_h_a(opponent, "", df_teams_curr)

            # Media pesata 70 / 30
            opponent_xGA_last5_per90 = (
                0.7 * opponent_xGA_overall +
                0.3 * opponent_xGA_split
            )

            xGA_last5_opp = 0.7 * xGA_overall_opp + 0.3 * xGA_split_opp
            GA_last5_opp  = 0.7 * GA_overall_opp  + 0.3 * GA_split_opp

            # ==========================
            # 🔹 PLAYER TEAM
            # ==========================

            # Split casa/trasferta
            team_xG_split = utils.get_xG_last5_team_h_a_mean(team, h_a_player, df_teams_curr)
            xG_split_team, Goal_split_team = utils.get_att_data_last5_team_h_a(team, h_a_player, df_teams_curr)

            # Overall
            team_xG_overall = utils.get_xG_last5_team_h_a_mean(team, "", df_teams_curr)
            xG_overall_team, Goal_overall_team = utils.get_att_data_last5_team_h_a(team, "", df_teams_curr)

            # Media pesata 70 / 30
            team_xG_90_min_last5 = (
                0.7 * team_xG_overall +
                0.3 * team_xG_split
            )

            xG_last5_team     = 0.7 * xG_overall_team   + 0.3 * xG_split_team
            Goal_last5_team   = 0.7 * Goal_overall_team + 0.3 * Goal_split_team

        else:
            #OPPONENT TEAM DATA
            opponent_xGA_last5_per90 = utils.get_xGA_last5_team_h_a_mean(opponent, "", df_teams)
            xGA_last5_opp, GA_last5_opp = utils.get_def_data_last5_team_h_a(opponent,"", df_teams)

            #PLAYER TEAM DATA
            team_xG_90_min_last5 = utils.get_xG_last5_team_h_a_mean(team, "", df_teams)
            xG_last5_team, Goal_last5_team = utils.get_att_data_last5_team_h_a(team, "", df_teams)
        
        # predizione xg futuro
        opponent_strength = utils.map_strength(opponent) 
        sum_xG_new = utils.predict_xg_next_match(model_xg, player_df, main_role, opponent_strength)
        sum_xG_new = utils.weighted_xg_vs_opponent_mixed(sum_xG_new, player_df, opponent_xGA_last5_per90, xGA_last5_opp, GA_last5_opp)

        sum_xG_new = utils.weighted_xg_team_mixed(sum_xG_new, df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team)

        sum_xG_new = utils.adjust_xg_by_minutes(sum_xG_new, player_df["minutes_played"].rolling(window=5, min_periods=1).mean())

        opponent = utils.normalize_team(opponent)
        opponent_strength = utils.map_strength(opponent) 

        xg_adj_pct = utils.compute_player_vs_strength_xg_adjustment(
            player_df,
            opponent_strength
        )

        sum_xG_new = sum_xG_new * (1 + xg_adj_pct)

        # 7️⃣ Costruisci feature row
        X_new = [[sum_xG_new,                                                                                                                
                  player_df["overperf_role_resid"].iloc[-1],
                  player_df["shot_quality_index"].iloc[-1],
                  player_df["finishing_form_resid"].iloc[-1]   ,
                  player_df["goal_signal"].iloc[-1]                                 
                  ]]

        feature_names = cols_to_check
        X_new_df = pd.DataFrame(X_new, columns=feature_names)

        df_records = pd.concat([df_records, X_new_df], axis=0)

        for col, val in X_new_df.iloc[0].items():
            print(f"  {col}: {val:.4f}")

        player_pos = player_df[categorical_features]

        # Aggiungi le dummy di posizione
        X_new_df = pd.concat([X_new_df.reset_index(drop=True), player_pos.tail(1).reset_index(drop=True)], axis=1)

        probs = utils.predict_probabilities_poisson(
        model=model,
        X_new_df=X_new_df,
        main_role=main_role,
        poisson_fn=utils.poisson_goal_probs,
        target="goal"
        )

        print(f"✅ Probabilità che {player} ({main_role}) segni contro {opponent}: {probs['p_any']:.2f}. XGA avversaria last5:{opponent_xGA_last5_per90:.2f}, GA avversaria last5:{GA_last5_opp:.2f}")

        results.append({
            "player": player,
            "team": team,
            "opponent": opponent,
            **probs
        })  

    return pd.DataFrame(results), model, lin, numeric_features, categorical_features

def main():

    parser = argparse.ArgumentParser(description="FantaModel")
    parser.add_argument("--fit", action="store_true", help="Vuoi riaddestrare il modello?")
    args = parser.parse_args()
    args.fit = True

    players = INPUT["players"]
    teams = INPUT["teams"]
    opponents = INPUT["opponents"]
    h_a = INPUT["h_a"]

    df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
    df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)

        
    results, df_mod = full_training_pipeline(df_orig, args)
    
    if results.keys().__contains__("model"):

        model = results["model"]
        lin_reg = results["lin_reg"]
        numeric_features = results["numeric_features"]
        categorical_features = results["categorical_features"]
        stats = results["stats"]

        print("Best Threshold:", results["best_threshold"])
        print("Metrics:", results["metrics"])
    else:
        lin_reg = results["lin_reg"]
        numeric_features = results["numeric_features"]
        categorical_features = results["categorical_features"]
        stats = results["stats"]
         #carico i modelli salvati
        models = utils.load_models()
        model = models["poiss_reg"]

    xg_model = utils.load_xg_model()

    #pred_df = predict_goal_probabilities(xg_model["catboost_regressor_xg"], players, teams, opponents,
                                     #df_mod, df_teams, df_teams_curr_season,
                                     #model, lin_reg,
                                     #numeric_features, categorical_features, stats, h_a)
    
    utils.save_models(model=model, scaler_xg=None,scaler=None,poly=None, lin_poly=None, lin=lin_reg, is_baseline=False) 

if __name__ == "__main__":
    main()