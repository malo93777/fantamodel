from catboost import CatBoostRegressor
from config import CURRENT_SEASON_TEAMS_FILE ,DATASET_DATA_DIR, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_FACTORS, INPUT, PROD_DATA_FILE_ASSIST
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import poisson
import re

# ============================================================
# GLOBALS
# ============================================================

current_season = CURRENT_SEASON
boosts = BOOST_FACTORS

players = INPUT["players"]
teams = INPUT["teams"]
opponents = INPUT["opponents"]

cols_to_check = ["sum_xA", "xA_last5"]
numeric_features = ["sum_xA", "xA_last5"]


# ============================================================
# UTILS
# ============================================================

def get_player_data(df: pd.DataFrame, player_name: str):
    df = df.copy()
    df["player_norm"] = df["player"].apply(lambda x: unidecode(str(x)).lower())
    player_norm = unidecode(player_name).lower()

    player_df = df[df["player_norm"] == player_norm]

    if player_df.empty:
        player_df = df[df["player_norm"].str.contains(rf"\b{re.escape(player_norm)}\b", na=False)]

    if player_df.empty:
        print(f"⚠️ Nessun giocatore trovato per '{player_name}'.")
        return pd.DataFrame()

    return player_df.sort_values("date").reset_index(drop=True)


def get_Xga_90min_opp_team(team: str, season: str, teams_df: pd.DataFrame):
    row = teams_df[(teams_df["Team"].str.lower() == team.lower()) & (teams_df["season"] == season)]
    return row["XGA_90min"].values[0] if not row.empty else np.nan


def weighted_xA_vs_opponent(base_xA, player_df, opponent_xGA_90min):
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(12).mean()
    if pd.isna(base_xA) or pd.isna(avg_opponent_xGA):
        return base_xA

    factor = np.clip(opponent_xGA_90min / avg_opponent_xGA, 0.75, 1.25)
    return base_xA * factor

def prepare_features(df_orig):

    df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_ASSIST)
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

    numeric_features = [
        "sum_xA", 
        "xA_last5"
    ]

    #df = df.dropna(subset=cols_to_check)
    df[numeric_features] = df[numeric_features].fillna(0)

    #****** CONTROLLI STATISTICI *******
    #utils.analyze_feature_skewness(df, cols_to_check)
    utils.multicoll_check(df, numeric_features)
 
    #trasf log sum_xA per ridurre skewness
    # Seleziona le features (X) e target (y)

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

    return df, numeric_features, categorical_features

# ============================================================
# MODELLO POISSON
# ============================================================

def train_poisson_model(X_train, y_train, cat_features):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weights = np.where(y_train == 1, pos_weight, 1)

    model = CatBoostRegressor(
        depth=6,
        iterations=600,
        learning_rate=0.005,
        loss_function="Poisson",
        random_seed=42,
        verbose=False,
        l2_leaf_reg=3,
        random_strength= 1.0,
        min_data_in_leaf=30,
        bootstrap_type="Bayesian",
        cat_features=cat_features
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

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
# CALIBRAZIONE α PER RUOLO
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
# CONVERSIONE λ → PROBABILITÀ POISSON
# ============================================================

def predict_assist_distribution(lam: float):
    lam = max(lam, 0)

    p0 = poisson.pmf(0, lam)
    p1 = poisson.pmf(1, lam)
    p2plus = 1 - (p0 + p1)

    return {
        "lambda": lam,
        "p0": float(p0),
        "p1": float(p1),
        "p2plus": float(p2plus),
        "p_any": float(1 - p0)
    }

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
# PREVISIONE PER GIOCATORE
# ============================================================

def predict_assist_probabilities(players, teams, opponents, df_orig, df_teams,df_teams_curr,
                                 model, role_alphas, features, numeric_features,h_a_player):

    results = []

    for player, team, opponent, h_a_player in zip(players, teams, opponents, h_a_player):

        player_df = get_player_data(df_orig, player)
        if player_df.empty:
            continue

        now = pd.Timestamp.now()
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
        player_df = player_df[player_df["date"] <= now]

        season = player_df["season"].iloc[-1]

        opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)

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
        
        sum_xA = player_df["sum_xA"].tail(12).mean()
        sum_xA = utils.progressive_weighted_mean(sum_xA, alpha=0.3)

        sum_xA_weighted = utils.weighted_xg_vs_opponent_mixed(sum_xA, player_df, opponent_xGA_90min_last5_per90, xGA_last5_opp, GA_last5_opp)

        sum_xA_weighted = utils.weighted_xg_team_mixed(sum_xA_weighted, df_teams, team_xG_90_min_last5,xG_last5_team,Goal_last5_team)

        xA_last5 = player_df["sum_xA"].tail(5).mean()

        X_new = pd.DataFrame([{
            "sum_xA": sum_xA_weighted,
            "xA_last5": xA_last5,
            "position": player_df["position"].iloc[-1]
        }])

        lam = float(model.predict(X_new)[0])
        role = X_new["position"].iloc[0]
        a = role_alphas.get(role, 0.8)

        lam_adjusted = lam * a

        probs = predict_assist_distribution(lam_adjusted)

        print(f"✅ Probabilità che {player} assista contro {opponent}: {probs['p_any']:.2f}. XGA avversaria last5:{opponent_xGA_90min_last5_per90:.2f}, GA avversaria last5:{GA_last5_opp:.2f}")

        results.append({
            "player": player,
            "team": team,
            "opponent": opponent,
            **probs
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
players = INPUT["players"]
teams = INPUT["teams"]
opponents = INPUT["opponents"]
h_a = INPUT["h_a"]

df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_ASSIST)
df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)

df = df_orig.copy()

df, numeric_features, categorical_features = prepare_features(df)

X = pd.concat([df[numeric_features], df[categorical_features]], axis=1)
y = df["assists"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.3, random_state=42, stratify=y_train_full
)


model = train_poisson_model(X_train, y_train, ["position"])

role_alphas = calibrate_role_alpha(model, X_val, y_val)

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
print(metrics)
print(best_threshold)

# Aggiungi y_true, pred prob e pred label
X_test["true_goal"] = y_test_bin.values
X_test["pred_prob"] = y_test_prob["p_any"]
X_test["pred_label"] = (y_test_prob["p_any"] >= best_threshold).astype(int)

# ============================================================
# PREVISIONE INPUT UTENTE
# ============================================================

results_df = predict_assist_probabilities(
    INPUT["players"],
    INPUT["teams"],
    INPUT["opponents"],
    df_orig,
    df_teams,
    df_teams_curr_season,
    model,
    role_alphas,
    cols_to_check,
    numeric_features,
    h_a
)

print(results_df)

utils.save_models_assist(model)
