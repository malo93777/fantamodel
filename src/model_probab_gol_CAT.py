from config import CURRENT_SEASON_TEAMS_FILE, GOALS_DATA_FILE_ALL_LEAGUES, DATASET_DATA_DIR, PROD_DATA_FILE_GOALS, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_RESID, BOOST_FACTORS_XGB, INPUT, MODEL_DIR, SCALER_DIR, CALIB_LOGISTIC_REG, SCALER, SERIE_A_TEAMS
import utils
from first_preproc import Preprocessor
import pandas as pd
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error,r2_score
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import imblearn
from unidecode import unidecode

### *** GLOBALS ***

# stagione corrente (es. 2025)
current_season = CURRENT_SEASON

# colonne da pesare
cols_to_weight = ["sum_xG", "n_shots", "xG_last5", "shots_last5", "goals_last5"]

boosts = BOOST_FACTORS_XGB

players = INPUT["players"]
teams = INPUT["teams"]
opponents = INPUT["opponents"]
### *** END  GLOBALS ***

def predict_goal_proba(lambda_pred, role, alpha_map, iso_models):
    """Applica α e calibrazione isotonic in base al ruolo"""
    alpha = alpha_map.get(role, np.mean(list(alpha_map.values())))
    p_raw = 1 - np.exp(-np.clip(alpha * lambda_pred, 0, None))
    if role in iso_models:
        return float(iso_models[role].predict([p_raw])[0])
    return float(p_raw)

# 3️⃣ Funzione helper che prende la posizione e restituisce il relativo alpha
def get_alpha_for_position(role):
    return role_alphas.get(role, 0.5)  # fallback se ruolo non presente

# 4️⃣ Applica alpha personalizzato a ogni record (train / val / test)
def apply_alpha_by_role(lambda_vec, roles):
    alphas = np.array([utils.get_alpha_for_role(r) for r in roles])
    return 1 - np.exp(-np.clip(alphas * lambda_vec, 0, None))

def predict_goal_probabilities(players, teams, opponents, df_orig, df_teams, df_teams_curr, model, lin, boosts, numeric_features, categorical_features):
    results = []

    df_records = pd.DataFrame()
    preproc = Preprocessor(serie_a_teams=SERIE_A_TEAMS)
    for player, team, opponent in zip(players, teams, opponents):
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

        #player_df["position_weighted"] = player_df["position"].map(position_weights).fillna(0.3)

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
        player_df["xG_last5"] = np.log1p(player_df["xG_last5"])

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
       
        # 4️⃣ Ottieni info squadre
        season = player_df["season"].iloc[-1]
        #opponent_xGA_90min = utils.get_Xga_90min_opp_team(opponent, season, df_teams)
        #team_xG_90_min = utils.get_Xg_90min_team(team, season, df_teams)

        opponent_xGA_90min_last5 = utils.get_xGA_last5_team(opponent, df_teams_curr)
        team_xG_90_min_last5 = utils.get_xG_last5_team(team, df_teams_curr)
        xGA_last5_opp, GA_last5_opp = utils.get_overperf_last5_team(opponent, df_teams_curr)

        # 5️⃣ Media storica del giocatore
        sum_xG_new = player_df["sum_xG"].mean()

        #Media ultime 12 partite del giocatore (status giocatore ultimi 3 mesi, utile per il Fanta)
        sum_xG_new = (player_df["sum_xG"].tail(12).mean())

        sum_xG_new = utils.weighted_xg_vs_opponent_mixed(sum_xG_new, player_df, opponent_xGA_90min_last5, xGA_last5_opp, GA_last5_opp)

        sum_xG_new = utils.weighted_xg_by_team_strength(sum_xG_new, player_df, team_xG_90_min_last5, df_teams)

        #sum_xG_new = utils.adjust_sumxg_by_defensive_factor(sum_xG_new, df_teams_curr["defensive_adjust_factor_last5"].iloc[-1])

        # Streak senza gol
        cold_penalty = utils.get_latest_cold_penalty(player_df)
        
        main_role = utils.get_main_position_weighted( player_df["position"], window=10, decay=0.8)

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

        # 8️⃣ Applica boost
        for feature, factor in boosts.items():
            X_new_df[feature] = X_new_df[feature] * factor

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
        
        print(f"✅ Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}. XGA avversaria last5:{opponent_xGA_90min_last5:.2f}, GA avversaria last5:{GA_last5_opp:.2f}")

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

    for boost in boosts.items():
        print(boost)
   # df_records.to_csv("records.csv")
    return pd.DataFrame(results)

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


df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)
df_teams_curr_season = pd.read_csv(DATASET_DATA_DIR / CURRENT_SEASON_TEAMS_FILE)
#PREPROCESSING

#copia df
df = df_orig.copy()

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

cols_to_check = [
    "sum_xG", 
    #"xG_last5",
    #"goals_per90_weighted_mean",
    "finishing_form",
    "overperf_role_resid",
    "shot_quality_index"
]

#df = utils.fill_missing_values_player_df(df, cols_to_check, season_ref=CURRENT_SEASON)

#df = df.dropna(subset=cols_to_check)
df[cols_to_check] = df[cols_to_check].fillna(0)

#****** CONTROLLI STATISTICI *******
#utils.analyze_feature_skewness(df, cols_to_check)

#utils.get_stat_desc(df, cols_to_check,"goals")
#collinearity check
utils.multicoll_check(df, cols_to_check)
#gestisco la y, ossia i goal trasformandola in booleana, se gol>0 allora 1, altrimenti 0
#df["goals"] = (df["goals"] > 0).astype(int)

#trasf log sum_xG per ridurre skewness
df["sum_xG"] = np.log1p(df["sum_xG"])
df["xG_last5"] = np.log1p(df["xG_last5"])
# Seleziona le features (X) e target (y)
y = df["is_goals"]
y_binary = (y > 0).astype(int)

numeric_features = cols_to_check

# ======================================
# 1️⃣ Calcolo della feature residua PRIMA dello split
# ======================================
from sklearn.linear_model import LinearRegression
# --- Calcolo residuo lineare (prima di qualsiasi scaling/log) ---

df["xg_mean_12"] = (
    df.groupby("player")["sum_xG"]
    .apply(lambda x: x.rolling(window=10, min_periods=3).mean())
    .reset_index(level=0, drop=True)
)
df["xg_mean_12"]=df["xg_mean_12"].fillna(0)

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

#categorical_features = list(pos_dummies.columns)

# Costruisci X finale
# Aggiungi le dummy di posizione
#X = pd.concat([X[numeric_features].reset_index(drop=True), df[categorical_features].reset_index(drop=True)], axis=1)

vif_df = pd.DataFrame({
    "feature": numeric_features,
    "VIF": [variance_inflation_factor(df[numeric_features].values, i) for i in range(len(numeric_features))]
})
print(vif_df)
'''
position_weights = {
    "F": 1.00,
    "FM": 0.85,
    "M": 0.65,
    "DM": 0.5,
    "D": 0.35,
    "DF": 0.35
}
'''
#df["position_weighted"] = df["position"].map(position_weights).fillna(0.3)
#numeric_features.append("position_weighted")
X =df[numeric_features]
X = pd.concat([X, df["position"]], axis=1)
X.head()
categorical_features = ["position"]

# --- Split train / test ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# --- Split train / validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
)
# --- Split train / validation ---

# Dataset: X (features), y (target: 0/1 se segna o no)
# Esempio: y = df["goal"]
# Calcolo pesi inversamente proporzionali alla frequenza
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, pos_weight, 1)
# ----------------------------
# 1️⃣ TRAINING MODELLO POISSON
# ----------------------------
'''

model = CatBoostRegressor(
    depth=6,
    iterations=800,
    learning_rate=0.015,

    l2_leaf_reg=10,
    random_strength=0,
    bagging_temperature=0.7,

    border_count=128,
    grow_policy="SymmetricTree",
    min_data_in_leaf=30,

    bootstrap_type='Bayesian',
    loss_function='Poisson',
    verbose=False,
    random_seed=42,
    cat_features=categorical_features,

    feature_weights = {
        "sum_xG": 1.0,                 # dominante
        "shot_quality_index": 1.5,     # seconda in importanza
        "finishing_form_resid": 0.7,   # influenza moderata
        "overperf_combined": 0.25 ,   # influenza bassa
        "position_weighted": 1
    }

)

RMSE=0.2209 | Params={'depth': 9, 'learning_rate': 0.005, 'l2_leaf_reg': 3, 'bagging_temperature': 1.0, 'iterations': 1200, 'random_strength': 0, 'min_data_in_leaf': 10,
 'bootstrap_type': 'Bayesian', 'loss_function': 'Poisson', 'verbose': False, 'random_seed': 42}
 
RMSE=0.2207 | Params={'depth': 9, 'learning_rate': 0.01, 'l2_leaf_reg': 6, 'bagging_temperature': 0, 'iterations': 800, 'random_strength': 0, 
 'min_data_in_leaf': 50, 'bootstrap_type': 'Bayesian', 'loss_function': 'Poisson', 'verbose': False, 'random_seed': 42}

RMSE=0.2209 | Params={'depth': 9, 'learning_rate': 0.005, 'l2_leaf_reg': 3, 'bagging_temperature': 1.0, 'iterations': 1200, 'random_strength': 0, 'min_data_in_leaf': 10, 
 'bootstrap_type': 'Bayesian', 'loss_function': 'Poisson', 'verbose': False, 'random_seed': 42
'''

#best_model, best_params = utils.tune_catboost_regressor(X_train, y_train, categorical_features, n_iter=15)
#final_model, best_w, best_rmse = utils.tune_feature_weights(X_train, y_train, categorical_features)

model = CatBoostRegressor(
    depth=9,
    learning_rate=0.01,
    l2_leaf_reg=10,
    bagging_temperature=0,
    iterations=1000,
    random_strength=2.0,
    min_data_in_leaf= 15,
    bootstrap_type='Bayesian',
    loss_function='Poisson',
    verbose=False,
    random_seed=42,
    cat_features=categorical_features
)

model.fit(X_train, y_train, sample_weight=sample_weights)

lambda_val = model.predict(X_val)
y = y_val.values if hasattr(y_val, "values") else y_val

role_alphas = {}
for role in X_val["position"].unique():
    mask = X_val["position"] == role
    lam = model.predict(X_val[mask])
    y = y_val[mask]
    
    best_a, best_brier = None, 1e9
    for a in np.linspace(0.3, 1.0, 40):
        p = 1 - np.exp(-np.clip(a * lam, 0, None))
        b = brier_score_loss(y, p)
        if b < best_brier:
            best_a, best_brier = a, b

    role_alphas[role] = best_a
    print(f"Ruolo {role}: best α = {best_a:.3f}, Brier = {best_brier:.5f}")

print(f"Best alpha: {best_a:.3f}  →  Brier val: {best_brier:.5f}")

# ----------------------------
# 2️⃣ CONVERSIONE OUTPUT → PROBABILITÀ
# ----------------------------
lambda_train = model.predict(X_train)
lambda_val   = model.predict(X_val)
lambda_test  = model.predict(X_test)
'''
# ✅ Applica shrink con best_a
y_train_proba = 1 - np.exp(-np.clip(best_a * lambda_train, 0, None))
y_val_proba   = 1 - np.exp(-np.clip(best_a * lambda_val, 0, None))
y_test_proba  = 1 - np.exp(-np.clip(best_a * lambda_test, 0, None))
'''
# 5️⃣ Applica la conversione separata per ciascun set
y_train_proba = apply_alpha_by_role(lambda_train, X_train["position"])
y_val_proba   = apply_alpha_by_role(lambda_val,   X_val["position"])
y_test_proba  = apply_alpha_by_role(lambda_test,  X_test["position"])

# ----------------------------
# 3️⃣ BINARIZZA per metriche classiche (threshold = 0.5)
# ----------------------------
y_train_pred = (y_train_proba >= 0.36).astype(int)
y_test_pred  = (y_test_proba  >= 0.36).astype(int)

# ----------------------------
# 4️⃣ METRICHE
# ----------------------------

utils.print_metrics(y_train, y_train_pred, y_train_proba, "train")
utils.print_metrics(y_test, y_test_pred, y_test_proba, "test")

# ----------------------------
# 5️⃣ CONFUSION MATRIX
# ----------------------------
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ----------------------------
# 6️⃣ CALIBRAZIONE PROBABILITÀ
# ----------------------------
base_rate = y_test.mean()
print(f"Baseline (freq. goal>0): {base_rate:.3f}")

print(f"Brier score: {brier_score_loss(y_test, y_test_proba):.5f}")

# ----------------------------
# 7️⃣ PRECISION-RECALL & SOGLIA OTTIMALE
# ----------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
avg_prec = average_precision_score(y_test, y_test_proba)

# Calcolo F1 per ogni soglia
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🔍 Miglior soglia trovata: {best_threshold:.3f}")
print(f"✅ F1 ottimale: {best_f1:.3f}")
print(f"   Precision: {precisions[best_idx]:.3f}")
print(f"   Recall:    {recalls[best_idx]:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
plt.axvline(best_threshold, color='r', linestyle='--', label=f"Best Thresh = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 vs Threshold")
plt.show()

# ----------------------------
# 8️⃣ METRICHE CON SOGLIA OTTIMALE
# ----------------------------
y_pred_opt = (y_test_proba >= best_threshold).astype(int)

precision_opt = precision_score(y_test, y_pred_opt)
recall_opt = recall_score(y_test, y_pred_opt)
f1_opt = f1_score(y_test, y_pred_opt)

print(f"\n🎯 Metriche con soglia ottimizzata ({best_threshold:.2f}):")
print(f"Precision: {precision_opt:.3f}")
print(f"Recall:    {recall_opt:.3f}")
print(f"F1 Score:  {f1_opt:.3f}")

# ----------------------------
# 9️⃣ PERMUTATION IMPORTANCE
# ----------------------------
result = permutation_importance(
    model, X_test, y_test, n_repeats=20, random_state=42
)

importance = model.get_feature_importance(prettified=True)
print(importance.head(10))


# Assicurati che X_test sia un DataFrame con gli stessi indici di y_test
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=model.feature_names_)

# Aggiungi y_true, pred prob e pred label
X_test["true_goal"] = y_test.values
X_test["pred_prob"] = y_test_proba
X_test["pred_label"] = y_test_pred

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

# 📊 Analisi di performance per ruolo
agg = (
    X_test_sorted
    .groupby("position")
    .agg(
        mean_prob=("pred_prob", "mean"),
        true_rate=("true_goal", "mean"),
        false_positive_rate=("pred_label", lambda x: ((x==1) & (X_test_sorted.loc[x.index, "true_goal"]==0)).mean()),
        false_negative_rate=("pred_label", lambda x: ((x==0) & (X_test_sorted.loc[x.index, "true_goal"]==1)).mean())
    )
    .sort_values("mean_prob", ascending=False)
)

print("\n📊 METRICHE PER RUOLO:")
print(agg.round(3))

#input utente

pred_df = predict_goal_probabilities(players, teams, opponents,
                                     df_orig, df_teams, df_teams_curr_season,
                                     model, lin_reg, boosts,
                                     numeric_features, categorical_features)


utils.save_models(model=model, scaler_xg=None,scaler=None,poly=None, lin_poly=None, lin=lin_reg, is_baseline=False) 