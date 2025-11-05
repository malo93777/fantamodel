from config import CURRENT_SEASON_TEAMS_FILE, GOALS_DATA_FILE_ALL_LEAGUES, DATASET_DATA_DIR, PROD_DATA_FILE_GOALS, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_RESID, BOOST_FACTORS_XGB, INPUT, MODEL_DIR, SCALER_DIR, CALIB_LOGISTIC_REG, SCALER, SERIE_A_TEAMS
import utils
from first_preproc import Preprocessor
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
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
from sklearn.calibration import CalibratedClassifierCV
import statsmodels.api as sm
import imblearn
from imblearn.over_sampling import SMOTE
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

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def predict_goal_probabilities(players, teams, opponents, df_orig, df_teams, df_teams_curr, model, best_a, lin, boosts, numeric_features, categorical_features):
    results = []

    df_records = pd.DataFrame()

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
        '''
        # --- Ottieni posizione e one-hot encode coerente col training ---
        if "position" in player_df.columns:
         player_position = utils.clean_position(player_df["position"].iloc[-1])
        else:
            player_position = None

        # Crea il vettore one-hot con le stesse colonne del training
        pos_features = {col: 0 for col in pos_dummies.columns}
        if player_position is not None:
            col_name = f"pos_{player_position}"
            if col_name in pos_features:
                pos_features[col_name] = 1

        # Converti in DataFrame
        pos_df = pd.DataFrame([pos_features])
        '''

        # applica al dataset
        player_df["position"] = player_df["position"].apply(utils.clean_position)

        # controlla i valori unici
        print(player_df["position"].unique())
        player_df["position"]= player_df["position"].dropna()
        # Conta le occorrenze
        counts = player_df["position"].value_counts(dropna=False)

        # Rimuovo i "None"
        player_df = player_df[player_df["position"] != "None"]

        # 3️⃣ Fill NaN con 0
        cols_to_check = ["sum_xG",  
                         "xG_last5",  
                         "goals_last5",
                         "finishing_form", #viene tolta e sostituita dal residuo         
                         #"cold_penalty",                       
                         #"opponent_xGA_90min",  
                         #"team_xG_90min"
                         #"minutes_played_last5"
                         ]
        
        
        player_df = utils.fill_missing_values_player_df(player_df, cols_to_check, season_ref=CURRENT_SEASON)

        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)

        player_df["sum_xG"] = np.log1p(player_df["sum_xG"])
 
        # Calcolo residuo polinomiale per finishing_form
        player_df["xg_mean_12"] = (
            player_df.groupby("player")["sum_xG"]
            .apply(lambda x: x.rolling(window=10, min_periods=3).mean())
            .reset_index(level=0, drop=True)
        )
        player_df["xg_mean_12"] = player_df["xg_mean_12"].fillna(0)

        player_df["finishing_form_resid"] = player_df["finishing_form"] - lin.predict(player_df[["xg_mean_12"]])

        cols_to_check.remove("finishing_form")
        cols_to_check.append("finishing_form_resid")
        
        # 4️⃣ Ottieni info squadre
        season = player_df["season"].iloc[-1]
        #opponent_xGA_90min = utils.get_Xga_90min_opp_team(opponent, season, df_teams)
        #team_xG_90_min = utils.get_Xg_90min_team(team, season, df_teams)

        opponent_xGA_90min_last5 = utils.get_xGA_last5_team(opponent, df_teams_curr)
        team_xG_90_min_last5 = utils.get_xG_last5_team(team, df_teams_curr)

        # 5️⃣ Media storica del giocatore
        #sum_xG_new = player_df["sum_xG"].mean()

        #Media ultime 12 partite del giocatore (status giocatore ultimi 3 mesi, utile per il Fanta)
        sum_xG_new = (player_df["sum_xG"].tail(12).mean())
        
        resid = player_df["finishing_form_resid"].iloc[-1]

        sum_xG_new = sum_xG_new * (1.0 + BOOST_RESID * resid)  #boost =1.0

        sum_xG_new = utils.weighted_xg_vs_opponent(sum_xG_new, player_df, opponent_xGA_90min_last5)   

        sum_xG_new = utils.weighted_xg_by_team_strength(sum_xG_new, player_df, team_xG_90_min_last5, df_teams)

        # Streak senza gol
        cold_penalty = utils.get_latest_cold_penalty(player_df)

        sum_xG_new = utils.penalize_xg_with_cold_penalty(sum_xG_new,cold_penalty, player_df["position"].iloc[-1]) 

        #sum_xG_new = utils.weighted_xg_by_team_strength(sum_xG_new, player_df, team_xG_90_min, df_teams)
        # RiCalcolo features rolling contando anche dati ultima partita
        if len(player_df) >= 5:
            # Prendi le ultime 5 partite, includendo l'ultima
            xG_last5 = player_df["sum_xG"].iloc[-5:].mean()
            goals_last5 = player_df["goals"].iloc[-5:].mean()
            minutes_last5 = player_df["minutes_played"].rolling(window=5, min_periods=1).mean()
        else:
            # Se ci sono meno di 5 partite, usa tutte le partite disponibili
            xG_last5 = player_df["sum_xG"].mean()
            goals_last5 =  player_df["goals"].mean()
            minutes_last5 = player_df["minutes_played"].mean()

        # 🔹 Streak senza gol
        cold_penalty = utils.get_latest_cold_penalty(player_df)
    
        # 6️⃣ Posizioni (dummy)
        #pos_dummy_df = get_positions(player_df, pos_dummies.columns)

        # 7️⃣ Costruisci feature row
        X_new = [[sum_xG_new,   
                  xG_last5,   
                  goals_last5, 
                  #cold_penalty,                 
                  #opponent_xGA_90min_last5,  
                  #team_xG_90_min,             
                  #minutes_last5.iloc[-1],
                  player_df["finishing_form_resid"].iloc[-1],                    
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

         # Converti λ → probabilità di segnare almeno un gol applicando alfa per evitarae overconfidence
        prob_goal = 1 - np.exp(-0.7 * np.clip(lambda_pred, 0, None))

        # Estrai lo scalare se è array di forma (1,)
        if isinstance(prob_goal, np.ndarray):
            prob_goal = prob_goal.item() 

        # Interpretazione binaria (se vuoi anche la previsione 0/1)
        pred = int(prob_goal >= best_threshold)  # Usa la soglia trovata nel training

        print(f"✅ Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}. XGA avversaria last5:{opponent_xGA_90min_last5:.2f}")

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

#*** DROP PARTITE FUTURE ***
now = pd.Timestamp.now()
# converto in datetime se non lo è già
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] <= now].reset_index(drop=True)

#df, lr_new = compute_finishing_form_resid(df, 10)

#analisi statistica
#correlazione tra variabili numeriche

#df["xG_last5"] = df["sum_xG"].rolling(5).mean() / df["sum_xG"].mean()

cols_to_check = [
    "sum_xG", 
    "xG_last5",
    "goals_last5",
    #"goals_per90_weighted_mean",
    "finishing_form",
    #"cold_penalty",
    #"opponent_xGA_90min",
    #"team_xG_90min"
    #"minutes_played_last5"
]

df = utils.fill_missing_values_player_df(df, cols_to_check, season_ref=CURRENT_SEASON)

#df = df.dropna(subset=cols_to_check)
df[cols_to_check] = df[cols_to_check].fillna(0)

#multicoll_check(df,["finishing_efficiency", "sum_xG"])
#corr = df[cols_to_check].corr(numeric_only=True)
#plt.figure(figsize=(12, 10))
#sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#plt.title("Correlation Matrix")
#plt.show()
#df.groupby(pd.qcut(df['finishing_efficiency_hist'], 4, duplicates='drop'))['is_goals'].mean().plot(kind='bar')
#plt.title('Finishing Efficiency vs Goal Probability')
#********** POSIZIONE *************
print(df["position"].unique())
plt.figure(figsize=(6,4))
df["position"].value_counts().plot(kind="bar", edgecolor="black")
plt.ylabel("numero osservazioni")
plt.xlabel("Ruolo")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

df = df[df["position"] != "GK"]
df = df[df["position"] != "GKS"]

# applica al dataset
df["position"] = df["position"].apply(utils.clean_position)

# controlla i valori unici
print(df["position"].unique())
df["position"]= df["position"].dropna()
# Conta le occorrenze

print(df.shape)
# Rimuovo i "None"
#df = df[df["position"] != "None"]
print(df.shape)

# One-hot encoding
pos_dummies = pd.get_dummies(df["position"], prefix="pos", dtype=int)

# Aggiungo le colonne one-hot a X
df = pd.concat([df, pos_dummies], axis=1)

#df = df.drop(columns=["position"])

# Aggiungo al dataset
#df = pd.concat([df, pos_dummies], axis=1)

#****** CONTROLLI STATISTICI *******
#utils.analyze_feature_skewness(df, cols_to_check)

#utils.get_stat_desc(df, cols_to_check,"goals")
#collinearity check
utils.multicoll_check(df, cols_to_check)
#gestisco la y, ossia i goal trasformandola in booleana, se gol>0 allora 1, altrimenti 0
#df["goals"] = (df["goals"] > 0).astype(int)

#trasf log sum_xG per ridurre skewness
df["sum_xG"] = np.log1p(df["sum_xG"])
# Seleziona le features (X) e target (y)
y = df["is_goals"]
y_binary = (y > 0).astype(int)

#******* boosting feature stato di forma giocatore (last5) e media cumulativa (cummean) *********
# 8️⃣ Applica boost
for feature, factor in boosts.items():
    df[feature] = df[feature] * factor

numeric_features = cols_to_check

# ======================================
# 1️⃣ Calcolo della feature residua PRIMA dello split
# ======================================
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

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
print("Media pred_poly:", pred_lin.mean())
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

from catboost import CatBoostRegressor, Pool

# Dataset: X (features), y (target: 0/1 se segna o no)
# Esempio: y = df["goal"]
# Calcolo pesi inversamente proporzionali alla frequenza
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, pos_weight, 1)
# ----------------------------
# 1️⃣ TRAINING MODELLO POISSON
# ----------------------------
model = CatBoostRegressor(
    depth=4,
    learning_rate=0.01,
    l2_leaf_reg=6,
    bagging_temperature=0,
    iterations=500,
    random_strength=1.5,
    bootstrap_type='Bayesian',
    loss_function='Poisson',
    verbose=False,
    random_seed=42,
    cat_features=categorical_features
)

model.fit(X_train, y_train, sample_weight=sample_weights)

lambda_val = model.predict(X_val)
y = y_val.values if hasattr(y_val, "values") else y_val

alphas = np.linspace(0.3, 1.0, 40)
best_a, best_brier = None, 1e9

for a in alphas:
    p = 1 - np.exp(-np.clip(a * lambda_val, 0, None))
    b = brier_score_loss(y, p)
    if b < best_brier:
        best_brier = b
        best_a = a

print(f"Best alpha: {best_a:.3f}  →  Brier val: {best_brier:.5f}")

# ----------------------------
# 2️⃣ CONVERSIONE OUTPUT → PROBABILITÀ
# ----------------------------
lambda_train = model.predict(X_train)
lambda_val   = model.predict(X_val)
lambda_test  = model.predict(X_test)

# ✅ Applica shrink con best_a
y_train_proba = 1 - np.exp(-np.clip(best_a * lambda_train, 0, None))
y_val_proba   = 1 - np.exp(-np.clip(best_a * lambda_val, 0, None))
y_test_proba  = 1 - np.exp(-np.clip(best_a * lambda_test, 0, None))

# ----------------------------
# 3️⃣ BINARIZZA per metriche classiche (threshold = 0.5)
# ----------------------------
y_train_pred = (y_train_proba >= 0.31).astype(int)
y_test_pred  = (y_test_proba  >= 0.31).astype(int)

'''
# ----------------------------
# 3️⃣ CALIBRAZIONE
# ----------------------------
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(y_val_proba, y_val)  # usa validation

# 3) Fit Platt / sigmoid (LogisticRegression su 1D)
platt = LogisticRegression(solver='lbfgs')
# reshape per sklearn: (n_samples, 1)
platt.fit(y_val_proba.reshape(-1, 1), y_val)

# Applichiamo la calibrazione ai test
y_test_proba_cal = iso.predict(y_test_proba)

y_test_proba_cal_sigmoid = platt.predict_proba(y_test_proba.reshape(-1, 1))[:, 1]

# Valutazione prima/dopo calibrazione
print("Brier before:", brier_score_loss(y_test, y_test_proba))
print("Brier after isotonic:", brier_score_loss(y_test, y_test_proba_cal))
print("Brier after sigmoid:", brier_score_loss(y_test, y_test_proba_cal_sigmoid))

# Curva di calibrazione
prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_test_proba_cal, n_bins=10)

plt.figure(figsize=(7,6))
plt.plot(prob_pred, prob_true, "o-", label="Before calibration")
plt.plot(prob_pred_cal, prob_true_cal, "o-", label="After isotonic")
plt.plot([0,1],[0,1],"--", color="gray")
plt.legend()
plt.xlabel("Predicted probability")
plt.ylabel("True fraction of goals")
plt.title("Calibration Curve")
plt.show()
'''
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

importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(importance.index, importance.values)
plt.title("Permutation Feature Importance")
plt.xlabel("Riduzione media di accuratezza")
plt.show()

print("\nFeature importance:\n", importance.tail(10))

#input utente

pred_df = predict_goal_probabilities(players, teams, opponents,
                                     df_orig, df_teams, df_teams_curr_season,
                                     model, best_a, lin_reg, boosts,
                                     numeric_features, categorical_features)


utils.save_models(model=model, scaler_xg=None,scaler=None,poly=None, lin_poly=None, lin=lin_reg, is_baseline=False) 