from config import DATASET_DATA_DIR, PROD_DATA_FILE_GOALS, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_FACTORS_XGB, INPUT, MODEL_DIR, SCALER_DIR, CALIB_LOGISTIC_REG, SCALER, SERIE_A_TEAMS
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

import numpy as np
import pandas as pd

def compute_goals_per90_weighted(df, window=15, min_games=3):
    """
    Calcola goals_per90_weighted_mean per ogni giocatore, con media mobile pesata e fallback.

    Parametri
    ----------
    df : pd.DataFrame
        Deve contenere le colonne ['player', 'match_date', 'goals', 'minutes_played'].
    window : int, default=15
        Numero massimo di partite considerate nella media mobile.
    min_games : int, default=3
        Numero minimo di partite richieste per calcolare la media pesata.
        Se un giocatore ha meno di min_games partite, usa la media stagionale o di carriera.

    Ritorna
    -------
    pd.Series
        Serie con la colonna goals_per90_weighted_mean allineata a df.
    """

    # Ordina per giocatore e data
    df = df.sort_values(["player", "date"]).copy()

    # Calcolo base: goals per 90 minuti
    #df["goals_per90"] = df["goals"] / (df["time"] / 90)
    #df["goals_per90"] = df["goals_per90"].replace([np.inf, -np.inf], np.nan).fillna(0)

    def rolling_weighted_avg(x):
        # Se meno di min_games partite → usa la media semplice
        if len(x) < min_games:
            return np.mean(x)
        # Pesi crescenti verso le partite più recenti
        weights = np.linspace(1, 2, len(x))
        return np.average(x, weights=weights)

    # Applica la rolling window per giocatore
    df["goals_per90_weighted_mean"] = (
        df.groupby("player", group_keys=False)["goals_per90"]
          .apply(lambda g: g.rolling(window=window, min_periods=min_games)
                           .apply(rolling_weighted_avg, raw=True))
    )

    # Fallback finale: per chi non ha valori → media totale del giocatore
    player_avg = df.groupby("player")["goals_per90"].transform("mean")
    df["goals_per90_weighted_mean"] = df["goals_per90_weighted_mean"].fillna(player_avg)

    return df["goals_per90_weighted_mean"]


def predict_goal_probabilities(players, teams, opponents, df_orig, df_teams, calib_model, lin_poly, boosts, pos_dummies, numeric_features):
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
        
        #player_df["goals_per90_weighted_mean"] = compute_goals_per90_weighted(player_df, window=15, min_games=3)

        #player_df = preproc.add_finishing_efficiency_hist(player_df, window=20)
    
        # Calcolo finishing_eff_weighted
        #player_df = preproc.weight_efficiency_shots(player_df)

        # Calcolo finishing_form
        #player_df = preproc.combine_sumxg_efficiency(player_df, use_rank=True)

        # 3️⃣ Fill NaN con 0
        cols_to_check = ["sum_xG",  
                         #"xG_last5",  
                         #"goals_last5",
                         #"goals_per90_weighted_mean",
                         "finishing_form",               
                         "opponent_xGA_90min"
                         ]
        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)
 
        # Calcolo residuo polinomiale per finishing_form
        #sumxg_scaled = scaler_xg.transform(player_df[["sum_xG"]])
        player_df["finishing_form_resid"] = player_df["finishing_form"] - lin_poly.predict(player_df[["sum_xG"]])
        player_df["finishing_form_resid"] = 1.0 * player_df["finishing_form_resid"] 

        cols_to_check.remove("finishing_form")
        cols_to_check.append("finishing_form_resid")
        
        # 4️⃣ Ottieni info squadre
        season = player_df["season"].iloc[-1]
        opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)

        # 5️⃣ Media storica del giocatore
        #sum_xG_new = player_df["sum_xG"].mean()

        #Media ultime 12 partite del giocatore (status giocatore ultimi 4 mesi, utile per il Fanta)
        sum_xG_new = (player_df["sum_xG"].tail(12).mean())
        sum_xG_new = weighted_xg_vs_opponent(player_df, opponent_xGA_90min)   

        # Calcolo goals_last5 per la riga da prevedere
        if len(player_df) >= 5:
            # Prendi le ultime 5 partite, includendo l'ultima
            goals_last5 = player_df["goals"].iloc[-5:].mean()
        else:
            # Se ci sono meno di 5 partite, usa tutte le partite disponibili
            goals_last5 = player_df["goals"].mean()

        player_df["xG_last5"] = player_df["sum_xG"].rolling(5).mean() / player_df["sum_xG"].mean()
    
        # 6️⃣ Posizioni (dummy)
        #pos_dummy_df = get_positions(player_df, pos_dummies.columns)

        # 7️⃣ Costruisci feature row
        X_new = [[sum_xG_new,        
                  opponent_xGA_90min,    
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

        # 🔟 Aggiungi categoriche (posizioni)
        #X_new_df = pd.concat([X_new_df.reset_index(drop=True), pos_dummy_df.reset_index(drop=True)], axis=1)

        # 🔮 Predizione
        prob_goal = calib_model.predict_proba(X_new_df)[0, 1]
        pred = calib_model.predict(X_new_df)[0]

        print(f"✅ Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}. XGA avversaria:{opponent_xGA_90min:.2f}")

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

        df_records.to_csv("records_temp.csv")

    for boost in boosts.items():
        print(boost)
    df_records.to_csv("records.csv")
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

def weighted_xg_vs_opponent(player_df, opponent_xGA_90min):
    """
    Calcola uno xG medio del giocatore pesato per la forza dell'avversario (xGA_90min).
    """
    # media xG del giocatore nelle ultime 12 partite
    base_xG = (player_df["sum_xG"].tail(12).mean()) 

    # forza media degli avversari affrontati nelle ultime 12 partite
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(12).mean()

    # se mancano valori, fallback alla media
    if pd.isna(base_xG) or pd.isna(avg_opponent_xGA):
        return base_xG

    # calcola fattore di correzione
    # se l’avversario concede più del normale → boost
    # se concede meno → penalità
    factor = opponent_xGA_90min / avg_opponent_xGA

    # limitiamo il fattore per non esplodere
    factor = np.clip(factor, 0.75, 1.25)

    # xG pesato
    weighted_xG = base_xG * factor
    return weighted_xG

def get_Xga_90min_opp_team(team: str, season: str, teams_df: pd.DataFrame) -> float:
    row = teams_df[(teams_df["Team"].str.lower() == team.lower()) & (teams_df["season"] == season)]
    if not row.empty:
        return row["XGA_90min"].values[0]
    else:
        return np.nan
    
def get_Xg_90min_team(team: str, season: str, teams_df: pd.DataFrame) -> float:
    row = teams_df[(teams_df["Team"].str.lower() == team.lower()) & (teams_df["season"] == season)]
    if not row.empty:
        return row["XG_90min"].values[0]
    else:
        return np.nan 

df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_GOALS)
df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)

#PREPROCESSING

#copia df
df = df_orig.copy()

#*** DROP PARTITE FUTURE ***
now = pd.Timestamp.now()
# converto in datetime se non lo è già
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] <= now].reset_index(drop=True)

#analisi statistica
#correlazione tra variabili numeriche

#df["xG_last5"] = df["sum_xG"].rolling(5).mean() / df["sum_xG"].mean()

cols_to_check = [
    "sum_xG", 
    #"xG_last5",
    #"goals_last5",
    #"goals_per90_weighted_mean",
    "finishing_form",
    "opponent_xGA_90min"
]

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

# applica al dataset
df["position"] = df["position"].apply(utils.clean_position)

# controlla i valori unici
print(df["position"].unique())
df["position"]= df["position"].dropna()
# Conta le occorrenze
counts = df["position"].value_counts(dropna=False)

# Rimuovo i "None"
df = df[df["position"] != "None"]

# One-hot encoding
pos_dummies = pd.get_dummies(df["position"], prefix="pos", dtype=int)

df = df.drop(columns=["position"])

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
#df["sum_xG"] = np.log1p(df["sum_xG"])
# Seleziona le features (X) e target (y)
y = df["is_goals"]
y_binary = (y > 0).astype(int)
X = df.drop(columns=["is_goals"])

#******* boosting feature stato di forma giocatore (last5) e media cumulativa (cummean) *********

numeric_features = cols_to_check

#*********    trovo i nan in X  *********
if X.isnull().values.any():
    print("Ci sono valori NaN in X")
    print(X[X.isnull().any(axis=1)])
    X = X.fillna(0)
    print("Dopo il fillna:")
    print(X[X.isnull().any(axis=1)])

#*** Standardizzazione feature numeriche ***

# ======================================
# 1️⃣ Calcolo della feature residua PRIMA dello split
# ======================================
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# --- Calcolo residuo lineare (prima di qualsiasi scaling/log) ---
lin_reg = LinearRegression()

# Fit su tutto il dataset
lin_reg.fit(X[["sum_xG"]], X["finishing_form"])

# Predizione lineare
pred_lin = lin_reg.predict(X[["sum_xG"]])

# Calcolo residuo
X["finishing_form_resid"] = X["finishing_form"] - pred_lin

X["finishing_form_resid"] = 1.0 * X["finishing_form_resid"]

print("Media finishing_form:", X["finishing_form"].mean())
print("Media pred_poly:", pred_lin.mean())
print("Media residuo:", X["finishing_form_resid"].mean())

plt.scatter(pred_lin, X["finishing_form_resid"], alpha=0.6)
plt.axhline(0, color='r')
plt.xlabel("Predizione regressione polinomiale")
plt.ylabel("Residuo (finishing_form - pred_poly)")
plt.show()

numeric_features.remove("finishing_form")
numeric_features.append("finishing_form_resid")

#trasformazione log sum_xG
#X["sum_xG"] = np.log1p(X["sum_xG"])

vif_df = pd.DataFrame({
    "feature": numeric_features,
    "VIF": [variance_inflation_factor(X[numeric_features].values, i) for i in range(len(numeric_features))]
})
print(vif_df)

X = X[numeric_features]
# --- Split train / test ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# --- Split train / validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# --- Addestramento modello base ---
#model = LogisticRegression(random_state=42, class_weight="balanced")
#model.fit(X_train, y_train)


model = XGBClassifier(
    random_state=42,
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,  # evita overfitting di piccole variazioni
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
)

model.fit(X_train, y_train)

# --- Calibrazione su validation ---
calib_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calib_model.fit(X_val, y_val)

# --- Valutazione su test set (mai visto) ---
y_pred_proba = calib_model.predict_proba(X_test)[:, 1]

#calib_model=model
# **************   metrics on train   ***************
y_train_pred = calib_model.predict(X_train)

train_log_loss = log_loss(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"Train Precision: {train_precision:.4f}")
print(f"Train Recall: {train_recall:.4f}")
print(f"Train F1 Score: {train_f1:.4f}")
print(f"Train Log Loss: {train_log_loss:.4f}\n")

# **************   metrics on test   ***************

# Fai previsioni sul set di test
y_pred = calib_model.predict(X_test)
y_prob = calib_model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Log Loss: {log_loss:.4f}")

# **************   Confusion Matrix   ***************

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Aggiungo le probabilità al DataFrame di test per l'analisi
X_test["probabilità"] = y_prob

#stampo prima 20 predizioni di test con probabilità
for i in range(20):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}, Probab: {X_test['probabilità'].iloc[i]:.4f}")

X_test = X_test.drop(columns=["probabilità"])

base_rate = y_test.mean()   # y_val binario: 1 se ha segnato almeno 1 gol
print("Baseline (freq. reali di goal>0):", base_rate)

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("Mean predicted prob")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve")
plt.show()

print("Brier score:", brier_score_loss(y_test, y_prob))

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Calcola Precision-Recall curve e Average Precision
avg_prec = average_precision_score(y_test, y_prob)

# Aggiungiamo anche F1-score per ogni soglia
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
'''
# Plot Precision-Recall
plt.figure(figsize=(8,6))
plt.plot(recalls, precisions, label=f'PR curve (AP={avg_prec:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
#plt.show()

# Plot F1 vs soglia
plt.figure(figsize=(8,6))
plt.plot(thresholds, f1_scores[:-1], label="F1-score")
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall & F1 vs Threshold")
plt.legend()
plt.grid(True)
#plt.show()
'''
'''
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(coef_df["Feature"].head(15), coef_df["Coefficient"].head(15))
plt.gca().invert_yaxis()
plt.title("Coefficiente (effetto positivo) - Gol")
plt.xlabel("Peso del coefficiente")
#plt.show()
'''
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

result = permutation_importance(
    model, X_test, y_test, n_repeats=30, random_state=42
)

importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=True)
print(importance)
plt.barh(importance.index, importance.values)
plt.title("Importanza feature (Permutation Importance)")
plt.xlabel("Riduzione media di accuratezza")
plt.show()

#input utente

pred_df = predict_goal_probabilities(players, teams, opponents,
                                     df_orig, df_teams,
                                     calib_model, lin_reg, boosts,
                                     pos_dummies, numeric_features)


utils.save_model(calib_model)