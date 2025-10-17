from config import DATASET_DATA_DIR, PROD_DATA_FILE, TEAMS_DATA_FILE, CURRENT_SEASON, BOOST_FACTORS, INPUT, PROD_DATA_FILE_ASSIST
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import imblearn
from imblearn.over_sampling import SMOTE
from unidecode import unidecode

### *** GLOBALS ***

# stagione corrente (es. 2025)
current_season = CURRENT_SEASON

# colonne da pesare
cols_to_weight = ["sum_xG", "xA_last5", "assist_last5"]

boosts = BOOST_FACTORS

players = INPUT["players"]
teams = INPUT["teams"]
opponents = INPUT["opponents"]
### *** END  GLOBALS ***

def predict_assist_probabilities(players, teams, opponents, df_orig, df_teams, calib_model, features, numeric_features, scaler):
    """
    Calcola la probabilità che ciascun giocatore fornisca un assist nella prossima partita.

    Args:
        players (list): Lista di nomi giocatori.
        teams (list): Squadre corrispondenti.
        opponents (list): Avversarie corrispondenti.
        df_orig (pd.DataFrame): Dataset principale con le statistiche per giocatore-partita.
        df_teams (pd.DataFrame): Dataset delle squadre con xG/xGA medi per stagione.
        calib_model: Modello di regressione logistica (già calibrato).
        numeric_features (list): Lista delle feature numeriche da scalare.
        scaler: Scaler addestrato sul training set (es. StandardScaler).

    Returns:
        pd.DataFrame: Tabella con probabilità previste di assist per ogni giocatore.
    """

    results = []

    for player, team, opponent in zip(players, teams, opponents):

        print(f"\n➡️ {player} ({team} vs {opponent})")

        # 1️⃣ Filtra il dataframe del giocatore
        player_df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")

        player_df = get_player_data(df_orig, player)

        if player_df.empty:
            print(f"⚠️ Nessun dato trovato per {player}")
            continue

        # 2️⃣ Rimuovi eventuali partite future
        now = pd.Timestamp.now()
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
        player_df = player_df[player_df["date"] <= now].reset_index(drop=True)

        if player_df.empty:
            print(f"⚠️ Nessuna partita valida (tutte future) per {player}")
            continue

        # 3️⃣ Riempi i NaN
        cols_to_check = features
        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)

        # 4️⃣ Recupera dati della squadra e avversario
        season = player_df["season"].iloc[-1]
        opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)
        #team_xG_90min = get_Xg_90min_team(team, season, df_teams)

        # 5️⃣ Calcola statistiche base del giocatore
        sum_xA = player_df["sum_xA"].tail(18).mean()

        sum_xG_new = weighted_xg_vs_opponent(sum_xA, player_df, opponent_xGA_90min)

        # 6️⃣ Costruisci vettore di input
        X_new = [[sum_xG_new, player_df["xA_last5"].iloc[-1],player_df["key_passes_last5"].iloc[-1]]]
    
        X_new_df = pd.DataFrame(X_new, columns=features)

        #elimino eventuali types obj
        X_new_df[numeric_features] = X_new_df[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0)

        # 7️⃣ Scaling numeriche
        X_new_df[numeric_features] = scaler.transform(X_new_df[numeric_features])

        # 8️⃣ Predici probabilità
        prob_assist = calib_model.predict_proba(X_new_df)[0, 1]

        # 9️⃣ Stampa risultato
        print(f"✅ Probabilità che {player} faccia un assist contro {opponent}: {prob_assist:.2f})")

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

def weighted_xg_vs_opponent(base_xA, player_df, opponent_xGA_90min):
    """
    Calcola uno xG medio del giocatore pesato per la forza dell'avversario (xGA_90min).
    """
    # forza media degli avversari affrontati nelle ultime 10 partite
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(18).mean()

    # se mancano valori, fallback alla media
    if pd.isna(base_xA) or pd.isna(avg_opponent_xGA):
        return base_xA

    # calcola fattore di correzione
    # se l’avversario concede più del normale → boost
    # se concede meno → penalità
    factor = opponent_xGA_90min / avg_opponent_xGA

    # limitiamo il fattore per non esplodere
    factor = np.clip(factor, 0.7, 1.3)

    # xG pesato
    weighted_xA = base_xA * factor
    return weighted_xA

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
    
# trasformazione che moltiplica (usata dopo lo StandardScaler)
def multiply_by_factor(X, factor=2.0):
    return X * factor


def multicoll_check(X, features):

    X = X[features].dropna()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)

def remove_nan(X):
    #*********    trovo i nan in X  *********
    if X.isnull().values.any():
        print("Ci sono valori NaN in X")
        print(X[X.isnull().any(axis=1)])
        X = X.fillna(0)
        print("Dopo il fillna:")
        print(X[X.isnull().any(axis=1)])

    return X

df_orig = pd.read_csv(DATASET_DATA_DIR / PROD_DATA_FILE_ASSIST)
df_teams = pd.read_csv(DATASET_DATA_DIR / TEAMS_DATA_FILE)

#PREPROCESSING

#copia df
df = df_orig.copy()

#*** DROP PARTITE FUTURE ***
now = pd.Timestamp.now()
# converto in datetime se non lo è già
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] <= now].reset_index(drop=True)

#df = get_shot_conversion_mean(df)

#drop nome giocatori e squadre
#df= df.drop(columns=[ "sum_xG","player", "match_id", "player_team", "opponent_team", "date", "is_home","games","time","xG","shots","npg","npxG"])

#analisi statistica
#correlazione tra variabili numeriche
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show(block=True)

#droppo righe con nan
cols_to_check = [
    "sum_xA",
    "xA_last5",
    "key_passes_last5"
]

#df = df.dropna(subset=cols_to_check)
df[cols_to_check] = df[cols_to_check].fillna(0)

#multicoll_check(df,["sum_xG", "n_shots"])

#gestisco la y, ossia i goal trasformandola in booleana, se gol>0 allora 1, altrimenti 0
#df["assists"] = (df["assists"] > 0).astype(int)

# Seleziona le features (X) e target (y)
y = df["assists"]
y_binary = (y > 0).astype(int)

X = df[cols_to_check]

#mask = X["season"] == CURRENT_SEASON

#X = X.drop(columns=["season", "position"])

#********** Standardizzazione feature numeriche, tolgo se uso xgboost o random forest(non lineari) *********

numeric_features = [   
    "sum_xA",
    "xA_last5",
    "key_passes_last5"
    
]

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# --- Split train / test ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# --- Split train / validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# --- Addestramento modello base ---
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# --- Calibrazione su validation ---
calib_model = CalibratedClassifierCV(model, method='sigmoid', cv="prefit")
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
test_log_loss = log_loss(y_test, y_pred)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")

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

base_rate = y_test.mean()   # y_val binario: 1 se ha segnato almeno 1 assist
print("Baseline (freq. reali di assist>0):", base_rate)

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

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(coef_df["Feature"].head(15), coef_df["Coefficient"].head(15))
plt.gca().invert_yaxis()
plt.title("Coefficiente (effetto positivo) - Assist")
plt.xlabel("Peso del coefficiente")
plt.show()

#input utente

INPUT = {
    "players": ["Lautaro", "Christian Pulisic", "Barella"],
    "teams": ["Inter", "AC Milan", "Inter"],
    "opponents": ["Verona", "Torino", "Sassuolo"]
}

results_df = predict_assist_probabilities(
    INPUT["players"], 
    INPUT["teams"], 
    INPUT["opponents"],
    df_orig, df_teams, calib_model, cols_to_check, numeric_features, scaler
)

print(results_df)