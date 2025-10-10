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

### *** GLOBALS ***

top_teams = ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Atalanta", "Lazio"]
mid_teams = ["Fiorentina", "Torino", "Bologna", "Sassuolo", "Udinese"]
weak_teams = ["Empoli", "Verona", "Cagliari", "Lecce", "Salernitana", "Frosinone", "Monza", "Genoa", "Sampdoria", "Spezia", "Pisa", "Cremonese", "Benevento"]

map_strength_dict = {
    'top': 3,
    'mid': 2,
    'weak': 1
}

# stagione corrente (es. 2025)
current_season = 2025

# colonne da pesare
cols_to_weight = ["sum_xG", "n_shots", "xG_last5", "shots_last5", "goals_last5"]

boosts = {#boostare sumxg è stata chiave per tenere bassi difensori ma alzare attaccanti
    "sum_xG": 2.5,
    "xG_last5": 2.5,
    "shots_last5": 1.0,
    "goals_last5": 1.0,
    "opponent_xGA_90min": 1.0,
    "team_xG_90min": 1.0,
}

players = ["Lautaro", "Christian Pulisic", "Pavlovic", "Orsolini", "barella"]
teams = ["Inter", "AC Milan", "Inter", "Bologna", "inter"]
opponents = ["Verona", "Torino", "Fiorentina", "Juventus", "Sassuolo"]
### *** END  GLOBALS ***

def predict_goal_probabilities(players, teams, opponents, df_orig, df_teams, calib_model, scaler, boosts, pos_dummies, numeric_features):
    results = []

    for player, team, opponent in zip(players, teams, opponents):
        print(f"\n➡️ {player} ({team} vs {opponent})")

        # 1️⃣ Filtra storico del giocatore
        player_df = df_orig[df_orig["player"].str.contains(player, case=False, na=False)].sort_values("date")
        if player_df.empty:
            print(f"⚠️ Nessun dato per {player}")
            continue

        # 2️⃣ Drop partite future
        now = pd.Timestamp.now()
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
        player_df = player_df[player_df["date"] <= now].reset_index(drop=True)

        # 3️⃣ Fill NaN con 0
        cols_to_check = ["sum_xG", "n_shots", "xG_last5", "shots_last5", "goals_last5", "team_xG_90min", "opponent_xGA_90min"]
        player_df[cols_to_check] = player_df[cols_to_check].fillna(0)

        # 4️⃣ Ottieni info squadre
        season = player_df["season"].iloc[-1]
        opponent_xGA_90min = get_Xga_90min_opp_team(opponent, season, df_teams)
        player_team_xG_90min = get_Xg_90min_team(team, season, df_teams)

        # 5️⃣ Media storica del giocatore
        #sum_xG_new = player_df["sum_xG"].mean()
        #n_shots_new = player_df["n_shots"].mean()

        #Media ultime 5 partite del giocatore (status giocatore ultimi 2 mesi, utile per il Fanta)
        sum_xG_new = (player_df["sum_xG"].tail(10).mean())

        sum_xG_new = weighted_xg_vs_opponent(player_df, opponent_xGA_90min)

        n_shots_new = (player_df["n_shots"].tail(10).mean())

        #print(f" {player} sum_xG_new: {sum_xG_new:.2f}\n")
        #print(f" {player} n_shots_new: {n_shots_new:.2f}\n")

        # 6️⃣ Posizioni (dummy)
        pos_dummy_df = get_positions(player_df, pos_dummies.columns)

        # 7️⃣ Costruisci feature row
        X_new = [[sum_xG_new, n_shots_new,
                  player_df["xG_last5"].iloc[-1],
                  player_df["shots_last5"].iloc[-1],
                  player_df["goals_last5"].iloc[-1],
                  player_team_xG_90min,
                  opponent_xGA_90min]]

        feature_names = ["sum_xG", "n_shots", "xG_last5", "shots_last5", "goals_last5", "team_xG_90min", "opponent_xGA_90min"]
        X_new_df = pd.DataFrame(X_new, columns=feature_names)

        # 8️⃣ Applica boost
        for feature, factor in boosts.items():
            X_new_df[feature] = X_new_df[feature] * factor

        # 9️⃣ Scala numeriche
        X_new_df[numeric_features] = scaler.transform(X_new_df[numeric_features])

        # 🔟 Aggiungi categoriche (posizioni)
        #X_new_df = pd.concat([X_new_df.reset_index(drop=True), pos_dummy_df.reset_index(drop=True)], axis=1)

        # 🔮 Predizione
        prob_goal = calib_model.predict_proba(X_new_df)[0, 1]
        pred = calib_model.predict(X_new_df)[0]

        print(f"✅ Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}. XGA avversaria:{opponent_xGA_90min:.2f}")

        results.append({
            "player": player,
            "team": team,
            "opponent": opponent,
            "prob_goal": prob_goal
        })

    for boost in boosts.items():
        print(boost)

    return pd.DataFrame(results)

def weighted_xg_vs_opponent(player_df, opponent_xGA_90min):
    """
    Calcola uno xG medio del giocatore pesato per la forza dell'avversario (xGA_90min).
    """
    # media xG del giocatore nelle ultime 10 partite
    base_xG = (player_df["sum_xG"].tail(10).mean()) 

    # forza media degli avversari affrontati nelle ultime 10 partite
    avg_opponent_xGA = player_df["opponent_xGA_90min"].tail(10).mean()

    # se mancano valori, fallback alla media
    if pd.isna(base_xG) or pd.isna(avg_opponent_xGA):
        return base_xG

    # calcola fattore di correzione
    # se l’avversario concede più del normale → boost
    # se concede meno → penalità
    factor = opponent_xGA_90min / avg_opponent_xGA

    # limitiamo il fattore per non esplodere
    factor = np.clip(factor, 0.5, 1.5)

    # xG pesato
    weighted_xG = base_xG * factor
    return weighted_xG

# funzione per applicare i pesi stagionali per ogni giocatore
def weight_last_3_seasons(group):
    # ordina le stagioni del giocatore in ordine crescente
    group = group.sort_values("season")
    
    # stagioni da pesare: le 3 precedenti rispetto alla stagione corrente
    for offset, weight in zip([3, 2, 1], [1.5, 2, 3]):
        target_season = current_season - offset
        mask = group["season"] == target_season
        group.loc[mask, cols_to_weight] = group.loc[mask, cols_to_weight] * weight
        
    return group

def map_strength(team: str) -> str:
    if not isinstance(team, str):
        return "unknown"
    team_lower = team.lower().strip()
    if any(sa.lower() in team_lower for sa in top_teams):
        return 3
    elif any(sa.lower() in team_lower for sa in mid_teams):
        return 2
    elif any(sa.lower() in team_lower for sa in weak_teams):
        return 1
    else:
        return 0

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

def clean_position(pos):
    # Restituisce "D", "M" o "F" in base al ruolo principale, altrimenti "None"
    roles = re.findall(r"[DMF]", str(pos).upper())
    if "F" in roles:
        return "F"
    elif "M" in roles:
        return "M"
    elif "D" in roles:
        return "D"
    else:
        return "None"

def get_positions(player_df, pos_dummies_index):
    # 1. Prendo l'ultima posizione nota del giocatore
    player_pos = player_df["position"].iloc[-1]

    # 2. Creo le dummies con le stesse colonne usate nel training
    player_pos = clean_position(player_df["position"].iloc[-1])
    pos_dummies = pd.get_dummies([f"pos_{player_pos}"], dtype=int)

    # Se nel training c'erano più colonne, devo riallinearle:
    pos_feature_names = pos_dummies_index  # <- quelle usate nel train
    for col in pos_feature_names:
        if col not in pos_dummies:
            pos_dummies[col] = 0

    # Riordino le colonne come nel training
    pos_dummies = pos_dummies[pos_feature_names]

    return pos_dummies

def multicoll_check(X, features):

    X = X[features].dropna()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)

# calcolo media stagionale per giocatore
def get_shot_conversion_mean(df):

    df["n_shots"] = df["n_shots"].replace(0, np.nan)  # evitiamo div/0
    df["shot_conversion_rate"] = df["goals"] / df["n_shots"]

    conversion_mean = (
        df.groupby(["player", "season"], as_index=False)["shot_conversion_rate"]
        .mean()
        .rename(columns={"shot_conversion_rate": "shot_conversion_rate_mean"})
    )

    # uniamo al df principale
    df = df.merge(conversion_mean, on=["player", "season"], how="left")

    # eventuali NaN (es. giocatori senza tiri in una stagione) → 0
    df["shot_conversion_rate_mean"] = df["shot_conversion_rate_mean"].fillna(0)

    df = df.drop(columns=["shot_conversion_rate"])

    return df

df_orig = pd.read_csv(".\PROD_shots_2025_preproc_Serie_A.csv")
df_teams = pd.read_csv("teams_2014_2025.csv")

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
df= df.drop(columns=[ "xG_per90","player", "match_id", "player_team", "opponent_team", "date", "xG_cummean", "is_home","games","time","goals_total","xG","assists","xA","shots","key_passes","npg","npxG","xGChain","xGBuildup", "goals_per90", "shots_per90"])

#analisi statistica
#correlazione tra variabili numeriche
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
#plt.show(block=True)

#df['player_team_strength'] = df['player_team_strength'].map(map_strength_dict)
#df['opponent_team_strength'] = df['opponent_team_strength'].map(map_strength_dict)

#fillna con 0 dei last5 in alternativa media delle rispettive colonne
'''
df["xG_last5"] = df["xG_last5"].fillna(df["sum_xG"].mean())
df["shots_last5"] = df["shots_last5"].fillna(df["n_shots"].mean())
df["goals_last5"] = df["goals_last5"].fillna(df["goals"].mean())
df["opponent_xGA_90min"] = df["opponent_xGA_90min"].fillna(df["opponent_xGA_90min"].mean())
df["team_xG_90min"] = df["team_xG_90min"].fillna(df["team_xG_90min"].mean())
'''
#droppo righe con nan
cols_to_check = [
    "n_shots",
    "xG_last5",
    "shots_last5",
    "goals_last5",
    "opponent_xGA_90min",
    "team_xG_90min"
]

#df = df.dropna(subset=cols_to_check)
df[cols_to_check] = df[cols_to_check].fillna(0)

#multicoll_check(df,["sum_xG", "n_shots"])

#********** POSIZIONE *************
print(df["position"].unique())

# applica al dataset
df["position"] = df["position"].apply(clean_position)

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

#corr = df.corr(numeric_only=True)
#plt.figure(figsize=(12, 10))
#sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#plt.title("Correlation Matrix")
#plt.show(block=True)

#gestisco la y, ossia i goal trasformandola in booleana, se gol>0 allora 1, altrimenti 0
df["goals"] = (df["goals"] > 0).astype(int)

# Seleziona le features (X) e target (y)
y = df["goals"]
y_binary = (y > 0).astype(int)
X = df.drop(columns=["goals"])

#******* boosting feature stato di forma giocatore (last5) e media cumulativa (cummean) *********

mask = X["season"] == 2025

# prima
print("Prima:", X.loc[mask, "xG_last5"].head())

for feature, factor in boosts.items():
    X.loc[mask, feature] = multiply_by_factor(X.loc[mask, feature], factor=factor)

# dopo
print("Dopo:", X.loc[mask, "xG_last5"].head())

X = X.drop(columns=["season"])

#********** Standardizzazione feature numeriche, tolgo se uso xgboost o random forest(non lineari) *********

numeric_features = ["sum_xG", "n_shots", "xG_last5", "shots_last5",
                    "goals_last5","opponent_xGA_90min",
                    "team_xG_90min"]

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

#*********    trovo i nan in X  *********
if X.isnull().values.any():
    print("Ci sono valori NaN in X")
    print(X[X.isnull().any(axis=1)])
    X = X.fillna(0)
    print("Dopo il fillna:")
    print(X[X.isnull().any(axis=1)])

# Dividi il dataset in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

#BILANCIO CON SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Crea e addestra il modello di regressione logistica

model = LogisticRegression()                #class_weight="balanced"

# modello base
'''
model = RandomForestClassifier(
    n_estimators= 500,
    min_samples_split= 5,
    min_samples_leaf= 1,
    max_features= 'sqrt',
    max_depth= 5,
    #class_weight='balanced_subsample'
)
'''

model.fit(X_train_res, y_train_res)

calib_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calib_model.fit(X_train, y_train)

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None]
}
'''
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='recall',
    cv=3,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
print(search.best_params_)
'''

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

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

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
#for i in range(20):
    #print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}, Probab: {X_test['probabilità'].iloc[i]:.4f}")

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

# esempio: soglia per recall >= 0.7
#target_precisions = 0.6
#idx = (precisions >= target_precisions).argmax()
#best_thr = thresholds[idx]
#print("Soglia ottimale:", best_thr)

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

# Trova soglia che massimizza recall (senza azzerare la precisione)
#best_idx = np.argmax(f1_scores)
#print(f"Soglia migliore per F1: {thresholds[best_idx]:.3f}")


#input utente

pred_df = predict_goal_probabilities(players, teams, opponents,
                                     df_orig, df_teams,
                                     calib_model, scaler, boosts,
                                     pos_dummies, numeric_features)


'''
player = input("Inserisci il nome del giocatore: ")
team = input("Inserisci la squadra del giocatore: ")
opponent = input("Inserisci la squadra avversaria: ")
is_home = int(input("Il giocatore gioca in casa? (1 per sì, 0 per no): "))

player = 'pavlovic'   #Lautaro Martínez
team = 'ac milan'
opponent = 'juventus'
#is_home = 1
# 1. Recupera storico del giocatore
player_df = df_orig[df_orig["player"].str.contains(player, case=False)].sort_values("date")

# 2. Calcola fillNa

player_df["xG_last5"] = player_df["xG_last5"].fillna(player_df["sum_xG"].mean())
player_df["shots_last5"] = player_df["shots_last5"].fillna(player_df["n_shots"].mean())
player_df["goals_last5"] = player_df["goals_last5"].fillna(player_df["goals"].mean())
player_df["opponent_xGA_90min"] = player_df["opponent_xGA_90min"].fillna(player_df["opponent_xGA_90min"].mean())
player_df["team_xG_90min"] = player_df["team_xG_90min"].fillna(player_df["team_xG_90min"].mean())

#*** DROP PARTITE FUTURE ***
now = pd.Timestamp.now()
# converto in datetime se non lo è già
player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
player_df = player_df[player_df["date"] <= now].reset_index(drop=True)

player_df[cols_to_check] = player_df[cols_to_check].fillna(0)
#player_df = player_df.dropna(subset=cols_to_check)

#*** shot conv mean*****
#player_df = get_shot_conversion_mean(player_df)

# 3. Determina forza squadre
#player_team_strength = map_strength(team)
#opponent_team_strength = map_strength(opponent)
opponent_xGA_90min = get_Xga_90min_opp_team(opponent, player_df["season"].iloc[-1], df_teams)
player_team_xG_90min = get_Xg_90min_team(team, player_df["season"].iloc[-1], df_teams)

#4. prevedo Xg e shots della partita futura ***CAMBIARE, PRENDE LA MEDIA DI TUTTI I GIOCATORI ***
#calcolo una media di quanti shots e xg fa in media il giocatore (non ho i dati della partita futura)
#qua dovrei applicare un coefficiente di forma in base alla stagione in corso e avversario
#prendo la colonna sum xg del giocatore
sum_xG_new = player_df["sum_xG"].mean() 
n_shots_new = player_df["n_shots"].mean()

#dati carriera in serie a
#xG_per90 = player_df["xG_per90"].mean()

#aggiungo positions
pos_dummies = get_positions(player_df, pos_dummies.columns)

# 5. Costruisci la riga finale
X_new = [[sum_xG_new, n_shots_new,
          player_df["xG_last5"].iloc[-1],player_df["shots_last5"].iloc[-1], player_df["goals_last5"].iloc[-1] ,    
          opponent_xGA_90min, player_team_xG_90min
         
          ]]  

feature_names = ["sum_xG", "n_shots",
                 "xG_last5","shots_last5","goals_last5",
              
                 "team_xG_90min", "opponent_xGA_90min"]

X_new_df = pd.DataFrame(X_new, columns=feature_names)

#boost
for feature, factor in boosts.items():
    X_new_df[feature] = multiply_by_factor(X_new_df[feature], factor=factor)

#******  scalo variabili numeriche  ******
X_new_df[numeric_features] = scaler.transform(X_new_df[numeric_features])

#concateno le categoriche
#X_new_df = pd.concat([X_new_df.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

# 5. Predici
pred = calib_model.predict(X_new_df)[0]
prob_goal = calib_model.predict_proba(X_new_df)[0,1]
print(pred)
print(f"Probabilità che {player} segni contro {opponent}: {prob_goal:.2f}")
'''