import joblib
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import config
from scipy.stats import skew, kurtosis

def save_models(model, scaler_xg, scaler, poly, lin_poly):
    """
    Salva il modello e lo scaler, chiedendo conferma se i file esistono già.
    """

    # Percorsi completi dei file
    model_path = config.MODEL_DIR / config.CALIB_LOGISTIC_REG if model is not None else None
    scaler_path = config.SCALER_DIR / config.SCALER if scaler is not None else None
    scaler_xg_path = config.SCALER_DIR / config.SCALER_XG if scaler_xg is not None else None
    poly_path = config.MODEL_DIR / config.POLY_TRANSFORMER if poly is not None else None
    lin_poly_path = config.MODEL_DIR / config.LIN_POLY if lin_poly is not None else None

    # Crea le cartelle se non esistono
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if scaler_path:
        os.makedirs(config.SCALER_DIR, exist_ok=True)
    if scaler_xg_path:
        os.makedirs(config.SCALER_DIR, exist_ok=True)
    if poly_path:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    if lin_poly_path:
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    # --- Salvataggio modello ---
    if model_path.exists():
        overwrite = input(f"⚠️ Il file '{model_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")
        else:
            joblib.dump(model, model_path)
            print(f"✅ Modello sovrascritto in: {model_path}")
    else:
        joblib.dump(model, model_path)
        print(f"✅ Modello salvato in: {model_path}")

    # --- Salvataggio scaler ---
    if scaler_path:
        if scaler_path.exists():
            overwrite = input(f"⚠️ Il file '{scaler_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio scaler annullato.")
                return
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler salvato in: {scaler_path}")
    # --- Salvataggio scaler_xg ---
    if scaler_xg_path:
        if scaler_xg_path.exists():
            overwrite = input(f"⚠️ Il file '{scaler_xg_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio scaler_xg annullato.")
                return
        joblib.dump(scaler_xg, scaler_xg_path)
        print(f"✅ Scaler XG salvato in: {scaler_xg_path}")
    # --- Salvataggio poly transformer ---
    if poly_path:
        if poly_path.exists():
            overwrite = input(f"⚠️ Il file '{poly_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio poly transformer annullato.")
                return
        joblib.dump(poly, poly_path)
        print(f"✅ Poly transformer salvato in: {poly_path}")
    # --- Salvataggio lin_poly model ---
    if lin_poly_path:
        if lin_poly_path.exists():
            overwrite = input(f"⚠️ Il file '{lin_poly_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
            if overwrite != "y":
                print("❌ Salvataggio lin_poly model annullato.")
                return
        joblib.dump(lin_poly, lin_poly_path)
        print(f"✅ Lin_poly model salvato in: {lin_poly_path}")

def save_model(model):
    # Percorsi completi dei file
    model_path = config.MODEL_DIR / config.XGB_MODEL_DIR if model is not None else None
    # --- Salvataggio modello ---
    if model_path.exists():
        overwrite = input(f"⚠️ Il file '{model_path.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
        if overwrite != "y":
            print("❌ Salvataggio modello annullato.")
        else:
            joblib.dump(model, model_path)
            print(f"✅ Modello sovrascritto in: {model_path}")
    else:
        joblib.dump(model, model_path)
        print(f"✅ Modello salvato in: {model_path}")

# trasformazione che moltiplica (usata dopo lo StandardScaler)
def multiply_by_factor(X, factor=2.0):
    return X * factor

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

def get_stat_desc(df, features, target_col):
    """
    Analisi descrittiva del dataset:
    - Distribuzione del target
    - Statistiche descrittive
    - Boxplot delle features
    - Matrice di correlazione e correlazioni col target

    Parametri:
    ----------
    df : pd.DataFrame
        Dataset completo
    features : list
        Lista delle colonne di feature numeriche
    target_col : str
        Nome della colonna target (es. 'assists')
    """

    # --- Distribuzione generale del target ---
    plt.figure(figsize=(5,4))
    sns.countplot(x=df[target_col])
    plt.title(f"Distribuzione del target ({target_col})")
    plt.xlabel(f"{target_col} (0 = no, 1 = sì)")
    plt.ylabel("Conteggio")
    plt.show()

    # --- Statistiche di base ---
    print("\n📈 Statistiche descrittive delle features numeriche:")
    print(df[features + [target_col]].describe().T)

    # --- Distribuzione delle feature principali ---
    df_melted = df[features].melt(value_vars=features)
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_melted, x="variable", y="value")
    plt.title("Distribuzione delle features (Boxplot)")
    plt.xticks(rotation=45)
    plt.show()

    # --- Correlazione numerica ---
    corr = df[features + [target_col]].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Matrice di correlazione con il target '{target_col}'")
    plt.show()

    # --- Correlazione diretta con il target ---
    corr_target = corr[target_col].sort_values(ascending=False)
    print(f"\n🔥 Correlazioni con il target '{target_col}':")
    print(corr_target)


def analyze_feature_skewness(df, feature_cols, plot=True):
    """
    Analizza l'asimmetria (skewness) e la curtosi delle features numeriche.
    Suggerisce trasformazioni log/sqrt per ridurre la skewness.
    """

    results = []

    for col in feature_cols:
        data = df[col].dropna()

        sk = skew(data)
        kt = kurtosis(data)

        # Suggerimento di trasformazione
        if sk > 1:
            suggestion = "log1p"
        elif 0.5 < sk <= 1:
            suggestion = "sqrt"
        else:
            suggestion = "none"

        results.append({
            "feature": col,
            "skewness": sk,
            "kurtosis": kt,
            "suggested_transform": suggestion
        })

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(data, bins=30, ax=axes[0], kde=True)
            axes[0].set_title(f"Distribuzione originale: {col}")

            if suggestion != "none":
                if suggestion == "log1p":
                    transformed = np.log1p(data)
                else:
                    transformed = np.sqrt(data)
                sns.histplot(transformed, bins=30, ax=axes[1], kde=True)
                axes[1].set_title(f"Distribuzione {suggestion}: {col}")
            else:
                axes[1].set_visible(False)

            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results).sort_values(by="skewness", ascending=False)
    print("\n📊 Analisi asimmetria delle feature numeriche:")
    print(results_df)
    return results_df

def shap_explanation(model, df, feature_columns):
    # 1. Assicurati che tutte le feature siano numeriche
    X = df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors='coerce')

    # 2. Rimuovi eventuali NaN creati da conversione
    X = X.fillna(0)

    # 3. Crea explainer e calcola shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 4. Plot
    shap.summary_plot(shap_values, X)

def process_positions(df, position_col="position"):
    """
    Pulisce e trasforma la colonna 'position' in un DataFrame, applicando il one-hot encoding.

    Parametri:
    ----------
    df : pd.DataFrame
        Il DataFrame contenente la colonna delle posizioni.
    position_col : str, default="position"
        Il nome della colonna che contiene le posizioni.

    Ritorna:
    --------
    pd.DataFrame
        Il DataFrame con le posizioni trasformate in colonne one-hot encoded.
    """
    # Stampa i valori unici iniziali (debug)
    print(f"Valori unici iniziali in '{position_col}':", df[position_col].unique())

    # Pulisci la colonna delle posizioni
    df[position_col] = df[position_col].apply(clean_position)

    # Stampa i valori unici dopo la pulizia (debug)
    print(f"Valori unici dopo la pulizia in '{position_col}':", df[position_col].unique())

    # Rimuovi valori NaN
    df = df.dropna(subset=[position_col])

    # Conta le occorrenze (debug)
    counts = df[position_col].value_counts(dropna=False)
    print(f"Occorrenze delle posizioni:\n{counts}")

    # Rimuovi i valori "None"
    df = df[df[position_col] != "None"]

    # Applica il one-hot encoding
    pos_dummies = pd.get_dummies(df[position_col], prefix="pos", dtype=int)

    # Rimuovi la colonna originale
    #df = df.drop(columns=[position_col])

    # Aggiungi le colonne one-hot encoded al DataFrame
    df = pd.concat([df, pos_dummies], axis=1)

    return df