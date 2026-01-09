import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import config
import re
import utils
from unidecode import unidecode

def normalize_team(name):
    if pd.isna(name):
        return None
    return name.strip().lower()

def map_strength(team):
    if team in config.TOP_TEAMS:
        return 'top'
    elif team in config.MID_TEAMS:
        return 'mid'
    else:
        return 'weak'


def add_team_strength_column(
    df,
    team_col,
    new_col='team_strength'
):
    """
    Aggiunge una colonna con la forza della squadra (top / mid / weak)

    Parameters
    ----------
    df : pd.DataFrame
    team_col : str
        Nome della colonna contenente le squadre
    new_col : str
        Nome della nuova colonna (default: team_strength)
    """

    df[new_col] = (
        df[team_col]
        .apply(normalize_team)
        .apply(map_strength)
    )

    return df

def load_data(csv_path: str) -> pd.DataFrame:
    """Carica il dataset da CSV"""
    return pd.read_csv(csv_path)


def clean_position(pos):
    pos = str(pos).upper()
    if "GK" in pos:
        return "GK"
    # Se contiene sia M che D, metti in M
    if "M" in pos and "D" in pos:
        return "M"
    if "FW" in pos:
        return "FW"
    if "AM" in pos:
        return "AM"
    if "M" in pos:
        return "M"
    if "D" in pos:
        return "D"
    return pos

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

def preprocess_data(df: pd.DataFrame):
    """
    Seleziona le feature e il target.
    Rimuove le righe con NaN nelle colonne usate.
    """
    df = add_team_strength_column(df, 'opponent_team', 'opponent_team_strength')
    df = add_team_strength_column(df, 'player_team', 'player_team_strength')

    features = [
        'voto_gds',
        'goals',
        'assists',
        'ammonizioni',
        'espulsioni',
        'rig_segnati',
        'rig_sbagliati',
        'position_clean',
        "opponent_team_strength",
        'player_team_strength'
    ]

    target = 'fantavoto'

    #per ora teniamo solo le colonne con season 2025
    df = df[df['season'] == config.CURRENT_SEASON]

    # Applica la pulizia della posizione
    if 'position' in df.columns:
        df['position_clean'] = df['position'].apply(clean_position)
        # Esempio: media fantavoto per posizione pulita
        print("\nMedia fantavoto per posizione pulita:")
        print(df.groupby('position_clean')['fantavoto'].mean().sort_values(ascending=False))

    
        # Tieni solo le colonne necessarie
    df_model = df[features + [target]].dropna()

      # Aggiungi il DataFrame delle squadre se disponibile

    X = df_model[features]
    y = df_model[target]

    return X, y

def pred_voto_prod(players, teams, opponents, h_a_players, df, model):

    predictions = []

    # lavoro solo sulla stagione corrente
    df = df[df['season'] == config.CURRENT_SEASON].copy()
    df['date'] = pd.to_datetime(df['date'])

    # pulizia posizione
    df['position_clean'] = df['position'].apply(clean_position)

    # media globale per posizione (fallback finale)
    pos_means = df.groupby('position_clean')['fantavoto'].mean()

    for player, team, opponent, h_a in zip(players, teams, opponents, h_a_players):

        player_df, player_full_name = get_player_data(df, player)
        if player_df.empty:
            continue

        player_df = player_df.sort_values('date')

        player_df = utils.add_home_away_column(player_df)

        # ---- rolling stats ultime 5 ----
        rolling_5 = player_df.tail(5)

        def safe_mean(col):
            if col in rolling_5.columns and rolling_5[col].notna().any():
                return rolling_5[col].mean()
            elif col in player_df.columns and player_df[col].notna().any():
                return player_df[col].mean()
            else:
                return 0.0

        # ---- costruzione features pre-match ----
        X_pred = pd.DataFrame([{
            'voto_gds': safe_mean('voto_gds'),
            'goals': safe_mean('goals'),
            'assists': safe_mean('assists'),
            'ammonizioni': safe_mean('ammonizioni'),
            'espulsioni': safe_mean('espulsioni'),
            'rig_segnati': safe_mean('rig_segnati'),
            'rig_sbagliati': safe_mean('rig_sbagliati'),
            'position_clean': rolling_5['position_clean'].mode().iloc[0]
        }])

        # fallback posizione se NaN
        if pd.isna(X_pred['position_clean'].iloc[0]):
            X_pred['position_clean'] = player_df['position_clean'].mode().iloc[0]

        # ---- encoding posizione (come in training!) ----
        X_pred = pd.get_dummies(X_pred, columns=['position_clean'], drop_first=True)

        # allinea colonne con il training
        for col in model.feature_names_:
            if col not in X_pred.columns:
                X_pred[col] = 0

        X_pred = X_pred[model.feature_names_]
        # ---- predizione ----
        fantavoto_pred = model.predict(X_pred)[0]
        # aggiustamento in base alla forza dell'avversario
        opponent = normalize_team(opponent)
        opponent_strength = map_strength(opponent)
        #fantavoto_pred = utils.adjust_fantavoto_by_opp(fantavoto_pred, opponent_strength, h_a)

        adj_opp_team = utils.compute_player_vs_strength_adjustment(
                player_df=player_df,
                target_opponent_strength=opponent_strength
            )
        adj_home_away = utils.compute_player_home_away_adjustment(
                player_df=player_df,
                target_ha=h_a
            )

        #da aggiungere aadj per squadra in cui gioca
        
        
        fantavoto_pred += adj_opp_team + adj_home_away

        print(f"Predicted fantavoto for {player_full_name} ({team} vs {opponent}, {h_a}): {fantavoto_pred:.2f}")

        predictions.append({
            'player': player_full_name,
            'team': team,
            'opponent': opponent,
            'home_away': h_a,
            'fantavoto_pred': round(float(fantavoto_pred), 2)
        })

    return pd.DataFrame(predictions)


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Allena un modello XGBoost e stampa RMSE"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=1000,
        depth=3,
        learning_rate=0.05,
        random_state=42,
        early_stopping_rounds=40,
        cat_features=["position_clean", "opponent_team_strength", "player_team_strength"]
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        #eval_metric='rmse',
        #early_stopping_rounds=20,
        verbose=False
    )

    print("\nBest iteration:", model.get_best_iteration())
    importance = model.get_feature_importance(prettified=True)
    print(importance.head(10))

    # ===================== TRAIN vs TEST METRICS =====================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
        f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
        f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
    #stampa 20 esmpi di predizioni
    print("\n=== Esempi di predizioni ===")
    for i in range(20):
        print(f"Predicted: {y_test_pred[i]:.2f} | Actual: {y_test.iloc[i]:.2f}")

    return model


def main():

    csv_path = config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI
    df_voti = load_data(csv_path)

    X, y = preprocess_data(df_voti)
    model = train_xgboost(X, y)

    pred_df = pred_voto_prod(
        config.INPUT["players"],
        config.INPUT["teams"],
        config.INPUT["opponents"],
        config.INPUT["h_a"],
        df_voti,
        model
        )



if __name__ == "__main__":
    main()
