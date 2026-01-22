from statistics import LinearRegression
import unicodedata
import joblib
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import config
import re
import utils
from unidecode import unidecode

def add_fanta_role(df_main, df_fanta_roles, debug=True):

    def assign_manual_roles(df, manual_roles):
        """
        Assegna ruoli manuali a giocatori specifici mantenendo gli indici originali.

        Args:
            df (pd.DataFrame): DataFrame principale con colonna 'player_norm' già normalizzata.
            manual_roles (dict): mappa {nome: ruolo}, esempio {'keinan davis': 'A', ...}

        Returns:
            pd.DataFrame: df con fanta_role aggiornato senza modificare gli indici.
        """

        # Assicuriamoci che fanta_role esista
        if 'fanta_role' not in df.columns:
            df['fanta_role'] = None

        # Normalizza chiavi del manual_roles
        manual_roles_norm = {k.lower(): v for k,v in manual_roles.items()}

        # Assegna i ruoli manuali
        for name_norm, role in manual_roles_norm.items():
            mask = df['player_norm'].str.contains(name_norm, regex=False, na=False)
            df.loc[mask, 'fanta_role'] = role

        # Mantieni gli indici originali senza droppare righe
        return df
    
    # -----------------------
    # Normalizzazione
    # -----------------------

    def normalize_fn(name):
        if not isinstance(name, str):
            return ""
        name = name.lower()
        special_map = {
            'ø':'o','æ':'ae','œ':'oe','ß':'ss','þ':'th',
            'č':'c','ć':'c','š':'s','ž':'z','đ':'d','ğ':'g',
            'ł':'l','ń':'n','ř':'r','ě':'e','ť':'t','ď':'d',
            'á':'a','à':'a','ä':'a','â':'a','é':'e','è':'e','ë':'e','ê':'e',
            'í':'i','ì':'i','ï':'i','î':'i','ó':'o','ò':'o','ö':'o','ô':'o',
            'ú':'u','ù':'u','ü':'u','û':'u','ñ':'n'
        }
        for k,v in special_map.items():
            name = name.replace(k,v)
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    PREFIXES = {'de','da','di','del','van','von','der','le','la','el','al','du','ze'}

    # -----------------------
    # Chiave: cognome completo senza iniziale
    # -----------------------
    def make_key_name(name):
        tokens = name.split()
        if not tokens:
            return ""
        # ignora prefissi iniziali
        first_token_idx = 0
        while first_token_idx < len(tokens) and tokens[first_token_idx] in PREFIXES:
            first_token_idx += 1
        if first_token_idx >= len(tokens):
            first_token_idx = 0

        # ultimo token è iniziale tipo "V."? togli punto
        last_token = tokens[-1]
        if len(last_token) == 2 and last_token.endswith('.'):
            last_token = last_token[0]

        # cognome = tutti i token da first_token_idx fino a penultimo se ultimo è iniziale
        if len(tokens) > 1 and last_token != tokens[-1]:
            surname_tokens = tokens[first_token_idx:-1]
        else:
            surname_tokens = tokens[first_token_idx:]

        key = " ".join(surname_tokens)
        return key.strip()

    # -----------------------
    # Copie difensive
    # -----------------------
    df = df_main.copy()
    df_roles = df_fanta_roles.copy()

    # Normalizza nomi e squadre
    df['player_norm'] = df['player'].apply(normalize_fn)
    df['team_norm'] = df['player_team'].apply(normalize_fn)
    df_roles['player_norm'] = df_roles['Nome'].apply(normalize_fn)
    df_roles['team_norm'] = df_roles['Squadra'].apply(normalize_fn)

    # Chiave per match
    df['key_name'] = df['player_norm'].apply(make_key_name)
    df_roles['key_name'] = df_roles['player_norm'].apply(make_key_name)

    # Conta duplicati sul cognome
    surname_counts = df_roles['key_name'].value_counts().to_dict()

    # Mappa key_name -> (ruolo, squadra se duplicato)
    role_map = {}
    for _, row in df_roles.iterrows():
        key = row['key_name']
        if surname_counts.get(key,0) > 1:
            role_map[key] = (row['R'], row['team_norm'])
        else:
            role_map[key] = (row['R'], None)

    # -----------------------
    # Inizializza fanta_role
    # -----------------------
    df['fanta_role'] = None

    # Mantieni SUB invariato
    if 'position' in df.columns:
        df.loc[df['position'].str.upper()=='SUB', 'fanta_role'] = 'SUB'

    # -----------------------
    # Match
    # -----------------------
    mask = df['fanta_role'].isna()
    for key, (role, team) in role_map.items():
        if "martinez" in key or "lautaro" in key:
            print("debug")
        mask2 = mask & df['key_name'].str.contains(key, regex=False)
        if team:  # se duplicato, usa squadra come filtro
            mask2 = mask2 & (df['team_norm'] == team)
        df.loc[mask2,'fanta_role'] = role

    # -----------------------
    # Debug
    # -----------------------
    if debug:
        df['debug_reason'] = None
        df.loc[df['fanta_role'].notna(),'debug_reason'] = 'MATCH_OK'
        df.loc[df['fanta_role'].isna(),'debug_reason'] = 'NO_MATCH'
        print("\n📊 FANTA ROLE DEBUG REPORT")
        print(f"Totale giocatori: {len(df)}")
        print(f"Match riusciti: {(df['debug_reason']=='MATCH_OK').sum()}")
        print(f"SUB: {(df['fanta_role']=='SUB').sum()}")
        print(f"Senza match: {(df['debug_reason']=='NO_MATCH').sum()}")
        problems = df[df['debug_reason']=='NO_MATCH']
        if not problems.empty:
            print("\n⚠️ Esempi senza match:")
            print(problems[['player','player_team','key_name']].head(10))
        # aggiungi stampa di nomi unioci che non hanno match
        unmatched_keys = problems['key_name'].unique()
        print("\n⚠️ Key names senza match:")
        for uk in unmatched_keys:
            print(f" - {uk}")
    # Pulizia colonne temporanee
    df.drop(columns=['team_norm','key_name','debug_reason'], inplace=True, errors='ignore')
    df = assign_manual_roles(df, config.manual_roles)
    return df

def normalize_team(name):
    if pd.isna(name):
        return None
    return name.strip().lower()

def map_strength(team):
    if team is None:
        return 'weak'
    team_norm = str(team).strip().lower()
    top_teams = [t.strip().lower() for t in config.TOP_TEAMS]
    mid_teams = [t.strip().lower() for t in config.MID_TEAMS]
    if team_norm in top_teams:
        return 'top'
    elif team_norm in mid_teams:
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

def safe_mean(col, player_df, rolling_5):
        if col in rolling_5.columns and rolling_5[col].notna().any():
            return rolling_5[col].mean()
        elif col in player_df.columns and player_df[col].notna().any():
            return player_df[col].mean()
        else:
            return 0.0

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
    if "Sub" in pos:
        return "SUB"
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
        'position_clean'
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

def pred_voto_prod(players, teams, opponents, h_a_players, df, model, preprocessor=None):
    # --- Carica dataset e modelli
    models_goal = utils.load_models()
    models_assist = utils.load_models_assist()
    model_xg = utils.load_xg_model()
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)
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

        #preprcessing come prima del training
        

        fanta_role = utils.get_main_position_weighted(player_df["fanta_role"], window=10, decay=0.8)
        real_role = utils.get_main_position_weighted(player_df["position_clean"], window=10, decay=0.8)
        # ---- rolling stats ultime 20 ----
        rolling_20 = player_df.tail(20)

        if "paz" in player_full_name.lower():
            print("debug")
        
        voto_base = utils.compute_base_voto_by_role(
           player_df=player_df,
            role=fanta_role
        )

        # aggiustamento in base alla forza dell'avversario
        opponent = normalize_team(opponent)
        opponent_strength = map_strength(opponent) 
        team_strength = map_strength(team)

        adj_opp_team = utils.compute_player_vs_strength_adjustment(
                player_df=player_df,
                target_opponent_strength=opponent_strength
        )
        
        #correzione in base a come la squadra del giocatore performa
        adj_team_player = utils.compute_player_team_strength_adjustment(
            player_df=player_df,
            target_team_strength=team_strength
        )

        #correzione in base a come il giocatore performa in casa o trasferta
        adj_home_away = utils.compute_player_home_away_adjustment(
                player_df=player_df,
                target_ha=h_a
        )    

        if fanta_role == 'D' or (fanta_role == 'C' and real_role == 'M'):
            consistency_adj = utils.compute_consistency_adjustment(player_df)
            voto_base += adj_opp_team + adj_home_away + adj_team_player + consistency_adj
        else:
            voto_base = voto_base + adj_opp_team + adj_home_away + adj_team_player

        goal_impact = utils.compute_feature_role_impact(
            player_df,
            fanta_role,
            config.ROLE_WEIGHTS_GOAL,
            feature_col='goals'
        )

        # === PREDIZIONE GOAL ===
        features_names_goal = list(models_goal["poiss_reg"].feature_names_)
        if "finishing_form_resid" in features_names_goal:
            features_names_goal.remove("finishing_form_resid")

        goal_proba = utils.get_goal_prob(
                model_xg["poisson_regressor_xg"],
                models_goal["poiss_reg"],
                features_names_goal,
                player, team, opponent, df_orig_goal, df_teams,
                df_teams_curr_season, models_goal["lin"], config.ROLE_STATS,
                h_a
        )

        goal_feature = goal_proba * goal_impact

        # === PREDIZIONE ASSIST ===
        features_names_assist = models_assist["poisson_reg_assist"].feature_names_
        assist_proba = utils.get_assist_prob(
                models_assist["poisson_reg_assist"], features_names_assist,
                player, team, opponent, df_orig_assist, df_teams,
                df_teams_curr_season, h_a
        )
   
        assist_impact = utils.compute_feature_role_impact(
            player_df,
            fanta_role,
            config.ROLE_WEIGHTS_ASSIST,
            feature_col='assists'
        )
        assist_feature = assist_proba * assist_impact

        print(f"Voto base pesato: {voto_base:.2f}")

        # ---- costruzione features pre-match ----
        X_pred = pd.DataFrame([{
            'voto_gds': voto_base,
            'goals': goal_feature,
            'assists': assist_feature,
            'ammonizioni': rolling_20['ammonizioni'].mean() if 'ammonizioni' in rolling_20.columns else 0.0,
            # 'espulsioni': rolling_20['espulsioni'].mean() if 'espulsioni' in rolling_20.columns else 0.0,
            'position_clean': fanta_role
        }])

        # fallback posizione se NaN
        if pd.isna(X_pred['position_clean'].iloc[0]):
            X_pred['position_clean'] = player_df['position_clean'].mode().iloc[0]

        # --- Gestione categoriche e standardizzazione come nel training ---
        if preprocessor is not None:
            # Se il model è una pipeline, NON applicare preprocessor.transform!
            fantavoto_pred = model.predict(X_pred)[0]
        
        #fantavoto_pred = utils.adjust_fantavoto_by_opp(fantavoto_pred, opponent_strength, h_a)
        #correzione in base a come il giocatore performe contro squadre di diversa forza  
 
        print(f"Predicted fantavoto for {player_full_name}, role: {fanta_role}, ({team} vs {opponent}, {h_a}): {fantavoto_pred:.2f}")

        index = utils.fantavoto_to_schierability_index(fantavoto_pred, fanta_role, config.ROLE_FANTAVOTO_STATS)  

        print(f"Schierability index: {index:.2f}")

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
  
    #bestmodel,params = utils.tune_catboost_regressor(X_train, y_train, cat_features=["position_clean"])
    #print(f"Best hyperparameters found:{params}")
 
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.01,
        random_state=42,
        early_stopping_rounds=50,
        cat_features=["position_clean"],
        loss_function='RMSE'
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
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

def train_log_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Allena un modello di regressione lineare e stampa MAE e MSE"""

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Suddividi in train e test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # PREPROCESSING E MODELLO
    # ===============================

    # Identifica feature numeriche e categoriche
    NUM_FEATURES = [
        'voto_gds',
        'goals',
        'assists',
        'ammonizioni'
    ]
    CAT_FEATURES = ['position_clean']
    # ---- preprocessors ----
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )   
    categorical_transformer = Pipeline(
        steps=[(
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            )
        )]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )
    # ---- model ----
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )
    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("\n=== TRAIN vs TEST ===")
    print(f"TRAIN -> MAE: {mean_absolute_error(y_train, y_train_pred):.4f} | "
          f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"TEST  -> MAE: {mean_absolute_error(y_test, y_test_pred):.4f} | "
          f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    
    return model, preprocessor

def main():

    train = False
    csv_path = config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI
    df_fanta_roles_path = config.DATASET_DATA_DIR / config.FANTA_RUOLI_FILE

    df_voti = load_data(csv_path)
    df_fanta_roles = load_data(df_fanta_roles_path)

     # Aggiungi fanta_role
    df_voti = add_fanta_role(df_voti, df_fanta_roles)

    X, y = preprocess_data(df_voti)

    if train:
        model = train_xgboost(X, y)
        #salva modello
        MODEL_PATH = config.MODEL_DIR_FV / config.FV_MODEL
        #chiedi all'utente se vuole salvare il modello
        if MODEL_PATH.exists():
                overwrite = input(f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
                if overwrite != "y":
                    print("❌ Salvataggio modello annullato.")
                else:
                    joblib.dump(model, MODEL_PATH)
                    print(f"✅ Modello sovrascritto in: {MODEL_PATH}")
        else:
            joblib.dump(model, MODEL_PATH)
            print(f"✅ Modello salvato in: {MODEL_PATH}")

    else:
        #carica modello
        model = utils.load_fv_model()

    model, scaler = train_log_regression(X, y)

    pred_df = pred_voto_prod(
        config.INPUT["players"],
        config.INPUT["teams"],
        config.INPUT["opponents"],
        config.INPUT["h_a"],
        df_voti,
        model,
        scaler
        )

if __name__ == "__main__":
    main()
