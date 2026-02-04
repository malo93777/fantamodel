from statistics import LinearRegression
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import config
import re
import numpy as np
import utils
from unidecode import unidecode
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")

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
        .apply(utils.normalize_team_name)
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

    #df["player_norm"] = df["player"].apply(utils.normalize_fn)
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
        return pd.DataFrame(), None

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
    #df = add_team_strength_column(df, 'opponent_team', 'opponent_team_strength')
    #df = add_team_strength_column(df, 'player_team', 'player_team_strength')

    features = [
        'voto_gds',
        'goals',
        'assists',
        'ammonizioni',
        'position_clean'
    ]

    target = 'fantavoto'

    #rimuovo tutti i Senza voto
    df = df[df['voto_gds'].notna()]

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

def preprocess_data_GK(df: pd.DataFrame, df_curr_teams: pd.DataFrame):
    """
    Seleziona le feature e il target.
    Rimuove le righe con NaN nelle colonne usate.
    """
    #df = add_team_strength_column(df, 'opponent_team', 'opponent_team_strength')
    #df = add_team_strength_column(df, 'player_team', 'player_team_strength')

    features = [
        'voto_gds',
        'goals',
        'ammonizioni',
        'player_team_strength'
    ]

    target = 'fantavoto'

    #PLAYER TEAM DATA
    #xGA_last5, GA_last5 = utils.get_xGA_last5_team_h_a_mean(opponent, "", df_curr_teams)

    df = df[df['fanta_role'] == 'P']

    #rimuovo tutti i Senza voto
    df = df[df['voto_gds'].notna()]

    #per ora teniamo solo le colonne con season 2025
    df = df[df['season'] == config.CURRENT_SEASON]
 
        # Tieni solo le colonne necessarie
    df_model = df[features + [target]].dropna()

      # Aggiungi il DataFrame delle squadre se disponibile

    X = df_model[features]
    y = df_model[target]

    return X, y

def pred_voto_prod(
        players,
        teams,
        opponents,
        h_a_players,
        df_voti,                
        df_orig_goal,
        df_orig_assist,
        df_teams,
        df_teams_curr_season,
        model_goal,
        model_assist,
        model_xg,
        pipeline
    ):

    predictions = []

    for player, team, opponent, h_a in zip(players, teams, opponents, h_a_players):
        if "jonathan" in player.lower():
            print(f"debug {player}")
        player_df, player_full_name = get_player_data(df_voti, player)
        if player_df.empty:
            continue
        
        #caso in cui giocatore ha cambiato squadra durante la stagione
        if isinstance(team, str) and "," in team:
            team = team.split(",")[-1].strip()

        player_df = player_df.sort_values('date')

        player_df = utils.add_home_away_column(player_df)     

        fanta_role = utils.get_main_position_weighted(player_df["fanta_role"], window=10, decay=0.8)
        real_role = utils.get_main_position_weighted(player_df["position_clean"], window=10, decay=0.8)
        
        # ---- rolling stats ultime 15 ----
        rolling_15 = player_df.tail(15)
        
        voto_base = utils.compute_base_voto_by_role(
           player_df=player_df,
            role=fanta_role
        )

        # aggiustamento in base alla forza dell'avversario
        opponent = utils.normalize_team_name(opponent)
        opponent_strength = map_strength(opponent) 
        team_strength = map_strength(team)

        adj_opp_team = utils.compute_player_vs_strength_adjustment(
                player_df=player_df,
                target_opponent_strength=opponent_strength
        )

        #correzione in base a come il giocatore performa in casa o trasferta
        adj_home_away = utils.compute_player_home_away_adjustment(
                player_df=player_df,
                target_ha=h_a
        ) 

        # *************  BONUS DIFENSORI  SE LORO SQUADRE concedono poco e MALUS se affrontano squadra che crea molto**************  
        # 4️⃣ Recupera dati della squadra e avversario
        if fanta_role == 'D':
            num_giornate = utils.count_matchdays(df_teams_curr_season)

            #se ho un numero sufficiente di giornate, applico discriminante home/away
            if num_giornate >= 15:

                #PLAYER TEAM DATA home/away
                xGA_last5, GA_last5 = utils.get_def_data_last5_team_h_a(team, h_a, df_teams_curr_season)
                xGA_last5 = xGA_last5/5    
            else:
                #PLAYER TEAM DATA
                xGA_last5, GA_last5 = utils.get_def_data_last5_team_h_a(team, "", df_teams_curr_season)
                xGA_last5 = xGA_last5/5

            bonus_defensive_adj = utils.compute_defensive_xga_bonus(
                fanta_role,
                team_xga_last5=xGA_last5,
                team_goal_against_last5=None,
                matchday=num_giornate,
                df_teams_curr_season=df_teams_curr_season
            )

            voto_base += bonus_defensive_adj

        # *************  AGGIUSTAMENTI CONSISTENZA PER DIFENSORI E MEDIANI **************

        if fanta_role == 'D' or (fanta_role == 'C' and real_role == 'M'):
            consistency_adj = utils.compute_consistency_adjustment(player_df)
            voto_base += adj_opp_team + adj_home_away + consistency_adj
        else:
            voto_base = voto_base + adj_opp_team + adj_home_away

        # ************  AGGIUSTAMENTO TENDENZA AL CARTELLINO GIALLO  **********
        yellowcard_adj = utils.compute_ammonizioni_adjustment(player_df, fanta_role)

        voto_base += yellowcard_adj
 
        # === PREDIZIONE GOAL ===
        features_names_goal = list(model_goal["poiss_reg"].feature_names_)
        if "finishing_form_resid" in features_names_goal:
            features_names_goal.remove("finishing_form_resid")

        #normalize name player
        norm_name = player_df['player_norm'].iloc[0]

        goal_proba = utils.get_goal_prob(
                model_xg["poisson_regressor_xg"],
                model_goal["poiss_reg"],
                features_names_goal,
                norm_name, team, opponent, df_orig_goal, df_teams,
                df_teams_curr_season, model_goal["lin"], config.ROLE_STATS,
                h_a
        )

        if goal_proba is None:
            print(f"⚠️ Impossibile calcolare la probabilità di goal per {player_full_name}. Impostata a 0.")
            goal_proba = 0.0

        goal_impact = utils.compute_feature_role_impact(
            player_df,
            fanta_role,
            config.ROLE_WEIGHTS_GOAL,
            feature_col='goals'
        )
        
        goal_feature = goal_proba * goal_impact

        # === PREDIZIONE ASSIST ===
        features_names_assist = model_assist["poisson_reg_assist"].feature_names_
        assist_proba = utils.get_assist_prob(
                model_assist["poisson_reg_assist"], features_names_assist,
                norm_name, team, opponent, df_orig_assist, df_teams,
                df_teams_curr_season, h_a
        )
        if assist_proba is None:
            print(f"⚠️ Impossibile calcolare la probabilità di assist per {player_full_name}. Impostata a 0.")
            assist_proba = 0.0

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
            'ammonizioni': rolling_15['ammonizioni'].mean() if 'ammonizioni' in rolling_15.columns else 0.0,
            'position_clean': fanta_role
        }])

        # fallback posizione se NaN
        if pd.isna(X_pred['position_clean'].iloc[0]):
            X_pred['position_clean'] = player_df['position_clean'].mode().iloc[0]

        # --- Gestione categoriche e standardizzazione come nel training ---
        if pipeline is not None:
            # Se il model è una pipeline, NON applicare preprocessor.transform!
            fantavoto_pred = pipeline.predict(X_pred)[0]

        print(f"Predicted fantavoto for {player_full_name}, role: {fanta_role}, ({team} vs {opponent}, {h_a}): {fantavoto_pred:.2f}")

        index = utils.fantavoto_to_schierability_index(fantavoto_pred, fanta_role, config.ROLE_FANTAVOTO_STATS)  

        if fanta_role == "C":
            print("debug")
        index_boost = utils.apply_fantarole_boost(index, fanta_role)

        print(f"Schierability REAl index: {index:.2f}")
        print(f"Schierability index BOOST: {index_boost:.2f}")

        ha_to_print = "casa" if h_a == "h" else "trasf."

        predictions.append({
        'Giocatore': player_full_name,
        'Index': index_boost,
        #'Squadra': team,
        'Avversario': opponent,
        'Campo': ha_to_print
        }) 

    return pd.DataFrame(predictions).reset_index(drop=True)

def pred_voto_prod_gk(
        players,
        teams,
        opponents,
        h_a_players,
        df_voti,                
        df_teams,
        df_teams_curr_season,
        pipeline
    ):

    predictions = []

    for player, team, opponent, h_a in zip(players, teams, opponents, h_a_players):

        player_df, player_full_name = get_player_data(df_voti, player)
        if player_df.empty:
            continue

        player_df = player_df.sort_values('date')

        player_df = utils.add_home_away_column(player_df)     

        fanta_role = utils.get_main_position_weighted(player_df["fanta_role"], window=10, decay=0.8)
        
        # ---- rolling stats ultime 15 ----
        rolling_15 = player_df.tail(15)
        
        voto_base = utils.compute_base_voto_by_role(
           player_df=player_df,
            role=fanta_role
        )

        # aggiustamento in base alla forza dell'avversario
        opponent = utils.normalize_team_name(opponent)
        opponent_strength = map_strength(opponent) 
        team_strength = map_strength(team)

        adj_opp_team = utils.compute_player_vs_strength_adjustment(
                player_df=player_df,
                target_opponent_strength=opponent_strength
        )

        #correzione in base a come il giocatore performa in casa o trasferta
        adj_home_away = utils.compute_player_home_away_adjustment(
                player_df=player_df,
                target_ha=h_a
        ) 

        # *************  BONUS SE LORO SQUADRE concedono poco e MALUS se affrontano squadra che crea molto**************  
        # 4️⃣ Recupera dati della squadra e avversario
        num_giornate = utils.count_matchdays(df_teams_curr_season)

        #se ho un numero sufficiente di giornate, applico discriminante home/away
        if num_giornate >= 15:

            h_a_opp = utils.get_h_a_opponent(h_a)

            #PLAYER TEAM DATA home/away
            xGA_last5, GA_last5 = utils.get_def_data_last5_team_h_a(team, h_a, df_teams_curr_season)
            GA_last5_per90= GA_last5/num_giornate

            #OPPONENT TEAM DATA
            xG_last5_team, Goal_last5_opponent = utils.get_att_data_last5_team_h_a(opponent, h_a_opp, df_teams_curr_season)
            Goal_last5_opponent_per90 = Goal_last5_opponent/5             
        else:
            #PLAYER TEAM DATA home/away
            xGA_last5, GA_last5 = utils.get_def_data_last5_team_h_a(team, "", df_teams_curr_season)
            GA_last5_per90= GA_last5/num_giornate

            #OPPONENT TEAM DATA
            xG_last5_team, Goal_last5_team = utils.get_att_data_last5_team_h_a(opponent, "", df_teams_curr_season)
            Goal_last5_team_per90 = Goal_last5_team/5

        bonus_defensive_adj = utils.compute_defensive_xga_bonus(
            fanta_role,
            team_xga_last5=None,
            team_goal_against_last5=GA_last5_per90,
            matchday=num_giornate,
            df_teams_curr_season=df_teams_curr_season,      
        )

        bonus_opponent_off_adj = utils.compute_opponent_offense_bonus(
            fanta_role,
            opponent_xg_last5=None,
            opponent_goal_last5=Goal_last5_opponent_per90,
            matchday=num_giornate,
            df_teams_curr_season=df_teams_curr_season,        
        )

        def_stats = utils.compute_clean_sheet(
            df_teams_curr_season,
            opponent_xg_last5=None,
            opponent_goal_last5=Goal_last5_opponent_per90,
        )

        voto_base += bonus_defensive_adj + bonus_opponent_off_adj + def_stats["bonus"]

        # *************  AGGIUSTAMENTI CONSISTENZA **************

        consistency_adj = utils.compute_consistency_adjustment(player_df)
        voto_base += adj_opp_team + adj_home_away + consistency_adj

        # ************  AGGIUSTAMENTO TENDENZA AL CARTELLINO GIALLO  **********
        yellowcard_adj = utils.compute_ammonizioni_adjustment(player_df, fanta_role)

        voto_base += yellowcard_adj 

        print(f"Voto base pesato: {voto_base:.2f}")

        #CALCOLO GOALS SUBITI PROSSIMA PARTITA
        gol_subiti_feature = def_stats["mean_ga_last5"] * def_stats["clean_sheet_prob"]

        # ---- costruzione features pre-match ----
        X_pred = pd.DataFrame([{
            'voto_gds': voto_base,
            'goals': gol_subiti_feature,
            'ammonizioni': rolling_15['ammonizioni'].mean() if 'ammonizioni' in rolling_15.columns else 0.0,
            'player_team_strength': team_strength
        }])

        # --- Gestione categoriche e standardizzazione come nel training ---
        if pipeline is not None:
            # Se il model è una pipeline, NON applicare preprocessor.transform!
            fantavoto_pred = pipeline.predict(X_pred)[0]

        print(f"Predicted fantavoto for {player_full_name}, role: {fanta_role}, ({team} vs {opponent}, {h_a}): {fantavoto_pred:.2f}")

        index = utils.fantavoto_to_schierability_index(fantavoto_pred, fanta_role, config.ROLE_FANTAVOTO_STATS)  

        index_boost = utils.apply_fantarole_boost(index, fanta_role)

        print(f"Schierability index: {index_boost:.2f}")

        ha_to_print = "casa" if h_a == "h" else "trasf."

        predictions.append({
        'Giocatore': player_full_name,
        'Index': index_boost,
        #'Squadra': team,
        'Avversario': opponent,
        'Campo': ha_to_print
        }) 

    return pd.DataFrame(predictions).reset_index(drop=True)

def train_log_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Allena un modello di regressione lineare e stampa MAE e MSE"""

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
            ("regressor", Ridge(alpha=2.0))
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
    
    # recupera nomi delle feature dopo il preprocessing
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # recupera coefficienti della Ridge
    coefs = model.named_steps["regressor"].coef_

    # dataframe ordinato per importanza assoluta
    feat_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs)
        })
        .sort_values("abs_coefficient", ascending=False)
    )

    print("\n=== FEATURE IMPORTANCE (RIDGE) ===")
    print(feat_importance.head(20))
    
    return model

def train_log_regression_GK(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Allena un modello di regressione lineare e stampa MAE e MSE"""

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
        'ammonizioni'
    ]
    CAT_FEATURES = ['player_team_strength']
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
            ("regressor", Ridge(alpha=2.0))
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
    
    # recupera nomi delle feature dopo il preprocessing
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # recupera coefficienti della Ridge
    coefs = model.named_steps["regressor"].coef_

    # dataframe ordinato per importanza assoluta
    feat_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs)
        })
        .sort_values("abs_coefficient", ascending=False)
    )

    print("\n=== FEATURE IMPORTANCE (RIDGE) ===")
    print(feat_importance.head(20))
    
    return model

def predizioni_per_ruolo(df_voti, next_games_df, pipeline=None, pipeline_gk=None, top_n=10):
    """
    Per ogni ruolo (D, C, A) calcola le predizioni di schierabilità
    e stampa una tabella ordinata per index con evidenziazione dei top_n.
    
    df_voti: dataframe con colonne ['player_name', 'player_team', 'position_clean', ...]
    next_games_df: dataframe con le prossime partite, colonne ['team', 'opponent', 'h_a']
    pipeline: pipeline modello fantavoto da passare a pred_voto_prod
    top_n: quanti top player evidenziare
    """
    #preprocesso df voti
    df_voti = utils.prepare_voto_dataframe(df_voti)
    
    #preprocesso df prossima giornata
    next_games_df = next_games_df.copy()
    next_games_df['home'] = next_games_df['home'].apply(utils.normalize_team_name)
    next_games_df['away'] = next_games_df['away'].apply(utils.normalize_team_name)

    #carico tutti i df per le probabilità gol/assist e dati squadre
    df_orig_goal = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS)
    df_orig_assist = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST)
    df_teams = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)
    df_teams_curr_season = pd.read_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE)

    # --- Carica dataset e modelli 
    model_goal = utils.load_models() 
    model_assist = utils.load_models_assist() 
    model_xg = utils.load_xg_model()

    #if pipeline_gk is None:
        #ruoli = ['D', 'C', 'A']
        #ruoli = ['A']
    #if pipeline is None and  pipeline_gk is not None:
        #ruoli = ['P']
    if pipeline is not None and pipeline_gk is not None:
        ruoli = ['P','D', 'C', 'A']
        

    risultati = {}
    
    for ruolo in ruoli:
        print(f"\n===== Ruolo: {ruolo} =====\n")

        # lista dei giocatori per ruolo
        players_role = df_voti[df_voti['fanta_role'] == ruolo]['player_norm'].tolist()
        
        #rimuovo duplicati
        players_role = list(dict.fromkeys(players_role))

        teams_role, opponents_role, ha_role = [], [], []
        
        #Costruizione giocatore-squadra avversaria prossima giornata
        for player in players_role:
            team = df_voti.loc[df_voti['player_norm'] == player, 'player_team'].iloc[0]
            if team is None or pd.isna(team): #prendendo l'ultima squadra, se il giocatore va all'estero a gennaio non trova più team
                print(f"Player {player} è andato all'estero, squadra non trovata")
                team = "squadra sconosciuta"
                #continue
            team = utils.normalize_team_name(team)
            teams_role.append(team)
            
            # cerca la prossima partita del team
            next_game = next_games_df[(next_games_df['home'] == team) | (next_games_df['away'] == team)]
            
            if not next_game.empty:
                row = next_game.iloc[0]  # prendi la prima prossima partita disponibile
                if team in row['home']:
                    h_a = 'h'
                    opponent = row['away']
                else:
                    h_a = 'a'
                    opponent = row['home']
            else:
                h_a = ""
                opponent = ""
            
            ha_role.append(h_a)
            opponents_role.append(opponent)
        
        if ruolo == 'P' and pipeline_gk is not None:
            # calcola le predizioni
            df_pred = pred_voto_prod_gk(players_role, teams_role, opponents_role, ha_role,
                                    df_voti, df_teams, df_teams_curr_season,                                 
                                    pipeline_gk)    
        elif pipeline is not None:
            # calcola le predizioni
            df_pred = pred_voto_prod(players_role, teams_role, opponents_role, ha_role,
                                    df_voti, df_orig_goal,df_orig_assist, df_teams, df_teams_curr_season,
                                    model_goal, model_assist, model_xg, 
                                    pipeline)
        
        df_pred_sorted = df_pred.sort_values('Index', ascending=False).reset_index(drop=True)

        df_pred_50 = df_pred_sorted.head(50)  # limita a top 50 per ruolo
        
        # evidenzia i top N
        def add_emoji(idx):
            if idx < top_n:
                return "🔥"
            else:
                return ""
        
        df_pred_50['Top'] = [add_emoji(i) for i in df_pred_50.index]
        
        # stampa la tabella
        display_cols = ['Top', 'Giocatore', 'Avversario', 'Campo', 'Index']
        print(df_pred_50[display_cols].to_string(index=False))
        risultati[ruolo] = df_pred_50

    return risultati

def main():

    train = False
    train_gk = False

    test = True
    test_gk = False

    csv_path = config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI
    df_fanta_roles_path = config.DATASET_DATA_DIR / config.FANTA_RUOLI_FILE
    next_games_path = config.DATASET_DATA_DIR / config.NEXT_GAMES_FILE
    df_curr_teams_path = config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE

    df_voti = load_data(csv_path)
    next_games_df = load_data(next_games_path)

    df_curr_teams = load_data(df_curr_teams_path)

    X, y = preprocess_data(df_voti)

    X_gk, y_gk = preprocess_data_GK(df_voti, df_curr_teams)

    if train:
        pipeline = train_log_regression(X, y)
        #salva modello
        MODEL_PATH = config.MODEL_DIR_FV / config.FV_MODEL
        #chiedi all'utente se vuole salvare il modello
        if MODEL_PATH.exists():
                overwrite = input(f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
                if overwrite != "y":
                    print("❌ Salvataggio modello annullato.")
                else:
                    joblib.dump(pipeline, MODEL_PATH)
                    print(f"✅ Modello sovrascritto in: {MODEL_PATH}")
        else:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"✅ Modello salvato in: {MODEL_PATH}")

    else:
        #carica modello
        pipeline = utils.load_fv_model()
        pipeline_gk = utils.load_fv_model_gk()

    if train_gk:
        pipeline = train_log_regression_GK(X_gk, y_gk)
        #salva modello
        MODEL_PATH = config.MODEL_DIR_FV / config.FV_MODEL_GK
        #chiedi all'utente se vuole salvare il modello
        if MODEL_PATH.exists():
                overwrite = input(f"⚠️ Il file '{MODEL_PATH.name}' esiste già. Vuoi sovrascriverlo? (y/n): ").strip().lower()
                if overwrite != "y":
                    print("❌ Salvataggio modello annullato.")
                else:
                    joblib.dump(pipeline, MODEL_PATH)
                    print(f"✅ Modello sovrascritto in: {MODEL_PATH}")
        else:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"✅ Modello salvato in: {MODEL_PATH}")

    else:
        #carica modello
        pipeline = utils.load_fv_model()
    if test:

        #pred_df = pred_voto_prod(config.INPUT["players"],config.INPUT["teams"],config.INPUT["opponents"],config.INPUT["h_a"],df_voti,
            #pipeline['fantavoto_model'])

        predizioni_per_ruolo(df_voti, next_games_df, pipeline=pipeline['fantavoto_model'], pipeline_gk=None, top_n=10)

    if test_gk:

        #pred_df = pred_voto_prod(config.INPUT["players"],config.INPUT["teams"],config.INPUT["opponents"],config.INPUT["h_a"],df_voti,
            #pipeline['fantavoto_model'])
        if train_gk:
            predizioni_per_ruolo(df_voti, next_games_df, pipeline=None, pipeline_gk=pipeline, top_n=10)
        else:
            predizioni_per_ruolo(df_voti, next_games_df, pipeline=None, pipeline_gk=pipeline_gk['fantavoto_model_gk'], top_n=10)

if __name__ == "__main__":
    main()
