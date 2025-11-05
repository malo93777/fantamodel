import pandas as pd
import ast  # per convertire le stringhe tipo "{'id':...}" in dict 
import config 
import unicodedata
import numpy as np
from sklearn.preprocessing import StandardScaler 

class Preprocessor:
    def __init__(self, serie_a_teams=None):
        self.serie_a_teams = serie_a_teams or []

    # ========================
    # Funzioni di supporto
    # ========================
    
    def add_opponent_team_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge una colonna 'opponent_team' al dataframe.
        Usa 'player_team', 'h_team' e 'a_team' per determinare la squadra avversaria.
        """

        def get_opponent(row):
            # Gioca in casa → avversario è away
            if row["player_team"] == row["h_team"]:
                return row["a_team"]
            # Gioca in trasferta → avversario è home
            elif row["player_team"] == row["a_team"]:
                return row["h_team"]
            # Non combacia (possibile errore di dati)
            else:
                return None

        df = df.copy()
        df["opponent_team"] = df.apply(get_opponent, axis=1)

        return df

    def normalize_name(self, name):
        if pd.isna(name):
            return ""
        # Converti in stringa
        name = str(name).strip()
        # Normalizza caratteri accentati (ü -> u)
        name = unicodedata.normalize("NFKD", name)
        name = name.encode("ascii", "ignore").decode("utf-8")
        # Pulisci spazi e porta in minuscolo
        name = name.lower().strip()
        # Se dopo tutto è vuoto, restituisci il nome originale lowercase
        if name == "":
            return str(name).lower()
        return name

    def extract_team_name(self, team_str):
        """Estrae solo il nome della squadra dal campo di understat."""
        if isinstance(team_str, str):
            try:
             return ast.literal_eval(team_str).get("title", team_str)
            except (ValueError, SyntaxError):
             return team_str
        elif isinstance(team_str, dict):
            return team_str.get("title", "")
        return str(team_str)

    def assign_league(self, team: str) -> str:
        """Assegna la lega a una squadra (Serie A o Other), basandosi su substring."""
        if not isinstance(team, str):
            return "Other"
        
        team_lower = team.lower()
        if any(sa_team.lower() in team_lower or team_lower in sa_team.lower() for sa_team in self.serie_a_teams):
            return "Serie A"
        
        return "Other"

    def add_finishing_efficiency_hist(self, df, window=20):
        """
        Calcola una metrica storica di efficienza di finalizzazione ('finishing_efficiency_hist')
        per ciascun giocatore sulle ultime `window` partite.

        Formula:
            finishing_eff = (rolling_goals / rolling_xG) * weight(shots)

        Dove:
        - rolling_* sono somme mobili sulle ultime `window` partite (shiftate per escludere la partita corrente)
       

        Parametri:
            df (pd.DataFrame): dataframe contenente almeno ['player', 'date', 'goals', 'sum_xG', 'shots']
            window (int): numero di partite considerate nella media mobile

        Ritorna:
            pd.DataFrame: con nuova colonna 'finishing_efficiency_hist'
        """
        df = df.sort_values(["player", "date"]).copy()
        eps = 1e-5

        # Calcolo cumulativo goals/xG
        df["finishing_efficiency"] = df["goals"] / (df["sum_xG"] + eps)

        # EMA per ogni giocatore
        df["finishing_efficiency_hist"] = (
            df.groupby("player")["finishing_efficiency"]
            .apply(lambda x: x.shift().ewm(span=window, min_periods=3).mean())
            .reset_index(level=0, drop=True)
        )

        # Clipping per outlier
        max_clip = df["finishing_efficiency_hist"].quantile(0.99)
        df["finishing_efficiency_hist"] = df["finishing_efficiency_hist"].clip(0, max_clip)

        # Fill iniziali
        df["finishing_efficiency_hist"] = df["finishing_efficiency_hist"].fillna(
            df["finishing_efficiency_hist"].median()
        )

        return df

    def weight_efficiency_shots(self, df):
        """
        Aggiunge una colonna 'finishing_eff_weighted' che combina
        l'efficienza di finalizzazione con l'esperienza (numero totale di tiri storici).

        Formula:
            finishing_eff_weighted = finishing_efficiency_hist * weight(shots_hist)

        Dove:
        - shots_hist è il cumulativo di tiri fino alla partita precedente
        - weight(shots_hist) è una funzione logaritmica che cresce lentamente con i tiri

        Parametri:
            df (pd.DataFrame): dataframe con colonne 'player', 'shots', 'finishing_efficiency_hist'

        Ritorna:
            pd.DataFrame: con colonne 'shots_hist' e 'finishing_eff_weighted'
        """
        df = df.copy()

        df["shots_hist"] = df.groupby("player")["shots"].cumsum().shift(1)
        df["shots_hist"] = df["shots_hist"].fillna(0)

        # Peso logaritmico più realistico
        weight = np.log1p(df["shots_hist"]) / np.log1p(20)
        weight = np.clip(weight, 0, 1)

        df["finishing_eff_weighted"] = df["finishing_efficiency_hist"] * weight
        return df

    # ---------------------------------------------------------------
   
    def add_cold_penalty(self, df):
        """
        Combina xG e efficienza di finalizzazione in una metrica 'finishing_form',
        includendo penalità per assenza di gol e smoothing temporale per stabilità.
        La standardizzazione finale è su tutto il dataset (non per giocatore).

        Parametri:
            df (pd.DataFrame): dati con colonne ['player', 'date', 'sum_xG', 'finishing_eff_weighted', 'goals']
            use_rank (bool): se True usa ranks invece di z-score
            smooth_span (int): span per smoothing EMA
            mode (str): 'balanced' (default) o 'strict' per penalizzazioni più forti
        """
        df = df.copy()

        # 1️⃣ Calcolo streak di partite senza gol
        df["no_goal_streak"] = (
            df.groupby("player")["goals"]
            .apply(lambda g: g.eq(0).astype(int)
                .groupby(g.ne(0).cumsum()).cumsum().shift(1))
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        # 2️⃣ Penalità logistica più morbida (chi non segna da molto viene penalizzato)
        df["cold_penalty"] = 1 / (1 + np.exp(0.25 * (df["no_goal_streak"] - 8)))
     
        return df

    def compute_cold_penalty(self, df, streak_col="no_goal_streak", a=0.25, b=8, min_penalty=0.4):
        """
        Calcola la penalità 'cold_penalty' in base alla streak di partite senza gol.
        
        Formula base:
            cold_penalty = min_penalty + (1 - min_penalty) / (1 + exp(a * (streak - b)))

        Dove:
            - a: controlla la pendenza della curva (quanto rapidamente cala)
            - b: soglia centrale in cui la penalità inizia a diventare significativa
            - min_penalty: livello minimo raggiungibile (evita di azzerare tutto)

        Effetto:
            🔹 Penalizza pochissimo per 1–3 partite senza gol
            🔹 Decadimento più rapido dopo 6–8 partite
            🔹 Mantiene un minimo >0 per non annullare totalmente il contributo xG
        """
        df = df.copy()

        # Calcolo streak di partite senza gol (se non esiste)
        if streak_col not in df.columns:
            df["no_goal_streak"] = (
                df.groupby("player")["goals"]
                .apply(lambda g: g.eq(0).astype(int)
                    .groupby(g.ne(0).cumsum()).cumsum().shift(1))
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
            streak_col = "no_goal_streak"

        # Penalità logistica "shiftata"
        df["cold_penalty"] = min_penalty + (1 - min_penalty) / (1 + np.exp(a * (df[streak_col] - b)))

        return df


    def combine_sumxg_efficiency(self, df, use_rank=False):
            """
            Combina la pericolosità (xG generato) e l'efficienza (finishing_eff_weighted)
            in un unico indice 'finishing_form'.

            Due opzioni di normalizzazione:
            - use_rank=True → usa rank percentuali (0-1), robusti a outlier ma perdono scala metrica
            - use_rank=False → usa z-score (StandardScaler), più informativi per modelli lineari

            Formula:
                finishing_form = 0.5 * norm(sum_xG) + 0.5 * norm(finishing_eff_weighted)

            Parametri:
                df (pd.DataFrame)
                use_rank (bool): se True usa rank percentuali, altrimenti z-score

            Ritorna:
                pd.DataFrame: con nuova colonna 'finishing_form'
            """
            df = df.copy()

            if use_rank:
                # Versione rank percentuale
                df["finishing_form"] = (
                    0.5 * df["sum_xG"].rank(pct=True) +
                    0.5 * df["finishing_eff_weighted"].rank(pct=True)
                )
            else:
                # Versione z-score (mantiene informazione metrica)
                scaler = StandardScaler()
                z_sumxg = scaler.fit_transform(df[["sum_xG"]])
                z_eff = scaler.fit_transform(df[["finishing_eff_weighted"]])
                df["finishing_form"] = 0.5 * z_sumxg.flatten() + 0.5 * z_eff.flatten()

            return df

    def calculate_players_data_shots(self, df):

        #funzione per aggiungere al dataset dei tiri (già unito con quello dei giocatori per stagione)
        #le info su quanto tira per partita

        df["time"] = df["time"].astype(float)
        df["shots"] = df["shots"].astype(float)
        df["xG"] = df["xG"].astype(float)
        df["goals"] = df["goals"].astype(float)

        df["shots_per90"] = round(df["shots"] / df["time"] * 90, 2)
        df["xG_per90"] = round(df["xG"] / df["time"] * 90, 2)
        df["goals_per90"] = round(df["goals_total"] / df["time"] * 90, 2)

        return df 
    
    def calculate_players_data_assists(self, df): 

        #funzione per aggiungere al dataset dei assist (già unito con quello dei giocatori per stagione)
        #le info su quanto assiste per partita

        df["time"] = df["time"].astype(float)  
        df["xA"] = df["xA"].astype(float)
        df["assists"] = df["assists"].astype(float)

        df["xA_per90"] = round(df["xA"] / df["time"] * 90, 2)
        df["assists_per90"] = round(df["assists_total"] / df["time"] * 90, 2)

        return df 

    # ========================
    # Preprocessing dei tiri
    # ========================

    def add_missing_games(self, shots_df, matches_df, all_season_players):
    
        all_players = []

        # Pulisci prima i campi h e a del matches_df
        matches_df["h_team"] = matches_df["h"].apply(self.extract_team_name)
        matches_df["a_team"] = matches_df["a"].apply(self.extract_team_name)
        matches_df["datetime"] = pd.to_datetime(matches_df["datetime"], errors="coerce")

        # Conversioni base
        matches_df = matches_df.copy()
        matches_df["datetime"] = pd.to_datetime(matches_df["datetime"], errors="coerce")
        now = pd.Timestamp.now()
        matches_df = matches_df[matches_df["datetime"] <= now]

        matches_df["id"] = matches_df["id"].astype(str)
        shots_df["match_id"] = shots_df["match_id"].astype(str)

        all_season_players["player_name"] = all_season_players["player_name"].apply(self.normalize_name)

           # --- Itera su ciascun giocatore per stagione ---
        for (player_name, season), group in shots_df.groupby(["player", "season"]):
            player_team = group["player_team"].iloc[0]

            # --- Tutte le partite giocate dalla squadra in quella stagione ---
            team_matches = matches_df[
                (matches_df["team"] == player_team)
                & (matches_df["season"] == season)
            ]
            team_match_ids = team_matches["id"].astype(str).unique()
            player_match_ids = group["match_id"].astype(str).unique()

            # --- Numero ufficiale di presenze del giocatore ---
            season_row = all_season_players[
                (all_season_players["player_name"].str.lower() == player_name.lower())
                & (all_season_players["season"] == season)
            ]

            # --- Numero ufficiale di presenze del giocatore ---
            print(f"\n🔍 Confronto per il giocatore: {player_name} | stagione: {season}")
            print("👉 player_name.lower() =", player_name.lower())     

            season_row = all_season_players[
                (all_season_players["player_name"].str.lower() == player_name.lower())
                & (all_season_players["season"] == season)
            ]

            if season_row.empty:
                print(f"⚠️ Nessun dato di presenze per {player_name} ({season})")
                all_players.append(group)
                continue

            n_appearances = int(season_row["games"].iloc[0])

            # --- Partite mancanti (in cui non ha tirato ma ha giocato) ---
            n_current = len(player_match_ids)
            n_missing = max(0, n_appearances - n_current)

            if n_missing == 0:
                all_players.append(group)
                continue

            available_matches = list(set(team_match_ids) - set(player_match_ids))
            if len(available_matches) < n_missing:
                n_missing = len(available_matches)

            # --- Crea righe dummy per le partite senza tiri ---
            dummy_rows = []
            for mid in available_matches[:n_missing]:
                match_row = team_matches[team_matches["id"].astype(str) == mid].iloc[0]
                opponent = (
                    match_row["a_team"] if match_row["side"] == "h" else match_row["h_team"]
                )

                dummy_rows.append({
                    "player": player_name,
                    "season": season,
                    "match_id": mid,
                    "player_team": player_team,
                    "opponent_team": opponent,
                    "is_home": 1 if match_row["side"] == "h" else 0,
                    "sum_xG": 0.0,
                    "n_shots": 0,
                    "goals": 0,
                    "date": match_row["datetime"]
                })

            # --- Combina originali + dummy ---
            group = group.copy()
            group["date"] = pd.to_datetime(group["date"], errors="coerce")
            combined = pd.concat([group, pd.DataFrame(dummy_rows)], ignore_index=True)

            # --- Ordina e pulisci date ---
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
            combined = combined.sort_values("date").reset_index(drop=True)

            all_players.append(combined)

        # --- Output finale ordinato globalmente ---
        result = pd.concat(all_players, ignore_index=True)
        result = result.sort_values(["player", "season", "date"]).reset_index(drop=True)
        return result

    def preproc_goals_dataset(self, input_path: str, df_to_merge_path: str, is_SerieA=True) -> pd.DataFrame:
        """
        Preprocessa il dataset dei goals (analogo a preproc_assists_dataset, ma per i gol):
        - raggruppa per player+match
        - calcola sum_xG e goals per match
        - filtra Serie A
        - unisce con file all_season_players
        - calcola statistiche per partita basate sulla carriera (shots/goals/xG per90)
        """
        df = pd.read_csv(input_path)
        #creo colonna is_goal
        # Raggruppamento per player + match id
        df = (
            df.groupby(["player", "id"])
            .agg(
                sum_xG=("xG", "sum"),
                goals=("goals", "sum"),
                season=("season", "first"),
                date=("date", "first"),
                minutes_played = ("time", "sum"),
                h_team=("h_team", "first"),
                a_team=("a_team", "first"),
            )
            .reset_index()
        )

        if is_SerieA:
            # Assegna lega e filtra Serie A
            df["league"] = df["h_team"].apply(self.assign_league)
            df = df[df["league"] == "Serie A"]     

        # Ordino cronologicamente
        df = df.sort_values(["player", "date"])

        all_season_players = pd.read_csv(df_to_merge_path)

        # Normalizzazione nomi
        df["player"] = df["player"].apply(self.normalize_name)
        all_season_players["player_name"] = all_season_players["player_name"].apply(self.normalize_name)

        merged_df = df.merge(
            all_season_players,
            left_on=["player", "season"],
            right_on=["player_name", "season"],
            how="left"
        )

        missing_players = merged_df[merged_df["games"].isna()]["player"].unique()
        print(f"⚠️ {len(missing_players)} giocatori senza match nel file all_season_players:")
        print(missing_players)

        # Rinomino colonne doppie come fatto per gli altri preprocess
        merged_df = merged_df.rename(columns={
            "goals_x": "goals",
            "goals_y": "goals_total",
            "id_x": "match_id",
            "id_y": "player_id",
            "team_title": "player_team"
        })

        # Aggiungo colonna opponent team
        merged_df = self.add_opponent_team_column(merged_df)

        # Aggiungo info per partita sulla base della carriera (shots/xG/goals per90)
        merged_df = self.calculate_players_data_shots(merged_df)
        
        # Creo colonna booleana is_goals
        merged_df["is_goals"] = (merged_df["goals"] > 0).astype(int)

        # Drop colonne inutili
        merged_df = merged_df.drop(columns={"player_name", "player_id", "yellow_cards", "red_cards", "h_team", "a_team"})

        #  Calcolo rolling features (XG, shots, gol mean, time for match)
        merged_df = self.calculate_roll_features(merged_df)

        # Calcolo finishing efficiency
        merged_df = self.add_finishing_efficiency_hist(merged_df, window=10)

        # Calcolo finishing_eff_weighted
        merged_df = self.weight_efficiency_shots(merged_df)

        # Calcolo finishing_form
        merged_df = self.combine_sumxg_efficiency(merged_df, use_rank=True)

        # Calcolo cold_penalty
        merged_df = self.compute_cold_penalty(merged_df)

        if is_SerieA:
            # salva su file dedicato ai goals (creare config.GOALS_DATA_FILE nel caso non esista)
            merged_df.to_csv(config.DATASET_DATA_DIR / config.GOALS_DATA_FILE, index=False)
        else:
            # salva su file dedicato ai goals per tutti i campionati
            merged_df.to_csv(config.DATASET_DATA_DIR / config.GOALS_DATA_FILE_ALL_LEAGUES, index=False)

        return merged_df
    
    def preproc_assists_dataset(self, input_path: str, df_to_merge_path: str) -> pd.DataFrame:
        """
        Preprocessa il dataset degli assist:
        - rimuove NaN
        - crea is_assist
        - raggruppa per player+match
        - crea rolling features (ultime 5 partite)
        - filtra Serie A
        """
        df = pd.read_csv(input_path)

         # =======================
        # 4️⃣ Raggruppamento per match id
        # =======================
        df = (
            df.groupby(["player", "id"])
            .agg(
                sum_xA=("xA", "sum"), 
                assists=("assists", "sum"),         
                season=("season", "first"),                         
                date=("date", "first"),
                h_team=("h_team", "first"),
                a_team=("a_team", "first")
            )
            .reset_index()
        )

        # Assegna lega
        df["league"] = df["h_team"].apply(self.assign_league)
        df = df[df["league"] == "Serie A"]

        # Ordino cronologicamente
        df = df.sort_values(["player", "date"])
        
        all_season_players = pd.read_csv(df_to_merge_path)

        #normalizzazione nomi unicode, per non perdersi alcuni giocatori
        df["player"] = df["player"].apply(self.normalize_name)
        all_season_players["player_name"] = all_season_players["player_name"].apply(self.normalize_name)

        merged_df = df.merge(
        all_season_players,
        left_on=["player", "season"],
        right_on=["player_name", "season"],
        how="left"
        )     

        missing_players = merged_df[merged_df["games"].isna()]["player"].unique()
        print(f"⚠️ {len(missing_players)} giocatori senza match nel file all_season_players:")
        print(missing_players)

        #rinomino colonne doppie
        merged_df = merged_df.rename(columns={  "assists_x": "assists",
                                    "assists_y": "assists_total",
                                    "id_x" : "match_id",
                                    "id_y" : "player_id",
                                    "team_title": "player_team"})
        
        #aggiungo colonna opponent team
        merged_df = self.add_opponent_team_column(merged_df)

        #aggiungo info per partita sulla base della sua carriera in Serie A
        merged_df = self.calculate_players_data_assists(merged_df)

        #drop colonne inutili
        merged_df = merged_df.drop(columns={"player_name", "player_id", "yellow_cards", "red_cards", "h_team", "a_team"})

        merged_df.head()
        merged_df.to_csv(config.DATASET_DATA_DIR / config.ASSIST_DATA_FILE, index=False)

        return merged_df
    
    # Rolling Features
    def calculate_roll_features(self, df):
        '''
        Fun che prende in input un df e calcola media xG, tiri e gol delle ultime 5 partite 
        '''
        # Rolling features (ultime 5 partite)
        df["xG_last5"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["shots_last5"] = df.groupby("player")["shots"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["goals_last5"] = df.groupby("player")["goals"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        # Statistiche cumulative
        df["xG_cummean"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().expanding().mean()
        )

        #minuti per partita last5
        df = self.add_minutes_played_last5(df)

        return df
    
    def calculate_roll_features_assist(self, df):
        '''
        Fun che prende in input un df e calcola media xG, tiri e gol delle ultime 5 partite 
        '''
        # Rolling features (ultime 5 partite)
        df["xA_last5"] = df.groupby("player")["sum_xA"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        df["assist_last5"] = df.groupby("player")["assists"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        #divido key passes totali per 90 minuti giocati
        df["key_passes_per90"] = round(df["key_passes"] / df["time"] * 90, 2)

        return df

    # ========================
    # XG per 90 minuti squadre
    # ========================
    def create_Xg_90min_teams(self, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        """
        Crea colonna team_xG_90min e merge con dataset dei giocatori.
        """
        teams_df["XG_90min"] = round(teams_df["XG_90min"], 2)

        players_df = players_df.merge(
            teams_df.rename(columns={"Team": "player_team"})[["player_team", "season", "XG_90min"]],
            on=["player_team", "season"],
            how="left"
        )

        players_df.rename(columns={"XG_90min": "team_xG_90min"}, inplace=True)

        return teams_df, players_df
    
    def create_XgA_90min_opponent(self, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        """
        Crea colonna XgA_90min_opponent e merge con dataset dei giocatori.
        """
        teams_df["XGA_90min"] = round(teams_df["XGA_90min"], 2)

        players_df = players_df.merge(
            teams_df.rename(columns={"Team": "opponent_team"})[["opponent_team", "season", "XGA_90min"]],
            on=["opponent_team", "season"],
            how="left"
        )

        players_df.rename(columns={"XGA_90min": "opponent_xGA_90min"}, inplace=True)

        return teams_df, players_df

    def add_minutes_played_last5(self,df):
        """
        Calcola la media dei minuti giocati nelle ultime 5 partite per ogni giocatore.
        """
        if "minutes_played" not in df.columns:
            print("⚠️ Colonna 'minutes_played' mancante!")
            df["minutes_played_last5"] = 0
            return df

        df = df.sort_values("date")
        df["minutes_played_last5"] = (
            df["minutes_played"]
            .rolling(window=5, min_periods=1)
            .mean()
        )
        return df
    
def build_team_dataframe(team_data: dict) -> pd.DataFrame:
    """
    Converte un dizionario `team_data` in un DataFrame con tutte le partite
    e calcola le colonne xG_last5 e xGA_last5 per ogni squadra.

    Args:
        team_data (dict): dizionario del tipo {
            '94': {'id': '94', 'title': 'Verona', 'history': [ {...}, {...}, ... ]},
            '95': {'id': '95', 'title': 'Roma', 'history': [ {...}, {...}, ... ]},
            ...
        }

    Returns:
        pd.DataFrame: DataFrame completo con colonne xG_last5 e xGA_last5
    """

    rows = []

    # 1️⃣ Espandi il dizionario in righe
    for team_id, team in team_data.items():
        title = team.get("title", "")
        history = team.get("history", [])
        for match in history:
            rows.append({**match, "team_id": team_id, "team_name": title})

    # 2️⃣ Crea il DataFrame
    df = pd.DataFrame(rows)

    # Se non ci sono partite, ritorna subito
    if df.empty:
        return df

    # 3️⃣ Converti la colonna "date" in datetime e ordina
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_name", "date"]).reset_index(drop=True)

    # 4️⃣ Calcola medie mobili (ultime 5 partite, escludendo quella corrente)
    df["xG_last5"] = (
        df.groupby("team_name")["xG"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    df["xGA_last5"] = (
        df.groupby("team_name")["xGA"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    return df
