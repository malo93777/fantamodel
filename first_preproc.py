import pandas as pd
import ast  # per convertire le stringhe tipo "{'id':...}" in dict 

class Preprocessor:
    def __init__(self, serie_a_teams=None, top_teams=None, mid_teams=None, weak_teams=None):
        self.serie_a_teams = serie_a_teams or []
        self.top_teams = top_teams or []
        self.mid_teams = mid_teams or []
        self.weak_teams = weak_teams or []

    # ========================
    # Funzioni di supporto
    # ========================
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
        """Assegna la lega a una squadra (Serie A o Other)."""
        if not isinstance(team, str):
            return "Other"
        team_lower = team.lower()
        if any(sa.lower() in team_lower for sa in self.serie_a_teams):
            return "Serie A"
        return "Other"

    def map_strength(self, team: str) -> str:
        """Classifica la forza della squadra (top, mid, weak, other)."""
        if not isinstance(team, str):
            return "unknown"
        team_lower = team.lower().strip()
        if any(sa.lower() in team_lower for sa in self.top_teams):
            return "top"
        elif any(sa.lower() in team_lower for sa in self.mid_teams):
            return "mid"
        elif any(sa.lower() in team_lower for sa in self.weak_teams):
            return "weak"
        else:
            return "other"
        
    def calculate_players_data(self, df): 

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

    # ========================
    # Preprocessing dei tiri
    # ========================

    def add_missing_games(self, shots_df, matches_df):
        all_players = []

        # Pulisci prima i campi h e a del matches_df
        matches_df = matches_df.copy()
        matches_df["h_team"] = matches_df["h"].apply(self.extract_team_name)
        matches_df["a_team"] = matches_df["a"].apply(self.extract_team_name)
        matches_df["datetime"] = pd.to_datetime(matches_df["datetime"])

        for player, group in shots_df.groupby(["player", "season"]):
            player_name, season = player
            player_team = group["player_team"].iloc[0]

            # tutte le partite della squadra in quella stagione
            team_matches = matches_df[
                (matches_df["team"] == player_team) &
                (matches_df["season"] == season)
            ]
            
            team_match_ids = team_matches["id"].astype(str).unique()
            player_match_ids = group["match_id"].astype(str).unique()

            # id partite mancanti
            missing_ids = set(team_match_ids) - set(player_match_ids)

            # costruisco righe dummy per partite senza tiri
            dummy_rows = []
            for mid in missing_ids:
                match_row = team_matches[team_matches["id"].astype(str) == mid].iloc[0]
                opponent = match_row["a_team"] if match_row["side"] == "h" else match_row["h_team"]
                dummy = {
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
                }
                dummy_rows.append(dummy)

            # combino righe originali + dummy
            group["date"] = pd.to_datetime(group["date"])
            combined = pd.concat([group, pd.DataFrame(dummy_rows)], ignore_index=True)

            # ordina per data
            combined = combined.sort_values("date").reset_index(drop=True)

            #*** DROP PARTITE FUTURE ***
            now = pd.Timestamp.now()
            # converto in datetime se non lo è già
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"] <= now].reset_index(drop=True)

            all_players.append(combined)

        # dataframe finale ordinato globalmente
        result = pd.concat(all_players, ignore_index=True)
        result = result.sort_values(["player", "season", "date"]).reset_index(drop=True)
        return result

    def preproc_shots_dataset(self, input_path: str, df_to_merge_path: str) -> pd.DataFrame:
        """
        Preprocessa il dataset dei tiri:
        - rimuove NaN
        - crea is_goal
        - raggruppa per player+match
        - crea rolling features (ultime 5 partite)
        - crea cumulative mean
        - filtra Serie A
        """
        df = pd.read_csv(input_path)

        # Rimuovi i valori nulli
        df = df.dropna()

        # Crea colonna 'is_goal'
        df["is_goal"] = (df["result"] == "Goal").astype(int)

        # Raggruppa per giocatore e partita
        df = (
            df.groupby(["player", "match_id"])
            .agg(
                sum_xG=("xG", "sum"),
                n_shots=("id", "count"),
                goals=("is_goal", "sum"),
                h_a=("h_a", "first"),
                h_team=("h_team", "first"),
                a_team=("a_team", "first"),
                season=("season", "first"),
                date=("date", "first")
            )
            .reset_index()
        )

        # Assegna lega
        df["league"] = df["h_team"].apply(self.assign_league)
        df = df[df["league"] == "Serie A"]

        # Trova squadra e avversario del giocatore
        df["player_team"] = df.apply(
                    lambda row: row["h_team"] if row["h_a"] == "h" else row["a_team"], axis=1
        )
        df["opponent_team"] = df.apply(
              lambda row: row["a_team"] if row["h_a"] == "h" else row["h_team"], axis=1
        )

        # Colonna booleana is_home
        df["is_home"] = df["h_a"].apply(lambda x: 1 if x == "h" else 0)

        # Rimuovo colonne inutili
        df = df.drop(columns=["h_a", "h_team", "a_team", "league"])

        # Ordino cronologicamente
        df = df.sort_values(["player", "date"])

        '''
        # Rolling features (ultime 5 partite)
        df["xG_last5"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["shots_last5"] = df.groupby("player")["n_shots"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["goals_last5"] = df.groupby("player")["goals"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        # Statistiche cumulative
        df["xG_cummean"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().expanding().mean()
        )
        '''
        
        all_season_players = pd.read_csv(df_to_merge_path)

        merged_df = df.merge(
        all_season_players,
        left_on=["player", "season"],
        right_on=["player_name", "season"],
        how="left"
        )     

        #rinomino colonne doppie
        merged_df = merged_df.rename(columns={  "goals_x": "goals",
                                    "goals_y": "goals_total"})
        
                #aggiungo info su quanto tira il giocatore per partita sulla base della sua carriera in Serie A
        merged_df = self.calculate_players_data(merged_df)

        #drop colonne inutili
        merged_df = merged_df.drop(columns={"player_name", "id", "yellow_cards", "red_cards", "team_title"})

        merged_df.head()
        merged_df.to_csv("shots_2025.csv", index=False)

        return merged_df
    
    # Rolling Features
    def calculare_roll_features(self, df):
        '''
        Fun che prende in input un df e calcola media xG, tiri e gol delle ultime 5 partite 
        '''
        # Rolling features (ultime 5 partite)
        df["xG_last5"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["shots_last5"] = df.groupby("player")["n_shots"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["goals_last5"] = df.groupby("player")["goals"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        # Statistiche cumulative
        df["xG_cummean"] = df.groupby("player")["sum_xG"].transform(
            lambda x: x.shift().expanding().mean()
        )
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
