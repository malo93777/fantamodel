import pandas as pd
import ast

from unidecode import unidecode  # per convertire le stringhe tipo "{'id':...}" in dict 
import config 
import unicodedata
import numpy as np
from sklearn.preprocessing import StandardScaler
import utils
import re

class Preprocessor:
    def __init__(self, serie_a_teams=None):
        self.serie_a_teams = serie_a_teams or []

    # ========================
    # Funzioni di supporto
    # ========================

    def normalize_surname_name(self, name):
        """
        Assume input: COGNOME NOME
        Gestisce cognomi composti (es. de silvestri)
        rimuove accenti e apostrofi
        """

        if pd.isna(name):
            return ""

        name = str(name).strip().lower()

        # Rimuove accenti
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        # Rimuove apostrofi
        name = name.replace("'", " ").replace("’", " ")

        parts = name.split()

        if len(parts) < 2:
            return name

        # Ultima parola = nome
        first_name = parts[-1]

        # Tutto il resto = cognome (anche composto)
        surname = " ".join(parts[:-1])

        return f"{first_name} {surname}"

    def remove_middle_name(self, name):
        """
        Rimuove il secondo nome SOLO se non è una particella del cognome.
        Rimuove anche gli accenti.
        Preserva cognomi composti (de silvestri, van dijk, ecc.)
        """

        if pd.isna(name):
            return ""

        import unicodedata

        # lowercase + strip
        name = str(name).strip().lower()

        # 🔹 rimuove accenti
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))

        particles = {"de", "di", "da", "del", "della", "van", "von", "la", "le"}

        parts = name.split()

        # Se meno di 3 parti → nulla da rimuovere
        if len(parts) < 3:
            return " ".join(parts)

        # Caso: nome + particella + cognome → NON TOCCARE
        if parts[1] in particles:
            return " ".join(parts)

        # Caso: nome + secondo_nome + cognome
        return f"{parts[0]} {parts[-1]}"

    
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
    
    def add_team_strength_column(
        self,
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
            .apply(utils.normalize_team)
            .apply(utils.map_strength)
        )

        return df
    
    def mod_df_teams(self, df):

        #*** funzione per aggiungere colonne al df di base che non sono presenti nel dataset originale ***

        df_mod = df.copy()
        df_mod.index = df_mod.index + 1

        #aggiungo dati per differenza tra xG e gol, xGA e gol subiti
        df_mod['diff_XG_GOL'] = df_mod['xG'] - df_mod['G']
        df_mod['diff_xGA_GOLAG'] = df_mod['xGA'] - df_mod['GA']
        df_mod['XGA_90min'] = df_mod['xGA'] / df_mod['M']# xGAgainst per 90 minuti
        df_mod['XG_90min'] = df_mod['xG'] / df_mod['M'] # xG per 90 minuti

        #tronco a 2 valori decimali i float
        df_mod = df_mod.round({"diff_XG_GOL": 2, "diff_xGA_GOLAG": 2, "XGA_90min": 2})

        return df_mod

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

    def add_finishing_efficiency_hist(self, df, window=20, prod=False):
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
            prod (bool): se True non applica shift (usa anche gli ultimi valori disponibili)

        Ritorna:
            pd.DataFrame: con nuova colonna 'finishing_efficiency_hist'
        """
        df = df.sort_values(["player", "date"]).copy()
        eps = 1e-5

        # Calcolo cumulativo goals/xG
        df["finishing_efficiency"] = df["npgoals_perMatch"] / (df["npxG_perMatch"] + eps)

        # EMA per ogni giocatore
        if prod:
            # NO SHIFT in produzione → usa anche gli ultimi valori
            df["finishing_efficiency_hist"] = (
                df.groupby("player")["finishing_efficiency"]
                .apply(lambda x: x.ewm(span=window, min_periods=3).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            # VERSIONE TRAINING → SHIFT per evitare leakage
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


    def weight_efficiency_shots(self, df, prod=False):
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
            prod (bool): se True non applica shift sulla storia dei tiri

        Ritorna:
            pd.DataFrame: con colonne 'shots_hist' e 'finishing_eff_weighted'
        """
        df = df.copy()

        if prod:
            # NO SHIFT → usa anche l’ultimo match
            df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum()
        else:
            # VERSIONE TRAINING → SHIFT per evitare leakage
            df["shots_hist"] = df.groupby("player")["shots_perMatch"].cumsum().shift(1)

        df["shots_hist"] = df["shots_hist"].fillna(0)

        # Peso logaritmico più realistico
        weight = np.log1p(df["shots_hist"]) / np.log1p(20)
        weight = np.clip(weight, 0, 1)

        df["finishing_eff_weighted"] = df["finishing_efficiency_hist"] * weight
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

    # ---------------------------------------------------------------

    def compute_shot_quality(self, df, window=20, use_rank=True, prod=False):
        """
        Esegue in sequenza:
        1) add_finishing_efficiency_hist
        2) weight_efficiency_shots
        3) combine_sumxg_efficiency

        Se prod=True:
            - NON applica shift nei calcoli storici (usa tutti i dati disponibili).
        """

        merged_df = df.copy()

        # Calcolo finishing efficiency
        merged_df = self.add_finishing_efficiency_hist(
            merged_df, window=window, prod=prod
        )

        # Calcolo finishing_eff_weighted
        merged_df = self.weight_efficiency_shots(
            merged_df, prod=prod
        )

        # Calcolo finishing_form
        merged_df = self.combine_sumxg_efficiency(
            merged_df, use_rank=use_rank
        )

        return merged_df

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
                npgoals_perMatch=("npg", "sum"),
                shots_perMatch=("shots", "sum"),
                npxG_perMatch=("npxG", "sum"),
                xGChain_perMatch=("xGChain", "sum"),
                xGBuildUp_perMatch=("xGBuildup", "sum"),
                key_passes_perMatch=("key_passes", "sum"),          
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

        merged_df = self.add_team_strength_column(merged_df, 'opponent_team', 'opponent_team_strength')

        # Aggiungo info per partita sulla base della carriera (shots/xG/goals per90)
        merged_df = self.calculate_players_data_shots(merged_df)
        
        # Creo colonna booleana is_goals
        merged_df["is_goals"] = (merged_df["goals"] > 0).astype(int)

        # Drop colonne inutili
        merged_df = merged_df.drop(columns={"player_name", "player_id", "yellow_cards", "red_cards", "h_team", "a_team"})

        #  Calcolo rolling features (XG, shots, gol mean, time for match)
        merged_df = self.calculate_roll_features(merged_df)

        # Calcolo finishing_form
        merged_df = utils.compute_finishing_form(merged_df, window=12, use_rank=True, prod=False)

        # Calcolo shot quality
        merged_df = utils.compute_shot_quality_index_per_shot(merged_df, prod=False)

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

        merged_df = self.add_team_strength_column(merged_df, 'opponent_team', 'opponent_team_strength')

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
        df["shots_last5"] = df.groupby("player")["shots_perMatch"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
        df["goals_last5"] = df.groupby("player")["goals"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

        #sbagliata!!!
        df["goals_last5_sum"] = df.groupby("player")["goals"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).sum()
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
        rows = []

        #RIPARTITE DA QUA, MANCA XG_90
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
    
    def compute_shot_quality_index(self, df, window=15, player_col="player", prod=False):
        """
        Calcola un indice di qualità tiro normalizzato 0–1 SENZA SCALER.
        
        - prod=False → training (usa shift per evitare leakage)
        - prod=True  → produzione (non usa shift, usa tutto lo storico)
        """

        df = df.copy()
        eps = 1e-6

        # --------------------------------------------
        # 1) Efficienza di tiro logaritmica
        # --------------------------------------------
        df["shot_eff_log"] = np.log1p(df["npgoals_perMatch"]) - np.log1p(df["npxG_perMatch"] + eps)

        # --------------------------------------------
        # 2) Difficoltà del tiro (premia gol difficili)
        # --------------------------------------------
        df["shot_difficulty"] = df["npgoals_perMatch"] * (1 - df["npxG_perMatch"].clip(0, 1))

        # --------------------------------------------
        # 3) Indice grezzo
        # --------------------------------------------
        df["shot_quality_raw"] = (
             df["shot_eff_log"] +
             df["shot_difficulty"]
        )

        # --------------------------------------------
        # 4) Rolling window di stabilizzazione
        # --------------------------------------------
        df = df.sort_values([player_col, "date"])

        if prod:
            # 🚀 Produzione → usa anche la riga corrente
            df["shot_quality_roll"] = (
                df.groupby(player_col)["shot_quality_raw"]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        else:
            # 🎓 Training → shift per evitare leakage
            df["shot_quality_shifted"] = df.groupby(player_col)["shot_quality_raw"].shift(1)

            df["shot_quality_roll"] = (
                df.groupby(player_col)["shot_quality_shifted"]
                .rolling(window=window, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # fallback iniziale
            df["shot_quality_roll"] = df["shot_quality_roll"].fillna(df["shot_quality_shifted"])

        # --------------------------------------------
        # 5) Normalizzazione 0–1 senza scaler
        #    usando quantile clipping robusto (evita outlier)
        # --------------------------------------------

        # quantili robusti
        q01 = df["shot_quality_roll"].quantile(0.01)
        q99 = df["shot_quality_roll"].quantile(0.99)

        # protezione
        if q99 - q01 < 1e-6:
            df["shot_quality_index"] = 0.5
            return df

        # normalizzazione
        df["shot_quality_index"] = (df["shot_quality_roll"] - q01) / (q99 - q01)

        # clipping finale
        df["shot_quality_index"] = df["shot_quality_index"].clip(0, 1)

        return df
    
    def add_home_away_xg_features(self,df):

        df = df.copy()

        # xG prodotto in casa / fuori
        df["xG_h"] = (
            df.groupby("team_name")["xG"]
            .transform(lambda s: s.where(df["h_a"]=="h").fillna(0))
        )
        df["xG_a"] = (
            df.groupby("team_name")["xG"]
            .transform(lambda s: s.where(df["h_a"]=="a").fillna(0))
        )

        # xGA concesso in casa / fuori
        df["xGA_h"] = (
            df.groupby("team_name")["xGA"]
            .transform(lambda s: s.where(df["h_a"]=="h").fillna(0))
        )
        df["xGA_a"] = (
            df.groupby("team_name")["xGA"]
            .transform(lambda s: s.where(df["h_a"]=="a").fillna(0))
        )

        return df

    
    def build_team_dataframe(self,team_data: list) -> pd.DataFrame:
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
        for team in team_data:
            team_id = team.get("id", None)
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
        df["xG_last5_mean"] = (
            df.groupby("team_name")["xG"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )
        df["xGA_last5_mean"] = (
            df.groupby("team_name")["xGA"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        df = self.add_home_away_xg_features(df)

        return df

    def merge_voti_player(self, csv1_path, csv2_path, csv3_path):
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)

        # Rimuovo i senza voto_gds
        #df2 = df2[df2['voto_gds'].notna()]

        # Normalizzazione nomi
        df1["player_norm"] = df1["player"].apply(self.remove_middle_name)
        df2['player_norm'] = df2['player_norm'].apply(self.normalize_surname_name)
        df2
        ''''''
        columns_to_add = [
            'voto_gds','fantavoto','rig_segnati','rig_sbagliati',
            'ammonizioni','espulsioni','autogol'
        ]

        for col in columns_to_add:
            df1[col] = pd.NA

        df1['date'] = pd.to_datetime(df1['date'])

        for (player, season), group in df1.groupby(['player_norm','season']):
            group_sorted = group.sort_values('date')

            #se player inizia con "jon"
            if player.startswith("jonatha") or player.startswith("nicolo barel"):
                print(f"Processing player: {player} | season: {season} | matches: {len(group_sorted)}")   

            # Filtra df2 per giocatore e stagione
            df2_player = (
                df2[
                    (df2['player_norm'] == player.lower()) &
                    (df2['stagione'].astype(str).str.startswith(str(season)))
                ]
                .copy()
            )

            if df2_player.empty:
                #print(f"⚠️ Nessun dato voti per {player} ({season})")
                continue
            
            # ---- match_order base (uguale alla giornata) ----
            df2_player["match_order"] = df2_player["giornata"].astype(float)

            # ---- FIX giornata 16 rinviata (Serie A 2025-2026) ----
            if "2025-2026" in df2_player["stagione"].astype(str).iloc[0]:

                POSTPONED_TEAMS_2025_16 = {
                    "como", "milan", "inter", "lecce", "napoli",
                    "parma", "verona", "bologna"
                }

                squadra_player = (
                    df2_player["squadra"]
                    .astype(str)
                    .str.lower()
                    .iloc[0]
                )

                if squadra_player in POSTPONED_TEAMS_2025_16:

                    if 16 not in df2_player["giornata"].values:

                        if player.startswith("nico paz"):
                            print("debug")

                        # riga di riferimento (non fondamentale quale)
                        base_row = df2_player.iloc[-1]

                        new_row = {
                            "stagione": "2025-2026",
                            "giornata": 16,
                            "match_order": 20.5,   #  POSIZIONATA TRA 20 E 21
                            "squadra": base_row["squadra"],
                            "player": player,
                            "player_norm": player,
                            "voto_gds": 6,
                            "fantavoto": 6,
                            "gol": 0,
                            "assist": 0,
                            "ammonizioni": 0,
                            "espulsioni": 0,
                            "autogol": 0,
                            "rig_segnati": 0,
                            "rig_sbagliati": 0,
                            "is_virtual_match": True
                        }

                        df2_player = pd.concat(
                            [df2_player, pd.DataFrame([new_row])],
                            ignore_index=True
                        )
            
            # 🔑 ORDINE CORRETTO: NON per giornata
            df2_player = df2_player.sort_values("match_order")
            
            # CONTROLLO CHE IL DF COI VOTI NON ABBIA PIÙ RIGHE DEL DF1
            # rappgruppo df1 per player e season, ordino per data e confronto le dimensioni con df2_player
            df1_player = df1[
                (df1['player_norm'] == player) &
                (df1['season'] == season)
            ].copy()

            if df1_player.empty:
                print(f"⚠️ Nessun dato df1 per {player} ({season})")
                continue

            df1_player = df1_player.sort_values('date')

            if len(df1_player) != len(df2_player):
                print(f"⚠️ Differenza dimensioni per {player} ({season}): {len(df1_player)} vs {len(df2_player)}")
                df2_player = self.reconcile_df2_by_partita(
                    df2_player=df2_player,
                    df1_player=group_sorted
                )

            for col in columns_to_add:
                values = df2_player[col].tolist()
                while len(values) < len(group_sorted):
                    values.append(pd.NA)
                df1.loc[group_sorted.index, col] = values[:len(group_sorted)]

        # --- MERGE MATCH-LEVEL CON DF3 ---
        df3['player_norm'] = df3['player'].apply(self.remove_middle_name)
        df3['date'] = pd.to_datetime(df3['date'])

        df1 = df1.merge(
            df3[['player_norm', 'date', 'player_team', 'opponent_team']],
            on=['player_norm', 'date'],
            how='left'
        )

        # Rimuovi colonna temporanea
        df1 = df1.drop(columns=['player_norm'])

        return df1

    def reconcile_df2_by_partita(self, df2_player, df1_player):
        """
        Rimuove da df2_player le righe la cui 'partita' non è presente
        tra le partite ricostruite da df1_player (h_team + a_team).

        Args:
            df2_player (pd.DataFrame): voti giocatore (colonna 'partita')
            df1_player (pd.DataFrame): match-level (colonne 'h_team', 'a_team')

        Returns:
            pd.DataFrame: df2_player filtrato
        """
        # --- Normalizza squadre df1 ---
        h_norm = df1_player['h_team'].apply(
            lambda x: utils.normalize_team_name(x)
        )
        a_norm = df1_player['a_team'].apply(
            lambda x: utils.normalize_team_name(x)
        )

        partite = df2_player['partita'].apply(
            lambda x: utils.normalize_match_string(x)
        )
        # --- Costruzione partite valide ---
        valid_partite = (h_norm + " - " + a_norm).unique()
        valid_partite_set = set(valid_partite)

        valid_partite_set = set(valid_partite)

        # --- Maschera partite valide ---
        mask_valid = partite.isin(valid_partite_set)

        # --- Debug opzionale ---
        removed = df2_player.loc[~mask_valid, 'partita'].unique()
        if len(removed) > 0:
            print("⚠️ Partite rimosse da df2_player:")
            for p in removed:
                print("  -", p)

        # --- Filtra ---
        df2_player_clean = df2_player.loc[mask_valid].copy()

        return df2_player_clean

    def add_fanta_role(self,df_main, df_fanta_roles, debug=True):

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

    def normalize_team(self, name):
        if pd.isna(name):
            return None
        return name.strip().lower()

    def map_strength(self, team):
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
        self,
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
            .apply(self.normalize_team)
            .apply(self.map_strength)
        )

        return df