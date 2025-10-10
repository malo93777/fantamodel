from first_preproc import Preprocessor
import pandas as pd
from matchscraper import MatchScraper
from understatapi import UnderstatClient

# 1. Istanzia la classe
preproc = Preprocessor(
    serie_a_teams=["Milan", "Inter", "Juventus", "Roma", "Napoli", "Lazio", "Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Empoli", "Genoa", "Verona", "Lecce", "Udinese", "Monza", "Cagliari", "Frosinone", "Salernitana", "Chievo", "Spezia"], 
    top_teams=["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta"],
    mid_teams=["Bologna", "Fiorentina", "Torino", "Sassuolo", "Monza"],
    weak_teams=["Empoli", "Genoa", "Verona", "Lecce", "Udinese", "Cagliari", "Frosinone", "Salernitana"]
)

# 2. Preprocessa dataset tiri
shots_df = preproc.preproc_shots_dataset(
    input_path="shots_2025.csv", 
    df_to_merge_path="players_all_seasons.csv"
)

print("Preprocessed shots dataset:")
print(shots_df.head())

# 3.
 # fase del recuper partite mancanti
teams = shots_df["player_team"].unique().tolist()

matches_df = pd.DataFrame()

with UnderstatClient() as understat:
    scraper = MatchScraper(understat)
                
    # scarico tutte le partite
    matches_df = scraper.get_all_teams_matches(teams, 2014, 2025)

    #player_matches = scraper.get_player_matches("Marcus Thuram", shots_df, matches_df)


shots_df_completed = preproc.add_missing_games(shots_df, matches_df)

# 3. Calcolo rolling features (XG, shots, gol mean)
shots_df_completed = preproc.calculare_roll_features(shots_df_completed)

# 4. Carica dataset squadre (da understat scraper)
teams_df = pd.read_csv("teams_2014_2025.csv")

# 5. Crea feature XG per 90 min e unisci ai giocatori
teams_df, shots_df_completed = preproc.create_Xg_90min_teams(teams_df, shots_df_completed)

# 6. Crea feature XGA per 90 min della squadra avversaria
teams_df, shots_df_completed = preproc.create_XgA_90min_opponent(teams_df, shots_df_completed)

print("Dataset con team_xG_90min aggiunto:")
print(shots_df_completed.head())

#to csv
shots_df_completed.to_csv("PROD_shots_2025_preproc_Serie_A.csv", index=False)


