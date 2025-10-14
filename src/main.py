from first_preproc import Preprocessor
import pandas as pd
from matchscraper import MatchScraper
from understatapi import UnderstatClient
from scraper import Scraper
import config

shots_df_path = config.DATASET_DATA_DIR / config.SHOTS_DATA_FILE
all_season_player_df_path = config.DATASET_DATA_DIR / config.PLAYERS_ALL_SEASON_FILE

scraper = Scraper()
scraper.run(debug=False)

# 1. Istanzia la classe
preproc = Preprocessor(
    serie_a_teams=["Milan", "Inter", "Juventus", "Roma", "Napoli", "Lazio", "Atalanta", "Fiorentina", "Torino", "Bologna", "Sassuolo", "Empoli", "Genoa", "Verona", "Lecce", "Udinese", "Monza", "Cagliari", "Frosinone", "Salernitana", "Chievo", "Spezia"]
)

# 2. Preprocessa dataset tiri
shots_df = preproc.preproc_shots_dataset(
    input_path=shots_df_path, 
    df_to_merge_path=all_season_player_df_path
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
    match_df_path = matches_df.to_csv(config.DATASET_DATA_DIR / config.MATCH_DATA_FILE)

all_season_player_df = pd.read_csv(all_season_player_df_path)
shots_df_completed = preproc.add_missing_games(shots_df, matches_df, all_season_player_df)

# 3. Calcolo rolling features (XG, shots, gol mean)
shots_df_completed = preproc.calculare_roll_features(shots_df_completed)

# 4. Carica dataset squadre (da understat scraper)
teams_df = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

# 5. Crea feature XG per 90 min e unisci ai giocatori
teams_df, shots_df_completed = preproc.create_Xg_90min_teams(teams_df, shots_df_completed)

# 6. Crea feature XGA per 90 min della squadra avversaria
teams_df, shots_df_completed = preproc.create_XgA_90min_opponent(teams_df, shots_df_completed)

print("Dataset con team_xG_90min aggiunto:")
print(shots_df_completed.head())

#to csv
shots_df_completed.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE, index=False)
