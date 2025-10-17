from first_preproc import Preprocessor
import pandas as pd
from matchscraper import MatchScraper
from understatapi import UnderstatClient
from scraper import Scraper
from assist_scraper import AssistScraper
import config
import argparse

#*** GLOBALS ***
all_season_player_df_path = config.DATASET_DATA_DIR / config.PLAYERS_ALL_SEASON_FILE

preproc = Preprocessor(
        serie_a_teams=config.SERIE_A_TEAMS
    )
#***

def get_goals_data():

    shots_df_path = config.DATASET_DATA_DIR / config.SHOTS_DATA_FILE

    # 1. Scraping
    scraper = Scraper()
    scraper.run(debug=False)

    # 3. Preprocessa dataset tiri
    shots_df = preproc.preproc_shots_dataset(
        input_path=shots_df_path, 
        df_to_merge_path=all_season_player_df_path
    )

    print("Preprocessed shots dataset:")
    print(shots_df.head())

    # 4.
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

    # 5. Calcolo rolling features (XG, shots, gol mean)
    shots_df_completed = preproc.calculate_roll_features(shots_df_completed)

    # 6. Carica dataset squadre (da understat scraper)
    teams_df = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

    # 7. Crea feature XG per 90 min e unisci ai giocatori
    teams_df, shots_df_completed = preproc.create_Xg_90min_teams(teams_df, shots_df_completed)

    # 8. Crea feature XGA per 90 min della squadra avversaria
    teams_df, shots_df_completed = preproc.create_XgA_90min_opponent(teams_df, shots_df_completed)

    print("Dataset con team_xG_90min aggiunto:")
    print(shots_df_completed.head())

    #to csv
    shots_df_completed.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE, index=False)

def get_assists_data():

    assist_df_path = config.DATASET_DATA_DIR / config.ASSIST_DATA_FILE

    # 1. Scraping
    scraper = AssistScraper()
    scraper.run(debug=False)

    # 2. Preprocessing
    assist_df = preproc.preproc_assists_dataset(
        input_path=assist_df_path, 
        df_to_merge_path=all_season_player_df_path
    )

    # 3. fase del recupero partite mancanti (assumo che il csv esista già)
    #matches_df = pd.read_csv(config.DATASET_DATA_DIR / config.MATCH_DATA_FILE)  
    
    #all_season_player_df = pd.read_csv(all_season_player_df_path)
    #assist_df_completed = preproc.add_missing_games(assist_df, matches_df, all_season_player_df)

    # 5. Calcolo rolling features (Xa, assists mean)
    assist_df = preproc.calculate_roll_features_assist(assist_df)

    # 6. Carica dataset squadre (da understat scraper)
    teams_df = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

    # 7. Crea feature XG per 90 min e unisci ai giocatori
    teams_df, assist_df = preproc.create_Xg_90min_teams(teams_df, assist_df)

    # 8. Crea feature XGA per 90 min della squadra avversaria
    teams_df, assist_df = preproc.create_XgA_90min_opponent(teams_df, assist_df)

    print("Dataset con team_xG_90min aggiunto:")
    print(assist_df.head())

    #to csv
    assist_df.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_ASSIST, index=False)

def main():

    matches_df = pd.DataFrame()

    # ==========================
    # ARGOMENTI DA LINEA DI COMANDO
    # ==========================
    parser = argparse.ArgumentParser(description="FantaModel")
    parser.add_argument("--gol", action="store_true", help="Scraping e Prepocessing per il modello dei gol")
    parser.add_argument("--assist", action="store_true", help="Scraping e Prepocessing per il modello degli assist")

    args = parser.parse_args()
    args.gol = True
    args.assist = False
    # ==========================
    # ESECUZIONE
    # ==========================
    if args.gol and args.assist:
        print("⚙️  Scraping e Prepocessing sia di GOL che ASSIST...")
        get_goals_data()
        get_assists_data()
    elif args.gol:
        print("⚽  Scraping e Prepocessing GOL...")
        get_goals_data()
    elif args.assist:
        print("🎯  Scraping e Prepocessing ASSIST...")
        get_assists_data()
    else:
        print("❗ Nessun argomento specificato. Usa --gol e/o --assist")

if __name__ == "__main__":
    main()   