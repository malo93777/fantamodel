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

    raw_df_path = config.DATASET_DATA_DIR / config.RAW_DATA_FILE

    # 1. Scraping
    scraper = Scraper()
    scraper.run(debug=False)

    # 2. Preprocessa dataset tiri
    goals_df = preproc.preproc_goals_dataset(
        input_path=raw_df_path, 
        df_to_merge_path=all_season_player_df_path,
        is_SerieA=config.IS_SERIEA
    )

    print("Preprocessed goals dataset:")
    print(goals_df.head())

    if config.IS_SERIEA == False:
        return

    # 3. Carica dataset squadre (da understat scraper)
    teams_df = pd.read_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE)

    # 4. Crea feature XG per 90 min e unisci ai giocatori
    teams_df, goals_df = preproc.create_Xg_90min_teams(teams_df, goals_df)

    # 5. Crea feature XGA per 90 min della squadra avversaria
    teams_df, goals_df = preproc.create_XgA_90min_opponent(teams_df, goals_df)

    print("Dataset con team_xG_90min aggiunto:")
    print(goals_df.head())

    #to csv
    goals_df.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS, index=False)

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