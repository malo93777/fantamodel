from first_preproc import Preprocessor
import pandas as pd
from scraper import Scraper
from voti_scraper import VotiScraper
from fbref_nextgames_scraper import NextGamesScraper
from pianeta_fanta_infortuni_scraper import UnavailablePlayersScraper
import config
import argparse

#*** GLOBALS ***
all_season_player_df_path = config.DATASET_DATA_DIR / config.PLAYERS_ALL_SEASON_FILE

preproc = Preprocessor(
        serie_a_teams=config.SERIE_A_TEAMS
    )
#***

def get_goals_data():

    print("Starting goals data processing...")

    raw_df_path = config.DATASET_DATA_DIR / config.RAW_DATA_FILE

    # 1. Scraping
    scraper = Scraper()
    scraper.run_scraper()

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

    #print("Dataset con team_xG_90min aggiunto:")
    print(goals_df.head())

    #to csv
    goals_df.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS, index=False)

def get_assists_data():

    print("Starting assists data processing...")

    raw_df_path = config.DATASET_DATA_DIR / config.RAW_DATA_FILE

    # 1. Scraping

    # 2. Preprocessing
    assist_df = preproc.preproc_assists_dataset(
        input_path=raw_df_path, 
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

def get_voti_data():
    print("Starting voti data processing...")

    raw_df_path = config.DATASET_DATA_DIR / config.RAW_DATA_FILE
    voti_csv_path = config.DATASET_DATA_DIR / config.VOTI_DATA_FILE
    prod_goals_with_teams_player = config.DATASET_DATA_DIR / config.PROD_DATA_FILE_GOALS
    df_fanta_roles_path = config.DATASET_DATA_DIR / config.FANTA_RUOLI_FILE
    df_fanta_roles = pd.read_csv(df_fanta_roles_path) #file fantacalcio.it ruoli

    # 1. Scraping  TOLTO PER DEBUG
    scraper = VotiScraper()
    #scraper.run()

    # 2. Preprocessa dataset voti
    voti_df = preproc.merge_voti_player(
        raw_df_path, 
        voti_csv_path,
        prod_goals_with_teams_player
    )
    voti_df = preproc.add_fanta_role(voti_df, df_fanta_roles)

    voti_df = preproc.add_team_strength_column(voti_df, 'opponent_team', 'opponent_team_strength')
    voti_df = preproc.add_team_strength_column(voti_df, 'player_team', 'player_team_strength')

    print("Preprocessed voti dataset:")
    print(voti_df.head())

    #to csv
    voti_df.to_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI, index=False)

def get_next_games_data():

    print("Starting partite prossima giornata processing...")

    nextgames_scraper = NextGamesScraper()
    games_df = nextgames_scraper.run()

    print("Next games dataset:")
    print(games_df.head())
    #to csv
    games_df.to_csv(config.DATASET_DATA_DIR / config.NEXT_GAMES_FILE, index=False)

def get_infortunati():

    print("Starting scraping infortunati...")

    scraper = UnavailablePlayersScraper()
    players_out_df = scraper.run()
    #patch anguissa
    players_out_df['Giocatore'] = players_out_df['Giocatore'].replace(
            "Anguissa A.",
            "Anguissa Z."
        ) 

    print("Infortunati dataset:")
    print(players_out_df.head())

    #to csv
    players_out_df.to_csv(config.DATASET_DATA_DIR / config.INFORTUNATI_FILE, index=False)

def main():

    # ==========================
    # ARGOMENTI DA LINEA DI COMANDO
    # ==========================
    parser = argparse.ArgumentParser(description="FantaModel")
    parser.add_argument("--gol", action="store_true", help="Scraping e Prepocessing per il modello dei gol")
    parser.add_argument("--assist", action="store_true", help="Scraping e Prepocessing per il modello degli assist")
    parser.add_argument("--voti", action="store_true", help="Scraping e Prepocessing per il dataset dei voti")
    parser.add_argument("--nextgames", action="store_true", help="Scraping delle partite della prossima giornata")
    parser.add_argument("--infortunati", action="store_true", help="Scraping dei giocatori infortunati")
    args = parser.parse_args()
    args.gol = True
    args.assist = True
    args.voti = True
    args.nextgames = True
    args.infortunati = True
    # ==========================
    # ESECUZIONE
    # ==========================
        
    if args.gol and args.assist and args.voti and args.nextgames and args.infortunati:
        print("⚙️  Scraping e Prepocessing sia di GOL che ASSIST e VOTI che next games...")
        get_goals_data()
        get_assists_data()
        get_voti_data()
        get_next_games_data()
        get_infortunati()
    elif args.gol and args.assist and args.voti and args.nextgames:
        print("⚙️  Scraping e Prepocessing sia di GOL che ASSIST e VOTI ...")
        get_goals_data()
        get_assists_data()
        get_voti_data()
    elif args.gol and args.assist:
        print("⚙️  Scraping e Prepocessing sia di GOL che ASSIST...")
        get_goals_data()
        get_assists_data()
    elif args.gol:
        print("⚽  Scraping e Prepocessing solo GOL...")
        get_goals_data()
    elif args.assist:
        print("🎯  Scraping e Prepocessing solo ASSIST...")
        get_assists_data()
    elif args.voti:
        print("📝  Scraping e Prepocessing solo VOTI...")
        get_voti_data()
    elif args.nextgames:
        print("📅  Scraping solo Next Games...")
        get_next_games_data()
    elif args.infortunati:
        print("📅  Scraping solo infortunati...")
        get_infortunati()
    else:
        print("❗ Nessun argomento specificato. Usa --gol e/o --assist")

if __name__ == "__main__":
    main()   