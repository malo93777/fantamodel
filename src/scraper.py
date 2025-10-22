import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
from understatapi import UnderstatClient
from teamscraper import TeamsXGAScraper
import config

class Scraper:

    def run(self, debug):

        if debug == False:

            with UnderstatClient() as understat:           

                league_player_data = understat.league(league="Serie_A").get_player_data(season="2025")
                shots_df = pd.DataFrame()
                # Get the name and id of every player

                for index, player in enumerate(league_player_data):
                    player_id = player["id"]
                    player_name = player["player_name"]
                    print(f"Player ID: {player_id}, Player Name: {player_name}")

                    # Get the name and id of one of the player
                    player_id, player_name = league_player_data[index]["id"], league_player_data[index]["player_name"]
                    
                    # Get data for every match this player has taken in a league match (for all seasons)
                    player_match_data = understat.player(player=player_id).get_match_data()
                    player_match_data_df = pd.DataFrame(player_match_data)
                    #creo colonna player
                    player_match_data_df.insert(0, "player", player_name)    

                    if shots_df.empty:
                        shots_df = player_match_data_df
                    else:
                        #aaggiungo player_match_data_df senza header a shots_df
                        shots_df = pd.concat([shots_df, player_match_data_df], ignore_index=True)

            shots_df.to_csv(config.DATASET_DATA_DIR / config.RAW_DATA_FILE, index=False)
            print("Dati salvati in raw_data.csv")

            #Get overall data for a player in a season          
            players_seasons_df = pd.DataFrame()
            seasons = [str(year) for year in range(2014, 2026)]
            
            for season in seasons:
                print(f"Processing season: {season}")
                league_player_data_for_season = understat.league(league="Serie_A").get_player_data(season=season)
                league_player_data_for_season_df = pd.DataFrame(league_player_data_for_season)
                league_player_data_for_season_df['season'] = season
                if players_seasons_df.empty:
                    players_seasons_df = league_player_data_for_season_df
                else:                 
                    players_seasons_df = pd.concat([players_seasons_df, league_player_data_for_season_df], ignore_index=True)
            
            players_seasons_df.to_csv(config.DATASET_DATA_DIR / config.PLAYERS_ALL_SEASON_FILE, index=False)
            print("Dati salvati in players_all_seasons.csv")

            #*****  scraping classifiche stagioni per squadra*******
            teams_scraper = TeamsXGAScraper()
            seasons = [str(year) for year in range(2014, 2026)]
            teams_scraper.run(seasons) 
        else:
                     
            teams_scraper = TeamsXGAScraper()
            seasons = [str(year) for year in range(2014, 2026)]
            teams_scraper.run(seasons)       

if __name__ == "__main__":
    scraper = Scraper()
    scraper.run(False)
