import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
from understatapi import UnderstatClient
from teamscraper import TeamsXGAScraper

class Scraper:

    def create_XG90min_player_team(self, teams_df, players_df):
        #creo colonna XG_90min per ogni squadra arrondando a 2 decimali e salvo in nuovo csv    
        teams_df["XG_90min"] = round(teams_df["xG"] / (teams_df["M"]), 2)
        # Unisco il DataFrame dei giocatori con il DataFrame delle squadre per ottenere l'XG_90min della squadra del giocatore
        players_df = players_df.merge(
            teams_df.rename(columns={"Team": "player_team"})[["player_team", "season", "XG_90min"]],
            on=["player_team", "season"],
            how="left"
        )

        players_df.rename(columns={"XG_90min": "team_xG_90min"}, inplace=True)

        print(players_df.head(5))

        return players_df       

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
                    
                    # Get data for every shot this player has taken in a league match (for all seasons)
                    player_shot_data = understat.player(player=player_id).get_shot_data()
                    player_shot_data_df = pd.DataFrame(player_shot_data)       

                    if shots_df.empty:
                        shots_df = player_shot_data_df
                    else:
                        #aaggiungo player_shot_data_df senza header a shots_df
                        shots_df = pd.concat([shots_df, player_shot_data_df], ignore_index=True)

            shots_df.to_csv("shots_2025.csv", index=False)
            print("Dati salvati in shots_2025.csv")           

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
            
            players_seasons_df.to_csv("players_all_seasons.csv", index=False)
            print("Dati salvati in players_all_seasons.csv")
        else:
                     
            teams_scraper = TeamsXGAScraper()
            seasons = [str(year) for year in range(2014, 2026)]
            teams_scraper.run(seasons)       

if __name__ == "__main__":
    scraper = Scraper()
    scraper.run(False)
