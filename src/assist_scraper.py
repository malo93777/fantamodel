import pandas as pd
from understatapi import UnderstatClient
import config

class AssistScraper:

    def run(self, debug):
        if not debug:
            return 
        with UnderstatClient() as understat:           

                    league_player_data = understat.league(league="Serie_A").get_player_data(season=str(config.CURRENT_SEASON))
                    assist_df = pd.DataFrame()
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

                        if assist_df.empty:
                            assist_df = player_match_data_df
                        else:
                            #aggiungo player_match_data_df senza header a assist_df
                            assist_df = pd.concat([assist_df, player_match_data_df], ignore_index=True)

                    assist_df.to_csv(config.DATASET_DATA_DIR / config.ASSIST_DATA_FILE, index=False)
                    print("Dati salvati in assists_2025.csv")