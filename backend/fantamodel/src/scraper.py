import pandas as pd
import config
import first_preproc
import asyncio
import json
import aiohttp
from understat import Understat

class Scraper:

    def run_scraper(self):
        scraper = Scraper()
        asyncio.run(scraper.run())

    async def run(self):

        async with aiohttp.ClientSession() as session:

                understat = Understat(session)
                # -----------------------------
                # 1) TEAM DATA
                # -----------------------------
                print("Scarico dati squadre stagione corrente...")
                team_data = await understat.get_teams("Serie_A", str(config.CURRENT_SEASON))

                preprocessor = first_preproc.Preprocessor(config.SERIE_A_TEAMS)
                teams_df = preprocessor.build_team_dataframe(team_data)

                teams_df.to_csv(config.DATASET_DATA_DIR / config.CURRENT_SEASON_TEAMS_FILE, index=False)
                print("Salvato:", config.CURRENT_SEASON_TEAMS_FILE)

                # -----------------------------
                # 2) PLAYER MATCH DATA (tutte le stagioni)
                # -----------------------------
                print("Scarico giocatori della stagione corrente...")
                league_players = await understat.get_league_players("Serie_A", str(config.CURRENT_SEASON))

                shots_df = pd.DataFrame()

                for player in league_players:
                    pid = player["id"]
                    pname = player["player_name"]

                    print(f"→ Match di {pname} ({pid})")

                    matches = await understat.get_player_matches(pid)
                    df = pd.DataFrame(matches)
                    df.insert(0, "player", pname)

                    shots_df = pd.concat([shots_df, df], ignore_index=True)

                shots_df.to_csv(config.DATASET_DATA_DIR / config.RAW_DATA_FILE, index=False)
                print("Salvato RAW players match:", config.RAW_DATA_FILE)

                # -----------------------------
                # 3) PLAYER SEASONS DATA
                # -----------------------------
                print("Scarico dati giocatori per tutte le stagioni...")

                players_seasons_df = pd.DataFrame()
                seasons = [str(year) for year in range(2014, config.CURRENT_SEASON + 1)]

                for season in seasons:
                    print("Stagione:", season)
                    pdata = await understat.get_league_players("Serie_A", season)
                    pdf = pd.DataFrame(pdata)
                    pdf["season"] = season
                    players_seasons_df = pd.concat([players_seasons_df, pdf], ignore_index=True)

                players_seasons_df.to_csv(config.DATASET_DATA_DIR / config.PLAYERS_ALL_SEASON_FILE, index=False)
                print("Salvato: players_all_seasons.csv")

                # -----------------------------
                # 4) TEAM TABLES PER STAGIONE
                # -----------------------------
                print("Scarico dati squadre per tutte le stagioni...")

                all_teams_df = pd.DataFrame()
                seasons = [str(year) for year in range(2014, config.CURRENT_SEASON + 1)]
                for season in seasons:
                    print("Stagione:", season)
                    teamdata = await understat.get_league_table("Serie_A", season)
                    # Se la prima riga contiene i nomi delle colonne
                    if teamdata and isinstance(teamdata[0], list) and all(isinstance(x, str) for x in teamdata[0]):
                        columns = teamdata[0]
                        data = teamdata[1:]
                        teams_df = pd.DataFrame(data, columns=columns)
                    else:
                        teams_df = pd.DataFrame(teamdata)

                    teams_df["season"] = season
                    all_teams_df = pd.concat([all_teams_df, teams_df], ignore_index=True)                
                
                all_teams_df = preprocessor.mod_df_teams(all_teams_df)
                all_teams_df.to_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE, index=False)
                print("Data saved to teams_2014_2026.csv")    


if __name__ == "__main__":
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(Scraper().run())
