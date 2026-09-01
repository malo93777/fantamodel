import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import re
import config

class TeamsXGAScraper:

    def get_url_data(self, season):
        url = f"https://understat.com/league/Serie_A/{season}"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        return soup

    def load_teams(self, soup):

        scripts = soup.find_all("script")
        print(f"Trovati {len(scripts)} script nella pagina.")

        # Regex per catturare il JSON interno
        pattern = r"JSON\.parse\(\s*'([^']+)'\s*\)"
        
        teams_data = None

        # Proviamo a trovare il JSON corretto
        for i, script in enumerate(scripts):
            if not script.string:
                continue

            match = re.search(pattern, script.string, re.DOTALL)
            if not match:
                continue

            raw_json = match.group(1)

            try:
                decoded = raw_json.encode('utf-8').decode('unicode_escape')
                data = json.loads(decoded)
            except Exception:
                continue

            # Qui cerchiamo veramente i dati delle squadre
            if "teamsData" in data:
                print(f"Trovato teamsData nello script {i}")
                teams_data = data["teamsData"]
                break

            # A volte è annidato
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict) and "teamsData" in val:
                        print(f"Trovato teamsData annidato nello script {i}")
                        teams_data = val["teamsData"]
                        break

            if teams_data is not None:
                break

        if teams_data is None:
            raise ValueError("ERRORE: impossibile estrarre 'teamsData' — struttura Understat cambiata?")

        # 🎯 --- COSTRUZIONE DEL DATAFRAME COME FA LA TUA VERSIONE ORIGINALE ---
        rows = []

        for team_id, team_info in teams_data.items():
            team_name = team_info['title']
            history = team_info['history']

            matches = len(history)
            wins = sum(1 for h in history if int(h['scored']) > int(h['missed']))
            draws = sum(1 for h in history if int(h['scored']) == int(h['missed']))
            loses = matches - wins - draws
            goals_for = sum(int(h['scored']) for h in history)
            goals_against = sum(int(h['missed']) for h in history)
            points = wins * 3 + draws

            # expected stats
            xG = round(sum(float(h['xG']) for h in history), 2)
            xGA = round(sum(float(h['xGA']) for h in history), 2)
            npxG = round(sum(float(h['npxG']) for h in history), 2)
            npxGA = round(sum(float(h['npxGA']) for h in history), 2)

            rows.append({
                "Team": team_name,
                "M": matches,
                "W": wins,
                "D": draws,
                "L": loses,
                "G": goals_for,
                "GA": goals_against,
                "PTS": points,
                "xG": xG,
                "xGA": xGA,
                "npxG": npxG,
                "npxGA": npxGA
            })

        df_teams = self.mod_df_teams(pd.DataFrame(rows))
        return df_teams


    def run(self, seasons):
        all_teams_df = pd.DataFrame()
        for season in seasons:
            print(f"Processing season: {season}")
            soup = self.get_url_data(season)
            teams_df = self.load_teams(soup)
            teams_df['season'] = season  # Aggiungi colonna stagione
            all_teams_df = pd.concat([all_teams_df, teams_df], ignore_index=True)
        
        all_teams_df.to_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE, index=False)
        print("Data saved to teams_2014_2026.csv")

if __name__ == "__main__":
    seasons = [str(year) for year in range(2014, 2027)]
    scraper = TeamsXGAScraper()
    scraper.run(seasons)