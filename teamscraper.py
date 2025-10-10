import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import re

class TeamsXGAScraper:

    def get_url_data(self, season):
        url = f"https://understat.com/league/Serie_A/{season}"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        return soup

    def mod_df_teams(self, df):

        #*** funzione per aggiungere colonne al df di base che non sono presenti nel dataset originale ***

        df_mod = df.copy()
        df_mod.index = df_mod.index + 1

        #aggiungo dati per differenza tra xG e gol, xGA e gol subiti
        df_mod['diff_XG_GOL'] = df_mod['xG'] - df_mod['G']
        df_mod['diff_xGA_GOLAG'] = df_mod['xGA'] - df_mod['GA']
        df_mod['XGA_90min'] = df_mod['xGA'] / df_mod['M']# xGAgainst per 90 minuti
        df_mod['XG_90min'] = df_mod['xG'] / df_mod['M'] # xG per 90 minuti

        #tronco a 2 valori decimali i float
        df_mod = df_mod.round({"diff_XG_GOL": 2, "diff_xGA_GOLAG": 2, "XGA_90min": 2})

        return df_mod


    def load_teams(self, soup):

        #soup = get_url_data()
        # Trova lo script con "teamsData"
        scripts = soup.find_all("script")

        scripts = soup.find_all("script")
        print(f"Trovati {len(scripts)} script nella pagina.")

        for i, script in enumerate(scripts):
            if not script.string:
                continue
            text = script.string
            # cerca pattern JSON.parse(
            if "JSON.parse" in text:
                print(f"\n--- Script {i} con JSON.parse ---")
                # mostra i primi 500 caratteri per capire
                print(text[:500])
            # cerca parole chiave
            if re.search(r"\b(teamsData|leagueTable|standingsData|positions)\b", text):
                print(f"\n--- Script {i} con parola chiave alternativa ---")
                print(text[:500])

        target = None
        for script in scripts:
            if "teamsData" in script.text:
                target = script.string
                break

        # Estrai il JSON da JSON.parse(' ... ')
        start = target.find("JSON.parse('") + len("JSON.parse('")
        end = target.find("')", start)
        json_raw = target[start:end]
        json_raw = json_raw.encode('utf-8').decode('unicode_escape')

        teams_data = json.loads(json_raw)

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
            points = wins*3 + draws
            
        # Somma degli expected stats
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
        
        all_teams_df.to_csv("teams_2014_2025.csv", index=False)
        print("Data saved to teams_2014_2025.csv")

if __name__ == "__main__":
    seasons = [str(year) for year in range(2014, 2025)]
    scraper = TeamsXGAScraper()
    scraper.run(seasons)