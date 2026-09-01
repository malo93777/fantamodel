import requests
import pandas as pd
import config


class TeamsXGAScraper:

    def get_league_data(self, season):
        """
        Chiama direttamente l'endpoint AJAX interno di Understat,
        che restituisce già i dati in formato JSON (niente scraping HTML).
        """
        url = f"https://understat.com/getLeagueData/Serie_A/{season}"
        response = requests.get(url, headers={"X-Requested-With": "XMLHttpRequest"})

        print(f"[{season}] Status:", response.status_code)
        response.raise_for_status()  # solleva un errore chiaro se lo status non è 200

        data = response.json()
        print(f"[{season}] Chiavi disponibili nel JSON:", list(data.keys()))

        return data

    def load_teams(self, data, season):
        """
        Costruisce il DataFrame delle squadre a partire dal JSON
        restituito dall'endpoint getLeagueData.
        """
        teams_data = data.get("teams")

        if not teams_data:
            raise ValueError(
                f"ERRORE: 'teams' non trovato per la stagione {season}. "
                f"Chiavi disponibili: {list(data.keys())}"
            )

        rows = []

        for team_id, team_info in teams_data.items():
            team_name = team_info["title"]
            history = team_info["history"]

            matches = len(history)
            wins = sum(1 for h in history if int(h["scored"]) > int(h["missed"]))
            draws = sum(1 for h in history if int(h["scored"]) == int(h["missed"]))
            loses = matches - wins - draws
            goals_for = sum(int(h["scored"]) for h in history)
            goals_against = sum(int(h["missed"]) for h in history)
            points = wins * 3 + draws

            xG = round(sum(float(h["xG"]) for h in history), 2)
            xGA = round(sum(float(h["xGA"]) for h in history), 2)
            npxG = round(sum(float(h["npxG"]) for h in history), 2)
            npxGA = round(sum(float(h["npxGA"]) for h in history), 2)

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
                "npxGA": npxGA,
            })

        df_teams = self.mod_df_teams(pd.DataFrame(rows))
        return df_teams

    def mod_df_teams(self, df):
        # Mantieni qui la tua logica esistente di post-elaborazione del DataFrame
        return df

    def run(self, seasons):
        all_teams_df = pd.DataFrame()

        for season in seasons:
            print(f"Processing season: {season}")

            try:
                data = self.get_league_data(season)
                teams_df = self.load_teams(data, season)
            except Exception as e:
                print(f"[{season}] Errore, stagione saltata: {type(e).__name__}: {e}")
                continue

            teams_df["season"] = season
            all_teams_df = pd.concat([all_teams_df, teams_df], ignore_index=True)

        all_teams_df.to_csv(config.DATASET_DATA_DIR / config.TEAMS_DATA_FILE, index=False)
        print("Data saved to teams_2014_2026.csv")


if __name__ == "__main__":
    seasons = [str(year) for year in range(2014, 2027)]
    scraper = TeamsXGAScraper()
    scraper.run(seasons)