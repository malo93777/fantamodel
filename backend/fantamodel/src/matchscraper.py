import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import re

class MatchScraper():

    def __init__(self, client):
        
        """
        Inizializza lo scraper con un UnderstatClient già aperto.
        """
        self.client = client


    def get_all_teams_matches(self, teams: list, start: int = 2014, end: int = 2026) -> pd.DataFrame:
        """
        Scarica le partite per tutte le squadre fornite.
        """
        all_matches = []
        for team in teams:
            for season in range(start, end + 1):
                print(f"Downloading {team} season {season}...")
                try:
                    matches = self.client.team(team=team).get_match_data(season=str(season))
                    matches_df = pd.DataFrame(matches)
                    matches_df["season"] = season
                    matches_df["team"] = team
                    all_matches.append(matches_df)
                except Exception as e:
                    print(f"⚠️ Errore per {team}, stagione {season}: {e}")
                    continue
        return pd.concat(all_matches, ignore_index=True)

    def filter_team_matches(self, matches_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
        """
        Filtra dal DataFrame solo le partite giocate da una determinata squadra.
        """
        mask = (matches_df["h"].apply(lambda x: x["title"]) == team_name) | \
               (matches_df["a"].apply(lambda x: x["title"]) == team_name)
        return matches_df[mask].reset_index(drop=True)
    

    def get_player_matches(self, player_name: str, shots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra tutte le partite di un giocatore (sia team che opponent) incrociando con matches_df.
        """
        # prendo solo le righe del giocatore
        player_df = shots_df[shots_df["player"].str.contains(player_name, case=False, na=False)].copy()

        # join sulle colonne squadra e avversario
        merged = player_df.merge(
            matches_df,
            left_on=["player_team", "opponent_team", "season"],
            right_on=["team", "opponent", "season"],
            how="left"
        )

        return merged
    
    def enrich_shots_with_matches(self, shots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge le informazioni delle partite (xG, xGA ecc.) al dataset dei tiri,
        per tutti i giocatori in shots_df.
        """
        enriched_df = shots_df.merge(
            matches_df,
            left_on=["player_team", "opponent_team", "season"],
            right_on=["team", "opponent", "season"],
            how="left"
        )
        return enriched_df
