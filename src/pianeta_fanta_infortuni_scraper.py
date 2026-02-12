import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import pandas as pd

class UnavailablePlayersScraper:
    def __init__(self):
        self.url = "https://www.pianetafanta.it/giocatori-infortunati.asp"
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def run(self) -> pd.DataFrame:
        r = requests.get(self.url, headers=self.headers, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        players = []

        links = soup.find_all(
            "a",
            href=lambda x: x and "giocatori-statistiche-personali.asp?nomegio=" in x
        )

        for link in links:
            name = link.get_text(strip=True)
            if name:
                players.append(name.title())

        # rimuove duplicati e ordina
        players = sorted(set(players))

        # crea DataFrame
        df = pd.DataFrame(players, columns=["Giocatore"])

        return df


# 🔹 TEST STAND-ALONE
if __name__ == "__main__":
    scraper = UnavailablePlayersScraper()
    players = scraper.run()

    print(f"Trovati {len(players)} giocatori indisponibili:\n")
    for p in sorted(players):
        print(p)