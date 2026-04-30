import requests
import pandas as pd
from datetime import datetime
import config


class NextGamesScraper:
    def __init__(self):
        self.url = "https://www.fotmob.com/api/data/leagues?id=55&ccode3=ITA"

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            "Referer": "https://www.fotmob.com/it/leagues/55",
            "Origin": "https://www.fotmob.com"
        })

    def run(self) -> pd.DataFrame:
        r = self.session.get(self.url, timeout=10)

        print("STATUS:", r.status_code)
        print("CONTENT:", r.text[:200])

        if r.status_code != 200:
            raise ValueError(f"HTTP error: {r.status_code}")

        if not r.text.strip():
            raise ValueError("Risposta vuota")

        if r.text.startswith("<"):
            raise ValueError("Bloccato o endpoint errato (HTML ricevuto)")

        data = r.json()

        # 👇 struttura corretta FotMob
        matches = data.get("fixtures", {}).get("allMatches", [])

        rows = []

        for m in matches:
            status = m.get("status", {})

            # solo partite future
            if status.get("finished") is True:
                continue

            utc_time = status.get("utcTime")
            if not utc_time:
                continue

            dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))

            rows.append({
                "round": int(m.get("round", 0)),
                "date": dt.date(),
                "time": dt.time(),
                "home": m.get("home", {}).get("name"),
                "away": m.get("away", {}).get("name"),
                "match_id": m.get("id")
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        next_matchday = (
            df[df["round"] == config.NEXT_GIORNATA]
            .sort_values(["date", "time"])
            .reset_index(drop=True)
        )

        return next_matchday