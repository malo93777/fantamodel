import requests
import pandas as pd
from datetime import datetime, timezone
import config

class NextGamesScraper:
    def __init__(self):
        self.url = "https://www.fotmob.com/api/leagues?id=55"  # Serie A
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

    def run(self) -> pd.DataFrame:
        r = requests.get(self.url, headers=self.headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        matches = data["fixtures"]["allMatches"]
        rows = []

        for m in matches:
            status = m.get("status", {})

            # solo partite NON finite (future)
            if status.get("finished") is True and status.get("round") != config.NEXT_GIORNATA:
                continue

            utc_time = status.get("utcTime")
            if not utc_time:
                continue

            dt = datetime.fromisoformat(
                utc_time.replace("Z", "+00:00")
            )

            rows.append({
                "round": int(m["round"]),
                "date": dt.date(),
                "time": dt.time(),
                "home": m["home"]["name"],
                "away": m["away"]["name"],
                "match_id": m["id"]
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # 🔑 prossima giornata reale
        next_matchday = (
            df[df["round"] == config.NEXT_GIORNATA]
            .sort_values(["date", "time"])
            .reset_index(drop=True)
        )

        return next_matchday