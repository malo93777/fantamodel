"""
Fantagiaveno – Full Pipeline
============================

SEZIONE A: SCRAPING
SEZIONE B: FEATURE ENGINEERING

Autore: tu 😄
Uso etico:
- User-Agent
- time.sleep(2)
"""

# ============================================================
# IMPORT
# ============================================================

import time
import requests
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup
import re
import config
# ============================================================
# Classe VotiScraper
# ============================================================

class VotiScraper:
    def run(self):
        SQUADRE = [
            "Atalanta","Bologna","Cagliari","Empoli","Fiorentina",
            "Genoa","Inter","Juventus","Lazio","Lecce",
            "Milan","Monza","Napoli","Roma","Salernitana",
            "Sassuolo","Torino","Udinese","Verona",
            "Cremonese","Parma","Pisa","Como","Venezia",
            "Brescia","Spal","Frosinone","Cesena","Carpi","Benevento"
        ]

        MAX_GIORNATE = 38
        SLEEP_SEC = 1

        session = requests.Session()

        all_rows = []
        seasons_found = set()
        current_season = None

        for giornata in range(1, MAX_GIORNATE + 1):
            print(f"\n▶ Giornata {giornata}")

            giornata_rows = []

            for squadra in SQUADRE:
                try:
                    rows = scrape_match(session, giornata, squadra)

                    if not rows:
                        continue

                    stagione = rows[0]["stagione"]

                    # 🔹 prima stagione trovata
                    if current_season is None:
                        current_season = stagione
                        print(f"📅 Stagione rilevata: {stagione}")

                    # 🔹 cambio stagione automatico
                    elif stagione != current_season:
                        print(f"🔄 Cambio stagione: {current_season} → {stagione}")
                        current_season = stagione

                    seasons_found.add(stagione)

                    print(f"    ✅ {squadra}: {len(rows)} giocatori")
                    giornata_rows.extend(rows)

                    time.sleep(SLEEP_SEC)

                except Exception as e:
                    print(f"    ❌ {squadra}: {e}")

            # 🔹 se una giornata non produce dati → fine scraping
            if not giornata_rows:
                print("⛔ Nessun dato trovato → fine scraping")
                break

            all_rows.extend(giornata_rows)

        # 🔹 OUTPUT
        print(f"\n📊 Stagioni trovate: {sorted(seasons_found)}")

        df = pd.DataFrame(all_rows)
        df.to_csv("voti_fantagiaveno_raw.csv", index=False)

        print(f"💾 Salvati {len(df)} record in voti_fantagiaveno_raw.csv")

# ============================================================
# ====================== SEZIONE A ===========================
# ======================== SCRAPING ==========================
# ============================================================

BASE_URL = "https://www.fantagiaveno.it"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_SECONDS = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "it-IT,it;q=0.9",
    "Referer": "http://www.fantagiaveno.it/",
    "Connection": "keep-alive"
}

session = requests.Session()
session.headers.update(HEADERS)

# ------------------------------------------------------------
# UTILS SCRAPING
# ------------------------------------------------------------

def get_soup(url, params=None, session=None):
    if session is None:
        session = requests.Session()

    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")




def normalize_name(name: str) -> str:
    """Normalizza nome giocatore"""
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    return "".join(c for c in name if unicodedata.category(c) != "Mn")


def parse_int(x):
    try:
        return int(x)
    except:
        return 0


def parse_float(x):
    try:
        return float(x.replace(",", "."))
    except:
        return None

MATCH_RE = re.compile(r"id=(\d+)&ids=([^&]+)")

def discover_match_urls(seed_url):
    soup = get_soup(seed_url)
    urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "voti-fantacalcio-gazzetta.asp" in href:
            m = MATCH_RE.search(href)
            if m:
                full_url = BASE_URL + "/" + href.lstrip("/")
                urls.add(full_url)

    return sorted(urls)

def discover_all_match_ids():
    """
    Scansiona il sito e trova TUTTI gli id match reali
    """
    print("🔍 Scoperta URL partite...")
    match_ids = set()

    # pagina indice generica (basta una)
    url = "https://www.fantagiaveno.it"
    soup = get_soup(url)

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "voti-fantacalcio-gazzetta.asp?id=" in href:
            try:
                id_part = href.split("id=")[1].split("&")[0]
                match_ids.add(int(id_part))
            except:
                pass

    print(f"✅ Trovati {len(match_ids)} match.")
    return sorted(match_ids)

import re

def extract_season(soup):
    td = soup.find("td", class_="SottoTitoloPiccolo")
    if not td:
        return None

    text = td.get_text(strip=True).upper()

    # match "CAMPIONATO 2025 / 2026"
    m = re.search(r"CAMPIONATO\s+(\d{4})\s*/\s*(\d{4})", text)
    if not m:
        return None

    return f"{m.group(1)}-{m.group(2)}"

def set_stagione(session, stagione_id):
    url = "https://www.fantagiaveno.it/stagione.asp"
    params = {"id": stagione_id}
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()


# ------------------------------------------------------------
# SCRAPE SINGOLA PARTITA
# ------------------------------------------------------------

def scrape_match(session, giornata, squadra):
    url = "https://www.fantagiaveno.it/voti-fantacalcio-gazzetta.asp"
    params = {
        "id": giornata,
        "ids": squadra
    }

    soup = get_soup(url, params=params, session=session)
    stagione = extract_season(soup)

    table = find_votes_table(soup)
    if table is None:
        print(f"⚠️ Tabella voti NON trovata ({squadra}, G{giornata})")
        return []

    rows = table.find_all("tr")
    data = []

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 14:
            continue

        name_tag = tds[1].find("a")
        if not name_tag:
            continue

        player = name_tag.get_text(strip=True)

        voto = parse_float(tds[3].get_text(strip=True))
        fantavoto = parse_float(tds[2].get_text(strip=True))

        gol = parse_int(tds[4].get_text(strip=True))
        assist = parse_int(tds[5].get_text(strip=True))
        ammonizioni = parse_int(tds[6].get_text(strip=True))
        espulsioni = parse_int(tds[7].get_text(strip=True))
        autogol = parse_int(tds[8].get_text(strip=True))

        rig_txt = tds[9].get_text(strip=True)
        if "/" in rig_txt:
            segnati, tentati = rig_txt.split("/")
            rig_segnati = parse_int(segnati)
            rig_sbagliati = max(0, parse_int(tentati) - rig_segnati)
        else:
            rig_segnati = 0
            rig_sbagliati = 0

        data.append({
            "stagione": stagione,
            "giornata": giornata,
            "squadra": squadra,
            "player": player,
            "player_norm": normalize_name(player),
            "voto_gds": voto,
            "fantavoto": fantavoto,
            "gol": gol,
            "assist": assist,
            "ammonizioni": ammonizioni,
            "espulsioni": espulsioni,
            "autogol": autogol,
            "rig_segnati": rig_segnati,
            "rig_sbagliati": rig_sbagliati
        })

    return data

# ------------------------------------------------------------
# FANTAVOTO
# ------------------------------------------------------------

def compute_fantavoto(row):
    """
    Fantavoto Gazzetta
    Autogol = -3
    """
    if row["voto_gds"] is None:
        return None

    fv = row["voto_gds"]
    fv += 3 * row["gol"]
    fv += 1 * row["assist"]
    fv -= 0.5 * row["ammonizioni"]
    fv -= 1 * row["espulsioni"]
    fv -= 3 * row["autogol"]
    fv -= 3 * row["rig_sbagliati"]
    fv += 3 * row["rig_segnati"]

    return round(fv, 2)


# ============================================================
# ====================== SEZIONE B ===========================
# ================== FEATURE ENGINEERING =====================
# ============================================================

def add_fantavoto(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonna fantavoto"""
    df = df.copy()
    df["fantavoto"] = df.apply(compute_fantavoto, axis=1)
    return df


def add_rolling_fantamedia(
    df: pd.DataFrame,
    windows=(5, 10, 15),
    player_col="player_norm"
) -> pd.DataFrame:
    """
    Fantamedia rolling senza leakage (shift 1)
    """
    df = df.copy()
    df = df.sort_values([player_col, "stagione", "giornata"])

    for w in windows:
        df[f"fantamedia_{w}"] = (
            df.groupby(player_col)["fantavoto"]
              .apply(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
              .reset_index(level=0, drop=True)
        )

    return df


def build_match_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa feature-engineering
    """
    df = add_fantavoto(df)
    df = add_rolling_fantamedia(df)
    return df

def find_votes_table(soup):
    tables = soup.find_all("table")

    for table in tables:
        headers = table.find_all("th")
        header_texts = [h.get_text(strip=True).lower() for h in headers]

        if (
            "nome" in header_texts
            and "voto" in header_texts
            and "punti" in header_texts
        ):
            return table

    return None


# ============================================================
# ========================= MAIN =============================
# ============================================================

if __name__ == "__main__":
    scraper = VotiScraper()
    scraper.run()


