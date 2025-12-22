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

def get_soup(url, params=None):
    time.sleep(SLEEP_SECONDS)
    r = session.get(url, params=params, timeout=20)
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

# ------------------------------------------------------------
# SCRAPE SINGOLA PARTITA
# ------------------------------------------------------------

def scrape_match(url, stagione, giornata, squadra):

    soup = get_soup(url)
    rows = soup.find_all("tr")
    data = []

    for tr in rows:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 6:
            continue

        cols = [c.get_text(strip=True) for c in cells]

        player = cols[0]

        # filtri anti-spazzatura
        if not player:
            continue
        if player.upper() in ["ALLENATORE", "PAN.", "PAN"]:
            continue
        if player.replace(".", "").isdigit():
            continue

        voto_raw = cols[1]
        voto = parse_float(voto_raw) if voto_raw not in ["", "-", "SV"] else None

        row = {
            "stagione": stagione,
            "giornata": giornata,
            "squadra": squadra,
            "player": player,
            "player_norm": normalize_name(player),

            "voto_gds": voto,
            "gol": parse_int(cols[2]) if len(cols) > 2 else 0,
            "assist": parse_int(cols[3]) if len(cols) > 3 else 0,
            "ammonizioni": parse_int(cols[4]) if len(cols) > 4 else 0,
            "espulsioni": parse_int(cols[5]) if len(cols) > 5 else 0,
            "autogol": parse_int(cols[6]) if len(cols) > 6 else 0,
            "rig_sbagliati": parse_int(cols[7]) if len(cols) > 7 else 0,
            "rig_segnati": parse_int(cols[8]) if len(cols) > 8 else 0,
        }

        data.append(row)

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
        rows = table.find_all("tr")
        if len(rows) < 10:
            continue

        # prova a leggere la prima riga dati
        for tr in rows:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            first_cell = tds[0].get_text(strip=True)

            # se sembra un nome giocatore → è la tabella giusta
            if first_cell and any(c.isalpha() for c in first_cell):
                return table

    return None


# ============================================================
# ========================= MAIN =============================
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # CONFIG
    # ============================================================

    STAGIONI = [
        "2015-2016", "2016-2017", "2017-2018", "2018-2019",
        "2019-2020", "2020-2021", "2021-2022", "2022-2023",
        "2023-2024", "2024-2025", "2025-2026"
    ]

    # 380 partite ≈ stagione completa
    MATCH_ID_RANGE = range(1, 38)

    SQUADRE = config.SERIE_A_TEAMS

    OUTPUT_RAW = "fantagiaveno_raw_votes.csv"
    OUTPUT_FEAT = "fantagiaveno_features.csv"

    all_rows = []

    print("🚀 Inizio scraping Fantagiaveno")

    # ============================================================
    # SCRAPING LOOP
    # ============================================================

    for stagione in STAGIONI:
        print(f"\n📅 Stagione {stagione}")

        for match_id in MATCH_ID_RANGE:
            print(f"  ▶ Match ID {match_id}")

            for squadra in SQUADRE:
                print(f"    • Squadra: {squadra}")
                try:
                    url = f"{BASE_URL}/voti-fantacalcio-gazzetta.asp"
                    params = {
                        "id": match_id,
                        "ids": squadra
                    }

                    print(f"      → Richiesta: stagione={stagione}, match_id={match_id}, squadra={squadra}")
                    rows = scrape_match(
                        url=url,
                        stagione=stagione,
                        giornata=None,      # non disponibile direttamente
                        squadra=squadra
                    )

                    if rows:
                        print(f"      ✓ {len(rows)} righe trovate per {squadra} (match {match_id})")
                        all_rows.extend(rows)
                    else:
                        print(f"      ⚠️ Nessun dato trovato per {squadra} (match {match_id})")

                except Exception as e:
                    print(f"      ❌ Errore {stagione} ID {match_id} {squadra}: {e}")

    # ============================================================
    # DATAFRAME
    # ============================================================

    df_raw = pd.DataFrame(all_rows)

    if df_raw.empty:
        raise RuntimeError("❌ Nessun dato scaricato")

    # Rimuove duplicati veri
    df_raw = df_raw.drop_duplicates(
        subset=["stagione", "squadra", "player", "voto_gds"]
    )

    print(f"\n✅ Scraping completato: {len(df_raw)} righe")

    df_raw.to_csv(OUTPUT_RAW, index=False)
    print(f"💾 Salvato {OUTPUT_RAW}")

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    print("\n🧠 Costruzione feature")

    df_features = build_match_level_features(df_raw)
    df_features.to_csv(OUTPUT_FEAT, index=False)

    print(f"💾 Salvato {OUTPUT_FEAT}")
    print("\n🏁 PIPELINE COMPLETATA")
