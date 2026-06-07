"""
World Cup Countries - Match Stats Scraper (2020-2026)
======================================================
Sources:
  1. StatsBomb Open Data (GitHub) — event-level data → shots on target, xG, corners, goals
     Competitions: Euro 2020 · WC 2022 · AFCON 2023 · Copa América 2024 · Euro 2024
  2. martj42/international_results (GitHub) — match-level results for ALL competitions
     Used for qualification matches, Nations League, etc. (goals only, no advanced stats)

OUTPUT:
  Writes into the 'historial_selecciones_ml' table in the provided SQLite database.
  Schema matches historial_multiliga_ml:
    Date, Torneo, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HST, AST, HC, AC, xG_home, xG_away

USAGE:
  python wc_scraper.py --db /path/to/database_partidos.db

NOTES:
  - Filters to the 48 WC 2026 qualified countries only
  - Excludes Friendlies
  - Skips rows already present in the DB (deduplication by Date + HomeTeam + AwayTeam)
  - Runtime: ~5-10 min (depends on connection)
"""

import argparse
import io
import json
import sqlite3
import time
from collections import defaultdict

import pandas as pd
import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATE_FROM = "2020-01-01"
DATE_TO   = "2026-06-06"

TARGET_TABLE = "historial_selecciones_ml"

# StatsBomb team name → martj42 name (where they differ)
SB_TO_MARTJ = {
    "Cape Verde Islands": "Cape Verde",
    "Congo DR":           "DR Congo",
    "Côte d'Ivoire":      "Ivory Coast",
    "Korea Republic":     "South Korea",
}

# StatsBomb competitions (comp_id, season_id, Torneo label)
STATSBOMB_COMPS = [
    (55,   43,  "UEFA Euro 2020"),
    (43,  106,  "FIFA World Cup 2022"),
    (1267, 107, "African Cup of Nations 2023"),
    (223, 282,  "Copa America 2024"),
    (55,  282,  "UEFA Euro 2024"),
]

# Tournaments to exclude from martj42 (non-competitive)
FRIENDLY_KEYWORDS = [
    "Friendly", "friendly", "International Champions Cup",
    "FIFA Series", "CONIFA", "Island Games", "MSG Prime Minister's Cup",
    "King's Cup", "Baltic Cup", "CAFA Nations Cup", "Intercontinental Cup",
    "Pacific Games", "Asian Games",
]

SB_BASE    = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# ─── WC 2026 TEAMS (48) ──────────────────────────────────────────────────────

WC2026_TEAMS = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
    "Colombia", "Croatia", "Curaçao", "Czech Republic", "DR Congo",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
    "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
        except Exception as e:
            print(f"    [warn] {e}, retrying ({attempt+1}/{retries})...")
            time.sleep(2)
    return None

def sb_name(name):
    return SB_TO_MARTJ.get(name, name)

def is_wc_team(name):
    return sb_name(name) in WC2026_TEAMS

def is_friendly(tournament):
    return any(kw in tournament for kw in FRIENDLY_KEYWORDS)

def ftr(hg, ag):
    if hg is None or ag is None:
        return None
    if hg > ag:   return "H"
    elif hg < ag: return "A"
    else:         return "D"

# ─── DB HELPERS ──────────────────────────────────────────────────────────────

def load_existing_keys(conn):
    """Return a set of (Date, HomeTeam, AwayTeam) already in the table."""
    cur = conn.cursor()
    cur.execute(f"SELECT Date, HomeTeam, AwayTeam FROM {TARGET_TABLE}")
    return set(cur.fetchall())

def ensure_table(conn):
    """Create the table if it doesn't exist yet."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            Date      TEXT,
            Torneo    TEXT,
            HomeTeam  TEXT,
            AwayTeam  TEXT,
            FTHG      INTEGER,
            FTAG      INTEGER,
            FTR       TEXT,
            HST       REAL,
            AST       REAL,
            HC        REAL,
            AC        REAL,
            xG_home   REAL,
            xG_away   REAL
        )
    """)
    conn.commit()

def insert_rows(conn, rows, existing_keys):
    """Insert rows that aren't already in the DB."""
    inserted = 0
    skipped  = 0
    cur = conn.cursor()
    for r in rows:
        key = (r["Date"], r["HomeTeam"], r["AwayTeam"])
        if key in existing_keys:
            skipped += 1
            continue
        cur.execute(f"""
            INSERT INTO {TARGET_TABLE}
              (Date, Torneo, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
               HST, AST, HC, AC, xG_home, xG_away)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["Date"], r["Torneo"], r["HomeTeam"], r["AwayTeam"],
            r["FTHG"], r["FTAG"], r["FTR"],
            r["HST"], r["AST"], r["HC"], r["AC"],
            r["xG_home"], r["xG_away"],
        ))
        existing_keys.add(key)
        inserted += 1
    conn.commit()
    return inserted, skipped

# ─── STATSBOMB SCRAPER ────────────────────────────────────────────────────────

def parse_events(events):
    """Aggregate per-team stats from a StatsBomb event list."""
    stats = defaultdict(lambda: {
        "goals": 0, "shots_on_target": 0, "xg": 0.0, "corners": 0
    })
    for e in events:
        etype = e["type"]["name"]
        team  = e.get("team", {}).get("name", "")

        if etype == "Shot":
            shot    = e.get("shot", {})
            outcome = shot.get("outcome", {}).get("name", "")
            stats[team]["xg"] += shot.get("statsbomb_xg", 0.0)
            if outcome == "Goal":
                stats[team]["goals"] += 1
            if outcome in ("Goal", "Saved", "Saved to Post"):
                stats[team]["shots_on_target"] += 1

        elif etype == "Pass":
            if e.get("pass", {}).get("type", {}).get("name") == "Corner":
                stats[team]["corners"] += 1

    return dict(stats)


def scrape_statsbomb(existing_keys):
    rows = []

    for comp_id, season_id, torneo in STATSBOMB_COMPS:
        print(f"\n{'─'*60}")
        print(f"  {torneo}  (comp={comp_id}, season={season_id})")
        print(f"{'─'*60}")

        matches = get_json(f"{SB_BASE}/matches/{comp_id}/{season_id}.json")
        if not matches:
            print("  [skip] could not fetch match list")
            continue

        filtered = [
            m for m in matches
            if DATE_FROM <= m.get("match_date","") <= DATE_TO
            and (is_wc_team(m["home_team"]["home_team_name"])
                 or is_wc_team(m["away_team"]["away_team_name"]))
        ]
        print(f"  Matches in range with WC teams: {len(filtered)}")

        for i, m in enumerate(filtered, 1):
            match_id = m["match_id"]
            date     = m["match_date"]
            ht_sb    = m["home_team"]["home_team_name"]
            at_sb    = m["away_team"]["away_team_name"]
            ht       = sb_name(ht_sb)
            at       = sb_name(at_sb)
            hg       = m.get("home_score")
            ag       = m.get("away_score")

            print(f"  [{i:3d}/{len(filtered)}] {date}  {ht} vs {at} ... ", end="", flush=True)

            # Skip if already in DB
            if (date, ht, at) in existing_keys:
                print("already in DB, skipping")
                continue

            events = get_json(f"{SB_BASE}/events/{match_id}.json")
            if not events:
                print("NO EVENTS")
                continue

            ts = parse_events(events)

            def s(team, key):
                return round(ts.get(team, {}).get(key, 0), 4)

            rows.append({
                "Date":     date,
                "Torneo":   torneo,
                "HomeTeam": ht,
                "AwayTeam": at,
                "FTHG":     hg,
                "FTAG":     ag,
                "FTR":      ftr(hg, ag),
                "HST":      s(ht_sb, "shots_on_target"),
                "AST":      s(at_sb, "shots_on_target"),
                "HC":       s(ht_sb, "corners"),
                "AC":       s(at_sb, "corners"),
                "xG_home":  s(ht_sb, "xg"),
                "xG_away":  s(at_sb, "xg"),
            })
            print(f"✓  {hg}-{ag}  xG {s(ht_sb,'xg'):.2f}-{s(at_sb,'xg'):.2f}  "
                  f"SoT {s(ht_sb,'shots_on_target')}-{s(at_sb,'shots_on_target')}  "
                  f"Corners {s(ht_sb,'corners')}-{s(at_sb,'corners')}")
            time.sleep(0.4)

    return rows


# ─── MARTJ42 SCRAPER ─────────────────────────────────────────────────────────

def scrape_results(existing_keys):
    print(f"\n{'─'*60}")
    print("  martj42/international_results (goals only)")
    print(f"{'─'*60}")

    r = requests.get(RESULTS_URL, timeout=30)
    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])

    mask = (
        (df["date"] >= DATE_FROM) &
        (df["date"] <= DATE_TO) &
        (df["home_team"].isin(WC2026_TEAMS) | df["away_team"].isin(WC2026_TEAMS)) &
        (~df["tournament"].apply(is_friendly))
    )
    df = df[mask].copy()
    print(f"  Non-friendly WC-team matches: {len(df)}")

    rows = []
    for _, row in df.iterrows():
        date = str(row["date"].date())
        ht   = row["home_team"]
        at   = row["away_team"]
        hg   = int(row["home_score"])
        ag   = int(row["away_score"])

        if (date, ht, at) in existing_keys:
            continue   # already in DB (from StatsBomb pass or prior run)

        rows.append({
            "Date":     date,
            "Torneo":   row["tournament"],
            "HomeTeam": ht,
            "AwayTeam": at,
            "FTHG":     hg,
            "FTAG":     ag,
            "FTR":      ftr(hg, ag),
            "HST":      None,
            "AST":      None,
            "HC":       None,
            "AC":       None,
            "xG_home":  None,
            "xG_away":  None,
        })
    return rows


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database_partidos.db",
                        help="Path to the SQLite database file")
    args = parser.parse_args()

    print("=" * 60)
    print(" World Cup Countries — Match Stats Scraper")
    print(f" DB:     {args.db}")
    print(f" Table:  {TARGET_TABLE}")
    print(f" Period: {DATE_FROM} → {DATE_TO}")
    print(f" Teams:  {len(WC2026_TEAMS)} WC 2026 nations")
    print("=" * 60)

    conn = sqlite3.connect(args.db)
    ensure_table(conn)
    existing_keys = load_existing_keys(conn)
    print(f"\n  Rows already in {TARGET_TABLE}: {len(existing_keys)}")

    # 1. Full stats from StatsBomb
    sb_rows = scrape_statsbomb(existing_keys)
    sb_inserted, sb_skipped = insert_rows(conn, sb_rows, existing_keys)
    print(f"\n  StatsBomb: inserted {sb_inserted}, skipped {sb_skipped} (already in DB)")

    # 2. Results-only from martj42 (deduplicates against StatsBomb automatically)
    r_rows = scrape_results(existing_keys)
    r_inserted, r_skipped = insert_rows(conn, r_rows, existing_keys)
    print(f"  martj42:   inserted {r_inserted}, skipped {r_skipped} (already in DB)")

    conn.close()

    total = sb_inserted + r_inserted
    print(f"\n{'='*60}")
    print(f" DONE — {total} new rows written to '{TARGET_TABLE}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
