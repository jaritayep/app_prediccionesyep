"""
OddsPapi — World Cup Historical Scraper
=======================================
Recovers Pinnacle closing-line odds for World Cup matches
on June 17–23, 2026 using the OddsPapi API.

Output: one CSV per day  →  odds_data/worldcup_YYYYMMDD.csv

Markets captured (whatever Pinnacle had available):
  • 1X2  (Full Time Result)
  • Asian Handicap
  • Over/Under Goals  (all lines)
  • Total Corners
  • Shots on Target

HOW IT WORKS:
  Step 1 — GET /v4/fixtures   → find every World Cup fixture per day
  Step 2 — GET /v4/markets    → build a marketId→name/type lookup (once)
  Step 3 — GET /v4/historical-odds?fixtureId=...&bookmakers=pinnacle
           → get the FULL odds history; we keep only the LAST active
             price per outcome (= closest thing to the closing line
             that was available before kickoff)
"""

import requests
import time
import os
import json
from datetime import date, timedelta, datetime, timezone
from collections import defaultdict
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = "361710a7-f618-47da-933f-ee2144b92b42"
BASE_URL   = "https://api.oddspapi.io/v4"
BOOKMAKER  = "pinnacle"

FECHA_INICIO = date(2026, 6, 17)   # inclusive
FECHA_FIN    = date(2026, 6, 23)   # inclusive

# OddsPapi market IDs we care about (from their /v4/markets catalog)
# marketType values:  "1x2", "asian_handicap", "totals", "corners", "shots"
# We use keyword matching on marketName as a fallback in case IDs shift.
MARKET_KEYWORDS = {
    "1x2":           ["full time result", "match result", "1x2"],
    "handicap":      ["asian handicap", "handicap"],
    "total_goals":   ["over under full time", "total goals", "over/under"],
    "corners":       ["corner", "corners", "rincón"],
    "shots":         ["shots on target", "shot", "tiro"],
}

# API rate limits (from docs)
DELAY_FIXTURES = 2.1   # /fixtures cooldown: 2000ms
DELAY_HIST     = 5.1   # /historical-odds cooldown: 5000ms

WC_KEYWORDS = ["world cup", "copa del mundo", "mundial", "fifa world", "fifa", "world"]
# ↑ broad on purpose — the debug log will show the exact name OddsPapi uses;
#   tighten after confirming (e.g. change "world" to "world cup 2026")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def get(endpoint: str, params: dict = None, retries: int = 3) -> dict | list | None:
    url = f"{BASE_URL}{endpoint}"
    # API key must be a query parameter (not a header)
    all_params = {"apiKey": API_KEY}
    if params:
        all_params.update(params)
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=all_params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  ⚠ Rate limit (429). Waiting {wait}s …")
                time.sleep(wait)
            elif resp.status_code == 404:
                return None
            else:
                print(f"  ✗ HTTP {resp.status_code} on {endpoint}: {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Network error ({attempt+1}/{retries}): {e}")
            time.sleep(3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CATALOGUE  (fetched once, reused everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_mercados() -> dict:
    """Returns {marketId: {"name": str, "type": str, "handicap": float, "outcomes": {id: name}}}"""
    print("  → Loading market catalogue …")
    data = get("/markets") or []
    catalogue = {}
    for m in data:
        mid = m.get("marketId")
        if mid is None:
            continue
        catalogue[mid] = {
            "name":     m.get("marketName", "").lower(),
            "type":     m.get("marketType", "").lower(),
            "handicap": m.get("handicap", 0),
            "period":   m.get("period", ""),
            "outcomes": {o["outcomeId"]: o["outcomeName"] for o in m.get("outcomes", [])},
        }
    print(f"  ✅ {len(catalogue)} markets loaded.")
    return catalogue


def clasificar_mercado(market_info: dict) -> str | None:
    """Returns one of the MARKET_KEYWORDS keys, or None if not relevant."""
    name = market_info["name"]
    mtype = market_info["type"]
    for categoria, keywords in MARKET_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return categoria
    # fallback by marketType
    if mtype == "1x2":
        return "1x2"
    if mtype == "asian_handicap":
        return "handicap"
    if mtype == "totals" and "corner" not in market_info["name"]:
        return "total_goals"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def obtener_fixtures_dia(dia: date) -> list:
    """Returns World Cup fixtures on `dia` (UTC)."""
    desde = f"{dia}T00:00:00Z"
    hasta = f"{dia}T23:59:59Z"
    # Omit hasOdds — for finished games the API may return False even when
    # historical odds exist. sportId=10 + 24h window is sufficient.
    data = get("/fixtures", params={
        "sportId": 10,
        "from":    desde,
        "to":      hasta,
    }) or []

    # Debug: show every tournament name returned so we can tune filtering
    torneos = {}
    for f in data:
        t = f.get("tournamentName", "UNKNOWN")
        torneos[t] = torneos.get(t, 0) + 1
    if torneos:
        print(f"  \u2139 Tournaments on {dia}:")
        for t, n in sorted(torneos.items()):
            print(f"       {n}x  {t}")
    else:
        print(f"  \u2139 No fixtures at all returned for {dia}.")

    wc = []
    for f in data:
        nombre = f.get("tournamentName", "").lower()
        if any(kw in nombre for kw in WC_KEYWORDS):
            wc.append(f)
    return wc


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL ODDS PARSER
# ─────────────────────────────────────────────────────────────────────────────

def ultima_cuota_activa(snapshots: list) -> float | None:
    """
    From a list of price snapshots (newest first per API), returns the price
    of the most recent entry where active=True (i.e. the last offered price).
    Falls back to the first snapshot regardless of active flag if none active.
    """
    for snap in snapshots:
        if snap.get("active"):
            return round(float(snap["price"]), 4)
    if snapshots:
        return round(float(snapshots[0]["price"]), 4)
    return None


def parsear_historical_odds(data: dict, catalogue: dict) -> dict:
    """
    Parses the /historical-odds response into a flat dict ready for a CSV row.

    data structure:
      data["bookmakers"]["pinnacle"]["markets"][marketId]["outcomes"][outcomeId]
           ["players"]["0"] = [list of snapshots newest-first]
    """
    row = {}
    bk = data.get("bookmakers", {}).get(BOOKMAKER, {})
    markets = bk.get("markets", {})

    for mid_str, market_data in markets.items():
        mid = int(mid_str)
        info = catalogue.get(mid)
        if info is None:
            continue

        categoria = clasificar_mercado(info)
        if categoria is None:
            continue

        outcomes = market_data.get("outcomes", {})
        for oid_str, outcome_data in outcomes.items():
            oid = int(oid_str)
            outcome_name = info["outcomes"].get(oid, f"outcome_{oid}")
            snapshots = outcome_data.get("players", {}).get("0", [])
            precio = ultima_cuota_activa(snapshots)
            if precio is None:
                continue

            hdp = info.get("handicap", "")
            hdp_str = f"_{hdp}" if hdp not in (0, None, "") else ""

            if categoria == "1x2":
                col = f"1x2_{outcome_name.lower().replace(' ','_')}"
            elif categoria == "handicap":
                col = f"hdp_{outcome_name.lower()}{hdp_str}"
            elif categoria == "total_goals":
                col = f"goles_{hdp_str.lstrip('_')}_{outcome_name.lower()}"
            elif categoria == "corners":
                col = f"corners_{hdp_str.lstrip('_')}_{outcome_name.lower()}"
            elif categoria == "shots":
                col = f"shots_{hdp_str.lstrip('_')}_{outcome_name.lower()}"
            else:
                continue

            # Keep the last (most recent) value if duplicate col appears
            # (can happen with alt lines sharing same handicap)
            if col not in row:
                row[col] = precio

    return row


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  🌍 ODDSPAPI — WORLD CUP HISTORICAL SCRAPER")
    print(f"  📅 Período: {FECHA_INICIO} → {FECHA_FIN}")
    print("=" * 65)

    catalogue = cargar_mercados()
    time.sleep(1)

    os.makedirs("odds_data", exist_ok=True)

    dia = FECHA_INICIO
    while dia <= FECHA_FIN:
        print(f"\n{'─'*65}")
        print(f"  📅 {dia}")
        print(f"{'─'*65}")

        fixtures = obtener_fixtures_dia(dia)
        time.sleep(DELAY_FIXTURES)

        if not fixtures:
            print(f"  ⚠ No World Cup fixtures found for {dia}.")
            dia += timedelta(days=1)
            continue

        print(f"  → {len(fixtures)} fixture(s) found.")
        filas = []

        for fx in fixtures:
            fid   = fx["fixtureId"]
            home  = fx["participant1Name"]
            away  = fx["participant2Name"]
            liga  = fx.get("tournamentName", "World Cup")
            start = fx.get("startTime", "")
            print(f"\n  ⚽ {home} vs {away}  [{fid}]")

            hist = get("/historical-odds", params={
                "fixtureId":  fid,
                "bookmakers": BOOKMAKER,
            })
            time.sleep(DELAY_HIST)

            if not hist:
                print(f"     ⚠ No historical odds returned.")
                continue

            odds_row = parsear_historical_odds(hist, catalogue)

            if not odds_row:
                print(f"     ⚠ Odds parsed but empty — market IDs may not match catalogue.")
                continue

            # Convert UTC startTime to local
            try:
                dt_utc   = datetime.fromisoformat(start.replace("Z", "+00:00"))
                dt_local = dt_utc.astimezone()
                inicio_local = dt_local.strftime("%Y-%m-%d %H:%M")
            except Exception:
                inicio_local = start

            fila = {
                "liga":         liga,
                "home":         home,
                "away":         away,
                "inicio_utc":   start,
                "inicio_local": inicio_local,
                **odds_row,
            }
            filas.append(fila)
            mercados_encontrados = [k for k in odds_row if not k.startswith(("liga","home","away","inicio"))]
            print(f"     ✅ {len(mercados_encontrados)} market column(s) captured.")

        if not filas:
            print(f"\n  ⚠ No data rows for {dia}. Skipping file.")
            dia += timedelta(days=1)
            continue

        fecha_str = dia.strftime("%Y%m%d")
        csv_path  = f"odds_data/worldcup_{fecha_str}.csv"

        df = pd.DataFrame(filas)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"\n  💾 {len(filas)} row(s) → {csv_path}")
        preview_cols = ["home", "away", "inicio_local"]
        for c in ["1x2_1", "1x2_x", "1x2_2", "1x2_home", "1x2_draw", "1x2_away"]:
            if c in df.columns:
                preview_cols.append(c)
        print(df[[c for c in preview_cols if c in df.columns]].to_string(index=False))

        dia += timedelta(days=1)

    print(f"\n{'='*65}")
    print("✅ Done! Files saved in odds_data/")
    print("=" * 65)


if __name__ == "__main__":
    main()
