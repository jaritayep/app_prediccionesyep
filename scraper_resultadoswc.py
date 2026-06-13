"""
scraper_mundial.py
──────────────────
Scrapea resultados del Mundial 2026 desde dos fuentes de GitHub
(ambas accesibles sin bloqueo) y los registra en database_partidos.db:

  fixture_mundial          → se actualiza con el marcador final
  historial_selecciones_ml → tabla que usa el modelo ML

FUENTES:
  Fuente 1 (scores + goles):
    github.com/openfootball/world-cup.json → worldcup.json
    → Tiene: score FT, goles marcadores, grupo, ronda, estadio

  Fuente 2 (stats avanzadas: shots, corners, cards):
    github.com/martj42/international_results → goalscorers.csv + shootouts.csv
    → Tiene: shots, corners, cards cuando están disponibles
    → Como fallback: estimación desde historial de la DB

USO:
  python scraper_mundial.py                     # procesa ayer
  python scraper_mundial.py --fecha 2026-06-15  # fecha específica
  python scraper_mundial.py --dias 3            # últimos 3 días
  python scraper_mundial.py --dry-run           # simula sin escribir
"""

import sqlite3
import argparse
import sys
import time
import re
import logging
import csv
import io
import json
from datetime import date, timedelta, datetime
from typing import Optional

import requests
from thefuzz import process as fuzz_process

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper_mundial")

# ─── Configuración ─────────────────────────────────────────────────────────
DB_PATH = "database_partidos.db"

# GitHub RAW — sin bloqueo de IP
URL_OPENFOOTBALL = (
    "https://raw.githubusercontent.com/openfootball/"
    "world-cup.json/master/2026/worldcup.json"
)
URL_MARTJ42_RESULTS = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
URL_MARTJ42_GOALSCORERS = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/goalscorers.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
}

# Delay entre peticiones HTTP
REQUEST_DELAY = 1.5

# Normalización de nombres de equipos → estándar interno del modelo
ALIAS: dict[str, str] = {
    "USA":                    "United States",
    "South Korea":            "Korea Republic",
    "Czechia":                "Czech Republic",
    "Czech Republic":         "Czech Republic",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Bosnia":                 "Bosnia and Herzegovina",
    "Cape Verde Islands":     "Cape Verde",
    "Congo DR":               "DR Congo",
    "DRC":                    "DR Congo",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cote d'Ivoire":          "Ivory Coast",
    "Türkiye":                "Turkey",
    "IR Iran":                "Iran",
    "North Macedonia":        "North Macedonia",
    "FYR Macedonia":          "North Macedonia",
    "Curaçao":                "Curaçao",
    "Curacao":                "Curaçao",
}


# ─── Helpers genéricos ─────────────────────────────────────────────────────

def normalize(name: str) -> str:
    return ALIAS.get(name.strip(), name.strip()) if name else name


def safe_int(val, default: int = 0) -> int:
    try:
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return default


def fuzzy_match(query: str, pool: list[str], threshold: int = 78) -> Optional[str]:
    if not pool or not query:
        return None
    result = fuzz_process.extractOne(query, pool)
    if result and result[1] >= threshold:
        return result[0]
    return None


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_url(session: requests.Session, url: str) -> Optional[str]:
    """Descarga una URL con retry y delay."""
    for attempt in range(3):
        try:
            time.sleep(REQUEST_DELAY if attempt == 0 else REQUEST_DELAY * 2)
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            log.warning(f"Intento {attempt+1}/3 fallido para {url}: {e}")
    return None


# ─── Base de datos ─────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historial_selecciones_ml (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            Date     TEXT NOT NULL,
            HomeTeam TEXT NOT NULL,
            AwayTeam TEXT NOT NULL,
            FTHG     INTEGER DEFAULT 0,
            FTAG     INTEGER DEFAULT 0,
            FTR      TEXT    DEFAULT 'D',
            HS       INTEGER DEFAULT 0,
            AS_      INTEGER DEFAULT 0,
            HST      INTEGER DEFAULT 0,
            AST      INTEGER DEFAULT 0,
            HC       INTEGER DEFAULT 0,
            AC       INTEGER DEFAULT 0,
            HY       INTEGER DEFAULT 0,
            AY       INTEGER DEFAULT 0,
            HR       INTEGER DEFAULT 0,
            AR       INTEGER DEFAULT 0,
            xG_home  REAL    DEFAULT 0.0,
            xG_away  REAL    DEFAULT 0.0,
            Stage    TEXT    DEFAULT '',
            UNIQUE(Date, HomeTeam, AwayTeam)
        )
    """)
    conn.commit()
    return conn


def already_exists(conn, date_str: str, home: str, away: str) -> bool:
    cur = conn.execute(
        "SELECT COUNT(*) FROM historial_selecciones_ml "
        "WHERE Date=? AND HomeTeam=? AND AwayTeam=?",
        (date_str, home, away),
    )
    return cur.fetchone()[0] > 0


def get_fixture_row(conn, home: str, away: str, date_str: str) -> Optional[dict]:
    """Busca el fixture en fixture_mundial con fuzzy matching."""
    try:
        cur = conn.execute(
            "SELECT * FROM fixture_mundial WHERE Date LIKE ?",
            (f"{date_str}%",)
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        return None

    if not rows:
        return None

    homes = [r["HomeTeam"] for r in rows]
    best = fuzzy_match(home, homes)
    if not best:
        return None

    match_row = next((r for r in rows if r["HomeTeam"] == best), None)
    if not match_row:
        return None

    # También verificar away
    if fuzzy_match(away, [match_row["AwayTeam"]], 70):
        return match_row
    return None


def get_team_avg_stats(conn, team: str) -> dict:
    """
    Devuelve promedios históricos de shots/corners/cards de un equipo
    desde historial_selecciones_ml, para usar como fallback cuando
    no hay stats detalladas.
    """
    try:
        cur = conn.execute("""
            SELECT
                AVG(CASE WHEN HomeTeam=? THEN HST ELSE AST END) as avg_sot,
                AVG(CASE WHEN HomeTeam=? THEN HS  ELSE AS_ END) as avg_sh,
                AVG(CASE WHEN HomeTeam=? THEN HC  ELSE AC  END) as avg_c,
                AVG(CASE WHEN HomeTeam=? THEN HY  ELSE AY  END) as avg_y
            FROM historial_selecciones_ml
            WHERE HomeTeam=? OR AwayTeam=?
            ORDER BY Date DESC LIMIT 10
        """, (team, team, team, team, team, team))
        row = cur.fetchone()
        if row:
            return {
                "sot": safe_float(row["avg_sot"], 4.0),
                "sh":  safe_float(row["avg_sh"],  10.0),
                "c":   safe_float(row["avg_c"],   5.0),
                "y":   safe_float(row["avg_y"],   1.5),
            }
    except Exception:
        pass
    return {"sot": 4.0, "sh": 10.0, "c": 5.0, "y": 1.5}


def insert_result(conn, row: dict, dry_run: bool = False) -> bool:
    src = "ℹ️ stats estimadas" if row.get("_estimated") else "✅ stats reales"
    log.info(
        f"  {'[DRY-RUN] ' if dry_run else ''}INSERTAR: "
        f"{row['Date']} | {row['HomeTeam']} {row['FTHG']}-{row['FTAG']} {row['AwayTeam']} | "
        f"Shots {row['HST']}/{row['AST']} | Corners {row['HC']}/{row['AC']} | "
        f"Yellow {row['HY']}/{row['AY']} | {src}"
    )
    if dry_run:
        return True

    try:
        conn.execute("""
            INSERT INTO historial_selecciones_ml
                (Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,
                 HS,AS_,HST,AST,HC,AC,HY,AY,HR,AR,
                 xG_home,xG_away,Stage)
            VALUES
                (:Date,:HomeTeam,:AwayTeam,:FTHG,:FTAG,:FTR,
                 :HS,:AS_,:HST,:AST,:HC,:AC,:HY,:AY,:HR,:AR,
                 :xG_home,:xG_away,:Stage)
            ON CONFLICT(Date,HomeTeam,AwayTeam) DO UPDATE SET
                FTHG=excluded.FTHG, FTAG=excluded.FTAG, FTR=excluded.FTR,
                HS=excluded.HS,   AS_=excluded.AS_,
                HST=excluded.HST, AST=excluded.AST,
                HC=excluded.HC,   AC=excluded.AC,
                HY=excluded.HY,   AY=excluded.AY,
                HR=excluded.HR,   AR=excluded.AR,
                xG_home=excluded.xG_home, xG_away=excluded.xG_away,
                Stage=excluded.Stage
        """, row)
        conn.commit()
        return True
    except Exception as e:
        log.error(f"  Error DB: {e}")
        return False


def update_fixture_score(conn, home: str, away: str, date_str: str,
                         hg: int, ag: int, dry_run: bool = False):
    if dry_run:
        return
    try:
        # Intenta actualizar si fixture_mundial tiene columnas de score
        conn.execute("""
            UPDATE fixture_mundial SET FTHG=?, FTAG=?
            WHERE HomeTeam=? AND AwayTeam=? AND Date LIKE ?
        """, (hg, ag, home, away, f"{date_str}%"))
        conn.commit()
    except Exception:
        pass  # No es crítico si la tabla no tiene esas columnas


# ─── FUENTE 1: openfootball/world-cup.json ─────────────────────────────────

def fetch_openfootball(session: requests.Session, target_date: str) -> list[dict]:
    """
    Descarga worldcup.json y extrae partidos del target_date.
    Devuelve lista con: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
    goals_home (lista), goals_away (lista), Stage, venue.
    """
    log.info("📡 Fuente 1: openfootball/world-cup.json ...")
    raw = fetch_url(session, URL_OPENFOOTBALL)
    if not raw:
        log.warning("  openfootball no disponible.")
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"  JSON inválido: {e}")
        return []

    results = []
    for match in data.get("matches", []):
        if match.get("date") != target_date:
            continue

        score = match.get("score", {})
        ft = score.get("ft")
        if not ft or len(ft) != 2:
            log.debug(f"  Sin resultado: {match.get('team1')} vs {match.get('team2')}")
            continue

        hg, ag = safe_int(ft[0]), safe_int(ft[1])
        home = normalize(match.get("team1", ""))
        away = normalize(match.get("team2", ""))

        if not home or not away:
            continue

        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        stage = match.get("round", match.get("group", "World Cup 2026"))

        results.append({
            "Date":        target_date,
            "HomeTeam":    home,
            "AwayTeam":    away,
            "FTHG":        hg,
            "FTAG":        ag,
            "FTR":         ftr,
            "Stage":       stage,
            "_goals_home": len(match.get("goals1", [])),
            "_goals_away": len(match.get("goals2", [])),
            "_venue":      match.get("ground", ""),
        })
        log.info(f"  ✅ {home} {hg}-{ag} {away}  [{stage}]")

    if not results:
        log.info(f"  No hay partidos con resultado para {target_date}")
    return results


# ─── FUENTE 2: martj42/international_results ───────────────────────────────

def fetch_martj42_stats(session: requests.Session, target_date: str) -> dict:
    """
    Descarga results.csv de martj42 y devuelve un dict:
    { (home_norm, away_norm): {shots, corners, cards, ...} }

    La fuente no tiene shots/corners directamente, pero sí tiene
    datos complementarios que podemos usar para validar scores.

    Nota: el CSV de martj42 tiene columnas básicas (score) pero
    NO tiene shots/corners — esas stats se estiman desde historial.
    """
    log.info("📡 Fuente 2: martj42/international_results ...")
    raw = fetch_url(session, URL_MARTJ42_RESULTS)
    if not raw:
        log.warning("  martj42 no disponible.")
        return {}

    stats_map = {}
    try:
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            if row.get("date") != target_date:
                continue
            if "FIFA World Cup" not in row.get("tournament", ""):
                continue

            home = normalize(row.get("home_team", ""))
            away = normalize(row.get("away_team", ""))
            key = (home, away)

            try:
                hs = safe_int(row.get("home_score"))
                as_ = safe_int(row.get("away_score"))
            except Exception:
                continue

            stats_map[key] = {
                "FTHG": hs,
                "FTAG": as_,
                "tournament": row.get("tournament", ""),
                "neutral": row.get("neutral", "FALSE") == "TRUE",
            }
            log.info(f"  ✅ Confirmado: {home} {hs}-{as_} {away}")
    except Exception as e:
        log.warning(f"  Error parseando martj42 CSV: {e}")

    return stats_map


# ─── Enriquecer con stats estimadas si no hay datos reales ─────────────────

def enrich_with_estimated_stats(result: dict, conn: sqlite3.Connection) -> dict:
    """
    Si el partido no tiene stats reales (shots/corners/cards),
    estima valores basados en el historial de ambos equipos en la DB.
    Marca el registro con _estimated=True para logging.
    """
    home = result["HomeTeam"]
    away = result["AwayTeam"]

    h_stats = get_team_avg_stats(conn, home)
    a_stats = get_team_avg_stats(conn, away)

    result["HS"]  = result.get("HS")  or round(h_stats["sh"])
    result["AS_"] = result.get("AS_") or round(a_stats["sh"])
    result["HST"] = result.get("HST") or round(h_stats["sot"])
    result["AST"] = result.get("AST") or round(a_stats["sot"])
    result["HC"]  = result.get("HC")  or round(h_stats["c"])
    result["AC"]  = result.get("AC")  or round(a_stats["c"])
    result["HY"]  = result.get("HY")  or round(h_stats["y"])
    result["AY"]  = result.get("AY")  or round(a_stats["y"])
    result["HR"]  = result.get("HR", 0)
    result["AR"]  = result.get("AR", 0)
    result["xG_home"] = result.get("xG_home", 0.0)
    result["xG_away"] = result.get("xG_away", 0.0)
    result["_estimated"] = True

    return result


# ─── Pipeline principal ────────────────────────────────────────────────────

def run(target_date: str, dry_run: bool = False):
    log.info("=" * 62)
    log.info(f"🌍 Scraper Mundial 2026 — {target_date}")
    if dry_run:
        log.info("⚠️  MODO DRY-RUN — no se escribirá nada en la DB")
    log.info("=" * 62)

    conn    = get_conn()
    session = get_session()

    # 1. Scores desde openfootball (fuente primaria)
    of_results = fetch_openfootball(session, target_date)

    # 2. Confirmación/stats desde martj42 (fuente secundaria)
    m42_stats = fetch_martj42_stats(session, target_date)

    if not of_results and not m42_stats:
        log.warning(f"Ninguna fuente tiene resultados para {target_date}.")
        log.info("  Puede que los partidos aún no se hayan jugado, "
                 "o que el torneo no haya empezado.")
        conn.close()
        return

    # 3. Construir lista unificada priorizando openfootball para scores
    all_keys_seen: set[tuple] = set()
    final_results: list[dict] = []

    # 3a. Desde openfootball
    for r in of_results:
        key = (r["HomeTeam"], r["AwayTeam"])
        all_keys_seen.add(key)

        # Buscar nombre canónico en fixture_mundial para consistencia con el modelo
        fixture = get_fixture_row(conn, r["HomeTeam"], r["AwayTeam"], target_date)
        if fixture:
            r["HomeTeam"] = fixture["HomeTeam"]
            r["AwayTeam"] = fixture["AwayTeam"]
            if "Grupo" in fixture and fixture["Grupo"]:
                r["Stage"] = str(fixture["Grupo"])

        # Enriquecer con stats estimadas (shots/corners/cards)
        r = enrich_with_estimated_stats(r, conn)
        # Limpiar keys internas
        for k in ["_goals_home", "_goals_away", "_venue"]:
            r.pop(k, None)

        final_results.append(r)

    # 3b. Partidos en martj42 que no estaban en openfootball
    for (home, away), m42 in m42_stats.items():
        if (home, away) in all_keys_seen:
            continue  # Ya procesado
        # Construir entrada básica
        hg = m42["FTHG"]
        ag = m42["FTAG"]
        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        r = {
            "Date": target_date, "HomeTeam": home, "AwayTeam": away,
            "FTHG": hg, "FTAG": ag, "FTR": ftr,
            "Stage": "World Cup 2026",
        }
        fixture = get_fixture_row(conn, home, away, target_date)
        if fixture:
            r["HomeTeam"] = fixture["HomeTeam"]
            r["AwayTeam"] = fixture["AwayTeam"]
        r = enrich_with_estimated_stats(r, conn)
        final_results.append(r)
        log.info(f"  ➕ Solo en martj42: {home} {hg}-{ag} {away}")

    # 4. Insertar en DB
    inserted = 0
    skipped  = 0
    for result in final_results:
        if not dry_run and already_exists(
            conn, result["Date"], result["HomeTeam"], result["AwayTeam"]
        ):
            log.info(f"  ⏭️  Ya existe: {result['HomeTeam']} vs {result['AwayTeam']}")
            skipped += 1
            continue

        ok = insert_result(conn, result, dry_run)
        if ok:
            inserted += 1
            update_fixture_score(
                conn,
                result["HomeTeam"], result["AwayTeam"], result["Date"],
                result["FTHG"], result["FTAG"],
                dry_run,
            )

    # 5. Resumen
    log.info("─" * 62)
    log.info(f"Resumen {target_date}:")
    log.info(f"  Encontrados : {len(final_results)}")
    log.info(f"  Insertados  : {inserted}")
    log.info(f"  Ya existían : {skipped}")
    if any(r.get("_estimated") for r in final_results):
        log.info("  ⚠️  Algunos partidos usan stats estimadas desde historial "
                 "(shots/corners/cards no disponibles en fuentes públicas aún).")
    log.info("─" * 62)

    conn.close()


# ─── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraper Mundial 2026 → database_partidos.db"
    )
    parser.add_argument(
        "--fecha", type=str, default=None,
        help="Fecha YYYY-MM-DD (default: ayer)"
    )
    parser.add_argument(
        "--dias", type=int, default=1,
        help="Días hacia atrás a procesar (default: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula sin escribir en la DB"
    )
    args = parser.parse_args()

    if args.fecha:
        try:
            datetime.strptime(args.fecha, "%Y-%m-%d")
            dates = [args.fecha]
        except ValueError:
            log.error("Formato inválido. Usa YYYY-MM-DD")
            sys.exit(1)
    else:
        today = date.today()
        dates = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, args.dias + 1)
        ]

    for d in dates:
        run(d, dry_run=args.dry_run)