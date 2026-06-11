"""
Pinnacle Scraper — World Cup (Auto-Discovery)
=============================================
Scrapes odds for all World Cup games from today through the next 3 days.
No database required — fetches all available matches automatically.

Markets extracted:
  • 1X2 (Moneyline)
  • Handicap / Asian Spread
  • Totales de goles (Over/Under)
  • Team Totals (Home & Away)
  • Corners (Total, Home, Away)
  • Shots on Target (Total, Home, Away)
"""


import requests
import json
import time
import os
import pandas as pd
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://guest.api.arcadia.pinnacle.com/0.1"

HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "es-ES,es;q=0.9",
    "Origin": "https://www.pinnacle.com",
    "Referer": "https://www.pinnacle.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "x-api-key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",
}

SPORT_ID_SOCCER   = 29
PERIODO_PARTIDO   = 0          # full-match period
KEYWORDS_CORNERS  = ["corner", "córner", "rincón"]
KEYWORDS_SHOTS    = ["shot", "tiro", "disparo", "shots on"]
REQUEST_DELAY     = 1.5        # seconds between requests
DAYS_AHEAD        = 3          # today + 3 days

# Keywords used to identify the World Cup league from the full catalogue
WORLD_CUP_KEYWORDS = [
    "world cup",
    "copa del mundo",
    "mundial",
    "fifa world",
]

# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER & UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def get(endpoint: str, params: dict = None):
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"  ⚠ Rate limit (429). Esperando 65s...")
                time.sleep(65)
            elif resp.status_code == 404:
                return None
            else:
                print(f"  ✗ HTTP {resp.status_code} en {endpoint}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Error de red ({attempt+1}/3): {e}")
            time.sleep(3)
    return None


def dentro_del_rango(fecha_iso: str) -> bool:
    """Returns True if the date is between now and DAYS_AHEAD days from now."""
    ahora  = datetime.now(timezone.utc)
    inicio = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    fin    = inicio + timedelta(days=DAYS_AHEAD)
    try:
        fecha = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
        return inicio <= fecha < fin
    except Exception:
        return False


def a_decimal(precio_americano) -> float | None:
    """Converts American odds to decimal odds."""
    if precio_americano is None:
        return None
    p = float(precio_americano)
    return round(p / 100 + 1, 4) if p > 0 else round(100 / abs(p) + 1, 4)


def contiene_keyword(texto: str, keywords: list) -> bool:
    return any(kw in texto.lower() for kw in keywords)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DISCOVERY: FIND THE WORLD CUP LEAGUE ID
# ─────────────────────────────────────────────────────────────────────────────

def auto_descubrir_world_cup() -> dict | None:
    """
    Scans all soccer leagues on Pinnacle to find the World Cup.
    Returns {"league_id": int, "nombre": str} or None if not found.
    """
    print("  → Escaneando catálogo de ligas de Pinnacle (fútbol)...")
    data = get(f"/sports/{SPORT_ID_SOCCER}/leagues", {"all": "false"})
    if not data:
        print("  ⚠ No se pudo descargar el catálogo de ligas.")
        return None

    candidatos = []
    for league in data:
        nombre = league.get("name", "")
        nombre_lower = nombre.lower()
        if any(kw in nombre_lower for kw in WORLD_CUP_KEYWORDS):
            candidatos.append({"league_id": league.get("id"), "nombre": nombre})

    if not candidatos:
        print("  ⚠ No se encontró ninguna liga de World Cup en el catálogo.")
        print("  ℹ Ligas disponibles (muestra):")
        for l in data[:20]:
            print(f"       {l.get('id')} – {l.get('name')}")
        return None

    # If several candidates found (e.g. U20, Women, etc.) list them all and
    # pick the most likely main event (shortest name / highest ID activity).
    if len(candidatos) > 1:
        print(f"  ℹ Múltiples ligas World Cup encontradas:")
        for c in candidatos:
            print(f"       ID {c['league_id']} – {c['nombre']}")
        # Prefer the one without qualifiers ("Women", "U20", "Youth", etc.)
        main = [c for c in candidatos if not any(
            x in c["nombre"].lower() for x in ["women", "u20", "youth", "qualifier", "u17", "u23"]
        )]
        elegido = main[0] if main else candidatos[0]
    else:
        elegido = candidatos[0]

    print(f"  ✅ Liga seleccionada: '{elegido['nombre']}' (ID: {elegido['league_id']})")
    return elegido

# ─────────────────────────────────────────────────────────────────────────────
# MATCHUPS & ODDS
# ─────────────────────────────────────────────────────────────────────────────

def obtener_partidos(league_id: int) -> list:
    """Returns all parent matchups in the date window."""
    data = get(f"/leagues/{league_id}/matchups")
    if not data:
        return []

    partidos = []
    for m in data:
        if not dentro_del_rango(m.get("startTime", "")):
            continue
        if m.get("type") != "matchup" or m.get("parentId"):
            continue
        participantes = m.get("participants", [])
        if len(participantes) < 2:
            continue
        home = next((p["name"] for p in participantes if p.get("alignment") == "home"), "?")
        away = next((p["name"] for p in participantes if p.get("alignment") == "away"), "?")
        partidos.append({
            "id":     m["id"],
            "home":   home,
            "away":   away,
            "inicio": m.get("startTime", ""),
        })
    return partidos


def obtener_odds_straight(league_id: int) -> list:
    return get(f"/leagues/{league_id}/markets/straight") or []


def parsear_straight(matchup_id: int, odds_data: list) -> dict:
    res = {
        "1x2": {},
        "handicap": [],
        "total_goles": [],
        "tt_home": [],
        "tt_away": [],
    }
    for market in odds_data:
        if market.get("matchupId") != matchup_id or market.get("period") != PERIODO_PARTIDO:
            continue
        tipo    = market.get("type", "").lower()
        precios = market.get("prices", [])
        es_alt  = market.get("altLineId") is not None

        if tipo == "moneyline":
            for p in precios:
                d = p.get("designation", "").lower()
                v = a_decimal(p.get("price"))
                if "home"        in d: res["1x2"]["home"] = v
                elif "away"      in d: res["1x2"]["away"] = v
                elif "draw" in d or "tie" in d: res["1x2"]["draw"] = v

        elif tipo == "spread":
            for p in precios:
                res["handicap"].append({
                    "lado":      p.get("designation", "").lower(),
                    "handicap":  p.get("points"),
                    "odds":      a_decimal(p.get("price")),
                    "es_alt":    es_alt,
                })

        elif tipo == "total":
            for p in precios:
                res["total_goles"].append({
                    "linea":  p.get("points"),
                    "lado":   p.get("designation", "").lower(),
                    "odds":   a_decimal(p.get("price")),
                    "es_alt": es_alt,
                })

        elif tipo == "team_total":
            equipo = market.get("side", "").lower()
            clave  = "tt_home" if equipo == "home" else "tt_away"
            for p in precios:
                res[clave].append({
                    "linea":  p.get("points"),
                    "lado":   p.get("designation", "").lower(),
                    "odds":   a_decimal(p.get("price")),
                    "es_alt": es_alt,
                })
    return res


def parsear_related(matchup_id: int) -> dict:
    """Fetches corners and shots props from related matchups."""
    resultado = {}
    related   = get(f"/matchups/{matchup_id}/related") or []
    time.sleep(REQUEST_DELAY)
    if not related:
        return resultado

    id_a_nombre = {
        r["id"]: r.get("units", r.get("name", ""))
        for r in related if r.get("id")
    }
    odds_related = get(f"/matchups/{matchup_id}/markets/related/straight") or []
    time.sleep(REQUEST_DELAY)

    odds_por_id: dict[int, list] = {}
    for market in odds_related:
        mid = market.get("matchupId")
        if mid:
            odds_por_id.setdefault(mid, []).append(market)

    for sub_id, nombre in id_a_nombre.items():
        if not nombre:
            continue
        es_corner = contiene_keyword(nombre, KEYWORDS_CORNERS)
        es_shot   = contiene_keyword(nombre, KEYWORDS_SHOTS)
        if not es_corner and not es_shot:
            continue
        prefijo = "corners" if es_corner else "shots"

        for market in odds_por_id.get(sub_id, []):
            if market.get("period") != PERIODO_PARTIDO:
                continue
            if market.get("altLineId") is not None:
                continue
            tipo    = market.get("type", "").lower()
            precios = market.get("prices", [])

            if tipo == "total":
                entrada: dict = {}
                for p in precios:
                    entrada[p.get("designation", "").lower()] = a_decimal(p.get("price"))
                    entrada["linea"] = p.get("points")
                if entrada:
                    resultado[f"{prefijo}_total"] = entrada

            elif tipo == "team_total":
                side = market.get("side", "").lower()
                entrada = {}
                for p in precios:
                    entrada[p.get("designation", "").lower()] = a_decimal(p.get("price"))
                    entrada["linea"] = p.get("points")
                if entrada and side in ["home", "away"]:
                    resultado[f"{prefijo}_{side}"] = entrada

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def scrapear_world_cup(league_id: int, nombre_liga: str) -> list:
    partidos = obtener_partidos(league_id)
    if not partidos:
        print(f"  ⚠ No hay partidos de {nombre_liga} en los próximos {DAYS_AHEAD} días.")
        return []

    print(f"  → {len(partidos)} partido(s) encontrado(s). Descargando cuotas principales...")
    odds_straight_data = obtener_odds_straight(league_id)
    time.sleep(REQUEST_DELAY)

    resultados = []
    for partido in partidos:
        mid, home_name, away_name = partido["id"], partido["home"], partido["away"]
        print(f"\n  ⚽ {home_name} vs {away_name}")

        straight = parsear_straight(mid, odds_straight_data)

        print(f"     → Buscando props (Córners / Tiros)...")
        related  = parsear_related(mid)

        if related:
            print(f"     ✅ Props encontrados: {', '.join(related.keys())}")
        else:
            print(f"     ⚠ Sin props disponibles aún")

        # Convert UTC time to local time
        fecha_dt    = datetime.fromisoformat(partido["inicio"].replace("Z", "+00:00"))
        fecha_local = fecha_dt.astimezone()

        fila: dict = {
            "liga":          nombre_liga,
            "home":          home_name,
            "away":          away_name,
            "inicio_utc":    partido["inicio"],
            "inicio_local":  fecha_local.strftime("%Y-%m-%d %H:%M"),
        }

        # ── 1X2 ──────────────────────────────────────────────────────────────
        for k, v in straight["1x2"].items():
            fila[f"1x2_{k}"] = v

        # ── Handicap (main lines only, max 2 sides) ───────────────────────────
        for h in [h for h in straight["handicap"] if not h["es_alt"]][:2]:
            fila[f"hdp_{h['lado']}_{h['handicap']}"] = h["odds"]

        # ── Totales de goles (main lines, max 4) ─────────────────────────────
        for t in [t for t in straight["total_goles"] if not t["es_alt"]][:4]:
            fila[f"goles_{t['linea']}_{t['lado']}"] = t["odds"]

        # ── Team Totals ───────────────────────────────────────────────────────
        for equipo, src in [("home", "tt_home"), ("away", "tt_away")]:
            for t in [t for t in straight[src] if not t["es_alt"]][:2]:
                fila[f"goles_tt_{equipo}_{t['linea']}_{t['lado']}"] = t["odds"]

        # ── Corners ───────────────────────────────────────────────────────────
        for subtipo in ("total", "home", "away"):
            datos = related.get(f"corners_{subtipo}", {})
            if datos:
                for lado in ("over", "under"):
                    if lado in datos:
                        fila[f"corners_{subtipo}_{datos.get('linea', '?')}_{lado}"] = datos[lado]

        # ── Shots ─────────────────────────────────────────────────────────────
        for subtipo in ("total", "home", "away"):
            datos = related.get(f"shots_{subtipo}", {})
            if datos:
                for lado in ("over", "under"):
                    if lado in datos:
                        fila[f"shots_{subtipo}_{datos.get('linea', '?')}_{lado}"] = datos[lado]

        resultados.append(fila)
        print(f"     ✓ Completado")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    hoy = datetime.now()
    print("=" * 65)
    print("  🌍 PINNACLE WORLD CUP SCRAPER")
    print(f"  📅 Período: {hoy.strftime('%Y-%m-%d')} → {(hoy + timedelta(days=DAYS_AHEAD)).strftime('%Y-%m-%d')}")
    print("=" * 65)

    liga = auto_descubrir_world_cup()
    if not liga:
        print("\n❌ No se encontró la liga del World Cup en Pinnacle.")
        print("   El torneo puede no estar disponible en este momento.")
        return

    print(f"\n{'─'*65}")
    print(f"  {liga['nombre'].upper()}")
    print(f"{'─'*65}")

    todos = scrapear_world_cup(liga["league_id"], liga["nombre"])

    if not todos:
        print("\n⚠ No se encontraron partidos en el rango de fechas.")
        return

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("odds_data", exist_ok=True)
    fecha_str = hoy.strftime("%Y%m%d")
    csv_path  = f"odds_data/worldcup_{fecha_str}.csv"
    json_path = f"odds_data/worldcup_{fecha_str}.json"

    df = pd.DataFrame(todos)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ ¡Éxito! {len(todos)} partido(s) guardado(s).")
    print(f"📂 CSV  → {csv_path}")
    print(f"📂 JSON → {json_path}")
    print("=" * 65)

    # ── Quick preview ─────────────────────────────────────────────────────────
    print("\n📋 Vista previa:")
    preview_cols = ["liga", "home", "away", "inicio_local", "1x2_home", "1x2_draw", "1x2_away"]
    available = [c for c in preview_cols if c in df.columns]
    print(df[available].to_string(index=False))


if __name__ == "__main__":
    main()