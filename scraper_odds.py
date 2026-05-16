"""
Pinnacle Scraper — Guest API (sin Selenium, sin login)
=======================================================
Usa el endpoint público guest.api.arcadia.pinnacle.com que consume
el propio frontend de Pinnacle. No requiere cuenta ni API key.

Scrapea partidos de HOY y MAÑANA de las Top 5 ligas europeas:
  - Premier League (Inglaterra)
  - La Liga (España)
  - Serie A (Italia)
  - Bundesliga (Alemania)
  - Ligue 1 (Francia)

Mercados capturados:
  STRAIGHT (un call por liga):
    - 1X2 (moneyline)
    - Hándicap asiático (spreads)
    - Total de goles — partido completo y por equipo

  RELATED (dos calls por partido):
    - Córners — total y por equipo
    - Tiros a puerta — total y por equipo

Instalación:
    pip install requests pandas

Uso:
    python pinnacle_scraper.py
"""

import requests
import json
import time
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
    "x-api-key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",  # token público del frontend
}

# IDs de liga en Pinnacle (Soccer sportId = 29)
LIGAS = {
    "Premier League": {"league_id": 1980, "pais": "Inglaterra"},
    "La Liga":        {"league_id": 2036, "pais": "España"},
    "Serie A":        {"league_id": 2037, "pais": "Italia"},
    "Bundesliga":     {"league_id": 1842, "pais": "Alemania"},
    "Ligue 1":        {"league_id": 2141, "pais": "Francia"},
}

# Período 0 = partido completo, 1 = primer tiempo
PERIODO_PARTIDO = 0

# Keywords para detectar córners y tiros en los nombres de related matchups
KEYWORDS_CORNERS = ["corner", "córner", "rincón"]
KEYWORDS_SHOTS   = ["shot", "tiro", "disparo", "shots on"]

# Delay entre requests (segundos) — respetar rate limits de Pinnacle
REQUEST_DELAY = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get(endpoint: str, params: dict = None):
    """GET a la guest API de Pinnacle con retry y manejo de rate limit."""
    url = f"{BASE_URL}{endpoint}"
    for intento in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"  ⚠ Rate limit (429). Esperando 65s...")
                time.sleep(65)
            elif resp.status_code == 404:
                return None  # sin datos, no reintentar
            else:
                print(f"  ✗ HTTP {resp.status_code} en {endpoint}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Error de red ({intento+1}/3): {e}")
            time.sleep(3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def es_hoy_o_manana(fecha_iso: str) -> bool:
    """True si la fecha cae dentro de las próximas 48 horas (UTC)."""
    ahora  = datetime.now(timezone.utc)
    inicio = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    fin    = inicio + timedelta(days=2)
    try:
        fecha = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
        return inicio <= fecha < fin
    except Exception:
        return False


def a_decimal(precio_americano) -> float | None:
    """Convierte odds americanas a formato decimal."""
    if precio_americano is None:
        return None
    p = float(precio_americano)
    return round(p / 100 + 1, 4) if p > 0 else round(100 / abs(p) + 1, 4)


def contiene_keyword(texto: str, keywords: list) -> bool:
    t = texto.lower()
    return any(kw in t for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────

def obtener_partidos(league_id: int, nombre_liga: str) -> list:
    """Devuelve partidos de hoy/mañana de la liga."""
    print(f"  → Partidos de {nombre_liga}...")
    data = get(f"/leagues/{league_id}/matchups")
    if not data:
        return []

    partidos = []
    for m in data:
        if not es_hoy_o_manana(m.get("startTime", "")):
            continue
        # Solo matchups principales (parentId ausente = no son sub-eventos)
        if m.get("type") != "matchup" or m.get("parentId"):
            continue
        participantes = m.get("participants", [])
        if len(participantes) < 2:
            continue

        home = next((p["name"] for p in participantes if p.get("alignment") == "home"), "?")
        away = next((p["name"] for p in participantes if p.get("alignment") == "away"), "?")
        partidos.append({"id": m["id"], "home": home, "away": away, "inicio": m.get("startTime", "")})

    print(f"     {len(partidos)} partidos encontrados")
    return partidos


# ─────────────────────────────────────────────────────────────────────────────
# ODDS STRAIGHT (1X2 · HANDICAP · TOTALES · TEAM TOTALS)
# ─────────────────────────────────────────────────────────────────────────────

def obtener_odds_straight(league_id: int) -> list:
    """Un solo call trae todas las odds straight de la liga."""
    return get(f"/leagues/{league_id}/markets/straight") or []


def parsear_straight(matchup_id: int, odds_data: list) -> dict:
    """Extrae y estructura las odds straight para un partido concreto."""
    res = {"1x2": {}, "handicap": [], "total_goles": [], "tt_home": [], "tt_away": []}

    for market in odds_data:
        if market.get("matchupId") != matchup_id:
            continue
        if market.get("period") != PERIODO_PARTIDO:
            continue

        tipo    = market.get("type", "").lower()
        precios = market.get("prices", [])
        es_alt  = market.get("altLineId") is not None

        if tipo == "moneyline":
            for p in precios:
                d = p.get("designation", "").lower()
                v = a_decimal(p.get("price"))
                if "home" in d:                   res["1x2"]["home"] = v
                elif "away" in d:                 res["1x2"]["away"] = v
                elif "draw" in d or "tie" in d:   res["1x2"]["draw"] = v

        elif tipo == "spread":
            for p in precios:
                res["handicap"].append({
                    "lado":     p.get("designation", "").lower(),
                    "handicap": p.get("points"),
                    "odds":     a_decimal(p.get("price")),
                    "es_alt":   es_alt,
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


# ─────────────────────────────────────────────────────────────────────────────
# ODDS RELATED (CÓRNERS · TIROS A PUERTA)
# ─────────────────────────────────────────────────────────────────────────────

def parsear_related(matchup_id: int) -> dict:
    """
    Obtiene córners y tiros a puerta vía dos endpoints específicos:
      GET /matchups/{id}/related                    → lista de sub-matchups
      GET /matchups/{id}/markets/related/straight   → odds de esos sub-matchups

    Retorna dict con claves: corners_total, corners_home, corners_away,
                                shots_total,   shots_home,   shots_away
    Cada clave contiene: {linea, over, under}
    """
    resultado = {}

    # 1) Lista de sub-matchups relacionados
    related = get(f"/matchups/{matchup_id}/related") or []
    time.sleep(REQUEST_DELAY)

    if not related:
        return resultado

    # id → nombre del sub-matchup
    id_a_nombre = {r["id"]: r.get("name", "") for r in related if r.get("id")}

    # 2) Odds de todos esos sub-matchups en un solo call
    odds_related = get(f"/matchups/{matchup_id}/markets/related/straight") or []
    time.sleep(REQUEST_DELAY)

    if not odds_related:
        return resultado

    # Agrupar odds por sub-matchup id
    odds_por_id: dict = {}
    for market in odds_related:
        mid = market.get("matchupId")
        if mid:
            odds_por_id.setdefault(mid, []).append(market)

    # Clasificar y parsear cada sub-matchup relevante
    for sub_id, nombre in id_a_nombre.items():
        es_corner = contiene_keyword(nombre, KEYWORDS_CORNERS)
        es_shot   = contiene_keyword(nombre, KEYWORDS_SHOTS)
        if not es_corner and not es_shot:
            continue

        # Determinar sub-tipo: total / home / away
        n = nombre.lower()
        if "home" in n:       subtipo = "home"
        elif "away" in n:     subtipo = "away"
        else:                 subtipo = "total"

        prefijo = "corners" if es_corner else "shots"
        clave   = f"{prefijo}_{subtipo}"

        # Parsear el mercado total (over/under) de este sub-matchup
        entrada = {}
        for market in odds_por_id.get(sub_id, []):
            if market.get("period") != PERIODO_PARTIDO:
                continue
            if market.get("type", "").lower() != "total":
                continue
            for p in market.get("prices", []):
                lado  = p.get("designation", "").lower()
                linea = p.get("points")
                odds  = a_decimal(p.get("price"))
                entrada[lado]    = odds
                entrada["linea"] = linea

        if entrada:
            resultado[clave] = entrada

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# FLUJO POR LIGA
# ─────────────────────────────────────────────────────────────────────────────

def scrapear_liga(nombre_liga: str, config: dict) -> list:
    league_id = config["league_id"]

    partidos = obtener_partidos(league_id, nombre_liga)
    if not partidos:
        return []
    time.sleep(REQUEST_DELAY)

    print(f"  → Odds straight {nombre_liga}...")
    odds_straight_data = obtener_odds_straight(league_id)
    time.sleep(REQUEST_DELAY)

    resultados = []
    for partido in partidos:
        mid       = partido["id"]
        home_name = partido["home"]
        away_name = partido["away"]

        # Parsear straight (1x2, handicap, goles, team totals)
        straight = parsear_straight(mid, odds_straight_data)

        # Parsear related (córners, tiros a puerta) — 2 calls HTTP
        print(f"     → Related: {home_name} vs {away_name}...")
        related = parsear_related(mid)

        # Formatear fechas
        fecha_dt    = datetime.fromisoformat(partido["inicio"].replace("Z", "+00:00"))
        fecha_local = fecha_dt.astimezone()

        # Construir fila del CSV/JSON
        fila = {
            "liga":         nombre_liga,
            "pais":         config["pais"],
            "partido_id":   mid,
            "home":         home_name,
            "away":         away_name,
            "inicio_utc":   partido["inicio"],
            "inicio_local": fecha_local.strftime("%Y-%m-%d %H:%M"),
        }

        # ── 1X2 ──
        for k, v in straight["1x2"].items():
            fila[f"1x2_{k}"] = v

        # ── Handicap asiático — línea principal ──
        hdps_main = [h for h in straight["handicap"] if not h["es_alt"]]
        for h in hdps_main[:2]:
            fila[f"hdp_{h['lado']}"]      = h["handicap"]
            fila[f"hdp_{h['lado']}_odds"] = h["odds"]

        # ── Total goles — línea principal ──
        totales_main = [t for t in straight["total_goles"] if not t["es_alt"]]
        for t in totales_main[:4]:
            fila[f"goles_{t['linea']}_{t['lado']}"] = t["odds"]

        # ── Team totals goles — línea principal ──
        for equipo, src in [("home", "tt_home"), ("away", "tt_away")]:
            tt_main = [t for t in straight[src] if not t["es_alt"]]
            for t in tt_main[:2]:
                fila[f"goles_tt_{equipo}_{t['linea']}_{t['lado']}"] = t["odds"]

        # ── Córners ──
        for subtipo in ("total", "home", "away"):
            datos = related.get(f"corners_{subtipo}", {})
            if datos:
                linea = datos.get("linea", "?")
                for lado in ("over", "under"):
                    if lado in datos:
                        fila[f"corners_{subtipo}_{linea}_{lado}"] = datos[lado]

        # ── Tiros a puerta ──
        for subtipo in ("total", "home", "away"):
            datos = related.get(f"shots_{subtipo}", {})
            if datos:
                linea = datos.get("linea", "?")
                for lado in ("over", "under"):
                    if lado in datos:
                        fila[f"shots_{subtipo}_{linea}_{lado}"] = datos[lado]

        # Raw para debugging
        fila["_raw"] = json.dumps({"straight": straight, "related": related}, ensure_ascii=False)

        resultados.append(fila)
        print(f"     ✓ {home_name} vs {away_name} ({fecha_local.strftime('%d/%m %H:%M')})")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    hoy    = datetime.now()
    manana = hoy + timedelta(days=1)

    print("=" * 65)
    print("  PINNACLE SCRAPER — TOP 5 LIGAS EUROPEAS")
    print(f"  Partidos: {hoy.strftime('%d/%m/%Y')} y {manana.strftime('%d/%m/%Y')}")
    print("  Mercados: 1X2 · Handicap · Goles · Team Totals · Córners · Tiros")
    print("=" * 65)

    todos = []
    for nombre_liga, config in LIGAS.items():
        print(f"\n{'─'*55}")
        print(f"  {nombre_liga.upper()} ({config['pais']})")
        print(f"{'─'*55}")
        try:
            filas = scrapear_liga(nombre_liga, config)
            todos.extend(filas)
        except Exception as e:
            print(f"  ❌ Error en {nombre_liga}: {e}")
        time.sleep(REQUEST_DELAY * 2)

    print(f"\n{'='*65}")
    print(f"  TOTAL: {len(todos)} partidos scrapeados")
    print("=" * 65)

    if not todos:
        print("\n⚠ Sin partidos para hoy/mañana en las ligas seleccionadas.")
        return []

    fecha_str = hoy.strftime("%Y%m%d")

    # JSON completo (incluye _raw para debug)
    json_path = f"pinnacle_{fecha_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {json_path}")

    # CSV limpio (sin _raw)
    df       = pd.DataFrame(todos)
    cols_csv = [c for c in df.columns if c != "_raw"]
    csv_path = f"pinnacle_{fecha_str}.csv"
    df[cols_csv].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV:  {csv_path}")

    # Preview consola
    cols_preview = ["liga", "home", "away", "inicio_local", "1x2_home", "1x2_draw", "1x2_away"]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print(f"\n📋 Preview:\n")
    print(df[cols_preview].head(10).to_string(index=False))

    # Resumen columnas de mercado
    mercado_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["1x2", "hdp", "goles", "corners", "shots"]
    )]
    print(f"\n📊 Columnas de mercado capturadas:")
    print(", ".join(mercado_cols))

    return todos


if __name__ == "__main__":
    main()