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
  - 1X2 (moneyline)
  - Hándicap asiático (spreads)
  - Total de goles (over/under)
  - Total de goles por equipo (team totals)
  - [Córners y tiros a puerta: disponibles vía specials si Pinnacle los ofrece]

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
    "Premier League":  {
        "league_id": 1980,
        "pais": "Inglaterra",
    },
    "La Liga": {
        "league_id": 2036,
        "pais": "España",
    },
    "Serie A": {
        "league_id": 2037,
        "pais": "Italia",
    },
    "Bundesliga": {
        "league_id": 1842,
        "pais": "Alemania",
    },
    "Ligue 1": {
        "league_id": 2141,
        "pais": "Francia",
    },
}

# Períodos de Pinnacle para fútbol
# 0 = partido completo, 1 = primer tiempo
PERIODO_PARTIDO = 0

# Delay entre requests para no saturar (segundos)
REQUEST_DELAY = 1.2


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get(endpoint: str, params: dict = None) -> dict | list | None:
    """Hace GET a la guest API de Pinnacle con manejo de errores y retry."""
    url = f"{BASE_URL}{endpoint}"
    for intento in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"  ⚠ Rate limit (429). Esperando 60s...")
                time.sleep(60)
            else:
                print(f"  ✗ HTTP {resp.status_code} en {endpoint}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Error de red ({intento+1}/3): {e}")
            time.sleep(3)
    return None


def es_hoy_o_manana(fecha_iso: str) -> bool:
    """True si la fecha está dentro de las próximas 48 horas."""
    ahora = datetime.now(timezone.utc)
    inicio = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    fin = inicio + timedelta(days=2)
    try:
        fecha = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
        return inicio <= fecha < fin
    except Exception:
        return False


def odds_decimal(precio_americano: float) -> float:
    """Convierte odds americanas a decimal."""
    if precio_americano is None:
        return None
    if precio_americano > 0:
        return round(precio_americano / 100 + 1, 4)
    else:
        return round(100 / abs(precio_americano) + 1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING DE PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────

def obtener_partidos(league_id: int, nombre_liga: str) -> list[dict]:
    """Obtiene la lista de matchups (partidos) de una liga."""
    print(f"\n  → Obteniendo partidos de {nombre_liga} (league_id={league_id})...")
    data = get(f"/leagues/{league_id}/matchups")
    if not data:
        return []

    partidos = []
    for m in data:
        # Filtrar partidos de hoy y mañana (no live ya empezados sin odds)
        starts = m.get("startTime", "")
        if not es_hoy_o_manana(starts):
            continue

        # Solo partidos principales (no props/alternates)
        if m.get("type") != "matchup" or m.get("parentId"):
            continue

        participantes = m.get("participants", [])
        if len(participantes) < 2:
            continue

        home = next((p["name"] for p in participantes if p.get("alignment") == "home"), "?")
        away = next((p["name"] for p in participantes if p.get("alignment") == "away"), "?")

        partidos.append({
            "id": m["id"],
            "liga": nombre_liga,
            "home": home,
            "away": away,
            "inicio": starts,
        })

    print(f"     {len(partidos)} partidos encontrados hoy/mañana")
    return partidos


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING DE ODDS
# ─────────────────────────────────────────────────────────────────────────────

def obtener_odds_straight(league_id: int) -> list[dict]:
    """
    Obtiene todas las odds 'straight' de la liga:
    moneyline (1X2), spread (hándicap asiático), totals, team totals.
    """
    data = get(f"/leagues/{league_id}/markets/straight")
    return data if data else []


def parsear_odds_partido(matchup_id: int, odds_data: list) -> dict:
    """
    Filtra y estructura las odds para un partido específico.
    Retorna dict con mercados: 1x2, handicap, total_goles, team_totals.
    """
    resultado = {
        "1x2": None,
        "handicap_asiatico": [],
        "total_goles": [],
        "team_total_home": [],
        "team_total_away": [],
    }

    for market in odds_data:
        if market.get("matchupId") != matchup_id:
            continue
        if market.get("period") != PERIODO_PARTIDO:
            continue

        tipo = market.get("type", "").lower()
        precios = market.get("prices", [])

        # ── 1X2 (Moneyline) ──
        if tipo == "moneyline":
            ml = {}
            for p in precios:
                desig = p.get("designation", "").lower()
                odds_raw = p.get("price")
                if "home" in desig:
                    ml["home"] = odds_decimal(odds_raw)
                elif "away" in desig:
                    ml["away"] = odds_decimal(odds_raw)
                elif "draw" in desig or "tie" in desig:
                    ml["draw"] = odds_decimal(odds_raw)
            if ml:
                resultado["1x2"] = ml

        # ── Hándicap Asiático (Spread) ──
        elif tipo == "spread":
            for p in precios:
                desig = p.get("designation", "").lower()
                hdp = p.get("points")
                precio = odds_decimal(p.get("price"))
                resultado["handicap_asiatico"].append({
                    "lado": desig,
                    "handicap": hdp,
                    "odds": precio,
                })

        # ── Total Goles (Over/Under) ──
        elif tipo == "total":
            alt = market.get("altLineId")  # None = línea principal
            for p in precios:
                desig = p.get("designation", "").lower()
                puntos = p.get("points")
                precio = odds_decimal(p.get("price"))
                resultado["total_goles"].append({
                    "linea": puntos,
                    "lado": desig,
                    "odds": precio,
                    "es_alternativa": alt is not None,
                })

        # ── Team Totals ──
        elif tipo == "team_total":
            equipo = market.get("side", "").lower()
            for p in precios:
                desig = p.get("designation", "").lower()
                puntos = p.get("points")
                precio = odds_decimal(p.get("price"))
                clave = f"team_total_{equipo}" if equipo in ("home", "away") else "team_total_home"
                resultado[clave].append({
                    "linea": puntos,
                    "lado": desig,
                    "odds": precio,
                })

    return resultado


def obtener_odds_specials(league_id: int, matchup_id: int) -> list[dict]:
    """
    Obtiene mercados especiales de un partido (córners, tiros a puerta, etc.)
    Nota: Pinnacle no siempre ofrece estos mercados para todas las ligas.
    """
    data = get(f"/leagues/{league_id}/matchups/{matchup_id}/related")
    if not data:
        return []

    especiales = []
    for item in data:
        nombre = item.get("name", "")
        nombre_lower = nombre.lower()
        if any(kw in nombre_lower for kw in ["corner", "córner", "shot", "tiro", "disparo"]):
            especiales.append({
                "nombre": nombre,
                "id": item.get("id"),
                "tipo": item.get("type"),
            })
    return especiales


# ─────────────────────────────────────────────────────────────────────────────
# FLUJO PRINCIPAL POR LIGA
# ─────────────────────────────────────────────────────────────────────────────

def scrapear_liga(nombre_liga: str, config: dict) -> list[dict]:
    """
    Scrapea una liga completa: partidos + odds de todos los mercados.
    """
    league_id = config["league_id"]
    partidos = obtener_partidos(league_id, nombre_liga)
    if not partidos:
        return []

    time.sleep(REQUEST_DELAY)

    # Obtener todas las odds de la liga en un solo call (eficiente)
    print(f"  → Obteniendo odds de {nombre_liga}...")
    odds_data = obtener_odds_straight(league_id)
    time.sleep(REQUEST_DELAY)

    resultados = []
    for partido in partidos:
        mid = partido["id"]
        odds_partido = parsear_odds_partido(mid, odds_data)

        # Intentar obtener specials (córners/tiros — opcionales)
        # Comentado por defecto para no exceder rate limits; descomentá si lo necesitás
        # specials = obtener_odds_specials(league_id, mid)
        # time.sleep(REQUEST_DELAY)

        fecha_dt = datetime.fromisoformat(partido["inicio"].replace("Z", "+00:00"))
        fecha_local = fecha_dt.astimezone()  # convierte a hora local

        fila = {
            "liga": nombre_liga,
            "pais": config["pais"],
            "partido_id": mid,
            "home": partido["home"],
            "away": partido["away"],
            "inicio_utc": partido["inicio"],
            "inicio_local": fecha_local.strftime("%Y-%m-%d %H:%M"),
            **{f"1x2_{k}": v for k, v in (odds_partido["1x2"] or {}).items()},
        }

        # Hándicap asiático — línea principal (primer hdp)
        hdps = odds_partido["handicap_asiatico"]
        if hdps:
            for h in hdps[:2]:  # home y away de la línea principal
                fila[f"hdp_{h['lado']}"] = h["handicap"]
                fila[f"hdp_{h['lado']}_odds"] = h["odds"]

        # Total goles — líneas no alternativas
        totales_main = [t for t in odds_partido["total_goles"] if not t["es_alternativa"]]
        for t in totales_main[:4]:  # over y under de las 2 primeras líneas
            fila[f"total_{t['linea']}_{t['lado']}"] = t["odds"]

        # Team totals — home y away
        for equipo in ("home", "away"):
            for t in odds_partido[f"team_total_{equipo}"][:2]:
                fila[f"tt_{equipo}_{t['linea']}_{t['lado']}"] = t["odds"]

        # Raw para debug / uso posterior
        fila["_raw_odds"] = json.dumps(odds_partido, ensure_ascii=False)

        resultados.append(fila)
        print(f"     ✓ {partido['home']} vs {partido['away']} ({fecha_local.strftime('%d/%m %H:%M')})")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    hoy = datetime.now()
    manana = hoy + timedelta(days=1)
    print("=" * 65)
    print("  PINNACLE SCRAPER — TOP 5 LIGAS EUROPEAS")
    print(f"  Partidos: {hoy.strftime('%d/%m/%Y')} y {manana.strftime('%d/%m/%Y')}")
    print("=" * 65)

    todos = []
    for nombre_liga, config in LIGAS.items():
        print(f"\n{'─'*55}")
        print(f"  Liga: {nombre_liga.upper()} ({config['pais']})")
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
        print("\n⚠ No se encontraron partidos para hoy/mañana.")
        print("  Esto puede pasar si no hay jornada en las ligas seleccionadas.")
        return []

    # ── Guardar resultados ──
    fecha_str = hoy.strftime("%Y%m%d")

    # JSON completo (incluye raw odds)
    json_path = f"pinnacle_{fecha_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON guardado: {json_path}")

    # CSV limpio (sin columna _raw_odds)
    df = pd.DataFrame(todos)
    cols_csv = [c for c in df.columns if c != "_raw_odds"]
    csv_path = f"pinnacle_{fecha_str}.csv"
    df[cols_csv].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV guardado:  {csv_path}")

    # Preview en consola
    print(f"\n📋 Preview ({min(5, len(todos))} primeros partidos):")
    cols_preview = ["liga", "home", "away", "inicio_local", "1x2_home", "1x2_draw", "1x2_away"]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print(df[cols_preview].head(10).to_string(index=False))

    return todos


if __name__ == "__main__":
    main()