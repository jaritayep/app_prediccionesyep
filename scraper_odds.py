"""
Pinnacle Scraper — Guest API (Automatizado con Base de Datos)
=======================================================
Usa el endpoint público de Pinnacle. 
Se conecta a tu SQLite local, filtra solo los partidos que te interesan,
extrae las cuotas y guarda todo ordenado en la carpeta /odds_data.
"""

import requests
import json
import time
import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from thefuzz import process

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

# Ligas principales mapeadas a los IDs de Pinnacle
LIGAS = {
    "Premier League": {"league_id": 1980, "pais": "Inglaterra"},
    "La Liga":        {"league_id": 2036, "pais": "España"},
    "Serie A":        {"league_id": 2037, "pais": "Italia"},
    "Bundesliga":     {"league_id": 1842, "pais": "Alemania"},
    "Ligue 1":        {"league_id": 2141, "pais": "Francia"},
}

PERIODO_PARTIDO = 0
KEYWORDS_CORNERS = ["corner", "córner", "rincón"]
KEYWORDS_SHOTS   = ["shot", "tiro", "disparo", "shots on"]
REQUEST_DELAY = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN A BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def obtener_partidos_automatizados():
    """Lee tu base de datos para saber qué partidos buscar hoy y mañana"""
    nombre_bd = "database_partidos.db" # <-- Asegúrate que sea el nombre de tu archivo DB
    
    try:
        conn = sqlite3.connect(nombre_bd)
        hoy = datetime.now().strftime("%Y-%m-%d")
        manana = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        
        query = f"""
            SELECT League, Local, Visita 
            FROM tabla_predicciones_limpia 
            WHERE date(Date) >= '{hoy}' AND date(Date) <= '{manana}'
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df['Local'].tolist(), df['Visita'].tolist()
    except Exception as e:
        print(f"Error leyendo base de datos: {e}")
        return [], []

# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER & UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def get(endpoint: str, params: dict = None):
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
                return None 
            else:
                print(f"  ✗ HTTP {resp.status_code} en {endpoint}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Error de red ({intento+1}/3): {e}")
            time.sleep(3)
    return None

def es_hoy_o_manana(fecha_iso: str) -> bool:
    ahora  = datetime.now(timezone.utc)
    inicio = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    fin    = inicio + timedelta(days=2)
    try:
        fecha = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
        return inicio <= fecha < fin
    except Exception:
        return False

def a_decimal(precio_americano) -> float | None:
    if precio_americano is None:
        return None
    p = float(precio_americano)
    return round(p / 100 + 1, 4) if p > 0 else round(100 / abs(p) + 1, 4)

def contiene_keyword(texto: str, keywords: list) -> bool:
    return any(kw in texto.lower() for kw in keywords)

# ─────────────────────────────────────────────────────────────────────────────
# PARTIDOS Y ODDS (LÓGICA PRINCIPAL)
# ─────────────────────────────────────────────────────────────────────────────

def obtener_partidos(league_id: int, nombre_liga: str) -> list:
    print(f"  → Consultando cartelera de {nombre_liga}...")
    data = get(f"/leagues/{league_id}/matchups")
    if not data: return []

    partidos = []
    for m in data:
        if not es_hoy_o_manana(m.get("startTime", "")): continue
        if m.get("type") != "matchup" or m.get("parentId"): continue
        
        participantes = m.get("participants", [])
        if len(participantes) < 2: continue

        home = next((p["name"] for p in participantes if p.get("alignment") == "home"), "?")
        away = next((p["name"] for p in participantes if p.get("alignment") == "away"), "?")
        partidos.append({"id": m["id"], "home": home, "away": away, "inicio": m.get("startTime", "")})

    return partidos

def obtener_odds_straight(league_id: int) -> list:
    return get(f"/leagues/{league_id}/markets/straight") or []

def parsear_straight(matchup_id: int, odds_data: list) -> dict:
    res = {"1x2": {}, "handicap": [], "total_goles": [], "tt_home": [], "tt_away": []}
    for market in odds_data:
        if market.get("matchupId") != matchup_id or market.get("period") != PERIODO_PARTIDO:
            continue

        tipo = market.get("type", "").lower()
        precios = market.get("prices", [])
        es_alt = market.get("altLineId") is not None

        if tipo == "moneyline":
            for p in precios:
                d = p.get("designation", "").lower()
                v = a_decimal(p.get("price"))
                if "home" in d: res["1x2"]["home"] = v
                elif "away" in d: res["1x2"]["away"] = v
                elif "draw" in d or "tie" in d: res["1x2"]["draw"] = v

        elif tipo == "spread":
            for p in precios:
                res["handicap"].append({"lado": p.get("designation", "").lower(), "handicap": p.get("points"), "odds": a_decimal(p.get("price")), "es_alt": es_alt})

        elif tipo == "total":
            for p in precios:
                res["total_goles"].append({"linea": p.get("points"), "lado": p.get("designation", "").lower(), "odds": a_decimal(p.get("price")), "es_alt": es_alt})

        elif tipo == "team_total":
            equipo = market.get("side", "").lower()
            clave = "tt_home" if equipo == "home" else "tt_away"
            for p in precios:
                res[clave].append({"linea": p.get("points"), "lado": p.get("designation", "").lower(), "odds": a_decimal(p.get("price")), "es_alt": es_alt})

    return res

def parsear_related(matchup_id: int, home_name: str, away_name: str) -> dict:
    resultado = {}
    related = get(f"/matchups/{matchup_id}/related") or []
    time.sleep(REQUEST_DELAY)
    if not related: return resultado

    id_a_nombre = {r["id"]: r.get("name", "") for r in related if r.get("id")}
    odds_related = get(f"/matchups/{matchup_id}/markets/related/straight") or []
    time.sleep(REQUEST_DELAY)

    odds_por_id = {}
    for market in odds_related:
        mid = market.get("matchupId")
        if mid: odds_por_id.setdefault(mid, []).append(market)

    for sub_id, nombre in id_a_nombre.items():
        es_corner = contiene_keyword(nombre, KEYWORDS_CORNERS)
        es_shot = contiene_keyword(nombre, KEYWORDS_SHOTS)
        if not es_corner and not es_shot: continue

        n = nombre.lower()
        h_in = home_name.lower() in n
        a_in = away_name.lower() in n

        # Lógica perfecta para asignar local, visita o totales
        if h_in and a_in:
            subtipo = "total"
        elif h_in:
            subtipo = "home"
        elif a_in:
            subtipo = "away"
        else:
            subtipo = "total"

        clave = f"{'corners' if es_corner else 'shots'}_{subtipo}"

        entrada = {}
        for market in odds_por_id.get(sub_id, []):
            if market.get("period") != PERIODO_PARTIDO or market.get("type", "").lower() != "total": continue
            for p in market.get("prices", []):
                entrada[p.get("designation", "").lower()] = a_decimal(p.get("price"))
                entrada["linea"] = p.get("points")

        if entrada: resultado[clave] = entrada

    return resultado

def scrapear_liga(nombre_liga: str, config: dict, locales_db: list, visitas_db: list) -> list:
    league_id = config["league_id"]
    partidos = obtener_partidos(league_id, nombre_liga)
    if not partidos: return []
    time.sleep(REQUEST_DELAY)

    # --- FILTRO INTELIGENTE CON LA DB ---
    partidos_filtrados = []
    for p in partidos:
        match_home = process.extractOne(p["home"], locales_db)
        match_away = process.extractOne(p["away"], visitas_db)
        
        if match_home and match_away and match_home[1] > 80 and match_away[1] > 80:
            partidos_filtrados.append(p)
            
    if not partidos_filtrados:
        print(f"  ⚠ Ningún partido de {nombre_liga} programado en tu DB para hoy/mañana.")
        return []

    print(f"  → Extrayendo líneas principales para {len(partidos_filtrados)} partidos...")
    odds_straight_data = obtener_odds_straight(league_id)
    time.sleep(REQUEST_DELAY)

    resultados = []
    for partido in partidos_filtrados:
        mid, home_name, away_name = partido["id"], partido["home"], partido["away"]
        
        straight = parsear_straight(mid, odds_straight_data)
        
        print(f"     → Buscando props (Córners/Tiros) para: {home_name} vs {away_name}...")
        # 🎯 AQUÍ ESTÁ LA CORRECCIÓN: Le pasamos los 3 datos requeridos
        related = parsear_related(mid, home_name, away_name)

        fecha_dt = datetime.fromisoformat(partido["inicio"].replace("Z", "+00:00"))
        fecha_local = fecha_dt.astimezone()

        fila = {
            "liga": nombre_liga,
            "home": home_name,
            "away": away_name,
            "inicio_local": fecha_local.strftime("%Y-%m-%d %H:%M"),
        }

        for k, v in straight["1x2"].items(): fila[f"1x2_{k}"] = v

        for h in [h for h in straight["handicap"] if not h["es_alt"]][:2]:
            fila[f"hdp_{h['lado']}_{h['handicap']}"] = h["odds"]

        for t in [t for t in straight["total_goles"] if not t["es_alt"]][:4]:
            fila[f"goles_{t['linea']}_{t['lado']}"] = t["odds"]

        for equipo, src in [("home", "tt_home"), ("away", "tt_away")]:
            for t in [t for t in straight[src] if not t["es_alt"]][:2]:
                fila[f"goles_tt_{equipo}_{t['linea']}_{t['lado']}"] = t["odds"]

        for subtipo in ("total", "home", "away"):
            datos = related.get(f"corners_{subtipo}", {})
            if datos:
                for lado in ("over", "under"):
                    if lado in datos: fila[f"corners_{subtipo}_{datos.get('linea', '?')}_{lado}"] = datos[lado]

        for subtipo in ("total", "home", "away"):
            datos = related.get(f"shots_{subtipo}", {})
            if datos:
                for lado in ("over", "under"):
                    if lado in datos: fila[f"shots_{subtipo}_{datos.get('linea', '?')}_{lado}"] = datos[lado]

        resultados.append(fila)
        print(f"     ✓ Partido completado")

    return resultados

# ─────────────────────────────────────────────────────────────────────────────
# MAIN Y EXPORTACIÓN A CARPETA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  🚀 PINNACLE API SCRAPER (Sincronizado con Base de Datos)")
    print("=" * 65)

    locales_db, visitas_db = obtener_partidos_automatizados()
    if not locales_db:
        print("⚠ No hay partidos en tu DB para hoy/mañana. Proceso abortado para no gastar recursos.")
        return []

    hoy = datetime.now()
    todos = []
    
    for nombre_liga, config in LIGAS.items():
        print(f"\n{'─'*55}")
        print(f"  {nombre_liga.upper()}")
        print(f"{'─'*55}")
        try:
            filas = scrapear_liga(nombre_liga, config, locales_db, visitas_db)
            todos.extend(filas)
        except Exception as e:
            print(f"  ❌ Error en {nombre_liga}: {e}")
        time.sleep(REQUEST_DELAY)

    if not todos:
        print("\n⚠ No se capturaron datos para los partidos de tu base de datos.")
        return []

    # --- CREACIÓN DE CARPETA Y GUARDADO ---
    os.makedirs("odds_data", exist_ok=True)
    fecha_str = hoy.strftime("%Y%m%d")

    csv_path = f"odds_data/pinnacle_{fecha_str}.csv"
    df = pd.DataFrame(todos)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*65}")
    print(f"✅ ¡Éxito! {len(todos)} partidos guardados.")
    print(f"📂 Archivo listo para la IA: {csv_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()