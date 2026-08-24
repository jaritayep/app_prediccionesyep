import sqlite3
import requests
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

DB_PATH = 'database_partidos.db'
TABLE = 'historial_multiliga_ml'

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 10

# Mismos códigos de football-data.co.uk que usa actualizador_database.py,
# mapeados a los slugs de liga que usa ESPN.
LIGAS_ESPN = {
    'E0': 'eng.1',    # Premier League
    'SP1': 'esp.1',   # La Liga
    'I1': 'ita.1',    # Serie A
    'D1': 'ger.1',    # Bundesliga
    'F1': 'fra.1',    # Ligue 1
}


def normalizar_nombre(nombre):
    # Mismo diccionario que actualizador_database.py, para mantener consistencia
    # con los nombres ya guardados en la DB. Los nombres de ESPN suelen venir
    # completos (ej. "Manchester United"), pero por si acaso los pasamos también
    # por este filtro.
    mapeo_especifico = {
        "Nott'm Forest": "Nottingham Forest",
        "Man Utd": "Manchester United",
        "Man United": "Manchester United",
        "Man City": "Manchester City",
        "Ath Bilbao": "Athletic Club",
        "Athletic Bilbao": "Athletic Club",
        "Atl Madrid": "Atletico Madrid",
        "Ath Madrid": "Atletico Madrid",
        "Atletico Madrid": "Atletico Madrid",
        "Atlético Madrid": "Atletico Madrid",
        "Atleti": "Atletico Madrid",
        "Barca": "Barcelona",
        "Barça": "Barcelona",
        "FC Barcelona": "Barcelona",
        "M'gladbach": "Borussia Monchengladbach",
        "M'Gladbach": "Borussia Monchengladbach",
        "Gladbach": "Borussia Monchengladbach",
        "Borussia M.Gladbach": "Borussia Monchengladbach",
        "Paris SG": "PSG",
        "Paris Saint Germain": "PSG",
        "Paris Saint-Germain": "PSG",
    }
    nombre_sucio = nombre.strip()
    return mapeo_especifico.get(nombre_sucio, nombre_sucio)


def get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️ Error fetching {url}: {e}")
        return None


def buscar_stat(stats_list, incluye, excluye=None):
    """Busca en la lista de statistics de ESPN una entrada cuyo nombre contenga
    alguno de los substrings en `incluye` (case-insensitive) y ninguno de `excluye`.
    Devuelve un int, o None si no encontró nada que matchee."""
    excluye = excluye or []
    for s in stats_list:
        texto = f"{s.get('name', '')} {s.get('displayName', '')} {s.get('abbreviation', '')}".lower()
        if any(k in texto for k in incluye) and not any(k in texto for k in excluye):
            val = s.get('displayValue', s.get('value'))
            try:
                return int(float(val))
            except (TypeError, ValueError):
                continue
    return None


def extraer_stats_equipo(stats_list):
    return {
        'shots_total': buscar_stat(stats_list, ['totalshots', 'shots'], excluye=['ontarget', 'target']),
        'shots_on_target': buscar_stat(stats_list, ['shotsontarget', 'shots on target', 'ontarget']),
        'corners': buscar_stat(stats_list, ['corner']),
        'fouls': buscar_stat(stats_list, ['foul']),
        'yellow': buscar_stat(stats_list, ['yellowcard', 'yellow card', 'yellow']),
        'red': buscar_stat(stats_list, ['redcard', 'red card']),
    }


def actualizar_desde_espn(dias_atras=15, pausa_entre_requests=0.3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    hoy = datetime.now()
    partidos_agregados = 0
    partidos_actualizados = 0
    partidos_revisados = 0
    debug_impreso = False  # solo mostramos el JSON crudo de stats una vez, para validar

    print(f"🚀 Iniciando extracción vía ESPN hidden API (últimos {dias_atras} días)...")

    for dias in range(dias_atras):
        fecha = hoy - timedelta(days=dias)
        fecha_espn = fecha.strftime('%Y%m%d')      # formato que pide ESPN
        fecha_str = fecha.strftime('%Y-%m-%d')      # formato que usa la DB

        for codigo, slug in LIGAS_ESPN.items():
            data = get(f"{ESPN_BASE}/{slug}/scoreboard", {"dates": fecha_espn})
            time.sleep(pausa_entre_requests)
            if not data:
                continue

            eventos = data.get("events", [])
            finalizados = [e for e in eventos if e["status"]["type"]["state"] == "post"]

            for ev in finalizados:
                partidos_revisados += 1
                comp = ev["competitions"][0]
                home_c = next(c for c in comp["competitors"] if c["homeAway"] == "home")
                away_c = next(c for c in comp["competitors"] if c["homeAway"] == "away")

                home = normalizar_nombre(home_c["team"]["displayName"])
                away = normalizar_nombre(away_c["team"]["displayName"])

                # Ya no saltamos automáticamente si el partido existe: lo consultamos
                # para decidir después si hace falta un INSERT (partido nuevo) o un
                # UPDATE puntual de HS/AS (partido ya guardado con otro valor).
                cursor.execute(
                    f"SELECT [HS], [AS] FROM {TABLE} WHERE HomeTeam = ? AND AwayTeam = ? AND Date = ?",
                    (home, away, fecha_str)
                )
                fila_existente = cursor.fetchone()

                gl = int(home_c.get("score", 0) or 0)
                gv = int(away_c.get("score", 0) or 0)
                ftr = 'H' if gl > gv else ('A' if gv > gl else 'D')

                # Traer el detalle del partido para las stats (corners, tiros, etc.)
                summary = get(f"{ESPN_BASE}/{slug}/summary", {"event": ev["id"]})
                time.sleep(pausa_entre_requests)
                if not summary:
                    continue

                box_teams = summary.get("boxscore", {}).get("teams", [])
                if len(box_teams) != 2:
                    print(f"⏩ Saltando {home} vs {away} ({fecha_str}) — sin boxscore de ESPN")
                    continue

                # boxscore.teams no siempre viene en orden home/away garantizado,
                # así que lo emparejamos por nombre de equipo.
                stats_por_equipo = {}
                for t in box_teams:
                    nombre_t = t.get("team", {}).get("displayName", "")
                    stats_por_equipo[nombre_t] = t.get("statistics", [])

                stats_home_raw = stats_por_equipo.get(home_c["team"]["displayName"], [])
                stats_away_raw = stats_por_equipo.get(away_c["team"]["displayName"], [])

                if not debug_impreso and (stats_home_raw or stats_away_raw):
                    print("\n🔍 DEBUG — stats crudas de ESPN para el primer partido "
                          "(revisa que buscar_stat() esté mapeando bien):")
                    print(stats_home_raw or stats_away_raw)
                    debug_impreso = True

                sh = extraer_stats_equipo(stats_home_raw)
                sa = extraer_stats_equipo(stats_away_raw)

                # 🛡️ ESCUDO: si ESPN no tiene ninguna stat para ningún equipo,
                # probablemente todavía no cargó el boxscore -> saltar.
                # (chequeamos None, no 0, porque un 0-0 real es válido)
                if (sh['shots_on_target'] is None and sa['shots_on_target'] is None
                        and sh['shots_total'] is None and sa['shots_total'] is None):
                    print(f"⏩ Saltando {home} vs {away} ({fecha_str}) — ESPN sin stats aún")
                    continue

                hst = sh['shots_on_target'] or 0
                ast = sa['shots_on_target'] or 0
                hc = sh['corners'] or 0
                ac = sa['corners'] or 0
                hf = sh['fouls'] or 0
                af = sa['fouls'] or 0
                hy = sh['yellow'] or 0
                ay = sa['yellow'] or 0
                hr = sh['red'] or 0
                ar = sa['red'] or 0

                # Tiros totales sí se recolectan (para poder comparar/actualizar),
                # pero quedan en None si ESPN no los trae, para no inventar un 0 falso.
                hs_nuevo = sh['shots_total']
                as_nuevo = sa['shots_total']

                if fila_existente is None:
                    # Partido nuevo -> INSERT completo
                    cursor.execute(f"""
                        INSERT INTO {TABLE}
                        ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC],
                         [HST], [AST], [HS], [AS], [HF], [AF], [HY], [AY], [HR], [AR])
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hs_nuevo, as_nuevo,
                          hf, af, hy, ay, hr, ar))

                    # Limpiar predicciones futuras para ese partido (anti-leakage,
                    # mismo criterio que actualizador_database.py)
                    cursor.execute("""
                        DELETE FROM tabla_predicciones_limpia
                        WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
                    """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))

                    print(f"✅ Ok (ESPN, nuevo): {fecha_str} | {home} {gl}-{gv} {away} | "
                          f"🎯 Tiros: {hs_nuevo}-{as_nuevo} | 🚩 Corners: {hc}-{ac}")
                    partidos_agregados += 1

                else:
                    # Partido ya existe -> sólo tocamos HS/AS, y sólo si ESPN trae
                    # un valor real (no None) que además sea distinto al guardado.
                    hs_actual, as_actual = fila_existente
                    cambios = {}
                    if hs_nuevo is not None and hs_nuevo != hs_actual:
                        cambios['HS'] = hs_nuevo
                    if as_nuevo is not None and as_nuevo != as_actual:
                        cambios['AS'] = as_nuevo

                    if cambios:
                        set_clause = ", ".join(f"[{col}] = ?" for col in cambios)
                        cursor.execute(f"""
                            UPDATE {TABLE} SET {set_clause}
                            WHERE HomeTeam = ? AND AwayTeam = ? AND Date = ?
                        """, (*cambios.values(), home, away, fecha_str))
                        print(f"🔄 Actualizado (ESPN): {fecha_str} | {home} vs {away} | "
                              f"HS/AS {hs_actual}/{as_actual} -> "
                              f"{cambios.get('HS', hs_actual)}/{cambios.get('AS', as_actual)}")
                        partidos_actualizados += 1

    conn.commit()
    conn.close()
    print(f"\n🏁 ¡Listo! Revisados {partidos_revisados} partidos finalizados. "
          f"Nuevos: {partidos_agregados} | Actualizados (HS/AS): {partidos_actualizados}.")


if __name__ == "__main__":
    actualizar_desde_espn()