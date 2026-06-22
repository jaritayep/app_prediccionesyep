import sqlite3
import requests
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_KEY  = "91b6393a-f882-4714-b2ab-26d2c6d77a1c"
BASE_URL = "https://api.balldontlie.io/fifa/worldcup/v1"
HEADERS  = {"Authorization": API_KEY}
DB_PATH  = "database_partidos.db"

# Pausa entre requests (segundos).
# Tier GOAT  → 600 req/min → usar 0.12
# Trial GOAT → 5 req/min   → usar 12.5
PAUSA_REQUEST = 0.15


# ─────────────────────────────────────────────
#  TRADUCTOR DE NOMBRES (BallDontLie → DB)
# ─────────────────────────────────────────────
MAPEO_EQUIPOS = {
    # CONCACAF
    "United States": "United States",
    "USA":           "United States",

    # África
    "DR Congo":                    "Congo DR",
    "Democratic Republic of Congo":"Congo DR",
    "Côte d'Ivoire":               "Ivory Coast",
    "Cote d'Ivoire":               "Ivory Coast",
    "Cape Verde":                  "Cape Verde Islands",
    "Cape Verde Islands":          "Cape Verde Islands",

    # Europa
    "Czech Republic":              "Czechia",
    "Bosnia & Herzegovina":        "Bosnia-Herzegovina",
    "Bosnia and Herzegovina":      "Bosnia-Herzegovina",

    # Asia
    "Korea Republic":              "South Korea",
    "Republic of Korea":           "South Korea",
    "South Korea":                 "South Korea",

    # CONMEBOL (ya suelen coincidir, pero por si acaso)
    "Curacao": "Curaçao",
}

def normalizar_equipo(nombre: str) -> str:
    """Traduce el nombre que devuelve la API al formato que usa la DB."""
    return MAPEO_EQUIPOS.get(nombre, nombre)


# ─────────────────────────────────────────────
#  HTTP CON REINTENTOS
# ─────────────────────────────────────────────
def _get(url: str, params: dict = None, max_reintentos: int = 3):
    """GET con reintentos automáticos. Devuelve dict JSON o None si falla."""
    for intento in range(max_reintentos):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                espera = 65
                print(f"  ⏳ Rate limit. Esperando {espera}s...")
                time.sleep(espera)
                continue
            if resp.status_code == 401:
                print("  🔑 Error 401: API key inválido o suscripción insuficiente (necesitas GOAT).")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Intento {intento + 1}/{max_reintentos} fallido: {e}")
            if intento < max_reintentos - 1:
                time.sleep(5)
    return None


def _get_paginado(url: str, params: dict) -> list:
    """Descarga todos los registros de un endpoint paginado con cursor."""
    todos = []
    cursor = None
    while True:
        p = {**params}
        if cursor:
            p["cursor"] = cursor
        data = _get(url, params=p)
        if not data:
            break
        todos.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(PAUSA_REQUEST)
    return todos


# ─────────────────────────────────────────────
#  CARGA DE DATOS DESDE LA API
# ─────────────────────────────────────────────
def obtener_partidos_completados() -> list:
    """
    Devuelve todos los partidos del Mundial 2026 con status='completed'.
    """
    print("📡 Descargando partidos del Mundial 2026...")
    partidos = _get_paginado(
        f"{BASE_URL}/matches",
        params={"seasons[]": 2026, "per_page": 100}
    )
    completados = [p for p in partidos if p.get("status") == "completed"
                   and p.get("home_team") and p.get("away_team")]
    print(f"   → {len(completados)} completados de {len(partidos)} totales")
    return completados


def obtener_stats_equipos(match_ids: list) -> dict:
    """
    Descarga team_match_stats para los IDs dados.
    Devuelve  {match_id: {"home": {...}, "away": {...}}}
    """
    resultado = {}
    # Procesar en batches de 10 para no generar URLs gigantes
    for i in range(0, len(match_ids), 10):
        batch = match_ids[i:i + 10]
        registros = _get_paginado(
            f"{BASE_URL}/team_match_stats",
            params={"match_ids[]": batch, "per_page": 100}
        )
        for stat in registros:
            mid = stat.get("match_id")
            if mid not in resultado:
                resultado[mid] = {"home": None, "away": None}
            if stat.get("is_home"):
                resultado[mid]["home"] = stat
            else:
                resultado[mid]["away"] = stat
        time.sleep(PAUSA_REQUEST)
    return resultado


def obtener_xg_partidos(match_ids: list) -> dict:
    """
    Agrega xG de todos los disparos por partido.
    Devuelve  {match_id: {"home_xg": float, "away_xg": float}}
    """
    resultado = {}
    for i in range(0, len(match_ids), 10):
        batch = match_ids[i:i + 10]
        disparos = _get_paginado(
            f"{BASE_URL}/match_shots",
            params={"match_ids[]": batch, "per_page": 100}
        )
        for disparo in disparos:
            mid = disparo.get("match_id")
            if mid not in resultado:
                resultado[mid] = {"home_xg": 0.0, "away_xg": 0.0}
            xg_val = disparo.get("xg") or 0.0
            if disparo.get("is_home"):
                resultado[mid]["home_xg"] = round(resultado[mid]["home_xg"] + xg_val, 4)
            else:
                resultado[mid]["away_xg"] = round(resultado[mid]["away_xg"] + xg_val, 4)
        time.sleep(PAUSA_REQUEST)
    return resultado


# ─────────────────────────────────────────────
#  HELPERS DE STATS
# ─────────────────────────────────────────────
def _int(d: dict, *claves) -> int:
    """Extrae el primer campo no-None de las claves dadas, como int."""
    for c in claves:
        v = d.get(c)
        if v is not None:
            return int(v)
    return 0


def _resultado(gl: int, gv: int) -> str:
    if gl > gv:
        return "H"
    elif gv > gl:
        return "A"
    return "D"


# ─────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────
def actualizar_mundial():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    print("=" * 60)
    print("🚀 Actualizador Mundial 2026 — BallDontLie API")
    print("=" * 60)

    try:
        # ── 1. Partidos completados ──────────────────────────────
        partidos = obtener_partidos_completados()
        if not partidos:
            print("⚠️  No hay partidos completados aún.")
            return

        # ── 2. Clasificar: nuevo / actualizar / saltar ───────────
        a_procesar = []
        for p in partidos:
            home = normalizar_equipo(p["home_team"]["name"])
            away = normalizar_equipo(p["away_team"]["name"])
            fecha = p["datetime"][:10]          # "2026-06-11"

            cur.execute("""
                SELECT HST, AST, HC, AC, xG_home, xG_away
                FROM historial_selecciones_ml
                WHERE Date=? AND HomeTeam=? AND AwayTeam=? AND Torneo='FIFA World Cup 2026'
            """, (fecha, home, away))
            existente = cur.fetchone()

            if existente:
                hst, ast, hc, ac, xgh, xga = existente
                # Si ya tiene estadísticas reales → nada que hacer
                if any([hst, ast, hc, ac, xgh, xga]):
                    continue
                # Existe pero stats = 0 → intentar rellenar
                p["_modo"] = "update"
            else:
                p["_modo"] = "insert"

            p["_home"] = home
            p["_away"] = away
            p["_fecha"] = fecha
            a_procesar.append(p)

        if not a_procesar:
            print("✅ La DB ya está al día. No hay nada nuevo que procesar.")
            return

        print(f"\n📊 {len(a_procesar)} partidos a procesar "
              f"({sum(1 for p in a_procesar if p['_modo']=='insert')} nuevos, "
              f"{sum(1 for p in a_procesar if p['_modo']=='update')} a actualizar)")

        # ── 3. Descargar stats y xG en batch ────────────────────
        ids = [p["id"] for p in a_procesar]

        print("\n📈 Descargando estadísticas de equipo...")
        stats_equipos = obtener_stats_equipos(ids)

        print("🎯 Descargando datos xG (disparos)...")
        xg_data = obtener_xg_partidos(ids)

        # ── 4. Insertar / actualizar ─────────────────────────────
        guardados = 0
        print()

        for p in a_procesar:
            mid   = p["id"]
            home  = p["_home"]
            away  = p["_away"]
            fecha = p["_fecha"]
            modo  = p["_modo"]

            gl  = p.get("home_score") or 0
            gv  = p.get("away_score") or 0
            ftr = _resultado(gl, gv)

            # Stats de equipo
            se    = stats_equipos.get(mid, {})
            h_s   = se.get("home") or {}
            a_s   = se.get("away") or {}

            # shots_on_target tiene varios nombres posibles según la versión de la API
            hst = _int(h_s, "shots_on_target", "shots_on_goal", "shotsOnTarget")
            ast = _int(a_s, "shots_on_target", "shots_on_goal", "shotsOnTarget")
            hc  = _int(h_s, "corners", "corner_kicks", "cornerKicks")
            ac  = _int(a_s, "corners", "corner_kicks", "cornerKicks")

            # xG
            xg     = xg_data.get(mid, {})
            xg_h   = round(xg.get("home_xg", 0.0), 4)
            xg_a   = round(xg.get("away_xg", 0.0), 4)

            # 🛡️ ESCUDO: si todo sigue en cero, la API no subió los stats aún
            if hst == 0 and ast == 0 and xg_h == 0.0 and xg_a == 0.0:
                print(f"⏩ Saltando {home} vs {away} ({fecha}) — stats no disponibles aún")
                continue

            torneo = "FIFA World Cup 2026"

            if modo == "update":
                cur.execute("""
                    UPDATE historial_selecciones_ml
                    SET FTHG=?, FTAG=?, FTR=?, HST=?, AST=?, HC=?, AC=?,
                        xG_home=?, xG_away=?
                    WHERE Date=? AND HomeTeam=? AND AwayTeam=? AND Torneo=?
                """, (gl, gv, ftr, hst, ast, hc, ac, xg_h, xg_a,
                      fecha, home, away, torneo))
                emoji_accion = "🔄"
            else:
                cur.execute("""
                    INSERT INTO historial_selecciones_ml
                    (Date, Torneo, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
                     HST, AST, HC, AC, xG_home, xG_away)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fecha, torneo, home, away, gl, gv, ftr,
                      hst, ast, hc, ac, xg_h, xg_a))
                emoji_accion = "✅"

            # Marcar fixture como FINISHED
            cur.execute("""
                UPDATE fixture_mundial
                SET Status = 'FINISHED'
                WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?
            """, (fecha, home, away))

            # Anti-leakage: eliminar predicciones del partido ya jugado
            cur.execute("""
                DELETE FROM tabla_predicciones_limpia
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha))

            print(f"{emoji_accion} {fecha} | {home} {gl}-{gv} {away} "
                  f"| SOT: {hst}-{ast} | Córners: {hc}-{ac} "
                  f"| xG: {xg_h:.2f}-{xg_a:.2f}")
            guardados += 1

        conn.commit()
        print(f"\n🏁 ¡Listo! {guardados} partidos procesados.")

    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    actualizar_mundial()