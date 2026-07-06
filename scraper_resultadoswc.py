import sqlite3
import requests
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_KEY  = "8940c7a6-d4af-430d-8ec5-42957db2e092"
BASE_URL = "https://api.balldontlie.io/fifa/worldcup/v1"
HEADERS  = {"Authorization": API_KEY}
DB_PATH  = "database_partidos.db"

# Pausa entre requests (segundos).
# Tier GOAT  → 600 req/min → usar 0.12
# Trial GOAT → 5 req/min   → usar 12.5
PAUSA_REQUEST = 0.15


# ─────────────────────────────────────────────
#  COLUMNAS NUEVAS: SPLIT TIEMPO REGULAR / TIEMPO EXTRA
#  (sólo se completan para partidos de fase eliminatoria)
# ─────────────────────────────────────────────
#  Convención:  sufijo _r  = tiempo regular (minutos 1-90, incluye descuento)
#               sufijo _et = tiempo extra   (minutos 91-120)
#
#  Fase            -> nombre de la ronda que devuelve la API (ej. "Round of 16")
#  EsEliminatoria  -> 1 si el partido es de fase eliminatoria, 0 si es de grupos
#  FueProrroga     -> 1 si el partido se definió en (o llegó a) tiempo extra
#  FuePenales      -> 1 si el partido se definió por penales
#  PSO_H / PSO_A   -> goles convertidos en la tanda de penales (NULL si no hubo)
#  FTHG_r / FTAG_r -> goles en tiempo regular (local / visita)
#  FTHG_et/FTAG_et -> goles en tiempo extra   (local / visita)
#  HST_r / AST_r   -> tiros AL ARCO en tiempo regular   ⚠️ best-effort, ver nota
#  HST_et/AST_et   -> tiros AL ARCO en tiempo extra     ⚠️ best-effort, ver nota
#  HS_total_r/AS_total_r   -> remates TOTALES en tiempo regular
#  HS_total_et/AS_total_et -> remates TOTALES en tiempo extra
#  xG_home_r/xG_away_r     -> xG en tiempo regular
#  xG_home_et/xG_away_et   -> xG en tiempo extra
#
#  IMPORTANTE - qué NO se pudo separar y por qué:
#  Faltas, tarjetas, posesión, pases, duelos, etc. sólo existen en la
#  API como UN número agregado de todo el partido (endpoint team_match_stats),
#  sin desglose por período. No hay un endpoint de "eventos de falta/tarjeta"
#  con minuto para poder cortarlos en regular/extra. Por eso esos campos se
#  dejan exactamente como estaban (total del partido) — inventar un split ahí
#  sería peor que no tenerlo, sobre todo alimentando un pipeline de ML.
#
#  🖊️ CORNERS (carga MANUAL): HC_t / AC_t / HC_r / AC_r / HC_et / AC_et
#  Se probó Sofascore para conseguir corners con minuto y bloqueó con 403
#  persistente (ni siquiera cloudscraper lo esquivó), así que esto se carga
#  a mano en vez de scrapearse:
#    HC_t / AC_t   -> corners TOTALES del partido (local/visita)
#    HC_r / AC_r   -> corners en TIEMPO REGULAR
#    HC_et / AC_et -> corners en TIEMPO EXTRA
#  Lo único que hace este script con estas columnas es autocompletar
#  HC_r/AC_r (y HC_et/AC_et=0) cuando el partido NO fue a prórroga, porque
#  ahí "regular" y "total" son lo mismo por definición — ver
#  autocompletar_corners_sin_prorroga() más abajo. Si el partido SÍ fue a
#  prórroga, las seis columnas se cargan a mano.
#
#  El split de tiros/goles/xG sí es real porque:
#   - Los goles por período vienen directo del endpoint /matches
#     (first_half_*, second_half_*, extra_time_*_score, *_score_penalties).
#   - match_shots trae cada disparo individual con su 'time_minute', así que
#     se puede contar/sumar por rango de minuto.
#
#  ⚠️ NOTA sobre tiros AL ARCO (HST_r/AST_r/HST_et/AST_et): para saber si un
#  disparo individual fue "al arco" hay que clasificar el campo 'shot_type'.
#  No pude confirmar el enum exacto contra una respuesta real de la API
#  (este entorno no tiene salida de red hacia api.balldontlie.io), así que
#  uso el criterio más estándar del rubro (gol o atajada = al arco) y agrego
#  un chequeo automático que avisa por consola si el total derivado no
#  coincide con el agregado de la API. Si ves ese aviso seguido, avisame con
#  un JSON de ejemplo de /match_shots y ajusto el criterio.
COLUMNAS_NUEVAS = [
    ("Fase",          "TEXT"),
    ("EsEliminatoria","INTEGER"),
    ("FueProrroga",   "INTEGER"),
    ("FuePenales",    "INTEGER"),
    ("PSO_H",         "INTEGER"),
    ("PSO_A",         "INTEGER"),
    ("FTHG_r",        "INTEGER"),
    ("FTAG_r",        "INTEGER"),
    ("FTHG_et",       "INTEGER"),
    ("FTAG_et",       "INTEGER"),
    ("HST_r",         "INTEGER"),
    ("AST_r",         "INTEGER"),
    ("HST_et",        "INTEGER"),
    ("AST_et",        "INTEGER"),
    ("HS_total_r",    "INTEGER"),
    ("AS_total_r",    "INTEGER"),
    ("HS_total_et",   "INTEGER"),
    ("AS_total_et",   "INTEGER"),
    ("xG_home_r",     "REAL"),
    ("xG_away_r",     "REAL"),
    ("xG_home_et",    "REAL"),
    ("xG_away_et",    "REAL"),
    # Corners — carga MANUAL (ver nota arriba)
    ("HC_t",          "INTEGER"),
    ("AC_t",          "INTEGER"),
    ("HC_r",          "INTEGER"),
    ("AC_r",          "INTEGER"),
    ("HC_et",         "INTEGER"),
    ("AC_et",         "INTEGER"),
]

# Tipos de remate considerados "al arco".
# Confirmado con datos reales (Suiza 2-0 Argelia, match_id 158, Round of 32):
# goal + save reconcilia EXACTO contra shots_on_target del agregado oficial
# en ambos lados (local 2+3=5 vs API=5; visita 0+2=2 vs API=2). El valor
# real de la API es "save" (sin "d" al final) — el typo "saved" era el bug.
TIPOS_A_PUERTA = {"goal", "save"}


def asegurar_columnas_extra(cur):
    """
    🎯 Mismo truco ninja que en fixture_mundial.py: intenta agregar cada
    columna y si ya existe, SQLite tira OperationalError y lo ignoramos.
    Así el script es idempotente sin importar cuántas veces corra.
    Asume que la tabla 'historial_selecciones_ml' ya existe (igual que
    el resto del script original).
    """
    for nombre, tipo in COLUMNAS_NUEVAS:
        try:
            cur.execute(f"ALTER TABLE historial_selecciones_ml ADD COLUMN {nombre} {tipo}")
        except sqlite3.OperationalError:
            pass


def autocompletar_corners_sin_prorroga(cur):
    """
    Los corners son de carga MANUAL (vos cargás HC_t/AC_t directo en la
    base). Esta función sólo hace la parte mecánica: si el partido NO fue
    a tiempo extra, "corners en tiempo regular" es por definición lo mismo
    que el total del partido — no hace falta cargarlo dos veces. También
    deja HC_et/AC_et en 0, porque si no hubo prórroga, no pudo haber
    corners en un período que no existió.

    Corre sobre TODA la tabla (no sólo los partidos de esta corrida), así
    agarra cualquier HC_t/AC_t que hayas cargado a mano desde la última vez
    que corriste el script.

    Si el partido SÍ fue a prórroga (FueProrroga=1), esto no toca nada:
    ahí HC_r/AC_r/HC_et/AC_et los cargás vos a mano, porque "regular" ya
    no es lo mismo que "total" y no hay forma de derivarlo automáticamente.
    """
    cur.execute("""
        UPDATE historial_selecciones_ml
        SET HC_r = HC_t, AC_r = AC_t, HC_et = 0, AC_et = 0
        WHERE (FueProrroga = 0 OR FueProrroga IS NULL)
          AND HC_t IS NOT NULL AND AC_t IS NOT NULL
          AND (HC_r IS NULL OR AC_r IS NULL OR HC_et IS NULL OR AC_et IS NULL)
    """)
    if cur.rowcount:
        print(f"🔁 Autocompleté corners (HC_r/AC_r/HC_et/AC_et) en {cur.rowcount} "
              f"partido(s) sin prórroga, a partir de HC_t/AC_t ya cargados.")


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
    "Cabo Verde":                  "Cape Verde Islands",   # ← nombre real en BDL

    # Europa
    "Czech Republic":              "Czechia",
    "Bosnia & Herzegovina":        "Bosnia-Herzegovina",
    "Bosnia and Herzegovina":      "Bosnia-Herzegovina",
    "Türkiye":                     "Turkey",               # ← BDL usa el nombre oficial turco

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
    No hace falta filtrar por ronda: cada partido trae su propio 'stage' y
    'group', así que la fase eliminatoria se detecta automáticamente más
    abajo (_es_eliminatoria) sin importar si es R16, cuartos, semis o final.
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
    Esto sigue siendo un agregado de TODO el partido (ver nota de columnas
    nuevas más arriba) — corners, faltas, posesión, etc. no se tocan.
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


def obtener_disparos_agregados(match_ids: list) -> dict:
    """
    Descarga match_shots para los IDs dados y agrega, por partido:
      - xG total (como antes, sin cambios de comportamiento)
      - xG, remates totales y remates al arco, separados en
        tiempo regular (minuto <= 90) vs tiempo extra (minuto > 90)

    El corte regular/extra asume que la API no reinicia el reloj en la
    prórroga (convención habitual: minutos 91-120 corridos). No lo pude
    confirmar contra una respuesta real en este entorno (sin acceso de red
    a api.balldontlie.io), así que el chequeo de cuadre en actualizar_mundial()
    va a avisar por consola si esta suposición no encaja con los totales
    que reporta team_match_stats.
    """
    resultado = {}
    sin_minuto = 0

    for i in range(0, len(match_ids), 10):
        batch = match_ids[i:i + 10]
        disparos = _get_paginado(
            f"{BASE_URL}/match_shots",
            params={"match_ids[]": batch, "per_page": 100}
        )
        for disparo in disparos:
            mid = disparo.get("match_id")
            if mid not in resultado:
                resultado[mid] = {
                    "home_xg": 0.0, "away_xg": 0.0,
                    "home_xg_r": 0.0, "away_xg_r": 0.0,
                    "home_xg_et": 0.0, "away_xg_et": 0.0,
                    "home_tiros_r": 0, "away_tiros_r": 0,
                    "home_tiros_et": 0, "away_tiros_et": 0,
                    "home_sot_r": 0, "away_sot_r": 0,
                    "home_sot_et": 0, "away_sot_et": 0,
                }
            r = resultado[mid]
            xg_val  = disparo.get("xg") or 0.0
            es_home = bool(disparo.get("is_home"))
            minuto  = disparo.get("time_minute")
            lado    = "home" if es_home else "away"

            # Total de siempre (comportamiento sin cambios)
            r[f"{lado}_xg"] = round(r[f"{lado}_xg"] + xg_val, 4)

            if minuto is None:
                sin_minuto += 1
                continue  # no se puede clasificar en regular/extra

            sufijo = "et" if minuto > 90 else "r"
            r[f"{lado}_xg_{sufijo}"]    = round(r[f"{lado}_xg_{sufijo}"] + xg_val, 4)
            r[f"{lado}_tiros_{sufijo}"] += 1
            if disparo.get("shot_type") in TIPOS_A_PUERTA:
                r[f"{lado}_sot_{sufijo}"] += 1

        time.sleep(PAUSA_REQUEST)

    if sin_minuto:
        print(f"  ⚠️  {sin_minuto} remate(s) sin 'time_minute' — no se pudieron ubicar en regular/prórroga.")

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


def _resultado(gl: int, gv: int, pen_l=None, pen_v=None) -> str:
    """
    H/A/D. OJO: en eliminatoria definida por penales, el marcador de
    90'+prórroga puede quedar empatado (gl == gv) pero SIEMPRE hay un
    ganador real — por eso se miran los penales antes de declarar empate.
    """
    if pen_l is not None and pen_v is not None:
        return "H" if pen_l > pen_v else "A"
    if gl > gv:
        return "H"
    elif gv > gl:
        return "A"
    return "D"


def _es_eliminatoria(p: dict) -> bool:
    """
    Un partido es de fase eliminatoria si no pertenece a ningún grupo.
    En la API, los partidos de grupos traen 'group': {...} y los de
    eliminatoria (R16, cuartos, semis, final, tercer puesto) traen
    'group': null. Esto funciona para cualquier ronda futura sin tocar
    el código.
    """
    return p.get('group') is None


def _goles_regulacion_y_et(p: dict):
    """
    Separa los goles de un partido en tiempo regular vs tiempo extra.
    Devuelve (gl_reg, gv_reg, gl_et, gv_et).
    Prioriza first_half + second_half (más explícito); si faltan, resta
    el marcador de prórroga al marcador final.
    """
    gl_final = p.get('home_score') or 0
    gv_final = p.get('away_score') or 0

    fh_l, sh_l = p.get('first_half_home_score'), p.get('second_half_home_score')
    fh_v, sh_v = p.get('first_half_away_score'), p.get('second_half_away_score')

    gl_reg = (fh_l + sh_l) if (fh_l is not None and sh_l is not None) else None
    gv_reg = (fh_v + sh_v) if (fh_v is not None and sh_v is not None) else None

    et_l = p.get('extra_time_home_score')
    et_v = p.get('extra_time_away_score')

    if gl_reg is None:
        gl_reg = gl_final - (et_l or 0)
    if gv_reg is None:
        gv_reg = gv_final - (et_v or 0)

    return gl_reg, gv_reg, (et_l or 0), (et_v or 0)


def _avisar_si_descuadra(etiqueta: str, total_api: int, derivado: int, contexto: str, tolerancia: int = 1):
    """
    Aviso de consola si el split derivado no reconcilia con el agregado de la API.
    Diferencias chicas en REMATES TOTALES son ruido normal entre team_match_stats
    y el shot log del proveedor — confirmado empíricamente en un partido de
    control (Suiza-Argelia, match_id 158): sin prórroga ni penales de por medio,
    igual el shot log trajo 13 remates locales contra 11 del agregado oficial.
    Por eso esa comparación usa una tolerancia más alta. TIROS AL ARCO sí debería
    cuadrar exacto (ya confirmado con ese mismo partido), así que ahí la
    tolerancia se mantiene en 1 para que un descuadre real siga avisando.
    """
    diferencia = abs(derivado - total_api)
    if total_api and diferencia > tolerancia:
        print(f"  ⚠️  Descuadre en {etiqueta} ({contexto}): "
              f"agregado API={total_api} vs regular+extra derivado={derivado} "
              f"(diferencia de {diferencia})")


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
        asegurar_columnas_extra(cur)
        conn.commit()

        autocompletar_corners_sin_prorroga(cur)
        conn.commit()

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
            es_elim = _es_eliminatoria(p)

            cur.execute("""
                SELECT HST, AST, HC, AC, xG_home, xG_away, FTHG_r, HST_r
                FROM historial_selecciones_ml
                WHERE Date=? AND HomeTeam=? AND AwayTeam=? AND Torneo='FIFA World Cup 2026'
            """, (fecha, home, away))
            existente = cur.fetchone()

            if existente:
                hst_g, ast_g, hc_g, ac_g, xgh_g, xga_g, fthg_r_g, hst_r_g = existente
                tiene_stats_base = any([hst_g, ast_g, hc_g, ac_g, xgh_g, xga_g])
                # Si es eliminatoria y todavía no tiene el split regular/extra,
                # hay que reprocesarlo aunque ya tenga las stats base — esto es
                # lo que permite "destapar" partidos que ya estaban guardados
                # desde antes de que este split existiera (p. ej. octavos que
                # ya corriste con la versión vieja del script).
                le_falta_split = es_elim and fthg_r_g is None and hst_r_g is None
                if tiene_stats_base and not le_falta_split:
                    continue
                p["_modo"] = "update"
            else:
                p["_modo"] = "insert"

            p["_home"]    = home
            p["_away"]    = away
            p["_fecha"]   = fecha
            p["_es_elim"] = es_elim
            a_procesar.append(p)

        if not a_procesar:
            print("✅ La DB ya está al día. No hay nada nuevo que procesar.")
            return

        n_elim = sum(1 for p in a_procesar if p["_es_elim"])
        print(f"\n📊 {len(a_procesar)} partidos a procesar "
              f"({sum(1 for p in a_procesar if p['_modo']=='insert')} nuevos, "
              f"{sum(1 for p in a_procesar if p['_modo']=='update')} a actualizar, "
              f"{n_elim} de fase eliminatoria)")

        # ── 3. Descargar stats y disparos en batch ──────────────
        ids = [p["id"] for p in a_procesar]

        print("\n📈 Descargando estadísticas de equipo...")
        stats_equipos = obtener_stats_equipos(ids)

        print("🎯 Descargando y clasificando disparos (xG, regular/prórroga)...")
        disparos_agg = obtener_disparos_agregados(ids)

        # ── 4. Insertar / actualizar ─────────────────────────────
        guardados = 0
        print()

        for p in a_procesar:
            mid   = p["id"]
            home  = p["_home"]
            away  = p["_away"]
            fecha = p["_fecha"]
            modo  = p["_modo"]
            es_elim = p["_es_elim"]

            gl  = p.get("home_score") or 0
            gv  = p.get("away_score") or 0
            pen_l = p.get("home_score_penalties")
            pen_v = p.get("away_score_penalties")
            ftr = _resultado(gl, gv, pen_l, pen_v)

            # Stats de equipo (agregado de todo el partido, sin cambios)
            se    = stats_equipos.get(mid, {})
            h_s   = se.get("home") or {}
            a_s   = se.get("away") or {}

            # shots_on_target tiene varios nombres posibles según la versión de la API
            hst = _int(h_s, "shots_on_target", "shots_on_goal", "shotsOnTarget")
            ast = _int(a_s, "shots_on_target", "shots_on_goal", "shotsOnTarget")
            hc  = _int(h_s, "corners", "corner_kicks", "cornerKicks")
            ac  = _int(a_s, "corners", "corner_kicks", "cornerKicks")

            # xG (total del partido, sin cambios)
            disp   = disparos_agg.get(mid, {})
            xg_h   = round(disp.get("home_xg", 0.0), 4)
            xg_a   = round(disp.get("away_xg", 0.0), 4)

            # 🛡️ ESCUDO: si todo sigue en cero, la API no subió los stats aún
            if hst == 0 and ast == 0 and xg_h == 0.0 and xg_a == 0.0:
                print(f"⏩ Saltando {home} vs {away} ({fecha}) — stats no disponibles aún")
                continue

            # Nombre de la ronda (informativo, se guarda para todos los partidos)
            stage_obj = p.get('stage')
            fase = stage_obj.get('name') if isinstance(stage_obj, dict) else (str(stage_obj) if stage_obj else "TBA")

            if es_elim:
                fthg_r, ftag_r, fthg_et, ftag_et = _goles_regulacion_y_et(p)

                hst_r,  ast_r  = disp.get("home_sot_r", 0),   disp.get("away_sot_r", 0)
                hst_et, ast_et = disp.get("home_sot_et", 0),  disp.get("away_sot_et", 0)
                h_tiros_r,  a_tiros_r  = disp.get("home_tiros_r", 0),  disp.get("away_tiros_r", 0)
                h_tiros_et, a_tiros_et = disp.get("home_tiros_et", 0), disp.get("away_tiros_et", 0)
                xg_h_r,  xg_a_r  = disp.get("home_xg_r", 0.0),  disp.get("away_xg_r", 0.0)
                xg_h_et, xg_a_et = disp.get("home_xg_et", 0.0), disp.get("away_xg_et", 0.0)

                fue_prorroga = (
                    p.get('extra_time_home_score') is not None or
                    p.get('extra_time_away_score') is not None or
                    pen_l is not None or pen_v is not None
                )
                fue_penales = pen_l is not None or pen_v is not None

                # Chequeo de cuadre: si esto avisa seguido, la suposición del
                # minuto 90 como corte y/o el criterio de "al arco" no encajan
                # con esta API — mandame un JSON de /match_shots y lo ajusto.
                ctx = f"{home} vs {away} ({fecha})"
                _avisar_si_descuadra("remates totales (local)",   _int(h_s, "shots_total"), h_tiros_r + h_tiros_et, ctx, tolerancia=3)
                _avisar_si_descuadra("remates totales (visita)",  _int(a_s, "shots_total"), a_tiros_r + a_tiros_et, ctx, tolerancia=3)
                _avisar_si_descuadra("tiros al arco (local)",     hst, hst_r + hst_et, ctx, tolerancia=1)
                _avisar_si_descuadra("tiros al arco (visita)",    ast, ast_r + ast_et, ctx, tolerancia=1)
            else:
                fthg_r = ftag_r = fthg_et = ftag_et = None
                hst_r = ast_r = hst_et = ast_et = None
                h_tiros_r = a_tiros_r = h_tiros_et = a_tiros_et = None
                xg_h_r = xg_a_r = xg_h_et = xg_a_et = None
                fue_prorroga = False
                fue_penales  = False

            torneo = "FIFA World Cup 2026"

            campos = {
                "FTHG": gl, "FTAG": gv, "FTR": ftr,
                "HST": hst, "AST": ast, "HC": hc, "AC": ac,
                "xG_home": xg_h, "xG_away": xg_a,
                "Fase": fase, "EsEliminatoria": int(es_elim),
                "FueProrroga": int(fue_prorroga), "FuePenales": int(fue_penales),
                "PSO_H": pen_l, "PSO_A": pen_v,
                "FTHG_r": fthg_r, "FTAG_r": ftag_r,
                "FTHG_et": fthg_et, "FTAG_et": ftag_et,
                "HST_r": hst_r, "AST_r": ast_r,
                "HST_et": hst_et, "AST_et": ast_et,
                "HS_total_r": h_tiros_r, "AS_total_r": a_tiros_r,
                "HS_total_et": h_tiros_et, "AS_total_et": a_tiros_et,
                "xG_home_r": xg_h_r, "xG_away_r": xg_a_r,
                "xG_home_et": xg_h_et, "xG_away_et": xg_a_et,
            }

            if modo == "update":
                set_clause = ", ".join(f"{col}=?" for col in campos)
                cur.execute(
                    f"UPDATE historial_selecciones_ml SET {set_clause} "
                    f"WHERE Date=? AND HomeTeam=? AND AwayTeam=? AND Torneo=?",
                    (*campos.values(), fecha, home, away, torneo)
                )
                emoji_accion = "🔄"
            else:
                columnas = ["Date", "Torneo", "HomeTeam", "AwayTeam", *campos.keys()]
                placeholders = ", ".join("?" for _ in columnas)
                cur.execute(
                    f"INSERT INTO historial_selecciones_ml ({', '.join(columnas)}) VALUES ({placeholders})",
                    (fecha, torneo, home, away, *campos.values())
                )
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

            extra_info = ""
            if es_elim:
                if fue_penales:
                    extra_info = f" [penales {pen_l}-{pen_v}]"
                elif fue_prorroga:
                    extra_info = " [prórroga]"

            print(f"{emoji_accion} {fecha} | {home} {gl}-{gv} {away}{extra_info} "
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