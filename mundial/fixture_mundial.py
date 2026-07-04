import sys
import sqlite3
import requests

# ==========================================
# 1. CONFIGURACIÓN DE FOOTBALL-DATA.ORG
# ==========================================
API_TOKEN = "c81aa18fa4974dda90812a83f1aec599"  # 🔴 Reemplaza con tu token
BASE_URL = "https://api.football-data.org/v4"

HEADERS = {
    "X-Auth-Token": API_TOKEN
}

# 🎯 Mapeo de alias amigables -> códigos de "stage" que usa la API.
# Así el script sirve para CUALQUIER ronda futura, no solo R16.
# Podés agregar más alias a la izquierda si querés escribirlos distinto.
ROUND_ALIASES = {
    "group": "GROUP_STAGE",
    "grupos": "GROUP_STAGE",
    "group_stage": "GROUP_STAGE",
    "r16": "LAST_16",
    "ro16": "LAST_16",
    "round_of_16": "LAST_16",
    "last_16": "LAST_16",
    "octavos": "LAST_16",
    "qf": "QUARTER_FINALS",
    "quarterfinals": "QUARTER_FINALS",
    "quarter_finals": "QUARTER_FINALS",
    "cuartos": "QUARTER_FINALS",
    "sf": "SEMI_FINALS",
    "semifinals": "SEMI_FINALS",
    "semi_finals": "SEMI_FINALS",
    "semis": "SEMI_FINALS",
    "third_place": "THIRD_PLACE",
    "tercer_puesto": "THIRD_PLACE",
    "final": "FINAL",
}


def resolver_stage(nombre_ronda):
    """Traduce lo que escribe el usuario (ej: 'r16') al código de stage de la API."""
    if nombre_ronda is None:
        return None
    clave = nombre_ronda.strip().lower()
    if clave in ROUND_ALIASES:
        return ROUND_ALIASES[clave]
    # Si ya vino en formato de la API (ej: "LAST_16"), lo dejamos pasar tal cual
    return nombre_ronda.strip().upper()


def construir_fixture_mundial_fd(stage=None):
    """
    Si 'stage' es None, trae TODO el fixture (comportamiento original).
    Si 'stage' tiene un valor (ej: 'LAST_16'), sólo trae y actualiza esa ronda,
    reemplazando los partidos que estaban en TBA por los datos reales.
    """
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()

    # 1. Crear tabla base (si no existe)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fixture_mundial (
        fixture_id INTEGER PRIMARY KEY,
        Date TEXT,
        Time TEXT,
        Round TEXT,
        HomeTeam TEXT,
        AwayTeam TEXT,
        Venue TEXT,
        City TEXT,
        Status TEXT
    )
    ''')

    # 🎯 TRUCO NINJA: Añadir la columna 'Grupo' dinámicamente si no existía antes
    try:
        cursor.execute("ALTER TABLE fixture_mundial ADD COLUMN Grupo TEXT")
    except sqlite3.OperationalError:
        pass  # Si la columna ya existe, SQLite tira un error que ignoramos silenciosamente

    conn.commit()

    # 2. Petición a la API (WC = World Cup, season = 2026)
    url = f"{BASE_URL}/competitions/WC/matches?season=2026"
    if stage:
        url += f"&stage={stage}"
        print(f"⏳ Conectando con football-data.org para buscar la ronda '{stage}' del Mundial 2026...")
    else:
        print("⏳ Conectando con football-data.org para buscar el Mundial 2026 (todas las rondas)...")

    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        print(f"🛑 Error conectando con la API (Código {res.status_code}).")
        conn.close()
        return

    data = res.json()
    matches = data.get('matches', [])

    if not matches:
        print("⚠️ La lista de partidos está vacía. Esa ronda todavía no está programada o el nombre de stage no existe.")
        conn.close()
        return

    print(f"📅 Se encontraron {len(matches)} partidos. Procesando...")
    partidos_guardados = 0
    partidos_actualizados_desde_tba = 0

    # 3. Procesamiento y Extracción de Grupo
    for match in matches:
        fixture_id = match.get('id')

        utc_date = match.get('utcDate')
        if utc_date:
            fecha = utc_date[:10]
            hora = utc_date[11:16]
        else:
            fecha, hora = "TBA", "TBA"

        round_stage = match.get('stage') or match.get('matchday') or "TBA"

        # 🎯 Extracción de Grupo (si aplica, en fases eliminatorias suele venir vacío)
        grupo = match.get('group') or "TBA"

        status = match.get('status', 'TBA')

        home_node = match.get('homeTeam') or {}
        home_team = home_node.get('name') or "TBA"

        away_node = match.get('awayTeam') or {}
        away_team = away_node.get('name') or "TBA"

        venue = match.get('venue') or "TBA"
        city = "TBA"

        # 🎯 Revisamos si ya existía una fila TBA para este partido, para poder
        # informar cuántos partidos "se destaparon" en esta corrida.
        cursor.execute(
            "SELECT HomeTeam, AwayTeam, Status FROM fixture_mundial WHERE fixture_id = ?",
            (fixture_id,)
        )
        fila_previa = cursor.fetchone()
        era_tba = fila_previa is not None and (
            fila_previa[0] == "TBA" or fila_previa[1] == "TBA" or fila_previa[2] == "TBA"
        )
        ahora_tiene_datos = home_team != "TBA" and away_team != "TBA"
        if era_tba and ahora_tiene_datos:
            partidos_actualizados_desde_tba += 1

        # 4. Inyección/actualización en base de datos (upsert por fixture_id).
        # Esto reemplaza automáticamente cualquier fila anterior en TBA para
        # ese mismo fixture_id, sea de la ronda que sea (R16, cuartos, etc.).
        cursor.execute("""
            INSERT OR REPLACE INTO fixture_mundial
            (fixture_id, Date, Time, Round, Grupo, HomeTeam, AwayTeam, Venue, City, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id,
            fecha,
            hora,
            str(round_stage),
            str(grupo),
            home_team,
            away_team,
            str(venue),
            city,
            status
        ))
        partidos_guardados += 1

    conn.commit()
    conn.close()

    print(f"✅ ¡Éxito! Se inyectaron/actualizaron {partidos_guardados} partidos en la tabla 'fixture_mundial'.")
    if partidos_actualizados_desde_tba:
        print(f"🔁 De esos, {partidos_actualizados_desde_tba} pasaron de TBA a equipos confirmados.")


if __name__ == "__main__":
    # Uso:
    #   python fixture_mundial.py            -> trae TODO el fixture (comportamiento original)
    #   python fixture_mundial.py r16        -> sólo trae/actualiza Octavos de Final
    #   python fixture_mundial.py qf         -> sólo trae/actualiza Cuartos de Final
    #   python fixture_mundial.py sf         -> Semifinales
    #   python fixture_mundial.py final      -> Final
    #   python fixture_mundial.py LAST_16    -> también acepta el código crudo de la API
    ronda_arg = sys.argv[1] if len(sys.argv) > 1 else None
    stage_resuelto = resolver_stage(ronda_arg)
    construir_fixture_mundial_fd(stage=stage_resuelto)