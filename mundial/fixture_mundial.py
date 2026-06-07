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

def construir_fixture_mundial_fd():
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
        pass # Si la columna ya existe, SQLite tira un error que ignoramos silenciosamente
        
    conn.commit()

    print("⏳ Conectando con football-data.org para buscar el Mundial 2026...")
    
    # 2. Petición a la API (WC = World Cup, season = 2026)
    url = f"{BASE_URL}/competitions/WC/matches?season=2026"
    res = requests.get(url, headers=HEADERS)
    
    if res.status_code != 200:
        print(f"🛑 Error conectando con la API (Código {res.status_code}).")
        conn.close()
        return

    data = res.json()
    matches = data.get('matches', [])
    
    if not matches:
        print("⚠️ La lista de partidos está vacía. El fixture aún no está programado.")
        conn.close()
        return

    print(f"📅 Se encontraron {len(matches)} partidos. Procesando...")
    partidos_guardados = 0

    # 3. Procesamiento y Extracción de Grupo
    for match in matches:
        utc_date = match.get('utcDate')
        if utc_date:
            fecha = utc_date[:10]
            hora = utc_date[11:16]
        else:
            fecha, hora = "TBA", "TBA"
            
        stage = match.get('stage') or match.get('matchday') or "TBA"
        
        # 🎯 NUEVA EXTRACCIÓN: Sacamos el Grupo de la API
        grupo = match.get('group') or "TBA"
        
        status = match.get('status', 'TBA')
        
        home_node = match.get('homeTeam') or {}
        home_team = home_node.get('name') or "TBA"
        
        away_node = match.get('awayTeam') or {}
        away_team = away_node.get('name') or "TBA"
        
        venue = match.get('venue') or "TBA"
        city = "TBA" 

        # 4. Inyección a base de datos (Ahora incluye la variable 'grupo')
        cursor.execute("""
            INSERT OR REPLACE INTO fixture_mundial 
            (fixture_id, Date, Time, Round, Grupo, HomeTeam, AwayTeam, Venue, City, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match.get('id'), 
            fecha, 
            hora, 
            str(stage), 
            str(grupo), # Inyectamos el grupo
            home_team, 
            away_team, 
            str(venue), 
            city, 
            status
        ))
        partidos_guardados += 1

    conn.commit()
    conn.close()
    
    print(f"✅ ¡Éxito! Se inyectaron/actualizaron {partidos_guardados} partidos con sus GRUPOS en la tabla 'fixture_mundial'.")

if __name__ == "__main__":
    construir_fixture_mundial_fd()