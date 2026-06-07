import sqlite3
import pandas as pd
import requests
import time
from thefuzz import fuzz

# ==========================================
# 1. CONFIGURACIÓN DE SPORTMONKS (v3)
# ==========================================
API_TOKEN = "L7guq7bKFrGy5wmXW5mEW8FCnipMFwexMi7K774Dv66pcp0kuAaw99TWmDyn" 
BASE_URL = "https://api.sportmonks.com/v3/football"

# Headers estándar para Sportmonks
HEADERS = {
    "Authorization": API_TOKEN,
    "Accept": "application/json"
}

def enriquecer_con_sportmonks():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    # 1. Extraemos SOLO los partidos vacíos usando el rowid invisible
    query_pendientes = """
        SELECT rowid, Date, Torneo, HomeTeam, AwayTeam 
        FROM historial_selecciones_ml 
        WHERE (HST IS NULL OR HST = 0.0 OR xG_home IS NULL)
    """
    df_pendientes = pd.read_sql(query_pendientes, conn)
    
    if df_pendientes.empty:
        print("✅ Tu base de datos ya está 100% llena. No hay datos nulos.")
        conn.close()
        return
        
    print(f"🔍 Se encontraron {len(df_pendientes)} partidos incompletos.")
    
    # Agrupamos por fecha para ahorrar peticiones a la API
    fechas_unicas = df_pendientes['Date'].unique()
    print(f"📅 Optimizando: Se harán solo {len(fechas_unicas)} llamadas a la API (una por día).")

    actualizados = 0

    # 2. MOTOR DE BÚSQUEDA POR FECHA
    for fecha in fechas_unicas:
        print(f"⏳ Descargando jornada del {fecha}...")
        
        # Endpoint v3: Busca partidos por fecha e INCLUYE participantes y estadísticas
        url = f"{BASE_URL}/fixtures/date/{fecha}?include=participants;statistics.type"
        
        res = requests.get(url, headers=HEADERS)
        
        if res.status_code == 429:
            print("🛑 Límite de peticiones de Sportmonks alcanzado. Pausando 60 segundos...")
            time.sleep(60)
            res = requests.get(url, headers=HEADERS) # Reintento
            
        if res.status_code != 200:
            print(f"⚠️ Error {res.status_code} en la fecha {fecha}: {res.text[:100]}")
            continue

        data_api = res.json().get('data', [])
        
        if not data_api:
            continue

        # Filtramos los partidos pendientes de nuestra DB que corresponden a esta fecha
        partidos_del_dia = df_pendientes[df_pendientes['Date'] == fecha]

        # 3. CRUCE DE DATOS (Fuzzy Matching)
        for _, fila_db in partidos_del_dia.iterrows():
            id_partido = fila_db['rowid']
            home_db = fila_db['HomeTeam']
            away_db = fila_db['AwayTeam']
            
            # Buscamos el partido exacto en la respuesta de la API
            match_encontrado = None
            for fixture in data_api:
                equipos = fixture.get('participants', [])
                if len(equipos) == 2:
                    # Sportmonks indica quién es local/visita en meta.location
                    equipo_1 = equipos[0]['name']
                    equipo_2 = equipos[1]['name']
                    
                    # Verificamos similitud de ambos equipos cruzados para asegurar que es el mismo partido
                    sim_home = max(fuzz.partial_ratio(home_db, equipo_1), fuzz.partial_ratio(home_db, equipo_2))
                    sim_away = max(fuzz.partial_ratio(away_db, equipo_1), fuzz.partial_ratio(away_db, equipo_2))
                    
                    if sim_home > 75 and sim_away > 75:
                        match_encontrado = fixture
                        break
            
            # Si encontramos el partido, extraemos las stats
            if match_encontrado:
                estadisticas = match_encontrado.get('statistics', [])
                participantes = match_encontrado.get('participants', [])
                
                # Identificamos los IDs internos de Sportmonks para saber de quién es cada estadística
                home_id, away_id = None, None
                for p in participantes:
                    if p.get('meta', {}).get('location') == 'home':
                        home_id = p['id']
                    elif p.get('meta', {}).get('location') == 'away':
                        away_id = p['id']

                # Variables por defecto
                hst, ast, hc, ac, xg_home, xg_away = 0, 0, 0, 0, 0.0, 0.0
                
                # Recorremos todas las estadísticas del partido
                for stat in estadisticas:
                    tipo_stat = stat.get('type', {}).get('name', '')
                    valor = stat.get('data', {}).get('value', 0)
                    pid = stat.get('participant_id')

                    # Clasificamos según el tipo de estadística y el equipo
                    if tipo_stat == 'Shots On Target':
                        if pid == home_id: hst = valor
                        elif pid == away_id: ast = valor
                    elif tipo_stat == 'Corners':
                        if pid == home_id: hc = valor
                        elif pid == away_id: ac = valor
                    elif tipo_stat == 'Expected Goals':
                        if pid == home_id: xg_home = valor
                        elif pid == away_id: xg_away = valor

                # 4. INYECCIÓN QUIRÚRGICA A SQLITE
                # Solo inyectamos si logramos extraer al menos una estadística clave
                if hc > 0 or hst > 0 or xg_home > 0:
                    cursor.execute("""
                        UPDATE historial_selecciones_ml 
                        SET xG_home = ?, xG_away = ?, HST = ?, AST = ?, HC = ?, AC = ?
                        WHERE rowid = ?
                    """, (xg_home, xg_away, hst, ast, hc, ac, id_partido))
                    
                    actualizados += 1
                    print(f"✅ Rellenado: {home_db} vs {away_db} (Córners: {hc}-{ac} | Tiros: {hst}-{ast} | xG: {xg_home}-{xg_away})")
                else:
                    print(f"⚠️ Partido encontrado ({home_db}), pero Sportmonks no tiene stats avanzadas para este torneo.")
                    
        # ⏱️ Pausa de seguridad para cuidar los límites de la API
        time.sleep(1.5)

    conn.commit()
    conn.close()
    print(f"\n🚀 ¡Proceso finalizado! Se rellenaron {actualizados} partidos internacional con calidad profesional.")

if __name__ == "__main__":
    enriquecer_con_sportmonks()