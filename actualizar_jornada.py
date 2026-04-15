import requests
import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta
import os

API_KEY = os.getenv('API_KEY', 'TU_API_KEY_PARA_PRUEBAS_LOCALES')
LIGAS = ['PL', 'PD', 'BL1', 'SA', 'FL1', 'PPL'] # Ligas principales
DB_NAME = "../database_partidos.db"

def actualizar_partidos_semana():
    headers = {'X-Auth-Token': API_KEY}
    all_matches = []
    
    # Definimos el rango: desde ahora hasta 7 días más
    hoy = datetime.now()
    proxima_semana = hoy + timedelta(days=7)
    
    print(f"🚀 Buscando partidos entre {hoy.date()} y {proxima_semana.date()}...")
    
    for liga in LIGAS:
        # Filtramos por estado SCHEDULED directamente en la URL
        url = f"https://api.football-data.org/v4/competitions/{liga}/matches?status=SCHEDULED"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                for m in matches:
                    fecha_partido = datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
                    
                    # FILTRO CRUCIAL: Solo los próximos 7 días
                    if hoy <= fecha_partido <= proxima_semana:
                        all_matches.append({
                            'League': liga,
                            'Date': m['utcDate'],
                            'Local': m['homeTeam']['shortName'],
                            'Visita': m['awayTeam']['shortName']
                        })
                print(f"✅ {liga}: {len(all_matches)} partidos válidos.")
            else:
                print(f"⚠️ Error en {liga}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Fallo de conexión en {liga}: {e}")
        
        # Pausa obligatoria de 10 seg (para no bloquear la cuenta gratuita)
        time.sleep(10)

    if all_matches:
        conn = sqlite3.connect(DB_NAME)
        df_jornada = pd.DataFrame(all_matches)
        
        # Esta es la tabla que "visualizaciones.py" busca para limpiar el desorden
        df_jornada.to_sql('tabla_predicciones_limpia', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n🔥 ¡LISTO! {len(all_matches)} partidos cargados en la base de datos.")
    else:
        print("\nℹ️ No se encontraron partidos programados para los próximos días.")

if __name__ == "__main__":
    actualizar_partidos_semana()
