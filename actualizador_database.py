import pandas as pd
import sqlite3
import time

def actualizar_desde_fbref():
    # Diccionario de URLs de FBRef para las 5 grandes ligas (Temporada 23-24 o 24-25)
    # FBRef usa IDs específicos por liga
    ligas_urls = {
        'PL': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        'PD': 'https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
        'BL1': 'https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures',
        'SA': 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures',
        'FL1': 'https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures'
    }
    
    conn = sqlite3.connect('database_partidos.db')
    
    for liga, url in ligas_urls.items():
        print(f"📡 Scrapeando {liga} desde FBRef...")
        try:
            # Leemos las tablas de la URL
            tablas = pd.read_html(url)
            df = tablas[0] # La primera tabla suele ser el fixture/resultados
            
            # Limpiamos: Solo filas con marcador (partidos jugados)
            df = df.dropna(subset=['Score'])
            
            # Aquí procesaríamos las columnas para que encajen con tu historial_multiliga_ml
            # FBRef requiere un poco de limpieza de nombres de columnas
            # ... (Lógica de limpieza) ...
            
            print(f"✅ {liga} actualizada.")
            time.sleep(5) # Respeto a los servidores de FBRef
        except Exception as e:
            print(f"❌ Error en {liga}: {e}")
            
    conn.close()