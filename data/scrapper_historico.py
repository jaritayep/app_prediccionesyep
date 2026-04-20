import pandas as pd
import sqlite3
import time

# Códigos de liga en Football-Data
LIGAS = {
    "EPL": "E0",      # Inglaterra
    "LaLiga": "SP1",  # España
    "Bundesliga": "D1",# Alemania
    "SerieA": "I1",   # Italia
    "Ligue1": "F1"    # Francia
}

# Temporadas a descargar
TEMPORADAS = ["2021", "2122", "2223", "2324", "2425", "2526"]
DB_NAME = 'database_partidos.db'

def reconstruir_base_datos():
    conn = sqlite3.connect(DB_NAME)
    all_data = []

    for liga_nombre, liga_cod in LIGAS.items():
        for temp in TEMPORADAS:
            url = f"https://www.football-data.co.uk/mmz4281/{temp}/{liga_cod}.csv"
            print(f"Descargando {liga_nombre} | Temp: {temp}...")
            
            try:
                df = pd.read_csv(url)
                # Seleccionamos el set completo de variables para el ML profesional
                columnas = [
                    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 
                    'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A'
                ]
                
                # Filtrar solo si las columnas existen en el CSV
                df_filtro = df[[col for col in columnas if col in df.columns]].copy()
                df_filtro['League'] = liga_nombre
                all_data.append(df_filtro)
                time.sleep(0.5) 
            except Exception as e:
                print(f"⚠️ Error en {liga_nombre} {temp}: {e}")

    df_final = pd.concat(all_data, ignore_index=True)
    
    # Limpieza de fechas para que SQL no se confunda
    df_final['Date'] = pd.to_datetime(df_final['Date'], dayfirst=True, errors='coerce')
    df_final = df_final.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    
    # Reemplazar la tabla vieja con los datos nuevos y limpios
    df_final.to_sql('historial_multiliga_ml', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"\n✅ ¡Base de datos reconstruida!")
    print(f"Total de partidos limpios: {len(df_final)}")
    print(f"Variables incluidas: Goles, Tiros, Tiros a Puerta, Córners, Faltas y Tarjetas.")

if __name__ == "__main__":
    reconstruir_base_datos()