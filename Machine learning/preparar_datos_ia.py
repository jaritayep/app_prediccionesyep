import sqlite3
import pandas as pd
import numpy as np

def normalizar_nombre(nombre):
    mapeo = {
        "Nott'm Forest": "Nottingham Forest", "Ath Bilbao": "Athletic Club",
        "Athletic Bilbao": "Athletic Club", "Atl Madrid": "Atletico Madrid",
        "Barca": "Barcelona", "Barça": "Barcelona", "Paris SG": "PSG",
        "M'gladbach": "Borussia Monchengladbach", "Paris Saint-Germain": "PSG"
    }
    return mapeo.get(nombre.strip(), nombre.strip())

def generar_dataset_ml():
    conn = sqlite3.connect('database_partidos.db')
    # Cargamos el historial completo
    df = pd.read_sql("SELECT * FROM historial_multiliga_ml ORDER BY League, Date", conn)
    
    # 1. Preparación de Fechas y Temporadas
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.slice(0, 10))
    # En Europa, si el mes es >= 8 (Agosto), es la temporada del año actual. Si no, del anterior.
    df['Temporada'] = np.where(df['Date'].dt.month >= 8, df['Date'].dt.year, df['Date'].dt.year - 1)

    # 2. Cálculo de Puntos Reales (Para la tabla histórica)
    df['Pts_H_Match'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['Pts_A_Match'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    print("📊 Calculando Tabla de Posiciones histórica (Puntos Acumulados)...")
    
    # Truco de eficiencia: Creamos un set de datos largo con todos los equipos y sus puntos
    h = df[['Date', 'League', 'Temporada', 'HomeTeam', 'Pts_H_Match']].rename(columns={'HomeTeam': 'Team', 'Pts_H_Match': 'Pts'})
    a = df[['Date', 'League', 'Temporada', 'AwayTeam', 'Pts_A_Match']].rename(columns={'AwayTeam': 'Team', 'Pts_A_Match': 'Pts'})
    df_puntos = pd.concat([h, a]).sort_values('Date')

    # Calculamos puntos acumulados por temporada, restando el partido actual (para no hacer trampa)
    df_puntos['Pts_Acum'] = df_puntos.groupby(['League', 'Temporada', 'Team'])['Pts'].cumsum() - df_puntos['Pts']

    # Unimos de vuelta al dataframe principal
    df = df.merge(df_puntos[['Date', 'Team', 'Pts_Acum']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Pts_Acum': 'Home_Season_Pts'}).drop('Team', axis=1)
    df = df.merge(df_puntos[['Date', 'Team', 'Pts_Acum']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Pts_Acum': 'Away_Season_Pts'}).drop('Team', axis=1)

    # 3. Promedios Móviles (Rendimiento reciente - Últimos 5)
    def get_rolling_stats(group, cols, n=5):
        return group[cols].shift(1).rolling(window=n, min_periods=n).mean()

    cols_stats = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'xG_home', 'xG_away']
    
    print("📈 Calculando promedios móviles y xG...")
    df_home_stats = df.groupby(['League', 'HomeTeam']).apply(lambda x: get_rolling_stats(x, cols_stats)).reset_index(level=[0,1], drop=True)
    df_home_stats.columns = [f"Home_{c}" for c in df_home_stats.columns]
    
    # 4. Consolidación Final
    df_final = pd.concat([df, df_home_stats], axis=1)

    # Variables de ingeniería: Diferenciales
    df_final['Diferencia_Tabla'] = df_final['Home_Season_Pts'] - df_final['Away_Season_Pts']
    df_final['Home_xG_Diff'] = df_final['Home_xG_home'] - df_final['Home_xG_away']
    
    # Eficiencia (Goles / xG)
    df_final['Home_Efficiency'] = df_final['Home_FTHG'] / (df_final['Home_xG_home'] + 0.01)

    # Limpieza de filas sin historial suficiente
    df_final = df_final.dropna(subset=['Home_FTHG', 'Home_Season_Pts'])
    
    # Guardar para el entrenamiento
    df_final.to_sql('dataset_entrenamiento_ia', conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Proceso terminado. Dataset con {len(df_final)} partidos y contexto de tabla guardado.")

if __name__ == "__main__":
    generar_dataset_ml()