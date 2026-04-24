import sqlite3
import pandas as pd
import numpy as np

def generar_dataset_ml():
    conn = sqlite3.connect('database_partidos.db')
    df = pd.read_sql("SELECT * FROM historial_multiliga_ml ORDER BY League, Date", conn)
    
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.slice(0, 10))
    df['Temporada'] = np.where(df['Date'].dt.month >= 8, df['Date'].dt.year, df['Date'].dt.year - 1)

    df['Pts_H_Match'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['Pts_A_Match'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    print("📊 Calculando Tabla de Posiciones y Fatiga Física...")
    
    # --- 1. CÁLCULO DE TABLA Y FATIGA EN UN SOLO PASO ---
    h = df[['Date', 'League', 'Temporada', 'HomeTeam', 'Pts_H_Match']].rename(columns={'HomeTeam': 'Team', 'Pts_H_Match': 'Pts'})
    a = df[['Date', 'League', 'Temporada', 'AwayTeam', 'Pts_A_Match']].rename(columns={'AwayTeam': 'Team', 'Pts_A_Match': 'Pts'})
    df_timeline = pd.concat([h, a]).sort_values(['Team', 'Date'])

    # Calculamos puntos acumulados
    df_timeline['Pts_Acum'] = df_timeline.groupby(['League', 'Temporada', 'Team'])['Pts'].cumsum() - df_timeline['Pts']
    
    # Calculamos DÍAS DE DESCANSO (Diferencia de días entre un partido y el anterior)
    df_timeline['Dias_Descanso'] = df_timeline.groupby('Team')['Date'].diff().dt.days
    # Si es el primer partido del equipo, asumimos 7 días de descanso normal
    df_timeline['Dias_Descanso'] = df_timeline['Dias_Descanso'].fillna(7.0)
    # Si el descanso es absurdamente largo (ej: parón de verano), lo capeamos a 14 días para no confundir a la IA
    df_timeline['Dias_Descanso'] = np.clip(df_timeline['Dias_Descanso'], 3, 14)

    # Unimos todo de vuelta (Puntos y Descanso Local)
    df = df.merge(df_timeline[['Date', 'Team', 'Pts_Acum', 'Dias_Descanso']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Pts_Acum': 'Home_Season_Pts', 'Dias_Descanso': 'Home_Rest'}).drop('Team', axis=1)
    
    # Unimos todo de vuelta (Puntos y Descanso Visita)
    df = df.merge(df_timeline[['Date', 'Team', 'Pts_Acum', 'Dias_Descanso']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Pts_Acum': 'Away_Season_Pts', 'Dias_Descanso': 'Away_Rest'}).drop('Team', axis=1)

    # --- 2. PROMEDIOS MÓVILES ---
    def get_rolling_stats(group, cols, n=5):
        return group[cols].shift(1).rolling(window=n, min_periods=n).mean()

    cols_stats = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'xG_home', 'xG_away']
    
    print("📈 Calculando rachas de los últimos 5 partidos...")
    df_home_stats = df.groupby(['League', 'HomeTeam']).apply(lambda x: get_rolling_stats(x, cols_stats)).reset_index(level=[0,1], drop=True)
    df_home_stats.columns = [f"Home_{c}" for c in df_home_stats.columns]
    
    df_final = pd.concat([df, df_home_stats], axis=1)

    # --- 3. SÚPER VARIABLES PARA LA IA ---
    df_final['Diferencia_Tabla'] = df_final['Home_Season_Pts'] - df_final['Away_Season_Pts']
    df_final['Home_xG_Diff'] = df_final['Home_xG_home'] - df_final['Home_xG_away']
    df_final['Home_Efficiency'] = df_final['Home_FTHG'] / (df_final['Home_xG_home'] + 0.01)
    
    # Ventaja Física: Positivo = Local llega más descansado, Negativo = Visita llega más descansada
    df_final['Ventaja_Fisica'] = df_final['Home_Rest'] - df_final['Away_Rest']

    df_final = df_final.dropna(subset=['Home_FTHG', 'Home_Season_Pts'])
    
    df_final.to_sql('dataset_entrenamiento_ia', conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Proceso terminado. Dataset con {len(df_final)} partidos listos para la IA.")

if __name__ == "__main__":
    generar_dataset_ml()