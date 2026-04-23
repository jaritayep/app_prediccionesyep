import sqlite3
import pandas as pd

def generar_dataset_ml():
    conn = sqlite3.connect('database_partidos.db')
    # Traemos todo, asegurándonos de que xG_home y xG_away existan
    df = pd.read_sql("SELECT * FROM historial_multiliga_ml ORDER BY League, Date", conn)
    
    # Función para calcular promedios de los últimos 5 partidos
    def get_rolling_stats(group, cols, n=5):
        return group[cols].shift(1).rolling(window=n, min_periods=n).mean()

    # --- LISTA ACTUALIZADA CON xG ---
    cols_to_avg = [
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 
        'HC', 'AC', 'HY', 'AY', 'xG_home', 'xG_away'
    ]
    
    print("🚀 Calculando rachas, promedios móviles y métricas de xG...")
    
    # Calculamos promedios móviles para el equipo local
    # (min_periods=n asegura que no usemos promedios con solo 1 o 2 partidos)
    df_home_stats = df.groupby(['League', 'HomeTeam']).apply(
        lambda x: get_rolling_stats(x, cols_to_avg)
    ).reset_index(level=[0,1], drop=True)
    
    # Renombramos para el dataset de entrenamiento
    df_home_stats.columns = [f"Home_{c}" for c in df_home_stats.columns]
    
    # Unimos los datos originales con los promedios
    df_final = pd.concat([df, df_home_stats], axis=1)

    # --- MÉTRICA EXTRA: Diferencial de xG ---
    # Esto le dice a la IA si el equipo suele dominar el área rival más que la propia
    df_final['Home_xG_Diff'] = df_final['Home_xG_home'] - df_final['Home_xG_away']

    # Eliminamos las filas iniciales que no tienen suficiente historial (NaNs)
    df_final = df_final.dropna(subset=['Home_FTHG', 'Home_xG_home'])
    
    # Guardamos en la tabla que lee el script de ML
    df_final.to_sql('dataset_entrenamiento_ia', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ Dataset actualizado: {len(df_final)} filas listas para el entrenamiento con xG.")

if __name__ == "__main__":
    generar_dataset_ml()