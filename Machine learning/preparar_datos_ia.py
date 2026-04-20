import sqlite3
import pandas as pd

def generar_dataset_ml():
    conn = sqlite3.connect('database_partidos.db')
    df = pd.read_sql("SELECT * FROM historial_multiliga_ml ORDER BY League, Date", conn)
    
    # Función para calcular promedios de los últimos 5 partidos
    def get_rolling_stats(group, cols, n=5):
        return group[cols].shift(1).rolling(window=n).mean()

    # Columnas que queremos promediar (Rendimiento)
    cols_to_avg = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY']
    
    print("Calculando rachas y promedios móviles...")
    # Calculamos para el equipo local
    df_home = df.groupby(['League', 'HomeTeam']).apply(lambda x: get_rolling_stats(x, cols_to_avg)).reset_index(level=[0,1], drop=True)
    df_home.columns = [f"Home_{c}" for c in df_home.columns]
    
    # Unimos y guardamos
    df_final = pd.concat([df, df_home], axis=1).dropna()
    
    df_final.to_sql('dataset_entrenamiento_ia', conn, if_exists='replace', index=False)
    conn.close()
    print("✅ Dataset listo con promedios móviles.")

if __name__ == "__main__":
    generar_dataset_ml()