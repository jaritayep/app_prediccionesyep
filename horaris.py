import sqlite3
import pandas as pd

def convertir_horarios_a_chile():
    print("⏳ Conectando a la base de datos...")
    conn = sqlite3.connect('database_partidos.db')
    
    # 1. Leer el fixture actual
    df = pd.read_sql("SELECT * FROM fixture_mundial", conn)
    
    if df.empty:
        print("⚠️ La tabla fixture_mundial está vacía.")
        return

    print("🌍 Convirtiendo horarios de UTC a Hora de Chile (America/Santiago)...")
    
    # 2. Unir fecha y hora actuales en un solo objeto Datetime y asignarle la zona UTC
    df['Datetime_UTC'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Datetime_UTC'] = df['Datetime_UTC'].dt.tz_localize('UTC')
    
    # 3. Convertir a la zona horaria de Chile
    df['Datetime_Chile'] = df['Datetime_UTC'].dt.tz_convert('America/Santiago')
    
    # 4. Separar de nuevo en las columnas originales Date y Time
    df['Date'] = df['Datetime_Chile'].dt.strftime('%Y-%m-%d')
    df['Time'] = df['Datetime_Chile'].dt.strftime('%H:%M')
    
    # Limpiar columnas temporales
    df = df.drop(columns=['Datetime_UTC', 'Datetime_Chile'])
    
    # 5. Sobrescribir la tabla en la base de datos
    df.to_sql('fixture_mundial', conn, if_exists='replace', index=False)
    
    conn.close()
    print("✅ ¡Éxito! Todos los partidos del Mundial ahora están en horario chileno.")

if __name__ == "__main__":
    convertir_horarios_a_chile()