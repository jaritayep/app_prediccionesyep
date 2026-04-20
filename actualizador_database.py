import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def auditoria_directa_csv():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    hoy = datetime.now()
    hace_7_dias = hoy - timedelta(days=7)
    
    # Códigos de las 5 grandes ligas en la página:
    # E0 = Premier, SP1 = LaLiga, I1 = Serie A, D1 = Bundesliga, F1 = Ligue 1
    codigos_ligas = ['E0', 'SP1', 'I1', 'D1', 'F1']
    
    # 🎭 EL TRUCO DE MAGIA: Nos disfrazamos de navegador web
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print("🚀 Iniciando extracción con Pandas (Modo Bypass de seguridad)...")

    try:
        dfs = []
        for codigo in codigos_ligas:
            # Apuntamos directo al CSV de la temporada actual (2526)
            url = f"https://www.football-data.co.uk/mmz4281/2526/{codigo}.csv"
            print(f"📥 Descargando liga {codigo}...")
            
            try:
                # pandas descarga el CSV pasando nuestras credenciales falsas de navegador
                df_liga = pd.read_csv(url, storage_options=headers)
                dfs.append(df_liga)
            except Exception as e:
                print(f"⚠️ No se pudo descargar {codigo}: {e}")
                continue
                
        if not dfs:
            print("❌ El sistema bloqueó todas las descargas.")
            return
            
        # Unimos todas las ligas en una sola tabla grande
        df = pd.concat(dfs, ignore_index=True)
        
        # Transformamos la fecha del CSV (que viene en formato Día/Mes/Año)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Filtramos los últimos 7 días
        mask_fechas = (df['Date'] >= hace_7_dias) & (df['Date'] <= hoy)
        df_recientes = df[mask_fechas].copy()
        
        if df_recientes.empty:
            print("⚠️ No hay partidos jugados en la última semana.")
            return

        print(f"🎯 Encontrados {len(df_recientes)} partidos. Guardando stats detalladas en la DB...")

        for _, row in df_recientes.iterrows():
            # Limpieza básica por si hay valores nulos
            if pd.isnull(row['HomeTeam']) or pd.isnull(row['AwayTeam']):
                continue
                
            fecha_str = row['Date'].strftime('%Y-%m-%d')
            home = str(row['HomeTeam']).strip()
            away = str(row['AwayTeam']).strip()
            
            # Goles y Resultado
            gl = int(row['FTHG']) if pd.notnull(row.get('FTHG')) else 0
            gv = int(row['FTAG']) if pd.notnull(row.get('FTAG')) else 0
            ftr = str(row.get('FTR', 'D'))
            
            # Córners
            hc = int(row['HC']) if pd.notnull(row.get('HC')) else 0
            ac = int(row['AC']) if pd.notnull(row.get('AC')) else 0
            
            # Tiros al arco
            hst = int(row['HST']) if pd.notnull(row.get('HST')) else 0
            ast = int(row['AST']) if pd.notnull(row.get('AST')) else 0
            
            # Amarillas
            hy = int(row['HY']) if pd.notnull(row.get('HY')) else 0
            ay = int(row['AY']) if pd.notnull(row.get('AY')) else 0

            # Guardar en Base de Datos (Auditoría)
            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hy, ay))
            
            # Eliminar el partido procesado de la tabla de predicciones futuras
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))
            
            print(f"✅ Ok: {fecha_str} | {home} {gl}-{gv} {away} | 🚩 {hc}-{ac} | 🎯 {hst}-{ast}")

        conn.commit()
        print("\n🏁 ¡Base de datos histórica actualizada con éxito!")

    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_directa_csv()
