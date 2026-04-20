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
    
    codigos_ligas = ['E0', 'SP1', 'I1', 'D1', 'F1']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }
    
    print("🚀 Iniciando extracción con Pandas (Stats Completas)...")

    try:
        dfs = []
        for codigo in codigos_ligas:
            url = f"https://www.football-data.co.uk/mmz4281/2526/{codigo}.csv"
            print(f"📥 Descargando liga {codigo}...")
            try:
                df_liga = pd.read_csv(url, storage_options=headers)
                dfs.append(df_liga)
            except Exception as e:
                print(f"⚠️ No se pudo descargar {codigo}: {e}")
                continue
                
        if not dfs:
            print("❌ El sistema bloqueó todas las descargas.")
            return
            
        df = pd.concat(dfs, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        mask_fechas = (df['Date'] >= hace_7_dias) & (df['Date'] <= hoy)
        df_recientes = df[mask_fechas].copy()
        
        if df_recientes.empty:
            print("⚠️ No hay partidos jugados en la última semana.")
            return

        print(f"🎯 Encontrados {len(df_recientes)} partidos. Guardando arsenal de stats en la DB...")

        for _, row in df_recientes.iterrows():
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
            
            # Tiros al arco (On Target)
            hst = int(row['HST']) if pd.notnull(row.get('HST')) else 0
            ast = int(row['AST']) if pd.notnull(row.get('AST')) else 0
            
            # Tiros Totales (NUEVO)
            hs = int(row['HS']) if pd.notnull(row.get('HS')) else 0
            as_shots = int(row['AS']) if pd.notnull(row.get('AS')) else 0
            
            # Faltas (NUEVO)
            hf = int(row['HF']) if pd.notnull(row.get('HF')) else 0
            af = int(row['AF']) if pd.notnull(row.get('AF')) else 0
            
            # Amarillas y Rojas (NUEVO)
            hy = int(row['HY']) if pd.notnull(row.get('HY')) else 0
            ay = int(row['AY']) if pd.notnull(row.get('AY')) else 0
            hr = int(row['HR']) if pd.notnull(row.get('HR')) else 0
            ar = int(row['AR']) if pd.notnull(row.get('AR')) else 0

            # Guardar en Base de Datos (Insertamos TODAS las columnas ahora)
            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HS], [AS], [HF], [AF], [HY], [AY], [HR], [AR])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hs, as_shots, hf, af, hy, ay, hr, ar))
            
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))
            
            print(f"✅ Ok: {fecha_str} | {home} {gl}-{gv} {away} | 🎯 Totales: {hs}-{as_shots}")

        conn.commit()
        print("\n🏁 ¡Base de datos histórica actualizada con TODAS las stats!")

    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_directa_csv()
