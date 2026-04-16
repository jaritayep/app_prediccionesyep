import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def auditoria_matchhistory_7_dias():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    hoy = datetime.now()
    hace_7_dias = hoy - timedelta(days=7)
    
    # MatchHistory usa los mismos nombres de ligas de soccerdata
    ligas = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
    
    print(f"🚀 Iniciando descarga rápida desde MatchHistory (football-data.co.uk)...")

    try:
        # Llamamos a MatchHistory en lugar de FBref
        mh = sd.MatchHistory(leagues=ligas, seasons="2025")
        
        # read_games() trae todo: marcadores, corners, tiros y tarjetas de una vez
        df = mh.read_games().reset_index()
        
        # Filtro de fechas
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        mask_fechas = (df['date'] >= hace_7_dias) & (df['date'] <= hoy)
        df_recientes = df[mask_fechas].copy()
        
        if df_recientes.empty:
            print("⚠️ No hay partidos finalizados en la última semana en esta fuente.")
            return

        print(f"🎯 Encontrados {len(df_recientes)} partidos. Guardando stats detalladas...")

        for _, row in df_recientes.iterrows():
            fecha_str = row['date'].strftime('%Y-%m-%d')
            home = row['home_team']
            away = row['away_team']
            
            # --- EXTRACCIÓN DE STATS DIRECTA Y SEGURA ---
            # MatchHistory ya usa el formato FTHG, FTAG, HC, etc.
            # Usamos pd.notnull para evitar errores si un partido reciente no tiene corners cargados aún
            gl = int(row['FTHG']) if pd.notnull(row.get('FTHG')) else 0
            gv = int(row['FTAG']) if pd.notnull(row.get('FTAG')) else 0
            ftr = row.get('FTR', 'D')
            
            # Córners (Home Corners / Away Corners)
            hc = int(row['HC']) if pd.notnull(row.get('HC')) else 0
            ac = int(row['AC']) if pd.notnull(row.get('AC')) else 0
            
            # Tiros al Arco (Home Shots on Target / Away Shots on Target)
            hst = int(row['HST']) if pd.notnull(row.get('HST')) else 0
            ast = int(row['AST']) if pd.notnull(row.get('AST')) else 0
            
            # Tarjetas Amarillas (Home Yellow / Away Yellow)
            hy = int(row['HY']) if pd.notnull(row.get('HY')) else 0
            ay = int(row['AY']) if pd.notnull(row.get('AY')) else 0

            # 4. Guardar en Base de Datos
            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hy, ay))
            
            # Limpiar tabla de predicciones
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))
            
            print(f"✅ Ok: {fecha_str} | {home} {gl}-{gv} {away} | 🚩 {hc}-{ac} | 🎯 {hst}-{ast} | 🟨 {hy}-{ay}")

        conn.commit()
        print("\n🏁 Proceso completado y subido con éxito.")

    except Exception as e:
        print(f"\n❌ Error durante la extracción: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_matchhistory_7_dias()

