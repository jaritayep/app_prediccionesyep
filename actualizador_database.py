import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def auditoria_forzada():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    # Ligas Big Five
    ligas = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
    
    print("🚀 INICIANDO AUDITORÍA AGRESIVA...")

    try:
        # Cargamos FBref para la temporada actual
        fbref = sd.FBref(leagues=ligas, seasons="2025")
        df = fbref.read_schedule()
        df = df.reset_index()

        # Aseguramos que la columna score sea string y no tenga nulos
        df = df[df['score'].notnull()].copy()
        
        # Filtro: Solo partidos que ocurrieron en ABRIL 2026 (para limpiar todo el mes)
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df_abril = df[df['date_str'].str.contains('2026-04')].copy()

        print(f"📊 Partidos con marcador encontrados en Abril: {len(df_abril)}")

        if df_abril.empty:
            print("❌ No se encontraron marcadores de Abril en FBref. ¿Quizás FBref no se ha actualizado?")
            return

        for _, row in df_abril.iterrows():
            fecha = row['date_str']
            home = row['home_team']
            away = row['away_team']
            score_raw = row['score']
            
            try:
                # Limpieza de marcador (FBref usa guiones largos '–')
                goles = score_raw.replace('–', '-').split('-')
                gl, gv = int(goles[0]), int(goles[1])
                ftr = 'H' if gl > gv else ('A' if gv > gl else 'D')
                
                # INSERTAR/ACTUALIZAR
                cursor.execute("""
                    INSERT OR REPLACE INTO historial_multiliga_ml 
                    ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0)
                """, (fecha, home, away, gl, gv, ftr))
                
                # LIMPIAR PREDICCIONES PENDIENTES
                cursor.execute("""
                    DELETE FROM tabla_predicciones_limpia 
                    WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
                """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha))
                
                print(f"✅ Procesado: {fecha} | {home} {gl}-{gv} {away}")
                
            except Exception as e:
                print(f"⚠️ Error procesando partido {home}-{away}: {e}")

        conn.commit()
        print("\n🏁 ¡PROCESO COMPLETADO! Base de datos guardada localmente.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN EL SCRIPT: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_forzada()
