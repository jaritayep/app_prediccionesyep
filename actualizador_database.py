import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def auditoria_big_five_ayer():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    # 1. Definir "Ayer" y las Ligas
    ayer = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    
    # Mapeo de ligas para SoccerData (Las 5 grandes)
    ligas = [
        "ENG-Premier League", 
        "ESP-La Liga", 
        "ITA-Serie A", 
        "GER-Bundesliga", 
        "FRA-Ligue 1"
    ]
    
    print(f"📅 Iniciando auditoría para la fecha: {ayer}")

    try:
        # 2. Inicializar SoccerData con todas las ligas juntas
        fbref = sd.FBref(leagues=ligas, seasons="2025")
        
        # 3. Obtener el calendario
        df = fbref.read_schedule()
        df = df.reset_index()

        # --- FILTRO CRÍTICO ---
        # 1. Que tengan marcador
        # 2. Que la fecha coincida exactamente con ayer
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df_ayer = df[(df['score'].notnull()) & (df['date_str'] == ayer)].copy()
        
        if df_ayer.empty:
            print(f"⚠️ No se encontraron partidos finalizados el {ayer} en las 5 grandes.")
            return

        print(f"🔎 Procesando {len(df_ayer)} partidos de ayer...")

        for _, row in df_ayer.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Limpiar el score
            try:
                goles = row['score'].replace('–', '-').split('-')
                gl, gv = int(goles[0]), int(goles[1])
                ftr = 'H' if gl > gv else ('A' if gv > gl else 'D')
            except:
                continue

            # --- MIGRACIÓN Y LIMPIEZA ---
            # Insertamos en historial
            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0)
            """, (ayer, home, away, gl, gv, ftr))
            
            # Borramos de predicciones (usamos LIKE para mayor seguridad)
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", ayer))
            
            print(f"✅ [{row['league']}] {home} {gl}-{gv} {away}")

        conn.commit()
        print(f"\n🏁 Auditoría terminada. Se procesaron {len(df_ayer)} partidos.")

    except Exception as e:
        print(f"❌ Error en auditoría: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_big_five_ayer()
