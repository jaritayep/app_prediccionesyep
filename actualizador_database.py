import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def auditoria_big_five_ayer():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    # 1. Definir RANGO (Últimos 7 días) para no dejar huecos
    hoy = datetime.now()
    fechas_a_revisar = [(hoy - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    
    ligas = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
    
    print(f"📅 Iniciando auditoría para el rango: {fechas_a_revisar[-1]} al {fechas_a_revisar[0]}")

    try:
        fbref = sd.FBref(leagues=ligas, seasons="2025")
        df = fbref.read_schedule()
        df = df.reset_index()

        # Creamos columna de fecha en string para filtrar fácil
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # FILTRO: Que tenga marcador Y que esté en nuestro rango de 7 días
        df_rango = df[(df['score'].notnull()) & (df['date_str'].isin(fechas_a_revisar))].copy()
        
        if df_rango.empty:
            print("⚠️ No se encontraron partidos nuevos en el rango de 7 días.")
            return

        print(f"🔎 Procesando {len(df_rango)} partidos encontrados...")

        for _, row in df_rango.iterrows():
            fecha_partido = row['date_str']
            home = row['home_team']
            away = row['away_team']
            
            try:
                goles = row['score'].replace('–', '-').split('-')
                gl, gv = int(goles[0]), int(goles[1])
                ftr = 'H' if gl > gv else ('A' if gv > gl else 'D')
            except:
                continue

            # MIGRACIÓN AL HISTORIAL
            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0)
            """, (fecha_partido, home, away, gl, gv, ftr))
            
            # LIMPIEZA DE PREDICCIONES
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_partido))
            
            print(f"✅ [{row['league']}] {home} {gl}-{gv} {away} ({fecha_partido})")

        conn.commit()
        print(f"\n🏁 Auditoría terminada. Base de datos actualizada.")

    except Exception as e:
        print(f"❌ Error en auditoría: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_big_five_ayer()
