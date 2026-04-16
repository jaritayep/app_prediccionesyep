import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def actualizar_stats_completas_7_dias():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    hoy = datetime.now()
    hace_7_dias = hoy - timedelta(days=7)
    
    ligas = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
    
    print(f"🚀 Iniciando scraping profundo de FBref (Desde {hace_7_dias.strftime('%Y-%m-%d')})")

    try:
        # IMPORTANTE: Temporada 2026 para que coincida con el calendario actual
        fbref = sd.FBref(leagues=ligas, seasons="2026")
        
        df_schedule = fbref.read_schedule().reset_index()
        df_schedule['date'] = pd.to_datetime(df_schedule['date']).dt.tz_localize(None)
        
        mask_fechas = (df_schedule['date'] >= hace_7_dias) & (df_schedule['date'] <= hoy) & (df_schedule['score'].notnull())
        df_recientes = df_schedule[mask_fechas].copy()
        
        if df_recientes.empty:
            print("⚠️ No hay partidos finalizados en los últimos 7 días.")
            return

        print(f"🎯 Encontrados {len(df_recientes)} partidos. Descargando detalles...")

        df_shoot = fbref.read_team_match_stats(stat_type="shooting").reset_index()
        df_misc = fbref.read_team_match_stats(stat_type="misc").reset_index()
        df_pass = fbref.read_team_match_stats(stat_type="passing_types").reset_index()
        
        df_shoot['date'] = pd.to_datetime(df_shoot['date']).dt.tz_localize(None)
        df_misc['date'] = pd.to_datetime(df_misc['date']).dt.tz_localize(None)
        df_pass['date'] = pd.to_datetime(df_pass['date']).dt.tz_localize(None)

        for _, match in df_recientes.iterrows():
            fecha_exacta = match['date']
            fecha_str = fecha_exacta.strftime('%Y-%m-%d')
            home = match['home_team']
            away = match['away_team']
            score_raw = str(match['score']).replace('–', '-')
            
            try:
                goles = score_raw.split('-')
                gl, gv = int(goles[0]), int(goles[1])
                ftr = 'H' if gl > gv else ('A' if gv > gl else 'D')
            except:
                continue 

            def get_stat(df_stats, team_name, match_date, stat_keyword):
                try:
                    fila = df_stats[(df_stats['team'] == team_name) & (df_stats['date'] == match_date)]
                    if fila.empty: return 0
                    columna = [col for col in fila.columns if stat_keyword in str(col)][0]
                    valor = fila.iloc[0][columna]
                    return int(valor) if pd.notnull(valor) else 0
                except:
                    return 0

            hst = get_stat(df_shoot, home, fecha_exacta, 'SoT')
            ast = get_stat(df_shoot, away, fecha_exacta, 'SoT')
            hy = get_stat(df_misc, home, fecha_exacta, 'CrdY')
            ay = get_stat(df_misc, away, fecha_exacta, 'CrdY')
            hc = get_stat(df_pass, home, fecha_exacta, 'CK')
            ac = get_stat(df_pass, away, fecha_exacta, 'CK')

            cursor.execute("""
                INSERT OR REPLACE INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HY], [AY], [HS], [AS])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hy, ay))
            
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))
            
            print(f"✅ Guardado: {fecha_str} | {home} {gl}-{gv} {away}")

        conn.commit()
        print("\n🏁 Proceso completado y guardado en la DB.")

    except Exception as e:
        print(f"\n❌ Error durante el scraping: {e}")
        raise e 
    finally:
        conn.close()

if __name__ == "__main__":
    actualizar_stats_completas_7_dias()
