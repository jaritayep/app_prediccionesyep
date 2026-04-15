import soccerdata as sd
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def auditoria_con_stats_reales():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    hoy = datetime.now()
    fechas_a_revisar = [(hoy - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    ligas = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]

    try:
        fbref = sd.FBref(leagues=ligas, seasons="2025")
        
        # 1. Traemos las estadísticas de "Manejo del balón" (donde FBref guarda corners y tiros)
        print("📊 Descargando estadísticas detalladas (esto puede tardar un poco)...")
        # 'passing' o 'shooting' suelen traer los tiros, pero para corners 
        # a veces necesitamos 'misc' o procesar el match_stats
        df_stats = fbref.read_team_match_stats(stat_type="misc") 
        df_stats = df_stats.reset_index()
        
        # 2. Filtrar por nuestro rango de fechas
        df_stats['date_str'] = df_stats['date'].dt.strftime('%Y-%m-%d')
        df_recientes = df_stats[df_stats['date_str'].isin(fechas_a_revisar)].copy()

        if df_recientes.empty:
            print("⚠️ No hay estadísticas nuevas en el rango de 7 días.")
            return

        # Agrupamos por partido para tener Local y Visita en la misma fila
        for (fecha, league, home, away), group in df_recientes.groupby(['date_str', 'league', 'home_team', 'away_team']):
            
            # Extraemos los datos del grupo (un equipo es index 0, el otro es index 1)
            # Nota: FBref devuelve una fila por equipo, las sumamos o mapeamos
            team1 = group.iloc[0]
            team2 = group.iloc[1]
            
            # Identificamos quién es local y quién visita para no cruzar cables
            # (FBref suele poner el marcador y stats por equipo)
            stats = {
                'HC': team1['corner_kicks'] if team1['is_home'] else team2['corner_kicks'],
                'AC': team2['corner_kicks'] if team1['is_home'] else team1['corner_kicks'],
                'HY': team1['cards_yellow'] if team1['is_home'] else team2['cards_yellow'],
                'AY': team2['cards_yellow'] if team1['is_home'] else team1['cards_yellow'],
            }

            # 3. UPDATE en la base de datos (rellenamos los ceros)
            cursor.execute("""
                UPDATE historial_multiliga_ml 
                SET HC = ?, AC = ?, HY = ?, AY = ?
                WHERE Date LIKE ? AND HomeTeam = ? AND AwayTeam = ?
            """, (stats['HC'], stats['AC'], stats['HY'], stats['AY'], f"{fecha}%", home, away))
            
            print(f"✅ Stats actualizadas: {home} vs {away} ({fecha})")

        conn.commit()
        print("\n🏁 ¡Historial enriquecido con estadísticas reales!")

    except Exception as e:
        print(f"❌ Error al extraer stats: {e}")
    finally:
        conn.close()
