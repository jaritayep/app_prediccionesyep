from understatapi import UnderstatClient
import pandas as pd
import sqlite3
import time

def normalizar_nombre(nombre):
    # Diccionario maestro para que Understat coincida con tu DB
    mapeo_especifico = {
        # Premier League
        "Nott'm Forest": "Nottingham Forest",
        "Nottingham Forest": "Nottingham Forest",
        "Man Utd": "Manchester United",
        "Manchester United": "Manchester United",
        "Man City": "Manchester City",
        "Manchester City": "Manchester City",
        
        # La Liga
        "Ath Bilbao": "Athletic Club",
        "Athletic Club": "Athletic Club",
        "Athletic Bilbao": "Athletic Club",
        "Atl Madrid": "Atletico Madrid",
        "Atletico Madrid": "Atletico Madrid",
        "Ath Madrid": "Atletico Madrid",
        "Barca": "Barcelona",
        "Barça": "Barcelona",
        "FC Barcelona": "Barcelona",
        "Barcelona": "Barcelona",
        
        # Bundesliga
        "M'gladbach": "Borussia Monchengladbach",
        "M'Gladbach": "Borussia Monchengladbach",
        "Gladbach": "Borussia Monchengladbach",
        "Borussia M.Gladbach": "Borussia Monchengladbach",
        
        # Ligue 1
        "Paris SG": "PSG",
        "PSG": "PSG",
        "Paris Saint Germain": "PSG",
        "Paris Saint-Germain": "PSG"
    }
    
    nombre_sucio = nombre.strip()
    return mapeo_especifico.get(nombre_sucio, nombre_sucio)

def sincronizacion_total_xg(year=2025):
    understat = UnderstatClient()
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    ligas_understat = ['EPL', 'La_Liga', 'Serie_A', 'Bundesliga', 'Ligue_1']

    for league in ligas_understat:
        print(f"📥 Sincronizando {league}...")
        try:
            matches = understat.league(league=league).get_match_data(season=year)
            df_xg = pd.DataFrame(matches)
            df_xg = df_xg[df_xg['isResult'] == True]
            
            actualizados = 0
            for _, match in df_xg.iterrows():
                home = normalizar_nombre(match['h']['title'])
                away = normalizar_nombre(match['a']['title'])
                xg_h = float(match['xG']['h'])
                xg_a = float(match['xG']['a'])
                
                # BUSQUEDA FLEXIBLE:
                # Buscamos el partido por equipos. Si hay varios (ej. ida y vuelta), 
                # el 'ORDER BY Date DESC' asegura que miremos el más reciente.
                cursor.execute("""
                    UPDATE historial_multiliga_ml 
                    SET xG_home = ?, xG_away = ?
                    WHERE HomeTeam = ? AND AwayTeam = ? 
                    AND (xG_home IS NULL OR xG_home = 0)
                """, (xg_h, xg_a, home, away))
                
                if cursor.rowcount > 0:
                    actualizados += 1
            
            print(f"✅ {league}: {actualizados} nuevos partidos recibieron xG.")
            
        except Exception as e:
            print(f"❌ Error en {league}: {e}")
        
        time.sleep(1)

    conn.commit()
    conn.close()
    print("\n🏁 ¡Sincronización terminada!")

if __name__ == "__main__":
    sincronizacion_total_xg()