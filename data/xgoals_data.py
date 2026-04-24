import sqlite3
import pandas as pd
from understatapi import UnderstatClient
from thefuzz import process
import time

def sincronizacion_masiva():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    # 1. Obtenemos nuestros equipos para la IA de thefuzz
    cursor.execute("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml")
    equipos_db = [row[0] for row in cursor.fetchall()]

    understat = UnderstatClient()
    ligas_understat = ['EPL', 'La_Liga', 'Serie_A', 'Bundesliga', 'Ligue_1']
    
    # 2. LAS TEMPORADAS FALTANTES
    temporadas = [2023, 2024, 2025] 
    traductor_automatico = {}

    print("🤖 Iniciando Descarga Histórica Masiva...")

    for year in temporadas:
        print(f"\n🚀 --- PROCESANDO TEMPORADA {year}/{year+1} ---")
        for league in ligas_understat:
            print(f"📥 Descargando {league}...")
            try:
                matches = understat.league(league=league).get_match_data(season=year)
                df_xg = pd.DataFrame(matches)
                df_xg = df_xg[df_xg['isResult'] == True] # Solo terminados
                
                actualizados = 0
                for _, match in df_xg.iterrows():
                    u_home = match['h']['title']
                    u_away = match['a']['title']
                    
                    # Cortamos la fecha para quedarnos solo con YYYY-MM-DD
                    fecha_corta = match['datetime'][:10] 
                    
                    # Traducción automática de equipos
                    if u_home not in traductor_automatico:
                        traductor_automatico[u_home] = process.extractOne(u_home, equipos_db)[0]
                    if u_away not in traductor_automatico:
                        traductor_automatico[u_away] = process.extractOne(u_away, equipos_db)[0]
                    
                    db_home = traductor_automatico[u_home]
                    db_away = traductor_automatico[u_away]
                    xg_h = float(match['xG']['h'])
                    xg_a = float(match['xG']['a'])
                    
                    # 3. El truco del LIKE: Busca la fecha ignorando el 00:00:00
                    cursor.execute("""
                        UPDATE historial_multiliga_ml 
                        SET xG_home = ?, xG_away = ?
                        WHERE HomeTeam = ? AND AwayTeam = ? AND Date LIKE ?
                    """, (xg_h, xg_a, db_home, db_away, f"{fecha_corta}%"))
                    
                    if cursor.rowcount > 0:
                        actualizados += 1
                
                print(f"✅ {league}: {actualizados} partidos guardados.")
                time.sleep(1) # Cuidamos el servidor de Understat
                
            except Exception as e:
                print(f"❌ Error en {league}: {e}")

    conn.commit()
    conn.close()
    print("\n🏁 ¡HISTORIAL COMPLETO SINCRONIZADO! Vuelve a correr tu Auditoría.")

if __name__ == "__main__":
    sincronizacion_masiva()