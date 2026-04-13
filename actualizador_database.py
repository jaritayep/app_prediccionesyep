import pandas as pd
import sqlite3
import time
import requests
import numpy as np

def limpiar_dato(valor):
    """Extrae solo el número de celdas que traen texto (ej: '5 (40%)' -> 5)"""
    try:
        if isinstance(valor, str):
            return int(''.join(filter(str.isdigit, valor.split()[0])))
        return int(valor)
    except:
        return 0

def actualizar_resultados_pro():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    ligas_urls = {
        'PL': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        'PD': 'https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
        'BL1': 'https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures',
        'SA': 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures',
        'FL1': 'https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures'
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for liga, url in ligas_urls.items():
        print(f"📡 Revisando jornada de {liga}...")
        try:
            response = requests.get(url, headers=headers)
            tablas = pd.read_html(response.text, extract_links="body")
            df = tablas[0]

            df_jugados = df[df['Score'].apply(lambda x: isinstance(x, tuple) and x[0] != '')].copy()

            for idx, row in df_jugados.iterrows():
                home = row['Home'][0]
                away = row['Away'][0]
                fecha = row['Date'][0]
                score_text = row['Score'][0]
                match_link = "https://fbref.com" + row['Match Report'][1]
                
                # 1. VERIFICACIÓN: ¿Existe? ¿Tiene stats?
                cursor.execute("SELECT HC FROM historial_multiliga_ml WHERE HomeTeam=? AND Date=?", (home, fecha))
                resultado = cursor.fetchone()
                
                if resultado is not None:
                    if resultado[0] is not None:
                        # Ya está completo, saltar
                        continue
                    else:
                        # Existe pero sin stats (como mencionaste), borrar para re-intentar
                        print(f"🔄 Datos incompletos para {home} vs {away}. Re-intentando...")
                        cursor.execute("DELETE FROM historial_multiliga_ml WHERE HomeTeam=? AND Date=?", (home, fecha))
                        conn.commit()

                # 2. ESPERA DE SEGURIDAD (1 MINUTO)
                print(f"⏳ Pausa de 60s antes de scrapear: {home} vs {away}...")
                time.sleep(60)

                # 3. SCRAPING DEL MATCH REPORT
                try:
                    res_match = requests.get(match_link, headers=headers)
                    match_tablas = pd.read_html(res_match.text)
                    
                    # Buscamos la tabla 'Team Stats'
                    df_stats = next(t for t in match_tablas if 'Possession' in t.values)
                    
                    goles_l, goles_v = map(int, score_text.split('–'))
                    ftr = 'H' if goles_l > goles_v else ('A' if goles_v > goles_l else 'D')

                    hst = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Shots on Target', na=False)].iloc[0,0])
                    ast = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Shots on Target', na=False)].iloc[0,2])
                    hc = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Corners', na=False)].iloc[0,0])
                    ac = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Corners', na=False)].iloc[0,2])
                    hy = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Cards', na=False)].iloc[0,0])
                    ay = limpiar_dato(df_stats[df_stats.iloc[:,1].str.contains('Cards', na=False)].iloc[0,2])

                    # 4. INSERTAR
                    nuevo_partido = {
                        'Date': fecha, 'HomeTeam': home, 'AwayTeam': away,
                        'FTHG': goles_l, 'FTAG': goles_v, 'FTR': ftr,
                        'HST': hst, 'AST': ast, 'HC': hc, 'AC': ac, 'HY': hy, 'AY': ay,
                        'HS': hst * 2, 'AS': ast * 2
                    }
                    
                    pd.DataFrame([nuevo_partido]).to_sql('historial_multiliga_ml', conn, if_exists='append', index=False)
                    print(f"✅ Guardado con éxito: {home} {goles_l}-{goles_v} {away}")

                except Exception as e:
                    print(f"⚠️ Error al obtener match report: {e}")

        except Exception as e:
            print(f"❌ Error grave en liga {liga}: {e}")

    conn.close()
    print("🚀 Proceso de actualización finalizado.")

if __name__ == "__main__":
    actualizar_resultados_pro()
