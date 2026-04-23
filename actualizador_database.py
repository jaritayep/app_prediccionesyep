import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
def normalizar_nombre(nombre):
    # Diccionario de correcciones manuales (El "Traductor")
    mapeo_especifico = {
        # Premier League (Los que ya arreglamos)
        "Nott'm Forest": "Nottingham Forest",
        "Man Utd": "Manchester United",
        "Man City": "Manchester City",
        
        # La Liga (Ajustes de Bilbao, Madrid y Barça)
        "Ath Bilbao": "Athletic Club",
        "Athletic Bilbao": "Athletic Club",
        "Atl Madrid": "Atletico Madrid",
        "Ath Madrid": "Atletico Madrid",
        "Atleti": "Atletico Madrid",
        "Barca": "Barcelona",
        "Barça": "Barcelona",
        "FC Barcelona": "Barcelona",
        
        # Bundesliga
        "M'gladbach": "Borussia Monchengladbach",
        "M'Gladbach": "Borussia Monchengladbach",
        "Gladbach": "Borussia Monchengladbach",
        
        # Ligue 1
        "Paris SG": "PSG",
        "Paris Saint Germain": "PSG",
        "Paris SG": "PSG"
    }
    
    nombre_sucio = nombre.strip()
    
    # Si el nombre está en nuestra lista de "rebeldes", lo cambiamos directo
    # Si no está, devolvemos el original para que pase a thefuzz
    return mapeo_especifico.get(nombre_sucio, nombre_sucio)

def auditoria_directa_csv():
    conn = sqlite3.connect('database_partidos.db')
    cursor = conn.cursor()
    
    hoy = datetime.now()
    hace_7_dias = hoy - timedelta(days=7)
    
    codigos_ligas = ['E0', 'SP1', 'I1', 'D1', 'F1']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }
    
    print("🚀 Iniciando extracción con Pandas (Stats Completas y Filtro Anti-Duplicados)...")

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

        # 🛡️ ESCUDO 1: Eliminar filas que vengan con los tiros completamente vacíos (NULL)
        df_recientes = df_recientes.dropna(subset=['HST', 'AST'])

        print(f"🎯 Encontrados {len(df_recientes)} partidos. Verificando duplicados y stats...")

        partidos_agregados = 0

        for _, row in df_recientes.iterrows():
            if pd.isnull(row['HomeTeam']) or pd.isnull(row['AwayTeam']):
                continue
                
            fecha_str = row['Date'].strftime('%Y-%m-%d')
            home = str(row['HomeTeam']).strip()
            away = str(row['AwayTeam']).strip()
            
            # Extracción de Tiros
            hst = int(row['HST']) if pd.notnull(row.get('HST')) else 0
            ast = int(row['AST']) if pd.notnull(row.get('AST')) else 0
            hs = int(row['HS']) if pd.notnull(row.get('HS')) else 0
            as_shots = int(row['AS']) if pd.notnull(row.get('AS')) else 0
            
            # 🛡️ ESCUDO 2: Si un partido marca 0 tiros totales para ambos, es un error de la web. Se salta.
            if hst == 0 and ast == 0 and hs == 0 and as_shots == 0:
                print(f"⏩ Saltando {home} vs {away} (La web aún no sube sus estadísticas)")
                continue

            # 🛡️ ESCUDO 3: Verificar si el partido ya está en la Base de Datos para no duplicarlo
            cursor.execute("SELECT 1 FROM historial_multiliga_ml WHERE HomeTeam = ? AND AwayTeam = ? AND Date = ?", (home, away, fecha_str))
            if cursor.fetchone():
                # Si entra aquí, es porque el partido ya existe. Lo saltamos en silencio o con aviso.
                # print(f"🔁 Saltando {home} vs {away} (Ya existe en el historial)")
                continue
            
            # Goles y Resultado
            gl = int(row['FTHG']) if pd.notnull(row.get('FTHG')) else 0
            gv = int(row['FTAG']) if pd.notnull(row.get('FTAG')) else 0
            ftr = str(row.get('FTR', 'D'))
            
            # Córners
            hc = int(row['HC']) if pd.notnull(row.get('HC')) else 0
            ac = int(row['AC']) if pd.notnull(row.get('AC')) else 0
            
            # Faltas
            hf = int(row['HF']) if pd.notnull(row.get('HF')) else 0
            af = int(row['AF']) if pd.notnull(row.get('AF')) else 0
            
            # Amarillas y Rojas
            hy = int(row['HY']) if pd.notnull(row.get('HY')) else 0
            ay = int(row['AY']) if pd.notnull(row.get('AY')) else 0
            hr = int(row['HR']) if pd.notnull(row.get('HR')) else 0
            ar = int(row['AR']) if pd.notnull(row.get('AR')) else 0

            # Guardar en Base de Datos (Insert normal, los duplicados ya fueron filtrados)
            cursor.execute("""
                INSERT INTO historial_multiliga_ml 
                ([Date], [HomeTeam], [AwayTeam], [FTHG], [FTAG], [FTR], [HC], [AC], [HST], [AST], [HS], [AS], [HF], [AF], [HY], [AY], [HR], [AR])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fecha_str, home, away, gl, gv, ftr, hc, ac, hst, ast, hs, as_shots, hf, af, hy, ay, hr, ar))
            
            # Limpiar Predicciones futuras (Prevenir el Leakage del 100%)
            cursor.execute("""
                DELETE FROM tabla_predicciones_limpia 
                WHERE (Local LIKE ? OR Visita LIKE ?) AND Date <= ?
            """, (f"%{home[:5]}%", f"%{away[:5]}%", fecha_str))
            
            print(f"✅ Ok: {fecha_str} | {home} {gl}-{gv} {away} | 🎯 Totales: {hs}-{as_shots}")
            partidos_agregados += 1

        conn.commit()
        print(f"\n🏁 ¡Actualización lista! Se guardaron {partidos_agregados} partidos nuevos reales.")

    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    auditoria_directa_csv()