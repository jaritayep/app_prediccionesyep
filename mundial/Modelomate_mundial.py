import sqlite3
import pandas as pd
import joblib
import numpy as np

def predecir_fixture():
    print("🧠 Despertando a la Inteligencia Artificial...")
    
    # 1. Cargar los modelos entrenados
    try:
        modelo = joblib.load('modelo_selecciones_rf.pkl')
        encoder = joblib.load('encoder_equipos_selecciones.pkl')
    except FileNotFoundError:
        print("🛑 Error: No se encontraron los archivos .pkl. Asegúrate de haber corrido el script de entrenamiento.")
        return

    # 2. Conectar a la base de datos
    conn = sqlite3.connect('database_partidos.db')
    
    # Cargar el historial (para calcular promedios de fuerza)
    df_historial = pd.read_sql("SELECT * FROM historial_selecciones_ml", conn)
    
    # Cargar los partidos futuros (Solo los que ya tienen equipos definidos, sin 'TBA')
    df_fixture = pd.read_sql("SELECT * FROM fixture_mundial WHERE HomeTeam != 'TBA' AND AwayTeam != 'TBA'", conn)
    
    if df_fixture.empty:
        print("⚠️ El fixture del Mundial aún tiene todos los equipos como 'TBA'.")
        print("💡 Prueba predictiva: Vamos a simular un partido clásico en su lugar.")
        # Simularemos un partido manualmente si el fixture oficial aún no está listo
        df_fixture = pd.DataFrame([{
            'HomeTeam': 'Argentina', 'AwayTeam': 'France', 'Date': '2026-07-19', 'Round': 'Final Simulada'
        }])

    print(f"📅 Se analizarán {len(df_fixture)} partidos futuros.")
    
    predicciones = []

    # 3. Motor de Predicción
    # 3. Motor de Predicción
    
    # 🎯 DICCIONARIO DE TRADUCCIÓN (Fixture -> Base de Datos)
    # Agrega aquí cualquier otra discrepancia de nombres que vayas encontrando
    TRADUCCION_EQUIPOS = {
        "Czechia": "Czech Republic",
        "South Korea": "Korea Republic",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Cape Verde Islands": "Cape Verde",
        "Congo DR": "DR Congo",
        "USA": "United States"
    }

    for _, partido in df_fixture.iterrows():
        home_original = partido['HomeTeam']
        away_original = partido['AwayTeam']
        
        # Traducimos el nombre si está en el diccionario; si no, dejamos el original
        home = TRADUCCION_EQUIPOS.get(home_original, home_original)
        away = TRADUCCION_EQUIPOS.get(away_original, away_original)
        
        # Validar que la IA conozca a los equipos (que estén en el Encoder)
        if home not in encoder.classes_ or away not in encoder.classes_:
            print(f"⚠️ Saltando {home} vs {away}: Uno de los equipos no tiene historial en la base de datos.")
            continue
            
        # Calcular fuerza del Local (Promedio de sus tiros y córners cuando juega de local)
        hist_home = df_historial[df_historial['HomeTeam'] == home]
        avg_hst = hist_home['HST'].mean() if not hist_home.empty else 4.0
        avg_hc = hist_home['HC'].mean() if not hist_home.empty else 5.0
        
        # Calcular fuerza de la Visita (Promedio de sus tiros y córners cuando juega de visita)
        hist_away = df_historial[df_historial['AwayTeam'] == away]
        avg_ast = hist_away['AST'].mean() if not hist_away.empty else 3.5
        avg_ac = hist_away['AC'].mean() if not hist_away.empty else 4.0

        # Calcular xG promedio para local y visita
        avg_xg_h = hist_home['xG_home'].mean() if not hist_home.empty and 'xG_home' in hist_home.columns and hist_home['xG_home'].mean() > 0 else 1.2
        avg_xg_a = hist_away['xG_away'].mean() if not hist_away.empty and 'xG_away' in hist_away.columns and hist_away['xG_away'].mean() > 0 else 1.0
        
        # Codificar los nombres a números para la IA
        home_code = encoder.transform([home])[0]
        away_code = encoder.transform([away])[0]
        
        # Ensamblar las variables exactas que espera el modelo
        # ['HomeTeam_Code', 'AwayTeam_Code', 'HST', 'AST', 'HC', 'AC', 'xG_home', 'xG_away']
        features_entrada = pd.DataFrame([[home_code, away_code, avg_hst, avg_ast, avg_hc, avg_ac, avg_xg_h, avg_xg_a]],
                                        columns=['HomeTeam_Code', 'AwayTeam_Code', 'HST', 'AST', 'HC', 'AC', 'xG_home', 'xG_away'])
        
        # 🎯 LA MAGIA: predict_proba devuelve [Prob_Visita, Prob_Empate, Prob_Local]
        probabilidades = modelo.predict_proba(features_entrada)[0]
        
        prob_away = probabilidades[0] * 100
        prob_draw = probabilidades[1] * 100
        prob_home = probabilidades[2] * 100
        
        # Calcular cuotas "Reales" (Fair Odds = 1 / Probabilidad decimal)
        cuota_justa_home = 1 / (probabilidades[2] + 0.001) # Se suma 0.001 para evitar dividir por 0
        cuota_justa_draw = 1 / (probabilidades[1] + 0.001)
        cuota_justa_away = 1 / (probabilidades[0] + 0.001)
        
        predicciones.append({
            'Ronda': partido.get('Round', 'Amistoso'),
            'Partido': f"{home} vs {away}",
            'Prob_Local': f"{prob_home:.1f}% (Cuota {cuota_justa_home:.2f})",
            'Prob_Empate': f"{prob_draw:.1f}% (Cuota {cuota_justa_draw:.2f})",
            'Prob_Visita': f"{prob_away:.1f}% (Cuota {cuota_justa_away:.2f})"
        })

    # Mostrar resultados en consola
    df_resultados = pd.DataFrame(predicciones)
    print("\n🔮 PREDICCIONES DEL ORÁCULO 🔮")
    print("-" * 60)
    print(df_resultados.to_string(index=False))
    print("-" * 60)
    
    conn.close()

if __name__ == "__main__":
    predecir_fixture()