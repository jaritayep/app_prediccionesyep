import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_ia():
    # 1. CONEXIÓN SEGURA
    # Buscamos el archivo en la misma carpeta donde vive este script
    db_name = 'premier_analytics_v3.db'
    
    if not os.path.exists(db_name):
        print(f"❌ Error: No se encuentra el archivo {db_name} en esta carpeta.")
        print(f"Directorio actual: {os.getcwd()}")
        return

    conn = sqlite3.connect(db_name)
    
    try:
        # 2. CARGA DE DATOS
        print("Leyendo datos históricos de la base de datos...")
        df = pd.read_sql("SELECT * FROM historial_multiliga_ml", conn)
        conn.close()

        # --- 3. PREPARACIÓN DE DATOS (Feature Engineering) ---
        # Definimos nuestro objetivo: Ganador (H=2, D=1, A=0)
        df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
        
        # Seleccionamos las variables de las que la IA aprenderá
        # HS: Tiros Local, AS: Tiros Visita, HST: Tiros a puerta Local, etc.
        features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A']
        
        # Limpieza rápida: eliminamos filas con nulos en estas columnas
        df_ml = df[features + ['Target']].dropna()
        
        X = df_ml[features]
        y = df_ml['Target']

        # --- 4. EL MODELO (Random Forest) ---
        print(f"Entrenando modelo con {len(df_ml)} partidos de toda Europa...")
        
        # Usamos 100 árboles de decisión para mayor precisión
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- 5. GUARDAR EL CEREBRO ---
        joblib.dump(model, 'modelo_ia.pkl')
        print("✅ ¡Éxito! Modelo entrenado y guardado como 'modelo_ia.pkl'")
        
    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")
        if "no such table" in str(e):
            print("Tip: Asegúrate de haber corrido el script de descarga histórica primero.")

if __name__ == "__main__":
    entrenar_ia()