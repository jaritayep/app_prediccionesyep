import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def entrenar_ia_super_pro():
    conn = sqlite3.connect('database_partidos.db')
    # Nota: Asegúrate de que 'dataset_entrenamiento_ia' tenga las columnas de xG
    df = pd.read_sql("SELECT * FROM dataset_entrenamiento_ia", conn)
    
    # --- INGENIERÍA DE VARIABLES (xG) ---
    # Creamos métricas de eficiencia: ¿El equipo aprovecha sus chances?
    # Usamos un pequeño valor (0.01) para evitar división por cero
    df['Home_Efficiency'] = df['Home_FTHG'] / (df['xG_home'] + 0.01)
    df['Away_Efficiency'] = df['Home_FTAG'] / (df['xG_away'] + 0.01)
    
    # --- VARIABLES ACTUALIZADAS ---
    features = [
        'Home_FTHG', 'Home_FTAG', 
        'Home_HS', 'Home_AS', 
        'Home_HST', 'Home_AST', 
        'Home_HC', 'Home_AC', 
        'Home_HY', 'Home_AY',
        'xG_home', 'xG_away',        # Agregamos xG directo
        'Home_Efficiency',           # Agregamos Eficiencia
        'Away_Efficiency'
    ]
    
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    # --- LÓGICA DE TIME DECAY ---
    df['Date'] = pd.to_datetime(df['Date'])
    fecha_reciente = df['Date'].max()
    df['dias_antiguedad'] = (fecha_reciente - df['Date']).dt.days
    
    # Mantengo tu factor 400 que es muy sólido
    df['peso_temporal'] = np.exp(-df['dias_antiguedad'] / 400)
    
    # Limpiamos filas con NaNs en las nuevas variables
    df_ml = df[features + ['Target', 'peso_temporal']].dropna()
    
    X = df_ml[features]
    y = df_ml['Target']
    weights = df_ml['peso_temporal']

    # Separación
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    weights_train = weights.iloc[idx_train]

    # Ajuste de hiperparámetros para las nuevas variables
    model = RandomForestClassifier(
        n_estimators=300,        # Subimos un poco por la mayor complejidad
        max_depth=12,            # Aumentamos ligeramente para que entienda la relación xG vs Goles
        min_samples_leaf=5,      # Evita que el modelo sea demasiado "ruidoso"
        random_state=42,
        class_weight='balanced'
    )
    
    # --- ENTRENAMIENTO ---
    model.fit(X_train, y_train, sample_weight=weights_train)

    score = model.score(X_test, y_test)
    print(f"🔥 SCORE CON xG Y PESO TEMPORAL: {score:.2%}")

    # --- IMPORTANCIA DE VARIABLES ---
    # Esto te dirá qué tanto está pesando el xG en tus predicciones
    importancias = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print("\n📊 Top Variables:")
    print(importancias.sort_values(by='importance', ascending=False).head(5))

    joblib.dump(model, './modelo_ia.pkl')
    conn.close()
    print("\n✅ Modelo guardado con éxito como 'modelo_ia.pkl'")

if __name__ == "__main__":
    entrenar_ia_super_pro()