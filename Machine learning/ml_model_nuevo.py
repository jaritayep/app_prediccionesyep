import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def entrenar_ia_super_pro():
    conn = sqlite3.connect('database_partidos.db')
    # Traemos también la fecha para calcular la antigüedad
    df = pd.read_sql("SELECT * FROM dataset_entrenamiento_ia", conn)
    
    # --- VARIABLES ACTUALIZADAS ---
    features = [
        'Home_FTHG', 'Home_FTAG', 'Home_HS', 'Home_AS', 
        'Home_HST', 'Home_AST', 'Home_HC', 'Home_AC', 'Home_HY', 'Home_AY'
    ]
    
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    # --- LÓGICA DE TIME DECAY (PESO POR ANTIGÜEDAD) ---
    df['Date'] = pd.to_datetime(df['Date'])
    fecha_reciente = df['Date'].max()
    
    # Calculamos días de diferencia
    df['dias_antiguedad'] = (fecha_reciente - df['Date']).dt.days
    
    # Función de peso: np.exp(-dias / factor)
    # Factor 400 significa que un partido de hace ~1 año (365 días) 
    # tiene la mitad de importancia que uno de hoy.
    df['peso_temporal'] = np.exp(-df['dias_antiguedad'] / 400)
    
    df_ml = df[features + ['Target', 'peso_temporal']].dropna()
    
    X = df_ml[features]
    y = df_ml['Target']
    weights = df_ml['peso_temporal']

    # Separamos manteniendo los índices para los pesos
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    # El peso de los datos de entrenamiento
    weights_train = weights.iloc[idx_train]

    # Modelo balanceado
    model = RandomForestClassifier(
        n_estimators=250, # Subimos un poco para captar el peso temporal
        max_depth=10,     # Reducimos profundidad para evitar que memorice el pasado (overfitting)
        random_state=42,
        class_weight='balanced' # Ayuda si hay pocos empates en los datos
    )
    
    # --- ENTRENAMIENTO CON PESOS ---
    model.fit(X_train, y_train, sample_weight=weights_train)

    score = model.score(X_test, y_test)
    print(f"🔥 SCORE CON PESO TEMPORAL: {score:.2%}")

    joblib.dump(model, './modelo_ia.pkl')
    conn.close()
    print("✅ Modelo guardado con éxito como 'modelo_ia.pkl'")

if __name__ == "__main__":
    entrenar_ia_super_pro()