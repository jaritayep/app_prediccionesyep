import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def entrenar_ia_super_pro():
    conn = sqlite3.connect('database_partidos.db')
    df = pd.read_sql("SELECT * FROM dataset_entrenamiento_ia", conn)
    
    # --- LAS 15 VARIABLES DEFINITIVAS ---
    features = [
        'Home_FTHG', 'Home_FTAG',     # Goles
        'Home_HS', 'Home_AS',         # Tiros
        'Home_HST', 'Home_AST',       # Tiros al arco
        'Home_HC', 'Home_AC',         # Córners
        'Home_HY', 'Home_AY',         # Tarjetas amarillas
        'Home_xG_home', 'Home_xG_away', # Goles Esperados
        'Home_Efficiency',            # Eficiencia del Local
        'Home_xG_Diff',               # Dominio (Local xG - Visita xG)
        'Diferencia_Tabla'            # Jerarquía (Puntos Local - Puntos Visita)
    ]
    
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    # --- LÓGICA DE TIME DECAY ---
    df['Date'] = pd.to_datetime(df['Date'])
    fecha_reciente = df['Date'].max()
    df['dias_antiguedad'] = (fecha_reciente - df['Date']).dt.days
    df['peso_temporal'] = np.exp(-df['dias_antiguedad'] / 400)
    
    # Limpiamos nulos asegurando que usemos las 15 columnas
    df_ml = df[features + ['Target', 'peso_temporal']].dropna()
    
    X = df_ml[features]
    y = df_ml['Target']
    weights = df_ml['peso_temporal']

    # Separación de datos
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    weights_train = weights.iloc[idx_train]

    # Modelo balanceado
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12,            
        min_samples_leaf=5,      
        random_state=42,
        class_weight='balanced'
    )
    
    # Entrenamiento
    model.fit(X_train, y_train, sample_weight=weights_train)
    score = model.score(X_test, y_test)
    print(f"🔥 SCORE CON TABLA DE POSICIONES Y xG: {score:.2%}")

    joblib.dump(model, './modelo_ia.pkl')
    conn.close()
    print("✅ Modelo guardado con éxito como 'modelo_ia.pkl'")

if __name__ == "__main__":
    entrenar_ia_super_pro()