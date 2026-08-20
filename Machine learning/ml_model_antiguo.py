import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def entrenar_ia_super_pro():
    conn = sqlite3.connect('database_partidos.db')
    df = pd.read_sql("SELECT * FROM dataset_entrenamiento_ia", conn)
    
    # Parche de rescate para el historial
    df['Home_xG_home'] = df['Home_xG_home'].fillna(df['Home_FTHG'])
    df['Home_xG_away'] = df['Home_xG_away'].fillna(df['Home_FTAG'])
    df['Home_xG_Diff'] = df['Home_xG_home'] - df['Home_xG_away']
    df['Home_Efficiency'] = df['Home_Efficiency'].fillna(1.0) 
    df['Diferencia_Tabla'] = df['Diferencia_Tabla'].fillna(0)
    df['Ventaja_Fisica'] = df['Ventaja_Fisica'].fillna(0)

    # --- LAS 16 VARIABLES DEFINITIVAS ---
    features = [
        'Home_FTHG', 'Home_FTAG',     
        'Home_HS', 'Home_AS',         
        'Home_HST', 'Home_AST',       
        'Home_HC', 'Home_AC',         
        'Home_HY', 'Home_AY',         
        'Home_xG_home', 'Home_xG_away', 
        'Home_Efficiency',            
        'Home_xG_Diff',               
        'Diferencia_Tabla',
        'Ventaja_Fisica'              # ¡La nueva métrica de fatiga!
    ]
    
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    df['Date'] = pd.to_datetime(df['Date'])
    fecha_reciente = df['Date'].max()
    df['dias_antiguedad'] = (fecha_reciente - df['Date']).dt.days
    df['peso_temporal'] = np.exp(-df['dias_antiguedad'] / 400)
    
    df_ml = df[features + ['Target', 'peso_temporal']].dropna()
    print(f"📚 Entrenando la IA Definitiva con {len(df_ml)} partidos válidos...")
    
    X = df_ml[features]
    y = df_ml['Target']
    weights = df_ml['peso_temporal']

    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    weights_train = weights.iloc[idx_train]

    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12,            
        min_samples_leaf=5,      
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train, sample_weight=weights_train)
    score = model.score(X_test, y_test)
    print(f"🔥 SCORE FINAL (16 Variables): {score:.2%}")

    joblib.dump(model, './modelo_ia.pkl')
    conn.close()
    print("✅ Modelo guardado con éxito como 'modelo_ia.pkl'")

if __name__ == "__main__":
    entrenar_ia_super_pro()