import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==========================================
# 1. EXTRACCIÓN Y LIMPIEZA DE DATOS
# ==========================================
def cargar_datos_limpios():
    print("📥 Extrayendo datos de la base local...")
    conn = sqlite3.connect('database_partidos.db')
    
    # xG excluido del modelo: 86 % de filas tienen xG=0 (almacenado como 0, no NaN).
    # Incluirlo crea un desfase entrenamiento/inferencia porque en inferencia se usan
    # promedios reales (0.3–1.5). Se revisará cuando la cobertura de xG supere el 50 %.
    query = """
        SELECT Date, Torneo, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HST, AST, HC, AC
        FROM historial_selecciones_ml
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Eliminamos cualquier fila que todavía tenga valores nulos en tiros o córners
    df = df.dropna(subset=['HST', 'AST', 'HC', 'AC'])

    # Filtramos solo filas con datos reales de tiros (descartamos partidos sin estadísticas)
    df = df[(df['HST'] + df['AST']) > 0]
    
    # Convertimos la fecha para ordenar cronológicamente
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Datos cargados: {len(df)} partidos 100% limpios con datos de tiros reales.")
    return df

# ==========================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (Feature Engineering)
# ==========================================
def preparar_features(df):
    print("⚙️ Construyendo variables predictivas...")
    
    # Codificamos los nombres de los equipos a números para que la IA los entienda
    le_equipos = LabelEncoder()
    
    # Unimos todos los equipos para tener un vocabulario único
    todos_los_equipos = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le_equipos.fit(todos_los_equipos)
    
    df['HomeTeam_Code'] = le_equipos.transform(df['HomeTeam'])
    df['AwayTeam_Code'] = le_equipos.transform(df['AwayTeam'])
    
    # Mapeamos el resultado final a números (H=2, D=1, A=0)
    mapa_resultados = {'H': 2, 'D': 1, 'A': 0}
    df['Target'] = df['FTR'].map(mapa_resultados)

    # Variables de entrada para el modelo (6 features, sin xG)
    # xG eliminado: cobertura insuficiente (~14 % de filas con datos reales).
    # Revisitar cuando la cobertura supere el 50 %.
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'HST', 'AST', 'HC', 'AC']
    
    X = df[features]
    y = df['Target']
    
    return X, y, le_equipos

# ==========================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO
# ==========================================
def entrenar_modelo():
    df = cargar_datos_limpios()
    X, y, le_equipos = preparar_features(df)
    
    # Separamos en datos de entrenamiento (80%) y prueba (20%)
    # shuffle=False es CRÍTICO en deportes: entrenamos con el pasado para predecir el futuro
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("🧠 Entrenando Random Forest Classifier con calibración isotónica...")
    # Pasamos el estimador SIN ajustar directamente a CalibratedClassifierCV.
    # Así el wrapper gestiona internamente los 5 folds: en cada fold entrena una
    # instancia nueva del RF y la calibra, evitando el ajuste redundante previo.
    # class_weight='balanced' corrige el desbalance (47 % local vs 30 % visitante).
    modelo_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    modelo = CalibratedClassifierCV(modelo_rf, method='isotonic', cv=5)
    modelo.fit(X_train, y_train)
    
    # Evaluamos el rendimiento en los datos de prueba (el futuro que el modelo no ha visto)
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    
    print("\n📊 RESULTADOS DE LA IA EN DATOS INÉDITOS:")
    print(f"Precisión Global (Accuracy): {precision:.2%}")
    print("-" * 40)
    print("Reporte detallado (0=Visita, 1=Empate, 2=Local):")
    print(classification_report(y_test, predicciones, zero_division=0))
    
    # ==========================================
    # 4. EXPORTACIÓN DEL CEREBRO
    # ==========================================
    # Guardamos el modelo entrenado y el codificador de equipos en archivos físicos
    archivos_exportados = {
        'modelo': 'modelo_selecciones_rf.pkl',
        'encoder': 'encoder_equipos_selecciones.pkl'
    }
    
    joblib.dump(modelo, archivos_exportados['modelo'])
    joblib.dump(le_equipos, archivos_exportados['encoder'])
    
    print(f"\n💾 ¡Sistema Guardado! Archivos generados:")
    print(f" - {archivos_exportados['modelo']} (El algoritmo predictivo)")
    print(f" - {archivos_exportados['encoder']} (El traductor de nombres de países)")

if __name__ == "__main__":
    entrenar_modelo()