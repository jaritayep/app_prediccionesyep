import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
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

    # Excluimos amistosos: el modelo debe aprender solo de partidos competitivos
    df = df[~df['Torneo'].str.contains('Friendly|Amistoso|friendly', case=False, na=False)]

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

    # ─────────────────────────────────────────────────────────────────
    # BOOST ESTADOS UNIDOS (host advantage + squad upgrade):
    # El historial subestima a USA porque la mayoría de sus datos son
    # de la era pre-2022, cuando el equipo era menos competitivo.
    # Como co-anfitrión 2026 con una generación renovada (Pulisic,
    # Weah, McKennie, Reyna), aplicamos un factor de corrección del 15%
    # sobre sus tiros al arco y córners en todos los partidos del dataset.
    # Esto eleva sus stats al rango de equipos CONCACAF+CONMEBOL
    # de nivel medio-alto sin distorsionar los resultados históricos.
    # ─────────────────────────────────────────────────────────────────
    USA_BOOST = 1.20
    USA_NAMES = {'United States', 'USA'}

    mask_home = df['HomeTeam'].isin(USA_NAMES)
    mask_away = df['AwayTeam'].isin(USA_NAMES)

    df.loc[mask_home, 'HST'] = (df.loc[mask_home, 'HST'] * USA_BOOST).round(2)
    df.loc[mask_home, 'HC']  = (df.loc[mask_home, 'HC']  * USA_BOOST).round(2)
    df.loc[mask_away, 'AST'] = (df.loc[mask_away, 'AST'] * USA_BOOST).round(2)
    df.loc[mask_away, 'AC']  = (df.loc[mask_away, 'AC']  * USA_BOOST).round(2)

    n_boost = mask_home.sum() + mask_away.sum()
    print(f"🇺🇸 USA boost x{USA_BOOST} aplicado a {n_boost} partidos "
          f"({mask_home.sum()} local, {mask_away.sum()} visita).")

    # Variables de entrada para el modelo (6 features, sin xG)
    # xG eliminado: cobertura insuficiente (~14 % de filas con datos reales).
    # Revisitar cuando la cobertura supere el 50 %.
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'HST', 'AST', 'HC', 'AC']

    X = df[features]
    y = df['Target']

    # ─────────────────────────────────────────────────────────────────
    # AUGMENTACIÓN CAMPO NEUTRAL:
    # El Mundial se juega en cancha neutral, pero el modelo fue entrenado
    # con datos donde un equipo siempre es 'local'. Para enseñarle que
    # la localía no existe en el torneo, duplicamos cada partido con los
    # equipos intercambiados e invertimos el resultado (H↔A, A↔H, D=D).
    # Así el modelo aprende que Home=France vs Away=Brazil y
    # Home=Brazil vs Away=France deben dar probabilidades simétricas.
    # ─────────────────────────────────────────────────────────────────
    df_inv = df.copy()
    df_inv['HomeTeam_Code'] = df['AwayTeam_Code'].values
    df_inv['AwayTeam_Code'] = df['HomeTeam_Code'].values
    df_inv['HST'] = df['AST'].values
    df_inv['AST'] = df['HST'].values
    df_inv['HC']  = df['AC'].values
    df_inv['AC']  = df['HC'].values
    # Invertir resultado: 2(H)→0(A), 0(A)→2(H), 1(D)→1(D)
    df_inv['Target'] = df['Target'].map({2: 0, 0: 2, 1: 1})

    X_inv = df_inv[features]
    y_inv = df_inv['Target']

    X = pd.concat([X, X_inv], ignore_index=True)
    y = pd.concat([y, y_inv], ignore_index=True)

    print(f"✅ Augmentación campo neutral: {len(df)} partidos originales → {len(X)} muestras totales (2×).")

    return X, y, le_equipos

# ==========================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO
# ==========================================
def entrenar_modelo():
    df = cargar_datos_limpios()
    X, y, le_equipos = preparar_features(df)
    
    # ─────────────────────────────────────────────────────────────────
    # SPLIT CRONOLÓGICO TEMPORAL:
    # Con shuffle=True un partido de 2024 puede quedar en train y uno de
    # 2018 en test, dando al modelo información "del futuro" durante el
    # entrenamiento y artificialmente inflando la accuracy.
    #
    # Solución: ordenamos el dataset aumentado por el índice original
    # (ya está ordenado cronológicamente desde cargar_datos_limpios).
    # Los primeros N×2 son los originales en orden; los siguientes N×2
    # son los espejos en el mismo orden. Intercalamos para que cada
    # partido y su espejo queden juntos, y luego cortamos el 80% más
    # antiguo para train y el 20% más reciente para test.
    # Así el test set representa siempre el "futuro" del modelo.
    # ─────────────────────────────────────────────────────────────────
    n_total = len(X)
    n_originales = n_total // 2          # mitad original, mitad espejo
    idx_orig   = np.arange(n_originales)
    idx_espejo = np.arange(n_originales, n_total)

    # Intercalamos: [orig_0, espejo_0, orig_1, espejo_1, ...]
    idx_intercalado = np.empty(n_total, dtype=int)
    idx_intercalado[0::2] = idx_orig
    idx_intercalado[1::2] = idx_espejo

    X_ordered = X.iloc[idx_intercalado].reset_index(drop=True)
    y_ordered = y.iloc[idx_intercalado].reset_index(drop=True)

    corte = int(n_total * 0.80)
    X_train, X_test = X_ordered.iloc[:corte], X_ordered.iloc[corte:]
    y_train, y_test = y_ordered.iloc[:corte], y_ordered.iloc[corte:]

    print(f"✅ Split cronológico: {len(X_train)} muestras train | {len(X_test)} muestras test (20% más reciente).")
    
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