import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

from ia_features import (
    N_FORMA, MESES_TABLA, DESCANSO_MIN, DESCANSO_MAX,
    nuevo_historial, perspectiva_equipo, promedio_ponderado, construir_fila_features,
)

# ── Rutas basadas en la ubicación real del archivo, no en el cwd desde donde ──
# se ejecute el script. Esto evita el bug de que sqlite3.connect() cree una DB
# vacía nueva si corres el script parado en la carpeta equivocada.
BASE_DIR = Path(__file__).resolve().parent      # machine learning/
ROOT_DIR = BASE_DIR.parent                       # proyecto app/ (donde vive la DB real)

DB_NAME    = ROOT_DIR / 'database_partidos.db'
HIST_TABLE = 'historial_multiliga_ml'   # misma tabla que usa visualizaciones.py en producción
MODEL_PATH = ROOT_DIR / 'modelo_ia.pkl'  # se guarda directo donde visualizaciones.py lo busca


def entrenar_ia_super_pro():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f"SELECT * FROM {HIST_TABLE}", conn)
    conn.close()

    columnas_requeridas = ['Date', 'HomeTeam', 'AwayTeam', 'FTR',
                            'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
                            'HC', 'AC', 'HY', 'AY']
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {HIST_TABLE}: {faltantes}")

    # format='mixed' porque la columna Date tiene filas con hora ("...00:00:00")
    # y filas solo con fecha ("YYYY-MM-DD") mezcladas. .dt.normalize() trunca
    # todo a medianoche pero mantiene el tipo datetime64 (necesario para las
    # restas de fechas con pd.Timedelta más abajo, a diferencia de .dt.date).
    df['Date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date').reset_index(drop=True)

    # Parche de xG: si falta, se usa el gol real anotado EN ESE partido (mismo criterio
    # que antes), pero esto solo afecta al estado histórico que alimenta la forma futura,
    # nunca se usa como feature del propio partido que se está prediciendo.
    if 'xG_home' not in df.columns: df['xG_home'] = np.nan
    if 'xG_away' not in df.columns: df['xG_away'] = np.nan
    df['xG_home'] = df['xG_home'].fillna(df['FTHG'])
    df['xG_away'] = df['xG_away'].fillna(df['FTAG'])

    # ── Estado incremental por equipo, recorrido en orden cronológico ──
    # Para cada partido, leemos el estado ANTES de actualizarlo con su propio
    # resultado. Así el partido nunca se ve a sí mismo como feature (sin fuga).
    forma        = defaultdict(nuevo_historial)
    pts_eventos  = defaultdict(list)     # [(fecha, puntos), ...] dentro de la ventana
    pts_suma     = defaultdict(float)
    ultima_fecha = {}
    ventana_tabla = pd.Timedelta(days=30 * MESES_TABLA)

    filas, targets, fechas = [], [], []

    for _, row in df.iterrows():
        home, away, fecha = row['HomeTeam'], row['AwayTeam'], row['Date']

        # 1. Forma reciente ANTES de este partido (misma lógica que get_recent_stats
        #    en producción, ya con perspectiva local/visita corregida)
        forma_h = promedio_ponderado(forma[home])
        forma_a = promedio_ponderado(forma[away])

        # 2. Puntos de "tabla" dentro de la ventana de meses (podar eventos viejos)
        for equipo in (home, away):
            eventos = pts_eventos[equipo]
            while eventos and (fecha - eventos[0][0]) > ventana_tabla:
                _, pts_viejos = eventos.pop(0)
                pts_suma[equipo] -= pts_viejos
        dif_tabla = pts_suma[home] - pts_suma[away]

        # 3. Días de descanso ANTES de actualizar la última fecha jugada
        descanso_h = (min(max((fecha - ultima_fecha[home]).days, DESCANSO_MIN), DESCANSO_MAX)
                      if home in ultima_fecha else 7)
        descanso_a = (min(max((fecha - ultima_fecha[away]).days, DESCANSO_MIN), DESCANSO_MAX)
                      if away in ultima_fecha else 7)
        ventaja_fisica = descanso_h - descanso_a

        filas.append(construir_fila_features(forma_h, forma_a, dif_tabla, ventaja_fisica))
        targets.append(row['FTR'])
        fechas.append(fecha)

        # 4. AHORA sí, actualizar el estado con el resultado de este partido
        #    (queda disponible para partidos futuros, nunca para este)
        forma[home].appendleft(perspectiva_equipo(row, es_local=True))
        forma[away].appendleft(perspectiva_equipo(row, es_local=False))

        pts_h = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        pts_a = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        pts_eventos[home].append((fecha, pts_h)); pts_suma[home] += pts_h
        pts_eventos[away].append((fecha, pts_a)); pts_suma[away] += pts_a
        ultima_fecha[home] = fecha
        ultima_fecha[away] = fecha

    df_feat = pd.DataFrame(filas)
    df_feat['Target'] = pd.Series(targets).map({'H': 2, 'D': 1, 'A': 0}).values
    df_feat['Date']   = pd.Series(fechas).values
    df_feat['dias_antiguedad'] = (df_feat['Date'].max() - df_feat['Date']).dt.days
    df_feat['peso_temporal']   = np.exp(-df_feat['dias_antiguedad'] / 400)

    feature_cols = [c for c in df_feat.columns if c not in ('Target', 'Date', 'peso_temporal', 'dias_antiguedad')]

    antes = len(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols + ['Target'])
    despues = len(df_feat)
    if despues < antes:
        print(f"⚠️  Se descartaron {antes - despues} partidos con datos incompletos ({despues} válidos)")

    print(f"📚 Entrenando con {despues} partidos y {len(feature_cols)} variables (features sin fuga de datos)...")

    # ── Split TEMPORAL, no aleatorio: se evalúa sobre los partidos más
    #    recientes, nunca sobre partidos anteriores a los de entrenamiento ──
    df_feat = df_feat.sort_values('Date').reset_index(drop=True)
    corte = int(len(df_feat) * 0.8)
    train, test = df_feat.iloc[:corte], df_feat.iloc[corte:]

    X_train, y_train, w_train = train[feature_cols], train['Target'], train['peso_temporal']
    X_test,  y_test           = test[feature_cols],  test['Target']

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    score = model.score(X_test, y_test)
    print(f"🔥 SCORE (evaluado en los últimos {len(test)} partidos, orden cronológico): {score:.2%}")
    print("   (Este número ahora debería ser más bajo que antes — el score previo estaba")
    print("    inflado porque el modelo veía, indirectamente, el resultado del propio partido.)")

    # Se guarda el modelo JUNTO al orden exacto de columnas que espera, para que
    # visualizaciones.py nunca tenga que adivinar/hardcodear el orden del vector.
    joblib.dump({'model': model, 'feature_cols': feature_cols}, MODEL_PATH)
    print(f"✅ Modelo guardado en '{MODEL_PATH}' junto con el orden exacto de {len(feature_cols)} columnas")


if __name__ == "__main__":
    entrenar_ia_super_pro()