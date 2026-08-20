"""
Módulo compartido de features para el modelo de predicción de partidos.

IMPORTANTE: tanto ml_model_nuevo.py (entrenamiento) como visualizaciones.py
(inferencia) DEBEN importar todo desde aquí en vez de recalcular sus propias
versiones de "forma reciente". Ese fue exactamente el bug original: el
training usaba stats crudas del propio partido y visualizaciones.py usaba
promedios de partidos pasados SIN corregir por local/visita — dos contratos
de features distintos alimentando el mismo modelo. Si un archivo cambia esta
lógica sin tocar el otro, el problema vuelve a aparecer.
"""
import numpy as np
from collections import deque

# ── Constantes del "contrato" de features. Cambiar aquí, no en los otros archivos. ──
N_FORMA = 5                          # nº de partidos recientes usados por equipo
PESOS_FORMA = np.array([5, 4, 3, 2, 1])  # más peso al partido más reciente
MESES_TABLA = 9                      # ventana para puntos de "tabla" (forma de temporada)
DESCANSO_MIN, DESCANSO_MAX = 3, 14   # límites razonables para días de descanso

FEATURE_STATS = [
    'goles_favor', 'goles_contra',
    'tiros_favor', 'tiros_contra',
    'tiros_arco_favor', 'tiros_arco_contra',
    'corners_favor', 'corners_contra',
    'amarillas_favor', 'amarillas_contra',
    'xg_favor', 'xg_contra',
]

# Valores por defecto cuando un equipo no tiene historial suficiente
# (equipo recién ascendido, debut en la base de datos, etc.)
DEFAULT_FORMA = {
    'goles_favor': 0.0, 'goles_contra': 0.0,
    'tiros_favor': 0.0, 'tiros_contra': 0.0,
    'tiros_arco_favor': 0.0, 'tiros_arco_contra': 0.0,
    'corners_favor': 0.0, 'corners_contra': 0.0,
    'amarillas_favor': 0.0, 'amarillas_contra': 0.0,
    'xg_favor': 1.0, 'xg_contra': 1.0,
}


def nuevo_historial():
    """Deque vacío con el tamaño correcto para acumular la forma de un equipo."""
    return deque(maxlen=N_FORMA)


def perspectiva_equipo(row, es_local):
    """Convierte una fila del historial (Home*/Away*) a la perspectiva
    'propio vs rival' de UN equipo, sin importar si jugó de local o visita.

    Este es el fix del bug de get_recent_stats(): antes se promediaban las
    columnas HomeTeam/AwayTeam tal cual, mezclando partidos donde el equipo
    fue local con partidos donde fue visita, sin corregir la perspectiva.
    """
    g = lambda h, a: row[h] if es_local else row[a]
    return {
        'goles_favor':       g('FTHG', 'FTAG'),
        'goles_contra':      g('FTAG', 'FTHG'),
        'tiros_favor':       g('HS', 'AS'),
        'tiros_contra':      g('AS', 'HS'),
        'tiros_arco_favor':  g('HST', 'AST'),
        'tiros_arco_contra': g('AST', 'HST'),
        'corners_favor':     g('HC', 'AC'),
        'corners_contra':    g('AC', 'HC'),
        'amarillas_favor':   g('HY', 'AY'),
        'amarillas_contra':  g('AY', 'HY'),
        'xg_favor':          g('xG_home', 'xG_away'),
        'xg_contra':         g('xG_away', 'xG_home'),
    }


def promedio_ponderado(historial_deque):
    """Promedio ponderado (más peso a lo reciente) sobre un deque de dicts
    ya en perspectiva propia. historial_deque[0] = partido más reciente."""
    n = len(historial_deque)
    if n == 0:
        return dict(DEFAULT_FORMA)
    pesos = PESOS_FORMA[:n]
    w = pesos / pesos.sum()
    return {
        stat: float(np.average([d[stat] for d in historial_deque], weights=w))
        for stat in FEATURE_STATS
    }


def construir_fila_features(forma_h, forma_a, dif_tabla, ventaja_fisica):
    """Ensambla el dict de features finales a partir de la forma reciente de
    ambos equipos. Un dict de Python conserva el orden de inserción, así que
    esto define también el orden estable de columnas del modelo."""
    fila = {}
    for stat in FEATURE_STATS:
        fila[f'H_{stat}'] = forma_h[stat]
        fila[f'A_{stat}'] = forma_a[stat]
    fila['Efficiency']       = forma_h['goles_favor'] / (forma_h['xg_favor'] + 0.01)
    fila['xG_Diff']          = forma_h['xg_favor'] - forma_a['xg_favor']
    fila['Diferencia_Tabla'] = dif_tabla
    fila['Ventaja_Fisica']   = ventaja_fisica
    return fila