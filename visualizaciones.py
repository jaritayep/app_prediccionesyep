import streamlit as st
import sqlite3
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os
from thefuzz import process, fuzz
import math
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import re
from diccionario_alias import ALIAS_GLOBAL, normalizar_nombre
import sys
# Agrega la carpeta Machine learning/ al path para poder importar ia_features
# sin duplicar el archivo (single source of truth para el contrato de features)
sys.path.insert(0, str(Path(__file__).resolve().parent / "Machine learning"))

from ia_features import (
    N_FORMA, PESOS_FORMA, FEATURE_STATS, DEFAULT_FORMA,
    perspectiva_equipo, promedio_ponderado, construir_fila_features,
)


def poisson_prob(lamba_val, k):
    """Calcula la probabilidad de que ocurran exactamente k eventos"""
    if lamba_val <= 0: return 0
    return (math.exp(-lamba_val) * (lamba_val**k)) / math.factorial(k)

def poisson_prob_over(promedio, umbral):
    """Calcula la probabilidad de que ocurra MÁS que el umbral (distribución Poisson).
    Renombrada desde prob_over para evitar colisión con la variable local prob_over = 0.5
    que se define más abajo en el scope de Streamlit."""
    if promedio <= 0: return 0.05
    prob_acumulada = 0
    # Sumamos las probabilidades de 0 hasta el umbral
    for k in range(int(umbral) + 1):
        prob_acumulada += poisson_prob(promedio, k)
    return 1 - prob_acumulada

prob_over = poisson_prob_over  # alias global: varias secciones llaman a prob_over(...) como función

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="AI Betting Lab Pro", page_icon="⚽")
DB_NAME = 'database_partidos.db'
MODEL_PATH = "modelo_ia.pkl"

st.markdown("""
    <style>
    /* 1. Tus estilos actuales */
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); color: white; }
    footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stMetric { background-color: #1e2129; padding: 10px; border-radius: 10px; }

    /* 2. El truco para evitar el teclado en el celular */
    .stSelectbox div[role='combobox'] input {
        caret-color: transparent;
        cursor: default;
    }
    </style>
    """, unsafe_allow_html=True)

CONFIG_FIJA = {'staticPlot': False, 'scrollZoom': False, 'doubleClick': 'reset', 'displayModeBar': False, 'showAxisDragHandles': False}

def corregir_nombre_equipo(nombre_api, lista_db):
    """Normaliza un nombre de equipo: primero via alias exacto, luego fuzzy matching contra la DB."""
    nombre_norm = normalizar_nombre(nombre_api)  # exact alias lookup first
    if not lista_db: return nombre_norm
    # If the alias-resolved name is directly in the DB, use it as-is
    if nombre_norm in lista_db:
        return nombre_norm
    # Otherwise fall back to fuzzy matching.
    # NOTA: usamos token_sort_ratio, NO token_set_ratio. token_set_ratio infla
    # el score cuando dos nombres comparten una sola palabra genérica ("City",
    # "United", "Real", etc.), lo que causaba falsos positivos graves
    # (ej: "Coventry City" -> "Man City" con score 67, y "Sociedad Real" ->
    # "Sociedad" con score 100). token_sort_ratio exige que el resto del
    # nombre también se parezca, y sigue detectando variantes reales
    # (typos, orden de palabras, abreviaciones no cubiertas por el alias).
    mejor_match, score = process.extractOne(nombre_norm.strip(), lista_db, scorer=fuzz.token_sort_ratio)
    return mejor_match if score > 72 else nombre_norm

def cargar_modelo():
    """Devuelve el bundle completo {'model': ..., 'feature_cols': [...]} que guarda
    ml_model_nuevo.py via joblib.dump(). Ya NO devuelve el modelo pelado —
    quien llame a esto debe desempaquetar bundle['model'] y bundle['feature_cols']."""
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def construir_forma_reciente(equipo, conn, hist_table='historial_multiliga_ml'):
    """Forma reciente de un equipo con perspectiva local/visita YA CORREGIDA,
    usando perspectiva_equipo() + promedio_ponderado() de ia_features — el
    MISMO contrato de features que usa ml_model_nuevo.py al entrenar.

    OJO: get_recent_stats() de más abajo NO corrige esta perspectiva (promedia
    columnas HomeTeam/AwayTeam crudas tal cual), por eso NO debe alimentar al
    modelo — solo sirve para los gráficos/stats de display de la UI."""
    q = (f'SELECT HomeTeam, AwayTeam, "FTHG", "FTAG", "HS", "AS", "HST", "AST", '
         f'"HC", "AC", "HY", "AY", "xG_home", "xG_away" '
         f'FROM {hist_table} '
         f'WHERE HomeTeam="{equipo}" OR AwayTeam="{equipo}" '
         f'ORDER BY Date DESC LIMIT {N_FORMA}')
    res = pd.read_sql(q, conn)
    if res.empty:
        return dict(DEFAULT_FORMA)
    res['xG_home'] = res['xG_home'].fillna(res['FTHG'])
    res['xG_away'] = res['xG_away'].fillna(res['FTAG'])
    res = res.fillna(0.0)
    historial = [perspectiva_equipo(row, es_local=(row['HomeTeam'] == equipo))
                 for _, row in res.iterrows()]
    return promedio_ponderado(historial)

def get_recent_stats(equipo, conn):
    # 1. Coma eliminada y columnas de xG añadidas
    q = f'SELECT "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "xG_home", "xG_away" FROM historial_multiliga_ml WHERE HomeTeam="{equipo}" OR AwayTeam="{equipo}" ORDER BY Date DESC LIMIT 5'
    res = pd.read_sql(q, conn)
    
    if res.empty: 
        return pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0], 
            index=['FTHG','FTAG','HS','AS','HST','AST','HC','AC','HY','AY', 'xG_home', 'xG_away']
        )
        
    pesos = np.array([5, 4, 3, 2, 1])[:len(res)]
    
    # 3. Limpieza preventiva: Si por algún motivo hay un hueco sin xG, lo rellenamos con 1.0 para que np.average no falle
    res = res.fillna(1.0)
    
    return pd.Series({col: np.average(res[col], weights=pesos/pesos.sum()) for col in res.columns})
def get_recent_stats_wc(equipo, conn):
    """Stats ponderadas (últimas 5 partidas) de una selección, desde historial_selecciones_ml.
    Corrige la perspectiva: FTHG/HST/HC son del equipo cuando juega de local,
    FTAG/AST/AC son del equipo cuando juega de visita.
    """
    q = (f'SELECT HomeTeam,AwayTeam,"FTHG","FTAG","HST","AST","HC","AC" '
         f'FROM historial_selecciones_ml '
         f'WHERE HomeTeam="{equipo}" OR AwayTeam="{equipo}" '
         f'ORDER BY Date DESC LIMIT 5')
    res = pd.read_sql(q, conn)
    if res.empty:
        return pd.Series([0.0, 0.0, 4.0, 3.5, 4.5, 4.0],
                         index=['FTHG', 'FTAG', 'HST', 'AST', 'HC', 'AC'])
    res = res.fillna(0.0)
    pesos = np.array([5, 4, 3, 2, 1])[:len(res)]
    # Convertir siempre desde la perspectiva del equipo consultado
    # tiros_eq = tiros propios; tiros_riv = tiros del oponente (para AST/AC)
    gf, gc, tiros_eq, tiros_riv, corners_eq, corners_riv = [], [], [], [], [], []
    for _, row in res.iterrows():
        if row['HomeTeam'] == equipo:
            gf.append(row['FTHG']); gc.append(row['FTAG'])
            tiros_eq.append(row['HST']); tiros_riv.append(row['AST'])
            corners_eq.append(row['HC']); corners_riv.append(row['AC'])
        else:
            gf.append(row['FTAG']); gc.append(row['FTHG'])
            tiros_eq.append(row['AST']); tiros_riv.append(row['HST'])
            corners_eq.append(row['AC']); corners_riv.append(row['HC'])
    w = pesos / pesos.sum()
    return pd.Series({
        'FTHG': np.average(gf,          weights=w),
        'FTAG': np.average(gc,          weights=w),
        'HST':  np.average(tiros_eq,    weights=w),
        'AST':  np.average(tiros_riv,   weights=w),   # tiros del oponente (perspectiva rival)
        'HC':   np.average(corners_eq,  weights=w),
        'AC':   np.average(corners_riv, weights=w),   # córners del oponente (perspectiva rival)
    })

def obtener_puntos_temporada(equipo, conn):
    # Calcula la jerarquía del equipo
    query = f"SELECT FTR, HomeTeam, AwayTeam FROM historial_multiliga_ml WHERE (HomeTeam='{equipo}' OR AwayTeam='{equipo}') AND Date >= date('now', '-9 months')"
    df = pd.read_sql(query, conn)
    pts = 0
    for _, row in df.iterrows():
        if row['HomeTeam'] == equipo and row['FTR'] == 'H': pts += 3
        elif row['AwayTeam'] == equipo and row['FTR'] == 'A': pts += 3
        elif row['FTR'] == 'D': pts += 1
    return pts

def obtener_dias_descanso(equipo, conn):
    # Calcula cuántos días pasaron desde su último partido
    q = f"SELECT Date FROM historial_multiliga_ml WHERE HomeTeam='{equipo}' OR AwayTeam='{equipo}' ORDER BY Date DESC LIMIT 1"
    res = pd.read_sql(q, conn)
    if not res.empty:
        ultima_fecha = pd.to_datetime(res.iloc[0]['Date'][:10])
        hoy = pd.Timestamp.now().normalize()
        dias = (hoy - ultima_fecha).days
        return min(max(dias, 3), 14) # Mantenemos los límites lógicos
    return 7

conn = sqlite3.connect(DB_NAME)


def predecir_analisis_dia(home_team, away_team, modelo_intl, encoder_intl, conn):
    """Pipeline unificado para partidos de selecciones en Análisis del Día.
    Usa el mismo blend aditivo FIFA 80/20 que el bloque es_mundial."""
    import math as _math

    hist_table = "historial_selecciones_ml"

    df_sh = pd.read_sql(
        f'SELECT HomeTeam,AwayTeam,FTHG,FTAG,HST,AST,HC,AC '
        f'FROM {hist_table} '
        f'WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" '
        f'ORDER BY Date DESC LIMIT 6', conn)
    df_sa = pd.read_sql(
        f'SELECT HomeTeam,AwayTeam,FTHG,FTAG,HST,AST,HC,AC '
        f'FROM {hist_table} '
        f'WHERE HomeTeam="{away_team}" OR AwayTeam="{away_team}" '
        f'ORDER BY Date DESC LIMIT 6', conn)

    def concat_mean_fn(s1, s2, default):
        combined = pd.concat([s1, s2]).reset_index(drop=True)
        if combined.empty or combined.isna().all():
            return default
        n = len(combined)
        base_w = np.array([5, 4, 3, 2, 1], dtype=float)
        w = base_w[:n] if n <= len(base_w) else np.concatenate([base_w, np.ones(n - len(base_w))])
        mask = combined.notna()
        if mask.sum() == 0:
            return default
        v = np.average(combined[mask], weights=w[mask.values])
        return float(v) if pd.notna(v) else default

    df_sh_home = df_sh[df_sh['HomeTeam'] == home_team]
    df_sh_away = df_sh[df_sh['AwayTeam'] == home_team]
    df_sa_home = df_sa[df_sa['HomeTeam'] == away_team]
    df_sa_away = df_sa[df_sa['AwayTeam'] == away_team]

    hst = concat_mean_fn(df_sh_home['HST'], df_sh_away['AST'], 4.0)
    hc  = concat_mean_fn(df_sh_home['HC'],  df_sh_away['AC'],  4.5)
    ast = concat_mean_fn(df_sa_home['AST'], df_sa_away['HST'], 3.5)
    ac  = concat_mean_fn(df_sa_home['AC'],  df_sa_away['HC'],  4.0)

    gf_h = concat_mean_fn(df_sh_home['FTHG'], df_sh_away['FTAG'], 1.3)
    gc_h = concat_mean_fn(df_sh_home['FTAG'], df_sh_away['FTHG'], 1.3)
    gf_a = concat_mean_fn(df_sa_home['FTHG'], df_sa_away['FTAG'], 1.3)
    gc_a = concat_mean_fn(df_sa_home['FTAG'], df_sa_away['FTHG'], 1.3)

    xg_h = (gf_h + gc_a) / 2
    xg_a = (gf_a + gc_h) / 2

    if home_team in encoder_intl.classes_ and away_team in encoder_intl.classes_:
        h_c = encoder_intl.transform([home_team])[0]
        a_c = encoder_intl.transform([away_team])[0]
        X_normal    = pd.DataFrame([[h_c, a_c, hst, ast, hc, ac]],
                                   columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC'])
        X_invertido = pd.DataFrame([[a_c, h_c, ast, hst, ac, hc]],
                                   columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC'])
        probs_n = modelo_intl.predict_proba(X_normal)[0]
        probs_i = modelo_intl.predict_proba(X_invertido)[0]
        p_h_raw = (float(probs_n[2]) + float(probs_i[0])) / 2
        p_d_raw = (float(probs_n[1]) + float(probs_i[1])) / 2
        p_a_raw = (float(probs_n[0]) + float(probs_i[2])) / 2
    else:
        p_h_raw, p_d_raw, p_a_raw = 0.33, 0.34, 0.33

    _UEFA = {
        "France","Spain","England","Germany","Portugal","Netherlands",
        "Italy","Belgium","Croatia","Switzerland","Denmark","Austria",
        "Poland","Serbia","Ukraine","Czech Republic","Hungary","Slovakia",
        "Romania","Turkey","Scotland","Wales","Greece","Slovenia",
        "Albania","Georgia","Norway","Sweden","Finland",
        "Bosnia and Herzegovina","North Macedonia","Kosovo","Montenegro",
        "Bulgaria","Luxembourg","Belarus","Ireland","Northern Ireland",
        "Iceland","Israel"
    }
    _CONMEBOL = {
        "Argentina","Brazil","Uruguay","Colombia","Chile",
        "Ecuador","Peru","Paraguay","Venezuela","Bolivia"
    }
    try:
        _df_fifa = pd.read_csv('fifa_ranking_2026.csv')
        _df_fifa.columns = [c.strip() for c in _df_fifa.columns]
        _TRAD_CSV = {
            "Francia":"France","España":"Spain","Argentina":"Argentina",
            "Inglaterra":"England","Portugal":"Portugal","Brasil":"Brazil",
            "Países Bajos":"Netherlands","Marruecos":"Morocco","Bélgica":"Belgium",
            "Alemania":"Germany","Croacia":"Croatia","Italia":"Italy",
            "Colombia":"Colombia","Senegal":"Senegal","México":"Mexico",
            "Estados Unidos":"United States","Uruguay":"Uruguay","Japón":"Japan",
            "Suiza":"Switzerland","Dinamarca":"Denmark","Irán":"Iran",
            "Turquía":"Turkey","Ecuador":"Ecuador","Austria":"Austria",
            "Corea del Sur":"Korea Republic","Nigeria":"Nigeria","Australia":"Australia",
            "Argelia":"Algeria","Egipto":"Egypt","Canadá":"Canada",
            "Noruega":"Norway","Ucrania":"Ukraine","Panamá":"Panama",
            "Costa de Marfil":"Ivory Coast","Polonia":"Poland","Rusia":"Russia",
            "Gales":"Wales","Suecia":"Sweden","Serbia":"Serbia",
            "Paraguay":"Paraguay","Chequia":"Czech Republic","Hungría":"Hungary",
            "Escocia":"Scotland","Túnez":"Tunisia","Camerún":"Cameroon",
            "RD Congo":"DR Congo","Grecia":"Greece","Eslovaquia":"Slovakia",
            "Venezuela":"Venezuela","Uzbekistán":"Uzbekistan","Costa Rica":"Costa Rica",
            "Malí":"Mali","Perú":"Peru","Chile":"Chile","Catar":"Qatar",
            "Rumanía":"Romania","Irak":"Iraq","Eslovenia":"Slovenia",
            "Irlanda":"Ireland","Sudáfrica":"South Africa","Arabia Saudita":"Saudi Arabia",
            "Burkina Faso":"Burkina Faso","Jordania":"Jordan","Albania":"Albania",
            "Bosnia":"Bosnia and Herzegovina","Honduras":"Honduras",
            "Macedonia Norte":"North Macedonia","EAU":"United Arab Emirates",
            "Cabo Verde":"Cape Verde","Irlanda Norte":"Northern Ireland",
            "Jamaica":"Jamaica","Georgia":"Georgia","Finlandia":"Finland",
            "Ghana":"Ghana","Islandia":"Iceland","Bolivia":"Bolivia",
            "Israel":"Israel","Kosovo":"Kosovo","Omán":"Oman",
            "Guinea":"Guinea","Montenegro":"Montenegro","Curazao":"Curaçao",
            "Haití":"Haiti","Siria":"Syria","Nueva Zelanda":"New Zealand",
            "Bulgaria":"Bulgaria","Gabón":"Gabon","Uganda":"Uganda",
            "Angola":"Angola","Benín":"Benin","Baréin":"Bahrain",
            "Zambia":"Zambia","Tailandia":"Thailand","China":"China",
            "Palestina":"Palestine","Guatemala":"Guatemala",
            "Bielorrusia":"Belarus","Luxemburgo":"Luxembourg",
            "Vietnam":"Vietnam","El Salvador":"El Salvador",
        }
        _df_fifa['country_en'] = _df_fifa['country'].str.strip().map(_TRAD_CSV)
        _pts_min, _pts_max = _df_fifa['points'].min(), _df_fifa['points'].max()
        _val_min, _val_max = _df_fifa['valor_total_mill_eur'].min(), _df_fifa['valor_total_mill_eur'].max()
        _df_fifa['pts_norm'] = (_df_fifa['points'] - _pts_min) / (_pts_max - _pts_min)
        _df_fifa['val_norm'] = (_df_fifa['valor_total_mill_eur'] - _val_min) / (_val_max - _val_min)
        _FIFA_SCORES = dict(zip(_df_fifa['country_en'], zip(_df_fifa['pts_norm'], _df_fifa['val_norm'])))
    except Exception:
        _FIFA_SCORES = {}

    def _score_sel(eq):
        if eq in _FIFA_SCORES:
            pts_n, val_n = _FIFA_SCORES[eq]
        else:
            pts_n, val_n = 0.3, 0.05
        conf = 1.0 if eq in _UEFA or eq in _CONMEBOL else 0.0
        return 0.40 * pts_n + 0.40 * val_n + 0.20 * conf

    _score_h = _score_sel(home_team)
    _score_a = _score_sel(away_team)
    _diff    = _score_h - _score_a

    # Blend aditivo: modelo RF 80% + FIFA ranking 20%
    _fifa_ph = 1 / (1 + _math.exp(-1.5 * _diff))
    _fifa_pa = 1 - _fifa_ph
    _fifa_pd = max(0.05, 0.5 - abs(_diff) * 0.3)
    _fifa_sum = _fifa_ph + _fifa_pa + _fifa_pd
    _fifa_ph /= _fifa_sum; _fifa_pa /= _fifa_sum; _fifa_pd /= _fifa_sum

    _W_MODEL = 0.80; _W_FIFA = 0.20
    p_h_aj = _W_MODEL * p_h_raw + _W_FIFA * _fifa_ph
    p_a_aj = _W_MODEL * p_a_raw + _W_FIFA * _fifa_pa
    p_d_aj = _W_MODEL * p_d_raw + _W_FIFA * _fifa_pd
    _suma  = p_h_aj + p_a_aj + p_d_aj
    prob_local  = p_h_aj / _suma
    prob_visita = p_a_aj / _suma
    prob_empate = p_d_aj / _suma

    # ─────────────────────────────────────────────────────────────────
    # PROYECCIÓN DE GOLES:
    # xg_h y xg_a ya fueron calculados arriba como promedios históricos
    # (ataque propio vs defensa rival). Los usamos directamente como
    # estimadores de goles esperados — sin redistribuir con prop_h/prop_a,
    # porque las probabilidades de resultado (H/D/A) no determinan cuántos
    # goles marca cada equipo; son variables relacionadas pero distintas.
    # ─────────────────────────────────────────────────────────────────
    pred_home = xg_h          # ya calculado: (gf_h + gc_a) / 2
    pred_away = xg_a          # ya calculado: (gf_a + gc_h) / 2
    promedio_goles = pred_home + pred_away

    # prob_over usa Poisson real (igual que el pipeline de clubes)
    # en lugar de la sigmoide aproximada que subestimaba el Over 2.5.
    prob_over_val = poisson_prob_over(promedio_goles, 2.5)

    return {
        'prob_local': prob_local,
        'prob_empate': prob_empate,
        'prob_visita': prob_visita,
        'pred_home': pred_home,
        'pred_away': pred_away,
        'promedio_goles': promedio_goles,
        'prob_over': prob_over_val,
        'hst': hst, 'ast': ast, 'hc': hc, 'ac': ac,
        'xg_h': xg_h, 'xg_a': xg_a,
    }

st.sidebar.title("Menú Principal")
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "Portafolio de Picks"])
st.sidebar.markdown("---")

if menu == "Análisis del Día":
    try:
        # 1. Cargar partidos de clubes
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        # ── Partidos visibles todo el día hasta las 23:59 hora Chile ──
        # Usamos la zona horaria de Santiago para que los partidos NO desaparezcan
        # cuando el servidor (UTC) cruza medianoche antes que el reloj local.
        # Ejemplo: a las 21:00 Santiago (00:00 UTC) un servidor en UTC devuelve
        # mañana con .now().normalize(), filtrando los partidos de hoy erróneamente.
        hoy = pd.Timestamp(pd.Timestamp.now(tz='America/Santiago').date())

        if not df_jornada.empty:
            # format='mixed' porque la columna Date mezcla filas con hora ("...00:00:00")
            # y filas solo con fecha ("YYYY-MM-DD") — mismo fix aplicado en ml_model_nuevo.py
            df_jornada['Date'] = pd.to_datetime(df_jornada['Date'], format='mixed').dt.tz_localize(None).dt.normalize()
            df_jornada = df_jornada[df_jornada['Date'] >= hoy]

        if not df_jornada.empty:
            df_jornada = df_jornada.sort_values(by='Date', ascending=True)
            df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%a %d/%m')

            # ── CSS compartido para las cards de partido ──
            st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] .match-card-wrap { width:100%; }
            </style>
            """, unsafe_allow_html=True)

            # ─────────────────────────────────────────────
            # SECCIÓN SELECTOR EN PANTALLA PRINCIPAL
            # Un único key global evita que el for sobreescriba
            # partido_texto con el valor de la última tab iterada.
            # ─────────────────────────────────────────────
            opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))

            # Inicializar selección global la primera vez
            if 'analisis_partido_sel' not in st.session_state:
                _primera_lista = (
                    df_jornada[df_jornada['Fecha_Display'] == opciones_fecha[0]]['Local'] + " vs " +
                    df_jornada[df_jornada['Fecha_Display'] == opciones_fecha[0]]['Visita']
                ).tolist()
                st.session_state['analisis_partido_sel'] = (
                    opciones_fecha[0],
                    _primera_lista[0] if _primera_lista else None
                )

            # Leer la selección actual ANTES de renderizar los tabs
            _sel_fecha, _sel_partido = st.session_state['analisis_partido_sel']

            # ── Tabs de fecha (máx 7 para no colapsar en móvil) ──
            tabs_fechas = st.tabs(opciones_fecha[:7])

            for i, tab in enumerate(tabs_fechas):
                fecha_label  = opciones_fecha[i]
                partidos_dia = df_jornada[df_jornada['Fecha_Display'] == fecha_label]

                lista_partidos = (partidos_dia['Local'] + " vs " + partidos_dia['Visita']).tolist()

                with tab:
                    for partido in lista_partidos:
                        _h, _a  = partido.split(" vs ")
                        # Un partido está activo solo si coincide fecha Y nombre
                        _activo = (_sel_fecha == fecha_label and _sel_partido == partido)
                        _border = "#5dade2" if _activo else "#2c3050"
                        _bg     = "#1a2540" if _activo else "#1e2129"
                        _sombra = "0 0 0 2px #5dade255" if _activo else "none"
                        _check  = "✦" if _activo else ""

                        st.markdown(f"""
                        <div style="
                            background:{_bg};border:1px solid {_border};border-radius:10px;
                            padding:10px 14px;margin-bottom:8px;
                            box-shadow:{_sombra};transition:all .2s;">
                          <div style="display:flex;align-items:center;justify-content:space-between;">
                            <span style="font-size:0.85rem;font-weight:700;color:#e8ecf5;flex:1;text-align:right;">
                              {_h}
                            </span>
                            <span style="color:#6c7a9c;font-size:0.72rem;font-weight:600;
                              margin:0 10px;white-space:nowrap;">VS</span>
                            <span style="font-size:0.85rem;font-weight:700;color:#e8ecf5;flex:1;text-align:left;">
                              {_a}
                            </span>
                            <span style="color:#5dade2;font-size:0.8rem;margin-left:8px;
                              min-width:12px;">{_check}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(
                            "Analizar →" if not _activo else "✔ Seleccionado",
                            key=f"btn_{i}_{partido}",
                            use_container_width=True,
                            type="primary" if _activo else "secondary",
                        ):
                            # Guardar (fecha, partido) en el key global y releer
                            st.session_state['analisis_partido_sel'] = (fecha_label, partido)
                            st.rerun()

            # La fuente de verdad es el key global, no el loop
            dia_sel_str   = st.session_state['analisis_partido_sel'][0]
            partido_texto = st.session_state['analisis_partido_sel'][1]

            if partido_texto is None:
                st.info("No hay partidos programados en la base de datos.")
                st.stop()

            st.divider()

            # ─────────────────────────────────────────────
            # CORRECCIÓN DE NOMBRES Y CARGA DE HISTÓRICO
            # ─────────────────────────────────────────────
            home_raw, away_raw = partido_texto.split(" vs ")

            hist_table = "historial_multiliga_ml"
            # Unimos HomeTeam y AwayTeam: un equipo que sólo aparece como
            # visitante en el historial (ej. recién ascendido, o con pocos
            # partidos jugados de local) quedaba fuera de la lista y forzaba
            # un fuzzy match incorrecto contra otro equipo.
            equipos_db = pd.read_sql(
                f"SELECT DISTINCT HomeTeam AS Team FROM {hist_table} "
                f"UNION SELECT DISTINCT AwayTeam AS Team FROM {hist_table}",
                conn
            )['Team'].tolist()

            home_team = corregir_nombre_equipo(home_raw, equipos_db)
            away_team = corregir_nombre_equipo(away_raw, equipos_db)

            # --- SECCIÓN 1: ENCABEZADO CARD HTML ---
            _modo_badge = (
                '<span style="background:#1a3a1a;color:#27ae60;font-size:0.65rem;'
                'padding:2px 8px;border-radius:10px;letter-spacing:1px;font-weight:700;">⚽ CLUBES</span>'
            )
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,#1a1d27 0%,#1e2236 100%);
                border:1px solid #2c3050;border-radius:14px;
                padding:18px 16px 14px;margin-bottom:18px;
                box-shadow:0 4px 20px rgba(0,0,0,0.4);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    {_modo_badge}
                    <span style="color:#6c7a9c;font-size:0.72rem;">📅 {dia_sel_str}</span>
                </div>
                <div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:1.25rem;font-weight:800;color:#e8ecf5;text-align:right;flex:1;min-width:80px;">
                        {home_team}
                    </span>
                    <span style="background:#2c3050;color:#5dade2;font-size:0.9rem;font-weight:700;
                        padding:5px 12px;border-radius:20px;white-space:nowrap;">VS</span>
                    <span style="font-size:1.25rem;font-weight:800;color:#e8ecf5;text-align:left;flex:1;min-width:80px;">
                        {away_team}
                    </span>
                </div>
                <div style="margin-top:12px;height:2px;
                    background:linear-gradient(90deg,transparent,#3d4f8a,transparent);"></div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.subheader("Historial H2H")
                q_h2h = (
                    f'SELECT Date, HomeTeam as L, AwayTeam as V, FTHG as [GL], FTAG as [GV], FTR as R '
                    f'FROM {hist_table} '
                    f'WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") '
                    f'OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") '
                    f'ORDER BY Date DESC LIMIT 5'
                )
                df_h2h = pd.read_sql(q_h2h, conn)
                if not df_h2h.empty:
                    df_h2h['Date'] = pd.to_datetime(df_h2h['Date'], format='mixed').dt.strftime('%d/%m/%y')
                    st.dataframe(df_h2h, use_container_width=True, hide_index=True)
                else:
                    st.info("No existen enfrentamientos directos recientes en la base de datos.")

                st.subheader("Tendencia de Goles")
                q_trend = (
                    f'SELECT FTHG as [Local], FTAG as [Visita] FROM {hist_table} '
                    f'WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" '
                    f'ORDER BY Date DESC LIMIT 10'
                )
                df_trend = pd.read_sql(q_trend, conn)
                if not df_trend.empty:
                    st.line_chart(df_trend.iloc[::-1])

            with col2:
                st.subheader("IA Predictiva")

                # ── Variables de scope para uso posterior ──
                prob_local = prob_empate = prob_visita = 0.33
                prob_over  = 0.5
                xg_h = xg_a = 1.2
                pred_home = pred_away = 1.2
                stats_h_dict = stats_a_dict = {}
                prom_h = prom_a = {}
                tend_h = tend_a = []
                _tiros_h = _tiros_a = _corners_h = _corners_a = 0.0

                bundle = cargar_modelo()
                if bundle:
                    model_obj    = bundle['model']
                    feature_cols = bundle['feature_cols']

                    stats_h, stats_a   = get_recent_stats(home_team, conn), get_recent_stats(away_team, conn)
                    stats_h_dict, stats_a_dict = stats_h, stats_a

                    # Forma reciente con perspectiva corregida — el vector que de verdad
                    # espera el modelo (contrato definido en ia_features.py)
                    forma_h = construir_forma_reciente(home_team, conn)
                    forma_a = construir_forma_reciente(away_team, conn)

                    xg_h = forma_h['xg_favor']
                    xg_a = forma_a['xg_favor']
                    pts_h         = obtener_puntos_temporada(home_team, conn)
                    pts_a         = obtener_puntos_temporada(away_team, conn)
                    dif_tabla     = pts_h - pts_a
                    descanso_h    = obtener_dias_descanso(home_team, conn)
                    descanso_a    = obtener_dias_descanso(away_team, conn)
                    ventaja_fisica = descanso_h - descanso_a

                    fila_features = construir_fila_features(forma_h, forma_a, dif_tabla, ventaja_fisica)
                    X_pred = pd.DataFrame([fila_features])[feature_cols]

                    prob_ia     = model_obj.predict_proba(X_pred)[0]
                    prob_local  = float(prob_ia[2])
                    prob_empate = float(prob_ia[1])
                    prob_visita = float(prob_ia[0])

                    # pred_home/pred_away usan xg_h/xg_a (perspectiva corregida) para ser
                    # coherentes con los inputs del modelo que genera la torta.
                    pred_home      = (xg_h + forma_a['xg_contra']) / 2
                    pred_away      = (xg_a + forma_h['xg_contra']) / 2
                    promedio_goles = pred_home + pred_away
                    prob_over      = 1 / (1 + np.exp(-(promedio_goles - 2.5)))
                    _tiros_h       = stats_h['HST']
                    _tiros_a       = stats_a['AST']
                    _corners_h     = stats_h['HC']
                    _corners_a     = stats_a['AC']

                    # --- SECCIÓN 2: TORTA CON ANOTACIÓN CENTRAL ---
                    _outcomes  = {'LOCAL': prob_local, 'EMPATE': prob_empate, 'VISITA': prob_visita}
                    _dom_label = max(_outcomes, key=_outcomes.get)
                    _dom_pct   = f"{_outcomes[_dom_label]:.0%}"

                    fig_pie = px.pie(
                        values=[prob_local, prob_empate, prob_visita],
                        names=['Local', 'Empate', 'Visita'],
                        color=['Local', 'Empate', 'Visita'],
                        color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'},
                        hole=0.45
                    )
                    fig_pie.update_layout(
                        dragmode=False, margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[dict(
                            text=f"<b>{_dom_label}</b><br>{_dom_pct}",
                            x=0.5, y=0.5,
                            font=dict(size=13, color='#e8ecf5'),
                            showarrow=False, xanchor='center', yanchor='middle'
                        )]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)

                    # --- SECCIÓN 3: PANEL COMPACTO DE MÉTRICAS (4 columnas) ---
                    _BASE   = 1.2
                    _xg_total = pred_home + pred_away   # coherente con prob_over y la torta
                    _m1, _m2, _m3, _m4 = st.columns(4)
                    _m1.metric("xG Total",            f"{_xg_total:.2f}", f"{_xg_total - _BASE*2:+.2f}")
                    _m2.metric("Over 2.5",             f"{prob_over:.0%}", delta=None)
                    _m3.metric(f"{home_team[:8]}",  f"{pred_home:.2f}", f"{pred_home - _BASE:+.2f}")
                    _m4.metric(f"{away_team[:8]}",  f"{pred_away:.2f}", f"{pred_away - _BASE:+.2f}")

                    # --- SECCIÓN 4: MINI TABLA TIROS Y CÓRNERS ---
                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;margin:10px 0 6px;font-size:0.82rem;">
                      <thead>
                        <tr style="background:#1a1d27;">
                          <th style="padding:6px 8px;color:#6c7a9c;text-align:left;border-bottom:1px solid #2c3050;">Stat</th>
                          <th style="padding:6px 4px;color:#27ae60;text-align:center;border-bottom:1px solid #2c3050;">{home_team[:12]}</th>
                          <th style="padding:6px 4px;color:#c0392b;text-align:center;border-bottom:1px solid #2c3050;">{away_team[:12]}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style="background:#1e2129;">
                          <td style="padding:6px 8px;color:#aab0c0;">🎯 Tiros</td>
                          <td style="padding:6px 4px;text-align:center;font-weight:700;color:#e8ecf5;">{_tiros_h:.1f}</td>
                          <td style="padding:6px 4px;text-align:center;font-weight:700;color:#e8ecf5;">{_tiros_a:.1f}</td>
                        </tr>
                        <tr style="background:#22263a;">
                          <td style="padding:6px 8px;color:#aab0c0;">🚩 Córners</td>
                          <td style="padding:6px 4px;text-align:center;font-weight:700;color:#e8ecf5;">{_corners_h:.1f}</td>
                          <td style="padding:6px 4px;text-align:center;font-weight:700;color:#e8ecf5;">{_corners_a:.1f}</td>
                        </tr>
                      </tbody>
                    </table>
                    """, unsafe_allow_html=True)

            # --- 🎯 FIX: CONDICIONAL PARA TARJETAS ---
            st.divider()
            st.subheader("Disciplina y Tarjetas")

            cd1, cd2 = st.columns(2)
            with cd1:
                st.markdown("#### **Media Amarillas**")
                m1, m2 = st.columns(2)
                m1.metric(f"{home_team[:12]}", f"{stats_h_dict.get('HY', 0):.1f}")
                m2.metric(f"{away_team[:12]}", f"{stats_a_dict.get('AY', 0):.1f}")

            with cd2:
                q_cards = (
                    f'SELECT Date, (HY + AY) as Total FROM {hist_table} '
                    f'WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") '
                    f'OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") '
                    f'ORDER BY Date DESC LIMIT 5'
                )
                try:
                    df_cards = pd.read_sql(q_cards, conn)
                    if not df_cards.empty:
                        fig_cards = px.bar(df_cards, x='Date', y='Total', color_discrete_sequence=['#f1c40f'])
                        fig_cards.update_layout(dragmode=False, xaxis={'fixedrange': True}, yaxis={'fixedrange': True})
                        st.plotly_chart(fig_cards, use_container_width=True, config=CONFIG_FIJA)
                    else:
                        st.info("Sin datos suficientes de tarjetas para graficar H2H.")
                except Exception:
                    st.info("No se pudieron graficar las tarjetas.")

            # --- 🎯 NUEVA SECCIÓN: PROMEDIOS Y TENDENCIAS ---
            st.divider()
            st.subheader("Promedios y Tendencias (Últimos 10 Partidos)")

            def calcular_promedios_tendencias(equipo, df_hist):
                n = len(df_hist)
                if n == 0: return {}, []

                stats = {'gf':[], 'gc':[], 'gt':[], 'ce':[], 'ct':[], 'se':[], 'amarillas':[]}
                wins, no_loss = 0, 0

                for _, row in df_hist.iterrows():
                    is_home = (row['HomeTeam'] == equipo)
                    gf = row['FTHG'] if is_home else row['FTAG']
                    gc = row['FTAG'] if is_home else row['FTHG']
                    ce = row['HC'] if 'HC' in row and pd.notna(row['HC']) else 0
                    if not is_home: ce = row['AC'] if 'AC' in row and pd.notna(row['AC']) else 0
                    ct_h = row['HC'] if 'HC' in row and pd.notna(row['HC']) else 0
                    ct_a = row['AC'] if 'AC' in row and pd.notna(row['AC']) else 0
                    se = row['HST'] if 'HST' in row and pd.notna(row['HST']) else 0
                    if not is_home: se = row['AST'] if 'AST' in row and pd.notna(row['AST']) else 0
                    am_h = row['HY'] if 'HY' in row and pd.notna(row['HY']) else None
                    am_a = row['AY'] if 'AY' in row and pd.notna(row['AY']) else None
                    am = am_h if is_home else am_a
                    stats['gf'].append(gf); stats['gc'].append(gc); stats['gt'].append(gf + gc)
                    stats['ce'].append(ce); stats['ct'].append(ct_h + ct_a); stats['se'].append(se)
                    if am is not None: stats['amarillas'].append(am)
                    if gf > gc: wins += 1; no_loss += 1
                    elif gf == gc: no_loss += 1

                promedios = {
                    "Goles Anotados":    sum(stats['gf'])/n if n>0 else 0,
                    "Goles Recibidos":   sum(stats['gc'])/n if n>0 else 0,
                    "Córners a Favor":   sum(stats['ce'])/n if n>0 else 0,
                    "Córners en Partido":sum(stats['ct'])/n if n>0 else 0,
                    "Tiros al Arco":     sum(stats['se'])/n if n>0 else 0,
                }
                if stats['amarillas']:
                    promedios["Tarjetas Amarillas"] = sum(stats['amarillas'])/len(stats['amarillas'])

                tendencias = []
                def check_highest_over(lista, umbrales, texto):
                    if not lista: return
                    for umbral in sorted(umbrales, reverse=True):
                        count = sum(1 for x in lista if x > umbral)
                        if count / n >= 0.75:
                            if umbral == 0.5 and texto == "⚽ Goles del Equipo":
                                tendencias.append(f"⚽ Anota al menos 1 gol en {count}/{n} partidos")
                            else:
                                tendencias.append(f"{texto} (+{umbral}) en {count}/{n} partidos")
                            break

                check_highest_over(stats['gf'], [0.5, 1.5, 2.5], "⚽ Goles del Equipo")
                check_highest_over(stats['gt'], [1.5, 2.5, 3.5], "🥅 Goles en el Partido")
                check_highest_over(stats['ce'], [3.5, 4.5, 5.5], "🚩 Córners del Equipo")
                check_highest_over(stats['ct'], [7.5, 8.5, 9.5], "⛳ Córners en el Partido")
                check_highest_over(stats['se'], [2.5, 3.5, 4.5], "🎯 Tiros al Arco")
                if wins / n >= 0.75:   tendencias.append(f"🏆 Victorias en {wins}/{n} partidos")
                if no_loss / n >= 0.75: tendencias.append(f"🛡️ Invicto en {no_loss}/{n} partidos")
                return promedios, tendencias

            df_h10 = pd.read_sql(f'SELECT * FROM {hist_table} WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" ORDER BY Date DESC LIMIT 10', conn)
            df_a10 = pd.read_sql(f'SELECT * FROM {hist_table} WHERE HomeTeam="{away_team}" OR AwayTeam="{away_team}" ORDER BY Date DESC LIMIT 10', conn)

            prom_h, tend_h = calcular_promedios_tendencias(home_team, df_h10)
            prom_a, tend_a = calcular_promedios_tendencias(away_team, df_a10)

            ct1, ct2 = st.columns(2)

            # --- SECCIÓN 5: BADGES PILL POR CATEGORÍA ---
            def _badge_color(texto):
                t = texto.lower()
                if "gol" in t:                              return "#0d3320", "#2ecc71"
                if "córner" in t or "corner" in t:          return "#3a2000", "#e67e22"
                if "victoria" in t or "invicto" in t:       return "#0d1f40", "#5dade2"
                if "tiro" in t:                             return "#1a0d40", "#9b59b6"
                if "tarjeta" in t or "amarilla" in t:       return "#3a3000", "#f1c40f"
                return "#1e1e1e", "#aab0c0"

            def renderizar_columna(equipo, prom, tend):
                st.markdown(f"#### {equipo}")
                st.markdown("**Promedios**")
                tabla_md = "| Estadística | Promedio |\n|---|---|\n"
                for k, v in prom.items():
                    tabla_md += f"| {k} | **{v:.1f}** |\n"
                st.markdown(tabla_md)

                st.markdown("**Tendencias Altas (>75%)**")
                if tend:
                    _badges_html = '<div style="display:flex;flex-direction:column;gap:6px;margin-top:4px;">'
                    for t in tend:
                        _bg, _fg = _badge_color(t)
                        _badges_html += (
                            f'<span style="background:{_bg};color:{_fg};'
                            f'border:1px solid {_fg}33;border-radius:20px;'
                            f'padding:5px 12px;font-size:0.78rem;font-weight:600;'
                            f'display:inline-block;">{t}</span>'
                        )
                    _badges_html += '</div>'
                    st.markdown(_badges_html, unsafe_allow_html=True)
                else:
                    st.info("Sin tendencias consistentes en los últimos 10 partidos.")

            with ct1: renderizar_columna(home_team, prom_h, tend_h)
            with ct2: renderizar_columna(away_team, prom_a, tend_a)

            # --- SECCIÓN 6: SCOUT REPORT FINAL ---
            st.divider()
            _outcome_map = {'LOCAL': home_team, 'EMPATE': 'Empate', 'VISITA': away_team}
            _probs_dict  = {'LOCAL': prob_local, 'EMPATE': prob_empate, 'VISITA': prob_visita}
            _dom_key     = max(_probs_dict, key=_probs_dict.get)
            _dom_team    = _outcome_map[_dom_key]
            _dom_prob    = _probs_dict[_dom_key]
            _xg_total_str = f"{xg_h + xg_a:.2f}"
            _over_str     = f"{prob_over:.0%}"
            _over_label   = "favorable" if prob_over >= 0.55 else ("ajustada" if prob_over >= 0.45 else "baja")

            _scout_lines = [
                f"El modelo favorece a <b>{_dom_team}</b> con una probabilidad del <b>{_dom_prob:.0%}</b>. "
                f"Los xG combinados ({_xg_total_str}) sitúan el partido con expectativa de goles "
                f"{'alta' if xg_h + xg_a >= 2.5 else 'moderada'}.",
                f"La probabilidad Over 2.5 es <b>{_over_str}</b> ({_over_label}). "
                f"Se proyectan <b>{pred_home:.1f}</b> goles para {home_team} "
                f"y <b>{pred_away:.1f}</b> para {away_team}."
            ]
            _all_tend    = tend_h + tend_a
            _corner_tend = [t for t in _all_tend if "córner" in t.lower() or "corner" in t.lower()]
            _goal_tend   = [t for t in _all_tend if "gol" in t.lower()]
            if _corner_tend or _goal_tend:
                _extras = (_goal_tend + _corner_tend)[:2]
                _scout_lines.append(f"Tendencias clave: {' · '.join(_extras)}.")

            _report_html = (
                '<div style="background:linear-gradient(135deg,#10131e 0%,#14182b 100%);'
                'border:1px solid #2c3050;border-left:4px solid #5dade2;'
                'border-radius:12px;padding:16px 18px;margin-top:4px;">'
                '<div style="color:#5dade2;font-size:0.7rem;font-weight:700;'
                'letter-spacing:2px;margin-bottom:12px;">SCOUT REPORT · IA</div>'
            )
            for _line in _scout_lines:
                _report_html += f'<p style="color:#c8d0e0;font-size:0.84rem;line-height:1.55;margin:0 0 8px;">{_line}</p>'
            _report_html += '</div>'
            st.markdown(_report_html, unsafe_allow_html=True)

        else:
            st.info("No hay partidos programados en la base de datos.")

    except Exception as e:
        st.error(f"Error al cargar dashboard: {e}")


elif menu == "Auditoría (Resultados)":
    st.title("Auditoría de Precisión")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        fecha_audit = st.date_input("Selecciona fecha para auditar:",
                                    datetime.now() - timedelta(days=1))

    fecha_str = fecha_audit.strftime('%Y-%m-%d')

    # 1. Partidos de clubes
    df_clubes = pd.read_sql(
        "SELECT * FROM historial_multiliga_ml WHERE Date LIKE ?",
        conn, params=(f"{fecha_str}%",)
    )
    df_clubes['_es_mundial'] = False

    # 2. Partidos del Mundial desde historial_selecciones_ml
    try:
        df_wc = pd.read_sql(
            "SELECT * FROM historial_selecciones_ml WHERE Date LIKE ?",
            conn, params=(f"{fecha_str}%",)
        )
        df_wc['_es_mundial'] = True
        # historial_selecciones_ml puede no tener columnas de amarillas
        for _col in ['HY', 'AY']:
            if _col not in df_wc.columns:
                df_wc[_col] = 0
    except Exception:
        df_wc = pd.DataFrame()

    df_reales = (
        pd.concat([df_clubes, df_wc], ignore_index=True)
        if not df_wc.empty else df_clubes.copy()
    )

    if df_reales.empty:
        st.warning(f"No hay resultados en la base de datos para el {fecha_audit.strftime('%d/%m/%Y')}.")
    else:
        st.subheader(f"Jornada: {fecha_audit.strftime('%d/%m/%Y')}")

        _n_clubes = len(df_clubes)
        _n_wc     = len(df_wc) if not df_wc.empty else 0
        if _n_clubes > 0 and _n_wc > 0:
            st.caption(
                f"🏟️ {_n_clubes} partido{'s' if _n_clubes != 1 else ''} de clubes  •  "
                f"🌍 {_n_wc} partido{'s' if _n_wc != 1 else ''} del Mundial"
            )

        # --- FUNCIÓN DE TOLERANCIA INTELIGENTE ---
        def evaluar_precision(real, proyectado, margen):
            """Devuelve el ícono y color basado en el acierto o la cercanía"""
            if real >= proyectado:
                return "✅", "#27ae60"  # Verde (Cumplido)
            elif (proyectado - real) <= margen:
                return "⚠️", "#f39c12"  # Amarillo (Dentro del margen de error)
            else:
                return "❌", "#c0392b"  # Rojo (Fallado)

        total_predicciones = 0
        cumplidas          = 0
        casi_cumplidas     = 0
        resultados_procesados = []

        with st.spinner('Calculando precisión contra proyecciones IA...'):
            for _, r in df_reales.iterrows():
                es_wc = bool(r.get('_es_mundial', False))

                # Usar stats de la tabla correcta según el tipo de partido
                if es_wc:
                    sh = get_recent_stats_wc(r['HomeTeam'], conn)
                    sa = get_recent_stats_wc(r['AwayTeam'], conn)
                else:
                    sh = get_recent_stats(r['HomeTeam'], conn)
                    sa = get_recent_stats(r['AwayTeam'], conn)

                if sh is not None and sa is not None and len(sh) > 0 and len(sa) > 0:
                    # 1. Proyecciones
                    proj_goles_total = (sh['FTHG'] + sh['FTAG'] + sa['FTHG'] + sa['FTAG']) / 2
                    proj_goles_home  = (sh['FTHG'] + sa['FTAG']) / 2   # Ataque local vs defensa visita
                    proj_goles_away  = (sa['FTHG'] + sh['FTAG']) / 2   # Ataque visita vs defensa local
                    proj_corners     = sh['HC'] + sa['AC']
                    proj_tiros       = sh['HST'] + sa['AST']

                    # 2. Resultados Reales
                    real_goles_home  = r['FTHG']
                    real_goles_away  = r['FTAG']
                    real_goles_total = real_goles_home + real_goles_away
                    real_corners     = r['HC'] + r['AC']
                    real_tiros       = r['HST'] + r['AST']

                    # 3. Conteo de aciertos (goles totales + corners)
                    if real_goles_total >= proj_goles_total:
                        cumplidas += 1
                    elif (proj_goles_total - real_goles_total) <= 0.5:
                        casi_cumplidas += 1

                    if real_corners >= proj_corners:
                        cumplidas += 1
                    elif (proj_corners - real_corners) <= 1.5:
                        casi_cumplidas += 1

                    total_predicciones += 2

                    # 4. Construir lista de métricas
                    stats_partido = [
                        ("Goles Total",              proj_goles_total, real_goles_total, 0.5),
                        (f"Goles {r['HomeTeam']}",   proj_goles_home,  real_goles_home,  0.5),
                        (f"Goles {r['AwayTeam']}",   proj_goles_away,  real_goles_away,  0.5),
                        ("Córners Total",             proj_corners,     real_corners,     1.5),
                        ("Tiros al Arco",             proj_tiros,       real_tiros,       1.5),
                    ]

                    # Amarillas solo para clubes (historial_selecciones_ml puede no tenerlas)
                    if not es_wc:
                        proj_amarillas = sh['HY'] + sa['AY']
                        real_amarillas = r['HY'] + r['AY']
                        stats_partido.append(("Amarillas", proj_amarillas, real_amarillas, 1.0))

                    resultados_procesados.append({
                        'fila':       r,
                        'es_mundial': es_wc,
                        'stats':      stats_partido,
                    })

        # --- MÉTRICAS SUPERIORES ---
        if total_predicciones > 0:
            tasa_exacta   = cumplidas / total_predicciones
            tasa_flexible = (cumplidas + casi_cumplidas) / total_predicciones

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Tasa Verde (Exacta)",  f"{tasa_exacta:.1%}", f"{cumplidas} de {total_predicciones}")
            with col_m2:
                st.metric("Tasa Amarilla (Casi)", f"{(casi_cumplidas / total_predicciones):.1%}",
                          f"{casi_cumplidas} en el margen", delta_color="off")
            with col_m3:
                st.metric("Eficacia Flexible", f"{tasa_flexible:.1%}", "Verdes + Amarillos")

            st.divider()

            # --- ACORDEONES POR PARTIDO ---
            for res in resultados_procesados:
                r     = res['fila']
                icono = "🌍" if res['es_mundial'] else "🏟️"
                titulo = f"{icono} {r['HomeTeam']} {int(r['FTHG'])} - {int(r['FTAG'])} {r['AwayTeam']}"

                with st.expander(titulo):
                    cols = st.columns(2)
                    for i, (label, p, re, margen) in enumerate(res['stats']):
                        real_val = re if pd.notnull(re) else 0

                        check, color = evaluar_precision(real_val, p, margen)

                        with cols[i % 2]:
                            st.markdown(f"""
                            <div style="border-left: 5px solid {color}; padding: 8px; margin-bottom: 10px; background-color: #1e2129; border-radius: 5px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #888; font-size: 0.75rem; font-weight: bold;">{label.upper()}</span>
                                    <span style="font-size: 1.2rem;">{check}</span>
                                </div>
                                <div style="margin-top: 5px;">
                                    <span style="font-size: 0.9rem; color: #bbb;">IA:</span>
                                    <span style="font-size: 1rem; font-weight: bold; color: {color};">{p:.1f}</span>
                                    <span style="color: #555; margin: 0 5px;">|</span>
                                    <span style="font-size: 0.9rem; color: #bbb;">Real:</span>
                                    <span style="font-size: 1rem; font-weight: bold;">{int(real_val)}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No se pudieron calcular proyecciones (Faltan datos históricos de los equipos).")
elif menu == "Portafolio de Picks":
    st.title("Portafolio de Inversión (Flat Staking Híbrido)")
    
    API_KEY = "3ec28dbd498ab9985e9792b3f50a8902" 
    
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portafolio_historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            HomeTeam TEXT,
            AwayTeam TEXT,
            Mercado TEXT,
            Cuota REAL,
            Prob_IA REAL,
            Edge REAL,
            Stake REAL,
            Estado TEXT DEFAULT 'Pendiente',
            Beneficio_Neto REAL DEFAULT 0
        )
    """)
    conn.commit()

    tab1, tab2 = st.tabs(["Escáner en Vivo", "Rendimiento Histórico"])

    with tab1:
        st.markdown("### Escáner de Ineficiencias vs Pinnacle")
        st.caption("Cruzando modelo ML de Clubes y Poisson contra líneas de Pinnacle. (Edge 2% - 15%)")
        
        try:
            equipos_clubes = pd.read_sql(
                "SELECT DISTINCT HomeTeam AS Team FROM historial_multiliga_ml "
                "UNION SELECT DISTINCT AwayTeam AS Team FROM historial_multiliga_ml",
                conn
            )['Team'].tolist()
            
            directorio_odds = Path("odds_data")
            archivos_csv = list(directorio_odds.glob("*.csv")) if directorio_odds.exists() else []
            
            df_master_odds = pd.DataFrame()
            fechas_disponibles = []

            if archivos_csv:
                lista_dfs = []
                for f in archivos_csv:
                    try:
                        # Los archivos de odds del Mundial no se consideran en el portafolio
                        if 'worldcup' in f.name.lower():
                            continue

                        df_temp = pd.read_csv(f)
                        if df_temp.empty:
                            continue
                        
                        import re
                        match_dash = re.search(r'(\d{4})-(\d{2})-(\d{2})', f.name)
                        match_pure = re.search(r'(\d{4})(\d{2})(\d{2})', f.name)
                        
                        if match_dash:
                            fecha_fallback = match_dash.group(0)
                        elif match_pure:
                            fecha_fallback = f"{match_pure.group(1)}-{match_pure.group(2)}-{match_pure.group(3)}"
                        else:
                            import os
                            mtime = os.path.getmtime(f)
                            from datetime import datetime
                            fecha_fallback = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

                        df_temp.columns = [c.lower() for c in df_temp.columns]

                        if 'hometeam' in df_temp.columns and 'home' not in df_temp.columns: df_temp = df_temp.rename(columns={'hometeam': 'home'})
                        if 'awayteam' in df_temp.columns and 'away' not in df_temp.columns: df_temp = df_temp.rename(columns={'awayteam': 'away'})

                        if 'inicio_local' not in df_temp.columns or df_temp['inicio_local'].isna().all():
                            df_temp['inicio_local'] = fecha_fallback + " 12:00"
                        else:
                            df_temp['inicio_local'] = df_temp['inicio_local'].fillna(fecha_fallback + " 12:00")

                        lista_dfs.append(df_temp)
                    except Exception as e:
                        st.sidebar.error(f"Error en archivo {f.name}: {e}")
                
                if lista_dfs:
                    df_master_odds = pd.concat(lista_dfs, ignore_index=True)
                    df_master_odds = df_master_odds.drop_duplicates(subset=['home', 'away', 'inicio_local'])
                    df_master_odds['Fecha_Match'] = df_master_odds['inicio_local'].astype(str).str.strip().str.slice(0, 10)
                    df_master_odds = df_master_odds[df_master_odds['Fecha_Match'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]
                    fechas_disponibles = sorted(df_master_odds['Fecha_Match'].unique())

            c1, c2 = st.columns(2)
            with c1:
                inversion_total = st.number_input("Inversión TOTAL Portafolio ($)", min_value=1000, value=5000, step=500)
                modo_compuesto = st.toggle(
                    "Modo Compuesto (bankroll dinámico)",
                    key="modo_compuesto",
                    help="ON: el stake de cada día se calcula sobre tu bankroll actual (capital inicial + P&L acumulado). OFF: flat staking sobre el valor ingresado arriba."
                )
                if modo_compuesto:
                    _pnl_cerrado = pd.read_sql(
                        "SELECT COALESCE(SUM(Beneficio_Neto),0) as total FROM portafolio_historico WHERE Estado != 'Pendiente'",
                        conn
                    ).iloc[0]['total']
                    _bankroll_actual = inversion_total + _pnl_cerrado
                    _delta_str = f"{_pnl_cerrado:+,.0f}"
                    _color = "#2ecc71" if _pnl_cerrado >= 0 else "#e74c3c"
                    st.markdown(
                        f'<div style="background:#1a2a1a;border:1px solid {_color}33;border-radius:8px;'
                        f'padding:8px 12px;margin-top:4px;">'
                        f'<span style="color:#888;font-size:0.75rem;">Bankroll actual</span><br>'
                        f'<span style="font-size:1.2rem;font-weight:800;color:#e8ecf5;">'
                        f'${_bankroll_actual:,.0f}</span>'
                        f'<span style="color:{_color};font-size:0.8rem;margin-left:8px;">'
                        f'({_delta_str} P&L)</span></div>',
                        unsafe_allow_html=True
                    )
            with c2:
                if fechas_disponibles:
                    hoy_str = str(pd.Timestamp.now().date())
                    idx_hoy = fechas_disponibles.index(hoy_str) if hoy_str in fechas_disponibles else max(0, len(fechas_disponibles) - 1)
                    fecha_seleccionada = st.selectbox("Seleccionar Día del Portafolio:", fechas_disponibles, index=idx_hoy)
                else:
                    st.warning("No se encontraron partidos en la carpeta 'odds_data/'. Corre el scraper primero.")
                    fecha_seleccionada = None

            boton_disabled = fecha_seleccionada is None

            if st.button("Escanear Mercado", type="primary", disabled=boton_disabled):
                bundle_clubes = cargar_modelo()

                with st.spinner(f"Analizando los partidos del {fecha_seleccionada} con IA..."):
                    df_pinnacle = df_master_odds[df_master_odds['Fecha_Match'] == fecha_seleccionada]

                    # Nº de partidos distintos detectados ese día — usado más abajo
                    # para decidir si se arma portafolio o sólo se sugieren los picks.
                    n_partidos_dia = df_pinnacle[['home', 'away']].drop_duplicates().shape[0]

                    oportunidades = []
                    mercados_evaluados_completos = []
                    log_debug = []
                    
                    def buscar_cuota_segura(row, posibles_columnas):
                        for col in posibles_columnas:
                            if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != '':
                                return row[col]
                        return None

                    for index, row in df_pinnacle.iterrows():
                        h_csv = str(row['home'])
                        a_csv = str(row['away'])
                        fecha_partido = str(row['inicio_local']).split()[0] if pd.notna(row['inicio_local']) else str(pd.Timestamp.now().date())
                        
                        # El diccionario de alias manda: si hay match exacto por alias,
                        # se usa directamente y NO pasa por fuzzy matching (evita que
                        # nombres cortos/variantes conocidas dependan del score fuzzy).
                        h_alias = normalizar_nombre(h_csv)
                        a_alias = normalizar_nombre(a_csv)
                        h_db_match = (h_alias, 100) if h_alias in equipos_clubes else process.extractOne(h_csv, equipos_clubes, scorer=fuzz.token_sort_ratio)
                        a_db_match = (a_alias, 100) if a_alias in equipos_clubes else process.extractOne(a_csv, equipos_clubes, scorer=fuzz.token_sort_ratio)
                        
                        if not h_db_match or not a_db_match or h_db_match[1] < 80 or a_db_match[1] < 80:
                            continue
                            
                        h_db = h_db_match[0]
                        a_db = a_db_match[0]

                        # --- MOTOR DE IA (Clubes) ---
                        if not bundle_clubes: continue
                        model_clubes = bundle_clubes['model']
                        feature_cols_clubes = bundle_clubes['feature_cols']

                        stats_h = get_recent_stats(h_db, conn)
                        stats_a = get_recent_stats(a_db, conn)

                        forma_h = construir_forma_reciente(h_db, conn)
                        forma_a = construir_forma_reciente(a_db, conn)

                        pts_h = obtener_puntos_temporada(h_db, conn)
                        pts_a = obtener_puntos_temporada(a_db, conn)
                        dif_tabla = pts_h - pts_a
                        descanso_h = obtener_dias_descanso(h_db, conn)
                        descanso_a = obtener_dias_descanso(a_db, conn)
                        ventaja_fisica = descanso_h - descanso_a

                        fila_features = construir_fila_features(forma_h, forma_a, dif_tabla, ventaja_fisica)
                        X_pred = pd.DataFrame([fila_features])[feature_cols_clubes]

                        pred_probs = model_clubes.predict_proba(X_pred)[0]
                        prob_visita, prob_empate, prob_local = pred_probs[0], pred_probs[1], pred_probs[2]

                        pred_goles_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
                        pred_goles_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2
                        prom_goles_total = pred_goles_home + pred_goles_away
                        prom_corners_total = (stats_h['HC'] + stats_a['AC']) / 2
                        prom_shots_total = (stats_h['HST'] + stats_a['AST']) / 2

                        mercados_a_evaluar = [
                            ("Ganador (Local)", buscar_cuota_segura(row, ['1x2_home']), prob_local),
                            ("Empate", buscar_cuota_segura(row, ['1x2_draw']), prob_empate),
                            ("Ganador (Visita)", buscar_cuota_segura(row, ['1x2_away']), prob_visita)
                        ]

                        prob_over = poisson_prob_over  # alias local — evita NameError, ver línea ~23
                        def prob_under(promedio, umbral): return 1 - prob_over(promedio, umbral)
                        def prob_handicap(prom_favor, prom_contra, linea_hdp):
                            prob_acum = 0.0
                            for gf in range(15):
                                for gc in range(15):
                                    if (gf + linea_hdp) > gc:
                                        p_gf = (math.exp(-prom_favor) * (prom_favor**gf)) / math.factorial(gf)
                                        p_gc = (math.exp(-prom_contra) * (prom_contra**gc)) / math.factorial(gc)
                                        prob_acum += (p_gf * p_gc)
                            return prob_acum

                        for col_name, val in row.items():
                            if pd.isna(val) or str(val).strip() == '': continue
                            col_str = str(col_name).lower()
                            if col_str in ['es_mundial', 'liga', 'pais', 'partido_id', 'home', 'away', 'inicio_utc', 'inicio_local', 'fecha_match']: continue
                                
                            try:
                                val_num = float(val)
                                if val_num <= 1.0: continue
                            except ValueError: continue 
                                
                            # Mercados evaluados: 1x2, BTTS, Handicap, Goles Totales, Córners Totales
                            # Se excluyen tiros y goles por equipo (poco líquidos)
                            if 'btts' in col_str or 'ambos' in col_str:
                                prob_btts_si = (1 - math.exp(-pred_goles_home)) * (1 - math.exp(-pred_goles_away))
                                if 'yes' in col_str or 'si' in col_str: mercados_a_evaluar.append(("Ambos Anotan (Sí)", val_num, prob_btts_si))
                                elif 'no' in col_str: mercados_a_evaluar.append(("Ambos Anotan (No)", val_num, 1 - prob_btts_si))
                                continue

                            match = re.search(r'(-?\d+(?:\.\d+)?)', col_str)
                            if not match: continue
                            linea = float(match.group(1))

                            # Ignorar líneas asiáticas (.25 / .75) en TODOS los mercados
                            if round(abs(linea) % 1, 2) in (0.25, 0.75): continue

                            if 'hdp' in col_str or 'handicap' in col_str:
                                if 'home' in col_str: mercados_a_evaluar.append((f"Hándicap Local ({linea:+})", val_num, prob_handicap(pred_goles_home, pred_goles_away, linea)))
                                elif 'away' in col_str: mercados_a_evaluar.append((f"Hándicap Visita ({linea:+})", val_num, prob_handicap(pred_goles_away, pred_goles_home, linea)))
                            elif 'corners' in col_str or 'corner' in col_str:
                                # Mercados de córners — modelados con Poisson
                                if linea > 15.5: continue  # Cap para líneas irreales
                                if 'corners_home' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Local (+{linea})", val_num, prob_over(stats_h.get('HC', prom_corners_total), linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Local (-{linea})", val_num, prob_under(stats_h.get('HC', prom_corners_total), linea)))
                                elif 'corners_away' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Visita (+{linea})", val_num, prob_over(stats_a.get('AC', prom_corners_total), linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Visita (-{linea})", val_num, prob_under(stats_a.get('AC', prom_corners_total), linea)))
                                else:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Totales (+{linea})", val_num, prob_over(prom_corners_total, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Totales (-{linea})", val_num, prob_under(prom_corners_total, linea)))
                            elif ('goles' in col_str or 'total' in col_str) and 'tt_home' not in col_str and 'tt_away' not in col_str and 'shots' not in col_str:
                                # Cap: ignorar lineas de goles por encima de 5.5 (no son realistas)
                                if linea > 5.5: continue
                                if 'over' in col_str: mercados_a_evaluar.append((f"Goles Totales (+{linea})", val_num, prob_over(prom_goles_total, linea)))
                                elif 'under' in col_str: mercados_a_evaluar.append((f"Goles Totales (-{linea})", val_num, prob_under(prom_goles_total, linea)))

                        def evaluar_edge(mercado_nombre, prob_ia, cuota):
                            if cuota is None: return
                            try:
                                cuota_flt = float(cuota)
                                edge = prob_ia - (1 / cuota_flt)
                                log_debug.append(f"Evaluando: {h_db} - {mercado_nombre} | Cuota: {cuota_flt} | Prob IA: {prob_ia:.1%} | Edge: {edge:.2%}")
                                # Pool completo: cualquier mercado con edge positivo y cuota razonable
                                if edge > 0 and cuota_flt < 8.0:
                                    mercados_evaluados_completos.append((fecha_partido, h_db, a_db, mercado_nombre, cuota_flt, prob_ia, edge))
                                if 0.02 <= edge <= 0.15:
                                    oportunidades.append((fecha_partido, h_db, a_db, mercado_nombre, cuota_flt, prob_ia, edge))
                                    log_debug.append(f"   AÑADIDO AL PORTAFOLIO. Edge válido: {edge:.2%}")
                            except Exception: pass

                        for nombre_mkt, cuota_val, prob_ia in mercados_a_evaluar:
                            evaluar_edge(nombre_mkt, prob_ia, cuota_val)

                    with st.expander("🛠️ Ver Diagnóstico Completo del Robot Evaluador"):
                        for log_msg in log_debug: st.text(log_msg)

                    st.session_state['n_partidos_dia'] = n_partidos_dia

                    if oportunidades:
                        df_ops = pd.DataFrame(oportunidades, columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge'])
                        st.session_state['portafolio_escaneado'] = df_ops.sort_values(by='Edge', ascending=False).drop_duplicates(subset=['Home', 'Away', 'Mercado']).reset_index(drop=True)
                        st.session_state['pool_mercados'] = mercados_evaluados_completos
                    else:
                        st.caption("No se encontraron ineficiencias dentro del rango rentable (2% a 15%).")

            if 'portafolio_escaneado' in st.session_state:
                df_ops = st.session_state['portafolio_escaneado'].copy()
                df_ops['Partido'] = df_ops['Home'] + " vs " + df_ops['Away']
                df_ops['Edge_Str'] = (df_ops['Edge'] * 100).round(2).astype(str) + "%"
                df_ops['Prob_IA_Str'] = (df_ops['Prob_IA'] * 100).round(1).astype(str) + "%"
                
                columnas_base = ['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str', 'Date', 'Home', 'Away', 'Prob_IA', 'Edge']
                df_ops = df_ops[columnas_base]

                n_partidos_dia = st.session_state.get('n_partidos_dia', 0)

                # ══════════════════════════════════════════════════════════
                # Regla mínima de portafolio: si el día escaneado tiene menos
                # de 3 partidos, no se arma un portafolio con esa estructura de
                # riesgo (1-3-3-3) — sólo se muestran los picks encontrados como
                # sugerencias informativas, sin la opción de guardarlos.
                # ══════════════════════════════════════════════════════════
                if n_partidos_dia < 3:
                    st.warning(
                        f"Sólo se detectaron **{n_partidos_dia} partido(s)** este día — por debajo "
                        "del mínimo de 3 requerido para construir un portafolio. Los picks se muestran "
                        "abajo como **sugerencias** y no se agregan al portafolio."
                    )
                    df_sugeridos = df_ops.sort_values(by='Edge', ascending=False).reset_index(drop=True)
                    st.markdown(f"### 💡 Picks Sugeridos ({len(df_sugeridos)})")
                    st.caption("Picks con edge positivo encontrados en el escaneo — sólo informativos, no forman parte del portafolio.")
                    st.dataframe(
                        df_sugeridos[['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    TARGET_PICKS = 10

                    selected_indices = []
                    used_matches = set()
                    df_top_10_list = []

                    # ── Definición de niveles de riesgo por cuota ─────────────
                    BUCKETS = [
                        # (label, min_cuota, max_cuota)
                        ('🔴 Alto (>2.50)',        2.50, 9999.0),
                        ('🟡 Medio (1.90–2.49)',   1.90,  2.50),
                        ('🟢 Bajo (<1.90)',         0.0,   1.90),
                    ]
                    # Contadores por bucket (índice = posición en BUCKETS)
                    bucket_counts = [0, 0, 0]

                    # ── helpers ───────────────────────────────────────────────
                    def _mercados_elegidos(partido):
                        return {r.iloc[0]['Mercado'] for r in df_top_10_list if r.iloc[0]['Partido'] == partido}

                    # Mercados 1x2 — para evitar picks duplicados del mismo partido
                    MERCADOS_1X2 = {"Ganador (Local)", "Empate", "Ganador (Visita)"}

                    def _tiene_1x2(partido):
                        """Retorna True si ya hay un pick 1x2 (Local/Empate/Visita) de este partido."""
                        for r in df_top_10_list:
                            row0 = r.iloc[0]
                            if row0['Partido'] == partido and row0['Mercado'] in MERCADOS_1X2:
                                return True
                        return False

                    def _familia_mercado(mercado_nombre):
                        """
                        Devuelve el 'tipo' de mercado para detectar correlaciones dentro
                        del mismo partido.  Mercados de la misma familia sobre el mismo
                        partido son correlacionados: p.ej. "Goles Totales (+1.5)" y
                        "Goles Totales (+2.5)" apuntan al mismo resultado con distintas
                        líneas y no deben coexistir en el portafolio.
                        Retorna None para mercados que NO tienen restricción de familia
                        (p.ej. Hándicap, BTTS) — esos pueden convivir.
                        """
                        m = mercado_nombre.lower()
                        if 'goles totales' in m:      return 'goles_totales'
                        if 'goles local' in m:        return 'goles_local'
                        if 'goles visita' in m:       return 'goles_visita'
                        if 'córners totales' in m:    return 'corners_totales'
                        if 'córners local' in m:      return 'corners_local'
                        if 'córners visita' in m:     return 'corners_visita'
                        if 'tiros' in m:              return 'tiros'
                        # 1x2 se gestiona por separado con _tiene_1x2
                        return None   # sin restricción de familia

                    def _tiene_misma_familia(partido, mercado_nuevo):
                        """
                        Retorna True si ya hay en el portafolio un pick del mismo partido
                        con el mismo 'tipo' de mercado (familia) que el pick nuevo.
                        Cuando hay conflicto de familia, el pick con MAYOR edge ya ganó
                        su lugar primero (df_ops está ordenado desc por edge), por lo que
                        simplemente bloqueamos el nuevo.
                        """
                        familia = _familia_mercado(mercado_nuevo)
                        if familia is None:
                            return False   # mercado sin familia → sin restricción
                        for r in df_top_10_list:
                            row0 = r.iloc[0]
                            if row0['Partido'] == partido and _familia_mercado(row0['Mercado']) == familia:
                                return True
                        return False

                    def add_pick(row_or_idx, nivel_label, from_df=True):
                        """Agrega pick. from_df=True usa índice de df_ops, False recibe dict.
                        Bloquea:
                          · mismo partido+mercado exacto
                          · múltiples picks 1x2 del mismo partido
                          · múltiples picks de la misma familia de mercado (correlacionados)
                            → sólo entra el de mayor edge (df_ops está ordenado desc)
                        """
                        if from_df:
                            row = df_ops.loc[row_or_idx]
                            partido = row['Partido']
                            mercado = row['Mercado']
                            # Bloquear mismo partido+mercado exacto
                            if (partido, mercado) in {(r.iloc[0]['Partido'], r.iloc[0]['Mercado']) for r in df_top_10_list}:
                                return False
                            # Evitar dos picks 1x2 del mismo partido
                            if mercado in MERCADOS_1X2 and _tiene_1x2(partido):
                                return False
                            # Evitar mercados correlacionados del mismo partido (p.ej. +1.5 y +2.5 goles)
                            if _tiene_misma_familia(partido, mercado):
                                return False
                            used_matches.add(partido)
                            selected_indices.append(row_or_idx)
                            df_top_10_list.append(df_ops.loc[[row_or_idx]].assign(Nivel=nivel_label))
                        else:
                            d = row_or_idx
                            partido = d['Partido']
                            mercado = d['Mercado']
                            if (partido, mercado) in {(r.iloc[0]['Partido'], r.iloc[0]['Mercado']) for r in df_top_10_list}:
                                return False
                            # Evitar dos picks 1x2 del mismo partido
                            if mercado in MERCADOS_1X2 and _tiene_1x2(partido):
                                return False
                            # Evitar mercados correlacionados del mismo partido
                            if _tiene_misma_familia(partido, mercado):
                                return False
                            used_matches.add(partido)
                            df_top_10_list.append(pd.DataFrame([d]))
                        return True

                    def get_pool(min_c, max_c, edge_min, edge_max):
                        """Pool filtrado por cuota y edge, excluyendo pares partido+mercado ya usados."""
                        ya_usados = {(r.iloc[0]['Partido'], r.iloc[0]['Mercado']) for r in df_top_10_list}
                        mask = (
                            ~df_ops.apply(lambda r: (r['Partido'], r['Mercado']) in ya_usados, axis=1)
                            & (df_ops['Cuota'] >= min_c) & (df_ops['Cuota'] < max_c)
                            & (df_ops['Edge'] >= edge_min) & (df_ops['Edge'] <= edge_max)
                        )
                        return df_ops[mask]

                    def fill_bucket(bucket_idx, target_n, nivel_label, edge_min, edge_max):
                        """Intenta llenar el bucket hasta target_n picks."""
                        _, min_c, max_c = BUCKETS[bucket_idx]
                        added = 0
                        pool = get_pool(min_c, max_c, edge_min, edge_max)
                        for idx in pool.index:
                            if added >= target_n:
                                break
                            if add_pick(idx, nivel_label):
                                added += 1
                        return added

                    def faltantes_bucket(conteo_actual, target=3):
                        return max(0, target - conteo_actual)

                    # ══════════════════════════════════════════════════════════
                    # FASE 1 — estructura 1-3-3-3 con edge ESTÁNDAR (2%–15%)
                    # ══════════════════════════════════════════════════════════
                    EDGE_STD_MIN, EDGE_STD_MAX = 0.02, 0.15

                    # Golden Pick: mejor edge global (cualquier cuota)
                    for idx in df_ops[
                        (df_ops['Edge'] >= EDGE_STD_MIN)
                        & (df_ops['Edge'] <= EDGE_STD_MAX)
                    ].index:
                        if add_pick(idx, '⭐ Golden Pick'):
                            break

                    # Si no hay golden con estándar, usar el mejor edge disponible (≤30%)
                    if len(df_top_10_list) == 0:
                        for idx in df_ops[df_ops['Edge'] <= 0.30].index:
                            if add_pick(idx, '⭐ Golden Pick (ext.)'):
                                break

                    # Buckets estándar: 3 altos, 3 medios, 3 bajos
                    for bi in range(3):
                        lbl = BUCKETS[bi][0]
                        bucket_counts[bi] = fill_bucket(bi, 3, lbl, EDGE_STD_MIN, EDGE_STD_MAX)

                    # ══════════════════════════════════════════════════════════
                    # FASE 2 — ampliar edge HASTA 30% (en pasos), manteniendo
                    #          la estructura 1-3-3-3 bucket por bucket.
                    #          NUNCA superar 30% de edge máximo.
                    # ══════════════════════════════════════════════════════════
                    EDGE_PASOS = [
                        (0.01,  0.20),   # paso 1: relajar un poco
                        (0.005, 0.25),   # paso 2: abrir más
                        (0.001, 0.30),   # paso 3: hasta el máximo permitido (30%)
                    ]

                    for edge_min_exp, edge_max_exp in EDGE_PASOS:
                        if all(bucket_counts[bi] >= 3 for bi in range(3)):
                            break
                        lbl_suf = f'edge {edge_min_exp:.1%}–{edge_max_exp:.0%}'
                        for bi in range(3):
                            falt = faltantes_bucket(bucket_counts[bi])
                            if falt:
                                lbl = f'{BUCKETS[bi][0]} — {lbl_suf}'
                                added = fill_bucket(bi, falt, lbl, edge_min_exp, edge_max_exp)
                                bucket_counts[bi] += added
                    # La estructura 1-3-3-3 se preserva porque fill_bucket sólo toma
                    # picks del rango de cuota del bucket correspondiente.

                    # ══════════════════════════════════════════════════════════
                    # FASE 3 — estructura flexible con dos pasos:
                    #   Paso A: intentar completar la 1-3-3-3 con el pool completo
                    #           (≤30% edge, cada pick en su bucket por cuota).
                    #   Paso B: slots aún vacíos → rellenar SÓLO con cuota < 2.0.
                    # Permite repetir partido si el mercado es distinto,
                    # pero nunca dos picks 1x2 del mismo partido.
                    # ══════════════════════════════════════════════════════════
                    if len(df_top_10_list) < TARGET_PICKS:
                        faltantes_f3 = TARGET_PICKS - len(df_top_10_list)

                        pool_completo = st.session_state.get('pool_mercados', [])

                        if pool_completo:
                            df_f3 = pd.DataFrame(
                                pool_completo,
                                columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge']
                            )
                            df_f3['Partido'] = df_f3['Home'] + " vs " + df_f3['Away']
                            df_f3['Prob_IA_Str'] = (df_f3['Prob_IA'] * 100).round(1).astype(str) + "%"
                            df_f3['Edge_Str']    = (df_f3['Edge']    * 100).round(2).astype(str) + "%"

                            # Respetar el tope de edge del 30%
                            df_f3 = df_f3[df_f3['Edge'] <= 0.30]

                            # Construir set de mercados exactos ya elegidos (partido+mercado)
                            mercados_ya_en_portfolio = set()
                            for item in df_top_10_list:
                                r = item.iloc[0]
                                mercados_ya_en_portfolio.add((r['Partido'], r['Mercado']))

                            df_f3 = df_f3[
                                ~df_f3.apply(
                                    lambda r: (r['Partido'], r['Mercado']) in mercados_ya_en_portfolio,
                                    axis=1
                                )
                            ]

                            # Ordenar: preferir partidos nuevos, luego mayor edge
                            df_f3['es_partido_nuevo'] = (~df_f3['Partido'].isin(used_matches)).astype(int)
                            df_f3 = df_f3.sort_values(['es_partido_nuevo', 'Edge'], ascending=[False, False])

                            _picks_f3 = [0]  # contador mutable para uso dentro de _add_f3

                            def _add_f3(row_f3, nivel_f3, bi_f3):
                                """Intenta agregar un pick de Fase 3; retorna True si lo logró."""
                                partido_f3 = row_f3['Partido']
                                mercado_f3 = row_f3['Mercado']
                                if (partido_f3, mercado_f3) in mercados_ya_en_portfolio:
                                    return False
                                if mercado_f3 in MERCADOS_1X2 and _tiene_1x2(partido_f3):
                                    return False
                                if _tiene_misma_familia(partido_f3, mercado_f3):
                                    return False
                                entry = {
                                    'Date': row_f3['Date'], 'Home': row_f3['Home'], 'Away': row_f3['Away'],
                                    'Mercado': mercado_f3, 'Cuota': row_f3['Cuota'],
                                    'Prob_IA': row_f3['Prob_IA'], 'Edge': row_f3['Edge'],
                                    'Prob_IA_Str': row_f3['Prob_IA_Str'], 'Edge_Str': row_f3['Edge_Str'],
                                    'Partido': partido_f3, 'Nivel': nivel_f3
                                }
                                if partido_f3 not in used_matches:
                                    used_matches.add(partido_f3)
                                mercados_ya_en_portfolio.add((partido_f3, mercado_f3))
                                df_top_10_list.append(pd.DataFrame([entry]))
                                bucket_counts[bi_f3] += 1
                                _picks_f3[0] += 1
                                return True

                            # ── Paso A: completar slots 1-3-3-3 faltantes ──────────
                            # Golden si sigue vacío
                            n_golden_f3 = sum(
                                1 for d in df_top_10_list
                                if 'Golden' in str(d.iloc[0].get('Nivel', ''))
                            )
                            if n_golden_f3 == 0 and _picks_f3[0] < faltantes_f3:
                                for _, row_f3 in df_f3.iterrows():
                                    if _add_f3(row_f3, '⭐ Golden Pick (Flexible)', 0):
                                        break  # sólo 1 golden

                            # Buckets faltantes en orden alto → medio → bajo
                            for bi in range(3):
                                falt_bi = faltantes_bucket(bucket_counts[bi])
                                if falt_bi == 0 or _picks_f3[0] >= faltantes_f3:
                                    continue
                                _, min_c, max_c = BUCKETS[bi]
                                lbl_bi = f'{BUCKETS[bi][0]} — flexible'
                                cnt_bi = 0
                                for _, row_f3 in df_f3.iterrows():
                                    if cnt_bi >= falt_bi or _picks_f3[0] >= faltantes_f3:
                                        break
                                    if not (min_c <= row_f3['Cuota'] < max_c):
                                        continue
                                    if _add_f3(row_f3, lbl_bi, bi):
                                        cnt_bi += 1

                            # ── Paso B: slots aún vacíos → sólo cuota < 2.0 ───────
                            if _picks_f3[0] < faltantes_f3:
                                df_f3_bajo = df_f3[df_f3['Cuota'] < 2.0].copy()
                                for _, row_f3 in df_f3_bajo.iterrows():
                                    if _picks_f3[0] >= faltantes_f3:
                                        break
                                    _add_f3(row_f3, '🟢 Flexible bajo (<2.0)', 2)

                            if _picks_f3[0] > 0:
                                st.caption(
                                    f"Modo flexible (≤30% edge): {_picks_f3[0]} pick(s) añadidos fuera del rango estándar (2%–15%)."
                                )

                    faltantes = TARGET_PICKS - len(df_top_10_list)
                    if faltantes > 0:
                        st.caption(
                            f"Portafolio parcial: {len(df_top_10_list)}/10 picks — no hay suficiente mercado con edge positivo para los {faltantes} slots restantes."
                        )

                    df_top_10 = pd.concat(df_top_10_list).reset_index(drop=True) if df_top_10_list else pd.DataFrame()

                    # Asegurar columnas string para display
                    if not df_top_10.empty:
                        if 'Prob_IA_Str' not in df_top_10.columns:
                            df_top_10['Prob_IA_Str'] = (df_top_10['Prob_IA'] * 100).round(1).astype(str) + "%"
                        if 'Edge_Str' not in df_top_10.columns:
                            df_top_10['Edge_Str'] = (df_top_10['Edge'] * 100).round(2).astype(str) + "%"
                        if 'Partido' not in df_top_10.columns:
                            df_top_10['Partido'] = df_top_10['Home'] + " vs " + df_top_10['Away']

                    cols_mostrar = ['Nivel', 'Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']

                    df_mostrar_top = df_top_10[cols_mostrar].copy()
                    df_mostrar_top.insert(0, "✅ Añadir", True)
                
                    df_reserva = df_ops[~df_ops.index.isin(selected_indices)].reset_index(drop=True)

                    # ── Picks válidos fuera del portafolio (pool completo con edge positivo) ──
                    # Incluye tanto los del rango 2-15% que no entraron como los del pool
                    # completo (edge positivo < 2% o > 15%) que quedaron fuera.
                    _portfolio_pares = {(r.iloc[0]['Partido'], r.iloc[0]['Mercado']) for r in df_top_10_list}
                    _pool_completo = st.session_state.get('pool_mercados', [])
                    if _pool_completo:
                        df_reserva_full = pd.DataFrame(
                            _pool_completo,
                            columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge']
                        )
                        df_reserva_full['Partido'] = df_reserva_full['Home'] + " vs " + df_reserva_full['Away']
                        df_reserva_full['Prob_IA_Str'] = (df_reserva_full['Prob_IA'] * 100).round(1).astype(str) + "%"
                        df_reserva_full['Edge_Str'] = (df_reserva_full['Edge'] * 100).round(2).astype(str) + "%"
                        # Excluir lo que ya está en el portafolio
                        df_reserva_full = df_reserva_full[
                            ~df_reserva_full.apply(lambda r: (r['Partido'], r['Mercado']) in _portfolio_pares, axis=1)
                        ].drop_duplicates(subset=['Partido', 'Mercado']).sort_values('Edge', ascending=False).reset_index(drop=True)
                    else:
                        df_reserva_full = df_reserva.copy()
                        if 'Prob_IA_Str' not in df_reserva_full.columns:
                            df_reserva_full['Prob_IA_Str'] = (df_reserva_full['Prob_IA'] * 100).round(1).astype(str) + "%"
                        if 'Edge_Str' not in df_reserva_full.columns:
                            df_reserva_full['Edge_Str'] = (df_reserva_full['Edge'] * 100).round(2).astype(str) + "%"

                    df_mostrar_reserva = df_reserva_full[['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']].copy()
                    df_mostrar_reserva.insert(0, "✅ Añadir", False)

                    modo = "completo ✅" if len(df_top_10) >= TARGET_PICKS else f"parcial ({len(df_top_10)}/{TARGET_PICKS} picks)"
                    st.success(f"Escaneo listo. Portafolio {modo} — {len(df_top_10)} picks seleccionados.")

                    # ── Resumen de estructura 1-3-3-3 ─────────────────────────
                    modo_stake_riesgo = st.toggle(
                        "Stake por Nivel de Riesgo",
                        key="modo_stake_riesgo",
                        help="ON: Golden 1.5×, Bajo 1.2×, Medio 1.0×, Alto 0.5× de la unidad base. OFF: mismo stake para todos los picks."
                    )
                    n_golden = sum(1 for d in df_top_10_list if 'Golden' in str(d.iloc[0].get('Nivel','')))
                    _bc = bucket_counts  # [alto, medio, bajo]
                    col_g, col_h, col_m, col_l = st.columns(4)
                    col_g.metric("Golden", f"{n_golden}/1")
                    col_h.metric("Alto",   f"{_bc[0]}/3")
                    col_m.metric("Medio",  f"{_bc[1]}/3")
                    col_l.metric("Bajo",   f"{_bc[2]}/3")

                    st.markdown(f"### Portafolio ({len(df_top_10)} picks)")
                
                    edit_top10 = st.data_editor(
                        df_mostrar_top,
                        hide_index=True,
                        use_container_width=True,
                        key="editor_top10",
                        column_config={"✅ Añadir": st.column_config.CheckboxColumn(required=True)}
                    )
                
                    with st.expander(f"📂 Ver el resto de picks válidos ({len(df_reserva_full)} con edge positivo fuera del portafolio)"):
                        if not df_mostrar_reserva.empty:
                            st.caption("Todos los picks con edge positivo que NO entraron al portafolio — ordenados por edge de mayor a menor. Puedes marcar cualquiera como reemplazo.")
                            edit_reserva = st.data_editor(
                                df_mostrar_reserva,
                                hide_index=True,
                                use_container_width=True,
                                key="editor_reserva",
                                column_config={"✅ Añadir": st.column_config.CheckboxColumn(required=True)}
                            )
                        else:
                            st.info("No hay más picks de reserva. Se utilizaron todos los disponibles.")

                    if st.button("Guardar Portafolio Seleccionado", type="primary"):
                        indices_top = edit_top10[edit_top10["✅ Añadir"] == True].index
                        indices_res = edit_reserva[edit_reserva["✅ Añadir"] == True].index if not df_mostrar_reserva.empty else []
                    
                        df_final_top = df_top_10.iloc[indices_top]
                        df_final_res = df_reserva_full.iloc[indices_res] if not df_mostrar_reserva.empty else pd.DataFrame()
                        df_final_a_guardar = pd.concat([df_final_top, df_final_res])
                    
                        if df_final_a_guardar.empty:
                            st.warning("No seleccionaste ningún pick.")
                        else:
                            _modo_cmp = st.session_state.get("modo_compuesto", False)
                            _modo_riesgo = st.session_state.get("modo_stake_riesgo", False)
                            if _modo_cmp:
                                _pnl_hist = pd.read_sql(
                                    "SELECT COALESCE(SUM(Beneficio_Neto),0) as total FROM portafolio_historico WHERE Estado != 'Pendiente'",
                                    conn
                                ).iloc[0]['total']
                                _base_inversion = inversion_total + _pnl_hist
                            else:
                                _base_inversion = inversion_total

                            _n_picks = len(df_final_a_guardar)

                            # ── Días con menos de 8 picks → flat 500 por pick ──
                            if _n_picks < 8:
                                def _stake_para_pick(row, unidad):
                                    return 500.0
                                _stake_label = f"Flat $500/pick ({_n_picks} picks < 8)"
                            elif _modo_riesgo:
                                # Stake por nivel de riesgo: la unidad base se distribuye
                                # ponderada por multiplicador según nivel (Nivel column).
                                # Golden 1.5×, Bajo 1.2×, Medio 1.0×, Alto 0.5×
                                def _mult_nivel(nivel_str):
                                    n = str(nivel_str).lower()
                                    if 'golden' in n: return 1.5
                                    if 'bajo'   in n: return 1.2
                                    if 'medio'  in n: return 1.0
                                    if 'alto'   in n: return 0.5
                                    return 1.0  # fallback

                                _mults = df_final_a_guardar['Nivel'].apply(_mult_nivel) if 'Nivel' in df_final_a_guardar.columns else pd.Series([1.0]*_n_picks)
                                _suma_mults = _mults.sum()
                                _unidad_base = _base_inversion / _suma_mults if _suma_mults > 0 else _base_inversion / _n_picks

                                def _stake_para_pick(row, unidad=_unidad_base):
                                    return unidad * _mult_nivel(row.get('Nivel', ''))

                                _stake_label = f"Por Riesgo — unidad ${_unidad_base:,.0f}"
                            else:
                                stake_plano = _base_inversion / _n_picks
                                def _stake_para_pick(row, unidad=stake_plano):
                                    return unidad
                                _stake_label = f"Flat ${stake_plano:,.0f}/pick"

                            for _, row in df_final_a_guardar.iterrows():
                                _stake_pick = _stake_para_pick(row)
                                cursor.execute("""
                                    INSERT INTO portafolio_historico (Date, HomeTeam, AwayTeam, Mercado, Cuota, Prob_IA, Edge, Stake)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (row['Date'], row['Home'], row['Away'], row['Mercado'], row['Cuota'], row['Prob_IA'], row['Edge'], _stake_pick))
                            conn.commit()
                            _modo_badge = "Compuesto" if _modo_cmp else "Flat"
                            st.success(f"{_n_picks} picks guardados ({_modo_badge} — Bankroll: ${_base_inversion:,.0f} | {_stake_label})")
                            del st.session_state['portafolio_escaneado']

        except Exception as e:
            st.error(f"Error en la aplicación: {e}")

        # ── PORTAFOLIO ACTIVO (siempre visible, no requiere re-escanear) ──
        st.divider()
        st.markdown("### Portafolio Activo (Picks Pendientes)")
        df_activos = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente' ORDER BY Date ASC", conn)
        if df_activos.empty:
            st.info("No hay picks pendientes guardados. Escanea el mercado y guarda tu portafolio para verlo aquí.")
        else:
            # Agrupar por fecha para visualizar días distintos
            fechas_activas = sorted(df_activos['Date'].unique())
            for fecha in fechas_activas:
                df_dia = df_activos[df_activos['Date'] == fecha].copy()
                stake_dia = df_dia['Stake'].sum()
                retorno_potencial = (df_dia['Cuota'] * df_dia['Stake']).sum()
                with st.expander(f"{fecha}  —  {len(df_dia)} picks  |  Invertido: ${stake_dia:,.0f}  |  Retorno potencial: ${retorno_potencial:,.0f}", expanded=(fecha == fechas_activas[-1])):
                    df_mostrar_activos = df_dia[['HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Prob_IA', 'Edge', 'Stake']].copy()
                    df_mostrar_activos['Prob_IA'] = (df_mostrar_activos['Prob_IA'] * 100).round(1).astype(str) + "%"
                    df_mostrar_activos['Edge'] = (df_mostrar_activos['Edge'] * 100).round(2).astype(str) + "%"
                    df_mostrar_activos['Partido'] = df_mostrar_activos['HomeTeam'] + " vs " + df_mostrar_activos['AwayTeam']
                    df_mostrar_activos = df_mostrar_activos[['Partido', 'Mercado', 'Cuota', 'Prob_IA', 'Edge', 'Stake']]
                    st.dataframe(df_mostrar_activos, hide_index=True, use_container_width=True)

    with tab2:
        c_tit, c_btn = st.columns([0.75, 0.25])
        c_tit.subheader("Rendimiento Acumulado")
        
        if c_btn.button("Resetear Historial", use_container_width=True):
            cursor.execute("DELETE FROM portafolio_historico")
            conn.commit()
            st.toast("¡Historial borrado con éxito! Portafolio limpio.")
            st.rerun()
            
        if st.button("Liquidar Apuestas Pendientes", type="primary"):
            df_pendientes = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente'", conn)
            liquidadas = 0
            beneficio_reciente = 0.0
            stake_reciente = 0.0

            import re
            from thefuzz import fuzz as fuzz_lib
            from datetime import datetime, timedelta

            # Normalización via diccionario_alias centralizado
            def _liq_norm(nombre):
                """Normaliza un nombre de equipo via ALIAS_GLOBAL (case-insensitive)."""
                return normalizar_nombre(str(nombre))

            def _encontrar_fila(res_df, pick_home_raw, pick_away_raw):
                """Busca la fila del partido en res_df normalizando AMBOS lados
                (pick Y candidatos DB) antes de comparar. Esto resuelve el caso
                en que la DB guarda el mismo equipo con nombres distintos segun
                la temporada (ej. M'gladbach vs Borussia Monchengladbach)."""
                pick_h = _liq_norm(pick_home_raw)
                pick_a = _liq_norm(pick_away_raw)
                # 1. Comparar normalizando el lado DB tambien
                for _, cand in res_df.iterrows():
                    if _liq_norm(cand["HomeTeam"]) == pick_h and _liq_norm(cand["AwayTeam"]) == pick_a:
                        return cand
                # 2. Fuzzy sobre nombres ya normalizados en ambos lados
                best_row, best_score = None, 0
                for _, cand in res_df.iterrows():
                    db_h = _liq_norm(cand["HomeTeam"])
                    db_a = _liq_norm(cand["AwayTeam"])
                    h_s = fuzz_lib.token_set_ratio(pick_h, db_h)
                    a_s = fuzz_lib.token_set_ratio(pick_a, db_a)
                    combined = (h_s + a_s) / 2
                    if combined > best_score and h_s >= 70 and a_s >= 70:
                        best_score = combined
                        best_row = cand
                return best_row

            for _, pick in df_pendientes.iterrows():
                try:
                    fecha_dt = datetime.strptime(pick['Date'], '%Y-%m-%d')
                    fecha_inicio = (fecha_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                    fecha_fin    = (fecha_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception:
                    fecha_inicio = fecha_fin = pick['Date']

                q_res = f"""
                SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST, NULL AS FueProrroga, NULL AS FTHG_r, NULL AS FTAG_r, NULL AS HC_r, NULL AS AC_r, 0 AS EsMundial FROM historial_multiliga_ml WHERE Date BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                UNION ALL
                SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST, FueProrroga, FTHG_r, FTAG_r, HC_r, AC_r, 1 AS EsMundial FROM historial_selecciones_ml WHERE Date BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                """
                try:
                    res_real = pd.read_sql(q_res, conn)
                except Exception:
                    res_real = pd.DataFrame()

                if res_real.empty:
                    continue

                row = _encontrar_fila(res_real, str(pick['HomeTeam']), str(pick['AwayTeam']))
                if row is None:
                    continue

                hg  = row['FTHG'] if pd.notna(row.get('FTHG')) else None
                ag  = row['FTAG'] if pd.notna(row.get('FTAG')) else None
                if hg is None or ag is None:
                    continue
                hg, ag = int(hg), int(ag)

                # Prórroga: si el partido fue a tiempo extra, se considera empate
                # y se usa el resultado de tiempo reglamentario (FTHG_r/FTAG_r)
                # para 1X2, Ambos Anotan, Hándicap y Over/Under de goles.
                if pd.notna(row.get('FueProrroga')) and int(row['FueProrroga']) == 1:
                    hg_r, ag_r = row.get('FTHG_r'), row.get('FTAG_r')
                    if pd.notna(hg_r) and pd.notna(ag_r):
                        hg, ag = int(hg_r), int(ag_r)

                hc  = int(row['HC'])  if pd.notna(row.get('HC'))  else 0
                ac  = int(row['AC'])  if pd.notna(row.get('AC'))  else 0
                hst = int(row['HST']) if pd.notna(row.get('HST')) else 0
                ast = int(row['AST']) if pd.notna(row.get('AST')) else 0

                # Córners:
                # - Clubes: siempre HC/AC.
                # - Mundial CON prórroga (FueProrroga = 1): se liquida con TIEMPO
                #   REGLAMENTARIO (HC_r/AC_r), porque HC/AC incluye los córners de
                #   la prórroga. Si todavía no cargaste HC_r/AC_r a mano para este
                #   partido, el pick de córners se queda 'Pendiente'.
                # - Mundial SIN prórroga (FueProrroga = 0): HC/AC ya es el dato
                #   completo de tiempo reglamentario, igual que en clubes.
                es_wc = bool(row.get('EsMundial'))
                fue_prorroga = es_wc and pd.notna(row.get('FueProrroga')) and int(row['FueProrroga']) == 1

                if fue_prorroga:
                    hc_r_raw, ac_r_raw = row.get('HC_r'), row.get('AC_r')
                    corners_disponible = pd.notna(hc_r_raw) and pd.notna(ac_r_raw)
                    if corners_disponible:
                        hc_corner, ac_corner = int(hc_r_raw), int(ac_r_raw)
                else:
                    corners_disponible = True
                    hc_corner, ac_corner = hc, ac

                mkt = pick['Mercado']
                if "Córners" in mkt and not corners_disponible:
                    continue

                ganada = False
                if mkt == "Ganador (Local)":    ganada = (hg > ag)
                elif mkt == "Empate":            ganada = (hg == ag)
                elif mkt == "Ganador (Visita)":  ganada = (ag > hg)
                elif mkt == "Ambos Anotan (Sí)": ganada = (hg > 0 and ag > 0)
                elif mkt == "Ambos Anotan (No)": ganada = (hg == 0 or ag == 0)
                else:
                    match = re.search(r'\(([+-]\d+\.5)\)', mkt)
                    if match:
                        signo            = match.group(1)[0]
                        valor_linea      = float(match.group(1)[1:])
                        linea_matematica = float(match.group(1))
                        if "Hándicap" in mkt:
                            if "Local"    in mkt: ganada = (hg + linea_matematica > ag)
                            elif "Visita" in mkt: ganada = (ag + linea_matematica > hg)
                        else:
                            score = -1
                            if "Goles Local"       in mkt: score = hg
                            elif "Goles Visita"    in mkt: score = ag
                            elif "Goles"           in mkt: score = hg + ag
                            elif "Córners Local"   in mkt: score = hc_corner
                            elif "Córners Visita"  in mkt: score = ac_corner
                            elif "Córners"         in mkt: score = hc_corner + ac_corner
                            elif "Tiros Local"     in mkt: score = hst
                            elif "Tiros Visita"    in mkt: score = ast
                            elif "Tiros"           in mkt: score = hst + ast
                            if signo == '+': ganada = (score > valor_linea)
                            elif signo == '-': ganada = (score < valor_linea)

                estado    = 'Ganada' if ganada else 'Perdida'
                beneficio = (pick['Stake'] * pick['Cuota']) - pick['Stake'] if ganada else -pick['Stake']
                cursor.execute(
                    "UPDATE portafolio_historico SET Estado = ?, Beneficio_Neto = ? WHERE id = ?",
                    (estado, beneficio, pick['id'])
                )
                liquidadas         += 1
                beneficio_reciente += beneficio
                stake_reciente     += pick['Stake']

            conn.commit()
            if liquidadas > 0:
                yield_tanda = (beneficio_reciente / stake_reciente * 100) if stake_reciente > 0 else 0
                st.success(f"Se liquidaron {liquidadas} partidos. Beneficio de esta tanda: **${beneficio_reciente:,.0f}** (Yield: **{yield_tanda:.2f}%**)")
            else:
                st.info("No hay partidos nuevos terminados para liquidar.")
        df_hist = pd.read_sql("SELECT * FROM portafolio_historico", conn)

        st.divider()
        if not df_hist.empty:
            df_cerradas = df_hist[df_hist['Estado'] != 'Pendiente']
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            with c_res1:
                st.metric("Picks Cerrados", len(df_cerradas))
            with c_res2:
                ganadas = len(df_cerradas[df_cerradas['Estado'] == 'Ganada'])
                win_rate = (ganadas / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with c_res3:
                beneficio_total = df_cerradas['Beneficio_Neto'].sum()
                inversion_total_hist = df_cerradas['Stake'].sum()
                yield_global = (beneficio_total / inversion_total_hist * 100) if inversion_total_hist > 0 else 0
                st.metric("Yield (ROI)", f"{yield_global:.2f}%")
            with c_res4:
                st.metric("Ganancia Neta Global", f"${beneficio_total:,.0f}")

            st.divider()
            # ── Vista estándar (picks individuales) ───────────────
            # ── Descarga del portafolio ───────────────────────────
            st.write("**Historial de Picks**")
            col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1])
            df_dl = df_hist[['Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Beneficio_Neto', 'Prob_IA', 'Edge']].copy()

            with col_dl2:
                csv_bytes = df_dl.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar CSV",
                    data=csv_bytes,
                    file_name=f"portafolio_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_dl3:
                json_bytes = df_dl.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="Descargar JSON",
                    data=json_bytes,
                    file_name=f"portafolio_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            df_mostrar = df_hist[['Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Beneficio_Neto']].copy()
            df_mostrar['Es_Pendiente'] = df_mostrar['Estado'] == 'Pendiente'
            df_mostrar = df_mostrar.sort_values(by=['Es_Pendiente', 'Date'], ascending=[False, False]).drop(columns=['Es_Pendiente'])
            
            def color_estado(val):
                if val == 'Ganada': return 'color: #00FF00; font-weight: bold'
                elif val == 'Perdida': return 'color: #FF4B4B'
                elif val == 'Pendiente': return 'color: #FFD700'
                return ''
                
            st.dataframe(df_mostrar.style.map(color_estado, subset=['Estado']), hide_index=True, use_container_width=True)

        # ════════════════════════════════════════════════════════
        # PORTAFOLIO HISTÓRICO — siempre visible (sin toggle)
        # ════════════════════════════════════════════════════════
        st.divider()
        st.subheader("Vista Portafolio Histórico")
        import plotly.graph_objects as go

        # ══════════════════════════════════════════════════════
        # FUENTE 1: picks ya guardados en portafolio_historico
        # ══════════════════════════════════════════════════════
        df_db_hist = pd.read_sql("SELECT * FROM portafolio_historico", conn)
        fechas_en_db = set()
        big_portfolio_from_db = []
        if not df_db_hist.empty:
            df_db_hist['Date'] = df_db_hist['Date'].astype(str).str.strip().str.slice(0, 10)
            fechas_en_db = set(df_db_hist['Date'].unique())
            for _, r in df_db_hist.iterrows():
                big_portfolio_from_db.append({
                    'Date':    r['Date'],
                    'Home':    r['HomeTeam'],
                    'Away':    r['AwayTeam'],
                    'Mercado': r['Mercado'],
                    'Cuota':   r['Cuota'],
                    'Prob_IA': r.get('Prob_IA', 0.0),
                    'Edge':    r.get('Edge', 0.0),
                    'Stake':   r.get('Stake', 500),
                    'Estado':  r['Estado'],
                    'Beneficio_Neto': r.get('Beneficio_Neto', 0.0),
                    '_from_db': True,
                })

        # ══════════════════════════════════════════════════════
        # FUENTE 2: re-escanear CSVs de odds para dias SIN DB
        # ══════════════════════════════════════════════════════
        directorio_odds_hist = Path("odds_data")
        archivos_hist = list(directorio_odds_hist.glob("*.csv")) if directorio_odds_hist.exists() else []
        big_portfolio_from_csv = []

        if archivos_hist:
            bundle_clubes_hist = cargar_modelo()

            equipos_clubes_hist = pd.read_sql(
                "SELECT DISTINCT HomeTeam AS Team FROM historial_multiliga_ml "
                "UNION SELECT DISTINCT AwayTeam AS Team FROM historial_multiliga_ml",
                conn
            )['Team'].tolist()

            lista_dfs_hist = []
            for f_hist in archivos_hist:
                try:
                    # Los archivos de odds del Mundial no se consideran en el portafolio
                    if 'worldcup' in f_hist.name.lower():
                        continue

                    df_t = pd.read_csv(f_hist)
                    if df_t.empty:
                        continue
                    match_dash = re.search(r'(\d{4})-(\d{2})-(\d{2})', f_hist.name)
                    match_pure = re.search(r'(\d{4})(\d{2})(\d{2})', f_hist.name)
                    if match_dash:
                        fecha_fb = match_dash.group(0)
                    elif match_pure:
                        fecha_fb = f"{match_pure.group(1)}-{match_pure.group(2)}-{match_pure.group(3)}"
                    else:
                        import os as _os
                        fecha_fb = datetime.fromtimestamp(_os.path.getmtime(f_hist)).strftime('%Y-%m-%d')
                    df_t.columns = [c.lower() for c in df_t.columns]
                    if 'hometeam' in df_t.columns and 'home' not in df_t.columns:
                        df_t = df_t.rename(columns={'hometeam': 'home'})
                    if 'awayteam' in df_t.columns and 'away' not in df_t.columns:
                        df_t = df_t.rename(columns={'awayteam': 'away'})
                    if 'inicio_local' not in df_t.columns or df_t['inicio_local'].isna().all():
                        df_t['inicio_local'] = fecha_fb + " 12:00"
                    else:
                        df_t['inicio_local'] = df_t['inicio_local'].fillna(fecha_fb + " 12:00")
                    # Fecha_Match derivada de inicio_local (cada fila puede ser un dia distinto)
                    df_t['Fecha_Match'] = df_t['inicio_local'].astype(str).str.strip().str.slice(0, 10)
                    df_t = df_t[df_t['Fecha_Match'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]
                    # Incluir TODOS los dias del CSV; el loop de re-escaneo ya salta
                    # los dias cubiertos por la DB con "if fecha_d in fechas_en_db: continue"
                    if not df_t.empty:
                        lista_dfs_hist.append(df_t)
                except Exception:
                    continue

            if lista_dfs_hist:
                df_master_hist = pd.concat(lista_dfs_hist, ignore_index=True)
                df_master_hist = df_master_hist.drop_duplicates(subset=['home', 'away', 'inicio_local'])
                fechas_csv_nuevas = sorted(df_master_hist['Fecha_Match'].unique())
            else:
                df_master_hist    = pd.DataFrame()
                fechas_csv_nuevas = []
        else:
            df_master_hist    = pd.DataFrame()
            fechas_csv_nuevas = []

        # ══════════════════════════════════════════════════════
        # Combinacion de ambas fuentes
        # ══════════════════════════════════════════════════════
        n_dias_db  = len(fechas_en_db)
        n_dias_csv = len(fechas_csv_nuevas)

        if n_dias_db == 0 and n_dias_csv == 0:
            st.warning("No hay datos en portafolio_historico ni archivos de odds CSV. No hay historial que mostrar.")
        else:
            st.caption(
                f"Fuente: **{n_dias_db} dias desde la base de datos** "
                + (f"+ **{n_dias_csv} dias re-escaneados desde odds CSVs**" if n_dias_csv > 0 else "")
            )

            fechas_hist_todas = sorted(
                list(fechas_en_db) + [f for f in fechas_csv_nuevas if f not in fechas_en_db]
            )

                # ── Función reutilizada del escáner para evaluar un día ──
            def _escanear_dia_hist(fecha_str, df_odds_dia):
                prob_over = poisson_prob_over   # alias local — poisson_prob_over definida al top del módulo
                oportunidades = []
                for _, row in df_odds_dia.iterrows():
                    h_csv = str(row['home'])
                    a_csv = str(row['away'])
                    fecha_partido = str(row['inicio_local']).split()[0] if pd.notna(row.get('inicio_local')) else fecha_str

                    # El diccionario de alias manda: match exacto por alias primero,
                    # fuzzy sólo como respaldo para variantes que el diccionario no conoce.
                    h_alias_cl = normalizar_nombre(h_csv)
                    a_alias_cl = normalizar_nombre(a_csv)
                    h_match_cl = (h_alias_cl, 100) if equipos_clubes_hist and h_alias_cl in equipos_clubes_hist else (process.extractOne(h_csv, equipos_clubes_hist, scorer=fuzz.token_sort_ratio) if equipos_clubes_hist else None)
                    a_match_cl = (a_alias_cl, 100) if equipos_clubes_hist and a_alias_cl in equipos_clubes_hist else (process.extractOne(a_csv, equipos_clubes_hist, scorer=fuzz.token_sort_ratio) if equipos_clubes_hist else None)

                    cl_score = min(h_match_cl[1] if h_match_cl else 0, a_match_cl[1] if a_match_cl else 0)

                    if cl_score >= 80:
                        h_db, a_db = h_match_cl[0], a_match_cl[0]
                    else:
                        continue

                    try:
                        if not bundle_clubes_hist:
                            continue
                        model_clubes_hist    = bundle_clubes_hist['model']
                        feature_cols_hist    = bundle_clubes_hist['feature_cols']

                        stats_h_loc = get_recent_stats(h_db, conn)
                        stats_a_loc = get_recent_stats(a_db, conn)

                        forma_h = construir_forma_reciente(h_db, conn)
                        forma_a = construir_forma_reciente(a_db, conn)

                        pts_h = obtener_puntos_temporada(h_db, conn)
                        pts_a = obtener_puntos_temporada(a_db, conn)
                        dif_tabla = pts_h - pts_a
                        descanso_h = obtener_dias_descanso(h_db, conn)
                        descanso_a = obtener_dias_descanso(a_db, conn)
                        ventaja_fisica = descanso_h - descanso_a

                        fila_features = construir_fila_features(forma_h, forma_a, dif_tabla, ventaja_fisica)
                        X_pred = pd.DataFrame([fila_features])[feature_cols_hist]

                        pred_probs = model_clubes_hist.predict_proba(X_pred)[0]
                        prob_visita, prob_empate, prob_local = pred_probs[0], pred_probs[1], pred_probs[2]
                        pred_goles_home = (stats_h_loc['FTHG'] + stats_a_loc['FTAG']) / 2
                        pred_goles_away = (stats_a_loc['FTHG'] + stats_h_loc['FTAG']) / 2
                        prom_goles_total = pred_goles_home + pred_goles_away
                        prom_corners_total = (stats_h_loc['HC'] + stats_a_loc['AC']) / 2
                        prom_shots_total = (stats_h_loc['HST'] + stats_a_loc['AST']) / 2
                    except Exception:
                        continue

                    # Evaluar mercados (mismo criterio que el escáner en vivo)
                    def _buscar_cuota(r, cols):
                        for c in cols:
                            if c in r.index and pd.notna(r[c]) and str(r[c]).strip() != '':
                                return r[c]
                        return None

                    mercados_ev = [
                        ("Ganador (Local)",   _buscar_cuota(row, ['1x2_home']), prob_local),
                        ("Empate",            _buscar_cuota(row, ['1x2_draw']), prob_empate),
                        ("Ganador (Visita)",  _buscar_cuota(row, ['1x2_away']), prob_visita),
                    ]

                    def _prob_under_loc(prom, umb): return 1 - prob_over(prom, umb)
                    def _prob_hdp(pf, pc, linea):
                        pa = 0.0
                        for gf in range(15):
                            for gc in range(15):
                                if (gf + linea) > gc:
                                    pa += (math.exp(-pf)*(pf**gf)/math.factorial(gf)) * (math.exp(-pc)*(pc**gc)/math.factorial(gc))
                        return pa

                    for col_name, val in row.items():
                        if pd.isna(val) or str(val).strip() == '': continue
                        col_str = str(col_name).lower()
                        if col_str in ['es_mundial','liga','pais','partido_id','home','away','inicio_utc','inicio_local','fecha_match']: continue
                        try:
                            val_num = float(val)
                            if val_num <= 1.0: continue
                        except ValueError: continue
                        if 'btts' in col_str or 'ambos' in col_str:
                            prob_btts = (1 - math.exp(-pred_goles_home)) * (1 - math.exp(-pred_goles_away))
                            if 'yes' in col_str or 'si' in col_str: mercados_ev.append(("Ambos Anotan (Sí)", val_num, prob_btts))
                            elif 'no' in col_str: mercados_ev.append(("Ambos Anotan (No)", val_num, 1 - prob_btts))
                            continue
                        m_linea = re.search(r'(-?\d+(?:\.\d+)?)', col_str)
                        if not m_linea: continue
                        linea = float(m_linea.group(1))
                        # Ignorar líneas asiáticas (.25 / .75) en TODOS los mercados
                        if round(abs(linea) % 1, 2) in (0.25, 0.75): continue
                        if 'hdp' in col_str or 'handicap' in col_str:
                            if 'home' in col_str: mercados_ev.append((f"Hándicap Local ({linea:+})", val_num, _prob_hdp(pred_goles_home, pred_goles_away, linea)))
                            elif 'away' in col_str: mercados_ev.append((f"Hándicap Visita ({linea:+})", val_num, _prob_hdp(pred_goles_away, pred_goles_home, linea)))
                        elif 'corners' in col_str:
                            hc_v = stats_h_loc.get('HC', prom_corners_total)
                            ac_v = stats_a_loc.get('AC', prom_corners_total)
                            if 'home' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Córners Local (+{linea})", val_num, prob_over(hc_v, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Córners Local (-{linea})", val_num, _prob_under_loc(hc_v, linea)))
                            elif 'away' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Córners Visita (+{linea})", val_num, prob_over(ac_v, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Córners Visita (-{linea})", val_num, _prob_under_loc(ac_v, linea)))
                            elif 'total' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Córners Totales (+{linea})", val_num, prob_over(prom_corners_total, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Córners Totales (-{linea})", val_num, _prob_under_loc(prom_corners_total, linea)))
                        elif 'shots' in col_str:
                            hst_v = stats_h_loc.get('HST', prom_shots_total)
                            ast_v = stats_a_loc.get('AST', prom_shots_total)
                            if 'home' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Tiros Local (+{linea})", val_num, prob_over(hst_v, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Tiros Local (-{linea})", val_num, _prob_under_loc(hst_v, linea)))
                            elif 'away' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Tiros Visita (+{linea})", val_num, prob_over(ast_v, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Tiros Visita (-{linea})", val_num, _prob_under_loc(ast_v, linea)))
                            elif 'total' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Tiros a Puerta Totales (+{linea})", val_num, prob_over(prom_shots_total, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Tiros a Puerta Totales (-{linea})", val_num, _prob_under_loc(prom_shots_total, linea)))
                        elif 'goles' in col_str or 'total' in col_str:
                            if 'tt_home' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Goles Local (+{linea})", val_num, prob_over(pred_goles_home, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Goles Local (-{linea})", val_num, _prob_under_loc(pred_goles_home, linea)))
                            elif 'tt_away' in col_str:
                                if 'over' in col_str: mercados_ev.append((f"Goles Visita (+{linea})", val_num, prob_over(pred_goles_away, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Goles Visita (-{linea})", val_num, _prob_under_loc(pred_goles_away, linea)))
                            else:
                                if 'over' in col_str: mercados_ev.append((f"Goles Totales (+{linea})", val_num, prob_over(prom_goles_total, linea)))
                                elif 'under' in col_str: mercados_ev.append((f"Goles Totales (-{linea})", val_num, _prob_under_loc(prom_goles_total, linea)))

                    for nombre_mk, cuota_mk, prob_mk in mercados_ev:
                        if cuota_mk is None: continue
                        try:
                            cuota_f = float(cuota_mk)
                            edge = prob_mk - (1 / cuota_f)
                            # Recopilar todo hasta 30% de edge; el portfolio builder
                            # aplica sus propias fases (1/2/3) con los rangos correctos.
                            if 0.001 <= edge <= 0.30:
                                oportunidades.append({
                                    'Date': fecha_partido, 'Home': h_db, 'Away': a_db,
                                    'Mercado': nombre_mk, 'Cuota': cuota_f,
                                    'Prob_IA': prob_mk, 'Edge': edge,
                                })
                        except Exception:
                            pass
                return oportunidades

            # ── Construir el big portfolio ────────────────────
            # Dias DB: usar picks tal como estan guardados
            big_portfolio_rows = list(big_portfolio_from_db)
            fechas_csv_vacias  = []  # dias con CSV pero sin picks con edge

            # Dias CSV: re-escanear solo los dias que no estan en DB
            if not df_master_hist.empty and fechas_csv_nuevas:
                with st.spinner("Re-escaneando dias adicionales desde odds CSVs..."):
                    stake_unitario = 500
                    for fecha_d in fechas_csv_nuevas:
                        if fecha_d in fechas_en_db:
                            continue  # ya cubierto por DB
                        df_dia_odds = df_master_hist[df_master_hist['Fecha_Match'] == fecha_d]
                        try:
                            ops_dia = _escanear_dia_hist(fecha_d, df_dia_odds)
                        except Exception:
                            fechas_csv_vacias.append(fecha_d)
                            continue
                        if not ops_dia:
                            fechas_csv_vacias.append(fecha_d)
                            continue
                        df_dia_ops = pd.DataFrame(ops_dia).sort_values('Edge', ascending=False)
                        df_dia_ops['Partido'] = df_dia_ops['Home'] + " vs " + df_dia_ops['Away']
                        df_dia_ops = df_dia_ops.drop_duplicates(subset=['Partido', 'Mercado']).reset_index(drop=True)

                        _HIST_1X2 = {"Ganador (Local)", "Empate", "Ganador (Visita)"}
                        _HIST_BUCKETS = [
                            ('alto',  2.50, 9999.0),
                            ('medio', 1.90,  2.50),
                            ('bajo',  0.0,   1.90),
                        ]
                        seleccionados = []
                        usados_partidos = set()
                        sel_mercados = set()  # (Partido, Mercado)

                        def _hist_tiene_1x2(partido):
                            for s in seleccionados:
                                if s['Partido'] == partido and s['Mercado'] in _HIST_1X2:
                                    return True
                            return False

                        def _hist_add(r_dict):
                            partido = r_dict['Partido']
                            mercado = r_dict['Mercado']
                            if (partido, mercado) in sel_mercados:
                                return False
                            if mercado in _HIST_1X2 and _hist_tiene_1x2(partido):
                                return False
                            seleccionados.append(r_dict)
                            usados_partidos.add(partido)
                            sel_mercados.add((partido, mercado))
                            return True

                        # ── Fase 1 hist: estructura 1-3-3-3, edge estándar 2%–15% ──
                        df_std = df_dia_ops[
                            (df_dia_ops['Edge'] >= 0.02) & (df_dia_ops['Edge'] <= 0.15)
                        ].copy()

                        # Golden pick
                        for _, r in df_std.iterrows():
                            if r['Partido'] not in usados_partidos:
                                if _hist_add(r.to_dict()):
                                    break

                        # Buckets estándar
                        _hist_bucket_counts = [0, 0, 0]
                        for bi, (_, min_c, max_c) in enumerate(_HIST_BUCKETS):
                            cnt = 0
                            for _, r in df_std.iterrows():
                                if cnt >= 3: break
                                if r['Partido'] in usados_partidos: continue
                                if min_c <= r['Cuota'] < max_c:
                                    if _hist_add(r.to_dict()):
                                        cnt += 1
                            _hist_bucket_counts[bi] = cnt

                        # ── Fase 2 hist: ampliar edge hasta 30%, mantener 1-3-3-3 ──
                        _HIST_EDGE_PASOS = [(0.01, 0.20), (0.005, 0.25), (0.001, 0.30)]
                        for e_min, e_max in _HIST_EDGE_PASOS:
                            if all(c >= 3 for c in _hist_bucket_counts):
                                break
                            df_exp = df_dia_ops[
                                (df_dia_ops['Edge'] >= e_min) & (df_dia_ops['Edge'] <= e_max)
                            ].copy()
                            for bi, (_, min_c, max_c) in enumerate(_HIST_BUCKETS):
                                falt = max(0, 3 - _hist_bucket_counts[bi])
                                if not falt: continue
                                cnt = 0
                                for _, r in df_exp.iterrows():
                                    if cnt >= falt: break
                                    if r['Partido'] in usados_partidos: continue
                                    if min_c <= r['Cuota'] < max_c:
                                        if _hist_add(r.to_dict()):
                                            cnt += 1
                                _hist_bucket_counts[bi] += cnt

                        # ── Fase 3 hist: flexible — Paso A 1-3-3-3, Paso B cuota<2.0 ──
                        if len(seleccionados) < 10:
                            df_flex = df_dia_ops[df_dia_ops['Edge'] <= 0.30].copy()
                            df_flex['es_nuevo'] = (~df_flex['Partido'].isin(usados_partidos)).astype(int)
                            df_flex = df_flex.sort_values(['es_nuevo', 'Edge'], ascending=[False, False])

                            # Paso A: completar 1-3-3-3
                            n_golden_h = sum(1 for s in seleccionados if 'Golden' in str(s.get('Nivel', '')))
                            if n_golden_h == 0 and len(seleccionados) < 10:
                                for _, r in df_flex.iterrows():
                                    if r['Partido'] not in usados_partidos:
                                        if _hist_add(r.to_dict()):
                                            break
                            for bi, (_, min_c, max_c) in enumerate(_HIST_BUCKETS):
                                falt_bi = max(0, 3 - _hist_bucket_counts[bi])
                                if not falt_bi or len(seleccionados) >= 10: continue
                                cnt = 0
                                for _, r in df_flex.iterrows():
                                    if cnt >= falt_bi or len(seleccionados) >= 10: break
                                    if not (min_c <= r['Cuota'] < max_c): continue
                                    if _hist_add(r.to_dict()):
                                        cnt += 1
                                _hist_bucket_counts[bi] += cnt

                            # Paso B: slots restantes, sólo cuota < 2.0
                            if len(seleccionados) < 10:
                                df_bajo = df_flex[df_flex['Cuota'] < 2.0].copy()
                                for _, r in df_bajo.iterrows():
                                    if len(seleccionados) >= 10: break
                                    _hist_add(r.to_dict())

                        # Si el día tiene menos de 8 picks, flat stake de 500 por pick (igual que tab1)
                        stake_dia = 500 if len(seleccionados) < 8 else stake_unitario
                        for s in seleccionados:
                            s['Stake'] = stake_dia
                            s['_from_db'] = False
                            big_portfolio_rows.append(s)

            if not big_portfolio_rows:
                st.info("No se encontraron oportunidades históricas con edge en los archivos de odds disponibles.")
            else:
                df_big = pd.DataFrame(big_portfolio_rows)

                # ── Liquidar contra el historial real ───────────
                # Normalización via diccionario_alias centralizado
                def _resolver_nombre(nombre):
                    """Devuelve el nombre normalizado via ALIAS_GLOBAL (case-insensitive)."""
                    return normalizar_nombre(str(nombre))

                def _liquidar_pick_hist(pick_row):
                    fecha_d = str(pick_row['Date'])
                    try:
                        fd = datetime.strptime(fecha_d, '%Y-%m-%d')
                        f_ini = (fd - timedelta(days=2)).strftime('%Y-%m-%d')
                        f_fin = (fd + timedelta(days=2)).strftime('%Y-%m-%d')
                    except Exception:
                        f_ini = f_fin = fecha_d

                    q_res = f"""
                    SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST, NULL AS FueProrroga, NULL AS FTHG_r, NULL AS FTAG_r, NULL AS HC_r, NULL AS AC_r, 0 AS EsMundial FROM historial_multiliga_ml WHERE Date BETWEEN '{f_ini}' AND '{f_fin}'
                    UNION ALL
                    SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST, FueProrroga, FTHG_r, FTAG_r, HC_r, AC_r, 1 AS EsMundial FROM historial_selecciones_ml WHERE Date BETWEEN '{f_ini}' AND '{f_fin}'
                    """
                    try:
                        res_df = pd.read_sql(q_res, conn)
                    except Exception:
                        return 'Pendiente', 0.0

                    if res_df.empty:
                        return 'Pendiente', 0.0

                    pick_home = _resolver_nombre(str(pick_row['Home']))
                    pick_away = _resolver_nombre(str(pick_row['Away']))

                    # Normalizamos también el lado de la BD: el mismo equipo puede
                    # estar guardado con distinta grafía según temporada
                    # (ej. "Nott'm Forest" vs "Nottingham Forest").
                    res_df['_HomeNorm'] = res_df['HomeTeam'].apply(_resolver_nombre)
                    res_df['_AwayNorm'] = res_df['AwayTeam'].apply(_resolver_nombre)

                    # 1. Intento exacto (ambos lados normalizados via alias)
                    exact = res_df[
                        (res_df['_HomeNorm'] == pick_home) &
                        (res_df['_AwayNorm'] == pick_away)
                    ]
                    if exact.empty:
                        # También probar con los nombres originales sin alias
                        orig_home = str(pick_row['Home'])
                        orig_away = str(pick_row['Away'])
                        exact = res_df[
                            (res_df['HomeTeam'] == orig_home) &
                            (res_df['AwayTeam'] == orig_away)
                        ]

                    if not exact.empty:
                        row_real = exact.iloc[0]
                    else:
                        # 2. Fallback fuzzy sobre ambos equipos
                        row_real = None
                        best_score = 0
                        for _, candidate in res_df.iterrows():
                            for ch, ca in [
                                (candidate['HomeTeam'], candidate['AwayTeam']),
                                (candidate['AwayTeam'], candidate['HomeTeam']),
                            ]:
                                h_s = fuzz.token_set_ratio(pick_home, ch)
                                a_s = fuzz.token_set_ratio(pick_away, ca)
                                combined = (h_s + a_s) / 2
                                if combined > best_score and h_s >= 75 and a_s >= 75:
                                    best_score = combined
                                    row_real = candidate
                        if row_real is None:
                            return 'Pendiente', 0.0

                    hg = row_real['FTHG'] if pd.notna(row_real.get('FTHG')) else None
                    ag = row_real['FTAG'] if pd.notna(row_real.get('FTAG')) else None
                    if hg is None or ag is None:
                        return 'Pendiente', 0.0

                    hg, ag = int(hg), int(ag)

                    # Prórroga: si el partido fue a tiempo extra, se considera empate
                    # y se usa el resultado de tiempo reglamentario (FTHG_r/FTAG_r)
                    # para 1X2, Ambos Anotan, Hándicap y Over/Under de goles.
                    if pd.notna(row_real.get('FueProrroga')) and int(row_real['FueProrroga']) == 1:
                        hg_r, ag_r = row_real.get('FTHG_r'), row_real.get('FTAG_r')
                        if pd.notna(hg_r) and pd.notna(ag_r):
                            hg, ag = int(hg_r), int(ag_r)

                    hc  = int(row_real['HC'])  if pd.notna(row_real.get('HC'))  else 0
                    ac  = int(row_real['AC'])  if pd.notna(row_real.get('AC'))  else 0
                    hst = int(row_real['HST']) if pd.notna(row_real.get('HST')) else 0
                    ast = int(row_real['AST']) if pd.notna(row_real.get('AST')) else 0

                    # Córners:
                    # - Clubes: siempre HC/AC.
                    # - Mundial CON prórroga (FueProrroga = 1): se liquida con TIEMPO
                    #   REGLAMENTARIO (HC_r/AC_r), porque HC/AC incluye los córners de
                    #   la prórroga. Si todavía no cargaste HC_r/AC_r a mano para este
                    #   partido, el pick de córners se queda 'Pendiente'.
                    # - Mundial SIN prórroga (FueProrroga = 0): HC/AC ya es el dato
                    #   completo de tiempo reglamentario, igual que en clubes.
                    es_wc = bool(row_real.get('EsMundial'))
                    fue_prorroga = es_wc and pd.notna(row_real.get('FueProrroga')) and int(row_real['FueProrroga']) == 1

                    if fue_prorroga:
                        hc_r_raw, ac_r_raw = row_real.get('HC_r'), row_real.get('AC_r')
                        corners_disponible = pd.notna(hc_r_raw) and pd.notna(ac_r_raw)
                        if corners_disponible:
                            hc_corner, ac_corner = int(hc_r_raw), int(ac_r_raw)
                    else:
                        corners_disponible = True
                        hc_corner, ac_corner = hc, ac

                    mkt = pick_row['Mercado']
                    if "Córners" in mkt and not corners_disponible:
                        return 'Pendiente', 0.0

                    ganada = False
                    if mkt == "Ganador (Local)":      ganada = (hg > ag)
                    elif mkt == "Empate":              ganada = (hg == ag)
                    elif mkt == "Ganador (Visita)":    ganada = (ag > hg)
                    elif mkt == "Ambos Anotan (Sí)":   ganada = (hg > 0 and ag > 0)
                    elif mkt == "Ambos Anotan (No)":   ganada = (hg == 0 or ag == 0)
                    else:
                        m_liq = re.search(r'\(([+-]\d+\.5)\)', mkt)
                        if m_liq:
                            signo = m_liq.group(1)[0]
                            val_l = float(m_liq.group(1)[1:])
                            lin_m = float(m_liq.group(1))
                            if "Hándicap" in mkt:
                                if "Local"   in mkt: ganada = (hg + lin_m > ag)
                                elif "Visita" in mkt: ganada = (ag + lin_m > hg)
                            else:
                                score_liq = -1
                                if "Goles Local"    in mkt: score_liq = hg
                                elif "Goles Visita" in mkt: score_liq = ag
                                elif "Goles"        in mkt: score_liq = hg + ag
                                elif "Córners Local"    in mkt: score_liq = hc_corner
                                elif "Córners Visita"   in mkt: score_liq = ac_corner
                                elif "Córners"          in mkt: score_liq = hc_corner + ac_corner
                                elif "Tiros Local"      in mkt: score_liq = hst
                                elif "Tiros Visita"     in mkt: score_liq = ast
                                elif "Tiros"            in mkt: score_liq = hst + ast
                                if signo == '+': ganada = (score_liq > val_l)
                                elif signo == '-': ganada = (score_liq < val_l)

                    estado = 'Ganada' if ganada else 'Perdida'
                    # Beneficio computed with placeholder stake=1; real stake applied later
                    return estado, ganada

                # ── Liquidar big portfolio ──────────────────────
                # DB picks: si Estado != Pendiente usar el valor guardado;
                #           si sigue Pendiente, intentar re-liquidar contra historial.
                # CSV picks: liquidar siempre contra historial.
                estados_list = []
                ganada_list  = []
                for _, pr in df_big.iterrows():
                    if pr.get('_from_db', False) and pr.get('Estado', 'Pendiente') != 'Pendiente':
                        # Resultado ya resuelto en DB → usar directamente
                        est_db = pr['Estado']
                        gan_db = (est_db == 'Ganada')
                        estados_list.append(est_db)
                        ganada_list.append(gan_db)
                    else:
                        # Pendiente (DB o CSV) → intentar liquidar contra historial
                        est, gan = _liquidar_pick_hist(pr)
                        estados_list.append(est)
                        ganada_list.append(gan)

                df_big['Estado']  = estados_list
                df_big['_Ganada'] = ganada_list  # bool para recalculo de stake
                df_big['Date']    = pd.to_datetime(df_big['Date'], format='mixed').dt.date

                # ── Sub-toggle: bankroll dinámico ────────────
                _col_toggle, _col_info = st.columns([1, 2])
                with _col_toggle:
                    _modo_dyn = st.toggle(
                        "Bankroll Dinámico",
                        key="hist_bankroll_dyn",
                        help="ON: el stake de cada día se calcula sobre el bankroll inicial + P&L acumulado de días anteriores. OFF: flat staking uniforme."
                    )
                with _col_info:
                    _bk_base_input = st.number_input(
                        "Bankroll inicial ($)",
                        min_value=100, value=5000, step=500,
                        key="hist_bankroll_base",
                        label_visibility="visible"
                    )

                # ── Sub-toggle: stake por nivel de riesgo ──────
                _modo_riesgo_hist = st.toggle(
                    "Stake por Nivel de Riesgo",
                    key="hist_modo_stake_riesgo",
                    help="ON: Golden 1.5×, Bajo 1.2×, Medio 1.0×, Alto 0.5× de la unidad base. OFF: mismo stake para todos los picks."
                )

                # ── Selector de Temporada ───────────────────────
                # Cada temporada agrupa los partidos desde julio del año N
                # hasta junio del año N+1 (calendario estándar de las ligas
                # europeas). Ajusta _MES_INICIO_TEMPORADA si se prefiere otro
                # mes de corte.
                _MES_INICIO_TEMPORADA = 7  # 7 = julio

                def _etiqueta_temporada(f):
                    _anio_ini = f.year if f.month >= _MES_INICIO_TEMPORADA else f.year - 1
                    return f"{str(_anio_ini)[-2:]}/{str(_anio_ini + 1)[-2:]}"

                df_big['_Temporada'] = df_big['Date'].apply(_etiqueta_temporada)
                _temporadas_disponibles = sorted(df_big['_Temporada'].unique(), reverse=True)
                _opciones_temporada = [f"Temporada {t}" for t in _temporadas_disponibles] + ["Todo el Historial"]

                st.markdown("**Temporada**")
                _temporada_sel = st.selectbox(
                    "Elige la temporada a visualizar:",
                    _opciones_temporada,
                    index=0,
                    key="hist_temporada_sel",
                    help="Agrupa los picks por temporada futbolística (julio a junio), para poder ver cada una por separado. 'Todo el Historial' muestra el acumulado completo de todas las temporadas.",
                    label_visibility="visible",
                )

                # Filtrar df_big según la temporada elegida
                if _temporada_sel != "Todo el Historial":
                    _temporada_activa = _temporada_sel.replace("Temporada ", "")
                    df_big = df_big[df_big['_Temporada'] == _temporada_activa].copy()
                df_big = df_big.drop(columns=['_Temporada'])

                # ── Asignar stakes reales día a día ──────────
                # Días con <8 picks → flat stake fijo de $500 por pick (igual que tab1, independiente del bankroll)
                _STAKE_DIA_CORTO = 500.0
                _UMBRAL_PICKS_CORTO = 8

                def _mult_nivel_hist(nivel_str):
                    n = str(nivel_str).lower()
                    if 'golden' in n: return 1.5
                    if 'bajo'   in n: return 1.2
                    if 'medio'  in n: return 1.0
                    if 'alto'   in n: return 0.5
                    return 1.0

                # Ordenar días cronológicamente
                fechas_con_picks = sorted(df_big['Date'].unique())
                bankroll_actual  = float(_bk_base_input)
                pnl_acum_dyn     = 0.0

                # Mapa fecha → unidad_base_por_pick (recalculado si dinámico)
                stake_por_fecha = {}
                for _fd in fechas_con_picks:
                    _picks_fd = df_big[df_big['Date'] == _fd]
                    _n_picks  = len(_picks_fd)
                    if _n_picks == 0:
                        continue

                    # Días con menos de 8 picks → stake fijo de $500 (ignora bankroll/modo)
                    if _n_picks < _UMBRAL_PICKS_CORTO:
                        _stake_hoy = _STAKE_DIA_CORTO
                    elif _modo_riesgo_hist:
                        # Unidad base = bankroll / suma de multiplicadores
                        _mults_fd = _picks_fd['Nivel'].apply(_mult_nivel_hist) if 'Nivel' in _picks_fd.columns else pd.Series([1.0]*_n_picks)
                        _suma_mults = _mults_fd.sum()
                        _stake_hoy = (bankroll_actual if _modo_dyn else float(_bk_base_input)) / _suma_mults if _suma_mults > 0 else (bankroll_actual if _modo_dyn else float(_bk_base_input)) / _n_picks
                    elif _modo_dyn:
                        _stake_hoy = bankroll_actual / _n_picks
                    else:
                        _stake_hoy = float(_bk_base_input) / _n_picks

                    stake_por_fecha[_fd] = _stake_hoy

                    # Calcular P&L del día para actualizar bankroll (solo cerrados)
                    _cerrados_fd = _picks_fd[_picks_fd['Estado'].isin(['Ganada', 'Perdida'])]
                    for _, _pr in _cerrados_fd.iterrows():
                        if _n_picks < _UMBRAL_PICKS_CORTO:
                            _s_pick = _STAKE_DIA_CORTO
                        elif _modo_riesgo_hist and 'Nivel' in _pr.index:
                            _s_pick = _stake_hoy * _mult_nivel_hist(_pr.get('Nivel', ''))
                        else:
                            _s_pick = _stake_hoy
                        if _pr['_Ganada']:
                            _ben = _s_pick * _pr['Cuota'] - _s_pick
                        else:
                            _ben = -_s_pick
                        pnl_acum_dyn += _ben
                    if _modo_dyn:
                        bankroll_actual = float(_bk_base_input) + pnl_acum_dyn

                # Aplicar stakes y calcular beneficios reales
                def _calc_beneficio(row):
                    _fd   = row['Date']
                    _n_fd = len(df_big[df_big['Date'] == _fd])
                    _is_db = row.get('_from_db', False)
                    # Días cortos siempre usan stake fijo, incluso para picks de DB
                    if _n_fd < _UMBRAL_PICKS_CORTO:
                        _s = _STAKE_DIA_CORTO
                    elif not _modo_dyn and not _modo_riesgo_hist and _is_db:
                        # Flat mode sin riesgo + día completo + DB pick: usar stake almacenado
                        _s   = row.get('Stake', float(_bk_base_input) / 10)
                        _ben = row.get('Beneficio_Neto', 0.0)
                        return _s, _ben
                    else:
                        _unidad = stake_por_fecha.get(_fd, float(_bk_base_input) / 10)
                        if _modo_riesgo_hist and 'Nivel' in row.index:
                            _s = _unidad * _mult_nivel_hist(row.get('Nivel', ''))
                        else:
                            _s = _unidad
                    if row['Estado'] == 'Ganada':
                        return _s, _s * row['Cuota'] - _s
                    elif row['Estado'] == 'Perdida':
                        return _s, -_s
                    else:
                        return _s, 0.0

                df_big[['Stake', 'Beneficio_Neto']] = df_big.apply(
                    lambda r: pd.Series(_calc_beneficio(r)), axis=1
                )

                # ── Días con portafolio chico (≤3 partidos) quedan fuera del histórico ──
                # Cuando un día tuvo 3 partidos o menos en juego, sus picks quedan
                # muy pegados entre sí (mismo puñado de encuentros), así que se
                # excluyen del cálculo agregado del portafolio histórico para no
                # distorsionar win rate / yield / equity. Se muestran aparte más abajo.
                df_big['_PartidoKey']  = df_big['Home'].astype(str) + " vs " + df_big['Away'].astype(str)
                _partidos_por_dia_big  = df_big.groupby('Date')['_PartidoKey'].transform('nunique')
                df_big_dias_chicos     = df_big[_partidos_por_dia_big <= 3].drop(columns=['_PartidoKey']).copy()
                df_big = df_big[_partidos_por_dia_big > 3].drop(columns=['_PartidoKey']).reset_index(drop=True)

                if not df_big_dias_chicos.empty:
                    _n_dias_chicos = df_big_dias_chicos['Date'].nunique()
                    st.info(
                        f"🔎 Dejamos **{_n_dias_chicos} día(s)** fuera de las estadísticas del portafolio "
                        "histórico porque tuvieron 3 partidos o menos en juego: con tan poca cancha, los "
                        "picks de ese día quedan demasiado correlacionados entre sí y podrían inflar o "
                        "distorsionar los números agregados. Eso sí — igual son picks que respaldamos, "
                        "así que los dejamos a la vista por separado en vez de descartarlos."
                    )
                    with st.expander(f"📂 Ver {_n_dias_chicos} día(s) con portafolio chico"):
                        for _fecha_chica in sorted(df_big_dias_chicos['Date'].unique()):
                            _picks_chicos = df_big_dias_chicos[df_big_dias_chicos['Date'] == _fecha_chica]
                            st.markdown(f"**{_fecha_chica}** — {len(_picks_chicos)} pick(s)")
                            _df_chico_show = _picks_chicos[['Home', 'Away', 'Mercado', 'Cuota', 'Estado']].copy()
                            _df_chico_show.columns = ['Local', 'Visita', 'Mercado', 'Cuota', 'Estado']
                            _df_chico_show['Cuota'] = _df_chico_show['Cuota'].round(2)
                            st.dataframe(_df_chico_show, hide_index=True, use_container_width=True)

                # ── Separar cerrados vs pendientes ───────────
                df_big_cerrado  = df_big[df_big['Estado'].isin(['Ganada', 'Perdida'])].copy()
                df_big_pendiente = df_big[df_big['Estado'] == 'Pendiente'].copy()

                n_dias_total    = df_big['Date'].nunique()
                n_dias_cerrados = df_big_cerrado['Date'].nunique() if not df_big_cerrado.empty else 0
                n_dias_pend     = df_big_pendiente['Date'].nunique() if not df_big_pendiente.empty else 0

                # ── Aviso de cobertura ────────────────────────
                if n_dias_pend > 0:
                    st.caption(
                        f"{n_dias_total} días con odds — "
                        f"{n_dias_cerrados} liquidados ✅ | "
                        f"{n_dias_pend} pendientes (sin resultado en DB aún)"
                    )

                if df_big_cerrado.empty:
                    st.warning("Ningún pick histórico tiene resultado registrado en la base de datos todavía.")
                else:
                    # ── KPIs globales ─────────────────────────
                    total_picks  = len(df_big_cerrado)
                    ganadas_big  = (df_big_cerrado['Estado'] == 'Ganada').sum()
                    win_rate_big = ganadas_big / total_picks * 100
                    stake_total  = df_big_cerrado['Stake'].sum()
                    pnl_total    = df_big_cerrado['Beneficio_Neto'].sum()
                    yield_big    = (pnl_total / stake_total * 100) if stake_total > 0 else 0.0
                    _modo_lbl    = "Dinámico" if _modo_dyn else "Flat"

                    # ── KPI: KPIs y rendimiento ────
                    st.markdown("#### 📊 KPIs y rendimiento")
                    if True:
                        _df_luck = df_big_cerrado[
                            df_big_cerrado['Prob_IA'].notna() &
                            (df_big_cerrado['Prob_IA'] > 0) &
                            df_big_cerrado['Cuota'].notna()
                        ].copy()
                        if len(_df_luck) >= 5:
                            # P&L esperado = Σ stake × (prob_IA × cuota − 1)
                            _exp_pnl = (
                                _df_luck['Stake'] *
                                (_df_luck['Prob_IA'] * _df_luck['Cuota'] - 1)
                            ).sum()
                            # Varianza = Σ p(1−p)(stake × cuota)²
                            _var_total = (
                                _df_luck['Prob_IA'] * (1 - _df_luck['Prob_IA']) *
                                (_df_luck['Stake'] * _df_luck['Cuota']) ** 2
                            ).sum()
                            _std_dev = float(np.sqrt(_var_total)) if _var_total > 0 else 1.0
                            _z_score = (pnl_total - _exp_pnl) / _std_dev

                            if _z_score >= 2.5:
                                _z_color = "#e74c3c"; _z_icon = "🚨"
                                _z_label = "Señal fuerte de retiro parcial"
                                _z_advice = "P&L supera 2.5σ sobre lo esperado. Alta probabilidad de componente de suerte. Considera retirar 30–50% del exceso."
                            elif _z_score >= 2.0:
                                _z_color = "#f39c12"; _z_icon = "⚠️"
                                _z_label = "Sobrerendimiento significativo (95%)"
                                _z_advice = "Estadísticamente sobreperformando. Considera retirar ~30% del exceso sobre P&L esperado como protección ante reversión a la media."
                            elif _z_score >= 1.64:
                                _z_color = "#f1c40f"; _z_icon = "📈"
                                _z_label = "Sobrerendimiento notable (90%)"
                                _z_advice = "Por encima de lo esperado, aún dentro del rango plausible. Monitorea la tendencia."
                            elif _z_score >= 0:
                                _z_color = "#2ecc71"; _z_icon = "✅"
                                _z_label = "Performance normal"
                                _z_advice = "El rendimiento está en línea con el edge esperado. Sin señal de suerte estructural."
                            else:
                                _z_color = "#5dade2"; _z_icon = "📉"
                                _z_label = "Por debajo de lo esperado"
                                _z_advice = "P&L actual por debajo del esperado por edge. Puede ser varianza negativa transitoria."

                            _lc1, _lc2, _lc3 = st.columns(3)
                            with _lc1:
                                st.metric(
                                    "P&L Esperado (Edge)",
                                    f"${_exp_pnl:,.0f}",
                                    help="Σ stake × (prob_IA × cuota − 1) para todos los picks cerrados con Prob_IA disponible."
                                )
                            with _lc2:
                                st.metric(
                                    "Z-Score",
                                    f"{_z_score:.2f}σ",
                                    help="Desviación estándar entre P&L real y el esperado por edge. Z > 2.0 = sobrerendimiento al 95% de confianza."
                                )
                            with _lc3:
                                st.markdown(f"""
                                <div style="background:#1a1d27;border:1px solid {_z_color}55;border-radius:10px;
                                    padding:11px 10px;text-align:center;">
                                    <div style="font-size:1.3rem;">{_z_icon}</div>
                                    <div style="font-size:0.72rem;font-weight:700;color:{_z_color};
                                        margin:4px 0;line-height:1.3;">{_z_label}</div>
                                    <div style="font-size:0.65rem;color:#8892a4;line-height:1.4;">{_z_advice}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            if _z_score >= 2.0:
                                _exceso  = pnl_total - _exp_pnl
                                _retirar = _exceso * 0.30
                                st.markdown(f"""
                                <div style="background:#1a1a2e;border-left:3px solid {_z_color};
                                    border-radius:6px;padding:8px 12px;margin:6px 0 4px;">
                                    💡 <b>Acción sugerida:</b> retirar ~<b style="color:{_z_color}">${_retirar:,.0f}</b>
                                    (30% del exceso de <b>${_exceso:,.0f}</b> sobre el P&L esperado por edge).
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(f"Se necesitan al menos 5 picks cerrados con Prob_IA para calcular el Z-score ({len(_df_luck)} disponibles).")

                    # ── KPIs globales ─────────────────────────────────────
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Picks Cerrados", f"{total_picks:,}")
                    k2.metric("Win Rate",       f"{win_rate_big:.1f}%")
                    k3.metric("Yield (ROI)",    f"{yield_big:.2f}%", _modo_lbl)
                    _balance_total = float(_bk_base_input) + pnl_total
                    k4.metric("Balance",        f"${_balance_total:,.0f}", f"${pnl_total:+,.0f}")

                    st.divider()

                    # ── Construir tabla de días (todos, cerrados + pendientes) ──
                    # Primero los cerrados agrupados
                    _dias_cerr = (
                        df_big_cerrado.groupby('Date')
                        .agg(
                            Picks   =('Estado', 'count'),
                            Ganadas =('Estado', lambda x: (x == 'Ganada').sum()),
                            Perdidas=('Estado', lambda x: (x == 'Perdida').sum()),
                            Invertido=('Stake', 'sum'),
                            PnL_Dia =('Beneficio_Neto', 'sum'),
                        )
                        .reset_index()
                    )
                    _dias_cerr['Estado_Dia'] = 'cerrado'

                    # Luego los pendientes agrupados
                    if not df_big_pendiente.empty:
                        _dias_pend = (
                            df_big_pendiente.groupby('Date')
                            .agg(Picks=('Estado', 'count'))
                            .reset_index()
                        )
                        _dias_pend['Ganadas']    = 0
                        _dias_pend['Perdidas']   = 0
                        _dias_pend['Invertido']  = _dias_pend['Date'].map(
                            lambda d: stake_por_fecha.get(d, 0) * len(df_big_pendiente[df_big_pendiente['Date'] == d])
                        )
                        _dias_pend['PnL_Dia']    = float('nan')
                        _dias_pend['Estado_Dia'] = 'pendiente'
                        _dias_big = pd.concat([_dias_cerr, _dias_pend], ignore_index=True)
                    else:
                        _dias_big = _dias_cerr.copy()

                    _dias_big = _dias_big.sort_values('Date').reset_index(drop=True)

                    # ── Agregar días CSV sin picks (odds disponibles pero sin edge) ──
                    if fechas_csv_vacias:
                        _sp_rows = []
                        _dates_ya = set(_dias_big['Date'].astype(str))
                        for _fv in sorted(fechas_csv_vacias):
                            try:
                                _fv_date = pd.Timestamp(_fv).date()
                            except Exception:
                                continue
                            if str(_fv_date) in _dates_ya:
                                continue  # ya cubierto con picks reales
                            _sp_rows.append({
                                'Date': _fv_date, 'Picks': 0, 'Ganadas': 0,
                                'Perdidas': 0, 'Invertido': 0.0,
                                'PnL_Dia': float('nan'), 'Estado_Dia': 'sin_picks'
                            })
                        if _sp_rows:
                            _dias_big = pd.concat(
                                [_dias_big, pd.DataFrame(_sp_rows)],
                                ignore_index=True
                            ).sort_values('Date').reset_index(drop=True)

                    _dias_big['Win%'] = (_dias_big['Ganadas'] / _dias_big['Picks'].replace(0, np.nan) * 100).round(1)

                    # Equity acumulada: sólo días cerrados contribuyen al P&L; pendientes se muestran como NaN
                    _pnl_acum = 0.0
                    _equity_vals = []
                    for _, _dr in _dias_big.iterrows():
                        if _dr['Estado_Dia'] == 'cerrado':
                            _pnl_acum += _dr['PnL_Dia']
                            _equity_vals.append(_pnl_acum)
                        else:
                            _equity_vals.append(float('nan'))
                    _dias_big['PnL_Acum'] = _equity_vals

                    # ── Curva de equity ───────────────────────
                    _dias_cerr_plot = _dias_big[_dias_big['Estado_Dia'] == 'cerrado']
                    _col_bars = ['#2ecc71' if v >= 0 else '#e74c3c' for v in _dias_cerr_plot['PnL_Dia']]

                    fig_big = go.Figure()
                    fig_big.add_trace(go.Bar(
                        x=_dias_cerr_plot['Date'].astype(str),
                        y=_dias_cerr_plot['PnL_Dia'],
                        name='P&L Día',
                        marker_color=_col_bars,
                        opacity=0.7,
                        yaxis='y2'
                    ))
                    fig_big.add_trace(go.Scatter(
                        x=_dias_cerr_plot['Date'].astype(str),
                        y=_dias_cerr_plot['PnL_Acum'],
                        name='Equity Acumulada',
                        mode='lines+markers',
                        line=dict(color='#5dade2', width=2.5),
                        marker=dict(size=7),
                    ))
                    # Pendientes como puntos grises en el eje equity
                    _dias_pend_plot = _dias_big[_dias_big['Estado_Dia'] == 'pendiente']
                    if not _dias_pend_plot.empty:
                        fig_big.add_trace(go.Scatter(
                            x=_dias_pend_plot['Date'].astype(str),
                            y=[_pnl_acum] * len(_dias_pend_plot),
                            name='Pendiente',
                            mode='markers',
                            marker=dict(color='#f39c12', size=9, symbol='circle-open'),
                        ))
                    # Días con odds pero sin picks con edge — marcadores grises con X
                    _dias_sp_plot = _dias_big[_dias_big['Estado_Dia'] == 'sin_picks']
                    if not _dias_sp_plot.empty:
                        fig_big.add_trace(go.Scatter(
                            x=_dias_sp_plot['Date'].astype(str),
                            y=[_pnl_acum] * len(_dias_sp_plot),
                            name='Sin picks',
                            mode='markers',
                            marker=dict(color='#636e72', size=8, symbol='x'),
                        ))
                    fig_big.update_layout(
                        dragmode=False,
                        plot_bgcolor='#1e2129',
                        paper_bgcolor='#1e2129',
                        font=dict(color='#aab0c0'),
                        legend=dict(orientation='h', y=1.08),
                        margin=dict(t=20, b=30, l=0, r=0),
                        yaxis =dict(title='Equity Acum. ($)', fixedrange=True, gridcolor='#2c3050'),
                        yaxis2=dict(title='P&L Día ($)', overlaying='y', side='right', fixedrange=True),
                        xaxis =dict(fixedrange=True),
                    )
                    st.plotly_chart(fig_big, use_container_width=True, config=CONFIG_FIJA)

                    # ── Tabla resumen día a día (todos los días) ──
                    def _color_pnl(val):
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            if val > 0: return 'color: #2ecc71; font-weight: bold'
                            elif val < 0: return 'color: #e74c3c'
                        return ''

                    _dias_mostrar = _dias_big[['Date','Picks','Ganadas','Perdidas','Win%','Invertido','PnL_Dia','PnL_Acum','Estado_Dia']].copy()
                    _dias_mostrar.columns = ['Fecha','Picks','✅ Won','❌ Lost','Win %','Invertido ($)','P&L Día ($)','Equity Acum. ($)','Estado']
                    _dias_mostrar['Invertido ($)']    = _dias_mostrar['Invertido ($)'].map('${:,.0f}'.format)
                    _dias_mostrar['P&L Día ($)']      = _dias_mostrar['P&L Día ($)'].apply(lambda v: round(v, 0) if not (isinstance(v, float) and np.isnan(v)) else '⏳')
                    _dias_mostrar['Equity Acum. ($)'] = _dias_mostrar['Equity Acum. ($)'].apply(lambda v: round(v, 0) if not (isinstance(v, float) and np.isnan(v)) else '⏳')
                    _dias_mostrar['Estado'] = _dias_mostrar['Estado'].map({'cerrado': '✅', 'pendiente': '⏳', 'sin_picks': '⚪'})

                    st.dataframe(
                        _dias_mostrar.style.map(_color_pnl, subset=['P&L Día ($)', 'Equity Acum. ($)']),
                        hide_index=True,
                        use_container_width=True
                    )

                    # ── Acordeones por día (todos) ────────────
                    st.markdown("#### Detalle por Día")
                    for _, _d in _dias_big.iterrows():
                        _fecha_d = str(_d['Date'])
                        _es_sp   = (_d['Estado_Dia'] == 'sin_picks')
                        _es_pend = (_d['Estado_Dia'] == 'pendiente')
                        if _es_sp:
                            _icon_d  = "⚪"
                            _enc_pnl = "Sin picks con edge este día"
                        elif _es_pend:
                            _icon_d  = "⏳"
                            _enc_pnl = "P&L: ⏳"
                        else:
                            _pnl_d   = _d['PnL_Dia']
                            _icon_d  = "🟢" if _pnl_d >= 0 else "🔴"
                            _enc_pnl = f"P&L: ${_pnl_d:+,.0f}  |  Acum: ${_d['PnL_Acum']:+,.0f}"
                        with st.expander(
                            f"{_icon_d} {_fecha_d}  —  {int(_d['Picks'])} picks  |  "
                            f"Won: {int(_d['Ganadas'])}  Lost: {int(_d['Perdidas'])}  |  {_enc_pnl}"
                        ):
                            if _es_sp:
                                st.caption("Odds disponibles en CSV pero ningún partido superó el filtro de edge (2–15%). El modelo no encontró valor ese día.")
                            else:
                                _picks_d = df_big[df_big['Date'] == _d['Date']]
                                _df_exp = _picks_d[['Home','Away','Mercado','Cuota','Stake','Estado','Beneficio_Neto']].copy()
                                _df_exp.columns = ['Local','Visita','Mercado','Cuota','Stake','Estado','Beneficio']
                                _df_exp['Cuota']     = _df_exp['Cuota'].round(2)
                                _df_exp['Stake']     = _df_exp['Stake'].map('${:,.0f}'.format)
                                _df_exp['Beneficio'] = _df_exp['Beneficio'].round(0)
                                def _ce_big(v):
                                    if v == 'Ganada':    return 'color: #2ecc71; font-weight: bold'
                                    elif v == 'Perdida': return 'color: #e74c3c'
                                    elif v == 'Pendiente': return 'color: #f39c12'
                                    return ''
                                st.dataframe(_df_exp.style.map(_ce_big, subset=['Estado']), hide_index=True, use_container_width=True)

conn.close()