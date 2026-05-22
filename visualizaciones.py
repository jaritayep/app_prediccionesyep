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


def poisson_prob(lamba_val, k):
    """Calcula la probabilidad de que ocurran exactamente k eventos"""
    if lamba_val <= 0: return 0
    return (math.exp(-lamba_val) * (lamba_val**k)) / math.factorial(k)

def prob_over(promedio, umbral):
    """Calcula la probabilidad de que ocurra MÁS que el umbral"""
    if promedio <= 0: return 0.05
    prob_acumulada = 0
    # Sumamos las probabilidades de 0 hasta el umbral
    for k in range(int(umbral) + 1):
        prob_acumulada += poisson_prob(promedio, k)
    return 1 - prob_acumulada
 
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
    if not lista_db: return nombre_api
    mejor_match, score = process.extractOne(nombre_api.strip(), lista_db, scorer=fuzz.token_set_ratio)
    return mejor_match if score >72 else nombre_api

def cargar_modelo():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

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

st.sidebar.title("⚽ Menú Principal")
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "BetBuilder Simulator", "Portafolio de Picks"])
st.sidebar.markdown("---")

if menu == "Análisis del Día":
    try:
        # Carga de datos inicial
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)

        # 1. Normalizar fechas y filtrar para mostrar solo HOY y el FUTURO
        df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.tz_localize(None).dt.normalize()

        # Obtenemos la fecha actual
        hoy = pd.Timestamp.now().normalize()
        df_jornada = df_jornada[df_jornada['Date'] >= hoy]

        if not df_jornada.empty:
            df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m')

            # 2. Selección de fecha y partido en la sidebar
            opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))
            dia_sel_str = st.sidebar.selectbox("📅 Seleccionar Día:", opciones_fecha)

            # Filtrar partidos del día seleccionado
            partidos_dia = df_jornada[df_jornada['Fecha_Display'] == dia_sel_str]
            partido_texto = st.sidebar.selectbox("🏟️ Partido:", partidos_dia['Local'] + " vs " + partidos_dia['Visita'])

            # Separar y corregir nombres
            home_raw, away_raw = partido_texto.split(" vs ")
            home_team = corregir_nombre_equipo(home_raw, equipos_db)
            away_team = corregir_nombre_equipo(away_raw, equipos_db)

            # --- RENDERIZADO DEL DASHBOARD ---
            st.title(f"{home_team} vs {away_team}")
            st.caption(f"📅 {dia_sel_str}")

            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.subheader("📊 Historial H2H")
                q_h2h = f'SELECT Date, HomeTeam as L, AwayTeam as V, FTHG as [GL], FTAG as [GV], FTR as R FROM historial_multiliga_ml WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") ORDER BY Date DESC LIMIT 5'
                df_h2h = pd.read_sql(q_h2h, conn)
                if not df_h2h.empty:
                    df_h2h['Date'] = pd.to_datetime(df_h2h['Date']).dt.strftime('%d/%m/%y')
                    st.dataframe(df_h2h, use_container_width=True, hide_index=True)

                st.subheader("📈 Tendencia de Goles")
                q_trend = f'SELECT FTHG as [Local], FTAG as [Visita] FROM historial_multiliga_ml WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" ORDER BY Date DESC LIMIT 10'
                st.line_chart(pd.read_sql(q_trend, conn).iloc[::-1])

            with col2:
                st.subheader("IA Predictiva")
                model = cargar_modelo()
                if model:
                    stats_h, stats_a = get_recent_stats(home_team, conn), get_recent_stats(away_team, conn)

                    # --- NUEVA PREPARACIÓN DE DATOS PARA LA IA (14 Variables) ---
                    # 1. Extraer xG (Uso .get() para evitar caídas si la DB devuelve nulo)
                    xg_h = stats_h.get('xG_home', 1.0) 
                    xg_a = stats_a.get('xG_away', 1.0)
                    xg_diff = xg_h - xg_a
                    pts_h = obtener_puntos_temporada(home_team, conn)
                    pts_a = obtener_puntos_temporada(away_team, conn)
                    dif_tabla = pts_h - pts_a
                    descanso_h = obtener_dias_descanso(home_team, conn)
                    descanso_a = obtener_dias_descanso(away_team, conn)
                    ventaja_fisica = descanso_h - descanso_a
                    
                    # 2. Calcular Eficiencia
                    eff_h = stats_h['FTHG'] / (xg_h + 0.01)
                    eff_a = stats_a['FTAG'] / (xg_a + 0.01)

                    # 3. Construir el array con el orden EXACTO de las 14 variables
                    input_data = [[
                        stats_h['FTHG'], stats_h['FTAG'], 
                        stats_h['HS'], stats_h['AS'], 
                        stats_h['HST'], stats_h['AST'], 
                        stats_h['HC'], stats_h['AC'], 
                        stats_h['HY'], stats_h['AY'],
                        xg_h,            # Variable 11 (xG Local)
                        xg_a,            # Variable 12 (xG Visita)
                        eff_h,           # Variable 13 (Eficiencia)
                        xg_diff,         # Variable 14 (Dominio)
                        dif_tabla,       # Variable 15 (Tabla)
                        ventaja_fisica   # Variable 16 (Fatiga)
                    ]]
                    
                    # Generar predicción
                    prob_ia = model.predict_proba(input_data)[0]

                    # Gráfico de Torta (Probabilidades)
                    fig_pie = px.pie(values=[prob_ia[2], prob_ia[1], prob_ia[0]], names=['Local', 'Empate', 'Visita'], color=['Local', 'Empate', 'Visita'], color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'}, hole=0.45)
                    fig_pie.update_layout(dragmode=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)

                    # --- LÓGICA DE PREDICCIÓN DE GOLES ---
                    pred_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
                    pred_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2
                    promedio_goles = pred_home + pred_away
                    prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))

                    # Métricas de Goles Actualizadas
                    c1, c2 = st.columns(2)
                    c1.metric("Goles Exp. (xG Total)", f"{(xg_h + xg_a):.2f}")
                    c2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                    st.progress(prob_over)

                    # Predicción Individual por Equipo
                    st.markdown("---")
                    cp_g1, cp_g2 = st.columns(2)
                    cp_g1.metric(f"Goles {home_team[:10]}", f"{pred_home:.2f}")
                    cp_g2.metric(f"Goles {away_team[:10]}", f"{pred_away:.2f}")
                    st.markdown("---")

                    st.markdown("#### **Tiros y Córners**")
                    cp1, cp2 = st.columns(2)
                    with cp1: st.write(f"Tiros: **{stats_h['HST']:.1f}** | **{stats_a['AST']:.1f}**")
                    with cp2: st.write(f"Córners: **{stats_h['HC']:.1f}** | **{stats_a['AC']:.1f}**")

            st.divider()
            st.subheader("🟨 Disciplina y Tarjetas")
            cd1, cd2 = st.columns(2)

            with cd1:
                st.markdown("#### **Media Amarillas**")
                m1, m2 = st.columns(2)
                m1.metric(f"{home_team[:12]}", f"{stats_h['HY']:.1f}")
                m2.metric(f"{away_team[:12]}", f"{stats_a['AY']:.1f}")

            with cd2:
                q_cards = f'SELECT Date, (HY + AY) as Total FROM historial_multiliga_ml WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") ORDER BY Date DESC LIMIT 5'
                df_cards = pd.read_sql(q_cards, conn)
                if not df_cards.empty:
                    fig_cards = px.bar(df_cards, x='Date', y='Total', color_discrete_sequence=['#f1c40f'])
                    fig_cards.update_layout(dragmode=False, xaxis={'fixedrange': True}, yaxis={'fixedrange': True})
                    st.plotly_chart(fig_cards, use_container_width=True, config=CONFIG_FIJA)
                    
        else:
            st.info("No hay partidos programados para hoy o los próximos días.")

    except Exception as e:
        st.error(f"Error al cargar dashboard: {e}")

elif menu == "Auditoría (Resultados)":
    st.title("🎯 Auditoría de Precisión (Flexible)")
    st.markdown("Audita las proyecciones de la IA incluyendo márgenes de error (⚠️) para resultados cercanos.")
 
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        fecha_audit = st.date_input("Selecciona fecha para auditar:",
                                    datetime.now() - timedelta(days=1))
 
    fecha_str = fecha_audit.strftime('%Y-%m-%d')

    query = "SELECT * FROM historial_multiliga_ml WHERE Date LIKE ?"
    df_reales = pd.read_sql(query, conn, params=(f"{fecha_str}%",))

    if df_reales.empty:
        st.warning(f"⚠️ No hay resultados en la base de datos para el {fecha_audit.strftime('%d/%m/%Y')}.")
    else:
        st.subheader(f"📊 Resumen de Jornada: {fecha_audit.strftime('%d/%m/%Y')}")

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
        cumplidas = 0
        casi_cumplidas = 0
        resultados_procesados = []

        with st.spinner('Calculando precisión contra proyecciones IA...'):
            for _, r in df_reales.iterrows():
                sh = get_recent_stats(r['HomeTeam'], conn)
                sa = get_recent_stats(r['AwayTeam'], conn)
 
                if sh is not None and sa is not None and len(sh) > 0 and len(sa) > 0:
                    # 1. Proyecciones (Ahora incluyendo equipos por separado)
                    proj_goles_total = (sh['FTHG'] + sh['FTAG'] + sa['FTHG'] + sa['FTAG']) / 2
 
                    # Lógica H2H: Ataque Local vs Defensa Visita / Ataque Visita vs Defensa Local
                    proj_goles_home = (sh['FTHG'] + sa['FTAG']) / 2
                    proj_goles_away = (sa['FTHG'] + sh['FTAG']) / 2
 
                    proj_corners = sh['HC'] + sa['AC']
                    proj_tiros = sh['HST'] + sa['AST']
                    proj_amarillas = sh['HY'] + sa['AY']
 
                    # 2. Resultados Reales
                    real_goles_home = r['FTHG']
                    real_goles_away = r['FTAG']
                    real_goles_total = real_goles_home + real_goles_away
                    real_corners = r['HC'] + r['AC']
                    real_tiros = r['HST'] + r['AST']
                    real_amarillas = r['HY'] + r['AY']
 
                    # 3. Verificación Global Superior (Solo medimos goles totales y corners para el resumen)
                    if real_goles_total >= proj_goles_total: cumplidas += 1
                    elif (proj_goles_total - real_goles_total) <= 0.5: casi_cumplidas += 1
 
                    if real_corners >= proj_corners: cumplidas += 1
                    elif (proj_corners - real_corners) <= 1.5: casi_cumplidas += 1
 
                    total_predicciones += 2
 
                    # Guardamos todas las métricas con su respectivo "Margen de Error"
                    resultados_procesados.append({
                        'fila': r,
                        'stats': [
                            # Formato: (Nombre, Proyectado, Real, Margen de Tolerancia)
                            ("Goles Total", proj_goles_total, real_goles_total, 0.5),
                            (f"Goles {r['HomeTeam']}", proj_goles_home, real_goles_home, 0.5),
                            (f"Goles {r['AwayTeam']}", proj_goles_away, real_goles_away, 0.5),
                            ("Córners Total", proj_corners, real_corners, 1.5),
                            ("Tiros al Arco", proj_tiros, real_tiros, 1.5),
                            ("Amarillas", proj_amarillas, real_amarillas, 1.0)
                        ]
                    })

        # --- MÉTRICAS SUPERIORES ---
        if total_predicciones > 0:
            tasa_exacta = cumplidas / total_predicciones
            tasa_flexible = (cumplidas + casi_cumplidas) / total_predicciones
 
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Tasa Verde (Exacta)", f"{tasa_exacta:.1%}", f"{cumplidas} de {total_predicciones}")
            with col_m2:
                st.metric("Tasa Amarilla (Casi)", f"{(casi_cumplidas / total_predicciones):.1%}", f"{casi_cumplidas} en el margen", delta_color="off")
            with col_m3:
                st.metric("Eficacia Flexible", f"{tasa_flexible:.1%}", "Verdes + Amarillos")
 
            st.divider()

            # --- ACORDEONES POR PARTIDO ---
            for res in resultados_procesados:
                r = res['fila']
                titulo = f"🏟️ {r['HomeTeam']} {int(r['FTHG'])} - {int(r['FTAG'])} {r['AwayTeam']}"
 
                with st.expander(titulo):
                    cols = st.columns(2)
                    for i, (label, p, re, margen) in enumerate(res['stats']):
                        real_val = re if pd.notnull(re) else 0
 
                        # Llamamos a nuestra nueva función de colores
                        check, color = evaluar_precision(real_val, p, margen)
 
                        # Alternamos columnas
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
elif menu == "BetBuilder Simulator":
    st.title("🛠️ BetBuilder Simulator")

    try:
        # 1. Cargar y normalizar datos de la jornada
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.tz_localize(None).dt.normalize()

        # Filtrar solo HOY y FUTURO
        hoy = pd.Timestamp.now().normalize()
        df_jornada = df_jornada[df_jornada['Date'] >= hoy]

        if df_jornada.empty:
            st.info("📅 No hay partidos programados para los próximos días.")
        else:
            # --- PASO 1: SELECCIONAR EL DÍA ---
            df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m')
            opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))

            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                dia_sel_str = st.selectbox("📅 Seleccionar Día:", opciones_fecha)

            # --- PASO 2: SELECCIONAR EL PARTIDO (Filtrado por el día elegido) ---
            partidos_del_dia = df_jornada[df_jornada['Fecha_Display'] == dia_sel_str]
            with c_sel2:
                partido_sel = st.selectbox(
                    "🏟️ Seleccionar Partido:",
                    partidos_del_dia['Local'] + " vs " + partidos_del_dia['Visita']
                )

            # Extraer y corregir nombres para la base de datos
            home_raw, away_raw = partido_sel.split(" vs ")
            home_team = corregir_nombre_equipo(home_raw, equipos_db)
            away_team = corregir_nombre_equipo(away_raw, equipos_db)

            # 3. CARGAR ESTADÍSTICAS Y CÁLCULOS
            stats_h = get_recent_stats(home_team, conn)
            stats_a = get_recent_stats(away_team, conn)

            # Predicción tradicional y extracción de xG (premium)
            pred_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
            pred_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2
            
            xg_h = stats_h.get('xG_home', pred_home)
            xg_a = stats_a.get('xG_away', pred_away)

            st.divider()
            col_config, col_ticket = st.columns([1.2, 1])

            with col_config:
                st.subheader("🎯 Configurar Mercados")

                # --- NUEVA LISTA DE MERCADOS EXPANDIDA ---
                mercado = st.selectbox("Seleccionar Mercado:", [
                    "Goles Totales", "Goles por Equipo", "BTTS (Ambos Anotan)", 
                    "Hándicap Asiático", "Córners Totales", "Córners por Equipo", 
                    "Tiros a Puerta Totales", "Tiros a Puerta por Equipo", "Doble Oportunidad"
                ])

                with st.container(border=True):
                    
                    if mercado == "Goles Totales":
                        l_g = st.selectbox("Línea:", [0.5, 1.5, 2.5, 3.5, 4.5], index=2)
                        t_g = st.radio("Predicción:", ["Over", "Under"], horizontal=True)
                        prom = pred_home + pred_away
                        prob = 1 / (1 + np.exp(-(prom - l_g))) if t_g == "Over" else 1 - (1 / (1 + np.exp(-(prom - l_g))))
                        desc_pick = f"{t_g} {l_g} Goles Totales"

                    elif mercado == "Goles por Equipo":
                        eq_sel = st.radio("Equipo:", [home_team, away_team], horizontal=True)
                        l_ge = st.selectbox("Línea de Goles:", [0.5, 1.5, 2.5], index=0)
                        t_ge = st.radio("Predicción:", ["Over", "Under"], horizontal=True, key="ge_t_bb")
                        val_p = pred_home if eq_sel == home_team else pred_away
                        prob = 1 / (1 + np.exp(-(val_p - l_ge))) if t_ge == "Over" else 1 - (1 / (1 + np.exp(-(val_p - l_ge))))
                        desc_pick = f"{eq_sel[:10]} {t_ge} {l_ge} Goles"
                        
                    elif mercado == "BTTS (Ambos Anotan)":
                        t_btts = st.radio("Predicción:", ["Sí", "No"], horizontal=True)
                        # Probabilidad Poisson de que marquen > 0 goles
                        prob_yes = (1 - np.exp(-xg_h)) * (1 - np.exp(-xg_a))
                        prob = prob_yes if t_btts == "Sí" else (1 - prob_yes)
                        desc_pick = f"Ambos Anotan: {t_btts}"
                        
                    elif mercado == "Hándicap Asiático":
                        eq_sel = st.radio("Equipo con Ventaja:", [home_team, away_team], horizontal=True)
                        l_hc = st.selectbox("Línea de Hándicap:", ["+1.5", "+2.5"])
                        # Calculamos la diferencia de dominio y le sumamos la ventaja virtual
                        dif_esperada = (xg_h - xg_a) if eq_sel == home_team else (xg_a - xg_h)
                        valor_hc = float(l_hc)
                        prob = 1 / (1 + np.exp(-(dif_esperada + valor_hc)))
                        desc_pick = f"Hándicap Asiático {eq_sel[:10]} {l_hc}"

                    elif mercado == "Córners Totales":
                        l_c = st.slider("Línea Córners Totales:", 5.5, 14.5, 8.5, 1.0)
                        t_c = st.radio("Predicción:", ["Over", "Under"], horizontal=True)
                        prom_c = stats_h['HC'] + stats_a['AC']
                        prob = 1 / (1 + np.exp(-(prom_c - l_c))) if t_c == "Over" else 1 - (1 / (1 + np.exp(-(prom_c - l_c))))
                        desc_pick = f"{t_c} {l_c} Córners Totales"
                        
                    elif mercado == "Córners por Equipo":
                        eq_sel = st.radio("Equipo:", [home_team, away_team], horizontal=True)
                        l_ce = st.slider("Línea Córners Equipo:", 2.5, 9.5, 4.5, 1.0)
                        t_ce = st.radio("Predicción:", ["Over", "Under"], horizontal=True)
                        prom_ce = stats_h['HC'] if eq_sel == home_team else stats_a['AC']
                        prob = 1 / (1 + np.exp(-(prom_ce - l_ce))) if t_ce == "Over" else 1 - (1 / (1 + np.exp(-(prom_ce - l_ce))))
                        desc_pick = f"{eq_sel[:10]} {t_ce} {l_ce} Córners"

                    elif mercado == "Doble Oportunidad":
                        opts = st.multiselect("Opciones:", ["Local", "Empate", "Visita"], default=["Local", "Empate"])
                        prob = len(opts) * 0.32 # Probabilidad base estimada
                        desc_pick = " o ".join(opts)

                    elif mercado == "Tiros a Puerta Totales":
                        l_t = st.number_input("Mínimo Tiros Totales:", 4, 20, 8)
                        prom_t = stats_h['HST'] + stats_a['AST']
                        prob = 1 / (1 + np.exp(-(prom_t - l_t)))
                        desc_pick = f"Más de {l_t} Tiros a Puerta Totales"
                        
                    elif mercado == "Tiros a Puerta por Equipo":
                        eq_sel = st.radio("Equipo:", [home_team, away_team], horizontal=True)
                        l_te = st.slider("Mínimo Tiros a Puerta:", 1, 10, 4)
                        prom_te = stats_h['HST'] if eq_sel == home_team else stats_a['AST']
                        prob = 1 / (1 + np.exp(-(prom_te - l_te)))
                        desc_pick = f"{eq_sel[:10]} Más de {l_te} Tiros a Puerta"

                # Lógica del Ticket
                if "ticket" not in st.session_state: st.session_state.ticket = []

                if st.button("➕ Añadir al Ticket"):
                    st.session_state.ticket.append({"desc": desc_pick, "prob": prob})
                    st.toast(f"Añadido: {desc_pick}")

            with col_ticket:
                st.subheader("📋 Tu Apuesta Combinada")
                if not st.session_state.ticket:
                    st.info("Añade mercados para ver la cuota final.")
                else:
                    p_final = 1.0
                    for i, item in enumerate(st.session_state.ticket):
                        c1, c2 = st.columns([3, 1])
                        c1.write(f"🔹 {item['desc']}")
                        c2.write(f"**{item['prob']:.0%}**")
                        p_final *= item['prob']

                    st.divider()
                    cuota = 1 / p_final if p_final > 0 else 100
                    st.metric("Probabilidad Total", f"{p_final:.1%}")
                    st.metric("Cuota Justa Calculada", f"{cuota:.2f}")

                    if st.button("🗑️ Limpiar Ticket"):
                        st.session_state.ticket = []
                        st.rerun()

    except Exception as e:
        st.error(f"Error en el Simulador: {e}")
elif menu == "Portafolio de Picks":
    st.title("📈 Portafolio de Inversión (Flat Staking)")
    
    API_KEY = "3ec28dbd498ab9985e9792b3f50a8902" # <--- Tu llave
    
    # 1. Crear tabla de historial si no existe
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

    # Tabs para organizar la UI
    tab1, tab2 = st.tabs(["🔍 Escáner en Vivo", "🏦 Rendimiento Histórico"])

    with tab1:
        st.markdown("### 🔍 Escáner de Ineficiencias vs Pinnacle")
        st.caption("Cruzando modelo ML, Poisson (Goles, Córners, Tiros) contra líneas de Pinnacle. (Edge 2% - 15%)")
        
        try:
            # 1. Cargamos tu Base de Datos de Equipos
            equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
            
            # --- AUTOMATIZACIÓN DE CARPETA Y CONSOLIDACIÓN DE FECHAS ---
            directorio_odds = Path("odds_data")
            archivos_csv = list(directorio_odds.glob("pinnacle_*.csv")) if directorio_odds.exists() else []
            
            df_master_odds = pd.DataFrame()
            fechas_disponibles = []

            if archivos_csv:
                lista_dfs = []
                for f in archivos_csv:
                    try:
                        df_temp = pd.read_csv(f)
                        # Nos aseguramos de que el archivo tenga la columna de tiempo
                        if 'inicio_local' in df_temp.columns:
                            lista_dfs.append(df_temp)
                    except Exception:
                        pass
                
                if lista_dfs:
                    # Combinamos todos y eliminamos partidos repetidos
                    df_master_odds = pd.concat(lista_dfs, ignore_index=True)
                    df_master_odds = df_master_odds.drop_duplicates(subset=['home', 'away', 'inicio_local'])
                    
                    # Extraemos la fecha limpia borrando la hora del texto
                    df_master_odds['Fecha_Match'] = df_master_odds['inicio_local'].astype(str).str.split().str[0]
                    
                    # Obtenemos los días únicos ordenados
                    fechas_disponibles = sorted(df_master_odds['Fecha_Match'].unique())

            c1, c2 = st.columns(2)
            with c1:
                # 🎯 CAMBIO 3: Ahora pides el total del portafolio, no por pick
                inversion_total = st.number_input("💰 Inversión TOTAL Portafolio ($)", min_value=1000, value=50000, step=500)
            with c2:
                if fechas_disponibles:
                    fecha_seleccionada = st.selectbox("📅 Seleccionar Día del Portafolio:", fechas_disponibles)
                else:
                    st.error("⚠️ No se encontraron partidos en la carpeta 'odds_data/'. ¡Corre el scraper primero!")
                    fecha_seleccionada = None

            boton_disabled = fecha_seleccionada is None

            if st.button("🔍 Escanear Mercado", type="primary", disabled=boton_disabled):
                model = cargar_modelo()
                if not model:
                    st.error("⚠️ No se encontró el archivo 'modelo_ia.pkl'.")
                    st.stop()

                with st.spinner(f"Analizando los partidos del {fecha_seleccionada} con Inteligencia Artificial..."):
                    # 🎯 FILTRADO AUTOMÁTICO: Extraemos solo las cuotas del día seleccionado
                    df_pinnacle = df_master_odds[df_master_odds['Fecha_Match'] == fecha_seleccionada]
                    
                    oportunidades = []
                    log_debug = []
                    
                    def buscar_cuota_segura(row, posibles_columnas):
                        for col in posibles_columnas:
                            if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != '':
                                return row[col]
                        return None

                    def prob_under(promedio, umbral):
                        return 1 - prob_over(promedio, umbral)
                        
                    def prob_handicap(prom_favor, prom_contra, linea_hdp):
                        # Motor Skellam (Poisson Doble) para Hándicap Asiático
                        # Calcula la probabilidad de que (Goles a Favor + Ventaja Hándicap) > Goles en Contra
                        prob_acum = 0.0
                        for gf in range(15):
                            for gc in range(15):
                                if (gf + linea_hdp) > gc:
                                    p_gf = (math.exp(-prom_favor) * (prom_favor**gf)) / math.factorial(gf)
                                    p_gc = (math.exp(-prom_contra) * (prom_contra**gc)) / math.factorial(gc)
                                    prob_acum += (p_gf * p_gc)
                        return prob_acum

                    for index, row in df_pinnacle.iterrows():
                        h_csv = str(row['home'])
                        a_csv = str(row['away'])
                        fecha_partido = str(row['inicio_local']).split()[0] if pd.notna(row['inicio_local']) else str(pd.Timestamp.now().date())
                        
                        h_db_match = process.extractOne(h_csv, equipos_db)
                        a_db_match = process.extractOne(a_csv, equipos_db)
                        
                        if not h_db_match or not a_db_match or h_db_match[1] < 80 or a_db_match[1] < 80:
                            continue
                            
                        h_db = h_db_match[0]
                        a_db = a_db_match[0]

                        # --- CÁLCULO DE VARIABLES DESDE TU DB ---
                        stats_h = get_recent_stats(h_db, conn)
                        stats_a = get_recent_stats(a_db, conn)
                        
                        xg_h = stats_h.get('xG_home', 1.0) 
                        xg_a = stats_a.get('xG_away', 1.0)
                        xg_diff = xg_h - xg_a
                        pts_h = obtener_puntos_temporada(h_db, conn)
                        pts_a = obtener_puntos_temporada(a_db, conn)
                        dif_tabla = pts_h - pts_a
                        descanso_h = obtener_dias_descanso(h_db, conn)
                        descanso_a = obtener_dias_descanso(a_db, conn)
                        ventaja_fisica = descanso_h - descanso_a
                        eff_h = stats_h['FTHG'] / (xg_h + 0.01)
                        eff_a = stats_a['FTAG'] / (xg_a + 0.01)

                        input_data = [[
                            stats_h['FTHG'], stats_h['FTAG'], stats_h['HS'], stats_h['AS'], 
                            stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'], 
                            stats_h['HY'], stats_h['AY'], xg_h, xg_a, eff_h, xg_diff, 
                            dif_tabla, ventaja_fisica
                        ]]
                        
                        pred_probs = model.predict_proba(input_data)[0]
                        prob_visita, prob_empate, prob_local = pred_probs[0], pred_probs[1], pred_probs[2]

                        # Promedios Base
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

                        # --- ESCÁNER DINÁMICO TOTAL (Goles, Hándicaps, Córners, Tiros, BTTS) ---
                        for col_name, val in row.items():
                            if pd.isna(val) or str(val).strip() == '':
                                continue
                            
                            col_str = str(col_name).lower()
                            
                            if col_str in ['liga', 'pais', 'partido_id', 'home', 'away', 'inicio_utc', 'inicio_local']:
                                continue
                                
                            try:
                                val_num = float(val)
                                if val_num <= 1.0: continue
                            except ValueError:
                                continue 
                                
                            # 🎯 RADAR 1: Mercados de Texto (BTTS / Ambos Anotan)
                            if 'btts' in col_str or 'ambos' in col_str:
                                # Probabilidad Poisson de que AMBOS hagan al menos 1 gol
                                prob_btts_si = (1 - math.exp(-pred_goles_home)) * (1 - math.exp(-pred_goles_away))
                                
                                if 'yes' in col_str or 'si' in col_str:
                                    mercados_a_evaluar.append(("Ambos Anotan (Sí)", val_num, prob_btts_si))
                                elif 'no' in col_str:
                                    mercados_a_evaluar.append(("Ambos Anotan (No)", val_num, 1 - prob_btts_si))
                                continue # Terminamos con esta columna, pasamos a la siguiente

                            # 🎯 RADAR 2: Mercados Asiáticos / Totales (Busca el .5)
                            match = re.search(r'(-?\d+\.5)', col_str)
                            if not match:
                                continue
                                
                            linea = float(match.group(1))
                            
                            # 1. HÁNDICAP ASIÁTICO
                            if 'hdp' in col_str or 'handicap' in col_str:
                                if 'home' in col_str:
                                    mercados_a_evaluar.append((f"Hándicap Local ({linea:+})", val_num, prob_handicap(pred_goles_home, pred_goles_away, linea)))
                                elif 'away' in col_str:
                                    mercados_a_evaluar.append((f"Hándicap Visita ({linea:+})", val_num, prob_handicap(pred_goles_away, pred_goles_home, linea)))
                                    
                            # 2. CÓRNERS (Totales y Por Equipo)
                            elif 'corners' in col_str:
                                if 'home' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Local (+{linea})", val_num, prob_over(stats_h['HC'], linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Local (-{linea})", val_num, prob_under(stats_h['HC'], linea)))
                                elif 'away' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Visita (+{linea})", val_num, prob_over(stats_a['AC'], linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Visita (-{linea})", val_num, prob_under(stats_a['AC'], linea)))
                                elif 'total' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Córners Totales (+{linea})", val_num, prob_over(prom_corners_total, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Córners Totales (-{linea})", val_num, prob_under(prom_corners_total, linea)))

                            # 3. TIROS A PUERTA (Totales y Por Equipo)
                            elif 'shots' in col_str:
                                if 'home' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Tiros Local (+{linea})", val_num, prob_over(stats_h['HST'], linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Tiros Local (-{linea})", val_num, prob_under(stats_h['HST'], linea)))
                                elif 'away' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Tiros Visita (+{linea})", val_num, prob_over(stats_a['AST'], linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Tiros Visita (-{linea})", val_num, prob_under(stats_a['AST'], linea)))
                                elif 'total' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Tiros a Puerta Totales (+{linea})", val_num, prob_over(prom_shots_total, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Tiros a Puerta Totales (-{linea})", val_num, prob_under(prom_shots_total, linea)))

                            # 4. GOLES (Totales y Por Equipo)
                            elif 'goles' in col_str or 'total' in col_str:
                                if 'tt_home' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Goles Local (+{linea})", val_num, prob_over(pred_goles_home, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Goles Local (-{linea})", val_num, prob_under(pred_goles_home, linea)))
                                elif 'tt_away' in col_str:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Goles Visita (+{linea})", val_num, prob_over(pred_goles_away, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Goles Visita (-{linea})", val_num, prob_under(pred_goles_away, linea)))
                                else:
                                    if 'over' in col_str: mercados_a_evaluar.append((f"Goles Totales (+{linea})", val_num, prob_over(prom_goles_total, linea)))
                                    elif 'under' in col_str: mercados_a_evaluar.append((f"Goles Totales (-{linea})", val_num, prob_under(prom_goles_total, linea)))

                        def evaluar_edge(mercado_nombre, prob_ia, cuota):
                            if cuota is None: return
                            try:
                                cuota_flt = float(cuota)
                                edge = prob_ia - (1 / cuota_flt)
                                
                                log_debug.append(f"📊 Evaluando: {h_db} - {mercado_nombre} | Cuota: {cuota_flt} | Prob IA: {prob_ia:.1%} | Edge: {edge:.2%}")
                                
                                # 🚨 FILTRO OPTIMIZADO: Edge entre 2% y 15%
                                if 0.02 <= edge <= 0.15:
                                    oportunidades.append((fecha_partido, h_db, a_db, mercado_nombre, cuota_flt, prob_ia, edge))
                                    log_debug.append(f"   ✨ ¡AÑADIDO AL PORTAFOLIO! Edge válido: {edge:.2%}")
                            except Exception as e:
                                pass

                        for nombre_mkt, cuota_val, prob_ia in mercados_a_evaluar:
                            evaluar_edge(nombre_mkt, prob_ia, cuota_val)

                    # --- RENDERS ---
                    with st.expander("🛠️ Ver Diagnóstico Completo del Robot Evaluador"):
                        for log_msg in log_debug: st.text(log_msg)

                    if oportunidades:
                        df_ops = pd.DataFrame(oportunidades, columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge'])
                        st.session_state['portafolio_escaneado'] = df_ops.sort_values(by='Edge', ascending=False).drop_duplicates(subset=['Home', 'Mercado']).reset_index(drop=True)
                    else:
                        st.warning("📊 No se encontraron ineficiencias dentro del rango rentable (2% a 15%).")

            # --- TABLA INTERACTIVA (ESTRUCTURA 3-3-3-1) ---
            if 'portafolio_escaneado' in st.session_state:
                df_ops = st.session_state['portafolio_escaneado'].copy()
                
                df_ops['Partido'] = df_ops['Home'] + " vs " + df_ops['Away']
                df_ops['Edge_Str'] = (df_ops['Edge'] * 100).round(2).astype(str) + "%"
                df_ops['Prob_IA_Str'] = (df_ops['Prob_IA'] * 100).round(1).astype(str) + "%"
                
                columnas_base = ['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str', 'Date', 'Home', 'Away', 'Prob_IA', 'Edge']
                df_ops = df_ops[columnas_base]

                # 🧠 🎯 CAMBIO 1: LÓGICA DE PARTIDOS ÚNICOS
                selected_indices = []
                used_matches = set() # Aquí guardaremos la memoria de los partidos ya elegidos
                df_top_10_list = []
                
                def add_pick(idx, nivel_label):
                    partido = df_ops.loc[idx, 'Partido']
                    # Si el partido ya tiene un pick, LO IGNORAMOS
                    if partido not in used_matches:
                        selected_indices.append(idx)
                        used_matches.add(partido) # Lo registramos para no volver a usarlo
                        df_top_10_list.append(df_ops.loc[[idx]].assign(Nivel=nivel_label))
                        return True
                    return False

                def get_available_pool(min_c, max_c):
                    # Filtra las cuotas y ELIMINA los partidos que ya están en el portafolio
                    disponibles = df_ops[~df_ops['Partido'].isin(used_matches) & (df_ops['Cuota'] >= min_c) & (df_ops['Cuota'] < max_c)]
                    # Elimina duplicados del MISMO partido dentro de esta piscina (se queda con el de mayor edge)
                    return disponibles.drop_duplicates(subset=['Partido'], keep='first')

                # 1. ⭐ Golden Pick 
                for idx in df_ops.index:
                    if add_pick(idx, '⭐ Golden Pick'): break
                
                # 2. 🔴 Alto Riesgo (Cuotas >= 2.50) -> Top 3
                pool_high = get_available_pool(2.50, 999.0)
                for idx in pool_high.head(3).index:
                    add_pick(idx, '🔴 Alto (>2.50)')
                    
                # 3. 🟡 Medio Riesgo (Cuotas 1.90 - 2.49) -> Top, Medio, Bajo
                pool_med = get_available_pool(1.90, 2.50)
                if not pool_med.empty:
                    idxs = [pool_med.index[0]]
                    if len(pool_med) >= 3: idxs.extend([pool_med.index[len(pool_med)//2], pool_med.index[-1]])
                    elif len(pool_med) == 2: idxs.append(pool_med.index[-1])
                    for idx in idxs: add_pick(idx, '🟡 Medio (1.90-2.49)')

                # 4. 🟢 Bajo Riesgo (Cuotas < 1.90) -> Top, Medio, Bajo
                pool_low = get_available_pool(0.0, 1.90)
                if not pool_low.empty:
                    idxs = [pool_low.index[0]]
                    if len(pool_low) >= 3: idxs.extend([pool_low.index[len(pool_low)//2], pool_low.index[-1]])
                    elif len(pool_low) == 2: idxs.append(pool_low.index[-1])
                    for idx in idxs: add_pick(idx, '🟢 Bajo (<1.90)')

                # --- ENSAMBLAJE ---
                df_top_10 = pd.concat(df_top_10_list).reset_index(drop=True) if df_top_10_list else pd.DataFrame()
                cols_mostrar = ['Nivel', 'Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']
                
                df_mostrar_top = df_top_10[cols_mostrar].copy()
                df_mostrar_top.insert(0, "✅ Añadir", True) # Estos vienen pre-marcados
                
                # Tabla 2: El banquillo de suplentes (Reserva)
                df_reserva = df_ops[~df_ops.index.isin(selected_indices)].reset_index(drop=True)
                df_mostrar_reserva = df_reserva[['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']].copy()
                df_mostrar_reserva.insert(0, "✅ Añadir", False) # Estos vienen desmarcados

                # --- RENDERIZADO VISUAL ---
                st.success(f"Escaneo listo. Se construyó un portafolio equilibrado usando {len(df_top_10)} picks recomendados.")
                st.markdown("### 🎯 Portafolio (3-3-3-1)")
                
                # Mostramos la tabla principal
                edit_top10 = st.data_editor(
                    df_mostrar_top,
                    hide_index=True,
                    use_container_width=True,
                    key="editor_top10",
                    column_config={"✅ Añadir": st.column_config.CheckboxColumn(required=True)}
                )
                
                # Mostramos la tabla de reserva minimizada
                with st.expander(f"📂 Ver el resto de picks válidos ({len(df_reserva)} en Reserva)"):
                    if not df_mostrar_reserva.empty:
                        st.caption("Si desmarcaste algún pick de arriba, puedes seleccionar reemplazos desde aquí.")
                        edit_reserva = st.data_editor(
                            df_mostrar_reserva,
                            hide_index=True,
                            use_container_width=True,
                            key="editor_reserva",
                            column_config={"✅ Añadir": st.column_config.CheckboxColumn(required=True)}
                        )
                    else:
                        st.info("No hay más picks de reserva. Se utilizaron todos los disponibles.")

                # --- BOTÓN DE GUARDADO UNIFICADO ---
                if st.button("💾 Guardar Portafolio Seleccionado", type="primary"):
                    indices_top = edit_top10[edit_top10["✅ Añadir"] == True].index
                    indices_res = edit_reserva[edit_reserva["✅ Añadir"] == True].index if not df_mostrar_reserva.empty else []
                    
                    df_final_top = df_top_10.iloc[indices_top]
                    df_final_res = df_reserva.iloc[indices_res] if not df_mostrar_reserva.empty else pd.DataFrame()
                    df_final_a_guardar = pd.concat([df_final_top, df_final_res])
                    
                    if df_final_a_guardar.empty:
                        st.warning("No seleccionaste ningún pick.")
                    else:
                        # 🎯 CAMBIO 3: División exacta del dinero total entre los picks seleccionados
                        stake_por_pick = inversion_total / len(df_final_a_guardar)
                        
                        for _, row in df_final_a_guardar.iterrows():
                            cursor.execute("""
                                INSERT INTO portafolio_historico (Date, HomeTeam, AwayTeam, Mercado, Cuota, Prob_IA, Edge, Stake)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (row['Date'], row['Home'], row['Away'], row['Mercado'], row['Cuota'], row['Prob_IA'], row['Edge'], stake_por_pick))
                        conn.commit()
                        st.toast(f"¡{len(df_final_a_guardar)} picks guardados! (Inversión por pick: ${stake_por_pick:,.0f})")
                        del st.session_state['portafolio_escaneado']
                        st.rerun()

        except Exception as e:
            st.error(f"Error en la aplicación: {e}")

    with tab2:
        # 🎯 CAMBIO 2: Botón rojo para limpiar toda la tabla de rendimiento
        c_tit, c_btn = st.columns([0.75, 0.25])
        c_tit.subheader("🏦 Rendimiento Acumulado")
        
        if c_btn.button("🗑️ Resetear Historial", use_container_width=True):
            cursor.execute("DELETE FROM portafolio_historico")
            conn.commit()
            st.toast("¡Historial borrado con éxito! Portafolio limpio.")
            st.rerun()
            
        # 1.Liquidar apuestas pendientes
        if st.button("⚖️ Liquidar Apuestas Pendientes", type="primary"):
            df_pendientes = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente'", conn)
            liquidadas = 0
            
            # Variables para calcular el Yield de esta liquidación específica
            beneficio_reciente = 0.0
            stake_reciente = 0.0
            
            import re
            
            for _, pick in df_pendientes.iterrows():
                q_res = f"SELECT * FROM historial_multiliga_ml WHERE Date = '{pick['Date']}' AND HomeTeam = '{pick['HomeTeam']}'"
                res_real = pd.read_sql(q_res, conn)
                
                if not res_real.empty:
                    row = res_real.iloc[0]
                    
                    hg = row['FTHG'] if pd.notna(row.get('FTHG')) else None
                    ag = row['FTAG'] if pd.notna(row.get('FTAG')) else None
                    
                    if hg is None or ag is None:
                        continue
                        
                    hc = row['HC'] if 'HC' in row and pd.notna(row['HC']) else 0
                    ac = row['AC'] if 'AC' in row and pd.notna(row['AC']) else 0
                    hst = row['HST'] if 'HST' in row and pd.notna(row['HST']) else 0
                    ast = row['AST'] if 'AST' in row and pd.notna(row['AST']) else 0
                    
                    mkt = pick['Mercado']
                    ganada = False
                    
                    # --- MOTOR DE RESOLUCIÓN ---
                    if mkt == "Ganador (Local)": ganada = (hg > ag)
                    elif mkt == "Empate": ganada = (hg == ag)
                    elif mkt == "Ganador (Visita)": ganada = (ag > hg)
                    elif mkt == "Ambos Anotan (Sí)": ganada = (hg > 0 and ag > 0)
                    elif mkt == "Ambos Anotan (No)": ganada = (hg == 0 or ag == 0)
                    else:
                        match = re.search(r'\(([+-]\d+\.5)\)', mkt)
                        if match:
                            signo = match.group(1)[0] 
                            valor_linea = float(match.group(1)[1:]) 
                            linea_matematica = float(match.group(1)) 
                            
                            if "Hándicap" in mkt:
                                if "Local" in mkt: ganada = (hg + linea_matematica > ag)
                                elif "Visita" in mkt: ganada = (ag + linea_matematica > hg)
                            else:
                                score = -1
                                if "Goles Local" in mkt: score = hg
                                elif "Goles Visita" in mkt: score = ag
                                elif "Goles" in mkt: score = hg + ag
                                elif "Córners Local" in mkt: score = hc
                                elif "Córners Visita" in mkt: score = ac
                                elif "Córners" in mkt: score = hc + ac
                                elif "Tiros Local" in mkt: score = hst
                                elif "Tiros Visita" in mkt: score = ast
                                elif "Tiros" in mkt: score = hst + ast
                                
                                if signo == '+': ganada = (score > valor_linea)
                                elif signo == '-': ganada = (score < valor_linea)

                    # --- CÁLCULO DE LIQUIDACIÓN ---
                    estado = 'Ganada' if ganada else 'Perdida'
                    beneficio = (pick['Stake'] * pick['Cuota']) - pick['Stake'] if ganada else -pick['Stake']
                    
                    cursor.execute("UPDATE portafolio_historico SET Estado = ?, Beneficio_Neto = ? WHERE id = ?", (estado, beneficio, pick['id']))
                    
                    # Sumamos a las variables recientes
                    liquidadas += 1
                    beneficio_reciente += beneficio
                    stake_reciente += pick['Stake']
            
            conn.commit()
            
            # Mensaje dinámico con el rendimiento de la tanda recién liquidada
            if liquidadas > 0: 
                yield_tanda = (beneficio_reciente / stake_reciente * 100) if stake_reciente > 0 else 0
                st.success(f"¡Se liquidaron {liquidadas} partidos! 📈 Beneficio de esta tanda: **${beneficio_reciente:,.0f}** (Yield: **{yield_tanda:.2f}%**)")
            else: 
                st.info("No hay partidos nuevos terminados para liquidar.")

        # 2. Mostrar Resultados Globales
        df_hist = pd.read_sql("SELECT * FROM portafolio_historico", conn)
        
        if not df_hist.empty:
            df_cerradas = df_hist[df_hist['Estado'] != 'Pendiente']
            
            # Ampliamos a 4 columnas para agregar el Yield
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            with c_res1:
                st.metric("Picks Cerrados", len(df_cerradas))
            with c_res2:
                ganadas = len(df_cerradas[df_cerradas['Estado'] == 'Ganada'])
                win_rate = (ganadas / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with c_res3:
                # Matemáticas del Yield Global: (Ganancia Neta Total / Dinero Total Invertido) * 100
                beneficio_total = df_cerradas['Beneficio_Neto'].sum()
                inversion_total = df_cerradas['Stake'].sum()
                yield_global = (beneficio_total / inversion_total * 100) if inversion_total > 0 else 0
                st.metric("Yield (ROI)", f"{yield_global:.2f}%")
            with c_res4:
                st.metric("Ganancia Neta Global", f"${beneficio_total:,.0f}")
                
            st.divider()
            st.write("📋 **Historial de Picks**")
            
            df_mostrar = df_hist[['Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Beneficio_Neto']].copy()
            df_mostrar['Es_Pendiente'] = df_mostrar['Estado'] == 'Pendiente'
            df_mostrar = df_mostrar.sort_values(by=['Es_Pendiente', 'Date'], ascending=[False, False]).drop(columns=['Es_Pendiente'])
            
            def color_estado(val):
                if val == 'Ganada': return 'color: #00FF00; font-weight: bold'
                elif val == 'Perdida': return 'color: #FF4B4B'
                elif val == 'Pendiente': return 'color: #FFD700'
                return ''
                
            st.dataframe(df_mostrar.style.map(color_estado, subset=['Estado']), hide_index=True, use_container_width=True)


conn.close()
