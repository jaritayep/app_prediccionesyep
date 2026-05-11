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
    tab1, tab2 = st.tabs(["🔍 Escáner en Vivo", "📊 Rendimiento Histórico"])

    with tab1:
        st.markdown("Cruzando probabilidades IA contra cuotas reales para guardar los 10 mejores picks.")
        
        ligas_api = {
            'EPL': 'soccer_epl', 'LaLiga': 'soccer_spain_la_liga',
            'SerieA': 'soccer_italy_serie_a', 'Bundesliga': 'soccer_germany_bundesliga',
            'Ligue1': 'soccer_france_ligue_one'
        }

        try:
            equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
            df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
            df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.tz_localize(None).dt.normalize()
            
            hoy = pd.Timestamp.now().normalize()
            manana = hoy + pd.Timedelta(days=1)
            df_jornada = df_jornada[(df_jornada['Date'] >= hoy) & (df_jornada['Date'] <= manana)]

            if df_jornada.empty:
                st.info("📅 No hay partidos programados para hoy ni mañana en la base de datos.")
            else:
                df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m/%Y')
                opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))

                c1, c2 = st.columns(2)
                with c1:
                    stake_fijo = st.number_input("💰 Inversión FIJA por Pick ($)", min_value=1000, value=5000, step=1000)
                with c2:
                    dia_seleccionado_str = st.selectbox("📅 Seleccionar Día a Escanear:", opciones_fecha)

                df_jornada_dia = df_jornada[df_jornada['Fecha_Display'] == dia_seleccionado_str]
                fecha_para_guardar = df_jornada_dia.iloc[0]['Date'].strftime('%Y-%m-%d')

                # --- BOTÓN DE ESCANEO ---
                if st.button("🔍 Escanear Mercado en Vivo", type="primary"):
                    with st.spinner(f"Buscando ineficiencias de mercado para el {dia_seleccionado_str}..."):
                        oportunidades = []
                        ligas_hoy = df_jornada_dia['League'].unique()
                        errores_api = 0

                        for liga in ligas_hoy:
                            if liga not in ligas_api: continue
                            sport_key = ligas_api[liga]
                            
                            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={API_KEY}&regions=eu,uk&markets=h2h,totals&oddsFormat=decimal"
                            response = requests.get(url)
                            
                            # AVISO DE ERROR SI LA API FALLA
                            if response.status_code != 200:
                                errores_api += 1
                                st.error(f"Error en la API con la liga {liga}: {response.text}")
                                continue
                                
                            datos_api = response.json()
                            partidos_liga_hoy = df_jornada_dia[df_jornada_dia['League'] == liga]

                            for partido_api in datos_api:
                                h_api, a_api = partido_api['home_team'], partido_api['away_team']
                                h_db = process.extractOne(h_api, equipos_db)[0]
                                a_db = process.extractOne(a_api, equipos_db)[0]
                                
                                if not ((partidos_liga_hoy['Local'] == h_db) & (partidos_liga_hoy['Visita'] == a_db)).any(): continue
                                if not partido_api['bookmakers']: continue
                                bookie = partido_api['bookmakers'][0] 
                                
                                cuota_h = cuota_a = cuota_o25 = 0
                                
                                for market in bookie['markets']:
                                    if market['key'] == 'h2h':
                                        for out in market['outcomes']:
                                            if out['name'] == h_api: cuota_h = out['price']
                                            elif out['name'] == a_api: cuota_a = out['price']
                                    elif market['key'] == 'totals':
                                        for out in market['outcomes']:
                                            if out['name'] == 'Over' and out.get('point') == 2.5: cuota_o25 = out['price']

                                stats_h = get_recent_stats(h_db, conn)
                                stats_a = get_recent_stats(a_db, conn)
                                xg_h = stats_h.get('xG_home', 1.0)
                                xg_a = stats_a.get('xG_away', 1.0)
                                pred_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
                                pred_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2

                                if cuota_h > 0:
                                    prob_ia = 1 / (1 + np.exp(-(xg_h - xg_a)))
                                    oportunidades.append((fecha_para_guardar, h_db, a_db, 'Local', cuota_h, prob_ia, prob_ia - (1/cuota_h)))
                                if cuota_a > 0:
                                    prob_ia = 1 / (1 + np.exp(-(xg_a - xg_h)))
                                    oportunidades.append((fecha_para_guardar, h_db, a_db, 'Visita', cuota_a, prob_ia, prob_ia - (1/cuota_a)))
                                if cuota_o25 > 0:
                                    prob_ia = 1 / (1 + np.exp(-((pred_home + pred_away) - 2.5)))
                                    oportunidades.append((fecha_para_guardar, h_db, a_db, '+2.5 Goles', cuota_o25, prob_ia, prob_ia - (1/cuota_o25)))

                        # SI TODO SALIÓ BIEN, GUARDAMOS EN LA MEMORIA DE STREAMLIT
                        if oportunidades:
                            df_ops = pd.DataFrame(oportunidades, columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob IA', 'Edge'])
                            df_validas = df_ops[(df_ops['Edge'] > 0.03) & (df_ops['Edge'] < 0.15)].copy()
                            st.session_state['portafolio_escaneado'] = df_validas.sort_values(by='Edge', ascending=False).groupby('Home').head(1).head(10).reset_index(drop=True)
                        elif errores_api == 0:
                            st.warning("No se encontraron cuotas para los partidos seleccionados o el mercado ya cerró.")

                # --- MOSTRAR RESULTADOS Y BOTÓN DE GUARDAR (FUERA DEL BOTÓN DE ESCANEO) ---
                if 'portafolio_escaneado' in st.session_state:
                    df_portfolio = st.session_state['portafolio_escaneado']
                    
                    if df_portfolio.empty:
                        st.info("El mercado es eficiente hoy. No hay Edge de valor entre 3% y 15%.")
                    else:
                        st.success(f"Se encontraron {len(df_portfolio)} picks.")
                        df_display = df_portfolio.copy()
                        df_display['Partido'] = df_display['Home'] + " vs " + df_display['Away']
                        df_display['Edge'] = (df_display['Edge'] * 100).round(2).astype(str) + "%"
                        df_display['Prob IA'] = (df_display['Prob IA'] * 100).round(1).astype(str) + "%"
                        
                        st.dataframe(df_display[['Partido', 'Mercado', 'Cuota', 'Prob IA', 'Edge']], hide_index=True)

                        if st.button("💾 Guardar Portafolio Seleccionado", type="primary"):
                            for _, row in df_portfolio.iterrows():
                                cursor.execute("""
                                    INSERT INTO portafolio_historico 
                                    (Date, HomeTeam, AwayTeam, Mercado, Cuota, Prob_IA, Edge, Stake)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (row['Date'], row['Home'], row['Away'], row['Mercado'], row['Cuota'], row['Prob IA'], row['Edge'], stake_fijo))
                            conn.commit()
                            st.toast("¡Portafolio guardado! Revisar en la pestaña Rendimiento.")
                            
                            # Limpiamos la memoria para evitar guardarlo dos veces
                            del st.session_state['portafolio_escaneado']
                            st.rerun()

        except Exception as e:
            st.error(f"Error procesando: {e}")

    with tab2:
        st.subheader("🏦 Rendimiento Acumulado")
        
        # 1. BOTÓN MÁGICO: Liquidar apuestas pendientes (Cruza con tu actualización de 3 días)
        if st.button("⚖️ Liquidar Apuestas Pendientes", type="primary"):
            df_pendientes = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente'", conn)
            liquidadas = 0
            
            for _, pick in df_pendientes.iterrows():
                # Buscar resultado real en el historial
                q_res = f"SELECT FTHG, FTAG, FTR FROM historial_multiliga_ml WHERE Date = '{pick['Date']}' AND HomeTeam = '{pick['HomeTeam']}'"
                res_real = pd.read_sql(q_res, conn)
                
                if not res_real.empty:
                    hg, ag, ftr = res_real.iloc[0]['FTHG'], res_real.iloc[0]['FTAG'], res_real.iloc[0]['FTR']
                    ganada = False
                    
                    if pick['Mercado'] == 'Local' and ftr == 'H': ganada = True
                    elif pick['Mercado'] == 'Visita' and ftr == 'A': ganada = True
                    elif pick['Mercado'] == '+2.5 Goles' and (hg + ag) > 2.5: ganada = True
                    
                    estado = 'Ganada' if ganada else 'Perdida'
                    beneficio = (pick['Stake'] * pick['Cuota']) - pick['Stake'] if ganada else -pick['Stake']
                    
                    cursor.execute("UPDATE portafolio_historico SET Estado = ?, Beneficio_Neto = ? WHERE id = ?", (estado, beneficio, pick['id']))
                    liquidadas += 1
            
            conn.commit()
            if liquidadas > 0: st.success(f"¡Se liquidaron {liquidadas} partidos finalizados!")
            else: st.info("No hay partidos nuevos terminados para liquidar.")

        # 2. Mostrar Resultados Globales
        df_hist = pd.read_sql("SELECT * FROM portafolio_historico", conn)
        
        if not df_hist.empty:
            df_cerradas = df_hist[df_hist['Estado'] != 'Pendiente']
            
            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                st.metric("Picks Cerrados", len(df_cerradas))
            with c_res2:
                ganadas = len(df_cerradas[df_cerradas['Estado'] == 'Ganada'])
                win_rate = (ganadas / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with c_res3:
                beneficio_total = df_cerradas['Beneficio_Neto'].sum()
                st.metric("Ganancia Neta Global", f"${beneficio_total:,.0f}")
                
            st.divider()
            st.write("📋 **Historial de Picks**")
            # Damos formato para mostrar
            df_mostrar = df_hist[['Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Beneficio_Neto']].sort_values(by='Date', ascending=False)
            st.dataframe(df_mostrar, hide_index=True, use_container_width=True)


conn.close()
