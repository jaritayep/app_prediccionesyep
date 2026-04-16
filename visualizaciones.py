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
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); color: white; }
    footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stMetric { background-color: #1e2129; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

CONFIG_FIJA = {'staticPlot': False, 'scrollZoom': False, 'doubleClick': 'reset', 'displayModeBar': False, 'showAxisDragHandles': False}

def corregir_nombre_equipo(nombre_api, lista_db):
    if not lista_db: return nombre_api
    mejor_match, score = process.extractOne(nombre_api.strip(), lista_db, scorer=fuzz.token_set_ratio)
    return mejor_match if score > 50 else nombre_api

def cargar_modelo():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def get_recent_stats(equipo, conn):
    q = f'SELECT "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY" FROM historial_multiliga_ml WHERE HomeTeam="{equipo}" OR AwayTeam="{equipo}" ORDER BY Date DESC LIMIT 5'
    res = pd.read_sql(q, conn)
    if res.empty: return pd.Series(0, index=['FTHG','FTAG','HS','AS','HST','AST','HC','AC','HY','AY'])
    pesos = np.array([5, 4, 3, 2, 1])[:len(res)]
    return pd.Series({col: np.average(res[col], weights=pesos/pesos.sum()) for col in res.columns})

conn = sqlite3.connect(DB_NAME)

st.sidebar.title("⚽ Menú Principal")
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "BetBuilder Simulator", "Comparador H2H"])
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
                    
                    # Preparación de datos para el modelo
                    input_data = [[stats_h['FTHG'], stats_h['FTAG'], stats_h['HS'], stats_h['AS'], stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'], stats_h['HY'], stats_h['AY']]]
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

                    # Métricas de Goles
                    c1, c2 = st.columns(2)
                    c1.metric("Goles Exp. (Total)", f"{promedio_goles:.2f}")
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
    st.title("🎯 Auditoría de Precisión")
    
    # --- 1. SELECTOR DE FECHA (El "Toggle") ---
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        # Selector de fecha (Ayer por defecto)
        fecha_audit = st.date_input("Selecciona fecha para auditar:", 
                                    datetime.now() - timedelta(days=1))
    
    fecha_str = fecha_audit.strftime('%Y-%m-%d')

    # --- 2. CARGA DE DATOS (Filtro Flexible) ---
    # Usamos LIKE para capturar fechas aunque tengan horas (ej: 2026-04-13 00:00:00)
    query = "SELECT * FROM historial_multiliga_ml WHERE Date LIKE ?"
    df_reales = pd.read_sql(query, conn, params=(f"{fecha_str}%",))

    if df_reales.empty:
        st.warning(f"⚠️ No hay resultados registrados en el historial para el {fecha_audit.strftime('%d/%m/%Y')}.")
        
        # DEBUG: Si no hay nada, veamos qué fechas existen realmente para orientarte
        st.info("Buscando fechas disponibles en la base de datos...")
        ultimas = pd.read_sql("SELECT DISTINCT Date FROM historial_multiliga_ml ORDER BY Date DESC LIMIT 3", conn)
        if not ultimas.empty:
            st.write("Últimas fechas con datos en el historial:", ultimas['Date'].tolist())
    else:
        st.subheader(f"📊 Resumen de Jornada: {fecha_audit.strftime('%d/%m/%Y')}")

        total_predicciones = 0
        cumplidas = 0
        resultados_procesados = []

        # Barra de progreso para el cálculo
        with st.spinner('Calculando precisión contra proyecciones IA...'):
            for _, r in df_reales.iterrows():
                # Obtenemos stats históricas de ambos equipos para recrear la proyección de la IA
                sh = get_recent_stats(r['HomeTeam'], conn)
                sa = get_recent_stats(r['AwayTeam'], conn)
                
                if sh and sa:
                    # Lo que la IA habría proyectado:
                    proj_goles = (sh['FTHG'] + sh['FTAG'] + sa['FTHG'] + sa['FTAG']) / 2
                    proj_corners = sh['HC'] + sa['AC']
                    
                    # Lo que pasó en realidad:
                    real_goles = r['FTHG'] + r['FTAG']
                    real_corners = r['HC'] + r['AC']
                    
                    # Verificación de aciertos
                    goles_ok = real_goles >= proj_goles
                    corners_ok = real_corners >= proj_corners
                    
                    if goles_ok: cumplidas += 1
                    if corners_ok: cumplidas += 1
                    total_predicciones += 2
                    
                    resultados_procesados.append({
                        'fila': r,
                        'sh': sh, 'sa': sa,
                        'proj_goles': proj_goles, 'proj_corners': proj_corners,
                        'goles_ok': goles_ok, 'corners_ok': corners_ok
                    })

        # --- 3. MÉTRICAS SUPERIORES ---
        if total_predicciones > 0:
            tasa_total = cumplidas / total_predicciones
            st.metric("Tasa de Cumplimiento Global", f"{tasa_total:.1%}", 
                      delta=f"{cumplidas}/{total_predicciones} Mercados Logrados")
            
            st.divider()

            # --- 4. ACORDEONES POR PARTIDO ---
            for res in resultados_procesados:
                r = res['fila']
                sh, sa = res['sh'], res['sa']
                
                # Formato del título del acordeón con el marcador real
                titulo = f"🏟️ {r['HomeTeam']} {int(r['FTHG'])} - {int(r['FTAG'])} {r['AwayTeam']}"
                
                with st.expander(titulo):
                    # Definimos las métricas a mostrar
                    metrica_data = [
                        ("Goles Total", res['proj_goles'], r['FTHG'] + r['FTAG']),
                        ("Córners", res['proj_corners'], r['HC'] + r['AC']),
                        ("Tiros Arco", sh['HST'] + sa['AST'], r['HST'] + r['AST']),
                        ("Amarillas", sh['HY'] + sa['AY'], r['HY'] + r['AY'])
                    ]

                    # Generamos el HTML para las cajas de resultados
                    cols = st.columns(2)
                    for i, (label, p, re) in enumerate(metrica_data):
                        real_val = re if pd.notnull(re) else 0
                        check = "✅" if real_val >= p else "❌"
                        color = "#27ae60" if real_val >= p else "#c0392b"
                        
                        # Alternamos entre columna 0 y 1
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div style="border-left: 5px solid {color}; padding: 8px; margin-bottom: 10px; background-color: #1e2129; border-radius: 5px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #888; font-size: 0.75rem; font-weight: bold;">{label.upper()}</span>
                                    <span>{check}</span>
                                </div>
                                <div style="margin-top: 5px;">
                                    <span style="font-size: 0.9rem; color: #bbb;">IA:</span> 
                                    <span style="font-size: 1rem; font-weight: bold;">{p:.1f}</span>
                                    <span style="color: #555; margin: 0 5px;">|</span>
                                    <span style="font-size: 0.9rem; color: #bbb;">Real:</span> 
                                    <span style="font-size: 1rem; font-weight: bold;">{int(real_val)}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No se pudieron calcular proyecciones para los partidos de esta fecha (Faltan datos históricos de los equipos).")
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

            # 3. CARGAR ESTADÍSTICAS Y CÁLCULOS DE GOLES
            stats_h = get_recent_stats(home_team, conn)
            stats_a = get_recent_stats(away_team, conn)

            # Predicción individual (Ataque propio vs Defensa rival)
            pred_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
            pred_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2

            st.divider()
            col_config, col_ticket = st.columns([1.2, 1])

            with col_config:
                st.subheader("🎯 Configurar Mercados")
                
                mercado = st.selectbox("Seleccionar Mercado:", [
                    "Goles Totales", "Goles por Equipo", "Córners Totales", "Doble Oportunidad", "Tiros a Puerta"
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

                    elif mercado == "Córners Totales":
                        l_c = st.slider("Línea Córners:", 5.5, 14.5, 8.5, 0.5)
                        t_c = st.radio("Predicción:", ["Over", "Under"], horizontal=True)
                        prom_c = stats_h['HC'] + stats_a['AC']
                        prob = 1 / (1 + np.exp(-(prom_c - l_c))) if t_c == "Over" else 1 - (1 / (1 + np.exp(-(prom_c - l_c))))
                        desc_pick = f"{t_c} {l_c} Córners"

                    elif mercado == "Doble Oportunidad":
                        opts = st.multiselect("Opciones:", ["Local", "Empate", "Visita"], default=["Local", "Empate"])
                        prob = len(opts) * 0.32 # Probabilidad estimada
                        desc_pick = " o ".join(opts)

                    elif mercado == "Tiros a Puerta":
                        l_t = st.number_input("Mínimo Tiros Totales:", 4, 20, 8)
                        prom_t = stats_h['HST'] + stats_a['AST']
                        prob = 1 / (1 + np.exp(-(prom_t - l_t)))
                        desc_pick = f"Más de {l_t} Tiros a Puerta"

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
                    st.metric("Cuota Justa", f"{cuota:.2f}")

                    if st.button("🗑️ Limpiar Ticket"):
                        st.session_state.ticket = []
                        st.rerun()

    except Exception as e:
        st.error(f"Error en el Simulador: {e}")
elif menu == "Comparador H2H":
elif menu == "Comparador H2H":
    st.title("⚖️ Comparador H2H Inteligente")
    st.markdown("Ajusta el análisis según el factor campo para obtener proyecciones más precisas.")

    # --- FUNCIÓN DE CÁLCULO AJUSTADA ---
    def obtener_stats_personalizadas(equipo, conn_db, modo, limite=10):
        """
        Calcula stats filtrando por: 'Local', 'Visitante' o 'Todas (Últimos 10)'
        """
        # Query para Local
        query_h = "SELECT FTHG as GF, FTAG as GC, HC as CF, AC as CC, HST as TF, AST as TC FROM historial_multiliga_ml WHERE HomeTeam = ? ORDER BY Date DESC LIMIT ?"
        # Query para Visita
        query_a = "SELECT FTAG as GF, FTHG as GC, AC as CF, HC as CC, AST as TF, HST as TC FROM historial_multiliga_ml WHERE AwayTeam = ? ORDER BY Date DESC LIMIT ?"
        
        if modo == "Solo Local":
            df = pd.read_sql(query_h, conn_db, params=(equipo, limite))
        elif modo == "Solo Visitante":
            df = pd.read_sql(query_a, conn_db, params=(equipo, limite))
        else:
            df_h = pd.read_sql(query_h, conn_db, params=(equipo, limite))
            df_a = pd.read_sql(query_a, conn_db, params=(equipo, limite))
            df = pd.concat([df_h, df_a]).sort_index(ascending=False).head(limite)

        if df.empty: return None
        return {
            'Goles a Favor': df['GF'].mean(),
            'Goles en Contra': df['GC'].mean(),
            'Córners a Favor': df['CF'].mean(),
            'Córners en Contra': df['CC'].mean(),
            'Tiros al Arco': df['TF'].mean()
        }

    # --- DICCIONARIO DE LIGAS (El mismo que ya tienes) ---
    query_todos = "SELECT DISTINCT HomeTeam FROM historial_multiliga_ml"
    equipos_db = sorted(pd.read_sql(query_todos, conn)['HomeTeam'].dropna().tolist())
    keywords_ligas = {
        "Premier League": ["Arsenal", "Aston", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Crystal", "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool", "Man", "Newcastle", "Nott", "Southampton", "Tottenham", "West Ham", "Wolves"],
        "La Liga": ["Alaves", "Athletic", "Atletico", "Barcelona", "Betis", "Celta", "Espanyol", "Getafe", "Girona", "Palmas", "Leganes", "Mallorca", "Osasuna", "Rayo", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Valladolid", "Villarreal"],
        "Serie A": ["Atalanta", "Bologna", "Cagliari", "Como", "Empoli", "Fiorentina", "Genoa", "Verona", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Parma", "Roma", "Torino", "Udinese", "Venezia"],
        "Bundesliga": ["Augsburg", "Bayer", "Bayern", "Bochum", "Dortmund", "Frankfurt", "Freiburg", "Heidenheim", "Hoffenheim", "Kiel", "Leipzig", "Mainz", "Monchengladbach", "Pauli", "Stuttgart", "Union", "Werder", "Wolfsburg"],
        "Ligue 1": ["Angers", "Auxerre", "Brest", "Havre", "Lens", "Lille", "Lyon", "Marseille", "Monaco", "Montpellier", "Nantes", "Nice", "Paris", "PSG", "Reims", "Rennes", "Etienne", "Strasbourg", "Toulouse"]
    }
    ligas_opciones = list(keywords_ligas.keys()) + ["Todas / Otras Ligas"]

    # --- UI: FILTROS DE LIGA Y EQUIPO ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        l_a = st.selectbox("Liga A", ligas_opciones, key="la")
        filt_a = [eq for eq in equipos_db if any(k.lower() in eq.lower() for k in keywords_ligas.get(l_a, []))] or equipos_db
        eq_a = st.selectbox("Equipo A", sorted(filt_a), key="ea")
        # EL NUEVO SELECTOR DE LOCALÍA
        modo_a = st.radio("Ver rendimiento de:", ["Juntas", "Solo Local", "Solo Visitante"], key="ma", horizontal=True)

    with col_b:
        l_b = st.selectbox("Liga B", ligas_opciones, key="lb", index=1)
        filt_b = [eq for eq in equipos_db if any(k.lower() in eq.lower() for k in keywords_ligas.get(l_b, []))] or equipos_db
        eq_b = st.selectbox("Equipo B", sorted(filt_b), key="eb")
        # EL NUEVO SELECTOR DE LOCALÍA
        modo_b = st.radio("Ver rendimiento de:", ["Juntas", "Solo Visitante", "Solo Local"], key="mb", horizontal=True)

    # --- RENDERIZADO DE RESULTADOS ---
    if eq_a and eq_b:
        st.divider()
        stats_a = obtener_stats_personalizadas(eq_a, conn, modo_a)
        stats_b = obtener_stats_personalizadas(eq_b, conn, modo_b)

        if stats_a and stats_b:
            st.markdown(f"<h3 style='text-align: center;'>{eq_a} ({modo_a}) vs {eq_b} ({modo_b})</h3>", unsafe_allow_html=True)
            
            metricas = [
                ("⚽ Goles a Favor", 'Goles a Favor'),
                ("🛡️ Goles en Contra", 'Goles en Contra'),
                ("🚩 Córners a Favor", 'Córners a Favor'),
                ("🎯 Tiros al Arco", 'Tiros al Arco')
            ]

            for icono, clave in metricas:
                val_a, val_b = stats_a[clave], stats_b[clave]
                diff_a, diff_b = val_a - val_b, val_b - val_a
                delta_color = "inverse" if clave == 'Goles en Contra' else "normal"

                c1, c2, c3 = st.columns([1, 2, 1])
                with c1: st.metric(label="", value=f"{val_a:.1f}", delta=f"{diff_a:.1f}", delta_color=delta_color)
                with c2: st.markdown(f"<div style='text-align: center; padding-top: 15px; font-weight: bold; color: #888;'>{icono}</div>", unsafe_allow_html=True)
                with c3: st.metric(label="", value=f"{val_b:.1f}", delta=f"{diff_b:.1f}", delta_color=delta_color)

            st.divider()
            st.subheader("Gráfico Comparativo Ajustado")
            df_grafico = pd.DataFrame({
                'Métrica': [m[1] for m in metricas],
                f"{eq_a} ({modo_a})": [stats_a[m[1]] for m in metricas],
                f"{eq_b} ({modo_b})": [stats_b[m[1]] for m in metricas]
            }).set_index('Métrica')
            st.bar_chart(df_grafico)
        else:
            st.info("Datos insuficientes para este filtro de localía.")



conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
