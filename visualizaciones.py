import streamlit as st
import sqlite3
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os
from thefuzz import process, fuzz 
import math

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
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "BetBuilder Simulator"])
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
    st.title("Auditoría de Precisión")
    
    # 1. Cargar datos
    df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
    # Traemos resultados reales
    df_reales = pd.read_sql("SELECT * FROM historial_multiliga_ml ORDER BY Date DESC", conn)

    # --- LIMPIEZA CRÍTICA DE FECHAS (Parche para evitar error de comparación) ---
    df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.tz_localize(None).dt.normalize()
    df_reales['Date'] = pd.to_datetime(df_reales['Date']).dt.tz_localize(None).dt.normalize()
    # ----------------------------------------------------------------------------

    if not df_reales.empty:
        # Buscamos la última fecha que tiene resultados cargados
        ultima_fecha_real = df_reales['Date'].max()
        st.subheader(f"📊 Resumen Jornada: {ultima_fecha_real.strftime('%d/%m/%Y')}")

        # Filtrar partidos de esa jornada que estaban en nuestras predicciones
        partidos_auditar = df_jornada[df_jornada['Date'] == ultima_fecha_real]
        
        # --- CÁLCULO DE TASA DE CUMPLIMIENTO TOTAL ---
        total_predicciones = 0
        cumplidas = 0

        # Primero recorremos para calcular la métrica global
        for _, fila in partidos_auditar.iterrows():
            # Buscamos coincidencia por equipo local y fecha exacta
            match_r = df_reales[(df_reales['HomeTeam'] == fila['Local']) & (df_reales['Date'] == ultima_fecha_real)].head(1)
            if not match_r.empty:
                r = match_r.iloc[0]
                sh, sa = get_recent_stats(fila['Local'], conn), get_recent_stats(fila['Visita'], conn)
                
                # Definimos qué cuenta como "acierto" (usamos Goles y Córners como base)
                # Si el promedio IA se cumplió o superó en la realidad
                if (r['FTHG'] + r['FTAG']) >= ((sh['FTHG']+sh['FTAG']+sa['FTHG']+sa['FTAG'])/2): cumplidas += 1
                if (r['HC'] + r['AC']) >= (sh['HC'] + sa['AC']): cumplidas += 1
                total_predicciones += 2 

        # Mostrar Métrica Superior
        if total_predicciones > 0:
            tasa_total = cumplidas / total_predicciones
            st.metric("Tasa de Cumplimiento Global", f"{tasa_total:.1%}", 
                      delta=f"{cumplidas}/{total_predicciones} Predicciones Logradas",
                      help="Porcentaje de mercados (Goles y Córners) donde el resultado real igualó o superó la proyección de la IA.")
        
        st.divider()

        # --- LISTADO DE PARTIDOS ---
        if partidos_auditar.empty:
            st.warning("No se encontraron predicciones guardadas para la última fecha con resultados.")
        else:
            for _, fila in partidos_auditar.iterrows():
                match_real = df_reales[(df_reales['HomeTeam'] == fila['Local']) & (df_reales['Date'] == ultima_fecha_real)].head(1)
                
                with st.expander(f"🏟️ {fila['Local']} vs {fila['Visita']}"):
                    if not match_real.empty:
                        r = match_real.iloc[0]
                        sh, sa = get_recent_stats(fila['Local'], conn), get_recent_stats(fila['Visita'], conn)
                        
                        metrica_data = [
                            ("Goles", (sh['FTHG']+sh['FTAG']+sa['FTHG']+sa['FTAG'])/2, r['FTHG']+r['FTAG']),
                            ("Tiros Arco", sh['HST'] + sa['AST'], r['HST'] + r['AST']),
                            ("Córners", sh['HC'] + sa['AC'], r['HC'] + r['AC']),
                            ("Amarillas", sh['HY'] + sa['AY'], r['HY'] + r['AY'])
                        ]

                        for label, p, re in metrica_data:
                            # Evitamos errores si el dato real es None/Null
                            real_val = re if pd.notnull(re) else 0
                            check = "✅" if real_val >= p else "❌"
                            color = "#27ae60" if real_val >= p else "#c0392b"
                            
                            st.markdown(f"""
                            <div style="border-left: 5px solid {color}; padding: 10px; margin-bottom: 8px; background-color: #1e2129; border-radius: 5px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: gray; font-size: 0.8rem;">{label.upper()}</span>
                                    <span>{check}</span>
                                </div>
                                <span style="font-size: 1.1rem;">IA: <b>{p:.1f}</b> | Real: <b>{real_val}</b></span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Resultado no disponible para este partido específico.")
    else:
        st.info("No hay datos históricos para auditar. Ejecuta el actualizador de base de datos.")

if menu == "BetBuilder":
    st.title("🛠️ BetBuilder Pro")
    
    # --- FUNCIONES DE SOPORTE LOCALES (Para evitar que falle si no están definidas) ---
    def get_stats_local(team, conexion):
        try:
            query = f"SELECT * FROM historial_multiliga_ml WHERE HomeTeam='{team}' OR AwayTeam='{team}' ORDER BY Date DESC LIMIT 10"
            df = pd.read_sql(query, conexion)
            if df.empty: return None
            return {
                'FTHG': df['FTHG'].mean(), 'FTAG': df['FTAG'].mean(),
                'HST': df['HST'].mean() if 'HST' in df else 4.0,
                'AST': df['AST'].mean() if 'AST' in df else 3.5,
                'HC': df['HC'].mean() if 'HC' in df else 5.0,
                'AC': df['AC'].mean() if 'AC' in df else 4.5,
                'HY': df['HY'].mean() if 'HY' in df else 2.0,
                'AY': df['AY'].mean() if 'AY' in df else 2.1
            }
        except: return None

    try:
        # 1. Verificar conexión a DB
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        
        # 2. Selectores de Equipo
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            h_team = st.selectbox("🏠 Local:", equipos_db, key="bb_h")
        with col_sel2:
            a_team = st.selectbox("🚀 Visitante:", equipos_db, key="bb_a")

        # 3. Cargar estadísticas para el cálculo
        stats_h = get_stats_local(h_team, conn)
        stats_a = get_stats_local(a_team, conn)

        if stats_h and stats_a:
            st.divider()
            c_config, c_res = st.columns([1, 1.5])

            with c_config:
                st.subheader("🎯 Configurar Pick")
                
                # --- GOLES ---
                st.markdown("##### **Mercado Goles**")
                linea_g = st.selectbox("Línea Goles:", [0.5, 1.5, 2.5, 3.5, 4.5], index=2)
                tipo_g = st.radio("Predicción Goles:", ["Over", "Under"], horizontal=True)

                # --- CÓRNERS ---
                st.markdown("##### **Mercado Córners**")
                linea_c = st.slider("Línea Córners:", 5.5, 15.5, 8.5, 1.0)
                tipo_c = st.radio("Predicción Córners:", ["Over", "Under"], horizontal=True, key="c_radio_bb")

                # --- RESULTADO ---
                st.markdown("##### **Doble Oportunidad**")
                mercado_res = st.multiselect("Opciones:", ["Local", "Empate", "Visita"], default=["Local", "Empate"])

                # --- TIROS ---
                st.markdown("##### **Tiros a Puerta**")
                linea_t = st.number_input("Mínimo Tiros Totales:", 4, 20, 8)

            with c_res:
                st.subheader("📊 Probabilidades")
                
                # CÁLCULOS (Matemática Sigmoide)
                # Goles
                prom_g = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
                prob_g = 1 / (1 + np.exp(-(prom_g - linea_g)))
                if tipo_g == "Under": prob_g = 1 - prob_g
                
                # Córners
                prom_c = stats_h['HC'] + stats_a['AC']
                prob_c = 1 / (1 + np.exp(-(prom_c - linea_c)))
                if tipo_c == "Under": prob_c = 1 - prob_c
                
                # Tiros
                prom_t = stats_h['HST'] + stats_a['AST']
                prob_t = 1 / (1 + np.exp(-(prom_t - linea_t)))

                # Resultado (Doble Oportunidad Simplificada si no carga el modelo)
                prob_res = len(mercado_res) * 0.30 # Estimación base si falla IA
                try:
                    m_ia = cargar_modelo()
                    if m_ia:
                        in_ia = [[stats_h['FTHG'], stats_h['FTAG'], 10, 10, stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'], stats_h['HY'], stats_h['AY']]]
                        p_ia = m_ia.predict_proba(in_ia)[0]
                        prob_res = 0
                        if "Local" in mercado_res: prob_res += p_ia[2]
                        if "Empate" in mercado_res: prob_res += p_ia[1]
                        if "Visita" in mercado_res: prob_res += p_ia[0]
                except: pass

                # --- MOSTRAR MÉTRICAS ---
                g1, g2 = st.columns(2)
                g1.metric(f"⚽ {tipo_g} {linea_g}", f"{prob_g:.1%}")
                g2.metric(f"🚩 {tipo_c} {linea_c}", f"{prob_c:.1%}")
                
                g3, g4 = st.columns(2)
                g3.metric("🛡️ Doble Oportunidad", f"{prob_res:.1%}")
                g4.metric(f"🎯 Tiros > {linea_t}", f"{prob_t:.1%}")

                # --- CUOTA JUSTA ---
                st.divider()
                prob_total = prob_g * prob_c * prob_res * prob_t
                cuota = 1 / prob_total if prob_total > 0.01 else 100.0
                
                st.success(f"🔥 **Probabilidad del Pick:** {prob_total:.1%}")
                st.info(f"💰 **Cuota Justa (Valor):** {cuota:.2f}")
                
                if prob_total > 0.40: st.balloons()

        else:
            st.error("❌ No hay datos históricos suficientes para estos equipos.")

    except Exception as e:
        st.error(f"⚠️ Error al cargar BetBuilder: {e}")

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
