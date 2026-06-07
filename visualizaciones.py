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
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "Portafolio de Picks", "Mundial 2026"])
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
            archivos_csv = list(directorio_odds.glob("*.csv")) if directorio_odds.exists() else []
            
            df_master_odds = pd.DataFrame()
            fechas_disponibles = []

            if archivos_csv:
                lista_dfs = []
                for f in archivos_csv:
                    try:
                        df_temp = pd.read_csv(f)
                        if df_temp.empty:
                            continue
                        
                        # 🔍 1. DETECTAR FECHA EN EL NOMBRE DEL ARCHIVO (Regex Infalible)
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

                        # 🔍 2. EVITAR CONFLICTOS DE MAYÚSCULAS/MINÚSCULAS
                        df_temp.columns = [c.lower() for c in df_temp.columns]

                        # Mapeo de compatibilidad por si cambiaste nombres de columnas en el tiempo
                        if 'hometeam' in df_temp.columns and 'home' not in df_temp.columns:
                            df_temp = df_temp.rename(columns={'hometeam': 'home'})
                        if 'awayteam' in df_temp.columns and 'away' not in df_temp.columns:
                            df_temp = df_temp.rename(columns={'awayteam': 'away'})

                        # 🔍 3. NORMALIZAR O INYECTAR COLUMNA DE TIEMPO
                        if 'inicio_local' not in df_temp.columns or df_temp['inicio_local'].isna().all():
                            df_temp['inicio_local'] = fecha_fallback + " 12:00"
                        else:
                            df_temp['inicio_local'] = df_temp['inicio_local'].fillna(fecha_fallback + " 12:00")

                        lista_dfs.append(df_temp)
                    except Exception as e:
                        st.sidebar.error(f"⚠️ Error en archivo {f.name}: {e}")
                
                if lista_dfs:
                    df_master_odds = pd.concat(lista_dfs, ignore_index=True)
                    df_master_odds = df_master_odds.drop_duplicates(subset=['home', 'away', 'inicio_local'])
                    
                    # Cortamos estrictamente los primeros 10 caracteres (YYYY-MM-DD)
                    df_master_odds['Fecha_Match'] = df_master_odds['inicio_local'].astype(str).str.strip().str.slice(0, 10)
                    
                    # Filtramos filas que mantengan la estructura de fecha real
                    df_master_odds = df_master_odds[df_master_odds['Fecha_Match'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]
                    fechas_disponibles = sorted(df_master_odds['Fecha_Match'].unique())

            # --- INTERFAZ VISUAL DE CONTROL ---
            c1, c2 = st.columns(2)
            with c1:
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
                    # Filtramos las cuotas maestras para quedarnos solo con el día elegido
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
        # 1. BOTÓN MÁGICO: Liquidar apuestas pendientes
        if st.button("⚖️ Liquidar Apuestas Pendientes", type="primary"):
            df_pendientes = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente'", conn)
            liquidadas = 0
            beneficio_reciente = 0.0
            stake_reciente = 0.0
            
            import re
            from thefuzz import process
            from datetime import datetime, timedelta
            
            for _, pick in df_pendientes.iterrows():
                # 🛡️ 1. Margen de Fecha (Buscamos el partido entre ayer, hoy y mañana)
                try:
                    fecha_dt = datetime.strptime(pick['Date'], '%Y-%m-%d')
                    fecha_inicio = (fecha_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                    fecha_fin = (fecha_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception:
                    fecha_inicio = pick['Date']
                    fecha_fin = pick['Date']

                q_res = f"SELECT * FROM historial_multiliga_ml WHERE Date BETWEEN '{fecha_inicio}' AND '{fecha_fin}'"
                res_real = pd.read_sql(q_res, conn)
                
                if not res_real.empty:
                    # 🛡️ 2. Fuzzy Matching para Nombres de Equipos (M'gladbach == Borussia Monchengladbach)
                    equipos_posibles = res_real['HomeTeam'].tolist()
                    match_fuzz = process.extractOne(pick['HomeTeam'], equipos_posibles)
                    
                    # Si hay coincidencia de al menos 75% de similitud de texto
                    if match_fuzz and match_fuzz[1] >= 75:
                        equipo_db_real = match_fuzz[0]
                        row = res_real[res_real['HomeTeam'] == equipo_db_real].iloc[0]
                        
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
                        
                        liquidadas += 1
                        beneficio_reciente += beneficio
                        stake_reciente += pick['Stake']
            
            conn.commit()
            
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
elif menu == "Mundial 2026":
    st.title("🏆 Oráculo Mundial 2026")
    st.markdown("Proyección oficial basada en formato FIFA 48 selecciones.")

    st.markdown("""
    <style>
    .wc-group-header {
        font-size: 1rem; font-weight: 700; color: #fff;
        background: #2c2f3a; padding: 6px 10px;
        border-radius: 6px 6px 0 0; margin-bottom: 0;
    }
    .wc-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 2px; }
    .wc-table th { background: #1a1d26; color: #888; padding: 4px 7px; text-align: left; font-weight: 600; }
    .wc-table td { padding: 5px 7px; border-bottom: 1px solid #2a2d3a; }
    .wc-pos1  { background: #0d3320; }
    .wc-pos2  { background: #0d3320; }
    .wc-pos3  { background: #1a3310; }
    .wc-pos4  { background: #1e2129; }
    .wc-match { font-size: 0.8rem; padding: 3px 0; color: #ccc; border-bottom: 1px solid #2a2d3a; }
    .wc-match:last-child { border-bottom: none; }
    .wc-winner { color: #2ecc71; font-weight: 700; }
    .bracket-round-title {
        font-size: 0.78rem; font-weight: 700; color: #888;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; text-align: center;
    }
    .bracket-card {
        background: #1e2129; border-radius: 8px;
        border-left: 3px solid #2ecc71;
        padding: 8px 12px; margin-bottom: 8px;
    }
    .bracket-label { font-size: 0.68rem; color: #666; margin-bottom: 3px; }
    .bracket-score { font-size: 0.92rem; }
    .bracket-card.final { border-left-color: #f1c40f; }
    .bracket-card.champion { border-left-color: #f39c12; background: #1a1700; }
    .champion-name { font-size: 1.3rem; font-weight: 700; color: #f1c40f; text-align: center; margin: 6px 0 2px; }
    .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
    </style>
    """, unsafe_allow_html=True)

    try:
        modelo_wc  = joblib.load('modelo_selecciones_rf.pkl')
        encoder_wc = joblib.load('encoder_equipos_selecciones.pkl')
        df_fixture_wc = pd.read_sql("SELECT * FROM fixture_mundial", conn)

        # ── Normalizar columna Grupo ──────────────────────────────────────────
        col_grupo = next(
            (c for c in df_fixture_wc.columns
             if c.strip().lower() in ('grupo', 'group', 'stage', 'fase')),
            None
        )
        col_home = next(
            (c for c in df_fixture_wc.columns
             if c.strip().lower() in ('hometeam', 'home', 'local', 'home_team')),
            None
        )
        col_away = next(
            (c for c in df_fixture_wc.columns
             if c.strip().lower() in ('awayteam', 'away', 'visita', 'away_team')),
            None
        )

        if not all([col_grupo, col_home, col_away]):
            st.error(f"Columnas detectadas en fixture_mundial: {list(df_fixture_wc.columns)}")
            st.stop()

        df_fixture_wc['_grupo'] = df_fixture_wc[col_grupo].astype(str).str.strip().str.upper()
        df_fixture_wc['_home']  = df_fixture_wc[col_home].astype(str).str.strip()
        df_fixture_wc['_away']  = df_fixture_wc[col_away].astype(str).str.strip()

        # ── Motor de predicción ───────────────────────────────────────────────
        def predecir_wc(h, a):
            """Devuelve dict con pts, goles y prob de cada equipo."""
            hst, hc, ast, ac = 4.0, 5.0, 3.5, 4.0
            classes = list(encoder_wc.classes_)
            if h in classes and a in classes:
                h_c = encoder_wc.transform([h])[0]
                a_c = encoder_wc.transform([a])[0]
                probs = modelo_wc.predict_proba(
                    pd.DataFrame([[h_c, a_c, hst, ast, hc, ac]],
                                 columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC'])
                )[0]
                # orden: A=0, D=1, H=2
                p_h, p_d, p_a = probs[2], probs[1], probs[0]
            else:
                p_h, p_d, p_a = 0.34, 0.32, 0.34
            pts_h, pts_a = (3, 0) if p_h > p_a else ((0, 3) if p_a > p_h else (1, 1))
            ganador = h if p_h >= p_a else a          # en KO no hay empate → gana el más probable
            return {
                "Pts_H": pts_h, "Pts_A": pts_a,
                "GH": max(0, round(p_h * 2.2)),
                "GA": max(0, round(p_a * 2.2)),
                "p_h": p_h, "p_a": p_a,
                "Ganador": ganador
            }

        # ═════════════════════════════════════════════════════════════════════
        #  CALCULAR GRUPOS — se hace ANTES de los tabs para alimentar tab_f
        # ═════════════════════════════════════════════════════════════════════
        grupos_keys = ['A','B','C','D','E','F','G','H','I','J','K','L']
        posiciones   = {}          # "1A" → equipo
        terceros_all = []          # (pts, goles_favor, grupo_letra, equipo)
        grupos_data  = {}          # letra → {tabla_df, partidos_list}

        for letra in grupos_keys:
            df_g = df_fixture_wc[df_fixture_wc['_grupo'] == letra]
            if df_g.empty:
                # Fallback: buscar "Grupo A" o "Group A"
                df_g = df_fixture_wc[df_fixture_wc['_grupo'].str.contains(f'\\b{letra}\\b', regex=True, na=False)]
            if df_g.empty:
                continue

            tabla  = {}     # equipo → {pts, gf, gc}
            partidos = []   # lista de dicts para mostrar

            for _, p in df_g.iterrows():
                h, a = p['_home'], p['_away']
                res = predecir_wc(h, a)

                for equipo in [h, a]:
                    if equipo not in tabla:
                        tabla[equipo] = {'Pts': 0, 'GF': 0, 'GC': 0, 'PJ': 0}

                tabla[h]['Pts'] += res['Pts_H']
                tabla[h]['GF']  += res['GH']
                tabla[h]['GC']  += res['GA']
                tabla[h]['PJ']  += 1
                tabla[a]['Pts'] += res['Pts_A']
                tabla[a]['GF']  += res['GA']
                tabla[a]['GC']  += res['GH']
                tabla[a]['PJ']  += 1
                partidos.append({'H': h, 'A': a, 'GH': res['GH'], 'GA': res['GA'],
                                 'Ganador': res['Ganador']})

            df_t = pd.DataFrame([
                {'Selección': eq, 'PJ': v['PJ'], 'Pts': v['Pts'],
                 'GF': v['GF'], 'GC': v['GC'], 'DG': v['GF'] - v['GC']}
                for eq, v in tabla.items()
            ]).sort_values(['Pts', 'DG', 'GF'], ascending=False).reset_index(drop=True)

            # Guardar posiciones 1° y 2°
            for rank_i, row in df_t.iterrows():
                posiciones[f"{rank_i+1}{letra}"] = row['Selección']
            # Guardar 3° para ranking de mejores terceros
            if len(df_t) >= 3:
                t = df_t.iloc[2]
                terceros_all.append((t['Pts'], t['DG'], t['GF'], letra, t['Selección']))

            grupos_data[letra] = {'tabla': df_t, 'partidos': partidos}

        # ── Elegir los 8 mejores terceros ─────────────────────────────────────
        terceros_all.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        mejores_terceros_set  = {x[4] for x in terceros_all[:8]}
        tercero_por_grupo     = {x[3]: x[4] for x in terceros_all}   # letra → equipo 3°

        def get_pos(clave):
            return posiciones.get(clave, f"({clave})")

        def mejor_3ro_de(grupos_str):
            """Devuelve el mejor 3° entre los grupos indicados (ej: 'A/B/C')."""
            candidatos = [g.strip() for g in grupos_str.split('/')]
            for pts, dg, gf, grp, equipo in terceros_all:
                if grp in candidatos and equipo in mejores_terceros_set:
                    return equipo
            return f"3°({grupos_str})"

        # ═════════════════════════════════════════════════════════════════════
        tab_g, tab_f = st.tabs(["📊 Grupos", "⚔️ Ruta a la Copa"])
        # ═════════════════════════════════════════════════════════════════════

        with tab_g:
            # Leyenda
            st.markdown("""
            <div style="margin-bottom:14px; font-size:0.8rem; color:#aaa;">
              <span style="background:#0d3320; padding:2px 8px; border-radius:4px; margin-right:6px;">■ 1° / 2°</span> Clasificado directo &nbsp;
              <span style="background:#1a3310; padding:2px 8px; border-radius:4px; margin-right:6px;">■ 3°</span> Posible mejor tercero
            </div>
            """, unsafe_allow_html=True)

            for fila_inicio in range(0, len(grupos_keys), 3):
                lote = grupos_keys[fila_inicio:fila_inicio+3]
                cols = st.columns(3)

                for col_idx, letra in enumerate(lote):
                    if letra not in grupos_data:
                        continue
                    df_t    = grupos_data[letra]['tabla']
                    partidos = grupos_data[letra]['partidos']

                    with cols[col_idx]:
                        # ── Tabla del grupo ──────────────────────────────
                        filas_html = ""
                        for rank_i, row in df_t.iterrows():
                            if rank_i < 2:
                                css = "wc-pos1"
                                ico = "🟢"
                            elif rank_i == 2 and row['Selección'] in mejores_terceros_set:
                                css = "wc-pos3"
                                ico = "🟡"
                            elif rank_i == 2:
                                css = "wc-pos3"
                                ico = "⚪"
                            else:
                                css = "wc-pos4"
                                ico = ""
                            dg_str = f"+{int(row['DG'])}" if row['DG'] > 0 else str(int(row['DG']))
                            filas_html += (
                                f'<tr class="{css}">'
                                f'<td style="color:#888">{rank_i+1}</td>'
                                f'<td>{ico} {row["Selección"]}</td>'
                                f'<td style="text-align:center"><b>{int(row["Pts"])}</b></td>'
                                f'<td style="text-align:center;color:#888">{dg_str}</td>'
                                f'</tr>'
                            )

                        st.markdown(
                            f'<div class="wc-group-header">Grupo {letra}</div>'
                            f'<table class="wc-table">'
                            f'<thead><tr>'
                            f'<th>#</th><th>Selección</th><th style="text-align:center">Pts</th><th style="text-align:center">DG</th>'
                            f'</tr></thead>'
                            f'<tbody>{filas_html}</tbody>'
                            f'</table>',
                            unsafe_allow_html=True
                        )

                        # ── Resultados simulados ─────────────────────────
                        match_html = '<div style="background:#161920;border-radius:0 0 6px 6px;padding:6px 10px;margin-bottom:14px;">'
                        for m in partidos:
                            if m['GH'] > m['GA']:
                                score = f'<span class="wc-winner">{m["H"]} {m["GH"]}</span>–{m["GA"]} {m["A"]}'
                            elif m['GA'] > m['GH']:
                                score = f'{m["H"]} {m["GH"]}–<span class="wc-winner">{m["GA"]} {m["A"]}</span>'
                            else:
                                score = f'{m["H"]} <span style="color:#f1c40f">{m["GH"]}–{m["GA"]}</span> {m["A"]}'
                            match_html += f'<div class="wc-match">{score}</div>'
                        match_html += '</div>'
                        st.markdown(match_html, unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        with tab_f:
        # ═════════════════════════════════════════════════════════════════════

            # ── RONDA DE 32 (16 partidos) ─────────────────────────────────
            # Cruces oficiales FIFA 2026 con los mejores terceros según tabla oficial
            cruces_r32_def = [
                ("2A",  "2B",              "2°A vs 2°B"),
                ("1C",  "2F",              "1°C vs 2°F"),
                ("1E",  "3_A/B/C/D/F",     "1°E vs 3°(A/B/C/D/F)"),
                ("1F",  "2C",              "1°F vs 2°C"),
                ("2E",  "2I",              "2°E vs 2°I"),
                ("1I",  "3_C/D/F/G/H",     "1°I vs 3°(C/D/F/G/H)"),
                ("1A",  "3_C/E/F/H/I",     "1°A vs 3°(C/E/F/H/I)"),
                ("1L",  "3_E/H/I/J/K",     "1°L vs 3°(E/H/I/J/K)"),
                ("1G",  "3_A/E/H/I/J",     "1°G vs 3°(A/E/H/I/J)"),
                ("1D",  "3_B/E/F/I/J",     "1°D vs 3°(B/E/F/I/J)"),
                ("1H",  "2J",              "1°H vs 2°J"),
                ("2K",  "2L",              "2°K vs 2°L"),
                ("1B",  "3_E/F/G/I/J",     "1°B vs 3°(E/F/G/I/J)"),
                ("2D",  "2G",              "2°D vs 2°G"),
                ("1K",  "2H",              "1°K vs 2°H"),
                ("1J",  "2E",              "1°J vs 2°E"),
            ]

            def resolver_equipo(clave):
                if clave.startswith("3_"):
                    return mejor_3ro_de(clave[2:])
                return get_pos(clave)

            def simular_ko(h, a, etiqueta=""):
                """Simula un partido KO (sin empate). Devuelve dict."""
                res = predecir_wc(h, a)
                # Si empate → gana el de mayor prob
                ganador = h if res['p_h'] >= res['p_a'] else a
                # En KO ajustar marcador para que no haya empate en la línea
                gh, ga = res['GH'], res['GA']
                if gh == ga:
                    if res['p_h'] >= res['p_a']: gh += 1
                    else: ga += 1
                return {'H': h, 'A': a, 'GH': gh, 'GA': ga,
                        'Ganador': ganador, 'Label': etiqueta}

            def render_partido(m, ronda_css=""):
                h, a, gh, ga = m['H'], m['A'], m['GH'], m['GA']
                lbl = m.get('Label', '')
                if gh > ga:
                    score_html = (f'<span class="wc-winner">{h}</span> '
                                  f'<span style="color:#aaa">{gh}–{ga}</span> {a}')
                elif ga > gh:
                    score_html = (f'{h} <span style="color:#aaa">{gh}–{ga}</span> '
                                  f'<span class="wc-winner">{a}</span>')
                else:
                    score_html = f'{h} <span style="color:#f1c40f">{gh}–{ga}</span> {a}'
                return (
                    f'<div class="bracket-card {ronda_css}">'
                    f'  <div class="bracket-label">{lbl}</div>'
                    f'  <div class="bracket-score">{score_html}</div>'
                    f'</div>'
                )

            # ── Simular R32 ───────────────────────────────────────────────
            r32_resultados = []
            for c_h, c_a, etq in cruces_r32_def:
                eq_h = resolver_equipo(c_h)
                eq_a = resolver_equipo(c_a)
                r32_resultados.append(simular_ko(eq_h, eq_a, etq))

            # Ganadores R32 → R16
            ganadores_r32 = [m['Ganador'] for m in r32_resultados]

            # ── Simular R16 (8 partidos) ──────────────────────────────────
            # Cruces por posición del bracket (pareados por bracket)
            pares_r16 = [
                (0,  1,  "Partido 1 bracket"),
                (2,  3,  "Partido 2 bracket"),
                (4,  5,  "Partido 3 bracket"),
                (6,  7,  "Partido 4 bracket"),
                (8,  9,  "Partido 5 bracket"),
                (10, 11, "Partido 6 bracket"),
                (12, 13, "Partido 7 bracket"),
                (14, 15, "Partido 8 bracket"),
            ]
            r16_resultados = []
            for i_h, i_a, etq in pares_r16:
                h = ganadores_r32[i_h] if i_h < len(ganadores_r32) else "TBD"
                a = ganadores_r32[i_a] if i_a < len(ganadores_r32) else "TBD"
                r16_resultados.append(simular_ko(h, a, etq))

            ganadores_r16 = [m['Ganador'] for m in r16_resultados]

            # ── Simular Cuartos (4 partidos) ──────────────────────────────
            qf_resultados = []
            for k in range(0, 8, 2):
                h = ganadores_r16[k]   if k   < len(ganadores_r16) else "TBD"
                a = ganadores_r16[k+1] if k+1 < len(ganadores_r16) else "TBD"
                qf_resultados.append(simular_ko(h, a, f"Cuartos {k//2+1}"))

            ganadores_qf = [m['Ganador'] for m in qf_resultados]

            # ── Simular Semis (2 partidos) ────────────────────────────────
            sf_resultados = [
                simular_ko(ganadores_qf[0], ganadores_qf[1], "Semifinal 1"),
                simular_ko(ganadores_qf[2], ganadores_qf[3], "Semifinal 2"),
            ] if len(ganadores_qf) >= 4 else []

            ganadores_sf   = [m['Ganador'] for m in sf_resultados]
            perdedores_sf  = [
                (sf_resultados[0]['A'] if sf_resultados[0]['Ganador'] == sf_resultados[0]['H']
                 else sf_resultados[0]['H']),
                (sf_resultados[1]['A'] if sf_resultados[1]['Ganador'] == sf_resultados[1]['H']
                 else sf_resultados[1]['H']),
            ] if sf_resultados else []

            # ── 3er puesto y Final ────────────────────────────────────────
            tercer_puesto = (simular_ko(perdedores_sf[0], perdedores_sf[1], "3° Puesto")
                             if len(perdedores_sf) == 2 else None)
            final = (simular_ko(ganadores_sf[0], ganadores_sf[1], "⚽ FINAL")
                     if len(ganadores_sf) == 2 else None)
            campeon = final['Ganador'] if final else "—"

            # ══════════════════════════════════════════════════════════════
            #  RENDER BRACKET
            # ══════════════════════════════════════════════════════════════

            # ── Campeón arriba ────────────────────────────────────────────
            if final:
                st.markdown(
                    f'<div class="bracket-card champion" style="max-width:360px;margin:0 auto 20px;">'
                    f'  <div style="text-align:center;color:#888;font-size:0.75rem;">🏆 CAMPEÓN PROYECTADO</div>'
                    f'  <div class="champion-name">🏆 {campeon}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown("---")

            # ── Final + 3° puesto ─────────────────────────────────────────
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown('<p class="bracket-round-title">⚽ Final</p>', unsafe_allow_html=True)
                if final:
                    st.markdown(render_partido(final, "final"), unsafe_allow_html=True)
            with col_f2:
                st.markdown('<p class="bracket-round-title">🥉 3° Puesto</p>', unsafe_allow_html=True)
                if tercer_puesto:
                    st.markdown(render_partido(tercer_puesto), unsafe_allow_html=True)

            st.markdown("---")

            # ── Semifinales ───────────────────────────────────────────────
            st.markdown('<p class="bracket-round-title">Semifinales</p>', unsafe_allow_html=True)
            cols_sf = st.columns(2)
            for idx, m in enumerate(sf_resultados):
                with cols_sf[idx]:
                    st.markdown(render_partido(m), unsafe_allow_html=True)

            st.markdown("---")

            # ── Cuartos ───────────────────────────────────────────────────
            st.markdown('<p class="bracket-round-title">Cuartos de Final</p>', unsafe_allow_html=True)
            cols_qf = st.columns(4)
            for idx, m in enumerate(qf_resultados):
                with cols_qf[idx]:
                    st.markdown(render_partido(m), unsafe_allow_html=True)

            st.markdown("---")

            # ── Ronda de 16 ──────────────────────────────────────────────
            st.markdown('<p class="bracket-round-title">Ronda de 16</p>', unsafe_allow_html=True)
            cols_r16 = st.columns(4)
            for idx, m in enumerate(r16_resultados):
                with cols_r16[idx % 4]:
                    st.markdown(render_partido(m), unsafe_allow_html=True)

            st.markdown("---")

            # ── Ronda de 32 ──────────────────────────────────────────────
            with st.expander("▶ Ver Ronda de 32 (16 partidos)"):
                cols_r32 = st.columns(4)
                for idx, m in enumerate(r32_resultados):
                    with cols_r32[idx % 4]:
                        st.markdown(render_partido(m), unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error(f"Error en Oráculo Mundial: {e}")
        st.code(traceback.format_exc())
conn.close()