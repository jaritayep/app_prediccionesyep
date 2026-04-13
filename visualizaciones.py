import streamlit as st
import sqlite3
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os
from thefuzz import process, fuzz 

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
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        
        # 1. Normalizar fechas para evitar duplicados por horas
        df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.normalize()
        df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m')
        
        # 2. Lista de fechas única y limpia
        opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))
        dia_sel_str = st.sidebar.selectbox("📅 Seleccionar Día:", opciones_fecha)
        
        # 3. Filtrar partidos del día
        partidos_dia = df_jornada[df_jornada['Fecha_Display'] == dia_sel_str]
        partido_texto = st.sidebar.selectbox("🏟️ Partido:", partidos_dia['Local'] + " vs " + partidos_dia['Visita'])
        
        home_raw, away_raw = partido_texto.split(" vs ")
        home_team = corregir_nombre_equipo(home_raw, equipos_db)
        away_team = corregir_nombre_equipo(away_raw, equipos_db)

        # --- AQUÍ EMPIEZA EL DASHBOARD ---
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
            st.subheader("🤖 IA Predictiva")
            model = cargar_modelo()
            if model:
                stats_h, stats_a = get_recent_stats(home_team, conn), get_recent_stats(away_team, conn)
                input_data = [[stats_h['FTHG'], stats_h['FTAG'], stats_h['HS'], stats_h['AS'], stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'], stats_h['HY'], stats_h['AY']]]
                prob_ia = model.predict_proba(input_data)[0]
                
                fig_pie = px.pie(values=[prob_ia[2], prob_ia[1], prob_ia[0]], names=['Local', 'Empate', 'Visita'], color=['Local', 'Empate', 'Visita'], color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'}, hole=0.45)
                fig_pie.update_layout(dragmode=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)
                
                promedio_goles = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
                prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))
                c1, c2 = st.columns(2)
                c1.metric("Goles Exp.", f"{promedio_goles:.2f}")
                c2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                st.progress(prob_over)

                st.markdown("#### **🎯 Tiros y Córners**")
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

    except Exception as e:
        st.error(f"Error al cargar dashboard: {e}")

elif menu == "Auditoría (Resultados)":
    st.title("⚖️ Auditoría de Precisión")
    
    # 1. Cargar predicciones y resultados
    df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
    df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.normalize()
    
    # 2. Traer resultados reales de los últimos días
    df_reales = pd.read_sql("SELECT Date, HomeTeam, AwayTeam, FTHG, FTAG, (FTHG+FTAG) as GolesTotales FROM historial_multiliga_ml WHERE Date >= date('now', '-7 days')", conn)
    df_reales['Date'] = pd.to_datetime(df_reales['Date']).dt.normalize()

    # Filtro de fecha para auditar
    fechas_audit = list(dict.fromkeys(df_jornada['Date'].dt.strftime('%A %d/%m').tolist()))
    dia_audit = st.sidebar.selectbox("📅 Auditar Día:", fechas_audit)
    
    partidos_dia = df_jornada[df_jornada['Date'].dt.strftime('%A %d/%m') == dia_audit]
    
    # --- CÁLCULO DE TASA DE EFECTIVIDAD (Basado en Goles) ---
    aciertos_goles = 0
    total_con_resultado = 0
    
    # Primero hacemos un barrido para la métrica superior
    for _, f in partidos_dia.iterrows():
        res_real = df_reales[(df_reales['HomeTeam'] == f['Local'])].head(1)
        if not res_real.empty:
            total_con_resultado += 1
            # Para la tasa, calculamos si se cumplió el Over 2.5 (o la métrica que prefieras)
            if res_real.iloc[0]['GolesTotales'] >= 2.5: aciertos_goles += 1

    if total_con_resultado > 0:
        tasa = aciertos_goles / total_con_resultado
        st.metric("Cumplimiento Goles (Over 2.5)", f"{tasa:.1%}", delta=f"{aciertos_goles}/{total_con_resultado} Partidos")
    else:
        st.info("Esperando resultados finales para calcular efectividad...")

    st.divider()

    # --- LISTADO DETALLADO ---
    for _, fila in partidos_dia.iterrows():
        # Buscamos las estadísticas ponderadas que la IA usó para predecir
        stats_h = get_recent_stats(fila['Local'], conn)
        stats_a = get_recent_stats(fila['Visita'], conn)
        prediccion_goles = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
        
        # Buscamos el resultado real
        match_real = df_reales[df_reales['HomeTeam'] == fila['Local']].head(1)
        
        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"#### {fila['Local']} vs {fila['Visita']}")
            
            if not match_real.empty:
                r = match_real.iloc[0]
                goles_r = int(r['GolesTotales'])
                
                # LÓGICA DEL TICK O EQUIS
                # Si los goles reales igualan o superan la predicción esperada
                simbolo = "✅" if goles_r >= prediccion_goles else "❌"
                color = "green" if goles_r >= prediccion_goles else "red"
                
                c2.metric("IA Esperaba", f"{prediccion_goles:.2f}")
                c3.markdown(f"<div style='background-color: #1e2129; padding: 10px; border-radius: 10px; text-align: center; border: 1px solid {color};'>"
                            f"<span style='font-size: 0.8rem; color: gray;'>REALIDAD</span><br>"
                            f"<span style='font-size: 1.5rem;'>{goles_r} {simbolo}</span>"
                            f"</div>", unsafe_allow_html=True)
            else:
                c2.write("En juego / Pendiente ⏳")
            st.write("") # Espaciador

elif menu == "BetBuilder Simulator":
    st.title("🛠️ BetBuilder AI")
    col_bb1, col_bb2 = st.columns(2)
    picks = []
    with col_bb1:
        for i in range(st.number_input("Eventos:", 1, 5, 2)):
            picks.append(st.slider(f"Prob. Pick {i+1} (%)", 1, 99, 50, key=f"bb_{i}")/100)
    with col_bb2:
        res_prob = np.prod(picks)
        st.metric("Probabilidad Total", f"{res_prob:.1%}")
        st.metric("Cuota Justa", f"{1/res_prob:.2f}" if res_prob > 0 else "0")

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
