import streamlit as st
import sqlite3
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os
from thefuzz import process, fuzz 

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(layout="wide", page_title="AI Betting Lab Pro", page_icon="⚽")
DB_NAME = 'database_partidos.db'
MODEL_PATH = "modelo_ia.pkl"

# --- BLOQUE PARA APARIENCIA DE APP MÓVIL ---
st.markdown("""
    <style>
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); color: white; }
    footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stMetric { background-color: #1e2129; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

CONFIG_FIJA = {
    'staticPlot': False, 'scrollZoom': False, 'doubleClick': 'reset',
    'displayModeBar': False, 'showAxisDragHandles': False
}

# --- FUNCIONES AUXILIARES ---
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

# --- CONEXIÓN ---
conn = sqlite3.connect(DB_NAME)

# --- NAVEGACIÓN LATERAL ---
st.sidebar.title("⚽ Menú Principal")
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "BetBuilder Simulator"])
st.sidebar.markdown("---")

# ==========================================
# PÁGINA 1: ANÁLISIS DEL DÍA
# ==========================================
if menu == "Análisis del Día":
    try:
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        df_jornada['Date'] = pd.to_datetime(df_jornada['Date'])
        
        # Filtro de Fecha
        fechas_disponibles = sorted(df_jornada['Date'].unique())
        fechas_str = [d.strftime('%A %d/%m') for d in pd.to_datetime(fechas_disponibles)]
        dia_sel_str = st.sidebar.selectbox("📅 Seleccionar Día:", fechas_str)
        
        partidos_dia = df_jornada[df_jornada['Date'].dt.strftime('%A %d/%m') == dia_sel_str]
        partido_texto = st.sidebar.selectbox("🏟️ Seleccionar Partido:", partidos_dia['Local'] + " vs " + partidos_dia['Visita'])
        
        home_raw, away_raw = partido_texto.split(" vs ")
        home_team = corregir_nombre_equipo(home_raw, equipos_db)
        away_team = corregir_nombre_equipo(away_raw, equipos_db)

        st.title(f"{home_team} vs {away_team}")
        
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.subheader("📊 Historial H2H")
            q_h2h = f'SELECT Date, HomeTeam as L, AwayTeam as V, FTHG as [GL], FTAG as [GV], FTR as R FROM historial_multiliga_ml WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") ORDER BY Date DESC LIMIT 5'
            df_h2h = pd.read_sql(q_h2h, conn)
            if not df_h2h.empty:
                df_h2h['Date'] = pd.to_datetime(df_h2h['Date']).dt.strftime('%d/%m/%y')
                st.dataframe(df_h2h, use_container_width=True, hide_index=True)
            st.subheader("📈 Goles Recientes")
            q_trend = f'SELECT FTHG as [Local], FTAG as [Visita] FROM historial_multiliga_ml WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" ORDER BY Date DESC LIMIT 10'
            st.line_chart(pd.read_sql(q_trend, conn).iloc[::-1])

        with col2:
            st.subheader("🤖 IA (Weighted)")
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
                c_g1, c_g2 = st.columns(2)
                c_g1.metric("Goles Exp.", f"{promedio_goles:.2f}")
                c_g2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                st.progress(prob_over)
    except:
        st.error("Error al cargar la jornada actual.")

# ==========================================
# PÁGINA 2: AUDITORÍA
# ==========================================
elif menu == "Auditoría (Resultados)":
    st.title("⚖️ Auditoría de Resultados")
    st.info("Aquí ves si los últimos partidos registrados coinciden con lo que sabías.")
    q_audit = "SELECT Date, HomeTeam as Local, AwayTeam as Visita, FTHG, FTAG, FTR as Ganador FROM historial_multiliga_ml ORDER BY Date DESC LIMIT 20"
    st.dataframe(pd.read_sql(q_audit, conn), use_container_width=True, hide_index=True)

# ==========================================
# PÁGINA 3: BETBUILDER
# ==========================================
elif menu == "BetBuilder Simulator":
    st.title("🛠️ BetBuilder AI")
    st.markdown("Calcula la probabilidad real de tu combinada.")
    col_bb1, col_bb2 = st.columns(2)
    picks = []
    with col_bb1:
        num_p = st.number_input("Eventos:", 1, 6, 2)
        for i in range(num_p):
            p = st.slider(f"Probabilidad Pick {i+1} (%)", 1, 99, 50)
            picks.append(p/100)
    with col_bb2:
        res_prob = np.prod(picks)
        st.metric("Probabilidad Total", f"{res_prob:.1%}")
        st.metric("Cuota Justa (Fair Odds)", f"{1/res_prob:.2f}" if res_prob > 0 else "0")
        if res_prob < 0.15: st.error("Riesgo muy alto.")

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
