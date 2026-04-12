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
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
        color: white;
    }
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    /* Estilo para que las métricas se vean bien en móvil */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURACIÓN DE GRÁFICOS FIJOS PARA MÓVIL ---
CONFIG_FIJA = {
    'staticPlot': False, 
    'scrollZoom': False,
    'doubleClick': 'reset',
    'displayModeBar': False,
    'showAxisDragHandles': False,
    'showAxisRangeEntryBoxes': False
}

# --- FUNCIONES AUXILIARES ---
def corregir_nombre_equipo(nombre_api, lista_db):
    if not lista_db: return nombre_api
    nombre_api = nombre_api.strip()
    if nombre_api in lista_db: return nombre_api
    mejor_match, score = process.extractOne(nombre_api, lista_db, scorer=fuzz.token_set_ratio)
    return mejor_match if score > 50 else nombre_api

def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def get_recent_stats(equipo, conn):
    q = f"""
        SELECT "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY" 
        FROM historial_multiliga_ml 
        WHERE HomeTeam='{equipo}' OR AwayTeam='{equipo}' 
        ORDER BY Date DESC LIMIT 5
    """
    res = pd.read_sql(q, conn)
    return res.mean().fillna(0)

# --- CONEXIÓN Y DATOS ---
conn = sqlite3.connect(DB_NAME)

try:
    equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
except:
    equipos_db = []

st.sidebar.title("⚽ Panel de Control")
st.sidebar.markdown("---")

try:
    df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
    nombres_ligas = {'PL': 'Premier League', 'PD': 'La Liga', 'BL1': 'Bundesliga', 'SA': 'Serie A', 'FL1': 'Ligue 1'}
    df_jornada['LeagueName'] = df_jornada['League'].map(nombres_ligas)
    
    liga_sel = st.sidebar.selectbox("Seleccionar Liga:", df_jornada['LeagueName'].unique())
    partidos_filtrados = df_jornada[df_jornada['LeagueName'] == liga_sel]
    
    partido_texto = st.sidebar.selectbox(
        "Seleccionar Partido:", 
        partidos_filtrados['Local'] + " vs " + partidos_filtrados['Visita']
    )
    
    home_raw, away_raw = partido_texto.split(" vs ")
    home_team = corregir_nombre_equipo(home_raw.strip(), equipos_db)
    away_team = corregir_nombre_equipo(away_raw.strip(), equipos_db)

except Exception as e:
    st.error("❌ No se encontró la tabla de la jornada.")
    st.stop()

# --- TÍTULO PRINCIPAL ---
st.title(f"{home_team} vs {away_team}")
if home_raw != home_team or away_raw != away_team:
    st.caption(f"🔧 Match: {home_raw}➔{home_team} | {away_raw}➔{away_team}")

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("📊 Historial H2H")
    q_h2h = f"""
        SELECT Date, HomeTeam as L, AwayTeam as V, 
               FTHG as [GL], FTAG as [GV], FTR as R 
        FROM historial_multiliga_ml 
        WHERE (HomeTeam='{home_team}' AND AwayTeam='{away_team}') 
           OR (HomeTeam='{away_team}' AND AwayTeam='{home_team}') 
        ORDER BY Date DESC LIMIT 5
    """
    df_h2h = pd.read_sql(q_h2h, conn)
    
    if not df_h2h.empty:
        df_h2h['Date'] = pd.to_datetime(df_h2h['Date']).dt.strftime('%d/%m/%y')
        st.dataframe(df_h2h, use_container_width=True, hide_index=True)
    else:
        st.info(f"Sin registros previos.")

    st.subheader("📈 Tendencia de Goles")
    q_trend = f"""
        SELECT FTHG as [Local], FTAG as [Visita] FROM historial_multiliga_ml 
        WHERE HomeTeam='{home_team}' OR AwayTeam='{home_team}' 
        ORDER BY Date DESC LIMIT 10
    """
    df_trend = pd.read_sql(q_trend, conn)
    st.line_chart(df_trend)

with col2:
    st.subheader("🤖 IA Predictiva")
    model = cargar_modelo()
    
    if model:
        stats_h = get_recent_stats(home_team, conn)
        stats_a = get_recent_stats(away_team, conn)

        input_data = [[
            stats_h['FTHG'], stats_h['FTAG'], stats_h['HS'], stats_h['AS'],
            stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'],
            stats_h['HY'], stats_h['AY']
        ]]

        prob_ia = model.predict_proba(input_data)[0]
        p_vis, p_emp, p_loc = prob_ia[0], prob_ia[1], prob_ia[2]

        fig_pie = px.pie(
            values=[p_loc, p_emp, p_vis], 
            names=['Local', 'Empate', 'Visita'],
            color=['Local', 'Empate', 'Visita'],
            color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'},
            hole=0.45
        )
        fig_pie.update_layout(dragmode=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)

        st.markdown("---")
        # MERCADO DE GOLES DESTACADO
        promedio_goles = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
        prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))
        
        c_g1, c_g2 = st.columns(2)
        c_g1.metric("Goles Exp.", f"{promedio_goles:.2f}")
        c_g2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
        st.progress(prob_over)

        # PROYECCIONES DE JUEGO
        st.markdown("#### **🎯 Proyecciones**")
        cp1, cp2 = st.columns(2)
        with cp1:
            st.caption("Tiros al Arco")
            st.write(f"L: **{stats_h['HST']:.1f}** | V: **{stats_a['AST']:.1f}**")
        with cp2:
            st.caption("Córners")
            st.write(f"L: **{stats_h['HC']:.1f}** | V: **{stats_a['AC']:.1f}**")

# --- SECCIÓN DE DISCIPLINA ---
st.divider()
st.subheader("🟨 Disciplina y Tarjetas")
cd1, cd2 = st.columns(2)

with cd1:
    st.markdown("#### **Media Amarillas**")
    m1, m2 = st.columns(2)
    m1.metric(f"{home_team[:10]}", f"{stats_h['HY']:.1f}")
    m2.metric(f"{away_team[:10]}", f"{stats_a['AY']:.1f}")

with cd2:
    q_cards = f"""
        SELECT Date, (HY + AY) as Total
        FROM historial_multiliga_ml 
        WHERE (HomeTeam = '{home_team}' AND AwayTeam = '{away_team}') 
           OR (HomeTeam = '{away_team}' AND AwayTeam = '{home_team}')
        ORDER BY Date DESC LIMIT 5
    """
    df_cards = pd.read_sql(q_cards, conn)
    if not df_cards.empty:
        fig_cards = px.bar(df_cards, x='Date', y='Total', color_discrete_sequence=['#f1c40f'])
        fig_cards.update_layout(dragmode=False, xaxis={'fixedrange': True}, yaxis={'fixedrange': True})
        st.plotly_chart(fig_cards, use_container_width=True, config=CONFIG_FIJA)
    else:
        st.info("Sin datos de tarjetas.")

# --- SECCIÓN INFERIOR: PATRONES ---
st.divider()
st.subheader("🧬 Peso de Variables (IA)")
if model:
    importancia = pd.DataFrame({
        'Feature': ['Goles L', 'Goles V', 'Tiros L', 'Tiros V', 'ST L', 'ST V', 'Córners L', 'Córners V', 'Amarillas L', 'Amarillas V'],
        'Weight': model.feature_importances_
    }).sort_values(by='Weight', ascending=True)
    
    fig_imp = px.bar(importancia, x='Weight', y='Feature', orientation='h', color_discrete_sequence=['#3498db'])
    fig_imp.update_layout(dragmode=False, xaxis={'fixedrange': True}, yaxis={'fixedrange': True})
    st.plotly_chart(fig_imp, use_container_width=True, config=CONFIG_FIJA)

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
