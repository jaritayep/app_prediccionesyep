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
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

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

st.sidebar.title("Panel de Control")
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
    st.error("❌ No se encontró la tabla de la jornada. Corre 'actualizar_jornada.py' primero.")
    st.stop()

# --- TÍTULO PRINCIPAL ---
st.title(f"🏆 {liga_sel}: {home_team} vs {away_team}")
if home_raw != home_team or away_raw != away_team:
    st.caption(f"🔧 Normalización: {home_raw} ➔ {home_team} | {away_raw} ➔ {away_team}")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Historial Directo (H2H)")
    q_h2h = f"""
        SELECT Date, HomeTeam as Local, AwayTeam as Visita, 
               FTHG as [Goles L], FTAG as [Goles V], FTR as Resultado 
        FROM historial_multiliga_ml 
        WHERE (HomeTeam='{home_team}' AND AwayTeam='{away_team}') 
           OR (HomeTeam='{away_team}' AND AwayTeam='{home_team}') 
        ORDER BY Date DESC LIMIT 5
    """
    df_h2h = pd.read_sql(q_h2h, conn)
    
    if not df_h2h.empty:
        df_h2h['Date'] = pd.to_datetime(df_h2h['Date']).dt.strftime('%d/%m/%Y')
        mapa_resultados = {'H': 'Local', 'A': 'Visita', 'D': 'Empate'}
        df_h2h['Resultado'] = df_h2h['Resultado'].map(mapa_resultados)
        df_h2h = df_h2h.rename(columns={'Date': 'Fecha'})
        st.dataframe(df_h2h, use_container_width=True, hide_index=True)
    else:
        st.info(f"Sin enfrentamientos previos registrados.")

    st.subheader("Tendencia de Goles")
    q_trend = f"""
        SELECT FTHG as [Goles Local], FTAG as [Goles Visita] FROM historial_multiliga_ml 
        WHERE HomeTeam='{home_team}' OR AwayTeam='{home_team}' 
        ORDER BY Date DESC LIMIT 10
    """
    df_trend = pd.read_sql(q_trend, conn)
    st.line_chart(df_trend)

with col2:
    st.subheader("Análisis Predictivo")
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
            hole=0.4
        )
        fig_pie.update_layout(dragmode=False)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        st.markdown("#### **Doble Oportunidad**")
        c1, c2 = st.columns(2)
        c1.metric("1X (Local o Empate)", f"{(p_loc + p_emp):.1%}")
        c2.metric("X2 (Visita o Empate)", f"{(p_vis + p_emp):.1%}")

        promedio_goles = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
        prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))
        st.progress(prob_over, text=f"Probabilidad Over 2.5: {prob_over:.1%}")

# --- SECCIÓN DE DISCIPLINA ---
st.divider()
st.subheader("Análisis de Disciplina y Tarjetas")
col_disc1, col_disc2 = st.columns(2)

with col_disc1:
    st.markdown("#### **Promedio Amarillas**")
    c1, c2 = st.columns(2)
    c1.metric(f"Media {home_team}", f"{stats_h['HY']:.1f}")
    c2.metric(f"Media {away_team}", f"{stats_a['AY']:.1f}")

with col_disc2:
    st.markdown("#### **Tarjetas en últimos H2H**")
    q_cards_h2h = f"""
        SELECT Date, (HY + AY) as Total_Amarillas
        FROM historial_multiliga_ml 
        WHERE (HomeTeam = '{home_team}' AND AwayTeam = '{away_team}') 
           OR (HomeTeam = '{away_team}' AND AwayTeam = '{home_team}')
        ORDER BY Date DESC LIMIT 5
    """
    df_cards_h2h = pd.read_sql(q_cards_h2h, conn)
    
    if not df_cards_h2h.empty:
        fig_cards = px.bar(df_cards_h2h, x='Date', y='Total_Amarillas', 
                           color_discrete_sequence=['#f1c40f'])
        fig_cards.update_layout(dragmode=False)
        st.plotly_chart(fig_cards, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Sin registros de tarjetas previos.")

# --- SECCIÓN INFERIOR: PATRONES ---
st.divider()
st.subheader("Patrones Relevantes del Modelo")
if model:
    importancia = pd.DataFrame({
        'Feature': ['Goles L', 'Goles V', 'Tiros L', 'Tiros V', 'ST L', 'ST V', 'Córners L', 'Córners V', 'Amarillas L', 'Amarillas V'],
        'Weight': model.feature_importances_
    }).sort_values(by='Weight', ascending=True)
    
    fig_imp = px.bar(importancia, x='Weight', y='Feature', orientation='h', 
                     color_discrete_sequence=['#3498db'])
    fig_imp.update_layout(dragmode=False)
    st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
