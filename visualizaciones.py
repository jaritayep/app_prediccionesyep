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

# --- ESTILO ---
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

# --- FUNCIONES ---
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

# --- NAVEGACIÓN ---
st.sidebar.title("⚽ Menú Principal")
menu = st.sidebar.radio("Ir a:", ["Análisis del Día", "Auditoría (Resultados)", "BetBuilder Simulator"])
st.sidebar.markdown("---")

if menu == "Análisis del Día":
    try:
        equipos_db = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        df_jornada['Date'] = pd.to_datetime(df_jornada['Date'])
        
        # --- CORRECCIÓN DE DUPLICADOS EN FECHAS ---
        # Creamos una columna temporal para mostrar en el selectbox
        df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m')
        
        # Obtenemos los días únicos basándonos en la fecha real (para que el orden sea correcto)
        fechas_unicas = sorted(df_jornada['Date'].unique())
        opciones_fecha = [d.strftime('%A %d/%m') for d in pd.to_datetime(fechas_unicas)]
        
        # El selectbox ahora usa la lista limpia y sin repetidos
        dia_sel_str = st.sidebar.selectbox("📅 Seleccionar Día:", opciones_fecha)
        
        # Filtramos los partidos que coincidan exactamente con ese string de fecha
        partidos_dia = df_jornada[df_jornada['Fecha_Display'] == dia_sel_str]
        
        partido_texto = st.sidebar.selectbox("🏟️ Partido:", partidos_dia['Local'] + " vs " + partidos_dia['Visita'])
        
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
                
                # --- GOLES ---
                promedio_goles = (stats_h['FTHG'] + stats_h['FTAG'] + stats_a['FTHG'] + stats_a['FTAG']) / 2
                prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))
                c1, c2 = st.columns(2)
                c1.metric("Goles Exp.", f"{promedio_goles:.2f}")
                c2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                st.progress(prob_over)

                # --- PROYECCIONES ---
                st.markdown("#### **🎯 Tiros y Córners**")
                cp1, cp2 = st.columns(2)
                with cp1:
                    st.write(f"Tiros: **{stats_h['HST']:.1f}** | **{stats_a['AST']:.1f}**")
                with cp2:
                    st.write(f"Córners: **{stats_h['HC']:.1f}** | **{stats_a['AC']:.1f}**")

        # --- SECCIÓN DE DISCIPLINA (RESTORED) ---
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
        st.error(f"Error: {e}")

elif menu == "Auditoría (Resultados)":
    st.title("⚖️ Auditoría de Resultados")
    st.dataframe(pd.read_sql("SELECT Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR FROM historial_multiliga_ml ORDER BY Date DESC LIMIT 20", conn), use_container_width=True)

elif menu == "BetBuilder Simulator":
    st.title("🛠️ BetBuilder AI")
    col_bb1, col_bb2 = st.columns(2)
    picks = []
    with col_bb1:
        for i in range(st.number_input("Eventos:", 1, 5, 2)):
            picks.append(st.slider(f"Prob. Pick {i+1} (%)", 1, 99, 50)/100)
    with col_bb2:
        res_prob = np.prod(picks)
        st.metric("Probabilidad Total", f"{res_prob:.1%}")
        st.metric("Cuota Justa", f"{1/res_prob:.2f}" if res_prob > 0 else "0")

conn.close()
# ABRIR CMD Y "cd C:\Users\sealj\OneDrive\Escritorio\proyecto_app" 
# luego ejecutar py -m streamlit run visualizaciones.py
