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
        # 1. Intentar cargar partidos de clubes
        df_jornada = pd.read_sql("SELECT * FROM tabla_predicciones_limpia", conn)
        hoy = pd.Timestamp.now().normalize()
        
        if not df_jornada.empty:
            df_jornada['Date'] = pd.to_datetime(df_jornada['Date']).dt.tz_localize(None).dt.normalize()
            df_jornada = df_jornada[df_jornada['Date'] >= hoy]
        
        es_mundial = False
        
        # 2. 🎯 MODO INTERNACIONAL (Fallback si no hay clubes)
        if df_jornada.empty:
            df_jornada = pd.read_sql("SELECT * FROM fixture_mundial WHERE HomeTeam != 'TBA' AND AwayTeam != 'TBA'", conn)
            df_jornada['Date'] = pd.to_datetime(df_jornada['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
            df_jornada = df_jornada[df_jornada['Date'] >= hoy]
            es_mundial = True
            st.info("🌍 **Modo Internacional Automático:** No hay partidos de clubes programados. Mostrando Fixture del Mundial.")

        if not df_jornada.empty:
            # 🎯 FIX: Ordenar cronológicamente antes de crear las etiquetas
            df_jornada = df_jornada.sort_values(by='Date', ascending=True)
            
            df_jornada['Fecha_Display'] = df_jornada['Date'].dt.strftime('%A %d/%m')

            # 3. Selección en la sidebar
            opciones_fecha = list(dict.fromkeys(df_jornada['Fecha_Display'].tolist()))
            dia_sel_str = st.sidebar.selectbox("📅 Seleccionar Día:", opciones_fecha)

            partidos_dia = df_jornada[df_jornada['Fecha_Display'] == dia_sel_str]
            
            if es_mundial:
                partidos_list = partidos_dia['HomeTeam'] + " vs " + partidos_dia['AwayTeam']
            else:
                partidos_list = partidos_dia['Local'] + " vs " + partidos_dia['Visita']
                
            partido_texto = st.sidebar.selectbox("🏟️ Partido:", partidos_list)

            # Separar y corregir nombres
            home_raw, away_raw = partido_texto.split(" vs ")
            
            hist_table = "historial_selecciones_ml" if es_mundial else "historial_multiliga_ml"
            equipos_db = pd.read_sql(f"SELECT DISTINCT HomeTeam FROM {hist_table}", conn)['HomeTeam'].tolist()
            
            home_team = corregir_nombre_equipo(home_raw, equipos_db)
            away_team = corregir_nombre_equipo(away_raw, equipos_db)

            # --- RENDERIZADO DEL DASHBOARD ---
            st.title(f"{home_team} vs {away_team}")
            st.caption(f"📅 {dia_sel_str}")

            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.subheader("📊 Historial H2H")
                q_h2h = f'SELECT Date, HomeTeam as L, AwayTeam as V, FTHG as [GL], FTAG as [GV], FTR as R FROM {hist_table} WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") ORDER BY Date DESC LIMIT 5'
                df_h2h = pd.read_sql(q_h2h, conn)
                if not df_h2h.empty:
                    df_h2h['Date'] = pd.to_datetime(df_h2h['Date']).dt.strftime('%d/%m/%y')
                    st.dataframe(df_h2h, use_container_width=True, hide_index=True)
                else:
                    st.info("No existen enfrentamientos directos recientes en la base de datos.")

                st.subheader("📈 Tendencia de Goles")
                q_trend = f'SELECT FTHG as [Local], FTAG as [Visita] FROM {hist_table} WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" ORDER BY Date DESC LIMIT 10'
                df_trend = pd.read_sql(q_trend, conn)
                if not df_trend.empty:
                    st.line_chart(df_trend.iloc[::-1])

            with col2:
                st.subheader("IA Predictiva")
                
                if es_mundial:
                    try:
                        modelo_intl = joblib.load('modelo_selecciones_rf.pkl')
                        encoder_intl = joblib.load('encoder_equipos_selecciones.pkl')
                        
                        df_sh = pd.read_sql(f'SELECT * FROM {hist_table} WHERE HomeTeam="{home_team}" OR AwayTeam="{home_team}" ORDER BY Date DESC LIMIT 6', conn)
                        df_sa = pd.read_sql(f'SELECT * FROM {hist_table} WHERE HomeTeam="{away_team}" OR AwayTeam="{away_team}" ORDER BY Date DESC LIMIT 6', conn)
                        
                        def seguro_mean(df, col, default):
                            return df[col].mean() if not df.empty and col in df.columns and pd.notna(df[col].mean()) else default

                        hst, hc = seguro_mean(df_sh, 'HST', 4.0), seguro_mean(df_sh, 'HC', 4.5)
                        ast, ac = seguro_mean(df_sa, 'AST', 3.5), seguro_mean(df_sa, 'AC', 4.0)
                        xg_h = seguro_mean(df_sh, 'xG_home', 1.2)
                        xg_a = seguro_mean(df_sa, 'xG_away', 1.0)
                        
                        # Separar partidos por rol para calcular goles correctamente
                        df_sh_home = df_sh[df_sh['HomeTeam'] == home_team]
                        df_sh_away = df_sh[df_sh['AwayTeam'] == home_team]
                        df_sa_home = df_sa[df_sa['HomeTeam'] == away_team]
                        df_sa_away = df_sa[df_sa['AwayTeam'] == away_team]

                        def concat_mean(s1, s2, default):
                            combined = pd.concat([s1, s2])
                            return combined.mean() if not combined.empty and pd.notna(combined.mean()) else default

                        # Goles anotados y recibidos de cada equipo (combinando ambos roles)
                        gf_h = concat_mean(df_sh_home['FTHG'], df_sh_away['FTAG'], 1.5)
                        gc_h = concat_mean(df_sh_home['FTAG'], df_sh_away['FTHG'], 1.0)
                        gf_a = concat_mean(df_sa_home['FTHG'], df_sa_away['FTAG'], 1.2)
                        gc_a = concat_mean(df_sa_home['FTAG'], df_sa_away['FTHG'], 1.3)

                        if home_team in encoder_intl.classes_ and away_team in encoder_intl.classes_:
                            h_c = encoder_intl.transform([home_team])[0]
                            a_c = encoder_intl.transform([away_team])[0]
                            X_input = pd.DataFrame([[h_c, a_c, hst, ast, hc, ac, xg_h, xg_a]], columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC','xG_home','xG_away'])
                            probs = modelo_intl.predict_proba(X_input)[0]
                            prob_visita, prob_empate, prob_local = probs[0], probs[1], probs[2]
                        else:
                            prob_visita, prob_empate, prob_local = 0.33, 0.34, 0.33

                        xg_h = (gf_h + gc_a) / 2
                        xg_a = (gf_a + gc_h) / 2
                        promedio_goles = xg_h + xg_a
                        prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))

                        fig_pie = px.pie(values=[prob_local, prob_empate, prob_visita], names=['Local', 'Empate', 'Visita'], color=['Local', 'Empate', 'Visita'], color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'}, hole=0.45)
                        fig_pie.update_layout(dragmode=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)

                        c1, c2 = st.columns(2)
                        c1.metric("Goles Exp. (xG Total)", f"{(xg_h + xg_a):.2f}")
                        c2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                        st.progress(prob_over)

                        st.markdown("---")
                        cp_g1, cp_g2 = st.columns(2)
                        cp_g1.metric(f"Goles {home_team[:10]}", f"{xg_h:.2f}")
                        cp_g2.metric(f"Goles {away_team[:10]}", f"{xg_a:.2f}")
                        st.markdown("---")

                        st.markdown("#### **Tiros y Córners**")
                        cp1, cp2 = st.columns(2)
                        with cp1: st.write(f"Tiros: **{hst:.1f}** | **{ast:.1f}**")
                        with cp2: st.write(f"Córners: **{hc:.1f}** | **{ac:.1f}**")
                        
                    except Exception as e:
                        st.error(f"Error cargando IA de selecciones: {e}")
                        
                else:
                    model = cargar_modelo()
                    if model:
                        stats_h, stats_a = get_recent_stats(home_team, conn), get_recent_stats(away_team, conn)
                        stats_h_dict, stats_a_dict = stats_h, stats_a

                        xg_h = stats_h.get('xG_home', 1.0) 
                        xg_a = stats_a.get('xG_away', 1.0)
                        xg_diff = xg_h - xg_a
                        pts_h = obtener_puntos_temporada(home_team, conn)
                        pts_a = obtener_puntos_temporada(away_team, conn)
                        dif_tabla = pts_h - pts_a
                        descanso_h = obtener_dias_descanso(home_team, conn)
                        descanso_a = obtener_dias_descanso(away_team, conn)
                        ventaja_fisica = descanso_h - descanso_a
                        
                        eff_h = stats_h['FTHG'] / (xg_h + 0.01)
                        eff_a = stats_a['FTAG'] / (xg_a + 0.01)

                        input_data = [[
                            stats_h['FTHG'], stats_h['FTAG'], stats_h['HS'], stats_h['AS'], 
                            stats_h['HST'], stats_h['AST'], stats_h['HC'], stats_h['AC'], 
                            stats_h['HY'], stats_h['AY'], xg_h, xg_a, eff_h, xg_diff, 
                            dif_tabla, ventaja_fisica
                        ]]
                        
                        prob_ia = model.predict_proba(input_data)[0]

                        fig_pie = px.pie(values=[prob_ia[2], prob_ia[1], prob_ia[0]], names=['Local', 'Empate', 'Visita'], color=['Local', 'Empate', 'Visita'], color_discrete_map={'Local': '#27ae60', 'Empate': '#7f8c8d', 'Visita': '#c0392b'}, hole=0.45)
                        fig_pie.update_layout(dragmode=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True, config=CONFIG_FIJA)

                        pred_home = (stats_h['FTHG'] + stats_a['FTAG']) / 2
                        pred_away = (stats_a['FTHG'] + stats_h['FTAG']) / 2
                        promedio_goles = pred_home + pred_away
                        prob_over = 1 / (1 + np.exp(-(promedio_goles - 2.5)))

                        c1, c2 = st.columns(2)
                        c1.metric("Goles Exp. (xG Total)", f"{(xg_h + xg_a):.2f}")
                        c2.metric("Prob. Over 2.5", f"{prob_over:.1%}")
                        st.progress(prob_over)

                        st.markdown("---")
                        cp_g1, cp_g2 = st.columns(2)
                        cp_g1.metric(f"Goles {home_team[:10]}", f"{pred_home:.2f}")
                        cp_g2.metric(f"Goles {away_team[:10]}", f"{pred_away:.2f}")
                        st.markdown("---")

                        st.markdown("#### **Tiros y Córners**")
                        cp1, cp2 = st.columns(2)
                        with cp1: st.write(f"Tiros: **{stats_h['HST']:.1f}** | **{stats_a['AST']:.1f}**")
                        with cp2: st.write(f"Córners: **{stats_h['HC']:.1f}** | **{stats_a['AC']:.1f}**")

            # --- 🎯 FIX: CONDICIONAL PARA TARJETAS ---
            st.divider()
            st.subheader("🟨 Disciplina y Tarjetas")
            
            if es_mundial:
                st.info("ℹ️ La base de datos de selecciones no incluye registro de tarjetas. Esta métrica es exclusiva del modelo de clubes.")
            else:
                cd1, cd2 = st.columns(2)
                with cd1:
                    st.markdown("#### **Media Amarillas**")
                    m1, m2 = st.columns(2)
                    m1.metric(f"{home_team[:12]}", f"{stats_h_dict.get('HY', 0):.1f}")
                    m2.metric(f"{away_team[:12]}", f"{stats_a_dict.get('AY', 0):.1f}")

                with cd2:
                    q_cards = f'SELECT Date, (HY + AY) as Total FROM {hist_table} WHERE (HomeTeam="{home_team}" AND AwayTeam="{away_team}") OR (HomeTeam="{away_team}" AND AwayTeam="{home_team}") ORDER BY Date DESC LIMIT 5'
                    try:
                        df_cards = pd.read_sql(q_cards, conn)
                        if not df_cards.empty:
                            fig_cards = px.bar(df_cards, x='Date', y='Total', color_discrete_sequence=['#f1c40f'])
                            fig_cards.update_layout(dragmode=False, xaxis={'fixedrange': True}, yaxis={'fixedrange': True})
                            st.plotly_chart(fig_cards, use_container_width=True, config=CONFIG_FIJA)
                        else:
                            st.info("Sin datos suficientes de tarjetas para graficar H2H.")
                    except Exception:
                        st.info("No se pudieron graficar las tarjetas.")
                    
        else:
            st.info("No hay partidos programados en la base de datos.")

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
    st.title("📈 Portafolio de Inversión (Flat Staking Híbrido)")
    
    API_KEY = "3ec28dbd498ab9985e9792b3f50a8902" 
    
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

    tab1, tab2 = st.tabs(["🔍 Escáner en Vivo", "🏦 Rendimiento Histórico"])

    with tab1:
        st.markdown("### 🔍 Escáner de Ineficiencias vs Pinnacle")
        st.caption("Cruzando modelos ML (Clubes y Selecciones) y Poisson contra líneas de Pinnacle. (Edge 2% - 15%)")
        
        try:
            equipos_clubes = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_multiliga_ml", conn)['HomeTeam'].tolist()
            equipos_wc = pd.read_sql("SELECT DISTINCT HomeTeam FROM historial_selecciones_ml", conn)['HomeTeam'].tolist()
            
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
                        
                        # 🎯 DETECCIÓN HÍBRIDA: Saber si el archivo es del Mundial
                        df_temp['Es_Mundial'] = 'worldcup' in f.name.lower()
                        
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

                        df_temp.columns = [c.lower() for c in df_temp.columns]
                        if 'es_mundial' in df_temp.columns: df_temp = df_temp.rename(columns={'es_mundial': 'Es_Mundial'})

                        if 'hometeam' in df_temp.columns and 'home' not in df_temp.columns: df_temp = df_temp.rename(columns={'hometeam': 'home'})
                        if 'awayteam' in df_temp.columns and 'away' not in df_temp.columns: df_temp = df_temp.rename(columns={'awayteam': 'away'})

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
                    df_master_odds['Fecha_Match'] = df_master_odds['inicio_local'].astype(str).str.strip().str.slice(0, 10)
                    df_master_odds = df_master_odds[df_master_odds['Fecha_Match'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]
                    fechas_disponibles = sorted(df_master_odds['Fecha_Match'].unique())

            c1, c2 = st.columns(2)
            with c1:
                inversion_total = st.number_input("💰 Inversión TOTAL Portafolio ($)", min_value=1000, value=5000, step=500)
            with c2:
                if fechas_disponibles:
                    hoy_str = str(pd.Timestamp.now().date())
                    idx_hoy = fechas_disponibles.index(hoy_str) if hoy_str in fechas_disponibles else max(0, len(fechas_disponibles) - 1)
                    fecha_seleccionada = st.selectbox("📅 Seleccionar Día del Portafolio:", fechas_disponibles, index=idx_hoy)
                else:
                    st.error("⚠️ No se encontraron partidos en la carpeta 'odds_data/'. ¡Corre el scraper primero!")
                    fecha_seleccionada = None

            boton_disabled = fecha_seleccionada is None

            if st.button("🔍 Escanear Mercado", type="primary", disabled=boton_disabled):
                modelo_clubes = cargar_modelo()
                try:
                    modelo_wc = joblib.load('modelo_selecciones_rf.pkl')
                    encoder_wc = joblib.load('encoder_equipos_selecciones.pkl')
                except:
                    modelo_wc = None

                with st.spinner(f"Analizando los partidos del {fecha_seleccionada} con IA..."):
                    df_pinnacle = df_master_odds[df_master_odds['Fecha_Match'] == fecha_seleccionada]
                    oportunidades = []
                    mercados_evaluados_completos = []
                    log_debug = []
                    
                    def buscar_cuota_segura(row, posibles_columnas):
                        for col in posibles_columnas:
                            if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != '':
                                return row[col]
                        return None

                    for index, row in df_pinnacle.iterrows():
                        h_csv = str(row['home'])
                        a_csv = str(row['away'])
                        es_mundial = row.get('Es_Mundial', False)
                        fecha_partido = str(row['inicio_local']).split()[0] if pd.notna(row['inicio_local']) else str(pd.Timestamp.now().date())
                        
                        lista_referencia = equipos_wc if es_mundial else equipos_clubes
                        h_db_match = process.extractOne(h_csv, lista_referencia)
                        a_db_match = process.extractOne(a_csv, lista_referencia)
                        
                        if not h_db_match or not a_db_match or h_db_match[1] < 80 or a_db_match[1] < 80:
                            continue
                            
                        h_db = h_db_match[0]
                        a_db = a_db_match[0]

                        # --- MOTOR HÍBRIDO DE IA ---
                        if es_mundial:
                            if not modelo_wc:
                                log_debug.append(f"⚠️ Omitiendo {h_db} vs {a_db}: No se encontró modelo_selecciones_rf.pkl")
                                continue
                                
                            df_sh = pd.read_sql(f'SELECT * FROM historial_selecciones_ml WHERE HomeTeam="{h_db}" OR AwayTeam="{h_db}" ORDER BY Date DESC LIMIT 6', conn)
                            df_sa = pd.read_sql(f'SELECT * FROM historial_selecciones_ml WHERE HomeTeam="{a_db}" OR AwayTeam="{a_db}" ORDER BY Date DESC LIMIT 6', conn)
                            
                            def seguro_mean(df, col, default):
                                return df[col].mean() if not df.empty and col in df.columns and pd.notna(df[col].mean()) else default

                            hst, hc = seguro_mean(df_sh, 'HST', 4.0), seguro_mean(df_sh, 'HC', 4.5)
                            ast, ac = seguro_mean(df_sa, 'AST', 3.5), seguro_mean(df_sa, 'AC', 4.0)
                            gf_h, gc_h = seguro_mean(df_sh, 'FTHG', 1.5), seguro_mean(df_sh, 'FTAG', 1.0)
                            gf_a, gc_a = seguro_mean(df_sa, 'FTHG', 1.2), seguro_mean(df_sa, 'FTAG', 1.3)
                            
                            if h_db in encoder_wc.classes_ and a_db in encoder_wc.classes_:
                                h_c = encoder_wc.transform([h_db])[0]
                                a_c = encoder_wc.transform([a_db])[0]
                                X_input = pd.DataFrame([[h_c, a_c, hst, ast, hc, ac]], columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC'])
                                probs = modelo_wc.predict_proba(X_input)[0]
                                prob_visita, prob_empate, prob_local = probs[0], probs[1], probs[2]
                            else:
                                prob_visita, prob_empate, prob_local = 0.33, 0.34, 0.33
                                
                            pred_goles_home = (gf_h + gc_a) / 2
                            pred_goles_away = (gf_a + gc_h) / 2
                            prom_goles_total = pred_goles_home + pred_goles_away
                            prom_corners_total = (hc + ac) / 2
                            prom_shots_total = (hst + ast) / 2
                            
                            stats_h = {'HC': hc, 'HST': hst}
                            stats_a = {'AC': ac, 'AST': ast}
                            
                        else:
                            if not modelo_clubes: continue
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
                            
                            pred_probs = modelo_clubes.predict_proba(input_data)[0]
                            prob_visita, prob_empate, prob_local = pred_probs[0], pred_probs[1], pred_probs[2]

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

                        def prob_under(promedio, umbral): return 1 - prob_over(promedio, umbral)
                        def prob_handicap(prom_favor, prom_contra, linea_hdp):
                            prob_acum = 0.0
                            for gf in range(15):
                                for gc in range(15):
                                    if (gf + linea_hdp) > gc:
                                        p_gf = (math.exp(-prom_favor) * (prom_favor**gf)) / math.factorial(gf)
                                        p_gc = (math.exp(-prom_contra) * (prom_contra**gc)) / math.factorial(gc)
                                        prob_acum += (p_gf * p_gc)
                            return prob_acum

                        for col_name, val in row.items():
                            if pd.isna(val) or str(val).strip() == '': continue
                            col_str = str(col_name).lower()
                            if col_str in ['es_mundial', 'liga', 'pais', 'partido_id', 'home', 'away', 'inicio_utc', 'inicio_local', 'fecha_match']: continue
                                
                            try:
                                val_num = float(val)
                                if val_num <= 1.0: continue
                            except ValueError: continue 
                                
                            if 'btts' in col_str or 'ambos' in col_str:
                                prob_btts_si = (1 - math.exp(-pred_goles_home)) * (1 - math.exp(-pred_goles_away))
                                if 'yes' in col_str or 'si' in col_str: mercados_a_evaluar.append(("Ambos Anotan (Sí)", val_num, prob_btts_si))
                                elif 'no' in col_str: mercados_a_evaluar.append(("Ambos Anotan (No)", val_num, 1 - prob_btts_si))
                                continue

                            match = re.search(r'(-?\d+\.5)', col_str)
                            if not match: continue
                            linea = float(match.group(1))
                            
                            if 'hdp' in col_str or 'handicap' in col_str:
                                if 'home' in col_str: mercados_a_evaluar.append((f"Hándicap Local ({linea:+})", val_num, prob_handicap(pred_goles_home, pred_goles_away, linea)))
                                elif 'away' in col_str: mercados_a_evaluar.append((f"Hándicap Visita ({linea:+})", val_num, prob_handicap(pred_goles_away, pred_goles_home, linea)))
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
                                # Pool completo: cualquier mercado con edge positivo y cuota razonable
                                if edge > 0 and cuota_flt < 8.0:
                                    mercados_evaluados_completos.append((fecha_partido, h_db, a_db, mercado_nombre, cuota_flt, prob_ia, edge))
                                if 0.02 <= edge <= 0.15:
                                    oportunidades.append((fecha_partido, h_db, a_db, mercado_nombre, cuota_flt, prob_ia, edge))
                                    log_debug.append(f"   ✨ ¡AÑADIDO AL PORTAFOLIO! Edge válido: {edge:.2%}")
                            except Exception: pass

                        for nombre_mkt, cuota_val, prob_ia in mercados_a_evaluar:
                            evaluar_edge(nombre_mkt, prob_ia, cuota_val)

                    with st.expander("🛠️ Ver Diagnóstico Completo del Robot Evaluador"):
                        for log_msg in log_debug: st.text(log_msg)

                    if oportunidades:
                        df_ops = pd.DataFrame(oportunidades, columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge'])
                        st.session_state['portafolio_escaneado'] = df_ops.sort_values(by='Edge', ascending=False).drop_duplicates(subset=['Home', 'Away', 'Mercado']).reset_index(drop=True)
                        st.session_state['pool_mercados'] = mercados_evaluados_completos
                    else:
                        st.warning("📊 No se encontraron ineficiencias dentro del rango rentable (2% a 15%).")

            if 'portafolio_escaneado' in st.session_state:
                df_ops = st.session_state['portafolio_escaneado'].copy()
                df_ops['Partido'] = df_ops['Home'] + " vs " + df_ops['Away']
                df_ops['Edge_Str'] = (df_ops['Edge'] * 100).round(2).astype(str) + "%"
                df_ops['Prob_IA_Str'] = (df_ops['Prob_IA'] * 100).round(1).astype(str) + "%"
                
                columnas_base = ['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str', 'Date', 'Home', 'Away', 'Prob_IA', 'Edge']
                df_ops = df_ops[columnas_base]

                TARGET_PICKS = 10

                selected_indices = []
                used_matches = set()
                df_top_10_list = []

                # ── helpers ───────────────────────────────────────────────
                def _mercados_elegidos(partido):
                    return {r.iloc[0]['Mercado'] for r in df_top_10_list if r.iloc[0]['Partido'] == partido}

                def add_pick(row_or_idx, nivel_label, from_df=True):
                    """Agrega pick; from_df=True usa índice de df_ops, False recibe dict."""
                    if from_df:
                        row = df_ops.loc[row_or_idx]
                        partido = row['Partido']
                        if partido in used_matches:
                            return False
                        used_matches.add(partido)
                        selected_indices.append(row_or_idx)
                        df_top_10_list.append(df_ops.loc[[row_or_idx]].assign(Nivel=nivel_label))
                    else:
                        d = row_or_idx
                        partido = d['Partido']
                        if mkt_already_added(partido, d['Mercado']):
                            return False
                        used_matches.add(partido)
                        df_top_10_list.append(pd.DataFrame([d]))
                    return True

                def mkt_already_added(partido, mercado):
                    return mercado in _mercados_elegidos(partido)

                def get_pool(min_c, max_c, edge_min=0.02, edge_max=0.15):
                    """Pool de df_ops filtrado por cuota y edge, excluyendo partidos ya usados."""
                    mask = (
                        ~df_ops['Partido'].isin(used_matches) &
                        (df_ops['Cuota'] >= min_c) & (df_ops['Cuota'] < max_c) &
                        (df_ops['Edge'] >= edge_min) & (df_ops['Edge'] <= edge_max)
                    )
                    return df_ops[mask].drop_duplicates(subset=['Partido'], keep='first')

                def fill_bucket(min_c, max_c, target_n, nivel_label, edge_min=0.02, edge_max=0.15):
                    """Intenta llenar un bucket hasta target_n picks dentro del rango de cuota.
                    Itera por todo el pool hasta completar target_n slots (1 partido por pick)."""
                    added = 0
                    pool = get_pool(min_c, max_c, edge_min, edge_max)
                    if pool.empty:
                        return 0
                    for idx in pool.index:
                        if added >= target_n:
                            break
                        if add_pick(idx, nivel_label):
                            added += 1
                    return added

                def faltantes_bucket(conteo_actual, target=3):
                    return max(0, target - conteo_actual)

                # ══════════════════════════════════════════════════════════
                # FASE 1 — estructura 3-3-3-1 con edge ESTÁNDAR (2%-15%)
                # ══════════════════════════════════════════════════════════

                # Golden Pick: mejor edge general (cuota cualquiera)
                for idx in df_ops.index:
                    if add_pick(idx, '⭐ Golden Pick'): break

                # Buckets estándar: Alto 3, Medio 3, Bajo 3
                n_high = fill_bucket(2.50, 999.0, 3, '🔴 Alto (>2.50)')
                n_med  = fill_bucket(1.90, 2.50,  3, '🟡 Medio (1.90-2.49)')
                n_low  = fill_bucket(0.0,  1.90,  3, '🟢 Bajo (<1.90)')

                # ══════════════════════════════════════════════════════════
                # FASE 2 — ampliar edge MANTENIENDO estructura 3-3-3-1 ESTRICTA
                # Se expande en pasos progresivos bucket por bucket.
                # ══════════════════════════════════════════════════════════
                EDGE_PASOS = [
                    (0.01,  0.20),   # paso 1: relajar un poco
                    (0.005, 0.25),   # paso 2: abrir más
                    (0.001, 0.35),   # paso 3: casi cualquier edge positivo
                ]

                if len(df_top_10_list) < TARGET_PICKS:
                    for edge_min_exp, edge_max_exp in EDGE_PASOS:
                        if len(df_top_10_list) >= TARGET_PICKS:
                            break
                        falt_high = faltantes_bucket(n_high)
                        falt_med  = faltantes_bucket(n_med)
                        falt_low  = faltantes_bucket(n_low)
                        lbl_suf = f'edge {edge_min_exp:.1%}–{edge_max_exp:.0%}'

                        if falt_high:
                            added = fill_bucket(2.50, 999.0, falt_high,
                                                f'🔴 Alto — {lbl_suf}',
                                                edge_min_exp, edge_max_exp)
                            n_high += added
                        if falt_med:
                            added = fill_bucket(1.90, 2.50, falt_med,
                                                f'🟡 Medio — {lbl_suf}',
                                                edge_min_exp, edge_max_exp)
                            n_med += added
                        if falt_low:
                            added = fill_bucket(0.0, 1.90, falt_low,
                                                f'🟢 Bajo — {lbl_suf}',
                                                edge_min_exp, edge_max_exp)
                            n_low += added

                # ══════════════════════════════════════════════════════════
                # FASE 3 — estructura flexible: llenar con lo que haya
                # Usa el pool completo (mercados_evaluados_completos) que
                # incluye TODO lo que tenga edge > 0, sin límite de rango.
                # Permite repetir partido si es mercado distinto.
                # ══════════════════════════════════════════════════════════
                if len(df_top_10_list) < TARGET_PICKS:
                    faltantes_f3 = TARGET_PICKS - len(df_top_10_list)

                    pool_completo = st.session_state.get('pool_mercados', [])

                    if pool_completo:
                        df_f3 = pd.DataFrame(
                            pool_completo,
                            columns=['Date', 'Home', 'Away', 'Mercado', 'Cuota', 'Prob_IA', 'Edge']
                        )
                        df_f3['Partido'] = df_f3['Home'] + " vs " + df_f3['Away']
                        df_f3['Prob_IA_Str'] = (df_f3['Prob_IA'] * 100).round(1).astype(str) + "%"
                        df_f3['Edge_Str']    = (df_f3['Edge']    * 100).round(2).astype(str) + "%"

                        # Excluir mercados exactos ya elegidos (partido+mercado)
                        mercados_ya_en_portfolio = set()
                        for item in df_top_10_list:
                            r = item.iloc[0]
                            mercados_ya_en_portfolio.add((r['Partido'], r['Mercado']))

                        df_f3 = df_f3[
                            ~df_f3.apply(lambda r: (r['Partido'], r['Mercado']) in mercados_ya_en_portfolio, axis=1)
                        ]

                        # Ordenar: mayor edge primero, preferir partidos nuevos
                        df_f3['es_partido_nuevo'] = (~df_f3['Partido'].isin(used_matches)).astype(int)
                        df_f3 = df_f3.sort_values(['es_partido_nuevo', 'Edge'], ascending=[False, False])

                        picks_f3 = 0
                        for _, row_f3 in df_f3.iterrows():
                            if picks_f3 >= faltantes_f3:
                                break
                            cuota_f3 = row_f3['Cuota']
                            if cuota_f3 >= 2.50:
                                nivel_f3 = '🔴 Flexible (alto)'
                            elif cuota_f3 >= 1.90:
                                nivel_f3 = '🟡 Flexible (medio)'
                            else:
                                nivel_f3 = '🟢 Flexible (bajo)'

                            entry = {
                                'Date': row_f3['Date'], 'Home': row_f3['Home'], 'Away': row_f3['Away'],
                                'Mercado': row_f3['Mercado'], 'Cuota': cuota_f3,
                                'Prob_IA': row_f3['Prob_IA'], 'Edge': row_f3['Edge'],
                                'Prob_IA_Str': row_f3['Prob_IA_Str'], 'Edge_Str': row_f3['Edge_Str'],
                                'Partido': row_f3['Partido'], 'Nivel': nivel_f3
                            }
                            # add_pick controla duplicados de partido; aquí permitimos mismo partido distinto mercado
                            partido_f3 = row_f3['Partido']
                            if partido_f3 not in used_matches:
                                used_matches.add(partido_f3)
                            df_top_10_list.append(pd.DataFrame([entry]))
                            picks_f3 += 1

                        if picks_f3 > 0:
                            st.warning(
                                f"⚠️ **Modo flexible activado:** Se añadieron {picks_f3} pick(s) "
                                f"con el mejor edge disponible fuera del rango estándar. "
                                f"Aparecen como 'Flexible' en la tabla — úsalos con criterio."
                            )

                faltantes = TARGET_PICKS - len(df_top_10_list)
                if faltantes > 0:
                    st.info(
                        f"ℹ️ Portafolio parcial: {len(df_top_10_list)}/10 picks. "
                        f"No hay suficiente mercado con edge positivo hoy para los {faltantes} slots restantes."
                    )

                df_top_10 = pd.concat(df_top_10_list).reset_index(drop=True) if df_top_10_list else pd.DataFrame()

                # Asegurar columnas string para display
                if not df_top_10.empty:
                    if 'Prob_IA_Str' not in df_top_10.columns:
                        df_top_10['Prob_IA_Str'] = (df_top_10['Prob_IA'] * 100).round(1).astype(str) + "%"
                    if 'Edge_Str' not in df_top_10.columns:
                        df_top_10['Edge_Str'] = (df_top_10['Edge'] * 100).round(2).astype(str) + "%"
                    if 'Partido' not in df_top_10.columns:
                        df_top_10['Partido'] = df_top_10['Home'] + " vs " + df_top_10['Away']

                cols_mostrar = ['Nivel', 'Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']

                df_mostrar_top = df_top_10[cols_mostrar].copy()
                df_mostrar_top.insert(0, "✅ Añadir", True)
                
                df_reserva = df_ops[~df_ops.index.isin(selected_indices)].reset_index(drop=True)
                df_mostrar_reserva = df_reserva[['Partido', 'Mercado', 'Cuota', 'Prob_IA_Str', 'Edge_Str']].copy()
                df_mostrar_reserva.insert(0, "✅ Añadir", False) 

                modo = "completo ✅" if len(df_top_10) >= TARGET_PICKS else f"parcial ({len(df_top_10)}/{TARGET_PICKS} picks)"
                st.success(f"Escaneo listo. Portafolio {modo} — {len(df_top_10)} picks seleccionados.")
                st.markdown(f"### 🎯 Portafolio ({len(df_top_10)} picks)")
                
                edit_top10 = st.data_editor(
                    df_mostrar_top,
                    hide_index=True,
                    use_container_width=True,
                    key="editor_top10",
                    column_config={"✅ Añadir": st.column_config.CheckboxColumn(required=True)}
                )
                
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

                if st.button("💾 Guardar Portafolio Seleccionado", type="primary"):
                    indices_top = edit_top10[edit_top10["✅ Añadir"] == True].index
                    indices_res = edit_reserva[edit_reserva["✅ Añadir"] == True].index if not df_mostrar_reserva.empty else []
                    
                    df_final_top = df_top_10.iloc[indices_top]
                    df_final_res = df_reserva.iloc[indices_res] if not df_mostrar_reserva.empty else pd.DataFrame()
                    df_final_a_guardar = pd.concat([df_final_top, df_final_res])
                    
                    if df_final_a_guardar.empty:
                        st.warning("No seleccionaste ningún pick.")
                    else:
                        stake_por_pick = inversion_total / len(df_final_a_guardar)
                        for _, row in df_final_a_guardar.iterrows():
                            cursor.execute("""
                                INSERT INTO portafolio_historico (Date, HomeTeam, AwayTeam, Mercado, Cuota, Prob_IA, Edge, Stake)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (row['Date'], row['Home'], row['Away'], row['Mercado'], row['Cuota'], row['Prob_IA'], row['Edge'], stake_por_pick))
                        conn.commit()
                        st.success(f"✅ ¡{len(df_final_a_guardar)} picks guardados en el portafolio activo! (Inversión por pick: ${stake_por_pick:,.0f})")
                        del st.session_state['portafolio_escaneado']

        except Exception as e:
            st.error(f"Error en la aplicación: {e}")

        # ── PORTAFOLIO ACTIVO (siempre visible, no requiere re-escanear) ──
        st.divider()
        st.markdown("### 📋 Portafolio Activo (Picks Pendientes)")
        df_activos = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente' ORDER BY Date ASC", conn)
        if df_activos.empty:
            st.info("No hay picks pendientes guardados. Escanea el mercado y guarda tu portafolio para verlo aquí.")
        else:
            # Agrupar por fecha para visualizar días distintos
            fechas_activas = sorted(df_activos['Date'].unique())
            for fecha in fechas_activas:
                df_dia = df_activos[df_activos['Date'] == fecha].copy()
                stake_dia = df_dia['Stake'].sum()
                retorno_potencial = (df_dia['Cuota'] * df_dia['Stake']).sum()
                with st.expander(f"📅 {fecha}  —  {len(df_dia)} picks  |  Invertido: ${stake_dia:,.0f}  |  Retorno potencial: ${retorno_potencial:,.0f}", expanded=(fecha == fechas_activas[-1])):
                    df_mostrar_activos = df_dia[['HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Prob_IA', 'Edge', 'Stake']].copy()
                    df_mostrar_activos['Prob_IA'] = (df_mostrar_activos['Prob_IA'] * 100).round(1).astype(str) + "%"
                    df_mostrar_activos['Edge'] = (df_mostrar_activos['Edge'] * 100).round(2).astype(str) + "%"
                    df_mostrar_activos['Partido'] = df_mostrar_activos['HomeTeam'] + " vs " + df_mostrar_activos['AwayTeam']
                    df_mostrar_activos = df_mostrar_activos[['Partido', 'Mercado', 'Cuota', 'Prob_IA', 'Edge', 'Stake']]
                    st.dataframe(df_mostrar_activos, hide_index=True, use_container_width=True)

    with tab2:
        c_tit, c_btn = st.columns([0.75, 0.25])
        c_tit.subheader("🏦 Rendimiento Acumulado")
        
        if c_btn.button("🗑️ Resetear Historial", use_container_width=True):
            cursor.execute("DELETE FROM portafolio_historico")
            conn.commit()
            st.toast("¡Historial borrado con éxito! Portafolio limpio.")
            st.rerun()
            
        if st.button("⚖️ Liquidar Apuestas Pendientes", type="primary"):
            df_pendientes = pd.read_sql("SELECT * FROM portafolio_historico WHERE Estado = 'Pendiente'", conn)
            liquidadas = 0
            beneficio_reciente = 0.0
            stake_reciente = 0.0
            
            import re
            from thefuzz import process
            from datetime import datetime, timedelta
            
            for _, pick in df_pendientes.iterrows():
                try:
                    fecha_dt = datetime.strptime(pick['Date'], '%Y-%m-%d')
                    fecha_inicio = (fecha_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                    fecha_fin = (fecha_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception:
                    fecha_inicio = pick['Date']
                    fecha_fin = pick['Date']

                # 🎯 LIQUIDADOR HÍBRIDO: Busca en ambas bases de datos unificadas
                q_res = f"""
                SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST FROM historial_multiliga_ml WHERE Date BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                UNION ALL
                SELECT HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HST, AST FROM historial_selecciones_ml WHERE Date BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                """
                
                try:
                    res_real = pd.read_sql(q_res, conn)
                except Exception:
                    # Fallback por si la tabla de selecciones no tiene algunas columnas menores
                    res_real = pd.DataFrame()
                
                if not res_real.empty:
                    equipos_posibles = res_real['HomeTeam'].tolist()
                    match_fuzz = process.extractOne(pick['HomeTeam'], equipos_posibles)
                    
                    if match_fuzz and match_fuzz[1] >= 75:
                        equipo_db_real = match_fuzz[0]
                        row = res_real[res_real['HomeTeam'] == equipo_db_real].iloc[0]
                        
                        hg = row['FTHG'] if pd.notna(row.get('FTHG')) else None
                        ag = row['FTAG'] if pd.notna(row.get('FTAG')) else None
                        
                        if hg is None or ag is None: continue
                            
                        hc = row['HC'] if 'HC' in row and pd.notna(row['HC']) else 0
                        ac = row['AC'] if 'AC' in row and pd.notna(row['AC']) else 0
                        hst = row['HST'] if 'HST' in row and pd.notna(row['HST']) else 0
                        ast = row['AST'] if 'AST' in row and pd.notna(row['AST']) else 0
                        
                        mkt = pick['Mercado']
                        ganada = False
                        
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
                
        df_hist = pd.read_sql("SELECT * FROM portafolio_historico", conn)

        # ── Importar portafolio externo ───────────────────────────
        st.subheader("📥 Importar Portafolio Guardado")
        with st.expander("Cargar archivo CSV o JSON para fusionar con el portafolio actual"):
            archivo_importar = st.file_uploader(
                "Sube un portafolio exportado (.csv o .json)",
                type=["csv", "json"],
                key="importar_portafolio"
            )
            if archivo_importar is not None:
                try:
                    if archivo_importar.name.endswith(".json"):
                        df_import = pd.read_json(archivo_importar)
                    else:
                        df_import = pd.read_csv(archivo_importar)

                    COLS_REQUERIDAS = {'Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado'}
                    if not COLS_REQUERIDAS.issubset(set(df_import.columns)):
                        st.error(f"❌ El archivo no tiene las columnas requeridas: {COLS_REQUERIDAS - set(df_import.columns)}")
                    else:
                        # Rellenar columnas opcionales si no vienen
                        if 'Beneficio_Neto' not in df_import.columns:
                            df_import['Beneficio_Neto'] = 0.0
                        if 'Prob_IA' not in df_import.columns:
                            df_import['Prob_IA'] = None
                        if 'Edge' not in df_import.columns:
                            df_import['Edge'] = None

                        # Preview
                        st.dataframe(df_import[['Date','HomeTeam','AwayTeam','Mercado','Cuota','Stake','Estado']].head(10), hide_index=True, use_container_width=True)
                        st.caption(f"Total filas en archivo: {len(df_import)}")

                        if st.button("⬆️ Fusionar con portafolio actual", type="primary"):
                            nuevos = 0
                            for _, row_i in df_import.iterrows():
                                # Evitar duplicados exactos (mismo partido + mercado + fecha)
                                existe = pd.read_sql(
                                    "SELECT COUNT(*) as n FROM portafolio_historico WHERE Date=? AND HomeTeam=? AND AwayTeam=? AND Mercado=?",
                                    conn,
                                    params=(str(row_i['Date']), str(row_i['HomeTeam']), str(row_i['AwayTeam']), str(row_i['Mercado']))
                                ).iloc[0]['n']
                                if existe == 0:
                                    cursor.execute(
                                        "INSERT INTO portafolio_historico (Date, HomeTeam, AwayTeam, Mercado, Cuota, Stake, Estado, Beneficio_Neto, Prob_IA, Edge) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                        (str(row_i['Date']), str(row_i['HomeTeam']), str(row_i['AwayTeam']), str(row_i['Mercado']),
                                         float(row_i['Cuota']), float(row_i['Stake']),
                                         str(row_i.get('Estado', 'Pendiente')), float(row_i.get('Beneficio_Neto', 0)),
                                         row_i.get('Prob_IA', None), row_i.get('Edge', None))
                                    )
                                    nuevos += 1
                            conn.commit()
                            st.success(f"✅ {nuevos} picks importados. {len(df_import) - nuevos} ya existían y se omitieron.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error al importar: {e}")

        st.divider()

        if not df_hist.empty:
            df_cerradas = df_hist[df_hist['Estado'] != 'Pendiente']
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            with c_res1:
                st.metric("Picks Cerrados", len(df_cerradas))
            with c_res2:
                ganadas = len(df_cerradas[df_cerradas['Estado'] == 'Ganada'])
                win_rate = (ganadas / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with c_res3:
                beneficio_total = df_cerradas['Beneficio_Neto'].sum()
                inversion_total_hist = df_cerradas['Stake'].sum()
                yield_global = (beneficio_total / inversion_total_hist * 100) if inversion_total_hist > 0 else 0
                st.metric("Yield (ROI)", f"{yield_global:.2f}%")
            with c_res4:
                st.metric("Ganancia Neta Global", f"${beneficio_total:,.0f}")

            st.divider()

            # ── Descarga del portafolio ───────────────────────────
            st.write("📋 **Historial de Picks**")
            col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1])
            df_dl = df_hist[['Date', 'HomeTeam', 'AwayTeam', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Beneficio_Neto', 'Prob_IA', 'Edge']].copy()

            with col_dl2:
                csv_bytes = df_dl.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv_bytes,
                    file_name=f"portafolio_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_dl3:
                json_bytes = df_dl.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_bytes,
                    file_name=f"portafolio_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
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
    st.title("Simulacion Mundial 2026")
    st.markdown("Proyección oficial basada en formato FIFA 48 selecciones.")

    st.markdown("""
    <style>
    .wc-group-header {
        font-size: 0.95rem; font-weight: 700; color: #fff;
        background: #2c2f3a; padding: 6px 10px;
        border-radius: 6px 6px 0 0; margin-bottom: 0;
    }
    .wc-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-bottom: 0; }
    .wc-table th { background: #1a1d26; color: #777; padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 1px solid #333; }
    .wc-table td { padding: 4px 6px; border-bottom: 1px solid #252830; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; }
    .wc-pos12 { background: #0a2e1a; }
    .wc-pos3  { background: #152a0a; }
    .wc-pos4  { background: #1e2129; color: #666; }
    .wc-matches { background: #14161e; border-radius: 0 0 6px 6px; padding: 5px 8px; margin-bottom: 16px; }
    .wc-match { font-size: 0.75rem; padding: 2px 0; color: #bbb; border-bottom: 1px solid #1e2129; }
    .wc-match:last-child { border-bottom: none; }
    .wc-win { color: #2ecc71; font-weight: 700; }
    .wc-draw { color: #f1c40f; }
    .bk-title {
        font-size: 0.72rem; font-weight: 700; color: #666;
        text-transform: uppercase; letter-spacing: 1.5px;
        text-align: center; margin: 4px 0 8px 0;
    }
    .bk-card {
        background: #1e2129; border-radius: 7px;
        border-left: 3px solid #2ecc71;
        padding: 7px 10px; margin-bottom: 7px; font-size: 0.82rem;
    }
    .bk-label { font-size: 0.63rem; color: #555; margin-bottom: 2px; }
    .bk-card.gold  { border-left-color: #f1c40f; }
    .bk-card.champ { border-left-color: #f39c12; background: #18150a; }
    .champ-name { font-size: 1.4rem; font-weight: 800; color: #f1c40f; text-align: center; padding: 4px 0; }
    </style>
    """, unsafe_allow_html=True)

    try:
        modelo_wc  = joblib.load('modelo_selecciones_rf.pkl')
        encoder_wc = joblib.load('encoder_equipos_selecciones.pkl')
        # Solo partidos de grupos (Grupo = GROUP_A … GROUP_L)
        df_fixture_wc = pd.read_sql(
            "SELECT * FROM fixture_mundial WHERE Grupo LIKE 'GROUP_%'", conn
        )
        # Columna auxiliar: solo la letra (GROUP_A → A)
        df_fixture_wc['_letra'] = df_fixture_wc['Grupo'].str.replace('GROUP_', '', regex=False).str.strip()

        # ── Motor de predicción ───────────────────────────────────────────

        # Diccionario de traducción de nombres de selecciones
        TRADUCCION_WC = {
            "Czechia": "Czech Republic", "South Korea": "Korea Republic",
            "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Cape Verde Islands": "Cape Verde",
            "Congo DR": "DR Congo", "USA": "United States"
        }

        # ── Carga del CSV FIFA 2026 ───────────────────────────────────────
        # Columnas: rank, country (en español), points, valor_total_mill_eur
        # Los nombres en español se traducen al inglés usado internamente.
        TRADUCCION_CSV = {
            "Francia": "France", "España": "Spain", "Argentina": "Argentina",
            "Inglaterra": "England", "Portugal": "Portugal", "Brasil": "Brazil",
            "Países Bajos": "Netherlands", "Marruecos": "Morocco", "Bélgica": "Belgium",
            "Alemania": "Germany", "Croacia": "Croatia", "Italia": "Italy",
            "Colombia": "Colombia", "Senegal": "Senegal", "México": "Mexico",
            "Estados Unidos": "United States", "Uruguay": "Uruguay", "Japón": "Japan",
            "Suiza": "Switzerland", "Dinamarca": "Denmark", "Irán": "Iran",
            "Turquía": "Turkey", "Ecuador": "Ecuador", "Austria": "Austria",
            "Corea del Sur": "Korea Republic", "Nigeria": "Nigeria", "Australia": "Australia",
            "Argelia": "Algeria", "Egipto": "Egypt", "Canadá": "Canada",
            "Noruega": "Norway", "Ucrania": "Ukraine", "Panamá": "Panama",
            "Costa de Marfil": "Ivory Coast", "Polonia": "Poland", "Rusia": "Russia",
            "Gales": "Wales", "Suecia": "Sweden", "Serbia": "Serbia",
            "Paraguay": "Paraguay", "Chequia": "Czech Republic", "Hungría": "Hungary",
            "Escocia": "Scotland", "Túnez": "Tunisia", "Camerún": "Cameroon",
            "RD Congo": "DR Congo", "Grecia": "Greece", "Eslovaquia": "Slovakia",
            "Venezuela": "Venezuela", "Uzbekistán": "Uzbekistan", "Costa Rica": "Costa Rica",
            "Malí": "Mali", "Perú": "Peru", "Chile": "Chile", "Catar": "Qatar",
            "Rumanía": "Romania", "Irak": "Iraq", "Eslovenia": "Slovenia",
            "Irlanda": "Ireland", "Sudáfrica": "South Africa", "Arabia Saudita": "Saudi Arabia",
            "Burkina Faso": "Burkina Faso", "Jordania": "Jordan", "Albania": "Albania",
            "Bosnia": "Bosnia and Herzegovina", "Honduras": "Honduras",
            "Macedonia Norte": "North Macedonia", "EAU": "United Arab Emirates",
            "Cabo Verde": "Cape Verde", "Irlanda Norte": "Northern Ireland",
            "Jamaica": "Jamaica", "Georgia": "Georgia", "Finlandia": "Finland",
            "Ghana": "Ghana", "Islandia": "Iceland", "Bolivia": "Bolivia",
            "Israel": "Israel", "Kosovo": "Kosovo", "Omán": "Oman",
            "Guinea": "Guinea", "Montenegro": "Montenegro", "Curazao": "Curaçao",
            "Haití": "Haiti", "Siria": "Syria", "Nueva Zelanda": "New Zealand",
            "Bulgaria": "Bulgaria", "Gabón": "Gabon", "Uganda": "Uganda",
            "Angola": "Angola", "Benín": "Benin", "Baréin": "Bahrain",
            "Zambia": "Zambia", "Tailandia": "Thailand", "China": "China",
            "Palestina": "Palestine", "Guatemala": "Guatemala",
            "Bielorrusia": "Belarus", "Luxemburgo": "Luxembourg",
            "Vietnam": "Vietnam", "El Salvador": "El Salvador",
        }

        try:
            df_fifa_ranking = pd.read_csv('fifa_ranking_2026.csv')
            df_fifa_ranking.columns = [c.strip() for c in df_fifa_ranking.columns]
            df_fifa_ranking['country_en'] = df_fifa_ranking['country'].str.strip().map(TRADUCCION_CSV)

            pts_min = df_fifa_ranking['points'].min()
            pts_max = df_fifa_ranking['points'].max()
            val_min = df_fifa_ranking['valor_total_mill_eur'].min()
            val_max = df_fifa_ranking['valor_total_mill_eur'].max()

            df_fifa_ranking['pts_norm']  = (df_fifa_ranking['points'] - pts_min) / (pts_max - pts_min)
            df_fifa_ranking['val_norm']  = (df_fifa_ranking['valor_total_mill_eur'] - val_min) / (val_max - val_min)

            FIFA_SCORES = dict(zip(df_fifa_ranking['country_en'], zip(
                df_fifa_ranking['pts_norm'], df_fifa_ranking['val_norm']
            )))
        except Exception:
            FIFA_SCORES = {}

        df_hist_wc = pd.read_sql("SELECT * FROM historial_selecciones_ml", conn)

        def _fuerza_seleccion(equipo):
            df_h = df_hist_wc[df_hist_wc['HomeTeam'] == equipo]
            df_a = df_hist_wc[df_hist_wc['AwayTeam'] == equipo]
            if df_h.empty and df_a.empty:
                return 4.0, 5.0
            hst = pd.concat([df_h['HST'], df_a['AST']]).mean() if not (df_h.empty and df_a.empty) else 4.0
            hc  = pd.concat([df_h['HC'],  df_a['AC']]).mean()  if not (df_h.empty and df_a.empty) else 5.0
            return hst, hc

        def predecir_wc(h_raw, a_raw):
            h = TRADUCCION_WC.get(h_raw, h_raw)
            a = TRADUCCION_WC.get(a_raw, a_raw)

            classes = list(encoder_wc.classes_)

            # 1. Predicción base del modelo (estadística pura)
            if h in classes and a in classes:
                hst, hc = _fuerza_seleccion(h)
                ast, ac = _fuerza_seleccion(a)
                h_c = encoder_wc.transform([h])[0]
                a_c = encoder_wc.transform([a])[0]
                # xG: promedio de partidos como local/visita, con defaults neutrales
                hist_h_home = df_hist_wc[df_hist_wc['HomeTeam'] == h]
                hist_a_away = df_hist_wc[df_hist_wc['AwayTeam'] == a]
                xg_h = hist_h_home['xG_home'].mean() if not hist_h_home.empty and hist_h_home['xG_home'].mean() > 0 else 1.2
                xg_a = hist_a_away['xG_away'].mean() if not hist_a_away.empty and hist_a_away['xG_away'].mean() > 0 else 1.0
                X = pd.DataFrame([[h_c, a_c, hst, ast, hc, ac, xg_h, xg_a]],
                                  columns=['HomeTeam_Code','AwayTeam_Code','HST','AST','HC','AC','xG_home','xG_away'])
                probs = modelo_wc.predict_proba(X)[0]
                # sklearn RandomForest ordena clases alfabéticamente: A(way)=0, D(raw)=1, H(ome)=2
                p_a_raw, p_d_raw, p_h_raw = float(probs[0]), float(probs[1]), float(probs[2])

                # Neutral venue correction: flatten home advantage (World Cup on neutral ground)
                NEUTRAL_FACTOR = 0.15  # reduces home advantage by 15%
                p_h_raw = p_h_raw * (1 - NEUTRAL_FACTOR) + p_d_raw * (NEUTRAL_FACTOR / 2)
                p_a_raw = p_a_raw * (1 - NEUTRAL_FACTOR) + p_d_raw * (NEUTRAL_FACTOR / 2)
                total = p_h_raw + p_d_raw + p_a_raw
                p_h_raw, p_d_raw, p_a_raw = p_h_raw/total, p_d_raw/total, p_a_raw/total
            else:
                p_h_raw, p_d_raw, p_a_raw = 0.33, 0.33, 0.33

            # 2. Score compuesto de 3 factores para medir la fuerza real de cada selección
            #
            #   - Puntos FIFA (40%): refleja el rendimiento reciente oficial
            #   - Valor de mercado (40%): proxy de calidad individual de plantilla
            #   - Bono confederación (20%): UEFA y CONMEBOL tienen mayor rodaje en mundiales
            #
            # Los tres factores se normalizan entre 0 y 1 y se combinan en un score único.
            # La diferencia de scores entre ambos equipos determina el multiplicador de probabilidad.

            UEFA_ES = {
                "France", "Spain", "England", "Germany", "Portugal", "Netherlands",
                "Italy", "Belgium", "Croatia", "Switzerland", "Denmark", "Austria",
                "Poland", "Serbia", "Ukraine", "Czech Republic", "Hungary", "Slovakia",
                "Romania", "Turkey", "Scotland", "Wales", "Greece", "Slovenia",
                "Albania", "Georgia", "Norway", "Sweden", "Finland",
                "Bosnia and Herzegovina", "North Macedonia", "Kosovo", "Montenegro",
                "Bulgaria", "Luxembourg", "Belarus", "Ireland", "Northern Ireland",
                "Iceland", "Israel"
            }
            CONMEBOL_ES = {
                "Argentina", "Brazil", "Uruguay", "Colombia", "Chile",
                "Ecuador", "Peru", "Paraguay", "Venezuela", "Bolivia"
            }

            def _score_seleccion(equipo):
                if equipo in FIFA_SCORES:
                    pts_n, val_n = FIFA_SCORES[equipo]
                else:
                    pts_n, val_n = 0.3, 0.05   # Equipo desconocido: valores bajos por defecto
                conf = 1.0 if equipo in UEFA_ES or equipo in CONMEBOL_ES else 0.0
                return 0.40 * pts_n + 0.40 * val_n + 0.20 * conf

            score_h = _score_seleccion(h)
            score_a = _score_seleccion(a)

            # Diferencia de scores: positivo = local es más fuerte
            diff_score = score_h - score_a

            # Multiplicador lineal: 1 punto de diferencia de score → 100% de ajuste
            # Con scores entre 0 y 1, el rango real es [-1, 1], dando multiplicadores de [0, 2]
            multiplicador_h = max(0.1, 1.0 + diff_score)
            multiplicador_a = max(0.1, 1.0 - diff_score)

            p_h_ajustada = p_h_raw * multiplicador_h
            p_a_ajustada = p_a_raw * multiplicador_a
            p_d_ajustada = p_d_raw * 0.9  # Reducimos ligeramente empates cuando hay diferencia de nivel

            # Normalizar para que sumen 1
            suma_probs = p_h_ajustada + p_a_ajustada + p_d_ajustada
            p_h = p_h_ajustada / suma_probs
            p_a = p_a_ajustada / suma_probs
            p_d = p_d_ajustada / suma_probs

            # 3. Cálculo de goles esperados (base para decidir resultado Y marcador)
            hist_h = df_hist_wc[df_hist_wc['HomeTeam'] == h]
            hist_a = df_hist_wc[df_hist_wc['AwayTeam'] == a]

            gf_h = hist_h['FTHG'].mean() if not hist_h.empty else 1.5
            gc_h = hist_h['FTAG'].mean() if not hist_h.empty else 1.0
            gf_a = hist_a['FTAG'].mean() if not hist_a.empty else 1.2
            gc_a = hist_a['FTHG'].mean() if not hist_a.empty else 1.5

            xg_home = (gf_h + gc_a) / 2
            xg_away = (gf_a + gc_h) / 2

            # Si un equipo tiene probabilidad aplastante (>60%), potenciamos sus goles
            if p_h > 0.60:
                xg_home += 1.0
                xg_away = max(0, xg_away - 0.5)
            elif p_a > 0.60:
                xg_away += 1.0
                xg_home = max(0, xg_home - 0.5)

            # 4. Resultado: el xG diferencial pondera si un empate es creíble.
            #
            # La diferencia de goles esperados mide cuánto separa realmente a los equipos.
            # - Si la diferencia es pequeña (< 0.4): el empate compite con el favorito.
            # - Si la diferencia es grande (> 0.9): el favorito gana con claridad.
            # - En el rango intermedio: se pondera suavemente entre ambos extremos.
            #
            # Para evitar que el modelo siempre elija el argmax (que ignora p_d),
            # boost_draw escala p_d según lo reñido que está el xG, y luego
            # volvemos a normalizar antes de decidir.

            xg_diff = abs(xg_home - xg_away)

            if xg_diff < 0.4:
                boost_draw = 1.6    # Partido muy igualado: el empate es muy probable
            elif xg_diff < 0.9:
                boost_draw = 1.2    # Ligera ventaja: el empate sigue siendo opción real
            else:
                boost_draw = 0.8    # Clara ventaja para uno: el empate pierde peso

            p_h_w = p_h
            p_a_w = p_a
            p_d_w = p_d * boost_draw
            total_w = p_h_w + p_a_w + p_d_w
            p_h_w /= total_w
            p_a_w /= total_w
            p_d_w /= total_w

            if p_d_w >= p_h_w and p_d_w >= p_a_w:
                pts_h, pts_a = 1, 1
            elif p_h_w >= p_a_w:
                pts_h, pts_a = 3, 0
            else:
                pts_h, pts_a = 0, 3

            # 5. Blindaje del marcador: ajustar xG flotantes antes de redondear
            if pts_h == 3 and xg_home <= xg_away:
                xg_home = xg_away + 0.6
            elif pts_a == 3 and xg_away <= xg_home:
                xg_away = xg_home + 0.6
            elif pts_h == 1 and pts_a == 1:
                avg = (xg_home + xg_away) / 2
                xg_home = xg_away = avg

            # Redondeamos una sola vez, al final, sobre los valores ya corregidos
            gh = int(round(xg_home))
            ga = int(round(xg_away))

            # Garantía residual post-redondeo
            if pts_h == 3 and gh <= ga:
                gh = ga + 1
            elif pts_a == 3 and ga <= gh:
                ga = gh + 1
            elif pts_h == 1 and pts_a == 1:
                gh = ga = max(gh, ga)

            return {
                "Pts_H": pts_h, "Pts_A": pts_a,
                "GH": gh, "GA": ga,
                "p_h": p_h, "p_a": p_a,
                "Ganador": h if p_h >= p_a else a
            }

        # ═════════════════════════════════════════════════════════════════
        #  FASE DE GRUPOS — calcular todo antes de los tabs
        # ═════════════════════════════════════════════════════════════════
        grupos_keys = list('ABCDEFGHIJKL')
        posiciones  = {}          # "1A" → equipo, "2B" → equipo …
        terceros    = []          # (pts, dg, gf, letra, equipo)
        grupos_data = {}          # letra → {tabla: df, partidos: list}

        for letra in grupos_keys:
            df_g = df_fixture_wc[df_fixture_wc['_letra'] == letra].copy()
            if df_g.empty:
                continue

            tabla    = {}
            partidos = []

            for _, p in df_g.iterrows():
                h, a = str(p['HomeTeam']).strip(), str(p['AwayTeam']).strip()
                res  = predecir_wc(h, a)

                for eq in (h, a):
                    if eq not in tabla:
                        tabla[eq] = {'Pts': 0, 'GF': 0, 'GC': 0, 'PJ': 0}

                tabla[h]['Pts'] += res['Pts_H']; tabla[h]['GF'] += res['GH']
                tabla[h]['GC']  += res['GA'];    tabla[h]['PJ'] += 1
                tabla[a]['Pts'] += res['Pts_A']; tabla[a]['GF'] += res['GA']
                tabla[a]['GC']  += res['GH'];    tabla[a]['PJ'] += 1
                partidos.append({'H': h, 'A': a, 'GH': res['GH'], 'GA': res['GA']})

            df_t = pd.DataFrame([
                {'Sel': eq, 'PJ': v['PJ'], 'Pts': v['Pts'],
                 'GF': v['GF'], 'GC': v['GC'], 'DG': v['GF'] - v['GC']}
                for eq, v in tabla.items()
            ]).sort_values(['Pts','DG','GF'], ascending=False).reset_index(drop=True)

            for ri, row in df_t.iterrows():
                posiciones[f"{ri+1}{letra}"] = row['Sel']
            if len(df_t) >= 3:
                t = df_t.iloc[2]
                terceros.append((t['Pts'], t['DG'], t['GF'], letra, t['Sel']))

            grupos_data[letra] = {'tabla': df_t, 'partidos': partidos}

        # ── 8 mejores terceros ────────────────────────────────────────────
        terceros.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        mejores_t8 = {x[4] for x in terceros[:8]}

        def get_pos(c):
            return posiciones.get(c, f"({c})")

        def mejor_3ro(grupos_str):
            candidatos = {g.strip() for g in grupos_str.split('/')}
            for _, _, _, grp, eq in terceros:
                if grp in candidatos and eq in mejores_t8:
                    return eq
            return f"3°({grupos_str})"

        # ═════════════════════════════════════════════════════════════════
        tab_g, tab_f = st.tabs(["📊 Grupos", "⚔️ Ruta a la Copa"])
        # ═════════════════════════════════════════════════════════════════

        with tab_g:
            st.markdown(
                '<div style="font-size:0.77rem;color:#888;margin-bottom:12px;">'
                '<span style="background:#0a2e1a;padding:2px 8px;border-radius:3px;margin-right:8px;">1° / 2°</span> Clasificado directo &nbsp;'
                '<span style="background:#152a0a;padding:2px 8px;border-radius:3px;margin-right:8px;">3°</span> Posible mejor tercero'
                '</div>', unsafe_allow_html=True
            )

            for batch_start in range(0, 12, 3):
                letras_batch = grupos_keys[batch_start:batch_start+3]
                cols = st.columns(3)
                for ci, letra in enumerate(letras_batch):
                    if letra not in grupos_data:
                        continue
                    df_t     = grupos_data[letra]['tabla']
                    partidos = grupos_data[letra]['partidos']

                    with cols[ci]:
                        # — Tabla —
                        filas = ""
                        for ri, row in df_t.iterrows():
                            if ri < 2:
                                css, ico = "wc-pos12", "🟢"
                            elif ri == 2 and row['Sel'] in mejores_t8:
                                css, ico = "wc-pos3", "🟡"
                            elif ri == 2:
                                css, ico = "wc-pos3", "⚪"
                            else:
                                css, ico = "wc-pos4", ""
                            dg = f"+{int(row['DG'])}" if row['DG'] > 0 else str(int(row['DG']))
                            # Truncar nombre largo
                            nombre = row['Sel'][:16] + ("…" if len(row['Sel']) > 16 else "")
                            filas += (
                                f'<tr class="{css}">'
                                f'<td style="color:#666;width:18px">{ri+1}</td>'
                                f'<td>{ico} {nombre}</td>'
                                f'<td style="text-align:center;width:28px"><b>{int(row["Pts"])}</b></td>'
                                f'<td style="text-align:center;width:28px;color:#888">{dg}</td>'
                                f'</tr>'
                            )
                        st.markdown(
                            f'<div class="wc-group-header">Grupo {letra}</div>'
                            f'<table class="wc-table"><thead><tr>'
                            f'<th>#</th><th>Selección</th>'
                            f'<th style="text-align:center">Pts</th>'
                            f'<th style="text-align:center">DG</th>'
                            f'</tr></thead><tbody>{filas}</tbody></table>',
                            unsafe_allow_html=True
                        )

                        # — Resultados —
                        html_m = '<div class="wc-matches">'
                        for m in partidos:
                            gh, ga = m['GH'], m['GA']
                            h_n = m['H'][:13] + ("…" if len(m['H']) > 13 else "")
                            a_n = m['A'][:13] + ("…" if len(m['A']) > 13 else "")
                            if gh > ga:
                                s = f'<span class="wc-win">{h_n} {gh}</span>–{ga} {a_n}'
                            elif ga > gh:
                                s = f'{h_n} {gh}–<span class="wc-win">{ga} {a_n}</span>'
                            else:
                                s = f'{h_n} <span class="wc-draw">{gh}–{ga}</span> {a_n}'
                            html_m += f'<div class="wc-match">{s}</div>'
                        html_m += '</div>'
                        st.markdown(html_m, unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════
        with tab_f:
        # ═════════════════════════════════════════════════════════════════

            def simular_ko(h, a, label=""):
                res = predecir_wc(h, a)
                gh, ga = res['GH'], res['GA']
                # Sin empate en KO
                if gh == ga:
                    if res['p_h'] >= res['p_a']: gh += 1
                    else: ga += 1
                ganador = h if gh > ga else a
                return {'H': h, 'A': a, 'GH': gh, 'GA': ga,
                        'Ganador': ganador, 'Label': label}

            def card(m, extra_css=""):
                h, a, gh, ga = m['H'], m['A'], m['GH'], m['GA']
                lbl = m.get('Label', '')
                hn = h[:16] + ("…" if len(h) > 16 else "")
                an = a[:16] + ("…" if len(a) > 16 else "")
                if gh > ga:
                    sc = f'<span class="wc-win">{hn}</span> <span style="color:#aaa">{gh}–{ga}</span> {an}'
                else:
                    sc = f'{hn} <span style="color:#aaa">{gh}–{ga}</span> <span class="wc-win">{an}</span>'
                return (f'<div class="bk-card {extra_css}">'
                        f'<div class="bk-label">{lbl}</div>'
                        f'<div>{sc}</div></div>')

            def resolver(c):
                return mejor_3ro(c[2:]) if c.startswith("3_") else get_pos(c)

            # ── Ronda de 32 ──────────────────────────────────────────────
            cruces_r32 = [
                ("2A",  "2B",           "2°A vs 2°B"),
                ("1C",  "2F",           "1°C vs 2°F"),
                ("1E",  "3_A/B/C/D/F",  "1°E vs 3°A/B/C/D/F"),
                ("1F",  "2C",           "1°F vs 2°C"),
                ("2E",  "2I",           "2°E vs 2°I"),
                ("1I",  "3_C/D/F/G/H",  "1°I vs 3°C/D/F/G/H"),
                ("1A",  "3_C/E/F/H/I",  "1°A vs 3°C/E/F/H/I"),
                ("1L",  "3_E/H/I/J/K",  "1°L vs 3°E/H/I/J/K"),
                ("1G",  "3_A/E/H/I/J",  "1°G vs 3°A/E/H/I/J"),
                ("1D",  "3_B/E/F/I/J",  "1°D vs 3°B/E/F/I/J"),
                ("1H",  "2J",           "1°H vs 2°J"),
                ("2K",  "2L",           "2°K vs 2°L"),
                ("1B",  "3_E/F/G/I/J",  "1°B vs 3°E/F/G/I/J"),
                ("2D",  "2G",           "2°D vs 2°G"),
                ("1K",  "2H",           "1°K vs 2°H"),
                ("1J",  "2E",           "1°J vs 2°E"),
            ]
            r32 = [simular_ko(resolver(ch), resolver(ca), lbl) for ch, ca, lbl in cruces_r32]
            gana_r32 = [m['Ganador'] for m in r32]

            # ── Ronda de 16 ──────────────────────────────────────────────
            # Cruces por pares consecutivos del bracket
            r16 = [simular_ko(gana_r32[i], gana_r32[i+1], f"R16 · P{i//2+1}")
                   for i in range(0, 16, 2)]
            gana_r16 = [m['Ganador'] for m in r16]

            # ── Cuartos ───────────────────────────────────────────────────
            qf = [simular_ko(gana_r16[i], gana_r16[i+1], f"Cuartos · {i//2+1}")
                  for i in range(0, 8, 2)]
            gana_qf = [m['Ganador'] for m in qf]

            # ── Semis ─────────────────────────────────────────────────────
            sf = [
                simular_ko(gana_qf[0], gana_qf[1], "Semifinal 1"),
                simular_ko(gana_qf[2], gana_qf[3], "Semifinal 2"),
            ]
            gana_sf = [m['Ganador'] for m in sf]
            pierde_sf = [
                (sf[0]['A'] if sf[0]['Ganador'] == sf[0]['H'] else sf[0]['H']),
                (sf[1]['A'] if sf[1]['Ganador'] == sf[1]['H'] else sf[1]['H']),
            ]

            # ── Final y 3° puesto ─────────────────────────────────────────
            tercer = simular_ko(pierde_sf[0], pierde_sf[1], "3° Puesto")
            final  = simular_ko(gana_sf[0],   gana_sf[1],   "⚽ Gran Final")
            campeon = final['Ganador']

            # ══════════════════════════════════════════════════════════════
            #  RENDER — de abajo a arriba: Campeón → Final → Semis → …
            # ══════════════════════════════════════════════════════════════

            # Campeón
            st.markdown(
                f'<div class="bk-card champ" style="max-width:340px;margin:0 auto 18px;">'
                f'<div style="text-align:center;color:#888;font-size:0.7rem;letter-spacing:1px;">🏆 CAMPEÓN PROYECTADO</div>'
                f'<div class="champ-name">🏆 {campeon}</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown("---")

            # Final + 3° puesto
            cf1, cf2 = st.columns(2)
            with cf1:
                st.markdown('<div class="bk-title">⚽ Gran Final</div>', unsafe_allow_html=True)
                st.markdown(card(final, "gold"), unsafe_allow_html=True)
            with cf2:
                st.markdown('<div class="bk-title">🥉 3° Puesto</div>', unsafe_allow_html=True)
                st.markdown(card(tercer), unsafe_allow_html=True)
            st.markdown("---")

            # Semis
            st.markdown('<div class="bk-title">Semifinales</div>', unsafe_allow_html=True)
            csf = st.columns(2)
            for i, m in enumerate(sf):
                with csf[i]:
                    st.markdown(card(m), unsafe_allow_html=True)
            st.markdown("---")

            # Cuartos
            st.markdown('<div class="bk-title">Cuartos de Final</div>', unsafe_allow_html=True)
            cqf = st.columns(4)
            for i, m in enumerate(qf):
                with cqf[i]:
                    st.markdown(card(m), unsafe_allow_html=True)
            st.markdown("---")

            # R16
            st.markdown('<div class="bk-title">Ronda de 16</div>', unsafe_allow_html=True)
            cr16 = st.columns(4)
            for i, m in enumerate(r16):
                with cr16[i % 4]:
                    st.markdown(card(m), unsafe_allow_html=True)
            st.markdown("---")

            # R32 en expander
            with st.expander("▶ Ver Ronda de 32 (16 partidos)"):
                cr32 = st.columns(4)
                for i, m in enumerate(r32):
                    with cr32[i % 4]:
                        st.markdown(card(m), unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error(f"Error en Oráculo Mundial: {e}")
        st.code(traceback.format_exc())
conn.close()