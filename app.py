import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

st.set_page_config(page_title="AI Football Predictor", page_icon="⚽", layout="wide")

st.title("⚽ AI Football Analyzer & Predictor")
st.caption("Calculadora de Probabilidades y Marcador Exacto mediante Distribución de Poisson")

# Sidebar - Parámetros de Entrada
st.sidebar.header("📊 Datos del Partido")

home_team = st.sidebar.text_input("Equipo Local", "Real Madrid")
away_team = st.sidebar.text_input("Equipo Visitante", "Barcelona")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    home_xg = col_a.number_input("xG Local", min_value=0.1, max_value=5.0, value=2.1, step=0.1)
with col_b:
    away_xg = col_b.number_input("xG Visitante", min_value=0.1, max_value=5.0, value=1.4, step=0.1)

# Simulación de Matriz de Marcadores
max_goals = 6
matrix = np.zeros((max_goals, max_goals))

for i in range(max_goals):
    for j in range(max_goals):
        prob_home = stats.poisson.pmf(i, home_xg)
        prob_away = stats.poisson.pmf(j, away_xg)
        matrix[i, j] = prob_home * prob_away

# Cálculo de Probabilidades de Resultados (1X2)
prob_home_win = np.sum(np.tril(matrix, -1)) * 100
prob_draw = np.sum(np.diag(matrix)) * 100
prob_away_win = np.sum(np.triu(matrix, 1)) * 100

# Cálculo de BTTS (Ambos Anotan)
btts_yes = (1 - stats.poisson.pmf(0, home_xg)) * (1 - stats.poisson.pmf(0, away_xg)) * 100
btts_no = 100 - btts_yes

# Total de Goles (Over/Under 2.5)
total_goals_prob = 0
for i in range(max_goals):
    for j in range(max_goals):
        if i + j > 2.5:
            total_goals_prob += matrix[i, j]
over_25 = total_goals_prob * 100
under_25 = 100 - over_25

# --- INTERFAZ DE USUARIO ---
st.subheader(f"Pronóstico: **{home_team}** vs **{away_team}**")

col1, col2, col3 = st.columns(3)
col1.metric("Victoria Local (1)", f"{prob_home_win:.1f}%", f"Cuota justa: {100/prob_home_win:.2f}")
col2.metric("Empate (X)", f"{prob_draw:.1f}%", f"Cuota justa: {100/prob_draw:.2f}")
col3.metric("Victoria Visitante (2)", f"{prob_away_win:.1f}%", f"Cuota justa: {100/prob_away_win:.2f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.write("### 🎯 Mercados Especiales")
    st.write(f"**Ambos Equipos Anotan (BTTS):** {btts_yes:.1f}% Sí | {btts_no:.1f}% No")
    st.write(f"**Más de 2.5 Goles (Over 2.5):** {over_25:.1f}%")
    st.write(f"**Menos de 2.5 Goles (Under 2.5):** {under_25:.1f}%")

with col_right:
    st.write("### 🥅 Marcadores Exactos Más Probables")
    scores = []
    for i in range(4):
        for j in range(4):
            scores.append({"Marcador": f"{i} - {j}", "Probabilidad (%)": round(matrix[i, j] * 100, 2)})
    
    df_scores = pd.DataFrame(scores).sort_values(by="Probabilidad (%)", ascending=False).head(5)
    st.dataframe(df_scores, hide_index=True, use_container_width=True)
