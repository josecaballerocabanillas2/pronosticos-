import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

st.set_page_config(page_title="AI Football Predictor PRO", page_icon="⚽", layout="wide")

st.title("⚽ AI Football Predictor & Live Analyzer")
st.caption("Calculadora Integral Pre-Partido y En Vivo (Distribución de Poisson)")

# --- SIDEBAR PRE-PARTIDO ---
st.sidebar.header("📊 Datos Pre-Partido")

home_team = st.sidebar.text_input("Equipo Local", "FC Porto")
away_team = st.sidebar.text_input("Equipo Visitante", "Manchester City")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    home_xg = col_a.number_input("xG Local (90m)", min_value=0.1, max_value=6.0, value=1.1, step=0.1)
with col_b:
    away_xg = col_b.number_input("xG Visitante (90m)", min_value=0.1, max_value=6.0, value=1.9, step=0.1)

home_xg_1h = home_xg * 0.45
away_xg_1h = away_xg * 0.45

max_g = 8

# Matriz Pre-Partido
matrix = np.zeros((max_g, max_g))
for i in range(max_g):
    for j in range(max_g):
        matrix[i, j] = stats.poisson.pmf(i, home_xg) * stats.poisson.pmf(j, away_xg)

matrix_1h = np.zeros((max_g, max_g))
for i in range(max_g):
    for j in range(max_g):
        matrix_1h[i, j] = stats.poisson.pmf(i, home_xg_1h) * stats.poisson.pmf(j, away_xg_1h)

def get_1x2(m):
    p1 = np.sum(np.tril(m, -1)) * 100
    px = np.sum(np.diag(m)) * 100
    p2 = np.sum(np.triu(m, 1)) * 100
    return p1, px, p2

def get_ou(m, line):
    over = np.sum([m[i, j] for i in range(max_g) for j in range(max_g) if i + j > line]) * 100
    under = 100 - over
    return over, under

p1, px, p2 = get_1x2(matrix)
btts_yes = (1 - stats.poisson.pmf(0, home_xg)) * (1 - stats.poisson.pmf(0, away_xg)) * 100
btts_no = 100 - btts_yes
p1_1h, px_1h, p2_1h = get_1x2(matrix_1h)

# --- PESTAÑAS ---
tab_live, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ ANÁLISIS EN VIVO",
    "🏆 Principales (1X2)", 
    "⚽ Goles & BTTS", 
    "🛡️ Hándicaps & Márgenes", 
    "⏱️ 1ª / 2ª Parte", 
    "🎯 Marcadores Exactos"
])

# --- TAB EN VIVO ---
with tab_live:
    st.subheader("⚡ Calculadora de Proyecciones En Vivo")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        minute = st.slider("Minuto del Partido", min_value=1, max_value=89, value=60)
    with col_l2:
        current_home_goals = st.number_input(f"Goles {home_team}", min_value=0, max_value=10, value=0)
    with col_l3:
        current_away_goals = st.number_input(f"Goles {away_team}", min_value=0, max_value=10, value=1)

    red_card = st.selectbox("Expulsiones (Tarjetas Rojas)", ["Ninguna", f"Roja para {home_team}", f"Roja para {away_team}"])

    time_remaining_ratio = (90 - minute) / 90.0

    # Ajuste de xG según minutos restantes
    rem_home_xg = home_xg * time_remaining_ratio
    rem_away_xg = away_xg * time_remaining_ratio

    # Factor de expulsión (-30% de peligro de gol para el expulsado, +15% para el rival)
    if red_card == f"Roja para {home_team}":
        rem_home_xg *= 0.7
        rem_away_xg *= 1.15
    elif red_card == f"Roja para {away_team}":
        rem_away_xg *= 0.7
        rem_home_xg *= 1.15

    # Matriz para los goles restantes
    matrix_live = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matrix_live[i, j] = stats.poisson.pmf(i, rem_home_xg) * stats.poisson.pmf(j, rem_away_xg)

    # Probabilidades de marcador final
    live_home_win = 0.0
    live_draw = 0.0
    live_away_win = 0.0
    live_over25 = 0.0

    for i in range(max_g):
        for j in range(max_g):
            final_h = current_home_goals + i
            final_a = current_away_goals + j
            p = matrix_live[i, j]
            if final_h > final_a:
                live_home_win += p
            elif final_h == final_a:
                live_draw += p
            else:
                live_away_win += p

            if (final_h + final_a) > 2.5:
                live_over25 += p

    live_home_win *= 100
    live_draw *= 100
    live_away_win *= 100
    live_over25 *= 100

    st.markdown("---")
    st.markdown(f"### 📊 Probabilidades Restantes (Minuto {minute}')")
    
    cl1, cl2, cl3 = st.columns(3)
    cl1.metric(f"Victoria {home_team}", f"{live_home_win:.1f}%", f"Cuota: {100/max(live_home_win,0.1):.2f}")
    cl2.metric("Empate Final", f"{live_draw:.1f}%", f"Cuota: {100/max(live_draw,0.1):.2f}")
    cl3.metric(f"Victoria {away_team}", f"{live_away_win:.1f}%", f"Cuota: {100/max(live_away_win,0.1):.2f}")

    st.markdown("### ⚽ Proyección de Goles Finales")
    st.write(f"**Más de 2.5 Goles en el Partido:** {live_over25:.1f}% | Cuota Justa: **{100/max(live_over25,0.1):.2f}**")
    st.write(f"**Menos de 2.5 Goles en el Partido:** {100-live_over25:.1f}% | Cuota Justa: **{100/max(100-live_over25,0.1):.2f}**")

# TAB 1
with tab1:
    st.subheader(f"Pronóstico Pre-Partido: {home_team} vs {away_team}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Victoria {home_team} (1)", f"{p1:.1f}%", f"Cuota: {100/max(p1,0.1):.2f}")
    c2.metric("Empate (X)", f"{px:.1f}%", f"Cuota: {100/max(px,0.1):.2f}")
    c3.metric(f"Victoria {away_team} (2)", f"{p2:.1f}%", f"Cuota: {100/max(p2,0.1):.2f}")

# TAB 2
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Líneas de Goles")
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            ov, un = get_ou(matrix, line)
            st.write(f"**Más de {line}:** {ov:.1f}% (Cuota {100/max(ov,0.1):.2f})")
    with col2:
        st.markdown("#### Ambos Anotan")
        st.write(f"**BTTS Sí:** {btts_yes:.1f}% | Cuota: **{100/max(btts_yes,0.1):.2f}**")

# TAB 3
with tab3:
    h_minus_15 = np.sum([matrix[i, j] for i in range(max_g) for j in range(max_g) if i - j > 1.5]) * 100
    st.write(f"**{home_team} -1.5:** {h_minus_15:.1f}% | Cuota: **{100/max(h_minus_15,0.1):.2f}**")

# TAB 4
with tab4:
    st.write(f"**1ª Parte Local:** {p1_1h:.1f}% | **Empate:** {px_1h:.1f}% | **Visitante:** {p2_1h:.1f}%")

# TAB 5
with tab5:
    scores = []
    for i in range(5):
        for j in range(5):
            prob = matrix[i, j] * 100
            scores.append({"Marcador": f"{i} - {j}", "Probabilidad (%)": round(prob, 2)})
    df_scores = pd.DataFrame(scores).sort_values(by="Probabilidad (%)", ascending=False).head(8)
    st.dataframe(df_scores, hide_index=True, use_container_width=True)
