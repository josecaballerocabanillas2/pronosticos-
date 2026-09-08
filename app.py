import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

st.set_page_config(
    page_title="QUANT FOOTBALL ENGINE v3", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ QUANT FOOTBALL ENGINE v3 — INSTITUTIONAL EDITION")
st.caption("Motor de Inferencia Cuantitativa: Monte Carlo (10k Runs), Dixon-Coles Correlated Poisson & Dynamic Game-State")

# ==========================================
# 1. MOTOR MATEMÁTICO AVANZADO
# ==========================================

def dixon_coles_tau(x: int, y: int, lambda_x: float, mu_y: float, rho: float = -0.13) -> float:
    """Ajuste de dependencia bivariada para marcadores de baja anotación."""
    if x == 0 and y == 0:
        return 1.0 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def build_dixon_coles_matrix(home_xg: float, away_xg: float, max_goals: int = 8, rho: float = -0.13) -> np.ndarray:
    """Construye y normaliza la matriz de densidad de probabilidad."""
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_i = stats.poisson.pmf(i, home_xg)
            p_j = stats.poisson.pmf(j, away_xg)
            tau = dixon_coles_tau(i, j, home_xg, away_xg, rho)
            matrix[i, j] = p_i * p_j * tau
    
    matrix = np.clip(matrix, 0, None)
    total_p = np.sum(matrix)
    return matrix / total_p if total_p > 0 else matrix

def run_monte_carlo_simulation(rem_home_xg: float, rem_away_xg: float, current_h: int, current_a: int, num_sims: int = 10000):
    """Simula el resto del partido N veces usando distribuciones de Poisson independientes por tiro/tiempo."""
    sim_h_goals = stats.poisson.rvs(rem_home_xg, size=num_sims)
    sim_a_goals = stats.poisson.rvs(rem_away_xg, size=num_sims)
    
    final_h = current_h + sim_h_goals
    final_a = current_a + sim_a_goals
    
    home_wins = np.sum(final_h > final_a) / num_sims * 100
    draws = np.sum(final_h == final_a) / num_sims * 100
    away_wins = np.sum(final_h < final_a) / num_sims * 100
    
    over_25 = np.sum((final_h + final_a) > 2.5) / num_sims * 100
    btts = np.sum((final_h > 0) & (final_a > 0)) / num_sims * 100
    
    return home_wins, draws, away_wins, over_25, btts

# ==========================================
# 2. PANEL LATERAL: CONFIGURACIÓN DE PARÁMETROS
# ==========================================

st.sidebar.header("🎛️ Parámetros del Encuentro")

home_team = st.sidebar.text_input("Equipo Local", "Marítimo Sub-23")
away_team = st.sidebar.text_input("Equipo Visitante", "Gil Vicente Sub-23")

st.sidebar.subheader("📈 Métricas de Rendimiento (Expected Goals)")
col_xg1, col_xg2 = st.sidebar.columns(2)
home_xg = col_xg1.number_input("xG Local (90m)", min_value=0.05, max_value=6.0, value=1.45, step=0.05)
away_xg = col_xg2.number_input("xG Visitante (90m)", min_value=0.05, max_value=6.0, value=1.65, step=0.05)

st.sidebar.subheader("⚙️ Calibración del Modelo")
league_avg_xg = st.sidebar.slider("Promedio de Goles de la Liga", 2.0, 3.8, 2.7, 0.1)
rho_param = st.sidebar.slider("Factor de Correlación (Rho)", -0.30, 0.00, -0.13, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("🏦 Capital & Risk Management")
bankroll = st.sidebar.number_input("Banca (€)", min_value=10.0, value=500.0, step=50.0)
kelly_fraction = st.sidebar.slider("Fracción de Kelly (Riesgo Controlado)", 0.05, 0.50, 0.15, 0.05)

# Matriz Pre-Partido
matrix_pre = build_dixon_coles_matrix(home_xg, away_xg, rho=rho_param)

# ==========================================
# 3. MÓDULOS Y PESTAÑAS
# ==========================================

tab_ev, tab_live, tab_montecarlo, tab_matrix = st.tabs([
    "🎯 Scanner +EV & Stake Kelly",
    "⚡ Análisis Live (Game-State)",
    "🎲 Simulación Monte Carlo (10k)",
    "📊 Matriz & Marcadores"
])

# ------------------------------------------
# TAB 1: SCANNER +EV & KELLY
# ------------------------------------------
with tab_ev:
    st.subheader("💡 Detección de Valor (+EV) y Gestión de Capital")
    
    p1_pre = np.sum(np.tril(matrix_pre, -1)) * 100
    px_pre = np.sum(np.diag(matrix_pre)) * 100
    p2_pre = np.sum(np.triu(matrix_pre, 1)) * 100

    col_o1, col_o2, col_o3 = st.columns(3)
    odd_1 = col_o1.number_input(f"Cuota {home_team}", min_value=1.01, value=2.60, step=0.01)
    odd_x = col_o2.number_input("Cuota Empate", min_value=1.01, value=3.20, step=0.01)
    odd_2 = col_o3.number_input(f"Cuota {away_team}", min_value=1.01, value=2.50, step=0.01)

    # EV y Kelly
    def calc_ev_and_stake(prob_pct, odd, bank, frac):
        p = prob_pct / 100.0
        ev = (p * odd) - 1.0
        b = odd - 1.0
        q = 1.0 - p
        f_star = (b * p - q) / b
        stake = max(0.0, f_star * bank * frac) if f_star > 0 else 0.0
        return ev, stake

    ev1, stake1 = calc_ev_and_stake(p1_pre, odd_1, bankroll, kelly_fraction)
    evx, stakex = calc_ev_and_stake(px_pre, odd_x, bankroll, kelly_fraction)
    ev2, stake2 = calc_ev_and_stake(p2_pre, odd_2, bankroll, kelly_fraction)

    st.markdown("---")
    
    data_ev = [
        {"Selección": home_team, "Prob. Modelo": f"{p1_pre:.2f}%", "Cuota Justa": f"{100/p1_pre:.2f}", "Cuota Casa": f"{odd_1:.2f}", "EV (%)": f"{ev1*100:+.2f}%", "Stake Recomendado": f"{stake1:.2f} €" if ev1 > 0 else "Sin Apuesta", "Estatus": "🔥 +EV DETECTADO" if ev1 > 0 else "⚪ Sin Valor"},
        {"Selección": "Empate", "Prob. Modelo": f"{px_pre:.2f}%", "Cuota Justa": f"{100/px_pre:.2f}", "Cuota Casa": f"{odd_x:.2f}", "EV (%)": f"{evx*100:+.2f}%", "Stake Recomendado": f"{stakex:.2f} €" if evx > 0 else "Sin Apuesta", "Estatus": "🔥 +EV DETECTADO" if evx > 0 else "⚪ Sin Valor"},
        {"Selección": away_team, "Prob. Modelo": f"{p2_pre:.2f}%", "Cuota Justa": f"{200/p2_pre:.2f}", "Cuota Casa": f"{odd_2:.2f}", "EV (%)": f"{ev2*100:+.2f}%", "Stake Recomendado": f"{stake2:.2f} €" if ev2 > 0 else "Sin Apuesta", "Estatus": "🔥 +EV DETECTADO" if ev2 > 0 else "⚪ Sin Valor"}
    ]
    st.dataframe(pd.DataFrame(data_ev), hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 2: ANÁLISIS EN VIVO (GAME-STATE)
# ------------------------------------------
with tab_live:
    st.subheader("⚡ Ajuste Dinámico por Game-State y Red Cards")
    
    l1, l2, l3 = st.columns(3)
    minuto = l1.slider("Minuto Actual", 1, 89, 37)
    goles_h = l2.number_input(f"Goles {home_team}", 0, 10, 0)
    goles_a = l3.number_input(f"Goles {away_team}", 0, 10, 1)

    expulsiones = st.radio("Tarjetas Rojas", ["Ninguna", f"Roja {home_team}", f"Roja {away_team}"], horizontal=True)

    # Tiempo restante y decay
    ratio_tiempo = (90 - minuto) / 90.0
    rem_h_xg = home_xg * ratio_tiempo
    rem_a_xg = away_xg * ratio_tiempo

    # Corrección por ventaja táctica (Game-State)
    diff = goles_h - goles_a
    if diff > 0:  # Local ganando
        rem_h_xg *= 0.85
        rem_a_xg *= 1.20
    elif diff < 0:  # Visitante ganando
        rem_h_xg *= 1.20
        rem_a_xg *= 0.85

    # Corrección por expulsión
    if expulsiones == f"Roja {home_team}":
        rem_h_xg *= 0.60
        rem_a_xg *= 1.30
    elif expulsiones == f"Roja {away_team}":
        rem_a_xg *= 0.60
        rem_h_xg *= 1.30

    m_live = build_dixon_coles_matrix(rem_h_xg, rem_a_xg, rho=rho_param)
    
    max_g = m_live.shape[0]
    p1_live = np.sum([m_live[i, j] for i in range(max_g) for j in range(max_g) if (goles_h + i) > (goles_a + j)]) * 100
    px_live = np.sum([m_live[i, j] for i in range(max_g) for j in range(max_g) if (goles_h + i) == (goles_a + j)]) * 100
    p2_live = np.sum([m_live[i, j] for i in range(max_g) for j in range(max_g) if (goles_h + i) < (goles_a + j)]) * 100

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Victoria {home_team}", f"{p1_live:.1f}%", f"Cuota Justa: {100/max(p1_live, 0.01):.2f}")
    m2.metric("Empate Final", f"{px_live:.1f}%", f"Cuota Justa: {100/max(px_live, 0.01):.2f}")
    m3.metric(f"Victoria {away_team}", f"{p2_live:.1f}%", f"Cuota Justa: {100/max(p2_live, 0.01):.2f}")

# ------------------------------------------
# TAB 3: SIMULACIÓN MONTE CARLO
# ------------------------------------------
with tab_montecarlo:
    st.subheader("🎲 Motor de Simulación Monte Carlo (10,000 Iteraciones)")
    st.markdown("Simula matemáticamente los minutos restantes basándose en distribuciones estocásticas por disparo.")
    
    if st.button("🚀 Ejecutar Simulación Monte Carlo"):
        with st.spinner("Simulando 10,000 finales alternativos del partido..."):
            mc1, mcx, mc2, mc_o25, mc_btts = run_monte_carlo_simulation(rem_h_xg, rem_a_xg, goles_h, goles_a)
            
            c_mc1, c_mc2, c_mc3 = st.columns(3)
            c_mc1.metric(f"Simulación: Victoria {home_team}", f"{mc1:.1f}%")
            c_mc2.metric("Simulación: Empate", f"{mcx:.1f}%")
            c_mc3.metric(f"Simulación: Victoria {away_team}", f"{mc2:.1f}%")
            
            st.markdown("---")
            st.write(f"**Probabilidad Simulatada de Over 2.5 Total:** {mc_o25:.1f}%")
            st.write(f"**Probabilidad Simulatada de Ambos Anotan (BTTS):** {mc_btts:.1f}%")

# ------------------------------------------
# TAB 4: MATRIZ COMPLETA
# ------------------------------------------
with tab_matrix:
    st.subheader("🎯 Marcadores Más Probables")
    
    scores = []
    for i in range(6):
        for j in range(6):
            p = matrix_pre[i, j] * 100
            scores.append({"Marcador": f"{i} - {j}", "Probabilidad (%)": f"{p:.2f}%", "Cuota Justa": f"{100/max(p, 0.01):.2f}"})
            
    st.dataframe(pd.DataFrame(scores).sort_values(by="Probabilidad (%)", ascending=False).head(10), hide_index=True, use_container_width=True)
