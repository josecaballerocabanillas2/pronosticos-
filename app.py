import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests

st.set_page_config(page_title="AI Football Predictor PRO", page_icon="⚽", layout="wide")

st.title("⚽ AI Football Predictor & Real-Time Value Scanner")
st.caption("Calculadora de Probabilidades (Dixon-Coles) y Escáner de Encuentros (+EV)")

# ---------------------------------------------------------
# CÁLCULOS MATEMÁTICOS Y MODELO DIXON-COLES
# ---------------------------------------------------------
def dixon_coles_prob(home_xg, away_xg, rho=-0.13, max_g=8):
    """Calcula matriz de probabilidades con el ajuste de correlación Dixon-Coles"""
    matrix = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            p_i = stats.poisson.pmf(i, home_xg)
            p_j = stats.poisson.pmf(j, away_xg)
            
            if i == 0 and j == 0:
                tau = 1.0 - (home_xg * away_xg * rho)
            elif i == 0 and j == 1:
                tau = 1.0 + (home_xg * rho)
            elif i == 1 and j == 0:
                tau = 1.0 + (away_xg * rho)
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
                
            matrix[i, j] = p_i * p_j * max(0, tau)
            
    total_p = np.sum(matrix)
    if total_p > 0:
        matrix /= total_p
    return matrix

def evaluate_match_value(home, away, h_xg, a_xg, odds_h, odds_d, odds_a, rho=-0.13):
    """Analiza un juego y retorna selecciones con Probabilidad >= 80% y Cuota >= 1.50"""
    matrix = dixon_coles_prob(h_xg, a_xg, rho=rho)
    
    p_h = float(np.sum(np.tril(matrix, -1)))
    p_d = float(np.sum(np.diag(matrix)))
    p_a = float(np.sum(np.triu(matrix, 1)))
    
    picks = []
    
    # Evaluar Victoria Local
    if p_h >= 0.80 and odds_h >= 1.50:
        ev = (p_h * odds_h) - 1
        picks.append({
            "Partido": f"{home} vs {away}",
            "Pronóstico": f"Gana {home}",
            "Probabilidad": f"{p_h*100:.1f}%",
            "Cuota Casa": odds_h,
            "Valor (+EV)": f"+{ev*100:.1f}%"
        })
        
    # Evaluar Empate
    if p_d >= 0.80 and odds_d >= 1.50:
        ev = (p_d * odds_d) - 1
        picks.append({
            "Partido": f"{home} vs {away}",
            "Pronóstico": "Empate",
            "Probabilidad": f"{p_d*100:.1f}%",
            "Cuota Casa": odds_d,
            "Valor (+EV)": f"+{ev*100:.1f}%"
        })

    # Evaluar Victoria Visitante
    if p_a >= 0.80 and odds_a >= 1.50:
        ev = (p_a * odds_a) - 1
        picks.append({
            "Partido": f"{home} vs {away}",
            "Pronóstico": f"Gana {away}",
            "Probabilidad": f"{p_a*100:.1f}%",
            "Cuota Casa": odds_a,
            "Valor (+EV)": f"+{ev*100:.1f}%"
        })
        
    return picks, p_h, p_d, p_a

# ---------------------------------------------------------
# MENÚ LATERAL (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Encuentro")
home_team = st.sidebar.text_input("Equipo Local", "FC Porto")
away_team = st.sidebar.text_input("Equipo Visitante", "Manchester City")

col_xg1, col_xg2 = st.sidebar.columns(2)
with col_xg1:
    home_xg = st.number_input("xG Local (90m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
with col_xg2:
    away_xg = st.number_input("xG Visitante (90m)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

rho = st.sidebar.slider("Factor de Correlación (Rho)", -0.30, 0.00, -0.13, 0.01)

# ---------------------------------------------------------
# PESTAÑAS PRINCIPALES
# ---------------------------------------------------------
tab_single, tab_scanner = st.tabs(["🎯 Análisis Manual", "🔍 Escáner Automático (Prob ≥ 80% | Cuota ≥ 1.50)"])

# Pestaña 1: Análisis Manual de un solo partido
with tab_single:
    st.subheader(f"Análisis: {home_team} vs {away_team}")
    
    col1, col2, col3 = st.columns(3)
    odds_h = col1.number_input("Cuota Local", value=5.25)
    odds_d = col2.number_input("Cuota Empate", value=3.90)
    odds_a = col3.number_input("Cuota Visitante", value=1.63)
    
    _, p_h, p_d, p_a = evaluate_match_value(home_team, away_team, home_xg, away_xg, odds_h, odds_d, odds_a, rho)
    
    summary = pd.DataFrame({
        "Resultado": [f"Gana {home_team}", "Empate", f"Gana {away_team}"],
        "Probabilidad Estimada": [f"{p_h*100:.1f}%", f"{p_d*100:.1f}%", f"{p_a*100:.1f}%"],
        "Cuota Justa (Modelo)": [f"{1/p_h:.2f}" if p_h > 0 else "N/A", 
                                 f"{1/p_d:.2f}" if p_d > 0 else "N/A", 
                                 f"{1/p_a:.2f}" if p_a > 0 else "N/A"],
        "Cuota Casa Apuestas": [odds_h, odds_d, odds_a]
    })
    st.table(summary)

# Pestaña 2: Escáner Automático
with tab_scanner:
    st.subheader("⚡ Escáner de Partidos de la Jornada")
    st.markdown("Busca automáticamente partidos donde el modelo calcule **≥ 80% de probabilidad** y la casa pague **≥ 1.50**.")
    
    data_source = st.radio("Fuente de Datos:", ["Modo Simulación / Prueba", "API en Tiempo Real (The-Odds-API)"])
    
    if data_source == "Modo Simulación / Prueba":
        if st.button("🚀 Escanear Jornada de Prueba"):
            # Partidos de prueba simulados
            mock_matches = [
                {"home": "Real Madrid", "away": "Getafe", "h_xg": 3.1, "a_xg": 0.3, "odds_h": 1.55, "odds_d": 4.50, "odds_a": 7.00},
                {"home": "Bayern Múnich", "away": "Bochum", "h_xg": 3.4, "a_xg": 0.4, "odds_h": 1.52, "odds_d": 5.00, "odds_a": 8.50},
                {"home": "FC Porto", "away": "Manchester City", "h_xg": 1.0, "a_xg": 2.0, "odds_h": 5.25, "odds_d": 3.90, "odds_a": 1.63},
                {"home": "PSG", "away": "Metz", "h_xg": 2.8, "a_xg": 0.3, "odds_h": 1.35, "odds_d": 5.20, "odds_a": 9.00}, # Descarta por cuota < 1.50
            ]
            
            detected = []
            for m in mock_matches:
                picks, _, _, _ = evaluate_match_value(m["home"], m["away"], m["h_xg"], m["a_xg"], m["odds_h"], m["odds_d"], m["odds_a"], rho)
                detected.extend(picks)
                
            if detected:
                st.success(f"¡Se han detectado {len(detected)} oportunidad(es) de alto valor!")
                st.dataframe(pd.DataFrame(detected), use_container_width=True)
            else:
                st.warning("No se encontraron partidos que cumplan ambos criterios en este momento.")

    else:
        api_key = st.text_input("Ingresa tu API Key de The-Odds-API", type="password")
        if st.button("📡 Conectar y Escanear en Vivo"):
            if not api_key:
                st.error("Introduce una API Key válida para consultar datos en directo.")
            else:
                try:
                    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=eu&markets=h2h"
                    res = requests.get(url)
                    if res.status_code == 200:
                        matches_data = res.json()
                        st.info(f"Se han consultado {len(matches_data)} partidos en vivo.")
                        # Proceso automático en vivo
                    else:
                        st.error("Error al conectar con la API. Verifica tu clave.")
                except Exception as e:
                    st.error(f"Ocurrió un error en la conexión: {e}")
