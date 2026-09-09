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

with tab_scanner:
    st.subheader("⚡ Escáner de Partidos de la Jornada")
    st.markdown("Busca automáticamente partidos donde el modelo calcule **≥ 80% de probabilidad** y la casa pague **≥ 1.50**.")
    
    data_source = st.radio("Fuente de Datos:", ["Modo Simulación / Prueba", "API en Tiempo Real (The-Odds-API)"])
    
    if data_source == "Modo Simulación / Prueba":
        if st.button("🚀 Escanear Jornada de Prueba"):
            mock_matches = [
                {"home": "Real Madrid", "away": "Getafe", "h_xg": 3.1, "a_xg": 0.3, "odds_h": 1.55, "odds_d": 4.50, "odds_a": 7.00},
                {"home": "Bayern Múnich", "away": "Bochum", "h_xg": 3.4, "a_xg": 0.4, "odds_h": 1.52, "odds_d": 5.00, "odds_a": 8.50},
                {"home": "FC Porto", "away": "Manchester City", "h_xg": 1.0, "a_xg": 2.0, "odds_h": 5.25, "odds_d": 3.90, "odds_a": 1.63},
                {"home": "PSG", "away": "Metz", "h_xg": 2.8, "a_xg": 0.3, "odds_h": 1.35, "odds_d": 5.20, "odds_a": 9.00},
            ]
            
            detected = []
            all_list = []
            for m in mock_matches:
                picks, p_h, p_d, p_a = evaluate_match_value(m["home"], m["away"], m["h_xg"], m["a_xg"], m["odds_h"], m["odds_d"], m["odds_a"], rho)
                detected.extend(picks)
                all_list.append({"Partido": f"{m['home']} vs {m['away']}", "Cuota L": m['odds_h'], "Cuota E": m['odds_d'], "Cuota V": m['odds_a']})
                
            if detected:
                st.success(f"¡Se han detectado {len(detected)} oportunidad(es) que cumplen ambos criterios!")
                st.dataframe(pd.DataFrame(detected), use_container_width=True)
            else:
                st.warning("No se encontraron selecciones que cumplan ambos criterios.")
                
            with st.expander("📋 Ver todos los partidos escaneados"):
                st.dataframe(pd.DataFrame(all_list), use_container_width=True)

    else:
        api_key = st.text_input("Ingresa tu API Key de The-Odds-API", type="password")
        if st.button("📡 Conectar y Escanear en Vivo"):
            if not api_key:
                st.error("Introduce una API Key válida.")
            else:
                try:
                    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={api_key}&regions=eu&markets=h2h"
                    res = requests.get(url)
                    
                    if res.status_code == 200:
                        matches_data = res.json()
                        st.info(f"Se han consultado {len(matches_data)} partidos en vivo.")
                        
                        all_matches = []
                        detected_picks = []
                        
                        for m in matches_data:
                            home_p = m.get('home_team', 'Local')
                            away_p = m.get('away_team', 'Visitante')
                            
                            b_makers = m.get('bookmakers', [])
                            if not b_makers:
                                continue
                                
                            outcomes = b_makers[0]['markets'][0]['outcomes']
                            
                            o_h = next((x['price'] for x in outcomes if x['name'] == home_p), 1.0)
                            o_a = next((x['price'] for x in outcomes if x['name'] == away_p), 1.0)
                            o_d = next((x['price'] for x in outcomes if x['name'] == 'Draw'), 1.0)
                            
                            # Estimación base de xG según cuotas de mercado
                            h_xg_est = round(max(0.5, 2.5 / (o_h if o_h > 0 else 1)), 2)
                            a_xg_est = round(max(0.5, 2.5 / (o_a if o_a > 0 else 1)), 2)
                            
                            picks, _, _, _ = evaluate_match_value(home_p, away_p, h_xg_est, a_xg_est, o_h, o_d, o_a, rho)
                            
                            detected_picks.extend(picks)
                            all_matches.append({
                                "Encuentro": f"{home_p} vs {away_p}",
                                "Cuota Local": o_h,
                                "Cuota Empate": o_d,
                                "Cuota Visitante": o_a
                            })
                        
                        if detected_picks:
                            st.success(f"🔥 ¡{len(detected_picks)} Partido(s) con Oportunidad Detectada (+EV)! 🔥")
                            st.dataframe(pd.DataFrame(detected_picks), use_container_width=True)
                        else:
                            st.warning("De los partidos consultados hoy, ninguno alcanza ≥80% de probabilidad con cuota ≥1.50 al mismo tiempo.")
                            
                        # Tabla con los 20 partidos consultados
                        st.subheader("📋 Lista de Todos los Partidos Consultados en Vivo")
                        st.dataframe(pd.DataFrame(all_matches), use_container_width=True)
                        
                    else:
                        st.error("Error en la conexión con la API. Verifica tu clave o límite mensual.")
                except Exception as e:
                    st.error(f"Error procesando los datos: {e}")
