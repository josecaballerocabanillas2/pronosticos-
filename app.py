import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests

st.set_page_config(page_title="AI Football Predictor PRO", page_icon="⚽", layout="wide")

st.title("⚽ AI Football Predictor & Multi-Market Scanner")
st.caption("Análisis Estadístico Avanzado: Ganador, Goles, Córneres y Ambos Marcan")

# ---------------------------------------------------------
# MOTOR DE PROBABILIDADES MULTI-MERCADO
# ---------------------------------------------------------
def dixon_coles_prob(home_xg, away_xg, rho=-0.13, max_g=8):
    """Calcula la matriz estocástica de marcadores posibles"""
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

def analyze_full_markets(home, away, h_xg, a_xg, odds_h, odds_d, odds_a, rho=-0.13):
    """Genera pronósticos extendidos para todos los mercados de un partido"""
    matrix = dixon_coles_prob(h_xg, a_xg, rho=rho)
    
    # 1. Mercado 1X2
    p_h = float(np.sum(np.tril(matrix, -1)))
    p_d = float(np.sum(np.diag(matrix)))
    p_a = float(np.sum(np.triu(matrix, 1)))
    
    # 2. Ambos Marcan (BTTS)
    p_btts_yes = float(np.sum(matrix[1:, 1:]))
    p_btts_no = 1.0 - p_btts_yes
    
    # 3. Total de Goles (Over/Under)
    total_goals_matrix = np.fromfunction(lambda i, j: i + j, matrix.shape)
    p_over_1_5 = float(np.sum(matrix[total_goals_matrix > 1.5]))
    p_over_2_5 = float(np.sum(matrix[total_goals_matrix > 2.5]))
    p_under_2_5 = 1.0 - p_over_2_5
    
    # 4. Estimación de Córneres (Basado en intensidad ofensiva xG)
    exp_corners = round((h_xg * 3.8) + (a_xg * 3.5) + 3.2, 1)
    p_corners_8_5 = min(0.95, round(0.50 + (exp_corners - 8.5) * 0.08, 2))
    
    # Compilar recomendaciones principales
    predictions = []
    
    # Selección 1X2
    if p_h >= 0.55:
        predictions.append(f"<b>Resultado:</b> Gana {home} ({p_h*100:.1f}%)")
    elif p_a >= 0.55:
        predictions.append(f"<b>Resultado:</b> Gana {away} ({p_a*100:.1f}%)")
    else:
        predictions.append("<b>Resultado:</b> Partido Abierto / Doble Oportunidad")
        
    # Selección Goles
    if p_over_2_5 >= 0.55:
        predictions.append(f"<b>Goles:</b> Más de 2.5 Goles ({p_over_2_5*100:.1f}%)")
    elif p_over_1_5 >= 0.75:
        predictions.append(f"<b>Goles:</b> Más de 1.5 Goles ({p_over_1_5*100:.1f}%)")
    else:
        predictions.append(f"<b>Goles:</b> Menos de 2.5 Goles ({p_under_2_5*100:.1f}%)")
        
    # Selección Ambos Marcan
    if p_btts_yes >= 0.55:
        predictions.append(f"<b>Ambos Marcan:</b> SÍ ({p_btts_yes*100:.1f}%)")
    else:
        predictions.append(f"<b>Ambos Marcan:</b> NO ({p_btts_no*100:.1f}%)")
        
    # Selección Córneres
    predictions.append(f"<b>Córneres Promedio:</b> ~{exp_corners} (Más de 8.5 Cantos: {p_corners_8_5*100:.0f}%)")
    
    return {
        "Encuentro": f"{home} vs {away}",
        "Pronóstico Recomendado": "<br>".join(predictions),
        "Prob. Victoria Local": f"{p_h*100:.1f}%",
        "Prob. Over 2.5 Goles": f"{p_over_2_5*100:.1f}%",
        "Prob. Both Teams Score": f"{p_btts_yes*100:.1f}%",
        "Expectativa Córneres": f"~{exp_corners}"
    }

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
tab_single, tab_scanner = st.tabs(["🎯 Análisis Manual", "🔍 Escáner Completo de Encuentros"])

with tab_single:
    st.subheader(f"Análisis Exhaustivo: {home_team} vs {away_team}")
    
    col1, col2, col3 = st.columns(3)
    odds_h = col1.number_input("Cuota Local", value=5.25)
    odds_d = col2.number_input("Cuota Empate", value=3.90)
    odds_a = col3.number_input("Cuota Visitante", value=1.63)
    
    res = analyze_full_markets(home_team, away_team, home_xg, away_xg, odds_h, odds_d, odds_a, rho)
    
    st.markdown("### 📋 Sugerencias de Apuesta Multimercado")
    st.write(res["Pronóstico Recomendado"], unsafe_allow_html=True)

with tab_scanner:
    st.subheader("⚡ Escáner Multimercado de la Jornada")
    st.markdown("Genera pronósticos concretos de **Ganador, Total de Goles, Ambos Marcan y Córneres** para cada partido de la lista.")
    
    api_key = st.text_input("Ingresa tu API Key de The-Odds-API", type="password")
    
    if st.button("📡 Analizar Partidos de Hoy"):
        if not api_key:
            st.error("Introduce tu API Key para cargar los partidos en vivo.")
        else:
            try:
                url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={api_key}&regions=eu&markets=h2h"
                res = requests.get(url)
                
                if res.status_code == 200:
                    matches_data = res.json()
                    st.info(f"Se están procesando {len(matches_data)} partidos...")
                    
                    full_analysis_list = []
                    
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
                        
                        # Inferencia estocástica de xG a partir de las cuotas del mercado
                        h_xg_est = round(max(0.6, 2.4 / (o_h if o_h > 0 else 1)), 2)
                        a_xg_est = round(max(0.6, 2.4 / (o_a if o_a > 0 else 1)), 2)
                        
                        # Generar predicción extendida
                        match_info = analyze_full_markets(home_p, away_p, h_xg_est, a_xg_est, o_h, o_d, o_a, rho)
                        full_analysis_list.append(match_info)
                    
                    df = pd.DataFrame(full_analysis_list)
                    
                    # Presentación limpia en pantalla
                    st.markdown("### 📊 Pronósticos Detallados por Partido")
                    st.write(df.to_html(escape=False), unsafe_allow_html=True)
                    
                else:
                    st.error("Error al conectar con el servidor de la API.")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar los datos: {e}")
