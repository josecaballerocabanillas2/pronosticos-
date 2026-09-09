import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="AI Football Predictor PRO", page_icon="⚽", layout="wide")

st.title("⚽ AI Football Predictor & Multi-Market Scanner")
st.caption("Análisis Estadístico Avanzado: Ganador, Goles, Córneres y Ambos Marcan (Calibrado)")

# ---------------------------------------------------------
# MOTOR DE PROBABILIDADES CALIBRADO
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

def analyze_full_markets(home, away, h_xg, a_xg, odds_h, odds_d, odds_a, commence_time="", rho=-0.13):
    """Genera pronósticos extendidos para todos los mercados de un partido"""
    matrix = dixon_coles_prob(h_xg, a_xg, rho=rho)
    
    # 1. Mercado 1X2
    p_h = float(np.sum(np.tril(matrix, -1)))
    p_d = float(np.sum(np.diag(matrix)))
    p_a = float(np.sum(np.triu(matrix, 1)))
    
    # 2. Mercado 1X2 al Descanso (45% del xG)
    matrix_ht = dixon_coles_prob(h_xg * 0.45, a_xg * 0.45, rho=rho)
    
    # 3. Ambos Marcan (BTTS)
    p_btts_yes = float(np.sum(matrix[1:, 1:]))
    p_btts_no = 1.0 - p_btts_yes
    
    # 4. Total de Goles (Over/Under)
    total_goals_matrix = np.fromfunction(lambda i, j: i + j, matrix.shape)
    p_over_0_5_ht = float(np.sum(matrix_ht[total_goals_matrix > 0.5]))
    p_over_1_5 = float(np.sum(matrix[total_goals_matrix > 1.5]))
    p_over_2_5 = float(np.sum(matrix[total_goals_matrix > 2.5]))
    p_under_2_5 = 1.0 - p_over_2_5
    
    # 5. Estimación de Córneres
    exp_corners = round((h_xg * 3.8) + (a_xg * 3.5) + 3.2, 1)
    p_corners_8_5 = min(0.95, round(0.50 + (exp_corners - 8.5) * 0.08, 2))
    
    # 6. Marcador Más Probable
    flat_idx = np.argmax(matrix)
    most_likely_score = np.unravel_index(flat_idx, matrix.shape)
    score_str = f"{most_likely_score[0]} - {most_likely_score[1]}"
    
    # Formatear hora
    formatted_time = "Por definir"
    if commence_time:
        try:
            dt = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
            formatted_time = dt.strftime("%d/%m/%Y - %H:%M UTC")
        except Exception:
            formatted_time = commence_time
            
    # Compilar recomendaciones
    predictions = []
    
    # 1X2 / Doble Oportunidad
    if p_h >= 0.58:
        predictions.append(f"<b>Resultado:</b> Gana {home} ({p_h*100:.1f}%)")
    elif p_a >= 0.58:
        predictions.append(f"<b>Resultado:</b> Gana {away} ({p_a*100:.1f}%)")
    elif p_h >= 0.40 and p_h > p_a:
        predictions.append(f"<b>Doble Oportunidad:</b> {home} o Empate (1X)")
    elif p_a >= 0.40 and p_a > p_h:
        predictions.append(f"<b>Doble Oportunidad:</b> {away} o Empate (X2)")
    else:
        predictions.append("<b>Resultado:</b> Partido Igualado")
        
    # Total de Goles
    if p_over_2_5 >= 0.50:
        predictions.append(f"<b>Total Goles:</b> Más de 2.5 Goles ({p_over_2_5*100:.1f}%)")
    elif p_over_1_5 >= 0.70:
        predictions.append(f"<b>Total Goles:</b> Más de 1.5 Goles ({p_over_1_5*100:.1f}%)")
    else:
        predictions.append(f"<b>Total Goles:</b> Menos de 2.5 Goles ({p_under_2_5*100:.1f}%)")
        
    # 1ª Parte
    if p_over_0_5_ht >= 0.65:
        predictions.append(f"<b>1ª Parte:</b> Más de 0.5 Goles HT ({p_over_0_5_ht*100:.1f}%)")
        
    # Ambos Marcan (BTTS) — Calibrado a umbral realista (50%+)
    if p_btts_yes >= 0.50 or (h_xg >= 1.15 and a_xg >= 1.15):
        predictions.append(f"<b>Ambos Marcan:</b> SÍ ({p_btts_yes*100:.1f}%)")
    else:
        predictions.append(f"<b>Ambos Marcan:</b> NO ({p_btts_no*100:.1f}%)")
        
    # Córneres y Marcador
    predictions.append(f"<b>Córneres Est.:</b> ~{exp_corners} (+8.5: {p_corners_8_5*100:.0f}%)")
    predictions.append(f"<b>Marcador Probable:</b> {score_str}")
    
    return {
        "Hora de Inicio": formatted_time,
        "Encuentro": f"<b>{home} vs {away}</b>",
        "Pronóstico Recomendado": "<br>".join(predictions),
        "Prob. Victoria Local": f"{p_h*100:.1f}%",
        "Prob. Over 2.5": f"{p_over_2_5*100:.1f}%",
        "Ambos Marcan": f"{p_btts_yes*100:.1f}%",
        "Córneres Est.": f"~{exp_corners}"
    }

# ---------------------------------------------------------
# MENÚ LATERAL
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Encuentro")
home_team = st.sidebar.text_input("Equipo Local", "FC Porto")
away_team = st.sidebar.text_input("Equipo Visitante", "Manchester City")

col_xg1, col_xg2 = st.sidebar.columns(2)
with col_xg1:
    home_xg = st.number_input("xG Local (90m)", min_value=0.1, max_value=5.0, value=1.4, step=0.1)
with col_xg2:
    away_xg = st.number_input("xG Visitante (90m)", min_value=0.1, max_value=5.0, value=1.3, step=0.1)

rho = st.sidebar.slider("Factor de Correlación (Rho)", -0.30, 0.00, -0.13, 0.01)

# ---------------------------------------------------------
# PESTAÑAS PRINCIPALES
# ---------------------------------------------------------
tab_single, tab_scanner = st.tabs(["🎯 Análisis Manual", "🔍 Escáner Completo con Horarios"])

with tab_single:
    st.subheader(f"Análisis Exhaustivo: {home_team} vs {away_team}")
    
    col1, col2, col3 = st.columns(3)
    odds_h = col1.number_input("Cuota Local", value=2.20)
    odds_d = col2.number_input("Cuota Empate", value=3.30)
    odds_a = col3.number_input("Cuota Visitante", value=3.10)
    
    res = analyze_full_markets(home_team, away_team, home_xg, away_xg, odds_h, odds_d, odds_a, rho=rho)
    
    st.markdown("### 📋 Sugerencias Multimercado Detalladas")
    st.write(res["Pronóstico Recomendado"], unsafe_allow_html=True)

with tab_scanner:
    st.subheader("⚡ Escáner Multimercado de la Jornada")
    st.markdown("Consigue el horario oficial del encuentro y un desglose completo de mercados de apuestas para cada partido.")
    
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
                        commence = m.get('commence_time', '')
                        
                        b_makers = m.get('bookmakers', [])
                        if not b_makers:
                            continue
                            
                        outcomes = b_makers[0]['markets'][0]['outcomes']
                        
                        o_h = next((x['price'] for x in outcomes if x['name'] == home_p), 1.0)
                        o_a = next((x['price'] for x in outcomes if x['name'] == away_p), 1.0)
                        o_d = next((x['price'] for x in outcomes if x['name'] == 'Draw'), 1.0)
                        
                        # Inferencia ajustada de xG según cuotas
                        # Se calcula un xG base proporcional a la probabilidad implícita
                        h_implied = 1.0 / o_h if o_h > 0 else 0.33
                        a_implied = 1.0 / o_a if o_a > 0 else 0.33
                        
                        h_xg_est = round(max(0.8, h_implied * 2.7), 2)
                        a_xg_est = round(max(0.8, a_implied * 2.7), 2)
                        
                        # Generar predicción extendida
                        match_info = analyze_full_markets(home_p, away_p, h_xg_est, a_xg_est, o_h, o_d, o_a, commence, rho)
                        full_analysis_list.append(match_info)
                    
                    df = pd.DataFrame(full_analysis_list)
                    
                    # Presentación limpia en pantalla
                    st.markdown("### 📊 Pronósticos y Horarios por Partido")
                    st.write(df.to_html(escape=False), unsafe_allow_html=True)
                    
                else:
                    st.error("Error al conectar con el servidor de la API.")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar los datos: {e}")
