import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests

st.set_page_config(page_title="Top Pronósticos de Alta Certeza", page_icon="⚽", layout="wide")

st.title("⚽ Escáner de Pronósticos Máxima Certeza")
st.caption("Filtra y muestra la jugada con mayor probabilidad matemática de ocurrir por partido.")

# ---------------------------------------------------------
# MOTOR DE PROBABILIDADES
# ---------------------------------------------------------
def dixon_coles_prob(home_xg, away_xg, rho=-0.13, max_g=8):
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

# ---------------------------------------------------------
# INTERFAZ STREAMLIT
# ---------------------------------------------------------
api_key_odds = st.text_input("Ingresa tu API Key de The-Odds-API", type="password")

if st.button("🚀 Escanear y Mostrar Mejores Pronósticos"):
    if not api_key_odds:
        st.error("Ingresa la API Key para consultar.")
    else:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={api_key_odds}&regions=eu&markets=h2h"
            res = requests.get(url)
            
            if res.status_code == 200:
                matches_data = res.json()
                results = []
                
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
                    
                    h_implied = 1.0 / o_h if o_h > 0 else 0.33
                    a_implied = 1.0 / o_a if o_a > 0 else 0.33
                    
                    h_xg_est = round(max(0.8, h_implied * 2.7), 2)
                    a_xg_est = round(max(0.8, a_implied * 2.7), 2)
                    
                    matrix = dixon_coles_prob(h_xg_est, a_xg_est)
                    
                    p_h = float(np.sum(np.tril(matrix, -1)))
                    p_d = float(np.sum(np.diag(matrix)))
                    p_a = float(np.sum(np.triu(matrix, 1)))
                    
                    p_1x = p_h + p_d
                    p_x2 = p_a + p_d
                    
                    total_goals_matrix = np.fromfunction(lambda i, j: i + j, matrix.shape)
                    p_over_0_5 = float(np.sum(matrix[total_goals_matrix > 0.5]))
                    p_over_1_5 = float(np.sum(matrix[total_goals_matrix > 1.5]))
                    p_under_3_5 = float(np.sum(matrix[total_goals_matrix < 3.5]))
                    p_under_4_5 = float(np.sum(matrix[total_goals_matrix < 4.5]))
                    
                    exp_corners = round((h_xg_est * 3.8) + (a_xg_est * 3.5) + 3.2, 1)
                    p_corners_7_5 = min(0.98, round(0.50 + (exp_corners - 7.5) * 0.08, 2))

                    # Evaluación de opciones
                    candidates = [
                        ("Gana " + home_p, p_h, o_h),
                        ("Gana " + away_p, p_a, o_a),
                        ("Doble Oportunidad 1X (" + home_p + " o Empate)", p_1x, round(1/p_1x, 2) if p_1x > 0 else 1.0),
                        ("Doble Oportunidad X2 (" + away_p + " o Empate)", p_x2, round(1/p_x2, 2) if p_x2 > 0 else 1.0),
                        ("Más de 0.5 Goles Totales", p_over_0_5, round(1/p_over_0_5, 2) if p_over_0_5 > 0 else 1.0),
                        ("Más de 1.5 Goles Totales", p_over_1_5, round(1/p_over_1_5, 2) if p_over_1_5 > 0 else 1.0),
                        ("Menos de 3.5 Goles Totales", p_under_3_5, round(1/p_under_3_5, 2) if p_under_3_5 > 0 else 1.0),
                        ("Menos de 4.5 Goles Totales", p_under_4_5, round(1/p_under_4_5, 2) if p_under_4_5 > 0 else 1.0),
                        ("Más de 7.5 Córneres", p_corners_7_5, 1.22)
                    ]
                    
                    # Ordenar por probabilidad descendente
                    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    top_pick = sorted_candidates[0]
                    second_pick = sorted_candidates[1]
                    
                    results.append({
                        "Partido": f"<b>{home_p} vs {away_p}</b>",
                        "Pronóstico #1 Máxima Certeza": f"🔥 <b>{top_pick[0]}</b><br>• Probabilidad: <b>{top_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{top_pick[2]:.2f}</b>",
                        "Pronóstico #2 Alternativa": f"🟢 <b>{second_pick[0]}</b><br>• Probabilidad: <b>{second_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{second_pick[2]:.2f}</b>"
                    })
                    
                df = pd.DataFrame(results)
                st.markdown("### 📊 Mejores Opciones por Partido")
                st.write(df.to_html(escape=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error procesando datos: {e}")
