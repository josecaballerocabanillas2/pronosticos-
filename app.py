import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
import dateutil.parser

st.set_page_config(page_title="Escáner Máxima Certeza (Hora Portugal)", page_icon="⚽", layout="wide")

st.title("⚽ Escáner de Pronósticos Máxima Certeza")
st.caption("Con columnas dedicadas para Ambos Marcan, Córneres Estimados y Pronósticos Principales (Hora de Portugal).")

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
                
                # Horario Portugal (UTC+1 / WET con cambio de hora estándar)
                portugal_tz = timezone(timedelta(hours=1))
                
                for m in matches_data:
                    home_p = m.get('home_team', 'Local')
                    away_p = m.get('away_team', 'Visitante')
                    
                    # Formatear Horario a Portugal
                    commence_time_raw = m.get('commence_time', '')
                    if commence_time_raw:
                        dt_utc = dateutil.parser.isoparse(commence_time_raw)
                        dt_portugal = dt_utc.astimezone(portugal_tz)
                        match_time = dt_portugal.strftime("%H:%M (%d/%m)")
                    else:
                        match_time = "Por definir"

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
                    
                    # Matrices
                    matrix = dixon_coles_prob(h_xg_est, a_xg_est)
                    matrix_ht = dixon_coles_prob(h_xg_est * 0.45, a_xg_est * 0.45)
                    matrix_2t = dixon_coles_prob(h_xg_est * 0.55, a_xg_est * 0.55)
                    
                    # 1. Ganador Final (1X2) y Doble Oportunidad
                    p_h = float(np.sum(np.tril(matrix, -1)))
                    p_d = float(np.sum(np.diag(matrix)))
                    p_a = float(np.sum(np.triu(matrix, 1)))
                    p_1x = p_h + p_d
                    p_x2 = p_a + p_d
                    
                    # 2. Ambos Marcan (BTTS)
                    p_btts_yes = float(np.sum(matrix[1:, 1:]))
                    p_btts_no = 1.0 - p_btts_yes
                    
                    # 3. Totales de Goles
                    total_goals_matrix = np.fromfunction(lambda i, j: i + j, matrix.shape)
                    p_over_0_5 = float(np.sum(matrix[total_goals_matrix > 0.5]))
                    p_over_1_5 = float(np.sum(matrix[total_goals_matrix > 1.5]))
                    p_under_3_5 = float(np.sum(matrix[total_goals_matrix < 3.5]))
                    p_under_4_5 = float(np.sum(matrix[total_goals_matrix < 4.5]))
                    
                    # 4. Estimación de Córneres
                    exp_corners = round((h_xg_est * 3.8) + (a_xg_est * 3.5) + 3.2, 1)
                    
                    # 5. Primer y Segundo Tiempo
                    p_h_ht = float(np.sum(np.tril(matrix_ht, -1)))
                    p_d_ht = float(np.sum(np.diag(matrix_ht)))
                    p_a_ht = float(np.sum(np.triu(matrix_ht, 1)))
                    
                    p_h_2t = float(np.sum(np.tril(matrix_2t, -1)))
                    p_d_2t = float(np.sum(np.diag(matrix_2t)))
                    p_a_2t = float(np.sum(np.triu(matrix_2t, 1)))

                    candidates = [
                        ("Gana " + home_p + " (Final)", p_h, o_h),
                        ("Empate (Final)", p_d, o_d),
                        ("Gana " + away_p + " (Final)", p_a, o_a),
                        ("1X (Doble Oportunidad Local/Empate)", p_1x, round(1/p_1x, 2) if p_1x > 0 else 1.0),
                        ("X2 (Doble Oportunidad Visita/Empate)", p_x2, round(1/p_x2, 2) if p_x2 > 0 else 1.0),
                        ("Más de 0.5 Goles Totales", p_over_0_5, round(1/p_over_0_5, 2) if p_over_0_5 > 0 else 1.0),
                        ("Más de 1.5 Goles Totales", p_over_1_5, round(1/p_over_1_5, 2) if p_over_1_5 > 0 else 1.0),
                        ("Menos de 3.5 Goles Totales", p_under_3_5, round(1/p_under_3_5, 2) if p_under_3_5 > 0 else 1.0),
                        ("Menos de 4.5 Goles Totales", p_under_4_5, round(1/p_under_4_5, 2) if p_under_4_5 > 0 else 1.0),
                        ("Gana " + home_p + " (1ª Parte)", p_h_ht, round(1/p_h_ht, 2) if p_h_ht > 0 else 1.0),
                        ("Empate (1ª Parte)", p_d_ht, round(1/p_d_ht, 2) if p_d_ht > 0 else 1.0),
                        ("Gana " + away_p + " (1ª Parte)", p_a_ht, round(1/p_a_ht, 2) if p_a_ht > 0 else 1.0),
                        ("Gana " + home_p + " (2ª Parte)", p_h_2t, round(1/p_h_2t, 2) if p_h_2t > 0 else 1.0),
                        ("Empate (2ª Parte)", p_d_2t, round(1/p_d_2t, 2) if p_d_2t > 0 else 1.0),
                        ("Gana " + away_p + " (2ª Parte)", p_a_2t, round(1/p_a_2t, 2) if p_a_2t > 0 else 1.0),
                    ]
                    
                    # Ordenar selecciones por probabilidad
                    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    top_pick = sorted_candidates[0]
                    second_pick = sorted_candidates[1]

                    # Determinar favorito en Ambos Marcan
                    if p_btts_yes >= p_btts_no:
                        btts_text = f"<b>SÍ</b> ({p_btts_yes*100:.1f}%)"
                    else:
                        btts_text = f"<b>NO</b> ({p_btts_no*100:.1f}%)"
                    
                    results.append({
                        "Horario (PT)": f"🇵🇹 <b>{match_time}</b>",
                        "Partido": f"<b>{home_p} vs {away_p}</b>",
                        "Ambos Marcan": btts_text,
                        "Córneres Est.": f"🚩 <b>{exp_corners}</b>",
                        "Pronóstico #1 Máxima Certeza": f"🔥 <b>{top_pick[0]}</b><br>• Probabilidad: <b>{top_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{top_pick[2]:.2f}</b>",
                        "Pronóstico #2 Alternativa": f"🟢 <b>{second_pick[0]}</b><br>• Probabilidad: <b>{second_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{second_pick[2]:.2f}</b>",
                        "max_prob": top_pick[1]
                    })
                
                results_sorted = sorted(results, key=lambda x: x["max_prob"], reverse=True)
                for item in results_sorted:
                    del item["max_prob"]
                
                df = pd.DataFrame(results_sorted)
                st.markdown("### 📊 Partidos Ordenados por Índice de Certeza Global")
                st.write(df.to_html(escape=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error procesando datos: {e}")
