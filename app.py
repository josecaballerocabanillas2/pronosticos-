import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
from datetime import datetime
import dateutil.parser

st.set_page_config(page_title="Escáner Completo de Pronósticos con Horarios", page_icon="⚽", layout="wide")

st.title("⚽ Escáner de Pronósticos Extendido con Horarios")
st.caption("Incluye horarios de inicio, Ambos Marcan, Resultado Final (1X2), Primer Tiempo (1T) y Segundo Tiempo (2T).")

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
                    
                    # Formatear Horario
                    commence_time_raw = m.get('commence_time', '')
                    if commence_time_raw:
                        dt = dateutil.parser.isoparse(commence_time_raw).astimezone()
                        match_time = dt.strftime("%H:%M (%d/%m)")
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
                    
                    # Matriz Partido Completo
                    matrix = dixon_coles_prob(h_xg_est, a_xg_est)
                    
                    # Matriz 1ª Parte (~45% del xG total)
                    matrix_ht = dixon_coles_prob(h_xg_est * 0.45, a_xg_est * 0.45)
                    
                    # Matriz 2ª Parte (~55% del xG total)
                    matrix_2t = dixon_coles_prob(h_xg_est * 0.55, a_xg_est * 0.55)
                    
                    # 1. Ganador Final (1X2)
                    p_h = float(np.sum(np.tril(matrix, -1)))
                    p_d = float(np.sum(np.diag(matrix)))
                    p_a = float(np.sum(np.triu(matrix, 1)))
                    
                    # 2. Ambos Marcan (BTTS)
                    p_btts_yes = float(np.sum(matrix[1:, 1:]))
                    p_btts_no = 1.0 - p_btts_yes
                    
                    # 3. Primer Tiempo (1T)
                    p_h_ht = float(np.sum(np.tril(matrix_ht, -1)))
                    p_d_ht = float(np.sum(np.diag(matrix_ht)))
                    p_a_ht = float(np.sum(np.triu(matrix_ht, 1)))
                    
                    # 4. Segundo Tiempo (2T)
                    p_h_2t = float(np.sum(np.tril(matrix_2t, -1)))
                    p_d_2t = float(np.sum(np.diag(matrix_2t)))
                    p_a_2t = float(np.sum(np.triu(matrix_2t, 1)))

                    candidates = [
                        ("Gana " + home_p + " (Partido Completo)", p_h, o_h),
                        ("Empate (Partido Completo)", p_d, o_d),
                        ("Gana " + away_p + " (Partido Completo)", p_a, o_a),
                        ("Ambos Marcan: SÍ", p_btts_yes, round(1/p_btts_yes, 2) if p_btts_yes > 0 else 1.0),
                        ("Ambos Marcan: NO", p_btts_no, round(1/p_btts_no, 2) if p_btts_no > 0 else 1.0),
                        ("Gana " + home_p + " (1ª Parte)", p_h_ht, round(1/p_h_ht, 2) if p_h_ht > 0 else 1.0),
                        ("Empate (1ª Parte)", p_d_ht, round(1/p_d_ht, 2) if p_d_ht > 0 else 1.0),
                        ("Gana " + away_p + " (1ª Parte)", p_a_ht, round(1/p_a_ht, 2) if p_a_ht > 0 else 1.0),
                        ("Gana " + home_p + " (2ª Parte)", p_h_2t, round(1/p_h_2t, 2) if p_h_2t > 0 else 1.0),
                        ("Empate (2ª Parte)", p_d_2t, round(1/p_d_2t, 2) if p_d_2t > 0 else 1.0),
                        ("Gana " + away_p + " (2ª Parte)", p_a_2t, round(1/p_a_2t, 2) if p_a_2t > 0 else 1.0),
                    ]
                    
                    # Ordenar por probabilidad descendente
                    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    top_pick = sorted_candidates[0]
                    second_pick = sorted_candidates[1]
                    third_pick = sorted_candidates[2]
                    
                    results.append({
                        "Horario": f"⏰ <b>{match_time}</b>",
                        "Partido": f"<b>{home_p} vs {away_p}</b>",
                        "Pronóstico #1 Máxima Certeza": f"🔥 <b>{top_pick[0]}</b><br>• Probabilidad: <b>{top_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{top_pick[2]:.2f}</b>",
                        "Pronóstico #2 Alternativa": f"🟢 <b>{second_pick[0]}</b><br>• Probabilidad: <b>{second_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{second_pick[2]:.2f}</b>",
                        "Pronóstico #3 Opción": f"🔵 <b>{third_pick[0]}</b><br>• Probabilidad: <b>{third_pick[1]*100:.1f}%</b><br>• Cuota Ref.: <b>{third_pick[2]:.2f}</b>"
                    })
                    
                df = pd.DataFrame(results)
                st.markdown("### 📊 Mejores Opciones por Partido")
                st.write(df.to_html(escape=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error procesando datos: {e}")
