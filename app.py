import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
import dateutil.parser

st.set_page_config(page_title="Escáner Máxima Certeza & Live", page_icon="⚽", layout="wide")

st.title("⚽ Escáner de Pronósticos (Pre-Partido & En Vivo)")
st.caption("Ordenado por horario de Portugal. Incluye ingreso de marcadores en vivo para recalcular probabilidades.")

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

def calculate_predictions(home_p, away_p, home_xg, away_xg, o_h, o_a, o_d, home_goals=0, away_goals=0):
    matrix = dixon_coles_prob(home_xg, away_xg)
    
    # Matriz para restarle el tiempo consumido
    p_h = float(np.sum(np.tril(matrix, -1)))
    p_d = float(np.sum(np.diag(matrix)))
    p_a = float(np.sum(np.triu(matrix, 1)))
    p_1x = p_h + p_d
    p_x2 = p_a + p_d
    
    # Evaluar si Ambos Marcan ya ocurrió en vivo o por probabilidad
    if home_goals > 0 and away_goals > 0:
        p_btts_yes = 1.0
        p_btts_no = 0.0
    else:
        p_btts_yes = float(np.sum(matrix[1:, 1:]))
        p_btts_no = 1.0 - p_btts_yes
    
    current_total_goals = home_goals + away_goals
    total_goals_matrix = np.fromfunction(lambda i, j: i + j + current_total_goals, matrix.shape)
    p_over_1_5 = float(np.sum(matrix[total_goals_matrix > 1.5]))
    p_under_3_5 = float(np.sum(matrix[total_goals_matrix < 3.5]))

    candidates = [
        ("Gana " + home_p + " (Final)", p_h, o_h),
        ("Empate (Final)", p_d, o_d),
        ("Gana " + away_p + " (Final)", p_a, o_a),
        ("1X (Doble Oportunidad Local/Empate)", p_1x, round(1/p_1x, 2) if p_1x > 0 else 1.0),
        ("X2 (Doble Oportunidad Visita/Empate)", p_x2, round(1/p_x2, 2) if p_x2 > 0 else 1.0),
        ("Más de 1.5 Goles Totales", p_over_1_5, round(1/p_over_1_5, 2) if p_over_1_5 > 0 else 1.0),
        ("Menos de 3.5 Goles Totales", p_under_3_5, round(1/p_under_3_5, 2) if p_under_3_5 > 0 else 1.0),
    ]
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return sorted_candidates[:3], p_btts_yes, p_btts_no

# ---------------------------------------------------------
# INTERFAZ STREAMLIT
# ---------------------------------------------------------
api_key_odds = st.text_input("Ingresa tu API Key de The-Odds-API", type="password")

if st.button("🚀 Escanear y Cargar Analizador Completo"):
    if not api_key_odds:
        st.error("Ingresa la API Key para consultar.")
    else:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={api_key_odds}&regions=eu&markets=h2h"
            res = requests.get(url)
            
            if res.status_code == 200:
                matches_data = res.json()
                results_pre = []
                
                portugal_tz = timezone(timedelta(hours=1))
                
                for m in matches_data:
                    home_p = m.get('home_team', 'Local')
                    away_p = m.get('away_team', 'Visitante')
                    
                    commence_time_raw = m.get('commence_time', '')
                    if commence_time_raw:
                        dt_utc = dateutil.parser.isoparse(commence_time_raw)
                        dt_portugal = dt_utc.astimezone(portugal_tz)
                        match_time = dt_portugal.strftime("%H:%M (%d/%m)")
                        sort_time = dt_portugal
                    else:
                        match_time = "Por definir"
                        sort_time = datetime.max.replace(tzinfo=portugal_tz)

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
                    
                    exp_corners = round((h_xg_est * 3.8) + (a_xg_est * 3.5) + 3.2, 1)
                    top_picks, p_btts_yes, p_btts_no = calculate_predictions(home_p, away_p, h_xg_est, a_xg_est, o_h, o_a, o_d)
                    
                    btts_detail = f"<b>SÍ:</b> {p_btts_yes*100:.1f}%<br><b>NO:</b> {p_btts_no*100:.1f}%"
                    
                    results_pre.append({
                        "Horario (PT)": f"🇵🇹 <b>{match_time}</b>",
                        "Partido": f"<b>{home_p} vs {away_p}</b>",
                        "Ambos Marcan": btts_detail,
                        "Córneres Est.": f"🚩 <b>{exp_corners}</b>",
                        "Pronóstico #1 Máxima Certeza": f"🔥 <b>{top_picks[0][0]}</b><br>• Probabilidad: <b>{top_picks[0][1]*100:.1f}%</b><br>• Cuota: <b>{top_picks[0][2]:.2f}</b>",
                        "Pronóstico #2 Alternativa": f"🟢 <b>{top_picks[1][0]}</b><br>• Probabilidad: <b>{top_picks[1][1]*100:.1f}%</b><br>• Cuota: <b>{top_picks[1][2]:.2f}</b>",
                        "Pronóstico #3 Opción": f"🔵 <b>{top_picks[2][0]}</b><br>• Probabilidad: <b>{top_picks[2][1]*100:.1f}%</b><br>• Cuota: <b>{top_picks[2][2]:.2f}</b>",
                        "sort_time": sort_time,
                        "home_p": home_p, "away_p": away_p,
                        "pre_picks": top_picks,
                        "h_xg": h_xg_est, "a_xg": a_xg_est,
                        "o_h": o_h, "o_a": o_a, "o_d": o_d
                    })
                
                results_sorted = sorted(results_pre, key=lambda x: x["sort_time"])
                
                tab1, tab2 = st.tabs(["📊 Tabla Pre-Partido", "⏱️ Actualizador LIVE Manual"])
                
                with tab1:
                    clean_results = []
                    for item in results_sorted:
                        row = item.copy()
                        del row["sort_time"]; del row["home_p"]; del row["away_p"]; del row["pre_picks"]
                        del row["h_xg"]; del row["a_xg"]; del row["o_h"]; del row["o_a"]; del row["o_d"]
                        clean_results.append(row)
                    
                    df = pd.DataFrame(clean_results)
                    st.markdown("### 📋 Partidos Ordenados Cronológicamente")
                    st.write(df.to_html(escape=False), unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### 🔄 Actualizador en Vivo por Marcador")
                    minute = st.slider("Minuto del partido:", 1, 89, 15)
                    
                    for idx, match in enumerate(results_sorted):
                        with st.expander(f"⚽ {match['home_p']} vs {match['away_p']}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                g_home = st.number_input(f"Goles {match['home_p']}", min_value=0, value=0, key=f"gh_{idx}")
                            with c2:
                                g_away = st.number_input(f"Goles {match['away_p']}", min_value=0, value=0, key=f"ga_{idx}")
                            
                            time_remaining_ratio = max(0.05, (90 - minute) / 90.0)
                            live_h_xg = round(match["h_xg"] * time_remaining_ratio, 2)
                            live_a_xg = round(match["a_xg"] * time_remaining_ratio, 2)
                            
                            live_picks, btts_y, btts_n = calculate_predictions(
                                match["home_p"], match["away_p"], 
                                live_h_xg, live_a_xg, match["o_h"], match["o_a"], match["o_d"],
                                home_goals=g_home, away_goals=g_away
                            )
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("#### 📌 Pre-Partido (Base)")
                                st.write(f"1️⃣ {match['pre_picks'][0][0]} ({match['pre_picks'][0][1]*100:.1f}%)")
                                st.write(f"2️⃣ {match['pre_picks'][1][0]} ({match['pre_picks'][1][1]*100:.1f}%)")
                                st.write(f"3️⃣ {match['pre_picks'][2][0]} ({match['pre_picks'][2][1]*100:.1f}%)")
                            
                            with col_b:
                                st.markdown(f"#### ⚡ En Vivo (Minuto {minute}' | {g_home}-{g_away})")
                                st.write(f"🔥 **1️⃣** {live_picks[0][0]} (**{live_picks[0][1]*100:.1f}%**)")
                                st.write(f"🟢 **2️⃣** {live_picks[1][0]} (**{live_picks[1][1]*100:.1f}%**)")
                                st.write(f"🔵 **3️⃣** {live_picks[2][0]} (**{live_picks[2][1]*100:.1f}%**)")
                                st.caption(f"Ambos Marcan en vivo: **SÍ ({btts_y*100:.0f}%)** | **NO ({btts_n*100:.0f}%)**")

        except Exception as e:
            st.error(f"Error procesando datos: {e}")
