import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
import sqlite3
from datetime import datetime

st.set_page_config(page_title="AI Football Predictor ABSOLUTE", page_icon="⚽", layout="wide")

# ---------------------------------------------------------
# BASE DE DATOS LOCAL
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect('bets_tracker.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, match TEXT, selection TEXT,
                  odds REAL, stake REAL, result TEXT, pnl REAL)''')
    conn.commit()
    conn.close()

init_db()

def add_bet(match, selection, odds, stake):
    conn = sqlite3.connect('bets_tracker.db')
    c = conn.cursor()
    c.execute("INSERT INTO bets (date, match, selection, odds, stake, result, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d"), match, selection, odds, stake, 'PENDIENTE', 0.0))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# MOTOR ESTADÍSTICO DE PROBABILIDADES
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

def calculate_kelly_stake(prob, odds, max_bank_pct=0.03):
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return 0.0
    f = (b * prob - q) / b
    fractional_kelly = f * 0.25
    return max(0.0, min(fractional_kelly, max_bank_pct))

# ---------------------------------------------------------
# INTERFAZ Y PESTAÑAS
# ---------------------------------------------------------
st.title("⚽ AI Football Predictor — Sistema Absoluto v4.1")
st.caption("Escaner Multimercado de Alta Certeza (≥80% Probabilidad)")

tab_scanner, tab_lineups, tab_tracker = st.tabs(["🔍 Escáner Avanzado (+EV)", "📋 Alineaciones y Bajas", "📈 Tracker de Yield y Banca"])

with tab_scanner:
    api_key_odds = st.text_input("1. API Key de The-Odds-API", type="password")
    api_key_football = st.text_input("2. API Key de API-Football (Opcional)", type="password")
    min_odds_val = st.number_input("Cuota Mínima Aceptable", value=1.15, step=0.05)

    if st.button("🚀 Ejecutar Análisis Absoluto"):
        if not api_key_odds:
            st.error("Ingresa la API Key de The-Odds-API para continuar.")
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
                        
                        exp_corners = round((h_xg_est * 3.8) + (a_xg_est * 3.5) + 3.2, 1)
                        p_corners_7_5 = min(0.98, round(0.50 + (exp_corners - 7.5) * 0.08, 2))

                        candidates = [
                            ("Gana " + home_p, p_h, o_h),
                            ("Gana " + away_p, p_a, o_a),
                            ("Doble Oportunidad 1X", p_1x, round(1 / p_1x * 1.05, 2) if p_1x > 0 else 1.0),
                            ("Doble Oportunidad X2", p_x2, round(1 / p_x2 * 1.05, 2) if p_x2 > 0 else 1.0),
                            ("Más de 0.5 Goles", p_over_0_5, round(1 / p_over_0_5 * 1.05, 2) if p_over_0_5 > 0 else 1.0),
                            ("Más de 1.5 Goles", p_over_1_5, round(1 / p_over_1_5 * 1.08, 2) if p_over_1_5 > 0 else 1.0),
                            ("Menos de 3.5 Goles", p_under_3_5, round(1 / p_under_3_5 * 1.08, 2) if p_under_3_5 > 0 else 1.0),
                            ("Más de 7.5 Córneres", p_corners_7_5, 1.25)
                        ]
                        
                        picks = []
                        for name, prob, est_odds in candidates:
                            if prob >= 0.80 and est_odds >= min_odds_val:
                                stake = calculate_kelly_stake(prob, est_odds) * 100
                                picks.append(f"🟢 <b>{name}</b><br>• Prob: <b>{prob*100:.1f}%</b> | Cuota Est.: <b>{est_odds:.2f}</b><br>• Stake: <b>{stake:.1f}% banca</b>")
                        
                        results.append({
                            "Partido": f"<b>{home_p} vs {away_p}</b>",
                            "Cuotas 1X2": f"{o_h:.2f} | {o_d:.2f} | {o_a:.2f}",
                            "Pronósticos Sugeridos (≥80% Acierto)": "<br><br>".join(picks) if picks else "<span style='color:gray;'>Sin selecciones que superen la cuota mínima fijada</span>"
                        })
                        
                    df = pd.DataFrame(results)
                    st.write(df.to_html(escape=False), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error procesando datos: {e}")

with tab_lineups:
    st.subheader("📋 Verificación de Bajas y Onces Titulares")
    fixture_id = st.text_input("ID del Partido (Fixture ID)")
    if st.button("Consultar Alineaciones Confirmadas"):
        if api_key_football and fixture_id:
            headers = {'x-apisports-key': api_key_football}
            url_lineups = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
            res_l = requests.get(url_lineups, headers=headers)
            if res_l.status_code == 200:
                st.json(res_l.json())
            else:
                st.error("No se pudieron obtener las alineaciones.")
        else:
            st.warning("Se requiere la API Key de API-Football y el ID del partido.")

with tab_tracker:
    st.subheader("📈 Registro Profesional de Rentabilidad (Yield)")
    with st.form("add_bet_form"):
        f_match = st.text_input("Partido")
        f_select = st.text_input("Selección Apostada")
        f_odds = st.number_input("Cuota", value=1.50, step=0.01)
        f_stake = st.number_input("Monto Apostado ($)", value=10.0, step=1.0)
        submitted = st.form_submit_button("Guardar Apuesta")
        if submitted:
            add_bet(f_match, f_select, f_odds, f_stake)
            st.success("Apuesta registrada.")
            
    conn = sqlite3.connect('bets_tracker.db')
    df_bets = pd.read_sql_query("SELECT * FROM bets", conn)
    conn.close()
    st.dataframe(df_bets)
