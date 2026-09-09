import streamlit as st
    conn = sqlite3.connect('bets_tracker.db')
    df_bets = pd.read_sql_query("SELECT * FROM bets", conn)
    conn.close()
    
    st.markdown("### 📊 Historial de Apuestas Registradas")
    st.dataframe(df_bets)
