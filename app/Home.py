import streamlit as st

st.set_page_config(page_title="Stock strategy tools", layout="wide")

pages = [
    st.Page("views/home.py", title="Αρχική", default=True),
    st.Page("views/charts.py", title="Διαγράμματα"),
    st.Page("views/backtest.py", title="Backtest"),
]
st.navigation(pages).run()
