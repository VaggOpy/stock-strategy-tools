import streamlit as st

st.set_page_config(page_title="Stock strategy tools", layout="wide")

pages = [
    st.Page("views/home.py", title="Αρχική", default=True),
    st.Page("views/charts.py", title="Διαγράμματα"),
    st.Page("views/backtest.py", title="Backtest"),
    st.Page("views/optimize.py", title="Υπερπαράμετροι"),
    st.Page("views/selection.py", title="Επιλογή μετοχών"),
    st.Page("views/text_strategy.py", title="Στρατηγική με λόγια"),
]
st.navigation(pages).run()
