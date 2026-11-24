import pandas as pd
import streamlit as st

from tradetool.data.prices import MANIFEST
from tradetool.data.universe import load_universe

st.title("Εργαλεία ανάλυσης και αξιολόγησης στρατηγικών")

st.markdown("""
Ειδικό θέμα, ΗΜΜΥ Πανεπιστημίου Θεσσαλίας.

Σελίδες:
- **Διαγράμματα**: τιμή μετοχής ή ETF σε διάφορους τύπους γραφήματος, με τεχνικούς δείκτες.

Επόμενα: backtest, επιλογή υπερπαραμέτρων, επιλογή μετοχών, στρατηγική σε φυσική γλώσσα.
""")

u = load_universe()
st.subheader("Δεδομένα")
c1, c2, c3 = st.columns(3)
c1.metric("Μετοχές", int((u["type"] == "stock").sum()))
c2.metric("ETFs", int((u["type"] == "etf").sum()))
if MANIFEST.exists():
    m = pd.read_csv(MANIFEST)
    c3.metric("Τελευταία μέρα", str(m["last"].max()))
    st.caption(f"Πηγές τιμών: {m['source'].value_counts().to_dict()}. Λίστα μετοχών από {u['snapshot'].iloc[0]} "
               "(σημερινή σύνθεση των δεικτών, άρα υπάρχει survivorship bias).")
else:
    st.info("Δεν έχουν κατέβει τιμές ακόμα: `python -m tradetool.data.prices`")
