import pandas as pd
import streamlit as st

from tradetool.data.prices import MANIFEST
from tradetool.data.universe import load_universe

st.title("Εργαλεία ανάλυσης και αξιολόγησης στρατηγικών")

st.markdown("""
Ειδικό θέμα, ΗΜΜΥ Πανεπιστημίου Θεσσαλίας.

Σελίδες:
- Διαγράμματα: τιμή μετοχής ή ETF σε διάφορους τύπους γραφήματος, με τεχνικούς δείκτες της επιλογής σου.
- Backtest: μια στρατηγική από τη βιβλιοθήκη σε μία μετοχή, με σύγκριση με buy & hold και SPY.
- Υπερπαράμετροι: χάρτης ευαισθησίας των παραμέτρων και walk-forward (επιλογή σε ένα διάστημα, έλεγχος
  στο επόμενο) με διάφορες μεθόδους.
- Επιλογή μετοχών: σκορ με momentum, θεμελιώδη και δύναμη κλάδου, το καλάθι των καλύτερων και το backtest του.
- Στρατηγική με λόγια: περιγραφή σε φυσική γλώσσα, που ένα LLM μετατρέπει σε κανόνα για backtest.

Σε όλα το σήμα βγαίνει στο κλείσιμο μιας μέρας και η συναλλαγή γίνεται στο άνοιγμα της επόμενης.
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
    st.info("Δεν έχουν κατέβει δεδομένα ακόμα: `python -m tradetool.data.setup` (περίπου 5 λεπτά). Οι σελίδες "
            "μίας μετοχής δουλεύουν και χωρίς αυτό, η επιλογή μετοχών το χρειάζεται.")
