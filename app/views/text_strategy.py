import json

import pandas as pd
import streamlit as st

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import compare
from tradetool.config import TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.llm.client import LLMError, default_client
from tradetool.llm.strategy import explain, facts, to_rule, unknown_numbers
from tradetool.rules.describe import describe
from views import results

EXAMPLES = [
    "Αγόρασε όταν ο EMA 10 περάσει πάνω από τον EMA 30 και πούλα όταν περάσει από κάτω. Stop στα 2 ATR.",
    "Buy when RSI(2) drops below 10 while the close is above the 200-day SMA, sell when the close gets back "
    "above the 5-day SMA.",
    "Αγόρασε όταν το κλείσιμο σπάσει το υψηλό των τελευταίων 50 ημερών και ο όγκος είναι μεγάλος. Βγες μετά από "
    "20 μέρες ή αν πέσει κάτω από τον EMA 20.",
    "Αγόρασε στα βουτήγματα όσο η μετοχή είναι σε ανοδική τάση.",
]


@st.cache_data
def universe():
    return load_universe()


@st.cache_data(ttl=3600)
def load(ticker):
    return prices.load(ticker, refresh=True)


@st.cache_resource
def llm():
    return default_client()


@st.cache_data(show_spinner="Το LLM γράφει τον κανόνα...")
def translate(text):
    return to_rule(text, llm())


@st.cache_data(show_spinner="Το LLM γράφει την εξήγηση...")
def explanation(text, rule_json, fact_text):
    return explain(llm(), text, json.loads(rule_json), fact_text)


u = universe()
with st.sidebar:
    labels = dict(zip(u["ticker"], u["ticker"] + "  " + u["name"].fillna("")))
    ticker = st.selectbox("Σύμβολο", list(labels), index=list(labels).index("AAPL"), format_func=labels.get)
    df = load(ticker)
    first = max(df.index[0] + pd.Timedelta(days=300), pd.Timestamp(TEST_START))
    if first >= df.index[-1]:
        first = df.index[0]
    period = st.date_input("Περίοδος", (first.date(), df.index[-1].date()),
                           min_value=df.index[0].date(), max_value=df.index[-1].date())
    cost = st.number_input("Κόστος ανά συναλλαγή (bps, κάθε πλευρά)", 0.0, 50.0, 5.0, 1.0)
    log = st.checkbox("Log κλίμακα")

st.title("Στρατηγική με λόγια")
st.caption("Γράψε μια στρατηγική όπως θα την έλεγες. Το LLM τη μετατρέπει σε κανόνα φτιαγμένο μόνο από τους "
           "δείκτες και τις συνθήκες του εργαλείου, όχι σε κώδικα. Ο κανόνας ελέγχεται πριν τρέξει και, αν έχει "
           "λάθη, γυρίζει πίσω στο LLM για διόρθωση (έως δύο φορές).")
try:
    st.caption(f"Μοντέλο: {llm().name}")
except LLMError as e:
    st.warning(str(e))
    st.stop()

example = st.selectbox("Παράδειγμα", ["(κανένα)"] + EXAMPLES)
text = st.text_area("Περιγραφή", "" if example == "(κανένα)" else example, height=110)
if st.button("Μετατροπή σε κανόνα και backtest", type="primary"):
    st.session_state["text"] = text.strip()
text = st.session_state.get("text")
if not text or len(period) != 2:
    st.stop()

try:
    t = translate(text)
except LLMError as e:
    st.error(str(e))
    st.stop()

if t.attempts > 1:
    with st.expander(f"Ο κανόνας χρειάστηκε {t.attempts} απαντήσεις από το LLM"):
        for i, problems in enumerate(t.problems, 1):
            st.markdown(f"Απάντηση {i}:\n" + "\n".join(f"- {p}" for p in problems))
if not t.ok:
    st.error("Το LLM δεν έγραψε κανόνα που να περνάει τον έλεγχο. Δοκίμασε να γράψεις την περιγραφή πιο συγκεκριμένα.")
    st.stop()
if t.unsupported:
    st.warning("Δεν γίνεται με τα κομμάτια του εργαλείου και παραλείφθηκε:\n" + "\n".join(f"- {s}" for s in t.unsupported))
if t.rule is None:
    if not t.unsupported:
        st.warning("Το LLM δεν έφτιαξε κανόνα και δεν είπε γιατί. Δοκίμασε να γράψεις την περιγραφή πιο συγκεκριμένα.")
    st.stop()

st.subheader(t.rule.get("name", "Ο κανόνας"))
st.markdown("\n".join(f"- {line}" for line in describe(t.rule)))
if t.assumptions:
    st.info("Υποθέσεις του LLM, όπου η περιγραφή δεν έλεγε κάτι:\n" + "\n".join(f"- {s}" for s in t.assumptions))
st.caption("Η περιγραφή παραπάνω βγαίνει από τον κανόνα με κώδικα και όχι από το LLM, οπότε είναι αυτό που θα τρέξει.")

start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
res = run(df, t.rule, start=start, end=end, cost_bps=cost)
if len(res.equity) < 2:
    st.warning("Η περίοδος είναι πολύ μικρή για backtest.")
    st.stop()
spy = load("SPY")
table = compare(res, df[df.index <= end], spy["close"])

st.subheader("Εξήγηση")
fact_text = facts(table, ticker, res.equity.index[0], res.equity.index[-1], cost)
try:
    words = explanation(text, json.dumps(t.rule, sort_keys=True), fact_text)
    st.write(words)
    made_up = unknown_numbers(words, fact_text, "\n".join(describe(t.rule)), text)
    note = "Γραμμένη από το LLM, που είδε μόνο τους αριθμούς του πίνακα παρακάτω."
    if made_up:
        note += f" Προσοχή: οι αριθμοί {', '.join(made_up)} δεν υπάρχουν στα αποτελέσματα."
    st.caption(note)
except LLMError as e:
    st.warning(f"Χωρίς εξήγηση: {e}")

results.show(res, table, df, spy, ticker, start, end, log)
