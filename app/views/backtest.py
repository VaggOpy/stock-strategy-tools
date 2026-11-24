import pandas as pd
import streamlit as st

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import compare
from tradetool.config import TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.rules.schema import library
from views import results


@st.cache_data
def universe():
    return load_universe()


@st.cache_data(ttl=3600)
def load(ticker):
    return prices.load(ticker, refresh=True)


rules = library()
u = universe()

with st.sidebar:
    labels = dict(zip(u["ticker"], u["ticker"] + "  " + u["name"].fillna("")))
    ticker = st.selectbox("Σύμβολο", list(labels), index=list(labels).index("AAPL"), format_func=labels.get)
    name = st.selectbox("Στρατηγική", list(rules), format_func=lambda k: rules[k]["name"])
    rule = rules[name]

    st.caption("Παράμετροι")
    values = {}
    for p, spec in rule.get("params", {}).items():
        step = spec.get("step", 1)
        if float(step).is_integer() and float(spec["default"]).is_integer():
            values[p] = st.slider(p, int(spec["low"]), int(spec["high"]), int(spec["default"]), int(step),
                                  key=f"{name}_{p}")
        else:
            values[p] = st.slider(p, float(spec["low"]), float(spec["high"]), float(spec["default"]), float(step),
                                  key=f"{name}_{p}")

    st.divider()
    df = load(ticker)
    first = max(df.index[0] + pd.Timedelta(days=300), pd.Timestamp(TEST_START))
    short = first >= df.index[-1]
    if short:  # listed less than a year ago, no room for a warm-up
        first = df.index[0]
    period = st.date_input("Περίοδος", (first.date(), df.index[-1].date()),
                           min_value=df.index[0].date(), max_value=df.index[-1].date())
    cost = st.number_input("Κόστος ανά συναλλαγή (bps, κάθε πλευρά)", 0.0, 50.0, 5.0, 1.0)
    log = st.checkbox("Log κλίμακα")

st.title(f"{ticker}: {rule['name']}")
st.caption(rule.get("description", ""))

if len(period) != 2:
    st.stop()
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
if short:
    st.info(f"Το {ticker} έχει δεδομένα μόνο από {df.index[0]:%d/%m/%Y}. Οι μακριοί δείκτες δεν προλαβαίνουν "
            "να υπολογιστούν, οπότε η στρατηγική μπορεί να μην κάνει καμία συναλλαγή.")
res = run(df, rule, values, start=start, end=end, cost_bps=cost)
if len(res.equity) < 2:
    st.warning("Η περίοδος είναι πολύ μικρή για backtest.")
    st.stop()
spy = load("SPY")
table = compare(res, df[df.index <= end], spy["close"])

results.show(res, table, df, spy, ticker, start, end, log)
