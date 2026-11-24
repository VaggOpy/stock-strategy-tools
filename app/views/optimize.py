import numpy as np
import pandas as pd
import streamlit as st

from tradetool.backtest.metrics import metrics
from tradetool.charts.backtest import equity_chart
from tradetool.charts.optimize import sensitivity_chart
from tradetool.config import TEST_END, TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.optimize.search import default_index, grid, returns_matrix, sharpe
from tradetool.optimize.walkforward import choose, folds, walk_forward, window
from tradetool.rules.schema import library

METHODS = {"fixed": "Σταθερές τιμές", "grid": "Grid (καλύτερο)", "optuna": "Optuna",
           "plateau": "Οροπέδιο", "topk": "Top-5 μαζί"}
# same searched parameters as experiments/walkforward_methods.py
SEARCH = {"ema_cross": ["fast", "slow"], "sma_trend": ["length"], "rsi_pullback": ["level", "exit_len"],
          "donchian_breakout": ["entry_len", "exit_len"]}


@st.cache_data
def universe():
    return load_universe()


@st.cache_data(show_spinner="Backtest σε όλο το πλέγμα παραμέτρων...")
def matrix(ticker, name):
    rule = library()[name]
    axes = grid(rule, SEARCH[name])
    valid = (lambda v: v["fast"] < v["slow"]) if name == "ema_cross" else None
    df = prices.load(ticker, end=TEST_END)
    R, P = returns_matrix(df, rule, axes, TEST_START, valid=valid)
    return R, P, axes, default_index(rule, axes)


@st.cache_data
def run_wf(ticker, name, first_test):
    R, P, axes, default = matrix(ticker, name)
    shape = [len(v) for v in axes.values()]
    rows, oos = walk_forward(R, P, shape, default, folds(R.index[-1], first_test))
    return pd.DataFrame(rows), oos


rules = library()
u = universe()
stocks = u[u["type"] == "stock"]

with st.sidebar:
    labels = dict(zip(stocks["ticker"], stocks["ticker"] + "  " + stocks["name"].fillna("")))
    ticker = st.selectbox("Μετοχή", list(labels), index=list(labels).index("MSFT"), format_func=labels.get)
    name = st.selectbox("Στρατηγική", list(SEARCH), format_func=lambda k: rules[k]["name"])
    st.caption(f"Παράμετροι που ψάχνονται: {', '.join(SEARCH[name])}. Οι υπόλοιπες μένουν στις προεπιλογές.")
    st.caption(f"Walk-forward: 3 χρόνια εκπαίδευση, 1 χρόνος έλεγχος, δεδομένα έως {TEST_END}.")

st.title(f"Επιλογή παραμέτρων: {ticker}, {rules[name]['name']}")

try:
    R, P, axes, default = matrix(ticker, name)
except KeyError:
    st.error(f"Δεν υπάρχουν δεδομένα για {ticker}.")
    st.stop()
if R.index[0] > pd.Timestamp(TEST_START) + pd.Timedelta(days=10):
    st.warning("Η μετοχή δεν έχει δεδομένα για όλη την περίοδο, τα πρώτα διαστήματα λείπουν.")

shape = [len(v) for v in axes.values()]
rows, oos = run_wf(ticker, name, "2020-01-01")
fl = folds(R.index[-1], "2020-01-01")
# only the years walk_forward could use (enough data in the train window)
used = set(rows["test_start"]) if len(rows) else set()
years = {f.test_start.year: f for f in fl if f.test_start in used}
if not years:
    st.info("Η μετοχή δεν έχει αρκετό ιστορικό για ούτε ένα διάστημα εκπαίδευσης.")
    st.stop()

st.subheader("Χάρτης ευαισθησίας")
year = st.select_slider("Χρονιά ελέγχου", list(years), value=list(years)[-2 if len(years) > 1 else -1])
f = years[year]
s_in = sharpe(window(R, f.train_start, f.train_end))
s_out = sharpe(window(R, f.test_start, f.test_end))
picks = choose(s_in, shape, default, seed=fl.index(f))  # same seed as walk_forward uses for this fold
st.caption(f"Εκπαίδευση {f.train_start:%d/%m/%Y} έως {f.train_end - pd.Timedelta(days=1):%d/%m/%Y}, "
           f"έλεγχος {f.test_start:%d/%m/%Y} έως {f.test_end - pd.Timedelta(days=1):%d/%m/%Y}. "
           "Κάθε κελί είναι το Sharpe ενός συνδυασμού. Τα σημάδια δείχνουν τι διάλεξε κάθε μέθοδος "
           "με βάση μόνο τον αριστερό χάρτη.")
st.plotly_chart(sensitivity_chart(axes, s_in, s_out, picks), width="stretch")

ok = ~np.isnan(s_in) & ~np.isnan(s_out)
rho = pd.Series(s_in[ok]).corr(pd.Series(s_out[ok]), method="spearman")
c1, c2, c3 = st.columns(3)
c1.metric("Συσχέτιση εντός/εκτός", f"{rho:.2f}")
c2.metric("Καλύτερος εκτός δείγματος", f"{np.nanmax(s_out):.2f}")
c3.metric("Διάμεσος εκτός δείγματος", f"{np.nanmedian(s_out):.2f}")
st.caption("Αν η συσχέτιση είναι κοντά στο 0, το ποιος συνδυασμός ήταν καλύτερος στο παρελθόν δεν λέει σχεδόν "
           "τίποτα για το ποιος θα είναι καλύτερος την επόμενη χρονιά.")

st.subheader("Walk-forward σε όλες τις χρονιές")
table = rows.pivot(index="test_start", columns="method", values="oos_sharpe")[list(METHODS)]
table.index = table.index.year
table.columns = [METHODS[c] for c in table.columns]
st.dataframe(table.style.format("{:.2f}"), width="stretch")

spy = prices.load("SPY", end=TEST_END)["close"]
first = min(r.index[0] for r in oos.values())
spy = spy[spy.index >= first]
curves = {METHODS[m]: (1 + oos[m]).cumprod() for m in ("grid", "fixed")}
curves["SPY"] = spy / spy.iloc[0]
st.plotly_chart(equity_chart(curves, height=480), width="stretch")

summary = pd.DataFrame({METHODS[m]: metrics((1 + r).cumprod()) for m, r in oos.items()})
summary["SPY"] = pd.Series(metrics(curves["SPY"]))
summary = summary.T[["cagr", "sharpe", "max_drawdown"]]
summary.columns = ["CAGR", "Sharpe", "Max drawdown"]
st.dataframe(summary.style.format({"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}"}),
             width="stretch")
st.caption("Μόνο τα διαστήματα ελέγχου, ενωμένα. Σε κάθε χρονιά κάθε μέθοδος χρησιμοποιεί τις παραμέτρους που "
           "διάλεξε με τα δεδομένα των τριών προηγούμενων ετών.")
