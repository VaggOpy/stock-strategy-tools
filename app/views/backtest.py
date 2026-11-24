import json

import pandas as pd
import streamlit as st

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import buy_and_hold, compare
from tradetool.charts.backtest import add_trades, equity_chart, rule_indicators
from tradetool.charts.price import price_chart
from tradetool.config import TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.rules.schema import library

LABELS = {"cagr": "CAGR", "sharpe": "Sharpe", "max_drawdown": "Max drawdown", "volatility": "Μεταβλητότητα",
          "calmar": "Calmar", "exposure": "Χρόνος στην αγορά", "trades": "Συναλλαγές", "win_rate": "Κερδοφόρες",
          "avg_trade": "Μέση συναλλαγή", "profit_factor": "Profit factor", "avg_days": "Μέρες ανά συναλλαγή",
          "beta_spy": "Beta vs SPY", "total_return": "Συνολική απόδοση", "corr_spy": "Συσχέτιση με SPY"}
PCT = {"total_return", "cagr", "max_drawdown", "volatility", "exposure", "win_rate", "avg_trade"}


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
    period = st.date_input("Περίοδος", (first.date(), df.index[-1].date()),
                           min_value=df.index[0].date(), max_value=df.index[-1].date())
    cost = st.number_input("Κόστος ανά συναλλαγή (bps, κάθε πλευρά)", 0.0, 50.0, 5.0, 1.0)
    log = st.checkbox("Log κλίμακα")

st.title(f"{ticker}: {rule['name']}")
st.caption(rule.get("description", ""))

if len(period) != 2:
    st.stop()
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
res = run(df, rule, values, start=start, end=end, cost_bps=cost)
spy = load("SPY")
table = compare(res, df[df.index <= end], spy["close"])

cols = st.columns(4)
for col, key in zip(cols, ["cagr", "sharpe", "max_drawdown", "exposure"]):
    v = table.loc["strategy", key]
    delta = None
    if key != "exposure":
        diff = v - table.loc["SPY", key]
        delta = f"{diff * 100:+.1f} μον. vs SPY" if key in PCT else f"{diff:+.2f} vs SPY"
    col.metric(LABELS[key], f"{v:.1%}" if key in PCT else f"{v:.2f}", delta)

idx = res.equity.index
curves = {"Στρατηγική": res.equity, f"Buy & hold {ticker}": buy_and_hold(df, idx),
          "SPY": buy_and_hold(spy, idx)}
st.plotly_chart(equity_chart(curves, log=log), width="stretch")

shown = table.rename(columns=LABELS, index={"strategy": "Στρατηγική", "buy & hold": f"Buy & hold {ticker}"})
fmt = {LABELS[k]: "{:.1%}" for k in PCT} | {LABELS[k]: "{:.2f}" for k in ("sharpe", "calmar", "profit_factor",
                                                                             "beta_spy", "avg_days")}
fmt |= {LABELS["trades"]: "{:.0f}", LABELS["corr_spy"]: "{:.2f}"}
st.dataframe(shown.style.format(fmt, na_rep=""), width="stretch")

st.subheader("Συναλλαγές στο γράφημα")
kind = "candles" if (end - start).days < 400 else "line"  # candles are unreadable over many years
fig = price_chart(df[df.index <= end], kind, rule_indicators(res.rule), ticker, start=start, volume=False,
                  height=600)
st.plotly_chart(add_trades(fig, res.trades), width="stretch")

with st.expander(f"Όλες οι συναλλαγές ({len(res.trades)})"):
    st.dataframe(res.trades.style.format({"entry_price": "{:.2f}", "exit_price": "{:.2f}", "return": "{:.1%}"}),
                 width="stretch")
with st.expander("Ο κανόνας σε JSON"):
    st.code(json.dumps(res.rule, indent=2), language="json")

st.caption("Το σήμα βγαίνει στο κλείσιμο και η εντολή εκτελείται στο άνοιγμα της επόμενης μέρας. "
           "Buy & hold και SPY ξεκινούν από το κλείσιμο της πρώτης μέρας, χωρίς κόστη.")
