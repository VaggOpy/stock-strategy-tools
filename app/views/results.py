"""Backtest results as shown on the backtest page and the text strategy page (not a page itself)."""
import json

import pandas as pd
import streamlit as st

from tradetool.backtest.metrics import buy_and_hold
from tradetool.charts.backtest import add_trades, equity_chart, rule_indicators
from tradetool.charts.price import price_chart

LABELS = {"cagr": "CAGR", "sharpe": "Sharpe", "max_drawdown": "Max drawdown", "volatility": "Μεταβλητότητα",
          "calmar": "Calmar", "exposure": "Χρόνος στην αγορά", "trades": "Συναλλαγές", "win_rate": "Κερδοφόρες",
          "avg_trade": "Μέση συναλλαγή", "profit_factor": "Profit factor", "avg_days": "Μέρες ανά συναλλαγή",
          "beta_spy": "Beta vs SPY", "total_return": "Συνολική απόδοση", "corr_spy": "Συσχέτιση με SPY"}
PCT = {"total_return", "cagr", "max_drawdown", "volatility", "exposure", "win_rate", "avg_trade"}


def show(res, table: pd.DataFrame, df: pd.DataFrame, spy: pd.DataFrame, ticker: str, start, end, log=False):
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
        st.dataframe(res.trades.style.format({"entry_price": "{:.2f}", "exit_price": "{:.2f}", "return": "{:.1%}",
                                              "pnl": "{:+.3f}"}),
                     width="stretch")
    with st.expander("Ο κανόνας σε JSON"):
        st.code(json.dumps(res.rule, indent=2), language="json")

    st.caption("Το σήμα βγαίνει στο κλείσιμο και η εντολή εκτελείται στο άνοιγμα της επόμενης μέρας. "
               "Buy & hold και SPY ξεκινούν από το κλείσιμο της πρώτης μέρας, χωρίς κόστη.")
