"""Backtest of a portfolio that is rebalanced to target weights from time to time.

Same timing as the single-stock engine: the weights chosen with the close of day t are traded at the
OPEN of the next trading day. Between rebalances the holdings are left alone, so their weights drift
with the prices. Costs are cost_bps on the value traded (both buys and sells).

Prices: open and close panels (dates x tickers). A stock without a price on a day keeps its last close
(e.g. a trading halt). A target for a stock without an open price on the execution day is dropped and
the rest of the weights are scaled up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    equity: pd.Series      # account value at each close, 1.0 at the close of the first signal date
    turnover: pd.Series    # value traded / account value, per rebalance (execution date)
    holdings: dict         # execution date -> weights right after the rebalance


def simulate(targets: dict, open_: pd.DataFrame, close: pd.DataFrame, cost_bps: float = 5.0,
             end=None) -> PortfolioResult:
    """targets: signal date -> pd.Series of weights (summing to 1, or less to keep the rest in cash)."""
    close = close if end is None else close[close.index <= pd.Timestamp(end)]
    open_ = open_.reindex(index=close.index, columns=close.columns)
    cols = close.columns
    c_last = close.ffill().to_numpy(float)
    o = open_.to_numpy(float)
    idx = close.index
    cost = cost_bps / 1e4

    # signal date t -> execution on the next trading day
    plan = {}
    for t, w in targets.items():
        i = idx.searchsorted(pd.Timestamp(t), side="right")
        if i < len(idx):
            plan[i] = w.reindex(cols).fillna(0.0).to_numpy(float)
    if not plan:
        raise ValueError("no target can be executed inside the price data")
    first = min(plan) - 1

    shares = np.zeros(len(cols))
    cash = 1.0
    equity = np.full(len(idx), np.nan)
    turnover, holdings = {}, {}
    for i in range(first, len(idx)):
        if i in plan:
            px = np.where(np.isnan(o[i]), c_last[i - 1], o[i])
            value_open = cash + np.nansum(shares * px)
            w = plan[i] * ~np.isnan(o[i])
            if w.sum() > 0:
                w = w / w.sum() * plan[i].sum()
            old_value = shares * px
            traded = np.nansum(np.abs(w * value_open - np.nan_to_num(old_value)))
            value_after = value_open - traded * cost
            new_shares = np.where(w > 0, w * value_after / np.where(np.isnan(o[i]), 1.0, o[i]), 0.0)
            cash = value_after - np.nansum(new_shares * px)
            shares = new_shares
            turnover[idx[i]] = traded / value_open
            holdings[idx[i]] = pd.Series(w, index=cols)[w > 0]
        equity[i] = cash + np.nansum(shares * c_last[i])

    return PortfolioResult(pd.Series(equity[first:], idx[first:], name="equity"),
                           pd.Series(turnover, name="turnover"), holdings)


def top_n(score: pd.Series, n: int) -> pd.Series:
    """Equal weights on the n best scores (ties broken by ticker so the result is repeatable)."""
    s = score.dropna()
    s = s.sort_index().sort_values(ascending=False, kind="stable")
    pick = s.index[:n]
    return pd.Series(1.0 / len(pick), index=pick) if len(pick) else pd.Series(dtype=float)
