"""Performance numbers for an equity curve, and the comparison with buy & hold and SPY."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradetool.data import prices

YEAR = 252


def metrics(equity: pd.Series, trades: pd.DataFrame | None = None, position: pd.Series | None = None) -> dict:
    equity = equity.dropna()
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    dd = equity / equity.cummax() - 1
    m = {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "volatility": r.std() * np.sqrt(YEAR),
        # risk-free rate taken as 0, so this is slightly flattering in 2023-25 when cash paid ~5%
        "sharpe": r.mean() / r.std() * np.sqrt(YEAR) if r.std() > 0 else np.nan,
        "max_drawdown": dd.min(),
    }
    m["calmar"] = m["cagr"] / -m["max_drawdown"] if m["max_drawdown"] < 0 else np.nan
    if position is not None:
        m["exposure"] = position.mean()
    if trades is not None:
        ret = trades["return"]
        m["trades"] = len(trades)
        m["win_rate"] = (ret > 0).mean() if len(ret) else np.nan
        m["avg_trade"] = ret.mean() if len(ret) else np.nan
        loss = -ret[ret < 0].sum()
        m["profit_factor"] = ret[ret > 0].sum() / loss if loss > 0 else np.nan
        m["avg_days"] = trades["days"].mean() if len(trades) else np.nan
    return m


def buy_and_hold(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Bought at the close of the first day of `index`, no costs. Same dates as the strategy."""
    c = df["close"].reindex(index).ffill()
    return c / c.iloc[0]


def spy(index: pd.DatetimeIndex) -> pd.Series:
    return buy_and_hold(prices.load("SPY"), index)


def compare(result, df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """One row each for the strategy, buy & hold of the same stock and SPY."""
    idx = result.equity.index
    bench = spy(idx) if bench is None else bench.reindex(idx).ffill() / bench.reindex(idx).ffill().iloc[0]
    rows = {"strategy": metrics(result.equity, result.trades, result.position),
            "buy & hold": metrics(buy_and_hold(df, idx)),
            "SPY": metrics(bench)}
    out = pd.DataFrame(rows).T
    r = result.equity.pct_change()
    b = bench.pct_change()
    out.loc["strategy", "beta_spy"] = r.cov(b) / b.var()
    out.loc["strategy", "corr_spy"] = r.corr(b)
    return out
