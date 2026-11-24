"""Scores for every stock at every rebalance date, using only what was known on that date.

Momentum uses closes up to the rebalance date. Fundamentals use filings made before it (see
data/fundamentals.py: a filing is usable from the next day). Each raw value is turned into a percentile
rank among the stocks that have it on that date, so the factors can be added together.
"""
from __future__ import annotations

import pandas as pd

from tradetool.data import fundamentals

MONTH = 21
MOMENTUM_MONTHS = (3, 6, 12)


def month_ends(index: pd.DatetimeIndex, start=None, end=None) -> pd.DatetimeIndex:
    """Last trading day of each month in index."""
    s = pd.Series(index, index=index)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).max().values)


def momentum(close: pd.DataFrame, dates) -> dict[str, pd.DataFrame]:
    """Return over 3, 6 and 12 months, leaving out the last month (short-term reversal), at each date.
    A stock needs a price on the date and 12 months + 1 month of history."""
    out = {}
    for m in MOMENTUM_MONTHS:
        r = close.shift(MONTH) / close.shift(MONTH * (m + 1)) - 1
        out[f"mom_{m}m"] = r.reindex(dates)
    listed = close.notna().cumsum() > MONTH * 13
    ok = (close.notna() & listed).reindex(dates)
    return {k: v.where(ok) for k, v in out.items()}


def fundamental_history(ticker: str) -> pd.DataFrame:
    """Revenue growth, EPS growth and operating cash flow (all over the last 12 months against the 12
    before), each from the day after the filing that made it known."""
    try:
        f = fundamentals.load(ticker)
    except KeyError:
        return pd.DataFrame()
    cols = {
        "rev_growth": fundamentals.history(f, "revenue", fundamentals.growth_of),
        "eps_growth": fundamentals.history(f, "eps_diluted", fundamentals.growth_of),
        "ocf_ttm": fundamentals.history(f, "op_cash_flow", fundamentals.ttm_of),
    }
    return pd.DataFrame(cols).sort_index()


MAX_AGE_DAYS = 200  # quarterly filers file at most ~130 days apart; older values are stale


def fundamental_panel(tickers: list[str], dates) -> dict[str, pd.DataFrame]:
    """dates x tickers for each fundamental field, the latest value known at each date. A value older
    than MAX_AGE_DAYS (the company stopped filing, or files only once a year) counts as missing."""
    dates = pd.DatetimeIndex(dates)
    fields: dict[str, dict] = {"rev_growth": {}, "eps_growth": {}, "ocf_ttm": {}}
    for t in tickers:
        h = fundamental_history(t)
        if h.empty:
            continue
        # value at date d = last filing made usable on or before d
        h = h.assign(_since=h.index)
        at = h.reindex(h.index.union(dates)).ffill().reindex(dates)
        fresh = (dates - pd.DatetimeIndex(at["_since"])).days <= MAX_AGE_DAYS
        for k in fields:
            if k in at:
                fields[k][t] = at[k].where(fresh)
    return {k: pd.DataFrame(v, index=dates).reindex(columns=tickers) for k, v in fields.items()}


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank in each row (date), 0 = worst, 1 = best. NaN stays NaN."""
    return x.rank(axis=1, pct=True)


def combine(parts: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Weighted mean of the available ranks. A missing part does not count, so a stock without
    fundamentals (e.g. a foreign filer) is scored on the rest. NaN only where all parts are NaN."""
    weights = weights or [1.0] * len(parts)
    total = sum(w * p.fillna(0) for w, p in zip(weights, parts))
    norm = sum(w * p.notna() for w, p in zip(weights, parts))
    return total / norm.where(norm > 0)
