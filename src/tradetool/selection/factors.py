"""Scores for every stock at every rebalance date, using only what was known on that date.

Momentum uses closes up to the rebalance date. Fundamentals use filings made before it (see
data/fundamentals.py: a filing is usable from the next day). Each raw value is turned into a percentile
rank among the stocks that have it on that date, so the factors can be added together.
"""
from __future__ import annotations

import numpy as np
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


def fundamental_history(ticker: str) -> dict[str, pd.Series]:
    """Revenue growth, EPS growth (last 12 months against the 12 before) and operating cash flow of the
    last 12 months, each recomputed after every filing of that item and usable from the next day."""
    try:
        f = fundamentals.load(ticker)
    except KeyError:
        return {}
    return {
        "rev_growth": fundamentals.history(f, "revenue", fundamentals.growth_of),
        "eps_growth": fundamentals.history(f, "eps_diluted", fundamentals.growth_of),
        "ocf_ttm": fundamentals.history(f, "op_cash_flow", fundamentals.ttm_of),
    }


MAX_AGE_DAYS = 200  # quarterly filers file at most ~130 days apart; older values are stale


def fundamental_panel(tickers: list[str], dates) -> dict[str, pd.DataFrame]:
    """dates x tickers for each fundamental field, the value after the last filing of that item made
    usable on or before each date. If that filing gave NaN (e.g. a quarter is missing) the field is NaN,
    not an older value. A value older than MAX_AGE_DAYS (the company stopped filing, or files only once
    a year) counts as missing."""
    dates = pd.DatetimeIndex(dates)
    fields: dict[str, dict] = {"rev_growth": {}, "eps_growth": {}, "ocf_ttm": {}}
    for t in tickers:
        h = fundamental_history(t)
        for k in fields:
            s = h.get(k)
            if s is None or len(s) == 0:
                continue
            s = s.sort_index()
            pos = s.index.searchsorted(dates, side="right") - 1
            last = pos.clip(0)
            age = (dates - s.index[last]).days
            fields[k][t] = pd.Series(np.where((pos >= 0) & (age <= MAX_AGE_DAYS), s.to_numpy()[last], np.nan),
                                     index=dates)
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
