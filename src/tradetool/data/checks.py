"""Sanity checks for the cached data. tests/test_data_cache.py runs them over every file when the
cache exists, so a change in the download code that breaks the data shows up in pytest.

Each function returns a list of problems (empty = fine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradetool.data.prices import MAX_GAP_DAYS

OHLC = ["open", "high", "low", "close"]


def price_problems(df: pd.DataFrame) -> list[str]:
    out = []
    if df.empty:
        return ["no rows"]
    missing = {"open", "high", "low", "close", "volume", "close_raw"} - set(df.columns)
    if missing:
        return [f"missing columns {sorted(missing)}"]
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        out.append("dates not sorted or duplicated")
    if (df.index.day_of_week > 4).any():
        out.append("weekend dates")
    if df[OHLC + ["volume", "close_raw"]].isna().any().any():
        out.append("NaN values")
    if (df[OHLC + ["close_raw"]] <= 0).any().any():
        out.append("prices <= 0")
    tol = 1e-6 * df["close"]
    if (df["high"] < df[["open", "close"]].max(axis=1) - tol).any():
        out.append(f"high below open/close on {(df['high'] < df[['open', 'close']].max(axis=1) - tol).sum()} days")
    if (df["low"] > df[["open", "close"]].min(axis=1) + tol).any():
        out.append(f"low above open/close on {(df['low'] > df[['open', 'close']].min(axis=1) + tol).sum()} days")
    if (df["volume"] <= 0).any():
        out.append("days with zero volume")
    gaps = df.index.to_series().diff().dt.days
    if (gaps > MAX_GAP_DAYS).any():
        out.append(f"gap of {int(gaps.max())} days")
    return out


def big_moves(df: pd.DataFrame, limit=0.5) -> pd.Series:
    """Daily closes that moved more than `limit`. Not an error by itself (MRNA really rose 177% in one
    day), but every one of them should be explainable."""
    r = df["close"].pct_change()
    return r[r.abs() > limit]


def fundamental_problems(f: pd.DataFrame) -> list[str]:
    out = []
    if f.empty:
        return out
    if f.duplicated(["concept", "end"]).any():
        out.append("same concept and period twice")
    if not np.isfinite(f["value"]).all():
        out.append("non-finite values")
    if (f["end"] > f["filed"]).any():
        out.append(f"{(f['end'] > f['filed']).sum()} values dated after their own filing")
    return out
