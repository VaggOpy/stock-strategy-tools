"""Parameter grid and the matrix of daily returns, one column per parameter combination.

Every combination is run once over the whole period and only its daily returns are kept. A window
(train or test) is then just a slice of rows. This is allowed because the engine is causal: the
return on day t depends only on data up to day t (tests/test_backtest.py checks it), so slicing the
full run gives the same numbers as a run that stopped at the end of the window. Positions opened
before a window are carried into it, as they would be in real trading.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from tradetool.backtest.engine import run

YEAR = 252


def grid(rule: dict, names: list[str]) -> dict[str, np.ndarray]:
    """Values of each searched parameter, from its low/high/step in the rule."""
    axes = {}
    for n in names:
        p = rule["params"][n]
        step = p.get("step", 1)
        axes[n] = np.round(np.arange(p["low"], p["high"] + step / 2, step), 6)
    return axes


def combos(axes: dict[str, np.ndarray]) -> list[dict]:
    """All combinations, in the same (C) order as np.zeros(shape).ravel()."""
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


def default_index(rule: dict, axes: dict[str, np.ndarray]) -> int:
    pos = [int(np.abs(v - rule["params"][n]["default"]).argmin()) for n, v in axes.items()]
    return int(np.ravel_multi_index(pos, [len(v) for v in axes.values()]))


def returns_matrix(df: pd.DataFrame, rule: dict, axes: dict, start, end=None, cost_bps=5.0,
                   valid=None) -> pd.DataFrame:
    """Daily returns from `start`, one column per combination. Combinations rejected by valid(values)
    (e.g. fast EMA longer than slow) are NaN."""
    cache: dict = {}
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    cols = {}
    for i, values in enumerate(combos(axes)):
        if valid is not None and not valid(values):
            continue
        eq = run(df, rule, values, start=start, cost_bps=cost_bps, cache=cache).equity
        cols[i] = eq.pct_change().fillna(0.0).to_numpy(np.float32)
    idx = df.index[df.index.searchsorted(pd.Timestamp(start)):]
    out = pd.DataFrame(np.nan, index=idx, columns=range(len(combos(axes))), dtype=np.float32)
    for i, r in cols.items():
        out[i] = r
    return out


def sharpe(r) -> np.ndarray:
    """Annualised Sharpe of each column (risk-free 0). A column that never traded gets 0, like cash."""
    r = np.asarray(r, dtype=float)
    sd = r.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = r.mean(axis=0) / sd * np.sqrt(YEAR)
    return np.where(sd > 0, s, np.where(np.isnan(sd), np.nan, 0.0))
