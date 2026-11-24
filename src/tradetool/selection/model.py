"""Stock selection: factor ranks at every month end, a weighted score, and the top N as a basket.

Factors (each a percentile rank among the stocks that have it, 1 = best):
  momentum      mean rank of the 3, 6 and 12 month returns, without the last month
  fundamentals  mean rank of revenue growth, EPS growth and positive operating cash flow (12 months)
  group_sic     strength of the stock's SEC industry group (mean momentum rank of its members)
  group_cluster the same with groups found by correlation clustering over the last year
A stock needs 13 months of prices to be scored at all. With `members`, only the stocks that were in the
index at each date are scored (see data/universe.add_sp500_dates).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from tradetool.backtest.portfolio import top_n
from tradetool.config import DATA_DIR, FUNDAMENTALS_DIR
from tradetool.data import prices
from tradetool.selection import factors as fx
from tradetool.selection.groups import group_panel

FACTORS = ["momentum", "fundamentals", "group_sic", "group_cluster"]


def sp500_members(universe: pd.DataFrame, dates, tickers) -> pd.DataFrame:
    """dates x tickers, True where the stock was already in the S&P 500 on that date."""
    added = pd.to_datetime(universe.set_index("ticker")["sp500_added"]).reindex(tickers)
    return pd.DataFrame({t: pd.DatetimeIndex(dates) >= a if pd.notna(a) else False for t, a in added.items()},
                        index=pd.DatetimeIndex(dates))


def build_factors(close: pd.DataFrame, dates, universe: pd.DataFrame, n_clusters: int = 25,
                  members: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """close: daily closes (dates x tickers) up to the last date needed. Returns rank panels (dates x tickers).
    members (dates x tickers, bool): if given, only these stocks are scored and ranked at each date."""
    dates = pd.DatetimeIndex(dates)
    mom = fx.momentum(close, dates)
    if members is not None:
        mom = {k: v.where(members.reindex(index=dates, columns=v.columns).fillna(False).astype(bool))
               for k, v in mom.items()}
    momentum = fx.rank(fx.combine([fx.rank(m) for m in mom.values()]))
    eligible = momentum.notna()

    fund = fx.fundamental_panel(list(close.columns), dates)
    ocf_pos = (fund["ocf_ttm"] > 0).astype(float).where(fund["ocf_ttm"].notna())
    parts = [fx.rank(fund["rev_growth"].where(eligible)), fx.rank(fund["eps_growth"].where(eligible)),
             fx.rank(ocf_pos.where(eligible))]
    fundamentals = fx.rank(fx.combine(parts)).where(eligible)

    daily = close.pct_change(fill_method=None)
    return {
        "momentum": momentum,
        "fundamentals": fundamentals,
        "group_sic": group_panel(momentum, "sic", universe=universe),
        "group_cluster": group_panel(momentum, "cluster", daily_returns=daily, n_groups=n_clusters),
    }


def score(ranks: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted mean of the factor ranks; only stocks with a momentum rank are scored."""
    names = [k for k, w in weights.items() if w > 0]
    s = fx.combine([ranks[k] for k in names], [weights[k] for k in names])
    return s.where(ranks["momentum"].notna())


def baskets(score_panel: pd.DataFrame, n: int = 20) -> dict:
    """signal date -> equal weights on the n best stocks."""
    return {d: top_n(score_panel.loc[d], n) for d in score_panel.index}


def _inputs_key(stocks, universe) -> str:
    """Changes when the stocks, the universe file, the fundamentals download or the factor code change,
    so the app never shows ranks computed by older code or data."""
    h = hashlib.sha256()
    h.update(",".join(stocks).encode())
    h.update(pd.util.hash_pandas_object(universe.astype(str), index=False).values.tobytes())
    summary = FUNDAMENTALS_DIR / "_summary.csv"
    h.update(str(summary.stat().st_mtime if summary.exists() else 0).encode())
    for f in ("factors.py", "groups.py", "model.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    h.update(Path(fx.fundamentals.__file__).read_bytes())
    return h.hexdigest()[:10]


def cached_factors(stocks: list[str], universe: pd.DataFrame, end=None,
                   sp500_then: bool = False) -> tuple[dict, pd.DataFrame]:
    """Factor ranks at every month end (and the last day of data), kept on disk because they take
    about two minutes. The file name has the last date of the prices, so new data means a new file.
    sp500_then: score only the stocks that were in the S&P 500 at each date. Returns (ranks, close)."""
    close = prices.panel(stocks, "close", end=end)
    tag = "sp500_then" if sp500_then else "current"
    path = DATA_DIR / "selection" / f"ranks_{tag}_{close.index[-1]:%Y-%m-%d}_{_inputs_key(stocks, universe)}.pkl"
    if path.exists():
        return pd.read_pickle(path), close
    dates = fx.month_ends(close.index)
    dates = dates[dates >= close.index[0] + pd.Timedelta(days=400)]
    members = sp500_members(universe, dates, close.columns) if sp500_then else None
    ranks = build_factors(close, dates, universe, members=members)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(ranks, path)
    return ranks, close
