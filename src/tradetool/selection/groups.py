"""Groups of related stocks and the strength of each group.

Two ways to form the groups:
- sic: the SEC industry code of each company (2 digits, e.g. 36 = electronic equipment). Groups with
  fewer than MIN_SIZE stocks go into the SIC division (first digit) so that a group is never one stock.
- cluster: stocks whose daily returns moved together over the last year, by hierarchical clustering
  on the correlation matrix. Recomputed at every date with the past year only.

Group strength at a date = mean momentum rank of the members. Every stock gets the percentile rank of
its group among all groups.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MIN_SIZE = 5
YEAR = 252


def sic_groups(universe: pd.DataFrame) -> pd.Series:
    """ticker -> group label."""
    u = universe.dropna(subset=["sic"])
    two = (u["sic"] // 100).astype(int)
    size = two.map(two.value_counts())
    label = np.where(size >= MIN_SIZE, "sic" + two.astype(str), "div" + (u["sic"] // 1000).astype(int).astype(str))
    return pd.Series(label, index=u["ticker"].to_numpy())


def cluster_groups(returns: pd.DataFrame, n_groups: int = 25) -> pd.Series:
    """returns: daily returns of the last year (dates x tickers). Stocks with more than 10% missing days
    are left out. Distance = sqrt(2 (1 - correlation)), which is the euclidean distance between the
    standardised return series, so Ward linkage applies. Average linkage gave one cluster with 60% of
    the stocks (the market itself) and a few small ones; Ward gives groups of similar size."""
    r = returns.loc[:, returns.notna().mean() > 0.9]
    if r.shape[1] < n_groups:
        return pd.Series(dtype=object)
    c = r.corr().fillna(0).to_numpy().copy()
    np.fill_diagonal(c, 1)
    d = np.sqrt(np.clip(2 * (1 - c), 0, None))
    z = linkage(squareform(d, checks=False), method="ward")
    labels = fcluster(z, n_groups, criterion="maxclust")
    return pd.Series([f"c{k}" for k in labels], index=r.columns)


def group_rank(member_score: pd.Series, groups: pd.Series, min_size: int = 3) -> pd.Series:
    """member_score: one value per stock at a date (e.g. momentum rank). Returns, per stock, the
    percentile rank of its group's mean score. Stocks in groups smaller than min_size get NaN."""
    df = pd.DataFrame({"score": member_score, "group": groups}).dropna()
    g = df.groupby("group")["score"].agg(["mean", "size"])
    g = g[g["size"] >= min_size]
    strength = g["mean"].rank(pct=True)
    return df["group"].map(strength).reindex(member_score.index)


def group_panel(score: pd.DataFrame, method: str, universe: pd.DataFrame | None = None,
                daily_returns: pd.DataFrame | None = None, n_groups: int = 25) -> pd.DataFrame:
    """Group rank of every stock at every date of score (dates x tickers)."""
    out = {}
    if method == "sic":
        groups = sic_groups(universe)
    for d in score.index:
        if method == "cluster":
            past = daily_returns.loc[:d].iloc[-YEAR:]
            groups = cluster_groups(past[score.columns[score.loc[d].notna()]], n_groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[d] = group_rank(score.loc[d], groups)
    return pd.DataFrame(out).T.reindex(columns=score.columns)
