"""Compare the adjusted daily returns of Alpaca and Yahoo for every symbol.

On a normal day both sources should give the same return. They differ on days with a corporate action
(split, dividend, spin-off) that one of them adjusts differently. A difference of a few percent on a
single day is enough to fake a crash or a rally in a backtest.

Run: python experiments/check_adjustments.py
"""
import time

import numpy as np
import pandas as pd

from tradetool.config import ROOT
from tradetool.data import prices
from tradetool.data.universe import load_universe

OUT = ROOT / "experiments" / "results"
LIMIT = 0.02  # daily returns differing by more than 2 points


def compare(ticker):
    a = prices.load(ticker)
    y = prices.fetch_yahoo(ticker)
    if y.empty:
        return None, []
    common = a.index.intersection(y.index)
    ra = a["close"].reindex(common).pct_change()
    ry = y["close"].reindex(common).pct_change()
    raw = a["close_raw"].reindex(common).pct_change()
    diff = (ra - ry).abs()
    days = [{"ticker": ticker, "date": d.date(), "raw": raw[d], "alpaca": ra[d], "yahoo": ry[d]}
            for d in diff.index[diff > LIMIT]]
    return {"ticker": ticker, "days": len(common), "bad_days": len(days),
            "corr": ra.corr(ry), "max_diff": diff.max()}, days


def main():
    t0 = time.time()
    summary, days = [], []
    for t in load_universe()["ticker"]:
        s, d = compare(t)
        if s:
            summary.append(s)
            days += d
    s = pd.DataFrame(summary)
    d = pd.DataFrame(days)
    OUT.mkdir(exist_ok=True)
    s.to_csv(OUT / "adjustment_check_summary.csv", index=False)
    d.to_csv(OUT / "adjustment_check_days.csv", index=False)
    print(f"{len(s)} symbols compared in {time.time() - t0:.0f}s")
    print(f"{(s['bad_days'] > 0).sum()} symbols with at least one day differing by more than {LIMIT:.0%}")
    print(f"{len(d)} such days in total, out of {s['days'].sum()}")
    if len(d):
        print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
