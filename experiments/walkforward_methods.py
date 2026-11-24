"""Which way of choosing parameters holds up out of sample?

Rolling walk-forward (3 years train, 1 year test, tests 2020 .. Oct 2026) for the four rules of the
library, on every stock with data for the whole period. Two parameters are searched per rule (one for
sma_trend), the rest stay at their defaults. Methods: see tradetool/optimize/walkforward.py, plus
"pooled": one combination for all stocks, the best on average over all stocks in the train window.

Run: python experiments/walkforward_methods.py   (about 20 minutes on 14 processes)
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tradetool.backtest.metrics import metrics
from tradetool.config import ROOT, TEST_END, TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.optimize.search import default_index, grid, returns_matrix, sharpe
from tradetool.optimize.walkforward import folds, oos_returns, walk_forward, window
from tradetool.rules.schema import library

OUT = ROOT / "experiments" / "results"
FIRST_TEST = "2020-01-01"
COST_BPS = 5


def fast_below_slow(v):
    return v["fast"] < v["slow"]


SEARCH = {
    "ema_cross": (["fast", "slow"], fast_below_slow),
    "sma_trend": (["length"], None),
    "rsi_pullback": (["level", "exit_len"], None),
    "donchian_breakout": (["entry_len", "exit_len"], None),
}


def setup(name):
    rule = library()[name]
    axes = grid(rule, SEARCH[name][0])
    return rule, axes, [len(v) for v in axes.values()], default_index(rule, axes)


def one_stock(args):
    name, ticker = args
    rule, axes, shape, default = setup(name)
    df = prices.load(ticker, end=TEST_END)
    R, P = returns_matrix(df, rule, axes, TEST_START, valid=SEARCH[name][1])
    fl = folds(R.index[-1], FIRST_TEST)
    rows, oos = walk_forward(R, P, shape, default, fl, cost_bps=COST_BPS)
    for r in rows:
        r.update(rule=name, ticker=ticker)
    s_in = np.array([sharpe(window(R, f.train_start, f.train_end)) for f in fl])
    # test windows plus a few days before, the pooled switch needs the position on the last train day
    keep = R.index >= pd.Timestamp(FIRST_TEST) - pd.Timedelta(days=10)
    return name, ticker, rows, oos, s_in, R[keep], P[keep]


def tickers():
    out = []
    for t in load_universe().query("type == 'stock'")["ticker"]:
        df = prices.load(t, end=TEST_END)
        if df.index[0] <= pd.Timestamp(TEST_START) - pd.Timedelta(days=300):
            out.append(t)
    return out


def basket(returns: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame(returns).mean(axis=1).fillna(0)


def main():
    stocks = tickers()
    spy = prices.load("SPY", end=TEST_END)["close"]
    spy_r = spy.pct_change()[spy.index >= FIRST_TEST]
    bh_r = basket({t: prices.load(t, end=TEST_END)["close"].pct_change() for t in stocks})
    bh_r = bh_r[bh_r.index >= FIRST_TEST]
    print(f"{len(stocks)} stocks, tests {FIRST_TEST} .. {TEST_END}")

    all_rows, curves, pooled_choices = [], {}, {}
    for name in SEARCH:
        rule, axes, shape, default = setup(name)
        t0 = time.time()
        with Pool(14) as pool:
            results = pool.map(one_stock, [(name, t) for t in stocks], chunksize=4)
        print(f"{name}: {np.prod(shape)} combinations x {len(stocks)} stocks in {time.time() - t0:.0f}s")

        # pooled: one choice per fold for all stocks, from the average train Sharpe
        fl = folds(results[0][5].index[-1], FIRST_TEST)
        mean_in = np.nanmean(np.stack([r[4] for r in results]), axis=0)
        pooled_cols = [int(np.nanargmax(m)) for m in mean_in]
        oos_by_method: dict[str, dict] = {}
        for _, ticker, rows, oos, s_in, R, P in results:
            all_rows += rows
            for m, r in oos.items():
                oos_by_method.setdefault(m, {})[ticker] = r
            parts, prev = [], None
            for n, f in enumerate(fl):
                r = oos_returns(R, P, f, [pooled_cols[n]], prev, COST_BPS)
                prev = [pooled_cols[n]]
                fixed_row = next(x for x in rows if x["fold"] == n and x["method"] == "fixed")
                all_rows.append({**fixed_row, "method": "pooled", "cols": [pooled_cols[n]],
                                 "is_sharpe": float(s_in[n, pooled_cols[n]]),
                                 "oos_sharpe": float(sharpe(r.to_numpy())), "oos_return": float((1 + r).prod() - 1)})
                parts.append(r)
            oos_by_method.setdefault("pooled", {})[ticker] = pd.concat(parts)
        for m, d in oos_by_method.items():
            curves[(name, m)] = basket(d)
        chosen = [{k: float(v[p]) for (k, v), p in zip(axes.items(), np.unravel_index(c, shape))}
                  for c in pooled_cols]
        pooled_choices[name] = chosen
        print("  pooled choice per fold:", chosen)

    rows = pd.DataFrame(all_rows)
    OUT.mkdir(exist_ok=True)
    rows.drop(columns="cols").assign(cols=rows["cols"].astype(str)).to_csv(OUT / "wf_rows.csv", index=False)

    # per stock and fold
    fixed = rows[rows["method"] == "fixed"].set_index(["rule", "ticker", "fold"])["oos_sharpe"]
    rows["beats_fixed"] = rows["oos_sharpe"].to_numpy() > fixed.reindex(
        pd.MultiIndex.from_frame(rows[["rule", "ticker", "fold"]])).to_numpy()
    rows["vs_median"] = rows["oos_sharpe"] - rows["oos_median_combo"]
    per = rows.groupby(["rule", "method"]).agg(is_sharpe=("is_sharpe", "mean"), oos_sharpe=("oos_sharpe", "mean"),
                                               beats_fixed=("beats_fixed", "mean"), vs_median=("vs_median", "mean"))
    per["decay"] = per["is_sharpe"] - per["oos_sharpe"]
    print("\nPer stock and fold (means):")
    print(per.round(3).to_string())

    rc = rows[rows["method"] == "fixed"].groupby(["rule", "test_start"])["rank_corr"].mean().unstack()
    print("\nSpearman correlation of in-sample and out-of-sample Sharpe over all combinations (mean over stocks):")
    print(rc.round(2).to_string())

    # baskets: equal weight over stocks, test windows joined
    b = {f"{rule} / {m}": metrics((1 + r).cumprod()) for (rule, m), r in curves.items()}
    b["buy & hold, same stocks"] = metrics((1 + bh_r.fillna(0)).cumprod())
    b["SPY"] = metrics((1 + spy_r.fillna(0)).cumprod())
    b = pd.DataFrame(b).T[["cagr", "volatility", "sharpe", "max_drawdown", "calmar"]]
    print("\nEqual-weight baskets, out of sample only:")
    print(b.round(3).to_string())

    yearly = pd.DataFrame({k: r.groupby(r.index.year).apply(lambda x: sharpe(x.to_numpy()))
                           for k, r in curves.items()}).T
    yearly.index = pd.MultiIndex.from_tuples(yearly.index, names=["rule", "method"])
    print("\nBasket Sharpe per test year:")
    print(yearly.astype(float).round(2).to_string())

    pd.Series(pooled_choices).to_json(OUT / "wf_pooled_choices.json", indent=1)
    per.to_csv(OUT / "wf_per_stock_summary.csv")
    rc.to_csv(OUT / "wf_rank_corr.csv")
    b.to_csv(OUT / "wf_baskets.csv")
    yearly.to_csv(OUT / "wf_baskets_yearly.csv")
    pd.DataFrame({f"{k[0]} / {k[1]}": (1 + v).cumprod() for k, v in curves.items()}).to_csv(OUT / "wf_basket_curves.csv")


if __name__ == "__main__":
    main()
