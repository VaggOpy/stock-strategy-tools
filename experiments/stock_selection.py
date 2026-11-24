"""Monthly stock selection: does a basket of the 20 best-scored stocks beat SPY, and which factor helps?

Every month end the stocks get factor ranks with the data known that day (selection/model.py), the
top 20 are bought with equal weights at the next open and held for a month. Compared with:
- SPY,
- all scored stocks with equal weights (same rebalancing and costs),
- 200 random baskets of 20 stocks from the same stocks every month.
The factor weights are also chosen by walk-forward (3 years train, 1 year test).

Everything runs on two universes:
- current: today's S&P 500 and Nasdaq-100 members over the whole period,
- sp500_then: at each date only the stocks that were already in the S&P 500 on that date.
The first one lets a strategy buy in 2020 a stock that joined the index in 2024 because it had risen,
which flatters anything that buys rising stocks. The second one removes that, but still misses the
stocks that left the index.

Run: python experiments/stock_selection.py   (about 6 minutes)
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from tradetool.backtest.metrics import metrics
from tradetool.backtest.portfolio import simulate, top_n
from tradetool.config import ROOT, TEST_END, TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.optimize.search import sharpe
from tradetool.optimize.walkforward import folds
from tradetool.selection import factors as fx
from tradetool.selection import model

OUT = ROOT / "experiments" / "results" / "stock_selection"
COST_BPS = 5
N = 20
N_RANDOM = 200
MIN_STOCKS = 200  # first month with at least this many scored stocks

VARIANTS = {
    "momentum": {"momentum": 1},
    "fundamentals": {"fundamentals": 1},
    "group_sic": {"group_sic": 1},
    "group_cluster": {"group_cluster": 1},
    "momentum + fundamentals": {"momentum": 1, "fundamentals": 1},
    "all (sic groups)": {"momentum": 1, "fundamentals": 1, "group_sic": 1},
    "all (cluster groups)": {"momentum": 1, "fundamentals": 1, "group_cluster": 1},
}


def summary(eq: pd.Series, spy_r: pd.Series, turnover: pd.Series | None = None) -> dict:
    m = metrics(eq)
    r = eq.pct_change()
    b = spy_r.reindex(r.index)
    m["beta_spy"] = r.cov(b) / b.var()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    m["turnover_per_year"] = turnover.sum() / years if turnover is not None else np.nan
    return m


def weight_grid(step=0.25):
    names = ["momentum", "fundamentals", "group_sic"]
    k = int(round(1 / step))
    return [dict(zip(names, (a * step, b * step, (k - a - b) * step)))
            for a, b in itertools.product(range(k + 1), repeat=2) if a + b <= k]


def run(label, close, open_, spy_close, u, dates, members=None):
    t0 = time.time()
    ranks = model.build_factors(close, dates, u, members=members)
    scored = ranks["momentum"].notna().sum(axis=1)
    dates = dates[scored.to_numpy() >= MIN_STOCKS]
    ranks = {k: v.loc[dates] for k, v in ranks.items()}
    print(f"\n=== {label}: {len(dates)} months ({dates[0].date()} .. {dates[-1].date()}), "
          f"{int(scored.loc[dates].min())}-{int(scored.loc[dates].max())} stocks scored, factors in {time.time() - t0:.0f}s")
    spy_r = spy_close.pct_change()
    curves, rows = {}, {}

    def record(name, res):
        curves[name] = res.equity
        rows[name] = summary(res.equity, spy_r, res.turnover)

    for name, w in VARIANTS.items():
        record(name, simulate(model.baskets(model.score(ranks, w), N), open_, close, COST_BPS))
    for n in (10, 40):
        record(f"all (sic groups), top {n}", simulate(
            model.baskets(model.score(ranks, VARIANTS["all (sic groups)"]), n), open_, close, COST_BPS))
    eligible = ranks["momentum"].notna()
    record("all stocks, equal weight", simulate(
        {d: pd.Series(1 / eligible.loc[d].sum(), index=eligible.columns[eligible.loc[d]]) for d in dates},
        open_, close, COST_BPS))
    first = curves["momentum"].index[0]
    spy_eq = spy_close[spy_close.index >= first] / spy_close[spy_close.index >= first].iloc[0]
    curves["SPY"] = spy_eq
    rows["SPY"] = summary(spy_eq, spy_r)

    rng = np.random.default_rng(0)
    rand = []
    for _ in range(N_RANDOM):
        targets = {d: pd.Series(1 / N, index=rng.choice(eligible.columns[eligible.loc[d]], N, replace=False))
                   for d in dates}
        rand.append(summary(simulate(targets, open_, close, COST_BPS).equity, spy_r))
    rand = pd.DataFrame(rand)
    for name in rows:
        for k in ("cagr", "sharpe"):
            rows[name][f"{k}_pct_of_random"] = (rand[k] < rows[name][k]).mean()

    # factor weights by walk-forward: the best train Sharpe on a grid, used for the next year
    grid = weight_grid()
    R = pd.DataFrame({i: simulate(model.baskets(model.score(ranks, w), N), open_, close, COST_BPS)
                     .equity.pct_change() for i, w in enumerate(grid)})
    chosen, targets = [], {}
    execution = {d: close.index[close.index.searchsorted(d, side="right")] for d in dates if d < close.index[-1]}
    for f in folds(R.index[-1], "2020-01-01"):
        train = R[(R.index >= f.train_start) & (R.index < f.train_end)]
        best = int(np.nanargmax(sharpe(train.fillna(0).to_numpy())))
        chosen.append({"test_year": f.test_start.year, **grid[best]})
        s = model.score(ranks, grid[best])
        targets.update({d: top_n(s.loc[d], N) for d, x in execution.items() if f.test_start <= x < f.test_end})
    wf = simulate(targets, open_, close, COST_BPS)  # starts at the close of the last day of 2019
    base = wf.equity.index[0]
    wf_rows = {}
    for name, e in {"weights by walk-forward": wf.equity, **{k: curves[k] for k in (
            "momentum", "all (sic groups)", "all stocks, equal weight", "SPY")}}.items():
        e = e[e.index >= base]
        wf_rows[name] = summary(e / e.iloc[0], spy_r)

    yearly = pd.DataFrame({k: v.resample("YE").last().pct_change().dropna() for k, v in curves.items()})
    yearly.index = yearly.index.year

    table = pd.DataFrame(rows).T
    cols = ["cagr", "volatility", "sharpe", "max_drawdown", "beta_spy", "turnover_per_year",
            "cagr_pct_of_random", "sharpe_pct_of_random"]
    print(f"random baskets: CAGR median {rand['cagr'].median():.3f} (5%-95%: {rand['cagr'].quantile(.05):.3f}"
          f" .. {rand['cagr'].quantile(.95):.3f}), Sharpe median {rand['sharpe'].median():.2f}")
    print(table[cols].round(3).to_string())
    print("Out of sample 2020 .. 2026, factor weights chosen by walk-forward:")
    print(pd.DataFrame(wf_rows).T[["cagr", "volatility", "sharpe", "max_drawdown"]].round(3).to_string())
    print(pd.DataFrame(chosen).to_string(index=False))
    print("Return per year:")
    print(yearly[["all (sic groups)", "momentum", "fundamentals", "group_sic", "all stocks, equal weight", "SPY"]]
          .round(3).to_string())

    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "summary.csv")
    rand.to_csv(out / "random_baskets.csv", index=False)
    pd.DataFrame(wf_rows).T.to_csv(out / "walkforward_weights.csv")
    (out / "walkforward_choices.json").write_text(json.dumps(chosen, indent=1))
    yearly.to_csv(out / "yearly.csv")
    pd.DataFrame(curves).to_csv(out / "curves.csv")
    # which stocks made the momentum basket's return: the ten biggest contributions
    res = simulate(model.baskets(model.score(ranks, VARIANTS["momentum"]), N), open_, close, COST_BPS)
    held = pd.DataFrame(res.holdings).T.reindex(close.index).ffill().fillna(0)
    contrib = (held.shift(1) * close.pct_change(fill_method=None).fillna(0)).loc[res.equity.index].sum()
    contrib.sort_values(ascending=False).head(15).to_csv(out / "momentum_top_contributors.csv")
    return table


def main():
    t0 = time.time()
    u = load_universe()
    stocks = u.query("type == 'stock'")["ticker"].tolist()
    close = prices.panel(stocks, "close", end=TEST_END)
    open_ = prices.panel(stocks, "open", end=TEST_END)
    spy_close = prices.load("SPY", end=TEST_END)["close"]
    dates = fx.month_ends(close.index, TEST_START, pd.Timestamp(TEST_END) - pd.Timedelta(days=20))

    a = run("current", close, open_, spy_close, u, dates)
    b = run("sp500_then", close, open_, spy_close, u, dates, members=model.sp500_members(u, dates, close.columns))
    both = pd.concat({"current": a, "sp500_then": b})[["cagr", "sharpe", "max_drawdown"]].unstack(0)
    print("\nBoth universes:")
    print(both.round(3).to_string())
    both.to_csv(OUT / "both_universes.csv")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
