"""The four rules of the library with their default parameters, on every stock of the universe that
has data for the whole test period. No optimisation here, this is the baseline.

Per stock: strategy vs buy & hold of the same stock. Overall: an equal-weight basket of the
per-stock strategies (each stock gets the same capital, in the stock or in cash) vs an
equal-weight basket of the same stocks held all the time, and vs SPY.

Run: python experiments/classic_strategies.py
"""
import time

import pandas as pd

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import buy_and_hold, metrics
from tradetool.config import ROOT, TEST_END, TEST_START
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.rules.schema import library

OUT = ROOT / "experiments" / "results"
COST_BPS = 5


def main():
    stocks = load_universe().query("type == 'stock'")["ticker"].tolist()
    data = {}
    for t in stocks:
        df = prices.load(t, end=TEST_END)
        if df.index[0] <= pd.Timestamp(TEST_START) - pd.Timedelta(days=300):
            data[t] = df
    print(f"{len(data)} of {len(stocks)} stocks have data for the whole period {TEST_START} .. {TEST_END}")

    spy = prices.load("SPY", end=TEST_END)
    idx = spy.index[spy.index >= TEST_START]
    spy_eq = buy_and_hold(spy, idx)

    rows, curves = [], {}
    t0 = time.time()
    for name, rule in library().items():
        rets = {}
        for t, df in data.items():
            res = run(df, rule, start=TEST_START, cost_bps=COST_BPS)
            m = metrics(res.equity, res.trades, res.position)
            bh = metrics(buy_and_hold(df, res.equity.index))
            rows.append({"rule": name, "ticker": t, **m, "bh_cagr": bh["cagr"], "bh_sharpe": bh["sharpe"],
                         "bh_max_drawdown": bh["max_drawdown"]})
            rets[t] = res.equity.pct_change()
        curves[name] = (1 + pd.DataFrame(rets).reindex(idx).mean(axis=1).fillna(0)).cumprod()
    print(f"{len(rows)} backtests in {time.time() - t0:.0f}s")

    per_stock = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    per_stock.to_csv(OUT / "classic_per_stock.csv", index=False)

    s = per_stock.groupby("rule").agg(
        median_cagr=("cagr", "median"), median_bh_cagr=("bh_cagr", "median"),
        median_sharpe=("sharpe", "median"), median_bh_sharpe=("bh_sharpe", "median"),
        median_maxdd=("max_drawdown", "median"), median_bh_maxdd=("bh_max_drawdown", "median"),
        exposure=("exposure", "mean"), trades_per_stock=("trades", "mean"), win_rate=("win_rate", "mean"))
    beats = per_stock.assign(cagr_win=per_stock["cagr"] > per_stock["bh_cagr"],
                             sharpe_win=per_stock["sharpe"] > per_stock["bh_sharpe"],
                             dd_win=per_stock["max_drawdown"] > per_stock["bh_max_drawdown"])
    s[["beats_bh_cagr", "beats_bh_sharpe", "smaller_dd"]] = beats.groupby("rule")[["cagr_win", "sharpe_win", "dd_win"]].mean()
    print("\nPer stock (medians, fractions of stocks):")
    print(s.round(3).T.to_string())

    bh_basket = (1 + pd.DataFrame({t: df["close"].pct_change() for t, df in data.items()})
                 .reindex(idx).mean(axis=1).fillna(0)).cumprod()
    baskets = {**{f"basket: {k}": v for k, v in curves.items()}, "basket: buy & hold": bh_basket, "SPY": spy_eq}
    b = pd.DataFrame({k: metrics(v) for k, v in baskets.items()}).T
    b["beta_spy"] = [v.pct_change().cov(spy_eq.pct_change()) / spy_eq.pct_change().var() for v in baskets.values()]
    print("\nEqual-weight baskets vs SPY:")
    print(b[["cagr", "volatility", "sharpe", "max_drawdown", "calmar", "beta_spy"]].round(3).to_string())

    s.to_csv(OUT / "classic_summary.csv")
    b.to_csv(OUT / "classic_baskets.csv")
    pd.DataFrame(baskets).to_csv(OUT / "classic_basket_curves.csv")


if __name__ == "__main__":
    main()
