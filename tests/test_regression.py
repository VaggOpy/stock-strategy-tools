"""Fixed results on fixed synthetic prices. If a change to the engine, the rules or the walk-forward
moves any of these numbers, either it is a bug or the behaviour changed on purpose. In the second
case update the numbers here in the same commit and say why in the message."""
import numpy as np
import pandas as pd
import pytest

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import metrics
from tradetool.optimize.search import returns_matrix
from tradetool.optimize.walkforward import folds, walk_forward
from tradetool.rules.schema import library

from helpers import random_prices

DF = random_prices(2600, seed=11)

# final equity, number of trades, Sharpe (start after one year, 5 bps)
ENGINE = {
    "donchian_breakout": (0.5699259009, 27, -0.2816559273),
    "ema_cross": (0.9772110827, 23, 0.0999125560),
    "rsi_pullback": (1.0841958361, 65, 0.1345000787),
    "sma_trend": (1.5454731394, 58, 0.3100657802),
}

# ema_cross on a 3x3 grid, out-of-sample growth of 1 for each method
WALK_FORWARD = {"fixed": 1.3926489904, "grid": 1.4958252143, "optuna": 1.5449144274,
                "plateau": 1.4656792605, "topk": 1.2056711369}
PICKS = {"fixed": [[5]] * 6, "grid": [[3], [0], [0], [8], [8], [8]],
         "optuna": [[0], [0], [0], [8], [8], [7]], "plateau": [[0], [0], [0], [0], [8], [8]]}


@pytest.mark.parametrize("name", ENGINE)
def test_engine_results_unchanged(name):
    res = run(DF, library()[name], start=DF.index[260], cost_bps=5)
    m = metrics(res.equity, res.trades, res.position)
    equity, trades, sharpe = ENGINE[name]
    assert res.equity.iloc[-1] == pytest.approx(equity, rel=1e-8)
    assert len(res.trades) == trades
    assert m["sharpe"] == pytest.approx(sharpe, rel=1e-8)


def test_walk_forward_results_unchanged():
    axes = {"fast": np.array([5, 10, 20]), "slow": np.array([30, 50, 100])}
    R, P = returns_matrix(DF, library()["ema_cross"], axes, "2017-01-03")
    rows, oos = walk_forward(R, P, [3, 3], 5, folds(DF.index[-1]), n_trials=8)
    for m, growth in WALK_FORWARD.items():
        assert (1 + oos[m]).prod() == pytest.approx(growth, rel=1e-6), m
    for m, picks in PICKS.items():
        assert [r["cols"] for r in rows if r["method"] == m] == picks, m


def test_selection_results_unchanged(monkeypatch):
    """30 synthetic stocks, momentum + both group factors, top 5 every month."""
    from test_selection import universe_prices

    from tradetool.backtest.portfolio import simulate
    from tradetool.selection import factors as fx
    from tradetool.selection import model

    monkeypatch.setattr(fx, "fundamental_history", lambda t: {})
    open_, close, u = universe_prices()
    dates = fx.month_ends(close.index, close.index[300])
    s = model.score(model.build_factors(close, dates, u, n_clusters=5),
                    {"momentum": 1, "group_sic": 1, "group_cluster": 1})
    res = simulate(model.baskets(s, 5), open_, close, cost_bps=5)
    assert res.equity.iloc[-1] == pytest.approx(1.6811545511, rel=1e-8)
    assert res.turnover.sum() == pytest.approx(30.0549438319, rel=1e-8)
    assert list(model.baskets(s, 5)[dates[-1]].index) == ["S09", "S17", "S27", "S11", "S05"]
