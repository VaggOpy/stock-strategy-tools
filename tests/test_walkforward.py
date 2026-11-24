import numpy as np
import pandas as pd
import pytest

from tradetool.backtest.engine import run
from tradetool.optimize.search import combos, default_index, grid, returns_matrix, sharpe
from tradetool.optimize.walkforward import choose, folds, smooth, walk_forward
from tradetool.rules.schema import library

from helpers import random_prices

RULE = library()["ema_cross"]
AXES = {"fast": np.array([5, 10, 20]), "slow": np.array([30, 50, 100])}
SHAPE = [3, 3]
START = "2017-01-03"


def test_folds_do_not_overlap():
    fl = folds("2026-10-30")
    assert len(fl) == 7
    for f, nxt in zip(fl, fl[1:]):
        assert f.train_end == f.test_start
        assert f.train_start == f.train_end - pd.DateOffset(years=3)
        assert f.test_end == nxt.test_start
    assert fl[-1].test_end == pd.Timestamp("2026-10-31")


def test_matrix_column_is_a_backtest():
    df = random_prices(2000)
    R = returns_matrix(df, RULE, AXES, START)
    values = {"fast": 10, "slow": 50}
    i = combos(AXES).index(values)
    eq = run(df, RULE, values, start=START).equity
    assert (1 + R[i].astype(float)).prod() == pytest.approx(eq.iloc[-1], rel=1e-4)


def test_plateau_prefers_region_over_spike():
    s = np.zeros((7, 7))
    s[0, 6] = 3.0          # lone peak
    s[2:5, 1:4] = 2.0      # broad region
    assert np.argmax(s) == 6
    assert np.unravel_index(int(np.nanargmax(smooth(s.ravel(), s.shape))), s.shape) == (3, 2)


def test_choose_ignores_invalid_cells():
    s = np.array([np.nan, 1.0, 0.5, np.nan, 2.0, 0.1])
    picks = choose(s, [2, 3], default=1, k=2, n_trials=10)
    assert picks["grid"] == [4]
    assert picks["topk"] == [4, 1]
    assert picks["fixed"] == [1]


def test_sharpe_of_cash_is_zero():
    r = np.zeros((300, 2))
    r[:, 1] = np.random.default_rng(0).normal(0.001, 0.01, 300)
    s = sharpe(r)
    assert s[0] == 0 and s[1] > 0


def test_walk_forward_no_lookahead():
    """Change all prices after T. Folds whose training ended before T must choose the same
    parameters, and out-of-sample returns before T must not change."""
    df = random_prices(2600)
    T = pd.Timestamp("2022-06-15")
    other = df.copy()
    after = other.index > T
    other.loc[after] = random_prices(2600, seed=7).loc[after].to_numpy()

    fl = folds(df.index[-1])
    rows_a, oos_a = walk_forward(returns_matrix(df, RULE, AXES, START), SHAPE, 4, fl, n_trials=8)
    rows_b, oos_b = walk_forward(returns_matrix(other, RULE, AXES, START), SHAPE, 4, fl, n_trials=8)
    a, b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    known = a["test_start"] <= T
    assert known.sum() >= 5 * 3  # folds 2020, 2021, 2022 for each method
    assert a.loc[known, "cols"].tolist() == b.loc[known, "cols"].tolist()
    for m in oos_a:
        pd.testing.assert_series_equal(oos_a[m][:T], oos_b[m][:T])
    # and the change after T does reach the later folds
    assert not a.loc[~known, "oos_sharpe"].equals(b.loc[~known, "oos_sharpe"])


def test_default_index():
    axes = grid(RULE, ["fast", "slow"])
    i = default_index(RULE, axes)
    pos = np.unravel_index(i, [len(v) for v in axes.values()])
    assert axes["fast"][pos[0]] == 20 and axes["slow"][pos[1]] == 50
