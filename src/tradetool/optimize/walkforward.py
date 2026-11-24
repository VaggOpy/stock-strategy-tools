"""Walk-forward comparison of ways to choose parameters.

For every fold the methods see only the Sharpe of each combination in the train window. The chosen
combination(s) are then judged on the test window that follows.

Methods:
  fixed    the rule's default (classic) values, no search
  grid     best combination in the train window
  optuna   TPE search over the same grid with a small budget
  plateau  best combination after averaging every cell with its neighbours, i.e. prefer a region
           of good values over a single peak
  topk     equal-weight mix of the k best combinations
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr

from tradetool.optimize.search import sharpe

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive, = test_start
    test_end: pd.Timestamp    # exclusive

    @property
    def test_start(self):
        return self.train_end


def folds(last_day, first_test="2020-01-01", train_years=3, test_years=1) -> list[Fold]:
    out, t = [], pd.Timestamp(first_test)
    last = pd.Timestamp(last_day)
    while t <= last:
        out.append(Fold(t - pd.DateOffset(years=train_years), t,
                        min(t + pd.DateOffset(years=test_years), last + pd.Timedelta(days=1))))
        t += pd.DateOffset(years=test_years)
    return out


def window(R: pd.DataFrame, a, b) -> pd.DataFrame:
    return R[(R.index >= a) & (R.index < b)]


def smooth(score: np.ndarray, shape) -> np.ndarray:
    """Mean of each cell and its neighbours (+-1 step in every parameter), NaN cells ignored."""
    s = score.reshape(shape)
    pad = np.pad(s, 1, constant_values=np.nan)
    stack = []
    for offset in np.ndindex(*(3,) * s.ndim):
        sl = tuple(slice(o, o + n) for o, n in zip(offset, s.shape))
        stack.append(pad[sl])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        m = np.nanmean(np.stack(stack), axis=0)
    return np.where(np.isnan(s), np.nan, m).ravel()


def pick_optuna(score: np.ndarray, shape, n_trials=30, seed=0) -> int:
    def objective(trial):
        pos = [trial.suggest_int(f"p{d}", 0, n - 1) for d, n in enumerate(shape)]
        v = score[np.ravel_multi_index(pos, shape)]
        return -10.0 if np.isnan(v) else float(v)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    pos = [study.best_params[f"p{d}"] for d in range(len(shape))]
    return int(np.ravel_multi_index(pos, shape))


def choose(score: np.ndarray, shape, default: int, k=5, n_trials=30, seed=0) -> dict[str, list[int]]:
    """Column(s) each method picks, from the train-window Sharpe of every combination."""
    order = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
    return {
        "fixed": [default],
        "grid": [int(order[0])],
        "optuna": [pick_optuna(score, shape, n_trials, seed)],
        "plateau": [int(np.nanargmax(smooth(score, shape)))],
        "topk": [int(i) for i in order[:k]],
    }


def oos_returns(R: pd.DataFrame, P: pd.DataFrame, f: Fold, cols: list[int], prev: list[int] | None,
                 cost_bps: float) -> pd.Series:
    """Daily returns of the chosen column(s) in the test window of fold f.

    If the choice is the same as in the previous fold, the position simply continues. Otherwise the
    old position is sold at the last close before the window (an order that can be placed in advance)
    and the new one is bought at the close of the first test day, when its state is known. The first
    day then earns nothing and pays the costs of both trades. Taking the new rule's return for that
    day instead would mean holding its position since a close where it had not been chosen yet."""
    r = window(R, f.test_start, f.test_end)[cols].mean(axis=1).astype(float)
    if prev is not None and sorted(prev) == sorted(cols):
        return r
    before = P[P.index < f.test_start]
    old = float(before[prev].iloc[-1].mean()) if prev is not None and len(before) else 0.0
    new = float(window(P, f.test_start, f.test_end)[cols].iloc[0].mean())
    r.iloc[0] = -(old + new) * cost_bps / 1e4
    return r


def walk_forward(R: pd.DataFrame, P: pd.DataFrame, shape, default: int, fold_list: list[Fold],
                 cost_bps: float = 5.0, **kw):
    """R, P from search.returns_matrix. Returns (rows, oos): one row per fold and method, and the
    test-window daily returns of each method joined over all folds."""
    rows, oos, prev = [], {}, {}
    for n, f in enumerate(fold_list):
        train, test = window(R, f.train_start, f.train_end), window(R, f.test_start, f.test_end)
        if len(train) < 200 or len(test) < 20:
            continue
        s_in, s_out = sharpe(train), sharpe(test)
        ok = ~np.isnan(s_in) & ~np.isnan(s_out)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant input, e.g. a stock where nothing traded
            rho = spearmanr(s_in[ok], s_out[ok]).statistic if ok.sum() > 2 else np.nan
        for method, cols in choose(s_in, shape, default, seed=n, **kw).items():
            r = oos_returns(R, P, f, cols, prev.get(method), cost_bps)
            prev[method] = cols
            oos.setdefault(method, []).append(r)
            rows.append({"fold": n, "test_start": f.test_start, "method": method, "cols": cols,
                         "is_sharpe": float(np.nanmean(s_in[cols])), "oos_sharpe": float(sharpe(r.to_numpy())),
                         "oos_return": float((1 + r).prod() - 1),
                         "rank_corr": rho, "oos_median_combo": float(np.nanmedian(s_out)),
                         "oos_best_combo": float(np.nanmax(s_out))})
    return rows, {m: pd.concat(v) for m, v in oos.items()}
