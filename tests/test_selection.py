import numpy as np
import pandas as pd
import pytest

from tradetool.backtest.portfolio import simulate, top_n
from tradetool.data import fundamentals as F
from tradetool.selection import factors as fx
from tradetool.selection import model
from tradetool.selection.groups import group_rank, sic_groups

from helpers import random_prices


def long_facts(seed=0, n=12):
    """Quarterly revenue and EPS with filing delays of 25-60 days (one quarter filed very late)."""
    rng = np.random.default_rng(seed)
    ends = pd.date_range("2020-03-31", periods=n, freq="QE")
    delays = rng.integers(25, 60, n)
    delays[5] = 150  # filed after the next quarter's filing
    rows = []
    for concept in ("revenue", "eps_diluted", "op_cash_flow"):
        vals = 100 * np.cumprod(1 + rng.normal(0.02, 0.05, n)) - (60 if concept == "op_cash_flow" else 0)
        rows += [{"concept": concept, "end": e, "value": v, "filed": e + pd.Timedelta(days=int(d)), "derived": False}
                 for e, v, d in zip(ends, vals, delays)]
    return pd.DataFrame(rows)


def test_history_equals_direct_point_in_time():
    """The per-filing history must give the same value as asking ttm_growth on every single day."""
    f = long_facts()
    h = F.history(f, "revenue", F.growth_of)
    o = F.history(f, "op_cash_flow", F.ttm_of)
    for day in pd.date_range("2020-06-01", "2023-06-30"):
        known = h[h.index <= day]
        got = known.iloc[-1] if len(known) else np.nan
        assert got == pytest.approx(F.ttm_growth(f, "revenue", day), nan_ok=True), day
        known = o[o.index <= day]
        got = known.iloc[-1] if len(known) else np.nan
        assert got == pytest.approx(F.ttm(f, "op_cash_flow", day), nan_ok=True), day


def test_fundamental_panel_is_point_in_time_and_goes_stale(monkeypatch):
    f = long_facts()
    monkeypatch.setattr(fx, "fundamental_history", lambda t: pd.DataFrame({
        "rev_growth": F.history(f, "revenue", F.growth_of),
        "eps_growth": F.history(f, "eps_diluted", F.growth_of),
        "ocf_ttm": F.history(f, "op_cash_flow", F.ttm_of)}))
    dates = pd.date_range("2022-01-31", "2024-12-31", freq="ME")
    p = fx.fundamental_panel(["X"], dates)["rev_growth"]["X"]
    for d in dates:
        direct = F.ttm_growth(f, "revenue", d)
        last_filing = f[f["filed"] < d]["filed"].max()
        if (d - last_filing).days - 1 <= fx.MAX_AGE_DAYS:
            assert p[d] == pytest.approx(direct, nan_ok=True), d
        else:
            assert np.isnan(p[d]), d  # nothing filed for too long


def test_momentum_skips_the_last_month():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.DataFrame({"A": np.arange(1, 401, dtype=float)}, index=idx)
    d = idx[350]
    m = fx.momentum(close, [d])
    i = 350
    assert m["mom_3m"].loc[d, "A"] == pytest.approx(close["A"].iloc[i - 21] / close["A"].iloc[i - 21 * 4] - 1)
    assert m["mom_12m"].loc[d, "A"] == pytest.approx(close["A"].iloc[i - 21] / close["A"].iloc[i - 21 * 13] - 1)
    young = fx.momentum(close, [idx[200]])
    assert np.isnan(young["mom_12m"].loc[idx[200], "A"])  # less than 13 months of prices


def test_group_rank_and_sic_groups():
    score = pd.Series({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1, "e": 0.2, "f": 0.3, "g": 1.0})
    groups = pd.Series({"a": "x", "b": "x", "c": "x", "d": "y", "e": "y", "f": "y", "g": "z"})
    r = group_rank(score, groups)
    assert r["a"] == r["b"] == 1.0 and r["d"] == 0.5
    assert np.isnan(r["g"])  # group z has one member only
    u = pd.DataFrame({"ticker": list("abcdefg"), "sic": [3674, 3672, 3670, 3661, 3600, 2834, 6021]})
    g = sic_groups(u)
    assert g["a"] == "sic36" and g["f"] == "div2" and g["g"] == "div6"


def test_top_n_is_repeatable():
    s = pd.Series({"b": 0.5, "a": 0.5, "c": 0.9, "d": np.nan})
    assert list(top_n(s, 2).index) == ["c", "a"]
    assert top_n(s, 2).sum() == pytest.approx(1.0)


def two_stocks(n=60):
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.DataFrame({"A": np.linspace(100, 130, n), "B": np.linspace(50, 40, n)}, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.001
    return open_, close


def test_portfolio_trades_at_next_open_with_costs():
    open_, close = two_stocks()
    t = close.index[10]
    res = simulate({t: pd.Series({"A": 1.0})}, open_, close, cost_bps=10)
    eq = res.equity
    assert eq.index[0] == t and eq.iloc[0] == 1.0
    shares = (1 - 0.001) / open_["A"].iloc[11]
    assert eq.iloc[1] == pytest.approx(shares * close["A"].iloc[11])
    assert eq.iloc[-1] == pytest.approx(shares * close["A"].iloc[-1])
    assert res.turnover.iloc[0] == pytest.approx(1.0)


def test_portfolio_rebalance_and_drift():
    open_, close = two_stocks()
    t1, t2 = close.index[5], close.index[30]
    res = simulate({t1: pd.Series({"A": 0.5, "B": 0.5}), t2: pd.Series({"B": 1.0})}, open_, close, cost_bps=0)
    h = res.holdings[close.index[31]]
    assert list(h.index) == ["B"]
    # between rebalances the weights drift: equity follows the sum of the two holdings
    i = 20
    sa = 0.5 / open_["A"].iloc[6]
    sb = 0.5 / open_["B"].iloc[6]
    assert res.equity.loc[close.index[i]] == pytest.approx(sa * close["A"].iloc[i] + sb * close["B"].iloc[i])


def universe_prices(n_stocks=30, n_days=900, seed=0):
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    closes, opens = {}, {}
    for i, t in enumerate(tickers):
        p = random_prices(n_days, seed=seed + i)
        closes[t], opens[t] = p["close"], p["open"]
    u = pd.DataFrame({"ticker": tickers, "sic": [2800 + 100 * (i % 6) + i for i in range(n_stocks)], "type": "stock"})
    return pd.DataFrame(opens), pd.DataFrame(closes), u


def run_selection(open_, close, u, monkeypatch):
    monkeypatch.setattr(fx, "fundamental_history", lambda t: pd.DataFrame())  # no fundamentals here
    dates = fx.month_ends(close.index, close.index[300])
    ranks = model.build_factors(close, dates, u, n_clusters=5)
    s = model.score(ranks, {"momentum": 1, "group_sic": 1, "group_cluster": 1})
    return ranks, simulate(model.baskets(s, 5), open_, close, cost_bps=5)


def test_selection_has_no_lookahead(monkeypatch):
    """Change every price after T: scores up to T and the equity up to T must stay the same."""
    open_, close, u = universe_prices()
    T = close.index[600]
    ranks_a, res_a = run_selection(open_, close, u, monkeypatch)
    other_open, other_close, _ = universe_prices(seed=50)
    after = close.index > T
    open_b, close_b = open_.copy(), close.copy()
    open_b.loc[after] = other_open.loc[after].to_numpy()
    close_b.loc[after] = other_close.loc[after].to_numpy()
    ranks_b, res_b = run_selection(open_b, close_b, u, monkeypatch)
    for k in ranks_a:
        if ranks_a[k].notna().any().any():
            pd.testing.assert_frame_equal(ranks_a[k][ranks_a[k].index <= T], ranks_b[k][ranks_b[k].index <= T])
    pd.testing.assert_series_equal(res_a.equity[:T], res_b.equity[:T])
    assert not res_a.equity.equals(res_b.equity)


def test_members_limit_who_is_scored(monkeypatch):
    open_, close, u = universe_prices()
    monkeypatch.setattr(fx, "fundamental_history", lambda t: pd.DataFrame())
    dates = fx.month_ends(close.index, close.index[300])
    u["sp500_added"] = [None] * 10 + [str(dates[5].date())] * 10 + ["1990-01-01"] * 10
    members = model.sp500_members(u, dates, close.columns)
    ranks = model.build_factors(close, dates, u, n_clusters=3, members=members)
    m = ranks["momentum"]
    assert m.iloc[:, :10].isna().all().all()            # never in the index
    assert m.iloc[:5, 10:20].isna().all().all()         # not yet in the index
    assert m.iloc[5:, 10:20].notna().all().all()
    assert m.iloc[0].max() == 1.0 and m.iloc[0].notna().sum() == 10  # ranked among members only
