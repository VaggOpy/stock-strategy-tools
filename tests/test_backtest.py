import numpy as np
import pandas as pd
import pytest

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import metrics
from tradetool.rules.schema import library, resolve, validate

from helpers import random_prices

RULES = library()


def flat_then(n_flat, moves, price=100.0):
    """Constant prices, then the given close-to-close moves. open = previous close."""
    close = np.concatenate([np.full(n_flat, price), price * np.cumprod(1 + np.array(moves))])
    open_ = np.concatenate([[price], close[:-1]])
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
                         "close": close, "volume": 1e6}, index=idx)


ABOVE_SMA3 = {"entry": [{"cond": "above", "left": {"price": "close"}, "right": {"ind": "SMA", "length": 3}}],
              "exit": [{"cond": "below", "left": {"price": "close"}, "right": {"ind": "SMA", "length": 3}}]}


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("cut", [400, 900, 1300])
def test_no_lookahead_cut(name, cut):
    """Results up to day T must not change when the data after T is removed or replaced."""
    df = random_prices()
    full = run(df, RULES[name], start=df.index[260])
    cut_df = run(df.iloc[:cut], RULES[name], start=df.index[260])

    other = df.copy()
    other.iloc[cut:] = random_prices(seed=99).iloc[cut:].to_numpy()
    changed = run(other, RULES[name], start=df.index[260])

    T = df.index[cut - 1]
    pd.testing.assert_series_equal(full.equity[:T], cut_df.equity[:T])
    pd.testing.assert_series_equal(full.equity[:T], changed.equity[:T])
    closed = full.trades[full.trades["exit_date"] <= T]
    pd.testing.assert_frame_equal(closed, cut_df.trades.iloc[:len(closed)])


def test_fill_is_next_open():
    # flat, then +5% on day 10: close > SMA(3) first on day 10, so the buy happens at the open of day 11
    df = flat_then(10, [0.05, 0.01, 0.01])
    res = run(df, ABOVE_SMA3, cost_bps=0)
    t = res.trades.iloc[0]
    assert t["entry_date"] == df.index[11]
    assert t["entry_price"] == df["open"].iloc[11]
    assert res.position.iloc[10] == 0 and res.position.iloc[11] == 1


def test_exit_is_next_open_and_costs():
    df = flat_then(10, [0.05, 0.0, -0.10, 0.0, 0.0])
    res = run(df, ABOVE_SMA3, cost_bps=10)
    t = res.trades.iloc[0]
    # exit signal at the close of day 12 (the -10% day), sold at the open of day 13
    assert t["exit_date"] == df.index[13]
    assert t["exit_price"] == df["open"].iloc[13]
    expected = (1 - 0.001) * (1 - 0.001) * t["exit_price"] / t["entry_price"] - 1
    assert t["return"] == pytest.approx(expected)
    assert res.equity.iloc[-1] == pytest.approx(1 + expected)


def test_stop_gap_and_intraday():
    rule = resolve(RULES["ema_cross"], {"fast": 5, "slow": 20, "stop": 2})
    df = random_prices(seed=3)
    res = run(df, rule, cost_bps=0)
    stops = res.trades[res.trades["reason"] == "stop"]
    assert len(stops) > 0
    for t in stops.itertuples():
        day = df.loc[t.exit_date]
        # never better than the open: either a gap (sold at the open) or the low touched the stop
        assert t.exit_price <= day["open"] + 1e-9
        assert t.exit_price >= day["low"] - 1e-9


def test_equity_matches_trades():
    df = random_prices(seed=5)
    for rule in RULES.values():
        res = run(df, rule, start=df.index[260], cost_bps=5)
        assert res.equity.iloc[-1] == pytest.approx((1 + res.trades["return"]).prod())
        assert res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - 1)


def test_no_trade_on_start_day():
    df = random_prices(seed=1)
    res = run(df, RULES["sma_trend"], start=df.index[400])
    assert res.equity.iloc[0] == 1.0
    assert res.trades["entry_date"].min() > df.index[400]


def test_negative_shift_is_rejected():
    bad = {"entry": [{"cond": "above", "left": {"price": "close", "shift": -1}, "right": {"value": 1}}]}
    assert validate(bad)


def test_validation_messages():
    assert validate({"entry": [{"cond": "above", "left": {"ind": "BB", "length": 20}, "right": {"price": "close"}}]})
    assert validate({"entry": [{"cond": "above", "left": {"ind": "EMA", "len": 20}, "right": {"price": "close"}}]})
    assert validate({"entry": [{"cond": "above", "left": {"ind": "EMA", "length": "$n"}, "right": {"value": 1}}]})
    assert validate({"entry": [{"cond": "above", "left": {"ind": "SMA"}, "right": {"value": 1}}], "code": "x"})
    for rule in RULES.values():
        assert validate(rule) == []
    unused = {"params": {"n": {"default": 20, "low": 5, "high": 50}},
              "entry": [{"cond": "above", "left": {"price": "close"}, "right": {"ind": "SMA", "length": 20}}]}
    assert validate(unused) == ["param n is defined but not used anywhere in the rule, remove it"]
    # ranges that a parameter search would reach: a stop of 0 and half days
    zero_stop = {**RULES["ema_cross"], "params": {**RULES["ema_cross"]["params"],
                                                  "stop": {"default": 2, "low": 0, "high": 4}}}
    assert validate(zero_stop) == ["with the params at their low: stop_atr must be greater than 0, got 0"]
    half_days = {**RULES["sma_trend"], "max_days": "$d", "params": {**RULES["sma_trend"]["params"],
                                                                     "d": {"default": 5, "low": 1, "high": 9, "step": 0.5}}}
    assert "max_days takes whole numbers" in validate(half_days)[0]


def test_metrics_known_values():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(1, 1.1, 253), idx)
    eq.iloc[100] = eq.iloc[99] * 0.8
    m = metrics(eq)
    assert m["total_return"] == pytest.approx(0.1)
    assert m["max_drawdown"] == pytest.approx(-0.2)


@pytest.mark.parametrize("op", [{"ind": "SMA", "length": 20.5}, {"ind": "EMA", "length": 0},
                                {"ind": "RSI", "length": -5}, {"ind": "SMA", "length": 5000}])
def test_bad_indicator_values_are_rejected(op):
    rule = {"entry": [{"cond": "above", "left": {"price": "close"}, "right": op}]}
    assert validate(rule)


def test_bad_param_range_and_values():
    rule = {"params": {"n": {"default": 20, "low": 0, "high": 50}},
            "entry": [{"cond": "above", "left": {"price": "close"}, "right": {"ind": "SMA", "length": "$n"}}]}
    assert validate(rule)  # low 0 is not a valid SMA length
    rule["params"]["n"]["low"] = 5
    assert validate(rule) == []
    with pytest.raises(ValueError):
        resolve(rule, {"n": 12.5})
    for bad in ({"stop_atr": -2}, {"max_days": 0}, {"max_days": 2.5}):
        assert validate({**rule, **bad})


def test_fixed_price_levels_are_rejected():
    """'close above 100' on adjusted prices is look-ahead: the old prices already contain later splits."""
    level = {"entry": [{"cond": "above", "left": {"price": "close"}, "right": {"value": 100}}]}
    assert validate(level)
    sma = {"entry": [{"cond": "below", "left": {"value": 50}, "right": {"ind": "SMA", "length": 20}}]}
    assert validate(sma)
    # scale-free comparisons stay allowed
    rsi = {"entry": [{"cond": "below", "left": {"ind": "RSI", "length": 2}, "right": {"value": 10}}]}
    macd = {"entry": [{"cond": "above", "left": {"ind": "MACD", "out": "hist"}, "right": {"value": 0}}]}
    assert validate(rsi) == [] and validate(macd) == []
