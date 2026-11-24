import numpy as np
import pandas as pd
import pytest

from tradetool.indicators import core
from tradetool.indicators.registry import INDICATORS


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
    high = close * (1 + rng.uniform(0, 0.02, 600))
    low = close * (1 - rng.uniform(0, 0.02, 600))
    idx = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def assert_no_lookahead(func, df, cut):
    """Value on day t must not change when the days after t are removed."""
    full, part = func(df), func(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


@pytest.mark.parametrize("key", list(INDICATORS))
@pytest.mark.parametrize("cut", [50, 250, 599])
def test_no_lookahead(prices, key, cut):
    assert_no_lookahead(INDICATORS[key].compute, prices, cut)


def test_check_catches_lookahead(prices):
    # a centered moving average uses future days, the check has to fail on it
    bad = lambda df: df[["close"]].rolling(5, center=True).mean()
    with pytest.raises(AssertionError):
        assert_no_lookahead(bad, prices, 250)


def test_sma_matches_mean(prices):
    s = core.sma(prices["close"], 10)
    assert s.iloc[9] == pytest.approx(prices["close"].iloc[:10].mean())
    assert s.iloc[:9].isna().all()


def test_ema_on_constant_is_constant():
    s = pd.Series([5.0] * 50)
    assert core.ema(s, 10).dropna().eq(5.0).all()


def test_rsi_range_and_extremes(prices):
    r = core.rsi(prices["close"]).dropna()
    assert r.between(0, 100).all()
    assert core.rsi(pd.Series(np.arange(1.0, 60.0))).iloc[-1] == pytest.approx(100)


def test_atr_constant_range():
    df = pd.DataFrame({"high": [11.0] * 40, "low": [9.0] * 40, "close": [10.0] * 40})
    assert core.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_macd_hist_is_difference(prices):
    m = core.macd(prices["close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])
