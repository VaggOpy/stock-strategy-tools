"""Technical indicators. Every value at day t uses only data up to and including day t
(rolling windows and ewm, never shift(-k) or centered windows). tests/test_indicators.py checks this.

RSI and ATR use Wilder's smoothing (alpha = 1/n), the standard definition.
"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    return close.rolling(length).mean()


def ema(close: pd.Series, length: int = 20) -> pd.Series:
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


def bollinger(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + std * sd, "lower": mid - std * sd})


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"upper": df["high"].rolling(length).max(), "lower": df["low"].rolling(length).min()})


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + up / down)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})
