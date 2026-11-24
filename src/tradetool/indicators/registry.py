"""The list of indicators the app (and later the strategy rules) can use.

For each one: its parameters with default and allowed range, and whether it is drawn on the
price chart ("overlay") or in its own panel below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from tradetool.indicators import core


@dataclass
class Param:
    name: str
    default: float
    low: float
    high: float
    step: float = 1


@dataclass
class Indicator:
    key: str
    label: str
    func: Callable
    params: list[Param]
    overlay: bool
    uses: str = "close"  # "close" or "ohlc"
    levels: list[float] = field(default_factory=list)  # reference lines, e.g. RSI 30/70
    outputs: list[str] = field(default_factory=list)  # column names when there is more than one line
    scale_free: bool = False  # True if the value does not depend on the price level (RSI 0-100)

    def compute(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        args = {p.name: kw.get(p.name, p.default) for p in self.params}
        args = {k: int(v) if float(v).is_integer() else v for k, v in args.items()}
        out = self.func(df["close"] if self.uses == "close" else df, **args)
        if isinstance(out, pd.Series):
            out = out.to_frame(self.key)
        return out

    def name_with(self, **kw) -> str:
        vals = [kw.get(p.name, p.default) for p in self.params]
        return f"{self.key}({', '.join(f'{v:g}' for v in vals)})"


INDICATORS = {i.key: i for i in [
    Indicator("SMA", "Simple moving average", core.sma, [Param("length", 50, 2, 400)], overlay=True),
    Indicator("EMA", "Exponential moving average", core.ema, [Param("length", 21, 2, 400)], overlay=True),
    Indicator("BB", "Bollinger bands", core.bollinger,
              [Param("length", 20, 5, 200), Param("std", 2.0, 0.5, 4.0, 0.1)], overlay=True,
              outputs=["mid", "upper", "lower"]),
    Indicator("DC", "Donchian channel", core.donchian, [Param("length", 20, 5, 250)], overlay=True, uses="ohlc",
              outputs=["upper", "lower"]),
    Indicator("RSI", "Relative strength index", core.rsi, [Param("length", 14, 2, 100)], overlay=False,
              levels=[30, 70], scale_free=True),
    Indicator("MACD", "MACD", core.macd,
              [Param("fast", 12, 2, 50), Param("slow", 26, 5, 200), Param("signal", 9, 2, 50)], overlay=False,
              outputs=["macd", "signal", "hist"]),
    Indicator("ATR", "Average true range", core.atr, [Param("length", 14, 2, 100)], overlay=False, uses="ohlc"),
]}
