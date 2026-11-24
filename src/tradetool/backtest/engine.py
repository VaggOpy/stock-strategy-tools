"""Backtest of one rule on one stock: long only, all capital in or out.

Timing (the reason this is a plain loop and not vectorised):
- signals are computed at the close of day t,
- the order is filled at the OPEN of day t+1, never at the close of the signal day,
- a stop is placed at fill price - k * ATR(14) of day t, so it is known when the trade opens.
  If day t+n opens below the stop we sell at the open (gap), if only the low touches it we sell
  at the stop price. The stop is also checked on the entry day itself.
Costs are paid on every fill, cost_bps per side (commission + spread + slippage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradetool.rules.schema import resolve
from tradetool.rules.signals import signals


@dataclass
class Result:
    equity: pd.Series      # account value at each close, starts at 1.0
    position: pd.Series    # 1 if holding the stock at the close, else 0
    trades: pd.DataFrame
    rule: dict             # resolved rule that was run


def run(df: pd.DataFrame, rule: dict, values: dict | None = None, start=None, end=None,
        cost_bps: float = 5.0, cache: dict | None = None) -> Result:
    """df: daily OHLC with the full history, so indicators are warmed up before `start`.
    The first order can be filled on the day after `start`. cache: see signals()."""
    rule = resolve(rule, values)
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    sig = signals(df, rule, cache)

    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    entry, exit_, atr = sig["entry"].to_numpy(), sig["exit"].to_numpy(), sig["atr"].to_numpy()
    stop_k, max_days = rule.get("stop_atr"), rule.get("max_days")
    cost = cost_bps / 1e4

    s = 0 if start is None else int(df.index.searchsorted(pd.Timestamp(start)))
    n = len(df)
    equity, position = np.full(n, np.nan), np.zeros(n)
    cash, shares, stop = 1.0, 0.0, np.nan
    trades, cur = [], None

    def sell(i, price, why):
        nonlocal cash, shares
        cash = shares * price * (1 - cost)
        cur.update(exit_date=df.index[i], exit_price=price, reason=why, pnl=cash - cur["cash_in"])
        cur["return"] = cash / cur.pop("cash_in") - 1
        trades.append(cur)
        shares = 0.0

    for i in range(s, n):
        held = shares > 0
        if held:
            if exit_[i - 1]:
                sell(i, o[i], "signal")
            elif max_days and i - cur["i"] >= max_days:
                sell(i, o[i], "time")
            elif o[i] <= stop:
                sell(i, o[i], "stop")

        if not held and i > s and entry[i - 1]:
            stop = o[i] - stop_k * atr[i - 1] if stop_k else -np.inf
            if not np.isnan(stop):
                shares = cash * (1 - cost) / o[i]
                cur = {"entry_date": df.index[i], "entry_price": o[i], "i": i, "cash_in": cash}
                cash = 0.0

        if shares > 0 and l[i] <= stop:
            sell(i, min(stop, o[i]), "stop")

        if shares > 0:
            position[i] = 1
        equity[i] = cash + shares * c[i]

    if shares > 0:  # still open at the end: valued at the last close, no exit cost
        cur.update(exit_date=pd.NaT, exit_price=c[-1], reason="open", pnl=shares * c[-1] - cur["cash_in"])
        cur["return"] = shares * c[-1] / cur.pop("cash_in") - 1
        trades.append(cur)

    t = pd.DataFrame(trades, columns=["entry_date", "entry_price", "exit_date", "exit_price", "reason", "return",
                                      "pnl", "i"])
    end_i = [n - 1 if pd.isna(d) else df.index.get_loc(d) for d in t["exit_date"]]
    t["days"] = np.array(end_i, dtype=int) - t["i"].to_numpy(dtype=int)
    t = t.drop(columns="i").astype({"entry_date": df.index.dtype, "exit_date": df.index.dtype,
                                    "entry_price": float, "exit_price": float, "return": float, "pnl": float,
                                    "reason": "str"})
    idx = df.index[s:]
    return Result(pd.Series(equity[s:], idx, name="equity"), pd.Series(position[s:], idx, name="position"), t, rule)
