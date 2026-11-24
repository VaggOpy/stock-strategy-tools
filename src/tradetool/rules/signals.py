"""Turn a resolved rule into daily entry/exit signals. A signal on day t uses only data up to the
close of day t; the engine acts on it at the next open."""
from __future__ import annotations

import pandas as pd

from tradetool.indicators import core
from tradetool.indicators.registry import INDICATORS


def operand(df: pd.DataFrame, op: dict, cache: dict) -> pd.Series:
    if "value" in op:
        return pd.Series(float(op["value"]), index=df.index)
    if "price" in op:
        s = df[op["price"]]
    else:
        ind = INDICATORS[op["ind"]]
        kw = {p.name: op[p.name] for p in ind.params if p.name in op}
        key = (ind.key, tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = ind.compute(df, **kw)
        out = cache[key]
        s = out[op["out"]] if ind.outputs else out.iloc[:, 0]
    return s.shift(op.get("shift", 0))


def condition(df: pd.DataFrame, c: dict, cache: dict) -> pd.Series:
    left, right = operand(df, c["left"], cache), operand(df, c["right"], cache)
    if c["cond"] == "above":
        return left > right
    if c["cond"] == "below":
        return left < right
    # NaN on the previous day gives False, so there is no "cross" right after the warm-up
    if c["cond"] == "cross_above":
        return (left > right) & (left.shift(1) <= right.shift(1))
    return (left < right) & (left.shift(1) >= right.shift(1))


def signals(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """rule must be resolved (numbers, no $params). Columns: entry, exit, atr."""
    cache: dict = {}
    entry = pd.Series(True, index=df.index)
    for c in rule["entry"]:
        entry &= condition(df, c, cache)
    exit_ = pd.Series(False, index=df.index)
    for c in rule.get("exit", []):
        exit_ |= condition(df, c, cache)
    return pd.DataFrame({"entry": entry, "exit": exit_, "atr": core.atr(df, 14)})
