"""Text -> JSON rule (checked, with retries) and backtest numbers -> explanation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from tradetool.llm import prompts
from tradetool.rules.describe import describe
from tradetool.rules.schema import validate


@dataclass
class Translation:
    rule: dict | None
    assumptions: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    attempts: int = 0       # answers the LLM gave, 1 = right the first time
    problems: list[list[str]] = field(default_factory=list)  # what was wrong in each rejected answer
    ok: bool = False        # False if it was still wrong after the last retry


def parse(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)  # some models wrap JSON in a code block
    data = json.loads(m.group(1) if m else text)
    if not isinstance(data, dict):
        raise ValueError("the answer must be a JSON object")
    return data


def check(data: dict) -> list[str]:
    if "rule" not in data:
        return ['the answer needs a "rule" key']
    if data["rule"] is None:
        return []
    if not isinstance(data["rule"], dict):
        return ['"rule" must be an object or null']
    return validate(data["rule"])


def _strings(x) -> list[str]:
    return [str(s) for s in x] if isinstance(x, list) else []


def to_rule(text: str, client, retries: int = 2) -> Translation:
    system = prompts.translate_system()
    messages = [{"role": "user", "text": f"Description: {text}"}]
    problems = []
    for attempt in range(1, retries + 2):
        answer = client.chat(system, messages, as_json=True)
        try:
            data = parse(answer)
            errors = check(data)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            errors = [f"the answer is not valid JSON: {e}"]
        if not errors:
            return Translation(data["rule"], _strings(data.get("assumptions")), _strings(data.get("unsupported")),
                               attempt, problems, ok=True)
        problems.append(errors)
        messages += [{"role": "assistant", "text": answer}, {"role": "user", "text": prompts.fix_message(errors)}]
    return Translation(None, attempts=retries + 1, problems=problems)


ROWS = {"strategy": "Strategy", "buy & hold": "Buy and hold of the same stock", "SPY": "SPY"}


def facts(table: pd.DataFrame, ticker: str, start, end, cost_bps: float) -> str:
    """The numbers the explanation may use, from compare()."""
    lines = [f"Stock {ticker}, from {pd.Timestamp(start):%d/%m/%Y} to {pd.Timestamp(end):%d/%m/%Y}, "
             f"cost {cost_bps:g} bps per trade side."]
    for key, label in ROWS.items():
        r = table.loc[key]
        lines.append(f"{label}: total return {r['total_return']:.1%}, CAGR {r['cagr']:.1%}, "
                     f"volatility {r['volatility']:.1%}, Sharpe {r['sharpe']:.2f}, max drawdown {r['max_drawdown']:.1%}")
    s = table.loc["strategy"]
    if s["trades"] > 0:
        lines.append(f"Strategy trades: {s['trades']:.0f}, winning {s['win_rate']:.0%}, average trade "
                     f"{s['avg_trade']:.1%}, average {s['avg_days']:.0f} days per trade, money in the stock "
                     f"{s['exposure']:.0%} of the days.")
    else:
        lines.append("The strategy made no trades in this period.")
    return "\n".join(lines)


def explain(client, text: str, rule: dict, fact_text: str) -> str:
    message = prompts.explain_message(text, describe(rule), fact_text)
    return client.chat(prompts.EXPLAIN, [{"role": "user", "text": message}]).strip()


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def unknown_numbers(explanation: str, *sources: str) -> list[str]:
    """Numbers in the explanation that are not in the sources, even after rounding.
    A rough check for made-up numbers: 12% is fine if the facts say 12.3%, 15% is not."""
    known = [float(x.replace(",", ".")) for s in sources for x in NUMBER.findall(s)]
    out = []
    for x in NUMBER.findall(explanation):
        decimals = len(re.split(r"[.,]", x)[1]) if re.search(r"[.,]", x) else 0
        v = float(x.replace(",", "."))
        if not any(round(k, decimals) == v for k in known):
            out.append(x)
    return out
