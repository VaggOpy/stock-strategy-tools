"""The texts sent to the LLM. The indicator list and the examples are built from the registry and the
rule library, so a new indicator or a new library rule reaches the prompt without editing it here."""
from __future__ import annotations

import json

from tradetool.indicators.registry import INDICATORS
from tradetool.rules.schema import CONDITIONS, PRICES, library, schema


def indicator_list() -> str:
    lines = []
    for ind in INDICATORS.values():
        args = ", ".join(f"{p.name}: {p.low:g}..{p.high:g} (usual {p.default:g})" for p in ind.params)
        line = f"- {ind.key} ({ind.label}): {args}"
        if ind.outputs:
            line += f'; several lines, choose one with "out": {", ".join(ind.outputs)}'
        if ind.scale_free:
            line += "; value 0-100, can be compared with fixed levels"
        lines.append(line)
    return "\n".join(lines)


def _example(rule: dict) -> str:
    answer = {"rule": {k: v for k, v in rule.items() if k != "description"}, "assumptions": [], "unsupported": []}
    return f"Description: {rule['description']}\nAnswer: {json.dumps(answer)}"


TRANSLATE = """You turn a description of a stock trading strategy into a JSON rule for a backtest engine.

How the engine works:
- one stock, daily bars, long only, all capital in the stock or all in cash;
- conditions are checked at the close of each day, orders are filled at the next day's open;
- "entry": a list of conditions that must ALL hold to buy;
- "exit": a list of conditions, ANY one of them is enough to sell;
- "stop_atr": k sells if the price falls k x ATR(14) below the entry price (optional);
- "max_days": n sells after n days in the trade (optional).

A condition is {{"cond": one of {conds}, "left": operand, "right": operand}}.
above / below compare the two operands on the same day. cross_above means left was not above right
the day before and is above it today; cross_below is the opposite.

An operand is one of:
- an indicator: {{"ind": "SMA", "length": 50}}, with its parameters as keys;
- a price: {{"price": "close"}}, one of {prices} (volume is not available);
- a number: {{"value": 30}}.
Any indicator or price can take "shift": k to use its value from k days earlier.

Indicators (parameter ranges are hard limits):
{indicators}

Important rules:
- Prices are adjusted for later splits and dividends, so an old price is rescaled after the fact.
  A price or a price-based indicator (SMA, EMA, BB, DC, ATR, MACD lines) must never be compared with a
  fixed number other than 0, because that level would come from the future. Only RSI can be compared
  with fixed levels. "Close above 100" or "price under $50" cannot be done: say so in "unsupported".
- A breakout above the highest high of the last n days is close above DC(n) upper with "shift": 1,
  so today's own high is not part of the channel.
- Every number that a user might want to tune goes into "params" with a sensible search range
  ("default", "low", "high", "step") and is used as "$name". The default is the number in the
  description. If the description gives no number, use the usual value and add an assumption.
- Fill "assumptions" with every choice the description did not make for you, in the language of the
  description. Fill "unsupported" with every part of the description the engine cannot do (news,
  fundamentals or earnings, volume, short selling, several stocks at once, position sizing, intraday
  timing, fixed price levels). Leave out the unsupported part and build the rest. If nothing useful
  is left, set "rule" to null.
- Give the rule a short "name".

Answer with JSON only, in this format:
{{"rule": {{...}} or null, "assumptions": ["..."], "unsupported": ["..."]}}

The full JSON schema of "rule":
{schema}

Examples:
{examples}
"""


def translate_system() -> str:
    return TRANSLATE.format(conds=", ".join(CONDITIONS), prices=", ".join(PRICES), indicators=indicator_list(),
                            schema=json.dumps(schema()), examples="\n\n".join(_example(r) for r in library().values()))


def fix_message(problems: list[str]) -> str:
    return ("The rule cannot run, it has these problems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\nAnswer again with the corrected JSON in the same format.")


EXPLAIN = """You explain the result of a strategy backtest to the user of a stock analysis tool.
Write in Greek, plain text, two or three short paragraphs, no headings and no lists.

- Use only the numbers given to you. Do not compute or estimate new numbers.
- Compare the strategy with buying and holding the same stock and with SPY (the S&P 500), on return
  and on risk (drawdown, volatility).
- Say how many trades there were and how much of the time the money was in the stock, and what this
  means for the result.
- End with what the result does not show: it is one stock and one past period, the parameters were not
  chosen out of sample, and a good past result is no proof for the future.
- No investment advice."""


def explain_message(text: str, rule_lines: list[str], facts: str) -> str:
    return (f"The user described the strategy like this:\n{text}\n\nWhat was run:\n"
            + "\n".join(f"- {line}" for line in rule_lines) + f"\n\nResults:\n{facts}")
