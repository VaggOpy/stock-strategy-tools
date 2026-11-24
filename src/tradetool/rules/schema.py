"""Strategy rules as JSON, built only from the indicators in the registry and four condition types.

A rule written by hand or by the LLM is checked here before it runs. Example:

    {"name": "EMA cross",
     "params": {"fast": {"default": 20, "low": 5, "high": 50},
                "slow": {"default": 50, "low": 20, "high": 200}},
     "entry": [{"cond": "cross_above", "left": {"ind": "EMA", "length": "$fast"},
                                       "right": {"ind": "EMA", "length": "$slow"}}],
     "exit":  [{"cond": "cross_below", "left": {"ind": "EMA", "length": "$fast"},
                                       "right": {"ind": "EMA", "length": "$slow"}}],
     "stop_atr": 3}

entry: all conditions must hold. exit: any of them is enough.
An operand is an indicator, a price column or a number. "shift": k uses the value of k days ago.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from tradetool.indicators.registry import INDICATORS

CONDITIONS = ["above", "below", "cross_above", "cross_below"]
PRICES = ["open", "high", "low", "close"]
LIBRARY = Path(__file__).parent / "library"
NAME = r"^[a-z_][a-z0-9_]*$"


def schema() -> dict:
    num = {"oneOf": [{"type": "number"}, {"type": "string", "pattern": r"^\$[a-z_][a-z0-9_]*$"}]}
    operand = {"oneOf": [
        {"type": "object", "required": ["ind"],
         "properties": {"ind": {"enum": list(INDICATORS)}, "out": {"type": "string"},
                        "shift": {"type": "integer", "minimum": 0, "maximum": 250}},
         "additionalProperties": num},
        {"type": "object", "required": ["price"], "additionalProperties": False,
         "properties": {"price": {"enum": PRICES}, "shift": {"type": "integer", "minimum": 0, "maximum": 250}}},
        {"type": "object", "required": ["value"], "additionalProperties": False, "properties": {"value": num}},
    ]}
    cond = {"type": "object", "required": ["cond", "left", "right"], "additionalProperties": False,
            "properties": {"cond": {"enum": CONDITIONS}, "left": operand, "right": operand}}
    param = {"type": "object", "required": ["default", "low", "high"], "additionalProperties": False,
             "properties": {"default": {"type": "number"}, "low": {"type": "number"},
                            "high": {"type": "number"}, "step": {"type": "number", "exclusiveMinimum": 0}}}
    return {
        "type": "object", "required": ["entry"], "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "params": {"type": "object", "patternProperties": {NAME: param}, "additionalProperties": False},
            "entry": {"type": "array", "minItems": 1, "items": cond},
            "exit": {"type": "array", "items": cond},
            "stop_atr": num,  # stop at entry price - k * ATR(14) of the signal day
            "max_days": num,  # exit at the open after this many days in the trade
        },
    }


def _operands(rule):
    for c in rule.get("entry", []) + rule.get("exit", []):
        yield c["left"]
        yield c["right"]


def _refs(rule):
    vals = [rule.get("stop_atr"), rule.get("max_days")]
    for op in _operands(rule):
        vals += [v for k, v in op.items() if k not in ("ind", "out", "price")]
    return {v[1:] for v in vals if isinstance(v, str) and v.startswith("$")}


def validate(rule: dict) -> list[str]:
    """Returns a list of problems in plain words (empty if the rule is fine).
    The messages go back to the LLM when it writes a bad rule, so they say what is allowed."""
    errors = [f"{'/'.join(map(str, e.path)) or 'rule'}: {e.message}"
              for e in jsonschema.Draft202012Validator(schema()).iter_errors(rule)]
    if errors:
        return errors

    params = rule.get("params", {})
    for name, p in params.items():
        if not p["low"] <= p["default"] <= p["high"]:
            errors.append(f"param {name}: default {p['default']} is outside [{p['low']}, {p['high']}]")
    for name in sorted(_refs(rule) - set(params)):
        errors.append(f"${name} is used but not defined in params")

    for op in _operands(rule):
        if "ind" not in op:
            continue
        ind = INDICATORS[op["ind"]]
        allowed = {p.name for p in ind.params}
        for k in set(op) - {"ind", "out", "shift"} - allowed:
            errors.append(f"{ind.key} has no parameter '{k}' (allowed: {', '.join(sorted(allowed))})")
        if ind.outputs and op.get("out") not in ind.outputs:
            errors.append(f"{ind.key} needs \"out\": one of {ind.outputs}")
        if not ind.outputs and "out" in op:
            errors.append(f"{ind.key} has a single line, remove \"out\"")
    return errors


def resolve(rule: dict, values: dict | None = None) -> dict:
    """Copy of the rule with every $param replaced by a number (defaults unless given in values)."""
    errors = validate(rule)
    if errors:
        raise ValueError("invalid rule:\n" + "\n".join(errors))
    vals = {k: p["default"] for k, p in rule.get("params", {}).items()}
    vals.update(values or {})

    def fill(v):
        if isinstance(v, str) and v.startswith("$"):
            v = vals[v[1:]]
            return int(v) if float(v).is_integer() else v
        return v

    out = copy.deepcopy(rule)
    for op in _operands(out):
        for k in list(op):
            op[k] = fill(op[k])
    for k in ("stop_atr", "max_days"):
        if k in out:
            out[k] = fill(out[k])
    return out


def library() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(LIBRARY.glob("*.json"))}
