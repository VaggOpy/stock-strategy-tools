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
    # an unused param would be a slider that changes nothing and a useless axis for the optimizer
    for name in sorted(set(params) - _refs(rule)):
        errors.append(f"param {name} is defined but not used anywhere in the rule, remove it")

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
        # a searched parameter must stay inside what the indicator accepts over its whole range
        for p in ind.params:
            v = op.get(p.name)
            if isinstance(v, str) and v[1:] in params:
                spec = params[v[1:]]
                if spec["low"] < p.low or spec["high"] > p.high:
                    errors.append(f"param {v[1:]}: range [{spec['low']}, {spec['high']}] goes outside "
                                  f"{ind.key} {p.name} limits [{p.low:g}, {p.high:g}]")
                if _whole(p) and not (float(spec["low"]).is_integer() and float(spec.get("step", 1)).is_integer()):
                    errors.append(f"param {v[1:]}: {ind.key} {p.name} takes whole numbers, use whole low and step")
    for c in rule["entry"] + rule.get("exit", []):
        for side, other in (("left", "right"), ("right", "left")):
            if "value" in c[other] and c[other]["value"] != 0 and _price_scaled(c[side]):
                errors.append(f"{c['cond']}: a price or a price-based indicator cannot be compared with a fixed "
                              "number. Adjusted prices are rescaled after later splits and dividends, so a fixed "
                              "level would use information from the future. Compare it with another price line, "
                              "or use RSI for fixed levels.")
    days = rule.get("max_days")
    if isinstance(days, str) and days[1:] in params:
        spec = params[days[1:]]
        if not (float(spec["low"]).is_integer() and float(spec.get("step", 1)).is_integer()):
            errors.append(f"param {days[1:]}: max_days takes whole numbers, use whole low and step")
    if errors:
        return errors
    # a search goes through the whole range, so the ends must run too, not only the default
    for end in ("default", "low", "high"):
        errors = _value_errors(_fill(rule, {k: p[end] for k, p in params.items()}))
        if errors:
            return errors if end == "default" else [f"with the params at their {end}: {e}" for e in errors]
    return []


def _price_scaled(op: dict) -> bool:
    return "price" in op or ("ind" in op and not INDICATORS[op["ind"]].scale_free)


def _whole(p) -> bool:
    return float(p.step).is_integer() and float(p.default).is_integer()


def _value_errors(rule: dict) -> list[str]:
    """Checks on a rule with numbers in place of $params."""
    errors = []
    for op in _operands(rule):
        if "ind" not in op:
            continue
        for p in INDICATORS[op["ind"]].params:
            if p.name not in op:
                continue
            v = op[p.name]
            if _whole(p) and not float(v).is_integer():
                errors.append(f"{op['ind']} {p.name} must be a whole number, got {v}")
            elif not p.low <= v <= p.high:
                errors.append(f"{op['ind']} {p.name} must be between {p.low:g} and {p.high:g}, got {v}")
    if "stop_atr" in rule and not rule["stop_atr"] > 0:
        errors.append(f"stop_atr must be greater than 0, got {rule['stop_atr']}")
    if "max_days" in rule and not (float(rule["max_days"]).is_integer() and rule["max_days"] >= 1):
        errors.append(f"max_days must be a whole number of at least 1, got {rule['max_days']}")
    return errors


def _fill(rule: dict, vals: dict) -> dict:
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


def resolve(rule: dict, values: dict | None = None) -> dict:
    """Copy of the rule with every $param replaced by a number (defaults unless given in values)."""
    errors = validate(rule)
    if errors:
        raise ValueError("invalid rule:\n" + "\n".join(errors))
    vals = {k: p["default"] for k, p in rule.get("params", {}).items()}
    vals.update(values or {})
    out = _fill(rule, vals)
    out.pop("params", None)  # nothing refers to them any more
    errors = _value_errors(out)
    if errors:
        raise ValueError("invalid values:\n" + "\n".join(errors))
    return out


def library() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(LIBRARY.glob("*.json"))}
