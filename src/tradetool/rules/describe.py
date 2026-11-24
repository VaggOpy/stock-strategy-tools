"""A rule in plain Greek, written by code and not by the LLM, so the user can check what will run."""
from __future__ import annotations

from tradetool.indicators.registry import INDICATORS
from tradetool.rules.schema import resolve

PRICE = {"open": "το άνοιγμα", "high": "το υψηλό", "low": "το χαμηλό", "close": "το κλείσιμο"}
COND = {"above": "πάνω από", "below": "κάτω από",
        "cross_above": "περνάει πάνω από", "cross_below": "περνάει κάτω από"}
OUT = {"upper": "πάνω", "lower": "κάτω", "mid": "μεσαία", "macd": "γραμμή", "signal": "σήμα", "hist": "ιστόγραμμα"}


def _num(v) -> str:
    return f"{v:g}".replace(".", ",")


def operand(op: dict) -> str:
    if "value" in op:
        return _num(op["value"])
    if "price" in op:
        text = PRICE[op["price"]]
    else:
        ind = INDICATORS[op["ind"]]
        args = ", ".join(_num(op.get(p.name, p.default)) for p in ind.params)
        text = f"{ind.key}({args})"
        if "out" in op:
            text += f" {OUT.get(op['out'], op['out'])}"
    k = op.get("shift", 0)
    if k == 1:
        text += " της προηγούμενης μέρας"
    elif k > 1:
        text += f" πριν από {k} μέρες"
    return text


def condition(c: dict) -> str:
    return f"{operand(c['left'])} {COND[c['cond']]} {operand(c['right'])}"


def describe(rule: dict, values: dict | None = None) -> list[str]:
    """One line per part of the rule, with the parameters at their defaults (or `values`)."""
    r = resolve(rule, values)
    entry = [condition(c) for c in r["entry"]]
    lines = ["Αγορά στο άνοιγμα της επόμενης μέρας όταν στο κλείσιμο "
             + ("ισχύει: " + entry[0] if len(entry) == 1 else "ισχύουν όλα: " + "· ".join(entry))]
    exits = [condition(c) for c in r.get("exit", [])]
    if exits:
        lines.append("Πώληση στο άνοιγμα της επόμενης μέρας όταν "
                     + ("ισχύει: " + exits[0] if len(exits) == 1 else "ισχύει ένα από: " + "· ".join(exits)))
    if "stop_atr" in r:
        lines.append(f"Stop στην τιμή αγοράς μείον {_num(r['stop_atr'])} × ATR(14)")
    if "max_days" in r:
        lines.append(f"Πώληση το αργότερο μετά από {_num(r['max_days'])} μέρες")
    if not exits and "stop_atr" not in r and "max_days" not in r:
        lines.append("Δεν υπάρχει κανόνας εξόδου: η θέση κρατιέται μέχρι το τέλος")
    return lines
