"""How well does an LLM turn a text description into the rule that was meant?

Every case in llm_cases.json has a description and the rule a person would write for it (or null if
the engine cannot do it). The LLM rule and the expected rule are run on five stocks over the test
period and their daily positions are compared. Two different JSON texts can be the same strategy
(above vs cross_above after an exit, a number vs a $param), so the JSON itself is not compared.

Kinds of cases:
- exact: everything is given,
- vague: "buy the dip", the expected rule is one usual reading, a different one is not a mistake,
- partial: part of it cannot be done (volume, earnings), the rest must be built and the part reported,
- unsupported: nothing can be done, rule must be null and the reason reported.

For the supported cases the LLM also explains the backtest on AAPL, and the numbers in the
explanation are checked against the numbers it was given.

Answers are cached in data/llm_cache, so a second run costs no requests.
Run: python experiments/llm_rules.py                    (Gemini, model from .env or gemini-3.5-flash-lite)
     python experiments/llm_rules.py --model gemini-3.8-flash
     python experiments/llm_rules.py --ollama qwen2.5:3b
"""
import argparse
import json

import numpy as np
import pandas as pd

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import compare
from tradetool.config import ROOT, TEST_END, TEST_START
from tradetool.data import prices
from tradetool.llm.client import Cached, Gemini, LLMError, Ollama
from tradetool.llm.strategy import explain, facts, to_rule, unknown_numbers
from tradetool.rules.describe import describe

CASES = ROOT / "experiments" / "llm_cases.json"
OUT = ROOT / "experiments" / "results" / "llm_rules"
STOCKS = ["SPY", "AAPL", "NVDA", "XOM", "JNJ"]


def positions(rule, data):
    return {s: run(df, rule, start=TEST_START, cost_bps=0).position.loc[TEST_START:] for s, df in data.items()}


def agreement(a, b):
    return float(np.mean([(a[s] == b[s]).mean() for s in a]))


def score(case, t, data):
    row = {"id": case["id"], "kind": case["kind"], "valid": t.ok, "attempts": t.attempts,
           "first_try": t.ok and t.attempts == 1, "built_rule": t.rule is not None,
           "unsupported_reported": len(t.unsupported) > 0, "assumptions": len(t.assumptions),
           "problems": " | ".join(p[0] for p in t.problems)}
    if case["rule"] is None:
        row["agreement"] = np.nan
        row["correct"] = t.ok and t.rule is None and len(t.unsupported) > 0
    else:
        row["agreement"] = agreement(positions(case["rule"], data), positions(t.rule, data)) if t.rule else 0.0
        row["correct"] = row["agreement"] >= 0.99 and (case["kind"] != "partial" or len(t.unsupported) > 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Gemini model name")
    ap.add_argument("--ollama", help="Ollama model name instead of Gemini")
    ap.add_argument("--no-explain", action="store_true")
    args = ap.parse_args()
    llm = Cached(Ollama(args.ollama) if args.ollama else Gemini(args.model))
    out = OUT / llm.name.replace("/", "_").replace(":", "-")

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    data = {s: prices.load(s, end=TEST_END) for s in STOCKS}
    rows, answers, notes = [], [], []
    for case in cases:
        try:
            t = to_rule(case["text"], llm)
        except LLMError as e:
            print(f"stopped at {case['id']}: {e}\n(answers so far are cached, run again later)")
            break
        rows.append(score(case, t, data))
        answers.append({"id": case["id"], **t.__dict__})
        r = rows[-1]
        print(f"{case['id']:22s} {case['kind']:12s} attempts {t.attempts}  correct {r['correct']!s:5s} "
              f"agreement {r['agreement']:.3f}  unsupported {t.unsupported}")

        if t.rule is not None and not args.no_explain:
            df = data["AAPL"]
            res = run(df, t.rule, start=TEST_START)
            fact_text = facts(compare(res, df), "AAPL", res.equity.index[0], res.equity.index[-1], 5)
            try:
                text = explain(llm, case["text"], t.rule, fact_text)
            except LLMError as e:
                print(f"no explanation for {case['id']}: {e}")
                continue
            made_up = unknown_numbers(text, fact_text, "\n".join(describe(t.rule)), case["text"])
            rows[-1]["explanation_unknown_numbers"] = len(made_up)
            notes.append(f"## {case['id']}\n\n{case['text']}\n\n{fact_text}\n\n{text}\n\n"
                         f"Numbers not in the facts: {made_up or 'none'}\n")

    if not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "cases.csv", index=False)
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "explanations.md").write_text("\n".join(notes), encoding="utf-8")
    by_kind = table.groupby("kind").agg(cases=("id", "size"), valid=("valid", "mean"), first_try=("first_try", "mean"),
                                        correct=("correct", "mean"), agreement=("agreement", "mean"),
                                        attempts=("attempts", "mean"))
    by_kind.loc["all"] = [len(table), table["valid"].mean(), table["first_try"].mean(), table["correct"].mean(),
                          table["agreement"].mean(), table["attempts"].mean()]
    by_kind.to_csv(out / "summary.csv")
    print(f"\n{llm.name}: {llm.calls} new requests, {llm.hits} from cache")
    print(by_kind.round(3).to_string())
    if "explanation_unknown_numbers" in table:
        n = table["explanation_unknown_numbers"].dropna()
        print(f"explanations: {len(n)}, with numbers not in the facts: {(n > 0).sum()}")


if __name__ == "__main__":
    main()
