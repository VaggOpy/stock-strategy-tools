import json

import pytest
import requests
from google.genai import errors

from tradetool.backtest.engine import run
from tradetool.backtest.metrics import buy_and_hold, compare
from tradetool.indicators.registry import INDICATORS
from tradetool.llm import client as llm_client
from tradetool.llm import prompts
from tradetool.llm.strategy import explain, facts, parse, to_rule, unknown_numbers
from tradetool.rules.describe import describe
from tradetool.rules.schema import library

from helpers import random_prices


class Scripted:
    """Gives the prepared answers in order and keeps what it was asked."""
    name = "test/scripted"
    version = "1"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def chat(self, system, messages, as_json=False):
        self.asked.append((system, [dict(m) for m in messages], as_json))
        return self.answers.pop(0)


GOOD = {"rule": library()["sma_trend"], "assumptions": ["used the 200-day SMA"], "unsupported": []}
FIXED_LEVEL = {"rule": {"entry": [{"cond": "above", "left": {"price": "close"}, "right": {"value": 100}}]},
               "assumptions": [], "unsupported": []}


def test_prompt_lists_every_indicator_and_valid_examples():
    text = prompts.translate_system()
    for key in INDICATORS:
        assert f"- {key} (" in text
    for rule in library().values():
        assert rule["description"] in text
    assert "{" in text and "{{" not in text  # format() left no doubled braces


def test_valid_first_answer():
    llm = Scripted(json.dumps(GOOD))
    t = to_rule("hold above the 200 day average", llm)
    assert t.ok and t.attempts == 1 and t.rule == GOOD["rule"]
    assert t.assumptions == ["used the 200-day SMA"]
    assert llm.asked[0][2] is True  # JSON mode


def test_rejected_rule_goes_back_with_the_problems():
    llm = Scripted(json.dumps(FIXED_LEVEL), json.dumps(GOOD))
    t = to_rule("buy when the close is above 100", llm)
    assert t.ok and t.attempts == 2
    assert "fixed number" in t.problems[0][0]
    retry = llm.asked[1][1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "fixed number" in retry[2]["text"]


def test_not_json_then_json_in_code_block():
    llm = Scripted("sure, here it is: {rule:", "```json\n" + json.dumps(GOOD) + "\n```")
    t = to_rule("x", llm)
    assert t.ok and t.attempts == 2 and "not valid JSON" in t.problems[0][0]


def test_gives_up_after_the_retries():
    llm = Scripted(*[json.dumps(FIXED_LEVEL)] * 3)
    t = to_rule("x", llm, retries=2)
    assert not t.ok and t.rule is None and t.attempts == 3 and len(t.problems) == 3


def test_nothing_supported():
    answer = {"rule": None, "assumptions": [], "unsupported": ["news sentiment"]}
    t = to_rule("buy on good news", Scripted(json.dumps(answer)))
    assert t.ok and t.rule is None and t.unsupported == ["news sentiment"]


def test_parse():
    assert parse('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        parse("[1, 2]")


def test_cache_answers_once(tmp_path):
    llm = Scripted("first", "second")
    cached = llm_client.Cached(llm, tmp_path)
    msg = [{"role": "user", "text": "hi"}]
    assert cached.chat("sys", msg) == "first"
    assert cached.chat("sys", msg) == "first"
    assert cached.chat("sys", [{"role": "user", "text": "other"}]) == "second"
    assert (cached.calls, cached.hits) == (2, 1)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["model"] == "test/scripted"


def test_describe_library_rules():
    lines = describe(library()["donchian_breakout"])
    assert lines[0] == "Αγορά στο άνοιγμα της επόμενης μέρας όταν στο κλείσιμο ισχύει: " \
                       "το κλείσιμο πάνω από DC(55) πάνω της προηγούμενης μέρας"
    assert lines[2] == "Stop στην τιμή αγοράς μείον 2 × ATR(14)"
    assert "ισχύουν όλα" in describe(library()["rsi_pullback"])[0]
    assert "μέχρι το τέλος" in describe({"entry": [{"cond": "above", "left": {"ind": "RSI"}, "right": {"value": 50}}]})[-1]


def test_unknown_numbers():
    src = "CAGR 12.34%, Sharpe 0.87, 2017"
    assert unknown_numbers("απόδοση 12,3% και 12%, Sharpe 0,87 από το 2017", src) == []
    assert unknown_numbers("απόδοση 15% και Sharpe 0,9", src) == ["15"]


def test_explain_gets_only_our_numbers():
    df = random_prices(800, seed=3)
    rule = library()["sma_trend"]
    res = run(df, rule, start=df.index[260])
    table = compare(res, df, bench=buy_and_hold(df, res.equity.index) * 1.0)
    text = facts(table, "XYZ", res.equity.index[0], res.equity.index[-1], 5)
    assert "Strategy:" in text and "SPY:" in text and "Strategy trades:" in text
    llm = Scripted("Η στρατηγική ...")
    assert explain(llm, "trend filter", rule, text) == "Η στρατηγική ..."
    system, messages, as_json = llm.asked[0]
    assert system == prompts.EXPLAIN and not as_json
    assert text in messages[0]["text"] and "SMA(200)" in messages[0]["text"]


def test_gemini_needs_a_key(monkeypatch):
    monkeypatch.setattr(llm_client, "env", lambda name: None)
    with pytest.raises(llm_client.LLMError, match="GEMINI_API_KEY"):
        llm_client.Gemini()


def test_retry_delay_from_429():
    per_minute = errors.ClientError(429, {"error": {"code": 429, "message": "x", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}})
    per_day = errors.ClientError(429, {"error": {"code": 429, "message": "x", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    assert llm_client._retry_delay(per_minute) == 13
    assert llm_client._retry_delay(per_day) is None


def test_ollama_not_running(monkeypatch):
    def refuse(*a, **k):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm_client.requests, "post", refuse)
    with pytest.raises(llm_client.LLMError, match="not running"):
        llm_client.Ollama(host="http://localhost:1").chat("s", [{"role": "user", "text": "x"}])
