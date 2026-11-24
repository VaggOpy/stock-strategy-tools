"""Check what each free data source actually returns and how fast.

    python experiments/check_apis.py            # all
    python experiments/check_apis.py yahoo sec  # some

Sources without a key in .env are skipped. Results go to experiments/results/check_apis/api_checks.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import requests

from tradetool.config import ROOT, env

sys.stdout.reconfigure(encoding="utf-8")

TODAY = dt.date.today()
RESULTS: dict[str, dict] = {}


def section(name):
    def deco(fn):
        def run():
            print(f"\n=== {name} ===")
            t0 = time.time()
            try:
                out = fn()
            except Exception as e:  # keep going with the other sources
                out = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            out["seconds"] = round(time.time() - t0, 1)
            RESULTS[name] = out
            for k, v in out.items():
                print(f"  {k}: {v}")
        run.source = name
        return run
    return deco


@section("yahoo")
def check_yahoo():
    import yfinance as yf

    out = {"status": "ok", "key_needed": False}
    spy = yf.Ticker("SPY").history(period="max", interval="1d", auto_adjust=False)
    out["spy_first_day"] = str(spy.index[0].date())
    out["spy_last_day"] = str(spy.index[-1].date())
    out["spy_rows"] = len(spy)
    out["columns"] = list(spy.columns)

    # 50 tickers x 10 years, to see speed and failures
    tickers = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V UNH XOM LLY JNJ PG MA HD COST MRK ABBV "
               "CVX PEP KO ADBE WMT CRM BAC NFLX AMD TMO ORCL LIN ACN MCD CSCO ABT DIS WFC INTC DHR "
               "QQQ IWM XLK XLF XLE XLV XLY XLP XLI XLU").split()
    t0 = time.time()
    batch = yf.download(tickers, start="2015-01-01", interval="1d", auto_adjust=False,
                        group_by="ticker", progress=False, threads=True)
    out["batch_50_tickers_10y_seconds"] = round(time.time() - t0, 1)
    got = [t for t in tickers if t in batch.columns.get_level_values(0) and batch[t]["Close"].notna().sum() > 0]
    out["batch_ok"] = f"{len(got)}/{len(tickers)}"

    # delisted names - does yahoo still have them?
    delisted = {}
    for t in ["SIVB", "FRC", "BBBY", "TWTR"]:
        h = yf.Ticker(t).history(period="max", interval="1d")
        delisted[t] = f"{len(h)} rows" + (f", last {h.index[-1].date()}" if len(h) else "")
    out["delisted_tickers"] = delisted
    return out


SEC_UA_FALLBACK = "UTH student research project tradetool"


def sec_get(url):
    ua = env("SEC_USER_AGENT") or SEC_UA_FALLBACK
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    return r


@section("sec_edgar")
def check_sec():
    out = {"status": "ok", "key_needed": False,
           "user_agent": "from .env" if env("SEC_USER_AGENT") else "generic fallback (set SEC_USER_AGENT)"}

    r = sec_get("https://www.sec.gov/files/company_tickers.json")
    out["company_tickers_http"] = r.status_code
    if r.status_code != 200:
        out["status"] = "blocked"
        return out
    tickers = r.json()
    out["companies_in_ticker_map"] = len(tickers)
    cik = {v["ticker"]: v["cik_str"] for v in tickers.values()}

    # every XBRL fact the company filed, each with its filing date
    r = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik['AAPL']:010d}.json")
    out["companyfacts_http"] = r.status_code
    facts = r.json()
    gaap = facts["facts"]["us-gaap"]
    out["aapl_us_gaap_concepts"] = len(gaap)
    out["aapl_size_mb"] = round(len(r.content) / 1e6, 1)
    rev_key = next(k for k in ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"] if k in gaap)
    rev = gaap[rev_key]["units"]["USD"]
    out["aapl_revenue_concept"] = rev_key
    out["aapl_revenue_points"] = len(rev)
    out["revenue_example_fields"] = sorted(rev[-1].keys())
    q = [x for x in rev if x.get("fp", "").startswith("Q") and x.get("form") == "10-Q"]
    if q:
        last = q[-1]
        lag = (dt.date.fromisoformat(last["filed"]) - dt.date.fromisoformat(last["end"])).days
        out["latest_10Q_revenue"] = f"period end {last['end']}, filed {last['filed']} (lag {lag} days)"
    eps = gaap.get("EarningsPerShareDiluted", {}).get("units", {}).get("USD/shares", [])
    out["aapl_eps_diluted_points"] = len(eps)
    out["earliest_filed"] = min(x["filed"] for c in gaap.values() for u in c["units"].values() for x in u)

    # SIC industry code, for grouping stocks later
    r = sec_get(f"https://data.sec.gov/submissions/CIK{cik['NVDA']:010d}.json")
    sub = r.json()
    out["nvda_sic"] = f"{sub.get('sic')} {sub.get('sicDescription')}"

    # SEC allows max 10 req/s
    t0 = time.time()
    codes = [sec_get(f"https://data.sec.gov/submissions/CIK{cik[t]:010d}.json").status_code
             for t in ["MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "LLY", "COST", "AMD"]]
    out["10_requests_seconds"] = round(time.time() - t0, 2)
    out["10_requests_status"] = sorted(set(codes))
    return out


@section("alpaca")
def check_alpaca():
    key, secret = env("ALPACA_API_KEY_ID"), env("ALPACA_API_SECRET_KEY")
    if not (key and secret):
        return {"status": "skipped", "reason": "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not in .env"}
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    out = {"status": "ok", "key_needed": True}
    url = "https://data.alpaca.markets/v2/stocks/bars"
    for feed in ["sip", "iex"]:
        p = {"symbols": "SPY", "timeframe": "1Day", "start": "2015-01-01", "end": str(TODAY - dt.timedelta(days=1)),
             "adjustment": "all", "feed": feed, "limit": 10000}
        r = requests.get(url, headers=h, params=p, timeout=30)
        rec = {"http": r.status_code}
        if r.ok:
            bars = r.json().get("bars", {}).get("SPY", [])
            rec.update(rows=len(bars), first=bars[0]["t"][:10] if bars else None,
                       last=bars[-1]["t"][:10] if bars else None)
        else:
            rec["message"] = r.text[:200]
        out[f"spy_daily_{feed}"] = rec
        out["rate_limit_header"] = r.headers.get("X-RateLimit-Limit")
    # free plan blocks the last 15 min of SIP data
    p = {"symbols": "SPY", "timeframe": "1Day", "start": str(TODAY - dt.timedelta(days=3)), "feed": "sip"}
    r = requests.get(url, headers=h, params=p, timeout=30)
    out["sip_recent_http"] = r.status_code
    p = {"symbols": "SIVB", "timeframe": "1Day", "start": "2022-01-01", "end": "2023-12-31", "feed": "sip"}
    r = requests.get(url, headers=h, params=p, timeout=30)
    out["delisted_SIVB_rows"] = len(r.json().get("bars", {}).get("SIVB", [])) if r.ok else f"http {r.status_code}"
    return out


@section("gemini")
def check_gemini():
    key = env("GEMINI_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "GEMINI_API_KEY not in .env"}
    from google import genai

    client = genai.Client(api_key=key)
    models = [m.name for m in client.models.list()]
    out = {"status": "ok", "key_needed": True, "models_available": len(models),
           "flash_models": [m for m in models if "flash" in m][:8]}
    model = next((m for m in models if "flash" in m and "lite" not in m), models[0]).removeprefix("models/")
    t0 = time.time()
    resp = client.models.generate_content(
        model=model, contents='Reply with only this JSON: {"ok": true}')
    out["test_model"] = model
    out["test_reply"] = resp.text.strip()[:80]
    out["test_latency_s"] = round(time.time() - t0, 1)
    return out


@section("ollama")
def check_ollama():
    host = env("OLLAMA_HOST") or "http://localhost:11434"
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
    except requests.ConnectionError:
        return {"status": "not running", "reason": f"nothing listening on {host} (Ollama not installed or not started)"}
    models = [m["name"] for m in r.json().get("models", [])]
    return {"status": "ok", "key_needed": False, "models": models or "none pulled yet"}


CHECKS = [check_yahoo, check_sec, check_alpaca, check_gemini, check_ollama]

if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for check in CHECKS:
        if not wanted or any(w in check.source for w in wanted):
            check()
    log = ROOT / "experiments" / "results" / "check_apis" / "api_checks.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": str(TODAY), **RESULTS}, ensure_ascii=False, default=str) + "\n")
    print(f"\nSaved to {log}")
