"""Point-in-time quarterly fundamentals from SEC EDGAR (companyfacts API).

How a value becomes "known":
- Every XBRL fact has the period (start, end) and the date it was filed. The same number shows up
  again a year later as a comparison column, so for each period we keep the FIRST filing only.
  That is the number the market saw first; later restatements are ignored (using an old number is
  stale, but never look-ahead).
- Companies report Q4 only inside the annual 10-K, and cash flow only year-to-date (3, 6, 9, 12
  months). Missing quarters are derived as differences of cumulative values with the same start,
  e.g. Q4 = FY - 9M, and are known from the later of the two filing dates.
- A value filed on day d is usable from day d+1 (filings often come after the close).
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from tradetool.config import FUNDAMENTALS_DIR, env

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# concept -> (unit, XBRL tags in order of preference). Companies change tags over time
# (e.g. Revenues -> RevenueFromContractWithCustomer... after ASC 606 in 2018), so all are merged.
CONCEPTS = {
    # banks report net revenue (after interest expense), that comes before the generic tag
    "revenue": ("USD", ["RevenueFromContractWithCustomerExcludingAssessedTax", "RevenuesNetOfInterestExpense",
                        "Revenues", "SalesRevenueNet",
                        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]),
    "net_income": ("USD", ["NetIncomeLoss", "ProfitLoss"]),
    "eps_diluted": ("USD/shares", ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted",
                                   "EarningsPerShareBasic"]),
    "op_cash_flow": ("USD", ["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
    "equity": ("USD", ["StockholdersEquity",
                       "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "shares_out": ("shares", ["EntityCommonStockSharesOutstanding"]),  # dei, cover page
}
INSTANT = {"equity", "shares_out"}
FORMS = ("10-Q", "10-K", "20-F", "40-F")

_last_call = [0.0]


def _sec_get(url):
    # SEC allows 10 requests/second
    wait = 0.12 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    r = requests.get(url, headers={"User-Agent": env("SEC_USER_AGENT", required=True)}, timeout=60)
    r.raise_for_status()
    return r.json()


def cik_map() -> dict[str, int]:
    path = FUNDAMENTALS_DIR / "company_tickers.json"
    if not path.exists():
        FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_sec_get(TICKERS_URL)))
    data = json.loads(path.read_text())
    # SEC writes BRK-B, the universe uses BRK.B
    return {v["ticker"].replace("-", "."): v["cik_str"] for v in data.values()}


def _raw_facts(facts: dict, concept: str) -> pd.DataFrame:
    unit, tags = CONCEPTS[concept]
    frames = []
    for prio, tag in enumerate(tags):
        for ns in ("us-gaap", "dei", "ifrs-full"):
            rows = facts.get(ns, {}).get(tag, {}).get("units", {}).get(unit)
            if rows:
                d = pd.DataFrame(rows)
                d["prio"] = prio
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["form"].str.startswith(FORMS)]
    if "start" not in d:
        d["start"] = None
    d["start"] = pd.to_datetime(d["start"])
    d["end"] = pd.to_datetime(d["end"])
    d["filed"] = pd.to_datetime(d["filed"])
    # first filing of each period; between tags with the same period prefer the higher priority tag
    d = d.sort_values(["filed", "prio"]).drop_duplicates(["start", "end"], keep="first")
    return d[["start", "end", "val", "filed"]]


def _quarterly_flow(d: pd.DataFrame) -> pd.DataFrame:
    """Quarter values from a mix of 3-month and year-to-date facts."""
    d = d.dropna(subset=["start"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    is_q = d["days"].between(80, 100)

    q = {}
    for r in d[is_q].itertuples():
        if r.end not in q or r.filed < q[r.end][1]:
            q[r.end] = (r.val, r.filed, False)

    # differences of cumulative values that share the same start (fiscal year start)
    for _, g in d.groupby("start"):
        g = g.sort_values("end")
        prev = None
        for r in g.itertuples():
            if prev is not None and 80 <= r.days - prev.days <= 100 and r.end not in q:
                q[r.end] = (r.val - prev.val, max(r.filed, prev.filed), True)
            prev = r

    # fallback for an annual value without a 9-month value: FY minus the three known quarters
    for r in d[d["days"].between(350, 380)].itertuples():
        if r.end in q:
            continue
        inside = sorted(e for e in q if r.start < e < r.end)
        if len(inside) == 3:
            q[r.end] = (r.val - sum(q[e][0] for e in inside), max([r.filed] + [q[e][1] for e in inside]), True)

    out = pd.DataFrame([(e, v, f, der) for e, (v, f, der) in q.items()], columns=["end", "value", "filed", "derived"])
    return out.sort_values("end")


def _instant(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("filed").drop_duplicates("end", keep="first")
    return pd.DataFrame({"end": d["end"], "value": d["val"], "filed": d["filed"], "derived": False})


def parse_facts(facts: dict) -> pd.DataFrame:
    """Long table: concept, end, value, filed, derived."""
    out = []
    for concept in CONCEPTS:
        d = _raw_facts(facts, concept)
        if d.empty:
            continue
        q = _instant(d) if concept in INSTANT else _quarterly_flow(d)
        q.insert(0, "concept", concept)
        out.append(q)
    if not out:
        return pd.DataFrame(columns=["concept", "end", "value", "filed", "derived"])
    return pd.concat(out, ignore_index=True)


def _path(ticker):
    return FUNDAMENTALS_DIR / f"{ticker}.parquet"


def download(tickers: list[str]) -> pd.DataFrame:
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    ciks = cik_map()
    summary = []
    for t in tickers:
        cik = ciks.get(t)
        row = {"ticker": t, "cik": cik}
        if cik is None:
            summary.append({**row, "status": "no CIK"})
            continue
        try:
            facts = _sec_get(FACTS_URL.format(cik=cik))["facts"]
            sub = _sec_get(SUBMISSIONS_URL.format(cik=cik))
        except requests.HTTPError as e:
            summary.append({**row, "status": f"http {e.response.status_code}"})
            continue
        q = parse_facts(facts)
        q.to_parquet(_path(t))
        rev = q[q["concept"] == "revenue"]
        summary.append({**row, "status": "ok", "sic": sub.get("sic"), "sic_desc": sub.get("sicDescription"),
                        "rows": len(q), "revenue_quarters": len(rev),
                        "first_quarter": rev["end"].min().date() if len(rev) else None})
    s = pd.DataFrame(summary)
    s.to_csv(FUNDAMENTALS_DIR / "_summary.csv", index=False)
    return s


def load(ticker: str) -> pd.DataFrame:
    p = _path(ticker)
    if not p.exists():
        download([ticker])
    return pd.read_parquet(p)


# ---- point-in-time access ----

def known_at(f: pd.DataFrame, date) -> pd.DataFrame:
    """Only facts that were public before `date` (filed on an earlier day)."""
    return f[f["filed"] < pd.Timestamp(date)]


def quarters(f: pd.DataFrame, concept: str, date=None) -> pd.Series:
    g = f[f["concept"] == concept] if date is None else known_at(f[f["concept"] == concept], date)
    return g.set_index("end")["value"].sort_index()


def _consecutive(ends) -> bool:
    gaps = pd.Series(ends).diff().dt.days.dropna()
    return len(gaps) > 0 and gaps.between(80, 100).all()


def ttm(f: pd.DataFrame, concept: str, date, lag: int = 0) -> float:
    """Sum of the last 4 quarters known at `date`. lag=4 gives the 4 quarters before those."""
    q = quarters(f, concept, date)
    q = q.iloc[len(q) - 4 - lag: len(q) - lag] if len(q) >= 4 + lag else q.iloc[:0]
    return float(q.sum()) if len(q) == 4 and _consecutive(q.index) else float("nan")


def ttm_growth(f: pd.DataFrame, concept: str, date) -> float:
    now, before = ttm(f, concept, date), ttm(f, concept, date, lag=4)
    return now / before - 1 if before and before > 0 else float("nan")
