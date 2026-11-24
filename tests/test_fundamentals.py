import pandas as pd
import pytest

from tradetool.data import fundamentals as F


def fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fy": 2025, "fp": "Q"}


@pytest.fixture
def facts():
    # fiscal year Oct-Sep, like Apple. Q1 and Q2 and Q3 reported as 3 months, cash flow only year-to-date,
    # Q4 only inside the 10-K. Every number is repeated a year later in the next filings.
    rev = [
        fact("2024-09-29", "2024-12-28", 100, "2025-01-31"),
        fact("2024-12-29", "2025-03-29", 90, "2025-05-02"),
        fact("2025-03-30", "2025-06-28", 80, "2025-08-01"),
        fact("2024-09-29", "2025-06-28", 270, "2025-08-01"),
        fact("2024-09-29", "2025-09-27", 380, "2025-10-31", "10-K"),
        # same quarter again in next year's 10-Q, with a restated value: must be ignored
        fact("2024-12-29", "2025-03-29", 95, "2026-05-01"),
    ]
    ocf = [
        fact("2024-09-29", "2024-12-28", 30, "2025-01-31"),
        fact("2024-09-29", "2025-03-29", 50, "2025-05-02"),
        fact("2024-09-29", "2025-06-28", 65, "2025-08-01"),
        fact("2024-09-29", "2025-09-27", 100, "2025-10-31", "10-K"),
    ]
    return {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
                        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}}}}


def test_q4_is_year_minus_nine_months(facts):
    f = F.parse_facts(facts)
    rev = F.quarters(f, "revenue")
    assert rev[pd.Timestamp("2025-09-27")] == 110
    q4 = f[(f.concept == "revenue") & (f.end == "2025-09-27")].iloc[0]
    assert q4["derived"] and q4["filed"] == pd.Timestamp("2025-10-31")


def test_cash_flow_from_year_to_date(facts):
    f = F.parse_facts(facts)
    assert F.quarters(f, "op_cash_flow").tolist() == [30, 20, 15, 35]


def test_first_filing_wins(facts):
    f = F.parse_facts(facts)
    assert F.quarters(f, "revenue")[pd.Timestamp("2025-03-29")] == 90


def test_not_known_before_filing_day(facts):
    f = F.parse_facts(facts)
    # the 10-K is filed on 2025-10-31: not usable that day, usable the next day
    assert pd.Timestamp("2025-09-27") not in F.quarters(f, "revenue", "2025-10-31").index
    assert pd.Timestamp("2025-09-27") in F.quarters(f, "revenue", "2025-11-01").index
    assert F.ttm(f, "revenue", "2025-10-31") != F.ttm(f, "revenue", "2025-11-01")


@pytest.mark.parametrize("date", ["2025-02-01", "2025-08-02", "2025-11-01", "2026-06-01"])
def test_no_lookahead(facts, date):
    """What we know at `date` must be the same if facts filed on or after `date` never existed."""
    full = F.parse_facts(facts)
    cut = {ns: {tag: {"units": {u: [r for r in rows if r["filed"] < date] for u, rows in d["units"].items()}}
                for tag, d in tags.items()} for ns, tags in facts.items()}
    part = F.parse_facts(cut)
    for concept in ["revenue", "op_cash_flow"]:
        pd.testing.assert_series_equal(F.quarters(full, concept, date), F.quarters(part, concept, date))


def test_ttm_needs_four_consecutive_quarters(facts):
    f = F.parse_facts(facts)
    assert F.ttm(f, "revenue", "2025-11-01") == 380
    assert pd.isna(F.ttm(f, "revenue", "2025-08-02"))  # only 3 quarters known


def test_failed_download_keeps_old_summary_and_goes_on(tmp_path, monkeypatch):
    import requests

    from tradetool.data import fundamentals as fm

    monkeypatch.setattr(fm, "FUNDAMENTALS_DIR", tmp_path)
    monkeypatch.setattr(fm, "cik_map", lambda: {"AAA": 1, "BBB": 2})
    pd.DataFrame([{"ticker": "AAA", "cik": 1, "status": "ok", "sic": 3674}]).to_csv(tmp_path / "_summary.csv",
                                                                                   index=False)

    def fake_get(url):
        if "CIK0000000001" in url:
            raise requests.ConnectionError("network down")
        return {"facts": {}} if "companyfacts" in url else {"sic": "2834", "sicDescription": "Pharma"}

    monkeypatch.setattr(fm, "_sec_get", fake_get)
    s = fm.download(["AAA", "BBB"]).set_index("ticker")
    assert s.loc["AAA", "status"] == "ok" and s.loc["AAA", "sic"] == 3674  # old row kept
    assert s.loc["BBB", "status"] == "ok"  # the failure did not stop the loop
