"""Checks over the local data cache. Skipped when data/ is not there (fresh clone, CI)."""
import numpy as np
import pandas as pd
import pytest

from tradetool.config import FUNDAMENTALS_DIR, PRICES_DIR
from tradetool.data.checks import big_moves, fundamental_problems, price_problems

price_files = sorted(PRICES_DIR.glob("*.parquet")) if PRICES_DIR.exists() else []
fund_files = sorted(FUNDAMENTALS_DIR.glob("*.parquet")) if FUNDAMENTALS_DIR.exists() else []
needs_prices = pytest.mark.skipif(not price_files, reason="no price cache")
needs_fund = pytest.mark.skipif(not fund_files, reason="no fundamentals cache")

# one-day moves over 50% that were checked against the news and are real
KNOWN_MOVES = {("MRNA", "2026-08-19"), ("ALNY", "2017-09-20"), ("AMD", "2016-04-22"), ("APA", "2020-03-09"),
               ("BE", "2024-11-15"), ("CVNA", "2023-06-08"), ("ECHO", "2025-08-26"), ("GL", "2024-04-11"),
               ("HOOD", "2021-08-04"), ("OXY", "2020-03-09"), ("PCG", "2019-01-14"), ("PDD", "2022-03-16"),
               ("TRGP", "2020-03-09"), ("TRGP", "2020-03-18"), ("PCG", "2019-01-24")}


@needs_prices
def test_price_files_are_clean():
    bad = {}
    for p in price_files:
        problems = price_problems(pd.read_parquet(p))
        if problems:
            bad[p.stem] = problems
    assert not bad, bad


@needs_prices
def test_all_prices_end_on_the_same_day():
    last = {p.stem: pd.read_parquet(p, columns=["close"]).index[-1] for p in price_files}
    assert len(set(last.values())) == 1, pd.Series(last).value_counts().head()


@needs_prices
def test_big_moves_are_known():
    unknown = []
    for p in price_files:
        for d in big_moves(pd.read_parquet(p)).index:
            if (p.stem, f"{d:%Y-%m-%d}") not in KNOWN_MOVES:
                unknown.append((p.stem, f"{d:%Y-%m-%d}"))
    assert not unknown, f"check these against the news, then add them to KNOWN_MOVES: {unknown}"


@needs_prices
def test_adjusted_equals_raw_on_split_free_days():
    """On a normal day the adjusted and the raw return are almost the same (dividends move them a
    little). A large difference without a split is a broken adjustment (e.g. RTX 2020-04-03)."""
    bad = []
    for p in price_files:
        df = pd.read_parquet(p)
        adj = np.log(df["close"]).diff()
        raw = np.log(df["close_raw"]).diff()
        split = (raw - adj).abs() > np.log(1.4)  # splits of 3:2 and more
        wrong = (adj.abs() - raw.abs() > 0.03) & ~split
        bad += [(p.stem, f"{d:%Y-%m-%d}") for d in df.index[wrong]]
    assert len(bad) <= 10, bad


@needs_fund
def test_fundamental_files_are_clean():
    bad = {p.stem: fundamental_problems(pd.read_parquet(p)) for p in fund_files}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, bad
