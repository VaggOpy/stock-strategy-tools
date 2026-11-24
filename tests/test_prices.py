import pandas as pd
import pytest

from tradetool.data.prices import clean, reconcile


def test_clean_drops_filled_days():
    idx = pd.bdate_range("2022-02-24", periods=6)
    df = pd.DataFrame({"open": [10, 9, 8, 8, 8, 12], "high": [11, 10, 9, 8, 8, 13], "low": [9, 8, 7, 8, 8, 11],
                       "close": [9, 8, 8, 8, 8, 12], "volume": [5e5, 6e5, 7e5, 0, 0, 4e5]}, index=idx)
    out = clean(df)
    assert len(out) == 4
    assert (out["volume"] > 0).all()


def test_clean_keeps_only_part_after_long_gap():
    # same symbol, two different companies with a year without trading in between
    old = pd.bdate_range("2016-01-01", "2017-08-31")
    new = pd.bdate_range("2019-04-02", "2019-12-31")
    df = pd.DataFrame({"close": 1.0, "volume": 1e6}, index=old.append(new))
    out = clean(df)
    assert out.index[0] == pd.Timestamp("2019-04-02")
    assert len(out) == len(new)


def _bars(close, raw=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    close = pd.Series(close, idx, dtype=float)
    raw = close if raw is None else pd.Series(raw, idx, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1e6, "close_raw": raw})


FLAT = [100.0, 101, 102, 101, 103, 104, 103, 105, 106, 105]


def test_reconcile_replaces_a_wrong_bar():
    y = _bars(FLAT)
    bad = FLAT.copy()
    bad[4] = 109.0  # Alpaca close not the official one on that day only
    fixed, fixes = reconcile(_bars(bad), y, pd.Series(0.0, y.index))
    assert [f["kind"] for f in fixes] == ["bar"]
    assert fixed["close"].iloc[4] == pytest.approx(103.0)
    assert fixed["close"].drop(fixed.index[4]).equals(_bars(bad)["close"].drop(fixed.index[4]))


def test_reconcile_wrong_bar_on_the_last_day():
    """Without a next day the bad bar must still be replaced, not taken for a corporate action that
    rescales the whole history."""
    bad = FLAT.copy()
    bad[-1] = 112.0
    fixed, fixes = reconcile(_bars(bad), _bars(FLAT), pd.Series(0.0, pd.bdate_range("2020-01-01", periods=10)))
    assert [f["kind"] for f in fixes] == ["bar"]
    assert fixed["close"].iloc[-1] == pytest.approx(105.0)
    assert fixed["close"].iloc[:-1].equals(_bars(bad)["close"].iloc[:-1])


def test_reconcile_fixes_a_missed_spinoff():
    # spin-off on day 5: the raw price drops 30%, Alpaca did not adjust it, Yahoo did
    raw = FLAT[:5] + [p * 0.7 for p in FLAT[5:]]
    a = _bars(raw)
    y = _bars([p * 0.7 for p in FLAT], raw=raw)
    fixed, fixes = reconcile(a, y, pd.Series(0.0, a.index))
    assert [f["kind"] for f in fixes] == ["adjustment"]
    r = fixed["close"].pct_change()
    assert r.iloc[5] == pytest.approx(FLAT[5] / FLAT[4] - 1)
    assert (fixed["close"].iloc[5:] == a["close"].iloc[5:]).all()  # after the action nothing changes


def test_reconcile_ignores_yahoo_phantom_and_splits():
    # Yahoo shows a +70% day that never happened; the stock also splits 2:1 on day 7 in Alpaca raw
    raw = FLAT[:7] + [p / 2 for p in FLAT[7:]]
    a = _bars(FLAT, raw=raw)
    ph = FLAT.copy()
    ph[3:] = [p * 1.7 for p in FLAT[3:]]
    y = _bars(ph, raw=ph)
    fixed, fixes = reconcile(a, y, pd.Series(0.0, a.index))
    assert fixes == []
    assert fixed.equals(a)


def test_price_checks_catch_broken_rows():
    from tradetool.data.checks import price_problems

    good = _bars(FLAT)
    assert price_problems(good) == []
    bad = good.copy()
    bad.iloc[2, bad.columns.get_loc("high")] = 50.0      # high below the close
    bad.iloc[5, bad.columns.get_loc("volume")] = 0
    problems = " ".join(price_problems(bad))
    assert "high below" in problems and "zero volume" in problems


def test_download_without_keys_uses_yahoo(tmp_path, monkeypatch):
    """A fresh clone has no .env: prices must still come (from Yahoo), not fail."""
    from tradetool.data import prices

    monkeypatch.setattr(prices, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(prices, "MANIFEST", tmp_path / "_manifest.csv")
    monkeypatch.setattr(prices, "FIXES", tmp_path / "_fixes.csv")
    monkeypatch.setattr(prices, "has_alpaca_keys", lambda: False)
    monkeypatch.setattr(prices, "fetch_alpaca", lambda *a, **k: pytest.fail("Alpaca called without keys"))
    monkeypatch.setattr(prices, "fetch_yahoo", lambda sym, start=None: _bars(FLAT))
    s = prices.download(["SPY", "AAPL"])
    assert set(s["source"]) == {"yahoo"}
    assert (tmp_path / "AAPL.parquet").exists() and (tmp_path / "SPY.parquet").exists()
