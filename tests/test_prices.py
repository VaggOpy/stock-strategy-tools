import pandas as pd

from tradetool.data.prices import clean


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
