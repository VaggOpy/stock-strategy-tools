"""Daily prices from Alpaca (main source), checked against Yahoo, which is also the fallback.
Cached as one parquet per ticker.

Adjusted prices change backwards in time after every split or dividend, so the cache is not
appended to: a refresh downloads the full history again (about 6 minutes for ~540 tickers, with the Yahoo check).

Columns: open, high, low, close, volume (split and dividend adjusted) and close_raw (as traded).
Without Alpaca keys everything comes from Yahoo, and close_raw is then only split adjusted.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from tradetool.config import PRICES_DIR, env

ALPACA_BARS = "https://data.alpaca.markets/v2/stocks/bars"
START = "2016-01-01"  # Alpaca SIP history starts here
MANIFEST = PRICES_DIR / "_manifest.csv"


def has_alpaca_keys() -> bool:
    return bool(env("ALPACA_API_KEY_ID") and env("ALPACA_API_SECRET_KEY"))


def _alpaca_headers():
    return {"APCA-API-KEY-ID": env("ALPACA_API_KEY_ID", required=True),
            "APCA-API-SECRET-KEY": env("ALPACA_API_SECRET_KEY", required=True)}


def _get(params, headers, tries=6):
    for i in range(tries):
        r = requests.get(ALPACA_BARS, params=params, headers=headers, timeout=60)
        if r.status_code == 429:  # rate limit, 200 calls/min on the free plan
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Alpaca rate limit, gave up")


def fetch_alpaca(symbols: list[str], start=START, end=None, adjustment="all") -> dict[str, pd.DataFrame]:
    """Daily bars for many symbols at once. end defaults to yesterday so we never get a half-finished day."""
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    headers = _alpaca_headers()
    rows: dict[str, list] = {}
    for i in range(0, len(symbols), 100):
        params = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": "1Day", "start": start, "end": end,
                  "adjustment": adjustment, "feed": "sip", "limit": 10000}
        while True:
            data = _get(params, headers)
            for sym, bars in data.get("bars", {}).items():
                rows.setdefault(sym, []).extend(bars)
            if not data.get("next_page_token"):
                break
            params["page_token"] = data["next_page_token"]

    out = {}
    for sym, bars in rows.items():
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["t"].str[:10])
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        out[sym] = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
    return out


def fetch_yahoo(symbol: str, start=START) -> pd.DataFrame:
    import yfinance as yf

    h = yf.Ticker(symbol.replace(".", "-")).history(start=start, interval="1d", auto_adjust=False)
    if h.empty:
        return h
    h.index = pd.to_datetime(h.index.date)
    h.index.name = "date"
    # adjust OHLC with the same factor yahoo uses for Adj Close
    f = h["Adj Close"] / h["Close"]
    df = pd.DataFrame({"open": h["Open"] * f, "high": h["High"] * f, "low": h["Low"] * f,
                       "close": h["Adj Close"], "volume": h["Volume"], "close_raw": h["Close"]})
    yesterday = pd.Timestamp(dt.date.today() - dt.timedelta(days=1))
    return df[df.index <= yesterday]


def _path(symbol):
    return PRICES_DIR / f"{symbol}.parquet"


MAX_GAP_DAYS = 30


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """1. Drop days with zero volume. Alpaca fills days when a stock was suspended or traded only
    OTC with the last price and volume 0 (NBIS 2022-24, SMCI 2018-20, EXE 2020-21). Nobody could
    trade at those prices.
    2. If the series has a gap longer than MAX_GAP_DAYS, keep only the part after the last gap.
    A return across such a gap is meaningless, and sometimes the symbol even belongs to another
    company before it (DOW was Dow Chemical until 2017, Dow Inc. from 2019)."""
    df = df[df["volume"] > 0]
    gaps = df.index.to_series().diff().dt.days > MAX_GAP_DAYS
    if gaps.any():
        df = df[df.index >= gaps[gaps].index[-1]]
    return df


DIFF_LIMIT = 0.02   # daily returns of the two sources differing by more than this get checked
MAX_REFEREE = 0.10  # Yahoo's version is only accepted if it is this close to the market move
# Both sources show another company's prices under this symbol before the listing date.
LISTED = {"SW": "2024-07-08"}  # Smurfit WestRock; before it the symbol carries WestRock's history
FIXES = PRICES_DIR / "_fixes.csv"
OHLC = ["open", "high", "low", "close"]


def reconcile(a: pd.DataFrame, y: pd.DataFrame, market: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """Correct Alpaca bars with Yahoo where the two disagree. Returns the bars and a list of fixes.

    Comparing all symbols (experiments/check_adjustments.py) showed two kinds of errors:
    1. A wrong bar: on some days of March 2020 Alpaca's close is not the official close (SPY 255.24
       instead of 248.11 on 2020-03-12). The raw closes of the two sources then differ on that day
       only. The bar is replaced by Yahoo's.
    2. A corporate action (mostly spin-offs) adjusted differently, e.g. RTX -71% on 2020-04-03 in
       Alpaca, -8% in Yahoo. The raw prices agree, only the adjustment differs. We keep the version
       whose return is closer to the market (SPY) that day, but take Yahoo's only if it is within
       MAX_REFEREE of it: Yahoo has its own wrong adjustments (DHR +61% on 2016-07-05). Taking Yahoo's
       version means rescaling everything before that day, like any adjustment does.
    """
    a = a.copy()
    common = a.index.intersection(y.index)
    if len(common) < 3:
        return a, []
    fixes = []
    # raw close ratio, constant except at splits (Yahoo's "Close" is split adjusted, Alpaca's raw is not)
    k = a["close_raw"].reindex(common) / y["close_raw"].reindex(common)
    ra, ry = a["close"].reindex(common).pct_change(), y["close"].reindex(common).pct_change()
    flagged = (ra - ry).abs() > DIFF_LIMIT
    blip = ((k / k.shift(1) - 1).abs() > 0.01) & ((k / k.shift(-1) - 1).abs() > 0.01)
    for d in common[blip & (flagged | flagged.shift(-1, fill_value=False))]:
        prev_k = k.shift(1)[d]
        f_a = a.at[d, "close"] / a.at[d, "close_raw"]
        f_y = y.at[d, "close"] / y.at[d, "close_raw"]
        for col in OHLC:
            a.at[d, col] = y.at[d, col] / f_y * prev_k * f_a
        a.at[d, "close_raw"] = y.at[d, "close_raw"] * prev_k
        fixes.append({"date": d.date(), "kind": "bar", "alpaca": ra[d], "yahoo": ry[d]})

    ra = a["close"].reindex(common).pct_change()
    m = market.reindex(common)
    for d in reversed(common[(ra - ry).abs() > DIFF_LIMIT]):
        err_a, err_y = abs(ra[d] - m[d]), abs(ry[d] - m[d])
        if err_y < err_a and err_y < MAX_REFEREE:
            a.loc[a.index < d, OHLC] *= (1 + ra[d]) / (1 + ry[d])
            fixes.append({"date": d.date(), "kind": "adjustment", "alpaca": ra[d], "yahoo": ry[d]})
    return a, fixes


def _market() -> pd.Series:
    p = _path("SPY")
    return pd.read_parquet(p)["close"].pct_change() if p.exists() else pd.Series(dtype=float)


def download(symbols: list[str], start=START) -> pd.DataFrame:
    """Download (or re-download) symbols and write the cache. Returns a summary per symbol.
    SPY goes first, it is the market reference when the sources are reconciled."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(symbols, key=lambda t: t != "SPY")
    if has_alpaca_keys():
        adj = fetch_alpaca(symbols, start, adjustment="all")
        raw = fetch_alpaca(symbols, start, adjustment="raw")
    else:  # no keys (e.g. someone who just cloned the repo): Yahoo only, nothing to check it against
        adj = raw = {}

    summary, all_fixes = [], []
    for sym in symbols:
        fixes = []
        if sym in adj and sym in raw:
            df = adj[sym].join(raw[sym]["close"].rename("close_raw"))
            source = "alpaca"
            try:
                y = fetch_yahoo(sym, start)
            except Exception:  # yfinance raises many kinds of errors when offline or rate limited
                y = pd.DataFrame()
            if y.empty:
                source = "alpaca (not checked)"
            else:
                market = df["close"].pct_change() if sym == "SPY" else _market()
                df, fixes = reconcile(df, y, market)
        else:
            try:
                df = fetch_yahoo(sym, start)
            except Exception:
                df = pd.DataFrame()
            source = "yahoo" if not df.empty else "missing"
        n = len(df)
        if sym in LISTED:
            df = df[df.index >= pd.Timestamp(LISTED[sym])]
        df = clean(df) if len(df) else df
        if not df.empty:
            df.to_parquet(_path(sym))
            fixes = [f for f in fixes if pd.Timestamp(f["date"]) >= df.index[0]]  # only the part we keep
        all_fixes += [{"ticker": sym, **f} for f in fixes]
        summary.append({"ticker": sym, "source": source, "rows": len(df), "dropped": n - len(df),
                        "fixed_days": len(fixes),
                        "first": df.index.min().date() if len(df) else None,
                        "last": df.index.max().date() if len(df) else None,
                        "downloaded": dt.date.today()})

    summary = pd.DataFrame(summary)
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST)
        summary = pd.concat([old[~old["ticker"].isin(summary["ticker"])], summary])
    summary.sort_values("ticker").to_csv(MANIFEST, index=False)

    fx = pd.DataFrame(all_fixes, columns=["ticker", "date", "kind", "alpaca", "yahoo"])
    if FIXES.exists():
        old = pd.read_csv(FIXES)
        fx = pd.concat([old[~old["ticker"].isin(symbols)], fx])
    fx.sort_values(["ticker", "date"]).to_csv(FIXES, index=False)
    return summary


def _stale(p) -> bool:
    return dt.date.fromtimestamp(p.stat().st_mtime) < dt.date.today()


def load(symbol: str, refresh: bool = False, end=None) -> pd.DataFrame:
    """Prices for one ticker from the cache; downloads it if it isn't there.
    refresh=True also re-downloads it when the file is from an earlier day (used by the app).
    end cuts the data at a fixed date, so experiments give the same result next month."""
    p = _path(symbol)
    if not p.exists() or (refresh and _stale(p)):
        try:
            download([symbol])
        except (requests.RequestException, RuntimeError):
            pass  # offline or no keys: use what is in the cache
    if not p.exists():
        raise KeyError(f"no price data for {symbol}")
    df = pd.read_parquet(p)
    return df[df.index <= pd.Timestamp(end)] if end is not None else df


def cached() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))


if __name__ == "__main__":
    import sys

    from tradetool.data.universe import load_universe

    tickers = sys.argv[1:] or load_universe()["ticker"].tolist()
    t0 = time.time()
    s = download(tickers)
    print(f"{len(s)} tickers in {time.time() - t0:.0f}s")
    print(s["source"].value_counts().to_string())
    short = s[s["rows"] < 2000]
    if len(short):
        print("\nless than ~8 years of data:")
        print(short.to_string(index=False))


def panel(symbols: list[str], field: str = "close", end=None) -> pd.DataFrame:
    """One column per symbol from the cache (dates x symbols). Symbols without a file are left out."""
    cols = {}
    for s in symbols:
        try:
            cols[s] = load(s, end=end)[field]
        except KeyError:
            continue
    return pd.DataFrame(cols).sort_index()
