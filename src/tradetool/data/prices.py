"""Daily prices from Alpaca (main source), with Yahoo as a fallback. Cached as one parquet per ticker.

Adjusted prices change backwards in time after every split or dividend, so the cache is not
appended to: a refresh downloads the full history again (about 2 minutes for ~540 tickers).

Columns: open, high, low, close, volume (split and dividend adjusted) and close_raw (as traded).
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


def download(symbols: list[str], start=START) -> pd.DataFrame:
    """Download (or re-download) symbols and write the cache. Returns a summary per symbol."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    adj = fetch_alpaca(symbols, start, adjustment="all")
    raw = fetch_alpaca(symbols, start, adjustment="raw")

    summary = []
    for sym in symbols:
        if sym in adj:
            df = adj[sym].join(raw[sym]["close"].rename("close_raw"))
            source = "alpaca"
        else:
            df = fetch_yahoo(sym, start)
            source = "yahoo" if not df.empty else "missing"
        n = len(df)
        df = clean(df) if n else df
        if not df.empty:
            df.to_parquet(_path(sym))
        summary.append({"ticker": sym, "source": source, "rows": len(df), "dropped": n - len(df),
                        "first": df.index.min().date() if len(df) else None,
                        "last": df.index.max().date() if len(df) else None,
                        "downloaded": dt.date.today()})

    summary = pd.DataFrame(summary)
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST)
        summary = pd.concat([old[~old["ticker"].isin(summary["ticker"])], summary])
    summary.sort_values("ticker").to_csv(MANIFEST, index=False)
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
