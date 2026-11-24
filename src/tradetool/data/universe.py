"""List of tickers we work with: S&P 500 + Nasdaq-100 + a set of ETFs.

The lists come from Wikipedia and are the *current* members, so anything built on them has
survivorship bias (companies that dropped out of the index are missing). The snapshot is saved
to universe.csv with its date so results can be reproduced.
"""
from __future__ import annotations

import datetime as dt
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UNIVERSE_FILE = Path(__file__).with_name("universe.csv")

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

ETFS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000", "DIA": "Dow Jones",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services",
    "SMH": "Semiconductors", "IBB": "Biotech", "KRE": "Regional Banks", "ITB": "Home Builders",
    "ARKK": "Innovation", "GLD": "Gold", "TLT": "20+ Year Treasuries",
}


def _tables(url: str) -> list[pd.DataFrame]:
    html = requests.get(url, headers={"User-Agent": "stock-strategy-tools/0.1"}, timeout=30).text
    return pd.read_html(StringIO(html))


def _find(tables, column):
    return next(t for t in tables if column in t.columns)


def fetch_universe() -> pd.DataFrame:
    sp = _find(_tables(SP500_URL), "GICS Sector")
    sp = sp.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                            "GICS Sub-Industry": "industry"})[["ticker", "name", "sector", "industry"]]
    sp["in_sp500"] = True

    ndx = _find(_tables(NDX_URL), "Ticker")
    # columns are "ICB Industry[2]" etc., footnote numbers change over time
    ndx.columns = [c.split("[")[0] for c in ndx.columns]
    ndx = ndx.rename(columns={"Ticker": "ticker", "Company": "name", "ICB Industry": "sector",
                              "ICB Subsector": "industry"})[["ticker", "name", "sector", "industry"]]
    ndx["in_ndx"] = True

    stocks = sp.merge(ndx, on="ticker", how="outer", suffixes=("", "_ndx"))
    for c in ["name", "sector", "industry"]:
        stocks[c] = stocks[c].fillna(stocks.pop(f"{c}_ndx"))
    stocks["type"] = "stock"

    etfs = pd.DataFrame({"ticker": list(ETFS), "name": list(ETFS.values()), "type": "etf"})
    out = pd.concat([stocks, etfs], ignore_index=True)
    out[["in_sp500", "in_ndx"]] = out[["in_sp500", "in_ndx"]].fillna(False).astype(bool)
    out["ticker"] = out["ticker"].str.strip().str.upper()
    out["snapshot"] = dt.date.today().isoformat()
    return out.drop_duplicates("ticker").sort_values(["type", "ticker"]).reset_index(drop=True)


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        fetch_universe().to_csv(UNIVERSE_FILE, index=False)
    return pd.read_csv(UNIVERSE_FILE)


if __name__ == "__main__":
    u = fetch_universe()
    u.to_csv(UNIVERSE_FILE, index=False)
    print(u["type"].value_counts().to_string())
    print("S&P 500:", u["in_sp500"].sum(), " Nasdaq-100:", u["in_ndx"].sum(),
          " both:", (u["in_sp500"] & u["in_ndx"]).sum())
    print(u.sample(5, random_state=0).to_string())
