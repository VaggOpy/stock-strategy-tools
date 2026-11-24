"""Get all the data the app and the experiments need, in one command:

    python -m tradetool.data.setup          prices + fundamentals snapshot
    python -m tradetool.data.setup --sec    fundamentals straight from the SEC (needs SEC_USER_AGENT)

Prices come from Alpaca when its keys are in .env, otherwise from Yahoo. They cannot be shared as files
(the data providers do not allow redistribution), so they are always downloaded. The SEC fundamentals
are public and come as a ready snapshot.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile

import requests

from tradetool.config import FUNDAMENTALS_DIR, FUNDAMENTALS_SNAPSHOT
from tradetool.data import fundamentals, prices
from tradetool.data.universe import load_universe


def get_snapshot(url=FUNDAMENTALS_SNAPSHOT):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if "/" not in n and not n.startswith(".")]
        z.extractall(FUNDAMENTALS_DIR, members=names)
    return len(names)


def make_snapshot(path):
    files = sorted(FUNDAMENTALS_DIR.glob("*.parquet")) + [FUNDAMENTALS_DIR / "_summary.csv"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    return len(files)


def main(args):
    if "--make-snapshot" in args:
        n = make_snapshot(FUNDAMENTALS_DIR.parent / "fundamentals.zip")
        print(f"{n} files in {FUNDAMENTALS_DIR.parent / 'fundamentals.zip'}")
        return

    u = load_universe()
    t0 = time.time()
    source = "Alpaca, checked against Yahoo" if prices.has_alpaca_keys() else "Yahoo (no Alpaca keys in .env)"
    print(f"prices for {len(u)} symbols from {source}...")
    s = prices.download(u["ticker"].tolist())
    print(f"  done in {time.time() - t0:.0f}s: {s['source'].value_counts().to_dict()}")

    stocks = u.query("type == 'stock'")["ticker"].tolist()
    if "--sec" in args:
        print(f"fundamentals for {len(stocks)} stocks from the SEC (about 17 minutes)...")
        fundamentals.download(stocks)
    else:
        print("fundamentals snapshot...")
        print(f"  {get_snapshot()} files")


if __name__ == "__main__":
    main(sys.argv[1:])
