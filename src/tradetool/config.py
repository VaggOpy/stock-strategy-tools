"""Paths and settings. API keys are read from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"  # local cache, not in git
PRICES_DIR = DATA_DIR / "prices"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
DOCS_DIR = ROOT / "docs"

load_dotenv(ROOT / ".env")


def env(name: str, required: bool = False) -> str | None:
    """Read a setting from .env / the environment. Empty values count as missing."""
    value = os.getenv(name) or None
    if required and value is None:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value

# Fixed period for experiments, so results don't change when new data is downloaded.
# Prices start 2016-01-04, the first year is warm-up for the 200-day indicators.
TEST_START = "2017-01-03"
TEST_END = "2026-10-30"

# SEC fundamentals as downloaded for the experiments (public data), so a fresh clone does not need
# 17 minutes of SEC requests. Made with: python -m tradetool.data.setup --make-snapshot
FUNDAMENTALS_SNAPSHOT = ("https://github.com/VaggOpy/stock-strategy-tools/releases/download/"
                         "data-2026-11-24/fundamentals.zip")
