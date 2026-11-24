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
