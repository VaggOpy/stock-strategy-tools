import numpy as np
import pandas as pd


def random_prices(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.015, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.015, n))
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)
