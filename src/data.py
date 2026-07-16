"""Data layer: download adjusted-close prices, cache them, build returns.

Design goals
------------
* Reproducibility: raw prices are cached to ``data/prices.csv`` so every later
  run uses identical inputs without re-downloading (Yahoo throttles heavily).
* No look-ahead is introduced here -- we only assemble the return panel.  All
  windowing / forecasting discipline lives in :mod:`src.var`.

Return conventions
------------------
* For the four ETFs we use **log returns**  r_t = ln(P_t / P_{t-1})  on the
  adjusted close (adjusted close already folds in dividends & splits).
* The **equal-weight (EW) portfolio** is a daily-rebalanced 25/25/25/25 basket.
  Log returns are not additive across assets, so the portfolio is built in
  simple-return space --  r_p = mean(simple returns)  -- and then converted back
  to a log return  ln(1 + r_p)  so it lives on the same scale as the single
  names.  This is the correct, standard construction for an equal-weight book.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

TICKERS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD")
PORTFOLIO_COL = "EW"
START = "2015-01-01"
END = "2025-12-31"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
_PRICES_CSV = os.path.join(_DATA_DIR, "prices.csv")


def _download_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices from Yahoo via yfinance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf  # imported lazily so tests don't require network

    tickers = list(tickers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,   # keep an explicit 'Adj Close' column
            group_by="column",
        )

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "yfinance returned no data (Yahoo is likely throttling this IP). "
            "Retry later, or drop a prices.csv into the data/ folder."
        )

    # With multiple tickers yfinance returns a column MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"].copy()
    else:  # single ticker
        prices = raw[["Adj Close"]].copy()
        prices.columns = tickers

    prices = prices[list(tickers)]           # stable column order
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.dropna(how="all").sort_index()
    return prices


def load_prices(
    tickers: Iterable[str] = TICKERS,
    start: str = START,
    end: str = END,
    force_download: bool = False,
    cache_path: str = _PRICES_CSV,
) -> pd.DataFrame:
    """Load adjusted-close prices, using the on-disk cache when available.

    Parameters
    ----------
    force_download : if True, ignore the cache and re-download.
    cache_path     : where the CSV cache lives (default ``data/prices.csv``).
    """
    tickers = list(tickers)
    if not force_download and os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if set(tickers).issubset(prices.columns):
            return prices[tickers].sort_index()

    prices = _download_prices(tickers, start, end)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_csv(cache_path)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns for each price column (first row dropped)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def build_returns(
    tickers: Iterable[str] = TICKERS,
    start: str = START,
    end: str = END,
    force_download: bool = False,
    cache_path: str = _PRICES_CSV,
) -> pd.DataFrame:
    """Return the full log-return panel: one column per ETF plus ``EW``.

    The EW portfolio is a daily-rebalanced equal-weight basket of the ETFs,
    expressed as a log return (see module docstring).
    """
    tickers = list(tickers)
    prices = load_prices(tickers, start, end, force_download, cache_path)

    log_ret = compute_log_returns(prices)

    # Equal-weight portfolio, built in simple-return space then log-transformed.
    simple_ret = prices.pct_change().dropna(how="all")
    ew_simple = simple_ret[tickers].mean(axis=1)
    log_ret[PORTFOLIO_COL] = np.log1p(ew_simple)

    # Drop any residual all-NaN / partial rows so every column is aligned.
    return log_ret.dropna()


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    r = build_returns()
    print(f"Loaded {len(r)} daily returns, {r.index[0].date()} -> {r.index[-1].date()}")
    print(r.describe().T[["mean", "std", "min", "max"]])
