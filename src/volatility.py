"""Volatility estimators, in increasing sophistication.

1. ``rolling_std``   -- rolling-window sample standard deviation (the baseline).
2. ``ewma_vol``      -- RiskMetrics exponentially-weighted moving average.
3. ``garch_forecast``-- one-step GARCH(1,1) conditional volatility via ``arch``.

All functions return **daily** volatilities (standard deviations) in return
(decimal) space, aligned to the input index.

A note on GARCH scaling
-----------------------
GARCH optimisers are numerically happier when the data are O(1) rather than
O(0.01).  We therefore fit on returns expressed in **percent** (``*100``) and
divide the resulting sigma back by 100.  This is purely a conditioning trick and
does not affect the model.

In-sample vs. out-of-sample
---------------------------
The functions here fit on whatever data you hand them.  ``run_analysis.py`` uses
the *full-sample* fit only for the "conditional vol vs. realised vol" picture.
The VaR **backtest** never uses these full-sample fits -- it uses the strict
walk-forward engine in :mod:`src.var`, which only ever sees data up to t-1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ---- RiskMetrics constant -------------------------------------------------
RISKMETRICS_LAMBDA = 0.94


def rolling_std(returns: pd.Series, window: int = 250) -> pd.Series:
    """Rolling-window sample standard deviation.

    Value at index t uses the ``window`` observations ending at t (inclusive).
    For use as a *forecast* of day t+1, callers must shift by one (the
    walk-forward engine handles this); as an in-sample descriptive estimator it
    is returned un-shifted here.
    """
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def ewma_vol(returns: pd.Series, lam: float = RISKMETRICS_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA conditional volatility.

    Variance recursion (zero-mean assumption, as in RiskMetrics):

        sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}

    The series is seeded with the sample variance of the first observations and
    returned as sigma_t (a *forecast* for day t built from information through
    t-1), so it is directly comparable across the sample.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")

    r = returns.to_numpy(dtype=float)
    n = len(r)
    var = np.empty(n)

    # Seed with the variance of the first min(30, n) squared returns.
    seed = min(30, n)
    var[0] = np.mean(r[:seed] ** 2)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2

    return pd.Series(np.sqrt(var), index=returns.index, name="ewma_vol")


def _fit_garch(returns_pct: np.ndarray, dist: str = "normal", params=None):
    """Fit (or fix) a constant-mean GARCH(1,1) on percent returns.

    If ``params`` is given, parameters are *fixed* (no optimisation) and only
    the conditional-variance recursion is filtered -- this is the cheap path
    used between periodic re-estimations in the walk-forward.
    """
    from arch import arch_model

    am = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if params is None:
            res = am.fit(disp="off", show_warning=False)
        else:
            res = am.fix(params)
    return res


def garch_forecast(returns: pd.Series, dist: str = "normal") -> pd.Series:
    """Full-sample in-sample conditional volatility from GARCH(1,1).

    Convenience wrapper used for the descriptive vol plots.  Returns sigma_t in
    decimal (return) space aligned to ``returns.index``.
    """
    r_pct = returns.to_numpy(dtype=float) * 100.0
    res = _fit_garch(r_pct, dist=dist)
    cond_vol_pct = np.asarray(res.conditional_volatility, dtype=float)
    return pd.Series(cond_vol_pct / 100.0, index=returns.index, name=f"garch_{dist}_vol")


def garch_params(returns: pd.Series, dist: str = "normal") -> pd.Series:
    """Return the fitted GARCH(1,1) parameter vector (for reporting)."""
    r_pct = returns.to_numpy(dtype=float) * 100.0
    res = _fit_garch(r_pct, dist=dist)
    return res.params
