"""Value-at-Risk / Expected-Shortfall calculators and the walk-forward engine.

Sign & reporting conventions
----------------------------
* Returns are daily log returns.  A **loss** is a negative return.
* VaR is reported as a **positive number** = the loss threshold that should be
  breached only ``(1 - confidence)`` of the time.  Example: a 99% VaR of 0.023
  means "we are 99% confident the daily loss will not exceed 2.3%".
* A **breach / exceedance** on day t occurs when  ``return_t < -VaR_t``  (the
  realised loss is larger than the forecast VaR).
* Daily drift is set to **zero** for the parametric methods.  For a 1-day
  horizon the mean (~0.02%/day) is negligible next to the volatility (~1%/day),
  and zero-mean is the RiskMetrics / regulatory convention.  This keeps the
  models directly comparable.

No look-ahead
-------------
:func:`walk_forward_var` guarantees every VaR forecast for day t is built only
from returns strictly before t.  At each step it takes the trailing ``window``
returns [t-window, t-1], produces sigma / quantiles from that block, and scores
the forecast against the *held-out* return on day t.  The engine never touches
day t (or later) when forming the day-t forecast.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .volatility import RISKMETRICS_LAMBDA

# Windows (business days) --------------------------------------------------
ESTIMATION_WINDOW = 500      # history required before the first forecast
ROLLING_VOL_WINDOW = 250     # window for rolling-std & historical simulation
GARCH_REFIT_EVERY = 10       # re-estimate GARCH params every N days (see README)

DEFAULT_CONF_LEVELS = (0.95, 0.99)
DEFAULT_ES_LEVEL = 0.975


# ===========================================================================
# Point calculators (single sigma / single window -> single VaR or ES)
# ===========================================================================
def var_parametric_normal(sigma: float, conf: float, mu: float = 0.0) -> float:
    """1-day parametric VaR under a normal distribution (positive loss)."""
    z = stats.norm.ppf(1.0 - conf)          # negative left-tail quantile
    return float(-(mu + sigma * z))


def _standardized_t_quantile(p: float, nu: float) -> float:
    """Quantile of a Student-t rescaled to **unit variance** (nu > 2)."""
    return float(stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu))


def var_parametric_t(sigma: float, conf: float, nu: float, mu: float = 0.0) -> float:
    """1-day parametric VaR under a (unit-variance) Student-t (positive loss)."""
    q = _standardized_t_quantile(1.0 - conf, nu)
    return float(-(mu + sigma * q))


def var_historical(window_returns: Sequence[float], conf: float) -> float:
    """Historical-simulation VaR: empirical (1-conf) quantile of past returns."""
    q = np.quantile(np.asarray(window_returns, dtype=float), 1.0 - conf)
    return float(-q)


def es_normal(sigma: float, conf: float, mu: float = 0.0) -> float:
    """Expected shortfall (CVaR) under a normal distribution (positive loss)."""
    alpha = 1.0 - conf
    z = stats.norm.ppf(alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / alpha))


def es_t(sigma: float, conf: float, nu: float, mu: float = 0.0) -> float:
    """Expected shortfall under a (unit-variance) Student-t (positive loss)."""
    alpha = 1.0 - conf
    q = stats.t.ppf(alpha, nu)                       # raw-t quantile
    es_raw = (stats.t.pdf(q, nu) / alpha) * (nu + q * q) / (nu - 1.0)
    es_std = es_raw / np.sqrt(nu / (nu - 2.0))       # rescale to unit variance
    return float(-(mu - sigma * es_std))


def es_historical(window_returns: Sequence[float], conf: float) -> float:
    """Historical-simulation ES: mean loss beyond the empirical VaR quantile."""
    r = np.asarray(window_returns, dtype=float)
    alpha = 1.0 - conf
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    if tail.size == 0:
        return float(-q)
    return float(-tail.mean())


# ===========================================================================
# Per-window volatility helpers (operate on the trailing estimation window)
# ===========================================================================
def _ewma_last_sigma(window_returns: np.ndarray, lam: float = RISKMETRICS_LAMBDA) -> float:
    """Final EWMA sigma after running the recursion across the window.

    Returns the forecast for the *next* day (built from the whole window).
    """
    r = window_returns
    seed = min(30, len(r))
    var = np.mean(r[:seed] ** 2)
    for t in range(1, len(r)):
        var = lam * var + (1.0 - lam) * r[t - 1] ** 2
    # one more step to forecast the day *after* the window's last observation
    var = lam * var + (1.0 - lam) * r[-1] ** 2
    return float(np.sqrt(var))


def _garch_one_step(window_pct: np.ndarray, dist: str, params=None):
    """Fit (params is None) or fix GARCH(1,1) on percent returns, forecast 1 step.

    Returns ``(sigma_decimal, params, nu)`` where sigma is next-day volatility in
    decimal (return) space and ``nu`` is the t degrees-of-freedom (or None).
    """
    from arch import arch_model

    am = arch_model(window_pct, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False) if params is None else am.fix(params)

    fc = res.forecast(horizon=1, reindex=False)
    sigma_pct = float(np.sqrt(fc.variance.to_numpy().ravel()[-1]))
    nu = float(res.params["nu"]) if dist == "t" else None
    return sigma_pct / 100.0, res.params, nu


# ===========================================================================
# Walk-forward engine
# ===========================================================================
# Methods this engine understands.
METHODS = ("normal_rolling", "ewma", "historical", "garch_normal", "garch_t")


def walk_forward_var(
    returns: pd.Series,
    methods: Sequence[str] = METHODS,
    conf_levels: Sequence[float] = DEFAULT_CONF_LEVELS,
    es_level: float = DEFAULT_ES_LEVEL,
    window: int = ESTIMATION_WINDOW,
    rolling_vol_window: int = ROLLING_VOL_WINDOW,
    refit_every: int = GARCH_REFIT_EVERY,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the strict walk-forward backtest for one return series.

    Returns
    -------
    dict[method] -> DataFrame indexed by evaluation date with columns:
        ``actual``               -- realised return on day t (held out),
        ``var_{conf}``           -- forecast VaR for each confidence level,
        ``es_{es_level}``        -- forecast Expected Shortfall.
    All methods share the *same* evaluation dates (indices ``window .. n-1``),
    so their exceedance statistics are directly comparable.
    """
    methods = list(methods)
    r = returns.to_numpy(dtype=float)
    idx = returns.index
    n = len(r)
    if n <= window:
        raise ValueError(f"Need more than {window} observations, got {n}.")

    conf_levels = list(conf_levels)
    eval_dates = idx[window:]
    var_cols = [f"var_{c:g}" for c in conf_levels]
    es_col = f"es_{es_level:g}"

    # pre-allocate result arrays per method
    out = {
        m: {c: np.full(n - window, np.nan) for c in var_cols + [es_col, "actual"]}
        for m in methods
    }

    # GARCH state (last fitted params, reused via fix() between re-estimations)
    g_params = {"garch_normal": None, "garch_t": None}
    g_dist = {"garch_normal": "normal", "garch_t": "t"}

    for i, t in enumerate(range(window, n)):
        win = r[t - window:t]                     # returns [t-window, t-1]
        actual = r[t]
        for m in methods:
            out[m]["actual"][i] = actual

        # --- rolling-std parametric normal --------------------------------
        if "normal_rolling" in methods:
            sig = float(np.std(win[-rolling_vol_window:], ddof=1))
            for c, col in zip(conf_levels, var_cols):
                out["normal_rolling"][col][i] = var_parametric_normal(sig, c)
            out["normal_rolling"][es_col][i] = es_normal(sig, es_level)

        # --- EWMA parametric normal ---------------------------------------
        if "ewma" in methods:
            sig = _ewma_last_sigma(win)
            for c, col in zip(conf_levels, var_cols):
                out["ewma"][col][i] = var_parametric_normal(sig, c)
            out["ewma"][es_col][i] = es_normal(sig, es_level)

        # --- historical simulation ----------------------------------------
        if "historical" in methods:
            hwin = win[-rolling_vol_window:]
            for c, col in zip(conf_levels, var_cols):
                out["historical"][col][i] = var_historical(hwin, c)
            out["historical"][es_col][i] = es_historical(hwin, es_level)

        # --- GARCH (normal and/or Student-t) ------------------------------
        for m in ("garch_normal", "garch_t"):
            if m not in methods:
                continue
            win_pct = win * 100.0
            refit = (g_params[m] is None) or (i % refit_every == 0)
            try:
                sig, params, nu = _garch_one_step(
                    win_pct, g_dist[m], params=None if refit else g_params[m]
                )
                g_params[m] = params
            except Exception:
                # optimiser hiccup: fall back to previous params, else EWMA sigma
                if g_params[m] is not None:
                    sig, _, nu = _garch_one_step(win_pct, g_dist[m], params=g_params[m])
                else:
                    sig, nu = _ewma_last_sigma(win), (8.0 if m == "garch_t" else None)
            for c, col in zip(conf_levels, var_cols):
                if m == "garch_t":
                    out[m][col][i] = var_parametric_t(sig, c, nu)
                else:
                    out[m][col][i] = var_parametric_normal(sig, c)
            out[m][es_col][i] = (
                es_t(sig, es_level, nu) if m == "garch_t" else es_normal(sig, es_level)
            )

        if progress and (i % 250 == 0):
            print(f"    {returns.name}: {i}/{n - window} days", flush=True)

    return {
        m: pd.DataFrame(out[m], index=eval_dates)[["actual"] + var_cols + [es_col]]
        for m in methods
    }
