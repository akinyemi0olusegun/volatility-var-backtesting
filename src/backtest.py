"""Backtesting: exceedances, Kupiec POF, Christoffersen, Basel traffic light.

Everything here consumes the walk-forward output from :mod:`src.var`
(a DataFrame with an ``actual`` column and one ``var_{conf}`` column) and turns
it into the statistics a model-validation team reports.

A breach on day t is  ``actual_t < -VaR_t``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Exceedances
# ===========================================================================
def exceedances(actual: pd.Series, var: pd.Series) -> pd.Series:
    """Boolean series: True on days the realised loss exceeded the VaR forecast."""
    a, v = actual.align(var, join="inner")
    return a < -v


# ===========================================================================
# Kupiec Proportion-of-Failures (unconditional coverage) test
# ===========================================================================
def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec POF likelihood-ratio test.

    Parameters
    ----------
    n : number of observations
    x : number of exceedances (failures)
    p : expected exceedance probability = 1 - confidence

    Returns
    -------
    (LR_pof, p_value) with the LR statistic ~ chi-square(1) under H0 that the
    true exceedance rate equals ``p``.
    """
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n

    # log-likelihood under H0 (rate = p)
    ll_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
    # log-likelihood under H1 (rate = pi), with 0*log(0) := 0
    ll_alt = 0.0
    if x > 0:
        ll_alt += x * np.log(pi)
    if n - x > 0:
        ll_alt += (n - x) * np.log(1.0 - pi)

    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


# ===========================================================================
# Christoffersen independence & conditional-coverage tests
# ===========================================================================
def christoffersen_independence(breaches: Sequence[bool]) -> tuple[float, float]:
    """Christoffersen (1998) independence test -- do breaches cluster?

    Tests H0: breach on day t is independent of a breach on day t-1, against a
    first-order Markov alternative.  LR ~ chi-square(1).
    """
    b = np.asarray(breaches, dtype=int)
    if b.size < 2:
        return float("nan"), float("nan")

    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _term(count, prob):
        return count * np.log(prob) if (count > 0 and prob > 0) else 0.0

    ll_null = _term(n00 + n10, 1.0 - pi) + _term(n01 + n11, pi)
    ll_alt = (
        _term(n00, 1.0 - pi01) + _term(n01, pi01)
        + _term(n10, 1.0 - pi11) + _term(n11, pi11)
    )
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def christoffersen_cc(n: int, x: int, p: float, breaches: Sequence[bool]) -> tuple[float, float]:
    """Conditional-coverage test: LR_cc = LR_pof + LR_ind ~ chi-square(2)."""
    lr_pof, _ = kupiec_pof(n, x, p)
    lr_ind, _ = christoffersen_independence(breaches)
    if np.isnan(lr_ind):
        return float("nan"), float("nan")
    lr_cc = lr_pof + lr_ind
    return float(lr_cc), float(1.0 - stats.chi2.cdf(lr_cc, df=2))


# ===========================================================================
# Basel traffic light (99% VaR, 250-day window)
# ===========================================================================
# Basel thresholds are defined for a 250-observation window at 99% VaR.
BASEL_GREEN_MAX = 4     # 0-4 exceptions
BASEL_YELLOW_MAX = 9    # 5-9 exceptions; 10+ is red


def basel_zone(exceptions: int) -> str:
    """Map an exception count (over 250 days, 99% VaR) to a Basel zone."""
    if exceptions <= BASEL_GREEN_MAX:
        return "green"
    if exceptions <= BASEL_YELLOW_MAX:
        return "yellow"
    return "red"


def traffic_light(breaches: pd.Series, window: int = 250) -> dict:
    """Basel traffic-light assessment over trailing 250-day windows.

    Returns the zone for the most recent 250-day window (the standard supervisory
    read), plus the worst zone seen anywhere in the sample and the max exceptions
    in any window -- useful colour for the README.
    """
    b = breaches.astype(int)
    if len(b) < window:
        exc = int(b.sum())
        return {"zone": basel_zone(exc), "recent_exceptions": exc,
                "worst_zone": basel_zone(exc), "max_window_exceptions": exc}

    rolling_exc = b.rolling(window).sum().dropna()
    recent = int(rolling_exc.iloc[-1])
    max_exc = int(rolling_exc.max())
    order = {"green": 0, "yellow": 1, "red": 2}
    worst = max((basel_zone(int(v)) for v in rolling_exc), key=lambda z: order[z])
    return {
        "zone": basel_zone(recent),
        "recent_exceptions": recent,
        "worst_zone": worst,
        "max_window_exceptions": max_exc,
    }


# ===========================================================================
# Per-(model, asset, confidence) summary row
# ===========================================================================
@dataclass
class BacktestResult:
    asset: str
    model: str
    confidence: float
    n_obs: int
    exceptions: int
    breach_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_p: float
    kupiec_pass: bool            # True = fail-to-reject H0 (model OK) at 5%
    christoffersen_ind_p: float
    cc_p: float
    traffic_light: str           # only meaningful at 99%; "n/a" otherwise

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    wf: pd.DataFrame,
    asset: str,
    model: str,
    conf: float,
    significance: float = 0.05,
) -> BacktestResult:
    """Full backtest of one walk-forward VaR series at one confidence level."""
    actual = wf["actual"]
    var = wf[f"var_{conf:g}"]
    br = exceedances(actual, var)

    n = int(len(br))
    x = int(br.sum())
    p = 1.0 - conf
    lr_pof, p_pof = kupiec_pof(n, x, p)
    _, p_ind = christoffersen_independence(br.to_numpy())
    _, p_cc = christoffersen_cc(n, x, p, br.to_numpy())

    tl = traffic_light(br)["zone"] if abs(conf - 0.99) < 1e-9 else "n/a"

    return BacktestResult(
        asset=asset,
        model=model,
        confidence=conf,
        n_obs=n,
        exceptions=x,
        breach_rate=x / n if n else float("nan"),
        expected_rate=p,
        kupiec_lr=lr_pof,
        kupiec_p=p_pof,
        kupiec_pass=bool(p_pof >= significance),
        christoffersen_ind_p=p_ind,
        cc_p=p_cc,
        traffic_light=tl,
    )


def summary_table(
    results_by_asset: dict[str, dict[str, pd.DataFrame]],
    conf_levels: Sequence[float] = (0.95, 0.99),
    significance: float = 0.05,
) -> pd.DataFrame:
    """Build the master summary table: (asset x model x confidence) -> stats.

    ``results_by_asset[asset][model]`` is a walk-forward DataFrame from
    :func:`src.var.walk_forward_var`.
    """
    rows = []
    for asset, by_model in results_by_asset.items():
        for model, wf in by_model.items():
            for conf in conf_levels:
                rows.append(evaluate(wf, asset, model, conf, significance).as_dict())
    df = pd.DataFrame(rows)
    return df.sort_values(["asset", "confidence", "model"]).reset_index(drop=True)
