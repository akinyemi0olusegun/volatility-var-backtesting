"""Regime-conditional validation.

The aggregate backtest in :mod:`src.backtest` answers "is the model calibrated
*on average* over 2015-2025?".  That can hide the truth: a model can look fine
overall while failing badly in a crisis.  This module slices the out-of-sample
period into **regimes** and re-runs the statistics *within each regime*, so we
can compare across all of them and measure **stability** -- does the breach rate
stay near target everywhere, or blow up in stress?

Two complementary definitions of "regime":

1. **Calendar / event regimes** -- hand-labelled market periods (pre-COVID calm,
   the COVID crash, the 2022 rate-hike bear, etc.).  Interpretable, narrative.
2. **Volatility-state regimes** -- data-driven, per asset: each day is tagged
   ``low`` / ``normal`` / ``high`` volatility by the terciles of its trailing
   21-day realised volatility.  Objective; directly tests calm-vs-stress
   calibration.  (This is an *ex-post* partition of realised outcomes for
   analysis -- it does not feed back into any forecast, so it introduces no
   look-ahead into the VaR itself.)

All functions consume the walk-forward DataFrames produced by
:func:`src.var.walk_forward_var` (an ``actual`` column plus ``var_{conf}``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .backtest import exceedances, kupiec_pof

TRADING_DAYS = 252

# --- Calendar regimes (start inclusive, end inclusive; None = open end) -----
# Boundaries chosen around well-known market events.  The walk-forward eval
# window only begins ~2017 (after the 500-day burn-in), so "pre-COVID" here
# means the calm 2017-2019 stretch.
# NOTE: boundaries are contiguous (no gaps) so every evaluation day lands in
# exactly one regime -- enforced by tests/test_regimes.py.
CALENDAR_REGIMES: list[tuple[str, str | None, str | None]] = [
    ("1. Pre-COVID calm",     None,         "2020-02-19"),
    ("2. COVID shock (2020)", "2020-02-20", "2020-12-31"),
    ("3. 2021 low-vol bull",  "2021-01-01", "2021-12-31"),
    ("4. 2022 rates bear",    "2022-01-01", "2022-12-31"),
    ("5. 2023-25 recovery",   "2023-01-01", None),
]
CALENDAR_ORDER = [name for name, _, _ in CALENDAR_REGIMES]

VOL_STATE_ORDER = ["low", "normal", "high"]


# ===========================================================================
# Regime label assignment
# ===========================================================================
def assign_calendar_regime(index: pd.DatetimeIndex) -> pd.Series:
    """Map each date to its calendar regime (ordered categorical)."""
    labels = pd.Series(index=index, dtype="object")
    for name, start, end in CALENDAR_REGIMES:
        mask = pd.Series(True, index=index)
        if start is not None:
            mask &= index >= pd.Timestamp(start)
        if end is not None:
            mask &= index <= pd.Timestamp(end)
        labels[mask.to_numpy()] = name
    return pd.Categorical(labels, categories=CALENDAR_ORDER, ordered=True)


def assign_vol_state(
    returns: pd.Series,
    eval_index: pd.DatetimeIndex,
    window: int = 21,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> pd.Series:
    """Tag each evaluation day low / normal / high by trailing realised vol.

    ``returns`` is the *full* return series (so the trailing window is available
    before ``eval_index`` starts); the result is restricted to ``eval_index``.
    Terciles are taken over the evaluation window only.
    """
    realised = returns.rolling(window).std(ddof=1)
    realised = realised.reindex(eval_index)
    lo, hi = realised.quantile(quantiles[0]), realised.quantile(quantiles[1])
    state = pd.Series("normal", index=eval_index, dtype="object")
    state[realised <= lo] = "low"
    state[realised > hi] = "high"
    return pd.Categorical(state, categories=VOL_STATE_ORDER, ordered=True)


# ===========================================================================
# Per-regime statistics
# ===========================================================================
def per_regime_stats(
    wf: pd.DataFrame,
    conf: float,
    regime_labels,
    regime_order: Sequence[str],
) -> pd.DataFrame:
    """Backtest + descriptive stats for one walk-forward series, by regime.

    Columns: regime, n_days, mean_ret_%, vol_ann_%, worst_day_%, exceptions,
    breach_rate_%, expected_%, kupiec_p.
    """
    actual = wf["actual"]
    var = wf[f"var_{conf:g}"]
    breach = exceedances(actual, var).reindex(actual.index)
    frame = pd.DataFrame({
        "ret": actual.to_numpy(),
        "breach": breach.to_numpy(),
        "regime": pd.Categorical(regime_labels, categories=list(regime_order), ordered=True),
    }, index=actual.index)

    rows = []
    for reg in regime_order:
        g = frame[frame["regime"] == reg]
        n = int(len(g))
        if n == 0:
            continue
        x = int(g["breach"].sum())
        _, p = kupiec_pof(n, x, 1.0 - conf)
        rows.append({
            "regime": reg,
            "n_days": n,
            "mean_ret_%": g["ret"].mean() * 100,
            "vol_ann_%": g["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS) * 100,
            "worst_day_%": g["ret"].min() * 100,
            "exceptions": x,
            "breach_rate_%": x / n * 100,
            "expected_%": (1.0 - conf) * 100,
            "kupiec_p": p,
        })
    return pd.DataFrame(rows)


def regime_table(
    results: dict[str, dict[str, pd.DataFrame]],
    returns: pd.DataFrame,
    conf: float,
    kind: str = "calendar",
) -> pd.DataFrame:
    """Long table of per-regime stats for every (asset, model, regime).

    ``kind`` is ``"calendar"`` or ``"volstate"``.
    """
    if kind not in ("calendar", "volstate"):
        raise ValueError("kind must be 'calendar' or 'volstate'")
    order = CALENDAR_ORDER if kind == "calendar" else VOL_STATE_ORDER

    out = []
    for asset, by_model in results.items():
        eval_index = next(iter(by_model.values())).index
        if kind == "calendar":
            labels = assign_calendar_regime(eval_index)
        else:
            labels = assign_vol_state(returns[asset], eval_index)
        for model, wf in by_model.items():
            stats = per_regime_stats(wf, conf, labels, order)
            stats.insert(0, "model", model)
            stats.insert(0, "asset", asset)
            out.append(stats)
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# Cross-regime stability
# ===========================================================================
def breach_rate_matrix(regime_long: pd.DataFrame, regime_order: Sequence[str]) -> pd.DataFrame:
    """Model x regime matrix of breach rate (%), averaged across assets."""
    piv = regime_long.pivot_table(
        index="model", columns="regime", values="breach_rate_%", aggfunc="mean", observed=False
    )
    cols = [c for c in regime_order if c in piv.columns]
    return piv[cols]


def stability_leaderboard(
    regime_long: pd.DataFrame,
    conf: float,
    regime_order: Sequence[str],
) -> pd.DataFrame:
    """Rank models by cross-regime stability of the breach rate.

    For each model:
      * ``worst_regime_rate_%``  -- highest cross-asset mean breach rate in any
        single regime (the true worst case),
      * ``regime_spread_%``      -- max - min of those per-regime mean rates,
      * ``regime_std_%``         -- std of the per-regime mean rates,
      * ``worst_cell_rate_%``    -- highest breach rate in any single
        (asset, regime) cell -- the most brutal stress point.
    Lower is better on every column.  Ranked by worst-regime rate.
    """
    expected = (1.0 - conf) * 100
    mat = breach_rate_matrix(regime_long, regime_order)
    rows = []
    for model, r in mat.iterrows():
        cells = regime_long[regime_long["model"] == model]["breach_rate_%"]
        rows.append({
            "model": model,
            "expected_%": expected,
            "worst_regime_rate_%": float(r.max()),
            "best_regime_rate_%": float(r.min()),
            "regime_spread_%": float(r.max() - r.min()),
            "regime_std_%": float(r.std(ddof=0)),
            "worst_cell_rate_%": float(cells.max()),
        })
    return pd.DataFrame(rows).sort_values(
        ["worst_regime_rate_%", "regime_spread_%"]
    ).reset_index(drop=True)
