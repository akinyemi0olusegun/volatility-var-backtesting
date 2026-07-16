"""Unit tests for regime-conditional validation.

Key invariant: regimes must **partition** the evaluation window -- every day
lands in exactly one regime, so per-regime counts sum back to the totals.
"""

import numpy as np
import pandas as pd
import pytest

from src import regimes


@pytest.fixture
def eval_index():
    # daily calendar spanning all five calendar regimes
    return pd.bdate_range("2017-06-01", "2025-12-30")


@pytest.fixture
def fake_wf(eval_index):
    """A synthetic walk-forward frame with a constant VaR and known breaches."""
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, len(eval_index)), index=eval_index)
    var99 = pd.Series(0.02, index=eval_index)
    var95 = pd.Series(0.015, index=eval_index)
    return pd.DataFrame({"actual": ret, "var_0.99": var99, "var_0.95": var95})


# ---------------------------------------------------------------------------
# Calendar regimes partition the sample (no gaps, no overlaps)
# ---------------------------------------------------------------------------
def test_calendar_regime_covers_every_day(eval_index):
    labels = pd.Series(regimes.assign_calendar_regime(eval_index))
    assert labels.notna().all()                     # no day falls through a gap
    assert set(labels.unique()) <= set(regimes.CALENDAR_ORDER)


def test_calendar_boundaries_are_contiguous():
    # each regime's end is the day before the next regime's start
    for (_, _, end), (_, start, _) in zip(regimes.CALENDAR_REGIMES[:-1],
                                          regimes.CALENDAR_REGIMES[1:]):
        assert pd.Timestamp(start) == pd.Timestamp(end) + pd.Timedelta(days=1)


def test_per_regime_counts_sum_to_total(fake_wf):
    labels = regimes.assign_calendar_regime(fake_wf.index)
    stats = regimes.per_regime_stats(fake_wf, 0.99, labels, regimes.CALENDAR_ORDER)
    total_breaches = int((fake_wf["actual"] < -fake_wf["var_0.99"]).sum())
    assert stats["exceptions"].sum() == total_breaches
    assert stats["n_days"].sum() == len(fake_wf)


# ---------------------------------------------------------------------------
# Volatility-state regimes
# ---------------------------------------------------------------------------
def test_vol_state_labels_and_partition(eval_index):
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0, 0.01, len(eval_index)), index=eval_index)
    state = pd.Series(regimes.assign_vol_state(ret, eval_index))
    assert set(state.dropna().unique()) <= set(regimes.VOL_STATE_ORDER)
    # terciles -> each state is a non-trivial share of the sample
    counts = state.value_counts(normalize=True)
    assert counts.min() > 0.15


def test_vol_state_high_has_higher_realised_vol(eval_index):
    # returns whose vol ramps up over time -> "high" state must sit at the fat end
    n = len(eval_index)
    scale = np.linspace(0.005, 0.03, n)
    rng = np.random.default_rng(2)
    ret = pd.Series(rng.normal(0, 1, n) * scale, index=eval_index)
    state = pd.Series(regimes.assign_vol_state(ret, eval_index), index=eval_index)
    realised = ret.rolling(21).std(ddof=1)
    assert realised[state == "high"].mean() > realised[state == "low"].mean()


# ---------------------------------------------------------------------------
# Stability leaderboard sanity
# ---------------------------------------------------------------------------
def test_stability_worst_ge_best():
    # minimal long-form frame: two models, two regimes
    df = pd.DataFrame({
        "asset": ["SPY"] * 4,
        "model": ["a", "a", "b", "b"],
        "regime": regimes.CALENDAR_ORDER[:2] * 2,
        "breach_rate_%": [1.0, 3.0, 1.1, 1.2],
    })
    lb = regimes.stability_leaderboard(df, 0.99, regimes.CALENDAR_ORDER[:2])
    assert (lb["worst_regime_rate_%"] >= lb["best_regime_rate_%"]).all()
    # model 'b' (tighter spread) should rank first on worst-regime rate
    assert lb.iloc[0]["model"] == "b"
