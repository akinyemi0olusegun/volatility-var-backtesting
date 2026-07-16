"""Unit tests: known-input VaR/ES values, Kupiec/Christoffersen on synthetic
breaches, Basel zones, and an end-to-end synthetic-normal coverage check.

Run with ``pytest`` from the repository root.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import backtest, var, volatility


# ---------------------------------------------------------------------------
# Parametric VaR: known normal quantiles
# ---------------------------------------------------------------------------
def test_parametric_normal_known_values():
    # sigma = 1, mu = 0  ->  VaR = |z_alpha|
    assert var.var_parametric_normal(1.0, 0.95) == pytest.approx(1.6448536, abs=1e-5)
    assert var.var_parametric_normal(1.0, 0.99) == pytest.approx(2.3263479, abs=1e-5)


def test_parametric_normal_scales_with_sigma_and_mu():
    # scales linearly in sigma; a positive drift lowers the loss VaR
    assert var.var_parametric_normal(2.0, 0.99) == pytest.approx(2 * 2.3263479, abs=1e-5)
    base = var.var_parametric_normal(1.0, 0.99, mu=0.0)
    assert var.var_parametric_normal(1.0, 0.99, mu=0.001) == pytest.approx(base - 0.001, abs=1e-9)


def test_student_t_has_fatter_tails_than_normal_at_99():
    # For equal (unit) variance, the t is fatter in the far tail -> larger 99% VaR
    sigma, nu = 1.0, 5.0
    assert var.var_parametric_t(sigma, 0.99, nu) > var.var_parametric_normal(sigma, 0.99)


# ---------------------------------------------------------------------------
# Historical simulation: exact numpy quantile
# ---------------------------------------------------------------------------
def test_historical_var_exact_quantile():
    r = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
    # 10% quantile via numpy linear interp: pos = 0.1*(10-1) = 0.9 -> -0.041
    assert var.var_historical(r, 0.90) == pytest.approx(0.041, abs=1e-9)


def test_historical_var_matches_numpy():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 5000)
    for conf in (0.95, 0.99):
        assert var.var_historical(r, conf) == pytest.approx(-np.quantile(r, 1 - conf))


# ---------------------------------------------------------------------------
# Expected shortfall
# ---------------------------------------------------------------------------
def test_es_normal_known_value():
    # ES_97.5 for standard normal = phi(z_0.025)/0.025 ~ 2.3378 (RiskMetrics value)
    assert var.es_normal(1.0, 0.975) == pytest.approx(2.33776, abs=1e-4)


def test_es_exceeds_var():
    sigma = 0.012
    assert var.es_normal(sigma, 0.975) > var.var_parametric_normal(sigma, 0.975)
    assert var.es_t(sigma, 0.975, 6.0) > var.var_parametric_t(sigma, 0.975, 6.0)


# ---------------------------------------------------------------------------
# Kupiec POF
# ---------------------------------------------------------------------------
def test_kupiec_perfect_coverage_gives_zero_lr():
    lr, p = backtest.kupiec_pof(n=1000, x=10, p=0.01)  # pi == p exactly
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_kupiec_rejects_gross_underestimation():
    # 5% observed vs 1% expected on 1000 days -> strong rejection
    lr, p = backtest.kupiec_pof(n=1000, x=50, p=0.01)
    assert lr > 20
    assert p < 0.01


def test_kupiec_zero_exceptions_is_finite():
    lr, p = backtest.kupiec_pof(n=1000, x=0, p=0.01)
    assert np.isfinite(lr) and np.isfinite(p)
    assert lr == pytest.approx(-2 * 1000 * np.log(0.99), abs=1e-6)


# ---------------------------------------------------------------------------
# Christoffersen independence
# ---------------------------------------------------------------------------
def test_christoffersen_flags_clustering():
    clustered = [1] * 10 + [0] * 90            # all breaches back-to-back
    _, p = backtest.christoffersen_independence(clustered)
    assert p < 0.05


def test_christoffersen_passes_iid_breaches():
    # Genuinely independent (iid) breaches should not be flagged as clustered.
    rng = np.random.default_rng(7)
    iid = (rng.random(3000) < 0.05).astype(int)
    _, p = backtest.christoffersen_independence(iid)
    assert p > 0.05


# ---------------------------------------------------------------------------
# Basel traffic light
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exc,zone", [(0, "green"), (4, "green"),
                                      (5, "yellow"), (9, "yellow"),
                                      (10, "red"), (25, "red")])
def test_basel_zone_boundaries(exc, zone):
    assert backtest.basel_zone(exc) == zone


# ---------------------------------------------------------------------------
# Exceedance wiring
# ---------------------------------------------------------------------------
def test_exceedances_flags_losses_beyond_var():
    idx = pd.date_range("2020-01-01", periods=4)
    actual = pd.Series([-0.03, 0.01, -0.06, -0.049], index=idx)
    varf = pd.Series([0.05, 0.05, 0.05, 0.05], index=idx)
    br = backtest.exceedances(actual, varf)
    assert br.tolist() == [False, False, True, False]  # only -0.06 < -0.05


# ---------------------------------------------------------------------------
# EWMA recursion sanity
# ---------------------------------------------------------------------------
def test_ewma_converges_to_constant_magnitude():
    # returns of constant magnitude 0.01 -> EWMA sigma should sit at ~0.01
    r = np.array([0.01, -0.01] * 300)
    sig = var._ewma_last_sigma(r)
    assert sig == pytest.approx(0.01, abs=1e-3)


def test_ewma_vol_series_shape_and_positive():
    s = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500))
    ev = volatility.ewma_vol(s)
    assert len(ev) == len(s)
    assert (ev > 0).all()


# ---------------------------------------------------------------------------
# End-to-end coverage on synthetic normal data
# ---------------------------------------------------------------------------
def test_synthetic_normal_coverage_passes_kupiec():
    """If returns really are N(0, sigma), parametric-normal VaR should be
    correctly calibrated and Kupiec should NOT reject."""
    rng = np.random.default_rng(42)
    sigma = 0.01
    n = 20000
    r = rng.normal(0.0, sigma, n)
    conf = 0.99
    var_f = var.var_parametric_normal(sigma, conf)          # constant, true sigma
    breaches = r < -var_f
    lr, p = backtest.kupiec_pof(n, int(breaches.sum()), 1 - conf)
    assert p > 0.05                                          # fail to reject H0
    # observed rate close to 1%
    assert breaches.mean() == pytest.approx(0.01, abs=0.003)
