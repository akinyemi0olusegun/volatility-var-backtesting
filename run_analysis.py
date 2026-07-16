"""One-command pipeline: regenerate every figure and table.

    python run_analysis.py            # full run (all assets, all models)
    python run_analysis.py --quick    # skip the two GARCH models (fast smoke run)

Outputs
-------
results/tables/summary.csv          master (asset x model x confidence) table
results/tables/summary.md           same, Markdown (pasted into the README)
results/tables/model_leaderboard.csv aggregate "which model won" ranking
results/tables/garch_params.csv     full-sample GARCH(1,1) fitted parameters
results/figures/vol_forecast_<A>.png     conditional vol vs realised, per asset
results/figures/var_breaches_<A>.png     returns + 99% VaR band + breaches, per asset
results/figures/breach_rate_comparison.png  breach-rate bars across models

The walk-forward VaR panel is also cached to results/walkforward.pkl so the
notebook can reload it without recomputing.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest, regimes, var, volatility
from src.data import PORTFOLIO_COL, TICKERS, build_returns

SEED = 42
np.random.seed(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_HERE, "results", "figures")
TAB_DIR = os.path.join(_HERE, "results", "tables")
PKL_PATH = os.path.join(_HERE, "results", "walkforward.pkl")

ASSET_ORDER = list(TICKERS) + [PORTFOLIO_COL]
TRADING_DAYS = 252
ANNUALISE = np.sqrt(TRADING_DAYS)

MODEL_LABELS = {
    "normal_rolling": "Normal (rolling 250d)",
    "ewma": "EWMA (lambda=0.94)",
    "historical": "Historical sim (250d)",
    "garch_normal": "GARCH(1,1)-Normal",
    "garch_t": "GARCH(1,1)-t",
}


# ===========================================================================
# Backtests
# ===========================================================================
def run_all_backtests(returns: pd.DataFrame, methods, refit_every: int) -> dict:
    """Walk-forward VaR for every asset; returns {asset: {model: DataFrame}}."""
    results = {}
    for asset in ASSET_ORDER:
        t0 = time.time()
        print(f"[backtest] {asset} ...", flush=True)
        results[asset] = var.walk_forward_var(
            returns[asset], methods=methods, refit_every=refit_every
        )
        print(f"[backtest] {asset} done in {time.time() - t0:0.1f}s", flush=True)
    return results


# ===========================================================================
# Tables
# ===========================================================================
def build_summary(results: dict) -> pd.DataFrame:
    return backtest.summary_table(results, conf_levels=(0.95, 0.99))


def summary_to_markdown(summary: pd.DataFrame) -> str:
    df = summary.copy()
    df["model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["confidence"] = (df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    df["breach_rate"] = (df["breach_rate"] * 100).round(2).astype(str) + "%"
    df["expected_rate"] = (df["expected_rate"] * 100).round(0).astype(int).astype(str) + "%"
    df["kupiec_lr"] = df["kupiec_lr"].round(2)
    df["kupiec_p"] = df["kupiec_p"].round(3)
    df["cc_p"] = df["cc_p"].round(3)
    df["kupiec_pass"] = np.where(df["kupiec_pass"], "PASS", "FAIL")
    cols = ["asset", "model", "confidence", "n_obs", "exceptions", "breach_rate",
            "expected_rate", "kupiec_lr", "kupiec_p", "kupiec_pass", "cc_p", "traffic_light"]
    df = df[cols].rename(columns={
        "n_obs": "N", "exceptions": "exc", "breach_rate": "breach%",
        "expected_rate": "exp%", "kupiec_lr": "Kupiec LR", "kupiec_p": "Kupiec p",
        "kupiec_pass": "Kupiec", "cc_p": "CC p", "traffic_light": "Basel"})
    return df.to_markdown(index=False)


def build_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ranking: Kupiec pass-rate and mean |breach - expected|."""
    rows = []
    for model, g in summary.groupby("model"):
        rows.append({
            "model": model,
            "kupiec_pass_rate": g["kupiec_pass"].mean(),
            "n_tests": len(g),
            "mean_abs_rate_error_bps": float(
                (g["breach_rate"] - g["expected_rate"]).abs().mean() * 1e4
            ),
            "red_zones_99": int(((summary["confidence"] == 0.99) &
                                 (summary["model"] == model) &
                                 (summary["traffic_light"] == "red")).sum()),
        })
    lb = pd.DataFrame(rows).sort_values(
        ["kupiec_pass_rate", "mean_abs_rate_error_bps"], ascending=[False, True]
    ).reset_index(drop=True)
    return lb


def build_garch_params(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample GARCH(1,1) params per asset (Normal & t) for the README."""
    rows = []
    for asset in ASSET_ORDER:
        for dist in ("normal", "t"):
            try:
                p = volatility.garch_params(returns[asset], dist=dist)
                row = {"asset": asset, "dist": dist,
                       "mu": p.get("mu"), "omega": p.get("omega"),
                       "alpha1": p.get("alpha[1]"), "beta1": p.get("beta[1]"),
                       "nu": p.get("nu", np.nan)}
                row["alpha+beta"] = (row["alpha1"] or 0) + (row["beta1"] or 0)
                rows.append(row)
            except Exception as e:  # pragma: no cover
                print(f"  [warn] GARCH-{dist} params failed for {asset}: {e}")
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================
def fig_vol_forecast(returns: pd.DataFrame, methods) -> None:
    """Conditional vol (annualised %) vs a realised proxy, one panel per asset."""
    fig, axes = plt.subplots(len(ASSET_ORDER), 1, figsize=(11, 2.6 * len(ASSET_ORDER)),
                             sharex=True)
    for ax, asset in zip(np.atleast_1d(axes), ASSET_ORDER):
        r = returns[asset]
        realised = r.rolling(21).std() * ANNUALISE * 100
        ax.plot(realised.index, realised, color="0.6", lw=0.8, label="Realised (21d)")
        ax.plot(r.index, volatility.rolling_std(r, 250) * ANNUALISE * 100,
                lw=1.0, label="Rolling 250d")
        ax.plot(r.index, volatility.ewma_vol(r) * ANNUALISE * 100,
                lw=1.0, label="EWMA 0.94")
        if "garch_normal" in methods:
            ax.plot(r.index, volatility.garch_forecast(r, "normal") * ANNUALISE * 100,
                    lw=1.0, label="GARCH-N")
        ax.set_title(f"{asset} — annualised volatility (%)", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vol_forecast_all.png"), dpi=120)
    plt.close(fig)


def fig_var_breaches(results: dict, conf: float = 0.99, model: str = "garch_t") -> None:
    """Returns with the VaR band and breach days highlighted, one panel per asset."""
    fallback = "normal_rolling"
    fig, axes = plt.subplots(len(ASSET_ORDER), 1, figsize=(11, 2.6 * len(ASSET_ORDER)),
                             sharex=True)
    for ax, asset in zip(np.atleast_1d(axes), ASSET_ORDER):
        by_model = results[asset]
        m = model if model in by_model else fallback
        wf = by_model[m]
        actual = wf["actual"] * 100
        var_line = -wf[f"var_{conf:g}"] * 100          # lower band (negative)
        br = backtest.exceedances(wf["actual"], wf[f"var_{conf:g}"])
        ax.plot(actual.index, actual, color="0.5", lw=0.5, label="Daily return %")
        ax.plot(var_line.index, var_line, color="tab:blue", lw=1.0,
                label=f"{int(conf*100)}% VaR")
        breach_pts = actual[br.values]
        ax.scatter(breach_pts.index, breach_pts, color="red", s=10, zorder=5,
                   label=f"Breach (n={int(br.sum())})")
        ax.set_title(f"{asset} — {MODEL_LABELS.get(m, m)}, {int(conf*100)}% VaR",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "var_breaches_all.png"), dpi=120)
    plt.close(fig)


def fig_breach_rate_comparison(summary: pd.DataFrame) -> None:
    """Grouped bars: mean breach rate per model vs expected, at 95% and 99%."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, conf in zip(axes, (0.95, 0.99)):
        sub = summary[summary["confidence"] == conf]
        means = sub.groupby("model")["breach_rate"].mean().reindex(MODEL_LABELS.keys())
        labels = [MODEL_LABELS[m] for m in means.index]
        bars = ax.bar(labels, means.values * 100, color="tab:blue", alpha=0.8)
        ax.axhline((1 - conf) * 100, color="red", ls="--", lw=1.2,
                   label=f"Expected {int((1-conf)*100)}%")
        for b, v in zip(bars, means.values * 100):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_title(f"Mean breach rate across assets — {int(conf*100)}% VaR")
        ax.set_ylabel("breach rate (%)")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend()
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "breach_rate_comparison.png"), dpi=120)
    plt.close(fig)


# ===========================================================================
# Regime-conditional validation
# ===========================================================================
def _heatmap(ax, mat: pd.DataFrame, title: str, expected: float,
             show_ylabels: bool = True) -> None:
    """Model x regime breach-rate heatmap; colour = distance from target."""
    import matplotlib.colors as mcolors

    data = mat.to_numpy(dtype=float)
    vmax = max(expected * 3, float(np.nanmax(data)))
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=expected, vmax=vmax)
    im = ax.imshow(data, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    if show_ylabels:
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in mat.index], fontsize=8)
    else:
        ax.set_yticklabels([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


def fig_regime_heatmap(cal: pd.DataFrame, vs: pd.DataFrame, conf: float = 0.99) -> None:
    """Two heatmaps: breach rate (%) by model x calendar regime and x vol-state."""
    order = list(var.METHODS)
    cal_mat = regimes.breach_rate_matrix(cal, regimes.CALENDAR_ORDER).reindex(order)
    vs_mat = regimes.breach_rate_matrix(vs, regimes.VOL_STATE_ORDER).reindex(order)
    expected = (1 - conf) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [5, 3]})
    _heatmap(axes[0], cal_mat, f"Breach rate % by calendar regime — {int(conf*100)}% VaR",
             expected)
    im = _heatmap(axes[1], vs_mat, f"Breach rate % by volatility state — {int(conf*100)}% VaR",
                  expected, show_ylabels=False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(f"breach rate % (target {expected:g}%)", fontsize=8)
    fig.suptitle("Regime-conditional VaR validation (green = near target, red = under-covered)",
                 fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "regime_heatmap.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def build_regime_outputs(results: dict, returns: pd.DataFrame) -> pd.DataFrame:
    """Write per-regime tables + stability leaderboard; return the leaderboard."""
    for conf in (0.95, 0.99):
        tag = f"{int(conf*100)}"
        cal = regimes.regime_table(results, returns, conf, "calendar")
        vs = regimes.regime_table(results, returns, conf, "volstate")
        cal.to_csv(os.path.join(TAB_DIR, f"regime_calendar_{tag}.csv"), index=False)
        vs.to_csv(os.path.join(TAB_DIR, f"regime_volstate_{tag}.csv"), index=False)
        if conf == 0.99:
            cal99, vs99 = cal, vs

    stability = regimes.stability_leaderboard(cal99, 0.99, regimes.CALENDAR_ORDER)
    stability.to_csv(os.path.join(TAB_DIR, "regime_stability_99.csv"), index=False)
    fig_regime_heatmap(cal99, vs99, 0.99)
    return stability


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the GARCH models for a fast smoke run")
    ap.add_argument("--refit-every", type=int, default=var.GARCH_REFIT_EVERY,
                    help="re-estimate GARCH params every N days (default 10)")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    methods = (["normal_rolling", "ewma", "historical"] if args.quick
               else list(var.METHODS))

    print(f"[data] loading returns (seed={SEED}) ...")
    returns = build_returns()
    print(f"[data] {len(returns)} obs, {returns.index[0].date()} -> {returns.index[-1].date()}")

    results = run_all_backtests(returns, methods, args.refit_every)
    with open(PKL_PATH, "wb") as fh:
        pickle.dump(results, fh)

    print("[tables] building summary ...")
    summary = build_summary(results)
    summary.to_csv(os.path.join(TAB_DIR, "summary.csv"), index=False)
    with open(os.path.join(TAB_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary_to_markdown(summary))

    leaderboard = build_leaderboard(summary)
    leaderboard.to_csv(os.path.join(TAB_DIR, "model_leaderboard.csv"), index=False)

    if not args.quick:
        gparams = build_garch_params(returns)
        gparams.to_csv(os.path.join(TAB_DIR, "garch_params.csv"), index=False)

    print("[regimes] regime-conditional validation ...")
    stability = build_regime_outputs(results, returns)

    print("[figures] rendering ...")
    fig_vol_forecast(returns, methods)
    fig_var_breaches(results)
    fig_breach_rate_comparison(summary)

    print("\n===== MODEL LEADERBOARD (aggregate) =====")
    print(leaderboard.to_string(index=False))
    print("\n===== REGIME STABILITY (99%, lower = more stable across regimes) =====")
    print(stability.round(2).to_string(index=False))
    print("\nWrote tables to results/tables/ and figures to results/figures/.")


if __name__ == "__main__":
    main()
