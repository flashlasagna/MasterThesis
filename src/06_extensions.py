"""
06_extensions.py
================
Tier-1 robustness and value-added extensions for "Forecasting Copper Prices:
A Replication and ML Extension of Buncic & Moretto (2014)".

These analyses sharpen and defend the thesis's central claims without altering
the core pipeline. All four read directly from the result objects already
produced by 02_dma_dms.py, 03_ml_models.py, and 04_hybrid.py — only the
forgetting-factor sensitivity (T8) re-runs the DMA engine.

Extensions
----------
T7  Pairwise Diebold-Mariano tests (HLN small-sample correction)
        Tests whether differences AMONG the top performers (DMA, Ridge,
        Elastic Net, Combo) are statistically significant. Sharpens the
        "convergence" claim from "close" to "statistically indistinguishable".
        NOTE: pairwise DM is valid here because these models are NOT nested in
        each other (DMA is not a special case of Ridge). This is distinct from
        comparison against the random walk, which IS nested and requires the
        Clark-West correction used in Table 2.

T8  Forgetting-factor sensitivity table
        DMA out-of-sample R^2 across a grid of lambda x alpha. Closes the
        "why these parameters?" question. Re-runs the DMA loop for each cell.

T9  Directional accuracy / hit-rate + Pesaran-Timmermann test
        Percentage of months each model predicts the correct sign of the
        copper return, with a formal PT test of directional predictability.
        Delivers on the "valuable to traders/hedgers" promise in the intro.
        Exact-zero forecasts are scored as MISSES (conservative convention).

T10 Predictor correlation matrix
        Pairwise correlations among the 18 predictors. Supports the
        Elastic-Net grouping discussion and motivates why model averaging
        finds so many near-equivalent specifications. Saved as CSV; a heatmap
        figure is produced in 05_evaluation-style if matplotlib is available.

Usage
-----
    from src.extensions import run_all_extensions
    run_all_extensions(
        df=df, predictor_cols=PREDICTOR_COLS,
        dma_result=dma_result, ml_results=ml_results,
        hybrid_results=hybrid_results,
        y=y, X=X, n_insample=120,
        output_dir='output/',
        run_sensitivity=True,        # set False to skip the T8 re-run
    )

References
----------
Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the Equality of
    Prediction Mean Squared Errors. International Journal of Forecasting,
    13(2), 281-291.
Pesaran, M.H. & Timmermann, A. (1992). A Simple Nonparametric Test of
    Predictive Performance. Journal of Business & Economic Statistics,
    10(4), 461-465.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

warnings.filterwarnings("ignore")


# ===========================================================================
# Shared helpers
# ===========================================================================

def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _newey_west_var(d: np.ndarray, bw: int) -> float:
    """
    Newey-West (Bartlett-kernel) long-run variance of the mean of series d.

    Returns the variance of the SAMPLE MEAN d-bar, i.e. includes the 1/T
    factor. Used by both the DM/HLN statistic and as a generic HAC estimator.
    """
    T = len(d)
    d_demeaned = d - d.mean()
    gamma0 = np.dot(d_demeaned, d_demeaned) / T
    lrv = gamma0
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1)
        gamma = np.dot(d_demeaned[lag:], d_demeaned[:-lag]) / T
        lrv += 2.0 * w * gamma
    lrv = max(lrv, 1e-12)
    return lrv / T  # variance of the mean


# ===========================================================================
# T7 — Pairwise Diebold-Mariano with HLN small-sample correction
# ===========================================================================

def diebold_mariano_hln(errors_a: np.ndarray,
                        errors_b: np.ndarray,
                        h: int = 1,
                        loss: str = 'squared') -> dict:
    """
    Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997)
    small-sample correction.

    Tests H0: equal predictive accuracy (E[loss_a - loss_b] = 0).
    A NEGATIVE statistic means model A has the smaller loss (A is better);
    a POSITIVE statistic means B is better. The two-sided p-value tests
    whether the accuracy difference is distinguishable from zero at all.

    The HLN correction (a) rescales the statistic by a finite-sample factor
    and (b) compares it to a Student-t distribution with (T-1) degrees of
    freedom rather than the standard Normal. For h=1 the loss differential
    has no MA structure, so the bandwidth used in the long-run variance is
    h-1 = 0 (i.e. only the contemporaneous variance), which is the textbook
    DM setting for one-step-ahead forecasts.

    Parameters
    ----------
    errors_a, errors_b : np.ndarray (T,)  forecast errors (actual - forecast)
    h    : int    forecast horizon (loss-differential MA order is h-1)
    loss : str    'squared' or 'absolute'

    Returns
    -------
    dict with keys: dm_stat, hln_stat, p_value (two-sided, from t_{T-1}),
                    mean_diff, better, T
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    if ea.shape != eb.shape:
        raise ValueError("error series must have equal length")
    T = len(ea)

    if loss == 'squared':
        la, lb = ea**2, eb**2
    elif loss == 'absolute':
        la, lb = np.abs(ea), np.abs(eb)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = la - lb            # loss differential; negative => A better
    d_bar = d.mean()

    # Long-run variance of d-bar using bandwidth = h-1 (DM convention)
    bw = max(0, h - 1)
    if bw == 0:
        var_dbar = np.var(d, ddof=0) / T
        var_dbar = max(var_dbar, 1e-12)
    else:
        var_dbar = _newey_west_var(d, bw)

    dm_stat = d_bar / np.sqrt(var_dbar)

    # HLN finite-sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    hln_stat = dm_stat * hln_factor

    # Two-sided p-value from Student-t with T-1 dof
    p_value = 2.0 * (1.0 - student_t.cdf(abs(hln_stat), df=T - 1))

    better = 'A' if d_bar < 0 else ('B' if d_bar > 0 else 'tie')

    return {
        'dm_stat':   float(dm_stat),
        'hln_stat':  float(hln_stat),
        'p_value':   float(p_value),
        'mean_diff': float(d_bar),
        'better':    better,
        'T':         int(T),
    }


def make_table7_dm(forecasts: dict,
                   actual: np.ndarray,
                   output_dir: str,
                   h: int = 1) -> pd.DataFrame:
    """
    T7: Pairwise HLN-corrected Diebold-Mariano matrix among the top models.

    Produces a long-form table (one row per unordered pair) AND a square
    matrix of HLN statistics / p-values, both saved to CSV.

    Parameters
    ----------
    forecasts : dict   {model_name: forecast_vector}; order is preserved
    actual    : np.ndarray (T,)  realised returns
    h         : int    forecast horizon

    Returns
    -------
    pd.DataFrame  long-form pairwise results
    """
    names = list(forecasts.keys())
    errors = {nm: actual - fc for nm, fc in forecasts.items()}

    long_rows = []
    n = len(names)
    stat_mat = pd.DataFrame(np.nan, index=names, columns=names)
    pval_mat = pd.DataFrame(np.nan, index=names, columns=names)

    for i in range(n):
        for j in range(n):
            if i == j:
                stat_mat.iloc[i, j] = 0.0
                pval_mat.iloc[i, j] = 1.0
                continue
            res = diebold_mariano_hln(errors[names[i]], errors[names[j]],
                                      h=h, loss='squared')
            # statistic sign convention: negative => row model (A) better
            stat_mat.iloc[i, j] = round(res['hln_stat'], 4)
            pval_mat.iloc[i, j] = round(res['p_value'], 4)
            if i < j:  # record each unordered pair once in long form
                winner = names[i] if res['better'] == 'A' else (
                         names[j] if res['better'] == 'B' else 'tie')
                long_rows.append({
                    'Model A':      names[i],
                    'Model B':      names[j],
                    'Mean loss diff (A-B)': round(res['mean_diff'], 5),
                    'HLN stat':     round(res['hln_stat'], 4),
                    'p-value':      round(res['p_value'], 4),
                    'Lower MSFE':   winner,
                    'Significant 5%': 'yes' if res['p_value'] < 0.05 else 'no',
                })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(output_dir, 'T7_dm_pairwise.csv'), index=False)
    stat_mat.to_csv(os.path.join(output_dir, 'T7_dm_stat_matrix.csv'))
    pval_mat.to_csv(os.path.join(output_dir, 'T7_dm_pval_matrix.csv'))
    print("  T7 saved (pairwise DM-HLN: long form + stat/pval matrices)")
    return long_df


# ===========================================================================
# T8 — Forgetting-factor sensitivity
# ===========================================================================

def make_table8_sensitivity(y: np.ndarray,
                            X: np.ndarray,
                            n_insample: int,
                            output_dir: str,
                            lambdas: list = (0.97, 0.98, 0.99),
                            alphas: list = (0.90, 0.95, 0.99),
                            kappa: float = 0.97,
                            dates=None,
                            baseline=(0.99, 0.95)) -> pd.DataFrame:
    """
    T8: DMA out-of-sample R^2 across a grid of lambda (state forgetting) and
    alpha (model forgetting). Closes the "why these parameters?" question.

    Re-runs the DMA engine once per (lambda, alpha) cell. With 3x3 = 9 cells
    at ~80s each, expect ~12 minutes. The baseline cell (0.99, 0.95) should
    reproduce the headline R^2 from Table 2 exactly, serving as a self-check.

    Returns
    -------
    pd.DataFrame  rows = lambda, cols = alpha, values = OOS R^2 (%)
    """
    # Local import to avoid a hard dependency when sensitivity is skipped
    from src.dma_dms import run_dma_dms

    grid = pd.DataFrame(index=[f'lambda={l}' for l in lambdas],
                        columns=[f'alpha={a}' for a in alphas], dtype=float)

    print(f"  T8: running {len(lambdas)}x{len(alphas)} DMA sensitivity grid...")
    for l in lambdas:
        for a in alphas:
            res = run_dma_dms(y=y, X=X, n_insample=n_insample,
                              lam=l, alpha=a, kappa=kappa,
                              dates=dates, verbose=False)
            r2 = res.r2_oos['DMA'] * 100.0
            grid.loc[f'lambda={l}', f'alpha={a}'] = round(r2, 2)
            tag = '  <- baseline' if (l, a) == tuple(baseline) else ''
            print(f"    lambda={l}, alpha={a}:  R2={r2:6.2f}%{tag}")

    grid.to_csv(os.path.join(output_dir, 'T8_ff_sensitivity.csv'))
    print("  T8 saved (forgetting-factor sensitivity grid)")
    return grid


# ===========================================================================
# T9 — Directional accuracy + Pesaran-Timmermann
# ===========================================================================

def pesaran_timmermann(actual: np.ndarray,
                       forecast: np.ndarray) -> dict:
    """
    Pesaran-Timmermann (1992) nonparametric test of directional / sign
    predictability.

    Tests H0: forecast direction is independent of actual direction.
    The statistic is asymptotically N(0,1) under H0; large positive values
    indicate genuine directional predictability.

    Sign convention for scoring: a correct "hit" requires sign(forecast) ==
    sign(actual). Exact-zero forecasts are scored as MISSES (they take
    sign 0, which equals sign(actual) only when the actual is also exactly
    zero — effectively never for continuous returns). This is the
    conservative convention requested for the thesis.

    Returns
    -------
    dict: hit_rate, pt_stat, p_value (one-sided), n
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    n = len(a)

    sa = np.sign(a)
    sf = np.sign(f)
    # Hit: same sign AND forecast sign is non-zero (zero forecast = miss)
    hits = ((sa == sf) & (sf != 0)).astype(float)
    hit_rate = hits.mean()

    # PT test uses indicator of POSITIVE direction
    # Define indicator z_t = 1 if actual>0, x_t = 1 if forecast>0
    py = np.mean(a > 0)                 # P(actual up)
    px = np.mean(f > 0)                 # P(forecast up)
    # P* = probability of a correct sign call under independence
    p_star = py * px + (1 - py) * (1 - px)
    # P_hat = realised proportion of correct UP/DOWN sign calls,
    # consistent with the PT formulation (zero forecast counts as "down")
    correct = ((a > 0) == (f > 0)).astype(float)
    p_hat = correct.mean()

    var_phat = p_star * (1 - p_star) / n
    var_pstar = (((2 * py - 1) ** 2) * px * (1 - px)
                 + ((2 * px - 1) ** 2) * py * (1 - py)
                 + 4 * py * px * (1 - py) * (1 - px) / n) / n
    var_diff = var_phat - var_pstar
    var_diff = max(var_diff, 1e-12)

    pt_stat = (p_hat - p_star) / np.sqrt(var_diff)
    p_value = 1.0 - norm.cdf(pt_stat)   # one-sided

    return {
        'hit_rate': float(hit_rate),
        'pt_stat':  float(pt_stat),
        'p_value':  float(p_value),
        'n':        int(n),
    }


def make_table9_directional(forecasts: dict,
                            actual: np.ndarray,
                            output_dir: str) -> pd.DataFrame:
    """
    T9: Directional hit-rate and Pesaran-Timmermann test for every model.

    Parameters
    ----------
    forecasts : dict  {model_name: forecast_vector}
    actual    : np.ndarray (T,)

    Returns
    -------
    pd.DataFrame  one row per model, sorted by hit-rate
    """
    rows = []
    for name, fc in forecasts.items():
        pt = pesaran_timmermann(actual, fc)
        rows.append({
            'Model':       name,
            'Hit rate (%)': round(pt['hit_rate'] * 100, 2),
            'PT stat':     round(pt['pt_stat'], 4),
            'p-value':     round(pt['p_value'], 4),
            'N':           pt['n'],
        })
    t9 = (pd.DataFrame(rows)
            .sort_values('Hit rate (%)', ascending=False)
            .set_index('Model'))
    t9.to_csv(os.path.join(output_dir, 'T9_directional.csv'))
    print("  T9 saved (directional accuracy + Pesaran-Timmermann)")
    return t9


# ===========================================================================
# T10 — Predictor correlation matrix
# ===========================================================================

def make_table10_correlation(df: pd.DataFrame,
                            predictor_cols: list,
                            output_dir: str,
                            make_heatmap: bool = True) -> pd.DataFrame:
    """
    T10: Pearson correlation matrix of the 18 predictors over the full sample.

    Supports the Elastic-Net grouping discussion and motivates the large
    population of near-equivalent DMA specifications. A heatmap figure is
    written in the 05_evaluation visual style when matplotlib is available.

    Returns
    -------
    pd.DataFrame  18x18 correlation matrix (short labels)
    """
    short = [c.replace('x_', '') for c in predictor_cols]
    corr = df[predictor_cols].corr()
    corr.index = short
    corr.columns = short
    corr.round(3).to_csv(os.path.join(output_dir, 'T10_predictor_corr.csv'))
    print("  T10 saved (predictor correlation matrix)")

    if make_heatmap:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            with plt.rc_context({'font.size': 8, 'figure.dpi': 150}):
                fig, ax = plt.subplots(figsize=(9, 8))
                im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1,
                               aspect='equal')
                ax.set_xticks(range(len(short)))
                ax.set_yticks(range(len(short)))
                ax.set_xticklabels(short, rotation=90, fontsize=7)
                ax.set_yticklabels(short, fontsize=7)
                # annotate cells with correlation values
                for i in range(len(short)):
                    for j in range(len(short)):
                        v = corr.values[i, j]
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                fontsize=5,
                                color='white' if abs(v) > 0.55 else 'black')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Pearson correlation', fontsize=8)
                ax.set_title('Figure A10: Predictor Correlation Matrix '
                             '(March 1998 – February 2026)', fontsize=10)
                plt.tight_layout()
                path = os.path.join(output_dir, 'F11_predictor_corr_heatmap.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved: {path}")
        except Exception as e:
            print(f"    Heatmap skipped ({e})")

    return corr.round(3)


# ===========================================================================
# Master runner
# ===========================================================================

def _collect_top_forecasts(dma_result, ml_results, hybrid_results) -> dict:
    """
    Assemble the forecast dict for the 'top models' comparison used by T7.
    Order matters for matrix readability: DMA, Ridge, ElasticNet, Combo.
    Falls back gracefully if a model is missing.
    """
    fc = {}
    fc['DMA'] = dma_result.dma_forecasts
    if 'Ridge' in ml_results:
        fc['Ridge'] = ml_results['Ridge'].forecasts
    if 'ElasticNet' in ml_results:
        fc['ElasticNet'] = ml_results['ElasticNet'].forecasts
    if 'Combo_DMA_EN' in hybrid_results:
        fc['Combo'] = hybrid_results['Combo_DMA_EN'].forecasts
    return fc


def _collect_all_forecasts(dma_result, ml_results, hybrid_results) -> dict:
    """Assemble the full forecast dict used by T9 (directional accuracy)."""
    fc = {'DMA': dma_result.dma_forecasts,
          'DMS': dma_result.dms_forecasts,
          'TVP': dma_result.tvp_forecasts}
    for name in ['OLS', 'Ridge', 'LASSO', 'ElasticNet', 'BayesianRidge',
                 'RandomForest', 'XGBoost']:
        if name in ml_results and not np.isnan(ml_results[name].msfe):
            fc[name] = ml_results[name].forecasts
    for hname, label in [('PIP_ElasticNet', 'PIP-EN'),
                         ('Stacking', 'Stacking'),
                         ('Combo_DMA_EN', 'Combo')]:
        if hname in hybrid_results:
            fc[label] = hybrid_results[hname].forecasts
    return fc


def run_all_extensions(df: pd.DataFrame,
                       predictor_cols: list,
                       dma_result,
                       ml_results: dict,
                       hybrid_results: dict,
                       y: np.ndarray,
                       X: np.ndarray,
                       n_insample: int,
                       output_dir: str = 'output/',
                       run_sensitivity: bool = True,
                       dates=None) -> dict:
    """
    Run all Tier-1 extensions (T7–T10) and save outputs.

    Parameters
    ----------
    df              : clean monthly panel
    predictor_cols  : 18 predictor column names
    dma_result      : DMAResult from run_dma_dms()
    ml_results      : dict from run_all_ml_models()
    hybrid_results  : dict from run_all_hybrids()
    y, X            : target and predictor arrays (needed for T8 re-run)
    n_insample      : burn-in length
    run_sensitivity : if False, skip the T8 forgetting-factor grid (the only
                      compute-heavy item)
    dates           : optional DatetimeIndex aligned with y

    Returns
    -------
    dict mapping table name -> DataFrame
    """
    _ensure_dir(output_dir)
    if dates is None:
        dates = pd.DatetimeIndex(df['date'])

    actual = dma_result.actual

    print("\n" + "=" * 55)
    print("06_extensions.py  —  Tier 1 extensions")
    print("=" * 55)

    out = {}

    # ── T7: pairwise DM-HLN among top models ─────────────────────────────
    print("\nT7: Pairwise Diebold-Mariano (HLN-corrected)")
    top_fc = _collect_top_forecasts(dma_result, ml_results, hybrid_results)
    out['T7'] = make_table7_dm(top_fc, actual, output_dir, h=1)
    print(out['T7'].to_string(index=False))

    # ── T9: directional accuracy + PT (cheap; do before optional T8) ─────
    print("\nT9: Directional accuracy + Pesaran-Timmermann")
    all_fc = _collect_all_forecasts(dma_result, ml_results, hybrid_results)
    out['T9'] = make_table9_directional(all_fc, actual, output_dir)
    print(out['T9'].to_string())

    # ── T10: predictor correlation matrix ────────────────────────────────
    print("\nT10: Predictor correlation matrix")
    out['T10'] = make_table10_correlation(df, predictor_cols, output_dir)

    # ── T8: forgetting-factor sensitivity (optional, compute-heavy) ──────
    if run_sensitivity:
        print("\nT8: Forgetting-factor sensitivity (re-runs DMA grid)")
        out['T8'] = make_table8_sensitivity(
            y=y, X=X, n_insample=n_insample, output_dir=output_dir,
            dates=dates)
        print(out['T8'].to_string())
    else:
        print("\nT8: Forgetting-factor sensitivity [SKIPPED — run_sensitivity=False]")

    print("\nAll Tier-1 extensions complete. Outputs in:", output_dir)
    return out


# ---------------------------------------------------------------------------
# Quick validation — run as script against the real pipeline
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline   import build_dataset, PREDICTOR_COLS
    from src.dma_dms    import run_dma_dms
    from src.ml_models  import run_all_ml_models
    from src.hybrid     import run_all_hybrids

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/'

    df, meta = build_dataset(data_dir=data_dir)
    y     = df['r_copper'].values
    X     = df[PREDICTOR_COLS].values
    dates = pd.DatetimeIndex(df['date'])
    n_is  = 120

    dma_result     = run_dma_dms(y=y, X=X, n_insample=n_is, dates=dates, verbose=True)
    ml_results     = run_all_ml_models(y=y, X=X, n_insample=n_is, dates=dates,
                                       include_lstm=False, verbose=True)
    hybrid_results = run_all_hybrids(y=y, X=X, dma_result=dma_result,
                                     ml_results=ml_results, n_insample=n_is,
                                     dates=dates, verbose=True)

    run_all_extensions(
        df=df, predictor_cols=PREDICTOR_COLS,
        dma_result=dma_result, ml_results=ml_results,
        hybrid_results=hybrid_results,
        y=y, X=X, n_insample=n_is,
        output_dir=output_dir,
        run_sensitivity=True,
        dates=dates,
    )