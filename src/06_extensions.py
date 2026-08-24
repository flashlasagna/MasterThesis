"""
06_extensions.py
================
Tier-1 robustness and value-added extensions for "Forecasting Copper Prices:
A Replication and ML Extension of Buncic & Moretto (2015)".

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

T22 Ex-post regime ranking vs real-time model selection
        Whether the regime-specific ranking of DMA and Elastic Net in
        Table 5 could have been exploited with information available at
        the forecast origin: recent-winner and discounted-MSFE rules,
        VIX-state rules, and two infeasible oracles as upper bounds.

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

from typing import Optional
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
                   h: int = 1,
                   suffix: str = '') -> pd.DataFrame:
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
    long_df.to_csv(os.path.join(output_dir, f'T7_dm_pairwise{suffix}.csv'), index=False)
    stat_mat.to_csv(os.path.join(output_dir, f'T7_dm_stat_matrix{suffix}.csv'))
    pval_mat.to_csv(os.path.join(output_dir, f'T7_dm_pval_matrix{suffix}.csv'))
    print(f"  T7{suffix} saved (pairwise DM-HLN: long form + stat/pval matrices)")
    return long_df


# ===========================================================================
# T17 — Power of the pairwise DM tests (minimum detectable difference)
# ===========================================================================

def make_table17_dm_power(forecasts: dict, actual: np.ndarray,
                          output_dir: str, bw: int = 5,
                          alpha: float = 0.05, power: float = 0.80) -> pd.DataFrame:
    """
    For each pair of leading models: the observed mean loss differential,
    the smallest differential detectable with `power` at level `alpha`
    (two-sided) given the Bartlett-HAC sd of d_t, the power at the
    observed value, and the sample size needed to detect the observed
    value at `power`. MSFE quantities are also expressed in percentage
    points of OOS R^2 by dividing by the random-walk MSFE.
    """
    from scipy.stats import norm
    _ensure_dir(output_dir)
    y = np.asarray(actual); T = len(y); msfe_rw = float(np.mean(y**2))
    z = norm.ppf(1 - alpha/2) + norm.ppf(power)
    names = list(forecasts); rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            d  = (y - forecasts[a])**2 - (y - forecasts[b])**2
            dc = d - d.mean(); v = np.sum(dc**2)/T
            for l in range(1, bw+1):
                v += 2*(1 - l/(bw+1))*np.sum(dc[l:]*dc[:-l])/T
            sd = np.sqrt(v); se = sd/np.sqrt(T); obs = d.mean(); mdd = z*se
            pw = (1 - norm.cdf(norm.ppf(1-alpha/2) - abs(obs)/se)
                  + norm.cdf(-norm.ppf(1-alpha/2) - abs(obs)/se))
            Tn = (z*sd/abs(obs))**2 if obs != 0 else np.inf
            rows.append({'Model A': a, 'Model B': b,
                         'Observed dMSFE': round(obs, 3),
                         'Observed dR2 (pp)': round(100*obs/msfe_rw, 2),
                         'HAC sd(d)': round(sd, 2),
                         'Detectable dMSFE': round(mdd, 3),
                         'Detectable dR2 (pp)': round(100*mdd/msfe_rw, 2),
                         'Power at observed': round(pw, 3),
                         'T needed': int(np.ceil(Tn)) if np.isfinite(Tn) else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(output_dir, 'T17_dm_power.csv'), index=False)
    print("  T17 saved (DM power / minimum detectable difference)")
    return tab


# ===========================================================================
# T20 — Horizon decomposition (dilution benchmark, single-month targets,
#       contribution of the lagged return)
# ===========================================================================

def make_table20_horizon_decomposition(y: np.ndarray, X: np.ndarray,
                                       dma_result, ml_results: dict,
                                       n_insample: int, predictor_cols: list,
                                       output_dir: str,
                                       dates: Optional[pd.DatetimeIndex] = None
                                       ) -> pd.DataFrame:
    """
    T20: why predictability vanishes beyond one month.
      (a) Dilution benchmark -- forecast the two-month return with the
          stored h=1 forecast for month t and zero for month t+1 (origin
          t-1 information only), scored against the RW on the two-month
          return. If this beats the direct h=2 models (Table 6), the
          second month contributes no signal.
      (b) Single-month targets -- forecast r_{t+k} alone from origin t-1
          for k=1,2 with Ridge and ElasticNet, purging the k intervening
          rows from the training set (gap=k).
      (c) Contribution of the lagged return -- re-estimate the h=1 Ridge
          and ElasticNet with x_r_copper_lag dropped.
    """
    from src.ml_models import run_ridge, run_elasticnet
    _ensure_dir(output_dir)
    T = len(y); y_oos = y[n_insample:]

    def _r2(fc, yy): return 100.0 * (1.0 - np.sum((yy - fc)**2) / np.sum(yy**2))

    rows = []
    # (a) dilution benchmark
    y2 = np.array([100.0 * ((1 + y[t]/100) * (1 + y[t+1]/100) - 1) for t in range(T - 1)])
    y2o = y2[n_insample:]; L = len(y2o)
    fc1 = {'DMA': dma_result.dma_forecasts}
    for k in ('Ridge', 'ElasticNet'):
        if k in ml_results: fc1[k] = ml_results[k].forecasts
    rows.append({'Exercise': 'Dilution benchmark: h=1 forecast for month t, zero for t+1, scored on 2-month return',
                 **{k: round(_r2(v[:L], y2o), 2) for k, v in fc1.items()}})
    # (b) single-month targets
    for k in (1, 2):
        ys = np.r_[y[k:], [np.nan]*k]; v = ~np.isnan(ys)
        d = dates[v] if dates is not None else None
        r_r = run_ridge(y=ys[v], X=X[v], n_insample=n_insample, dates=d, verbose=False, gap=k)
        r_e = run_elasticnet(y=ys[v], X=X[v], n_insample=n_insample, dates=d, verbose=False, gap=k)
        rows.append({'Exercise': f'Single month t+{k} from origin t-1 (gap={k})',
                     'Ridge': round(100*r_r.r2_oos, 2), 'ElasticNet': round(100*r_e.r2_oos, 2),
                     'EN CW p': round(r_e.cw_pval, 3)})
    # (c) without lagged return
    keep = [i for i, c in enumerate(predictor_cols) if c != 'x_r_copper_lag']
    Xn = X[:, keep]
    r_r = run_ridge(y=y, X=Xn, n_insample=n_insample, dates=dates, verbose=False)
    r_e = run_elasticnet(y=y, X=Xn, n_insample=n_insample, dates=dates, verbose=False)
    rows.append({'Exercise': 'h=1 without lagged copper return',
                 'Ridge': round(100*r_r.r2_oos, 2), 'ElasticNet': round(100*r_e.r2_oos, 2)})
    rows.append({'Exercise': 'AC(1), AC(2) of target',
                 'Ridge': round(float(pd.Series(y).autocorr(1)), 3),
                 'ElasticNet': round(float(pd.Series(y).autocorr(2)), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(output_dir, 'T20_horizon_decomposition.csv'), index=False)
    print("  T20 saved (horizon decomposition)")
    return tab


# ===========================================================================
# T21 — Predictability by regime: crisis contribution and ex-ante stress
# ===========================================================================

CRISIS_WINDOWS = [('2008-09-01', '2009-06-30'), ('2020-01-01', '2020-12-31'),
                  ('2022-01-01', '2022-12-31')]


def _cw_oneside(fc, y, bw=5):
    from scipy.stats import norm
    c = y**2 - (y - fc)**2 + fc**2; T = len(c); m = c.mean(); d = c - m
    v = np.sum(d**2) / T
    for l in range(1, bw + 1):
        v += 2 * (1 - l/(bw+1)) * np.sum(d[l:] * d[:-l]) / T
    s = m / np.sqrt(v / T)
    return float(s), float(1 - norm.cdf(s))


def make_table21_regimes(df: pd.DataFrame, dma_result, ml_results: dict,
                         hybrid_results: dict, n_insample: int, output_dir: str,
                         q: float = 0.75, roll_window: int = 36) -> pd.DataFrame:
    """
    T21: is the full-sample R^2 'normal' predictability?
      Panel A  ex-post crisis windows: conditional R^2, and the share of each
               model's total SFE reduction vs RW earned in crisis months.
      Panel B  ex-ante stress states, classified with information at the
               forecast origin only: VIX_{t-1} (already in x_t) above the
               q-quantile of its own expanding history; and trailing
               12-month realised copper volatility above its expanding
               q-quantile. Conditional R^2 and calm-state CW per model.
      Also: rolling `roll_window`-month R^2 paths (T21_rolling_r2.csv).
    """
    _ensure_dir(output_dir)
    y = np.asarray(dma_result.actual); d = pd.DatetimeIndex(dma_result.dates); N = len(y)
    fc = {'DMA': dma_result.dma_forecasts}
    for k in ('Ridge', 'ElasticNet'):
        if k in ml_results: fc[k] = ml_results[k].forecasts
    if 'Combo_DMA_EN' in hybrid_results: fc['Combo'] = hybrid_results['Combo_DMA_EN'].forecasts

    def _r2(f, yy): return 100.0 * (1.0 - np.sum((yy - f)**2) / np.sum(yy**2))

    crisis = np.zeros(N, bool)
    for a, b in CRISIS_WINDOWS: crisis |= (d >= a) & (d <= b)

    full = df.set_index('date')
    vix_hist = full['x_VIX']                       # lagged one month already
    vix = vix_hist.loc[d].values
    thr = np.array([vix_hist.loc[:dt].iloc[:-1].quantile(q) for dt in d])
    hi_vix = vix > thr
    rv = full[TARGET_COL if 'TARGET_COL' in globals() else 'r_copper'].rolling(12).std().shift(1)
    rv_o = rv.loc[d].values
    thr_rv = np.array([rv.loc[:dt].dropna().iloc[:-1].quantile(q) for dt in d])
    hi_rv = rv_o > thr_rv

    rows = []
    for name, f in fc.items():
        gain = y**2 - (y - f)**2
        rows.append({'Panel': 'A ex-post crisis', 'Model': name,
                     'Full R2': round(_r2(f, y), 2),
                     'High R2': round(_r2(f[crisis], y[crisis]), 2), 'N high': int(crisis.sum()),
                     'Low R2': round(_r2(f[~crisis], y[~crisis]), 2), 'N low': int((~crisis).sum()),
                     'SFE share high (%)': round(100 * gain[crisis].sum() / gain.sum(), 1),
                     'CW low': round(_cw_oneside(f[~crisis], y[~crisis])[0], 2)})
    for lab, m, flagged in [('B(i) ex-ante VIX', hi_vix, int((hi_vix & crisis).sum())),
                            ('B(ii) ex-ante copper vol', hi_rv, int((hi_rv & crisis).sum()))]:
        for name, f in fc.items():
            rows.append({'Panel': lab, 'Model': name, 'Full R2': round(_r2(f, y), 2),
                         'High R2': round(_r2(f[m], y[m]), 2), 'N high': int(m.sum()),
                         'Low R2': round(_r2(f[~m], y[~m]), 2), 'N low': int((~m).sum()),
                         'SFE share high (%)': np.nan, 'CW low': round(_cw_oneside(f[~m], y[~m])[0], 2),
                         'Crisis months flagged': flagged})
    tab = pd.DataFrame(rows)
    tab.loc[tab.Panel == 'A ex-post crisis', 'Share of months (%)'] = round(100 * crisis.mean(), 1)
    tab.loc[tab.Panel == 'A ex-post crisis', 'Share of squared returns (%)'] = round(100 * np.sum(y[crisis]**2) / np.sum(y**2), 1)

    roll = pd.DataFrame({f'roll{roll_window}_{k}': [_r2(f[i-roll_window:i], y[i-roll_window:i])
                                                   for i in range(roll_window, N + 1)]
                         for k, f in fc.items()}, index=d[roll_window-1:])

    tab.to_csv(os.path.join(output_dir, 'T21_regime_table.csv'), index=False)
    pd.DataFrame({'date': d, 'crisis': crisis, 'hi_vix': hi_vix, 'hi_rv': hi_rv, **fc, 'actual': y}
                 ).to_csv(os.path.join(output_dir, 'T21_regime_forecasts.csv'), index=False)
    roll.to_csv(os.path.join(output_dir, 'T21_rolling_r2.csv'))
    print("  T21 saved (regime table, flags, rolling R2)")
    return tab


# ===========================================================================
# T22 — Ex-post regime interpretation vs real-time model selection
# ===========================================================================

SUBPERIODS_T22 = [('Pre-GFC',       '2008-03-01', '2008-08-01'),
                  ('GFC',           '2008-09-01', '2009-06-01'),
                  ('Recovery',      '2009-07-01', '2019-12-01'),
                  ('COVID-19',      '2020-01-01', '2020-12-01'),
                  ('Supercycle',    '2021-01-01', '2021-12-01'),
                  ('Ukraine shock', '2022-01-01', '2022-12-01'),
                  ('Post-shock',    '2023-01-01', '2026-02-01')]


def make_table22_realtime_selection(dma_result, ml_results: dict, hybrid_results: dict,
                                    output_dir: str, windows=(6, 12, 24, 36),
                                    deltas=(0.90, 0.95, 1.00), warm_up: int = 6,
                                    min_state_obs: int = 6,
                                    stress_flag: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    T22: can the regime-specific ranking of DMA and Elastic Net (Table 5)
    be exploited in real time?

    The sub-period analysis shows Elastic Net ahead in the two V-shaped
    shocks (GFC, COVID-19) and DMA ahead in the gradual recovery and the
    2022 realignment. That ranking is known only ex post. This table asks
    what a forecaster could have earned from it using information available
    at each forecast origin, and bounds the answer from above with two
    oracles that use future information.

    Selection rules (all use forecast errors realised strictly BEFORE the
    origin t; the first `warm_up` origins, which have no history, use the
    equal-weight combination):
      (i)   Recent winner, k months: the model with the lower MSFE over the
            last k realised errors (k in `windows`), and over the full
            expanding history.
      (ii)  Discounted-MSFE weights (Stock & Watson, 2004): weights inversely
            proportional to the delta-discounted sum of past squared errors.
      (iii) VIX-state rule, fixed direction: Elastic Net when the ex-ante VIX
            flag of T21 is on, DMA otherwise. The DIRECTION of this rule is
            taken from Table 21, i.e. it is chosen with hindsight; it is
            reported to show the ceiling of a state-based switch, not as a
            feasible rule.
      (iv)  VIX-state rule, direction learned in real time: within the
            current VIX state, the model with the lower MSFE over all past
            origins in the same state (equal weight until `min_state_obs`
            such origins exist). Fully feasible.
    Oracles (infeasible, upper bounds):
      (v)   Best model in each sub-period of Table 5, chosen ex post.
      (vi)  Best model in each month, chosen ex post.

    Every strategy is compared with DMA and with the equal-weight
    combination by the HLN-corrected Diebold-Mariano test of T7 (h = 1).

    Outputs
    -------
    T22_realtime_selection.csv     headline table (R2, MSFE, DM vs DMA/Combo,
                                   number of switches, share of months on DMA)
    T22_selection_subperiod.csv    sub-period R2 of each strategy
    T22_selection_path.csv         monthly choice of each rule and the ex-post
                                   winner (for F14)
    """
    _ensure_dir(output_dir)
    y = np.asarray(dma_result.actual, dtype=float); N = len(y)
    d = pd.DatetimeIndex(dma_result.dates)
    f_dma = np.asarray(dma_result.dma_forecasts, dtype=float)
    f_en  = np.asarray(ml_results['ElasticNet'].forecasts, dtype=float)
    if 'Combo_DMA_EN' in hybrid_results:
        f_cmb = np.asarray(hybrid_results['Combo_DMA_EN'].forecasts, dtype=float)
    else:
        f_cmb = 0.5 * (f_dma + f_en)
    e_dma, e_en = (y - f_dma)**2, (y - f_en)**2

    if stress_flag is None:
        p_flag = os.path.join(output_dir, 'T21_regime_forecasts.csv')
        if os.path.exists(p_flag):
            stress_flag = pd.read_csv(p_flag)['hi_vix'].values.astype(bool)
        else:
            print("  T22: T21_regime_forecasts.csv not found; VIX-state rules skipped")
    hi = None if stress_flag is None else np.asarray(stress_flag, dtype=bool)

    def _r2(f, yy): return 100.0 * (1.0 - np.sum((yy - f)**2) / np.sum(yy**2))

    # ── selection rules ───────────────────────────────────────────────────
    def recent_winner(k=None):
        fc = f_cmb.copy(); pick = np.full(N, 'Combo', dtype=object)
        for t in range(warm_up, N):
            lo = 0 if k is None else max(0, t - k)
            use_dma = e_dma[lo:t].mean() <= e_en[lo:t].mean()
            fc[t] = f_dma[t] if use_dma else f_en[t]
            pick[t] = 'DMA' if use_dma else 'EN'
        return fc, pick

    def discounted_weights(delta):
        fc = f_cmb.copy(); w = np.full(N, 0.5)
        for t in range(warm_up, N):
            disc = delta ** np.arange(t - 1, -1, -1)
            s_dma, s_en = np.sum(disc * e_dma[:t]), np.sum(disc * e_en[:t])
            w[t] = (1.0 / s_dma) / (1.0 / s_dma + 1.0 / s_en)
            fc[t] = w[t] * f_dma[t] + (1.0 - w[t]) * f_en[t]
        return fc, w

    def vix_fixed():
        pick = np.where(hi, 'EN', 'DMA').astype(object)
        return np.where(hi, f_en, f_dma), pick

    def vix_learned():
        fc = f_cmb.copy(); pick = np.full(N, 'Combo', dtype=object)
        for t in range(warm_up, N):
            same = hi[:t] == hi[t]
            if same.sum() < min_state_obs:
                continue
            use_dma = e_dma[:t][same].mean() <= e_en[:t][same].mean()
            fc[t] = f_dma[t] if use_dma else f_en[t]
            pick[t] = 'DMA' if use_dma else 'EN'
        return fc, pick

    def oracle_subperiod():
        fc = f_cmb.copy(); pick = np.full(N, 'Combo', dtype=object)
        for _, a, b in SUBPERIODS_T22:
            m = (d >= a) & (d <= b)
            if m.sum() == 0:
                continue
            use_dma = e_dma[m].sum() <= e_en[m].sum()
            fc[m] = f_dma[m] if use_dma else f_en[m]
            pick[m] = 'DMA' if use_dma else 'EN'
        return fc, pick

    def oracle_month():
        use_dma = e_dma <= e_en
        return np.where(use_dma, f_dma, f_en), np.where(use_dma, 'DMA', 'EN').astype(object)

    strategies = [('DMA',                          'Single model',    f_dma, None),
                  ('ElasticNet',                   'Single model',    f_en,  None),
                  ('Combo (equal weight)',         'No selection',    f_cmb, None)]
    for k in windows:
        strategies.append((f'Recent winner, {k}m', 'Real-time', *recent_winner(k)))
    strategies.append(('Recent winner, expanding', 'Real-time', *recent_winner(None)))
    for dl in deltas:
        strategies.append((f'Discounted MSFE weights, delta={dl:.2f}', 'Real-time',
                           *discounted_weights(dl)))
    if hi is not None:
        strategies.append(('VIX state, fixed direction (ex post)', 'Hindsight direction',
                           *vix_fixed()))
        strategies.append(('VIX state, direction learned in real time', 'Real-time',
                           *vix_learned()))
    strategies.append(('Oracle: best model per sub-period', 'Infeasible', *oracle_subperiod()))
    strategies.append(('Oracle: best model each month',     'Infeasible', *oracle_month()))

    rows, paths = [], {'date': d}
    for name, kind, fc, pick in strategies:
        dm_d = diebold_mariano_hln(y - fc, y - f_dma, h=1)
        dm_c = diebold_mariano_hln(y - fc, y - f_cmb, h=1)
        if pick is None:
            n_sw, share = np.nan, np.nan
        elif pick.dtype == object:
            n_sw = int(np.sum(pick[1:] != pick[:-1]))
            share = 100.0 * np.mean(pick == 'DMA')
            paths[name] = pick
        else:                                   # continuous weights
            n_sw = np.nan; share = 100.0 * np.mean(pick)
        rows.append({'Strategy': name, 'Type': kind,
                     'MSFE': round(float(np.mean((y - fc)**2)), 3),
                     'R2_oos (%)': round(_r2(fc, y), 2),
                     'R2 excl. warm-up (%)': round(_r2(fc[warm_up:], y[warm_up:]), 2),
                     'HLN vs DMA': round(dm_d['hln_stat'], 2), 'p vs DMA': round(dm_d['p_value'], 3),
                     'HLN vs Combo': round(dm_c['hln_stat'], 2), 'p vs Combo': round(dm_c['p_value'], 3),
                     'Switches': n_sw, 'Share DMA (%)': None if share != share else round(share, 1)})
    tab = pd.DataFrame(rows)

    sub_rows = []
    for pname, a, b in SUBPERIODS_T22:
        m = (d >= a) & (d <= b)
        if m.sum() < 3:
            continue
        row = {'Period': pname, 'N': int(m.sum())}
        for name, _, fc, _ in strategies:
            row[name] = round(_r2(fc[m], y[m]), 2)
        sub_rows.append(row)
    sub = pd.DataFrame(sub_rows)

    paths['ex_post_winner'] = np.where(e_dma <= e_en, 'DMA', 'EN')
    paths['crisis'] = np.zeros(N, bool)
    for a, b in CRISIS_WINDOWS:
        paths['crisis'] |= (d >= a) & (d <= b)
    if hi is not None:
        paths['hi_vix'] = hi
    path = pd.DataFrame(paths)

    tab.to_csv(os.path.join(output_dir, 'T22_realtime_selection.csv'), index=False)
    sub.to_csv(os.path.join(output_dir, 'T22_selection_subperiod.csv'), index=False)
    path.to_csv(os.path.join(output_dir, 'T22_selection_path.csv'), index=False)
    print("  T22 saved (real-time selection, sub-period R2, selection path)")
    return tab


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
                ax.set_title('Predictor Correlation Matrix '
                             '(March 1998 – February 2026)', fontsize=10)
                plt.tight_layout()
                path = os.path.join(output_dir, 'F11_predictor_corr_heatmap.pdf')
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved: {path}")
        except Exception as e:
            print(f"    Heatmap skipped ({e})")

    return corr.round(3)


# ===========================================================================
# Master runner
# ===========================================================================


# ---------------------------------------------------------------------------
# T13 — DMA vs DMS: model-probability diagnostics and sub-period comparison
# ---------------------------------------------------------------------------

SUBPERIODS_T13 = [
    ('GFC',          '2008-09-01', '2009-06-30'),
    ('Recovery',     '2009-07-01', '2019-12-31'),
    ('COVID-19',     '2020-01-01', '2020-12-31'),
    ('Supercycle',   '2021-01-01', '2021-12-31'),
    ('Ukraine shock','2022-01-01', '2022-12-31'),
    ('Post-shock',   '2023-01-01', '2026-02-28'),
    ('Full OOS',     '2008-03-01', '2026-02-28'),
]


def make_table13_dma_dms(dma_result, output_dir: str) -> pd.DataFrame:
    """
    T13: Why DMA beats DMS — diagnostics on the prior model-probability
    vector at each forecast origin (recorded in DMAResult.diagnostics) and
    a sub-period comparison of the two estimators.

    Columns
    -------
    pi_max (%)   mean prior probability of the DMS-selected (top) model
    N_eff        mean effective number of models, 1 / sum_m pi_m^2
    Switch (%)   share of month-to-month transitions inside the period in
                 which the DMS-selected model changes
    DMA / DMS R2 out-of-sample R^2 vs random walk, in percent

    Also saves the full monthly diagnostic path and prints the DMA-vs-DMS
    HLN-corrected Diebold-Mariano test over the full window.
    """
    _ensure_dir(output_dir)
    d     = dma_result.diagnostics
    dates = pd.DatetimeIndex(dma_result.dates)
    y     = np.asarray(dma_result.actual)
    dma   = np.asarray(dma_result.dma_forecasts)
    dms   = np.asarray(dma_result.dms_forecasts)
    M     = float(dma_result.params['M'])

    def _r2(fc, yy):
        return 100.0 * (1.0 - np.sum((yy - fc)**2) / np.sum(yy**2))

    switch = (np.diff(d['top_model']) != 0)
    rows = []
    for label, a, b in SUBPERIODS_T13:
        m  = (dates >= a) & (dates <= b)
        mm = m[:-1] & m[1:]                      # transitions inside period
        rows.append({
            'Period':       label,
            'N':            int(m.sum()),
            'pi_max (%)':   round(100.0 * d['pi_max'][m].mean(), 4),
            'N_eff':        int(round(d['n_eff'][m].mean())),
            'Switch (%)':   round(100.0 * switch[mm].mean(), 1) if mm.any() else np.nan,
            'DMA R2 (%)':   round(_r2(dma[m], y[m]), 2),
            'DMS R2 (%)':   round(_r2(dms[m], y[m]), 2),
        })
    tab = pd.DataFrame(rows)

    # Headline diagnostics and DMA-vs-DMS DM test (full window)
    dm = diebold_mariano_hln(y - dma, y - dms, h=1)
    headline = {
        'pi_max_mean (%)':        100.0 * d['pi_max'].mean(),
        'pi_max_max (%)':         100.0 * d['pi_max'].max(),
        'pi_max_mean / uniform':  d['pi_max'].mean() * M,
        'N_eff_mean':             d['n_eff'].mean(),
        'N_eff_min':              d['n_eff'].min(),
        'N_eff_mean_share (%)':   100.0 * d['n_eff'].mean() / M,
        'mass_top10_max (%)':     100.0 * d['mass_top10'].max(),
        'switch_rate (%)':        100.0 * switch.mean(),
        'top_size_mean':          d['top_size'].mean(),
        'sd_fc_DMA':              dma.std(),
        'sd_fc_DMS':              dms.std(),
        'DMS_abs_err_larger (%)': 100.0 * np.mean(np.abs(y - dms) > np.abs(y - dma)),
        'DM_HLN_stat':            dm['hln_stat'],
        'DM_p_value':             dm['p_value'],
    }

    tab.to_csv(os.path.join(output_dir, 'T13_dma_dms_subperiod.csv'), index=False)
    pd.Series(headline).to_csv(os.path.join(output_dir, 'T13_dma_dms_headline.csv'),
                               header=['value'])
    pd.DataFrame({'date': dates, **{k: v for k, v in d.items()},
                  'dma_fc': dma, 'dms_fc': dms, 'actual': y}).to_csv(
        os.path.join(output_dir, 'T13_dma_dms_diagnostics.csv'), index=False)

    print("  T13 saved (sub-period table, headline diagnostics, monthly path)")
    print(f"  DMA vs DMS: HLN stat = {dm['hln_stat']:.3f}, p = {dm['p_value']:.3f}")
    return tab


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
                 'RandomForest', 'XGBoost', 'MLP']:
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

    # ── T7b: pairwise DM-HLN across model classes ────────────────────────
    #     Non-nested comparisons only (every model here is a distinct
    #     estimator, none a restricted case of another), so the plain
    #     DM/HLN test applies, as for T7. Documents which parts of the
    #     point-estimate ordering in Table 2 are statistically supported:
    #     linear-vs-XGBoost and Ridge-vs-OLS are, linear-vs-RandomForest
    #     and DMA-vs-DMS are not.
    print("\nT7b: Pairwise Diebold-Mariano across model classes")
    class_fc = {'DMA': dma_result.dma_forecasts, 'DMS': dma_result.dms_forecasts}
    for name in ['OLS', 'Ridge', 'ElasticNet', 'RandomForest', 'XGBoost', 'MLP']:
        if name in ml_results and not np.isnan(ml_results[name].msfe):
            class_fc[name] = ml_results[name].forecasts
    out['T7b'] = make_table7_dm(class_fc, actual, output_dir, h=1, suffix='_all')
    print(out['T7b'].to_string(index=False))

    # ── T17: power of the pairwise DM tests ───────────────────────────────
    print("\nT17: Minimum detectable difference for the DM tests")
    out['T17'] = make_table17_dm_power(top_fc, actual, output_dir)
    print(out['T17'].to_string(index=False))

    # ── T9: directional accuracy + PT (cheap; do before optional T8) ─────
    print("\nT9: Directional accuracy + Pesaran-Timmermann")
    all_fc = _collect_all_forecasts(dma_result, ml_results, hybrid_results)
    out['T9'] = make_table9_directional(all_fc, actual, output_dir)
    print(out['T9'].to_string())

    # ── T13: DMA vs DMS diagnostics (cheap; uses recorded probabilities) ─
    print("\nT13: DMA vs DMS — model-probability diagnostics")
    out['T13'] = make_table13_dma_dms(dma_result, output_dir)
    print(out['T13'].to_string(index=False))

    # ── T20: horizon decomposition (Ridge/EN re-fits, ~30s) ──────────────
    print("\nT20: Horizon decomposition")
    out['T20'] = make_table20_horizon_decomposition(
        y, X, dma_result, ml_results, n_insample, predictor_cols, output_dir, dates=dates)
    print(out['T20'].to_string(index=False))

    # ── T21: predictability by regime (ex-post and ex-ante) ───────────────
    print("\nT21: Predictability by regime")
    out['T21'] = make_table21_regimes(df, dma_result, ml_results, hybrid_results,
                                      n_insample, output_dir)
    print(out['T21'].to_string(index=False))

    # ── T22: ex-post regime ranking vs real-time model selection ──────────
    print("\nT22: Real-time selection between DMA and Elastic Net")
    out['T22'] = make_table22_realtime_selection(dma_result, ml_results, hybrid_results,
                                                 output_dir)
    print(out['T22'].to_string(index=False))

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

    # ── F13: rolling R2 by regime (needs T21) ─────────────────────────────
    try:
        from src.evaluation import plot_f13_rolling_r2_regimes
        plot_f13_rolling_r2_regimes(dma_result, output_dir)
    except Exception as exc:
        print(f"  [warning] F13 not produced: {exc}")

    # ── F14: real-time selection path vs ex-post winner (needs T22) ───────
    try:
        from src.evaluation import plot_f14_selection_path
        plot_f14_selection_path(output_dir)
    except Exception as exc:
        print(f"  [warning] F14 not produced: {exc}")

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
                                       verbose=True)
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