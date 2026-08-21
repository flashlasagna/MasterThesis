"""
02_dma_dms.py
=============
Dynamic Model Averaging and Selection (DMA/DMS) for copper return forecasting.
Replicates the econometric framework of Buncic & Moretto (2015).

Mathematical Framework
----------------------
The DMA/DMS model consists of two equations for each of M = 2^k candidate
models (indexed by m):

    Measurement : y_t       = x^(m)_{t-1} @ beta^(m)_t  +  u^(m)_t
    State       : beta^(m)_t = beta^(m)_{t-1}             +  eps^(m)_t

where u^(m)_t ~ N(0, H^(m)_t) and eps^(m)_t ~ N(0, Q^(m)_t).

Two simplifications from Raftery et al. (2010) avoid estimating H and Q:

    (1) Forgetting factor:  P^(m)_{t|t-1} = (1/lambda) * P^(m)_{t-1|t-1}
        => implies Q^(m)_t = (lambda^{-1} - 1) * P^(m)_{t-1|t-1}

    (2) EWMA variance:      H^(m)_t = kappa * H^(m)_{t-1} + (1-kappa) * u^2_{t-1}

Model probabilities are predicted using the alpha forgetting factor:

    pi^(m)_{t|t-1}  proportional to  (pi^(m)_{t-1|t-1})^alpha

and updated via Bayes' rule with the Normal predictive likelihood.

DMA forecast:  y_hat^DMA = sum_m  pi^(m)_{t|t-1} * y_hat^(m)_{t|t-1}
DMS forecast:  y_hat^DMS = y_hat^(m*)  where m* = argmax pi^(m)_{t|t-1}

Implementation Notes
--------------------
With k=18 predictors, M = 2^18 = 262,144 models. The P matrix is stored
as its diagonal only (the Koop & Korobilis 2012 diagonal approximation),
enabling fully vectorised numpy operations across all M models simultaneously.
This reduces per-step complexity from O(M * K_m^2) to O(M * K_m) and allows
the full T=336 run to complete in ~85 seconds.

Parameters (paper defaults)
---------------------------
    lambda (lam)  : 0.99    forgetting factor for state covariance
    alpha         : 0.95    forgetting factor for model probabilities
    kappa         : 0.97    EWMA smoothing parameter for H (per RiskMetrics 1996)

References
----------
Buncic, D. & Moretto, C. (2015). Forecasting Copper Prices with Dynamic
    Averaging and Selection Models. SSRN 2482015.
Koop, G. & Korobilis, D. (2012). Forecasting Inflation Using Dynamic Model
    Averaging. International Economic Review, 53(3), 867-886.
Raftery, A.E., Karny, M. & Ettler, P. (2010). Online Prediction Under Model
    Uncertainty via Dynamic Model Averaging. Technometrics, 52(1), 52-66.
RiskMetrics (1996). Technical Document, 4th Edition. J.P. Morgan/Reuters.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA = 0.99   # forgetting factor for state covariance
DEFAULT_ALPHA  = 0.95   # forgetting factor for model probabilities
DEFAULT_KAPPA  = 0.97   # EWMA smoothing for measurement variance


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class DMAResult:
    """
    Container for DMA/DMS estimation output.

    Attributes
    ----------
    dma_forecasts   : np.ndarray (T_oos,)  DMA one-step-ahead forecasts
    dms_forecasts   : np.ndarray (T_oos,)  DMS one-step-ahead forecasts
    tvp_forecasts   : np.ndarray (T_oos,)  TVP (full model) forecasts
    dates           : pd.DatetimeIndex     dates of forecast observations
    actual          : np.ndarray (T_oos,)  realised copper returns
    pip_path        : np.ndarray (T_oos, k)   posterior inclusion probabilities
    beta_dma_path   : np.ndarray (T_oos, k+1) DMA-averaged coefficients
    msfe            : dict                 MSFEs for DMA, DMS, TVP, RW, HA_exp
    r2_oos          : dict                 out-of-sample R^2 vs RW
    cw_stat         : dict                 Clark-West statistics vs RW
    cw_pval         : dict                 one-sided p-values for CW test
    params          : dict                 estimation parameters used
    runtime_seconds : float                wall-clock time for estimation
    """
    dma_forecasts   : np.ndarray = field(default_factory=lambda: np.array([]))
    dms_forecasts   : np.ndarray = field(default_factory=lambda: np.array([]))
    tvp_forecasts   : np.ndarray = field(default_factory=lambda: np.array([]))
    dates           : Optional[pd.DatetimeIndex] = None
    actual          : np.ndarray = field(default_factory=lambda: np.array([]))
    pip_path        : np.ndarray = field(default_factory=lambda: np.array([]))
    beta_dma_path   : np.ndarray = field(default_factory=lambda: np.array([]))
    msfe            : dict = field(default_factory=dict)
    r2_oos          : dict = field(default_factory=dict)
    cw_stat         : dict = field(default_factory=dict)
    cw_pval         : dict = field(default_factory=dict)
    params          : dict = field(default_factory=dict)
    runtime_seconds : float = 0.0
    # Model-probability diagnostics at each OOS origin (added for Section 5.1)
    #   'pi_max'     : probability of the DMS-selected (top) model
    #   'n_eff'      : effective number of models, 1 / sum_m pi_m^2
    #   'top_model'  : index of the DMS-selected model
    #   'top_size'   : number of predictors in the DMS-selected model
    #   'mass_top10' : cumulative probability of the 10 most probable models
    diagnostics     : dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _msfe(errors: np.ndarray) -> float:
    """Mean squared forecast error."""
    return float(np.mean(errors**2))


def _r2_oos(errors_model: np.ndarray, errors_rw: np.ndarray) -> float:
    """
    Campbell & Thompson (2008) out-of-sample R^2.
    R^2_os = 1 - MSFE(model) / MSFE(benchmark)
    Positive values indicate the model outperforms the benchmark.
    """
    return float(1.0 - _msfe(errors_model) / _msfe(errors_rw))


def _clark_west(errors_rw: np.ndarray,
                errors_model: np.ndarray,
                forecasts_rw: np.ndarray,
                forecasts_model: np.ndarray) -> tuple[float, float]:
    """
    Clark & West (2007) MSFE-adjusted t-statistic for nested model comparison.

    Tests H0: MSFE(RW) <= MSFE(model)  vs  H1: MSFE(RW) > MSFE(model).
    Rejection implies the model has significantly lower MSFE than the RW.

    Parameters
    ----------
    errors_rw     : forecast errors from random walk
    errors_model  : forecast errors from the model under evaluation
    forecasts_rw  : point forecasts from random walk
    forecasts_model: point forecasts from the model

    Returns
    -------
    cw_stat : float   Clark-West t-statistic
    p_value : float   one-sided p-value (lower = stronger evidence against H0)
    """
    # Standard DM component
    dm_seq = errors_rw**2 - errors_model**2
    # Bias-correction for nested model comparison
    adj_seq = (forecasts_rw - forecasts_model)**2
    cw_seq  = dm_seq + adj_seq

    cw_bar = cw_seq.mean()

    # HAC variance (Newey-West with automatic bandwidth)
    T = len(cw_seq)
    cw_demeaned = cw_seq - cw_bar
    # Bartlett kernel with bandwidth = floor(T^(1/3))
    bw = max(1, int(T**(1/3)))
    var_cw = np.var(cw_seq, ddof=1)
    for lag in range(1, bw + 1):
        weight = 1.0 - lag / (bw + 1)
        cov    = np.mean(cw_demeaned[lag:] * cw_demeaned[:-lag])
        var_cw += 2 * weight * cov
    var_cw = max(var_cw, 1e-12)  # numerical floor

    cw_stat = float(cw_bar / np.sqrt(var_cw / T))

    # One-sided p-value from standard Normal (asymptotic)
    from scipy.stats import norm
    p_value = float(1.0 - norm.cdf(cw_stat))

    return cw_stat, p_value


# ---------------------------------------------------------------------------
# Core DMA/DMS engine
# ---------------------------------------------------------------------------

def _build_inclusion_matrix(k: int) -> np.ndarray:
    """
    Build the (M, k+1) boolean inclusion matrix for all 2^k models.

    inclusion[m, 0]   = True  always (intercept)
    inclusion[m, j+1] = True  iff bit j is set in the binary representation of m

    Parameters
    ----------
    k : int   number of predictor variables (excluding intercept)

    Returns
    -------
    inclusion : np.ndarray (M, k+1) bool
    """
    M = 2**k
    inclusion = np.zeros((M, k + 1), dtype=bool)
    inclusion[:, 0] = True  # intercept always included
    for j in range(k):
        inclusion[:, j + 1] = ((np.arange(M) >> j) & 1).astype(bool)
    return inclusion


def run_dma_dms(y: np.ndarray,
                X: np.ndarray,
                n_insample: int,
                lam:   float = DEFAULT_LAMBDA,
                alpha: float = DEFAULT_ALPHA,
                kappa: float = DEFAULT_KAPPA,
                h:     int   = 1,
                dates: Optional[pd.DatetimeIndex] = None,
                verbose: bool = True) -> DMAResult:
    """
    Run the full DMA/DMS estimation and produce out-of-sample forecasts.

    Parameters
    ----------
    y          : np.ndarray (T,)     target variable (copper returns)
    X          : np.ndarray (T, k)   predictor matrix (k=18, NO intercept column)
    n_insample : int                 number of in-sample observations for
                                     initialising the Kalman filter (burn-in).
                                     Out-of-sample forecasts begin at t=n_insample.
    lam        : float               state covariance forgetting factor (default 0.99)
    alpha      : float               model probability forgetting factor (default 0.95)
    kappa      : float               EWMA smoothing for H (default 0.97)
    dates      : pd.DatetimeIndex    optional date index aligned with y
    verbose    : bool                print progress

    Returns
    -------
    DMAResult  dataclass with forecasts, PIPs, coefficients, and evaluation metrics
    """
    T, k = len(y), X.shape[1]
    M    = 2**k

    if verbose:
        print(f"DMA/DMS: k={k}, M={M:,}, T={T}, T_is={n_insample}, "
              f"T_oos={T - n_insample}, h={h}")
        print(f"  lambda={lam}, alpha={alpha}, kappa={kappa}")

    t_start = time.time()

    # ── Pre-build model structure ─────────────────────────────────────────
    inclusion = _build_inclusion_matrix(k)  # (M, k+1)

    # ── Initialise state (diffuse priors per Koop & Korobilis 2012) ───────
    # beta_all[m, j] : DMA-averaged parameter for model m, covariate j
    # P_all[m, j]    : diagonal of state covariance for model m
    # H_all[m]       : measurement variance for model m
    # pi_all[m]      : model probability
    beta_all = np.zeros((M, k + 1), dtype=np.float64)
    P_all    = np.where(inclusion, 100.0, 0.0).astype(np.float64)
    H_all    = np.ones(M, dtype=np.float64)
    pi_all   = np.ones(M, dtype=np.float64) / M

    # ── TVP (full model: all predictors) state ────────────────────────────
    # TVP uses the single model with ALL predictors (m = M-1)
    # We track it separately for the TVP forecast comparison
    k_tvp     = k + 1  # intercept + all k predictors
    beta_tvp  = np.zeros(k_tvp)
    P_tvp     = 100.0 * np.eye(k_tvp)
    H_tvp     = 1.0

    # ── Storage for OOS outputs ───────────────────────────────────────────
    T_oos         = T - n_insample
    dma_fc        = np.zeros(T_oos)
    dms_fc        = np.zeros(T_oos)
    tvp_fc        = np.zeros(T_oos)
    pip_path      = np.zeros((T_oos, k))
    beta_dma_path = np.zeros((T_oos, k + 1))
    # Store model probabilities at OOS dates (memory-efficient: not full path)
    # We only need current pi for updates

    # Historical average benchmark (expanding window)
    ha_fc = np.zeros(T_oos)

    # Model-probability diagnostics
    diag_pi_max   = np.zeros(T_oos)
    diag_n_eff    = np.zeros(T_oos)
    diag_top      = np.zeros(T_oos, dtype=np.int64)
    diag_top_size = np.zeros(T_oos, dtype=np.int64)
    diag_mass10   = np.zeros(T_oos)
    model_size    = inclusion[:, 1:].sum(axis=1)

    # ── Main loop ─────────────────────────────────────────────────────────
    oos_idx = 0

    for t in range(T):
        x_full = np.concatenate([[1.0], X[t]])  # forecast-origin predictors
        x_all  = np.where(inclusion, x_full[None, :], 0.0)        # (M, k+1)

        # ── Prediction step (all models) — state and pi reflect only data
        #    from target windows completed by the end of month t-1 ────────
        P_pred    = P_all / lam                                    # (M, k+1)
        y_hat_all = (x_all * beta_all).sum(axis=1)                 # (M,)

        # Predicted (prior) model probabilities — pure forgetting, no y_t
        pi_pred = pi_all**alpha
        pi_pred /= pi_pred.sum()

        # ── TVP prediction step (also pre-update) ─────────────────────────
        P_tvp_pred = P_tvp / lam
        y_hat_tvp  = float(x_full @ beta_tvp)

        # ── Record OOS forecasts BEFORE any update at time t ──────────────
        if t >= n_insample:
            # DMA: weighted average using prior (predicted) probabilities
            dma_fc[oos_idx] = float(pi_pred @ y_hat_all)

            # DMS: highest-prior-probability model
            best_m = int(np.argmax(pi_pred))
            dms_fc[oos_idx] = float(y_hat_all[best_m])

            # TVP: full model forecast (uses beta_tvp from the last update)
            tvp_fc[oos_idx] = y_hat_tvp

            # Diagnostics on the prior model-probability vector
            diag_pi_max[oos_idx]   = float(pi_pred[best_m])
            diag_n_eff[oos_idx]    = float(1.0 / np.sum(pi_pred**2))
            diag_top[oos_idx]      = best_m
            diag_top_size[oos_idx] = int(model_size[best_m])
            diag_mass10[oos_idx]   = float(np.sort(pi_pred)[-10:].sum())

            # Historical average over COMPLETED windows only: y[s] is fully
            # observed at origin t iff s + h - 1 <= t - 1, i.e. s < t-h+1
            n_done = t - h + 1
            ha_fc[oos_idx] = float(y[:n_done].mean()) if n_done > 0 else 0.0

            # DMA-averaged coefficients and PIPs use prior weights
            beta_dma_path[oos_idx] = (pi_pred[:, None] * beta_all * inclusion).sum(axis=0)
            pip_path[oos_idx]      = (pi_pred[:, None] * inclusion[:, 1:]).sum(axis=0)

            oos_idx += 1

        # ── Update with the most recent COMPLETED window ──────────────────
        # y[u] spans months u .. u+h-1 and is fully observed at the end of
        # month t = u + h - 1, i.e. u = t - h + 1. For h=1, u = t and this
        # block is identical to the standard one-step recursion. For h>1 the
        # filter runs with an (h-1)-month delay, which purges the overlap-
        # induced lookahead: no month inside the current forecast window
        # (t .. t+h-1) has yet influenced beta, P, H, or pi.
        u = t - h + 1
        if u < 0:
            # No completed window exists yet (first h-1 months only, deep in
            # burn-in). Leave the state at its prior; the P inflation is
            # deliberately not applied so the diffuse prior is not inflated
            # before any data arrive.
            continue

        y_u      = y[u]
        x_full_u = np.concatenate([[1.0], X[u]])
        x_all_u  = np.where(inclusion, x_full_u[None, :], 0.0)     # (M, k+1)

        # Model-by-model fit of the completed observation u
        y_hat_all_u = (x_all_u * beta_all).sum(axis=1)             # (M,)
        u_hat_all   = y_u - y_hat_all_u                            # (M,)
        F_all       = (x_all_u**2 * P_pred).sum(axis=1) + H_all    # (M,)

        # Normal predictive log-likelihood (numerically stable)
        log_f = -0.5 * (np.log(2.0 * np.pi * np.maximum(F_all, 1e-12))
                        + u_hat_all**2 / np.maximum(F_all, 1e-12))
        log_f -= log_f.max()  # subtract max for stability
        f_N   = np.exp(log_f)

        # Posterior model probabilities
        pi_all = pi_pred * f_N
        pi_all /= pi_all.sum()

        # TVP posterior update (same completed observation u)
        y_hat_tvp_u = float(x_full_u @ beta_tvp)
        u_hat_tvp   = y_u - y_hat_tvp_u
        F_tvp       = float(x_full_u @ P_tvp_pred @ x_full_u) + H_tvp
        G_tvp       = P_tvp_pred @ x_full_u / F_tvp
        beta_tvp    = beta_tvp + G_tvp * u_hat_tvp
        P_tvp       = P_tvp_pred - np.outer(G_tvp, x_full_u) @ P_tvp_pred
        H_tvp       = kappa * H_tvp + (1 - kappa) * u_hat_tvp**2

        # Kalman update for all models
        G_all     = P_pred * x_all_u / np.maximum(F_all[:, None], 1e-12)
        beta_all  = beta_all + G_all * u_hat_all[:, None]
        P_all     = P_pred - G_all * x_all_u * P_pred
        P_all     = np.maximum(P_all, 1e-10)  # numerical floor
        H_all     = kappa * H_all + (1 - kappa) * u_hat_all**2

        if verbose and (t + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t+1:3d}/{T}  elapsed={elapsed:.0f}s")

    runtime = time.time() - t_start
    if verbose:
        print(f"  Completed in {runtime:.1f}s")

    # ── Slice actual OOS returns ──────────────────────────────────────────
    y_oos    = y[n_insample:]
    oos_dates = dates[n_insample:] if dates is not None else None

    # ── Random walk benchmark (RW): forecast = 0 (driftless RW) ─────────
    rw_fc    = np.zeros(T_oos)
    err_rw   = y_oos - rw_fc
    err_dma  = y_oos - dma_fc
    err_dms  = y_oos - dms_fc
    err_tvp  = y_oos - tvp_fc
    err_ha   = y_oos - ha_fc

    # ── Evaluation metrics ────────────────────────────────────────────────
    msfe = {
        'RW':     _msfe(err_rw),
        'DMA':    _msfe(err_dma),
        'DMS':    _msfe(err_dms),
        'TVP':    _msfe(err_tvp),
        'HA_exp': _msfe(err_ha),
    }
    r2_oos = {
        'DMA':    _r2_oos(err_dma, err_rw),
        'DMS':    _r2_oos(err_dms, err_rw),
        'TVP':    _r2_oos(err_tvp, err_rw),
        'HA_exp': _r2_oos(err_ha,  err_rw),
    }
    cw_stat, cw_pval = {}, {}
    for name, err_m, fc_m in [('DMA', err_dma, dma_fc),
                               ('DMS', err_dms, dms_fc),
                               ('TVP', err_tvp, tvp_fc),
                               ('HA_exp', err_ha, ha_fc)]:
        stat, pval = _clark_west(err_rw, err_m, rw_fc, fc_m)
        cw_stat[name] = stat
        cw_pval[name] = pval

    result = DMAResult(
        dma_forecasts   = dma_fc,
        dms_forecasts   = dms_fc,
        tvp_forecasts   = tvp_fc,
        dates           = oos_dates,
        actual          = y_oos,
        pip_path        = pip_path,
        beta_dma_path   = beta_dma_path,
        msfe            = msfe,
        r2_oos          = r2_oos,
        cw_stat         = cw_stat,
        cw_pval         = cw_pval,
        params          = {'lambda': lam, 'alpha': alpha, 'kappa': kappa, 'h': h,
                           'k': k, 'M': M, 'T': T,
                           'n_insample': n_insample, 'T_oos': T_oos},
        runtime_seconds = runtime,
        diagnostics     = {'pi_max': diag_pi_max, 'n_eff': diag_n_eff,
                           'top_model': diag_top, 'top_size': diag_top_size,
                           'mass_top10': diag_mass10},
    )

    return result


# ---------------------------------------------------------------------------
# Multi-horizon forecasting (direct method)
# ---------------------------------------------------------------------------

def run_dma_multihorizon(y: np.ndarray,
                         X: np.ndarray,
                         n_insample: int,
                         horizons: list[int] = [1, 2, 3, 6],
                         lam:   float = DEFAULT_LAMBDA,
                         alpha: float = DEFAULT_ALPHA,
                         kappa: float = DEFAULT_KAPPA,
                         dates: Optional[pd.DatetimeIndex] = None,
                         verbose: bool = True) -> dict[int, DMAResult]:
    """
    Run DMA/DMS for multiple forecast horizons using the direct method.

    The direct method re-estimates a separate model for each horizon h using
    h-period holding returns as the target variable:
        r^h_{t+h} = 100 * (S_{t+h} / S_t - 1)
    and predictors X_t (same as one-step, lagged h periods).

    This matches Section 5.4 of Buncic & Moretto (5).

    Parameters
    ----------
    horizons : list[int]   forecast horizons in months (default [1, 2, 3, 6])

    Returns
    -------
    dict mapping horizon -> DMAResult
    """
    T = len(y)
    results = {}

    for h in horizons:
        if verbose:
            print(f"\n{'='*55}")
            print(f"Horizon h={h}")
            print(f"{'='*55}")

        # h-period holding return aligned with X_t (which carries info from t-1):
        # y_h[t] = 100 * (prod_{j=0}^{h-1} (1 + r_{t+j}/100) - 1)
        # For h=1 this collapses to y_h[t] = y[t], matching the single-horizon
        # convention. The last (h-1) rows are NaN and dropped via `valid`.
        y_h = np.full(T, np.nan)
        for t in range(T - h + 1):
            compound = np.prod(1.0 + y[t:t+h] / 100.0) - 1.0
            y_h[t] = 100.0 * compound

        # Direct method: align X_t with y^h_{t} (target is h steps ahead)
        # Use the same X_t as predictor; drop last h rows where y_h is NaN
        valid = ~np.isnan(y_h)
        y_h_valid = y_h[valid]
        X_valid   = X[valid]
        dates_h   = dates[valid] if dates is not None else None

        n_is_h = n_insample  # same burn-in
        if len(y_h_valid) <= n_is_h:
            print(f"  Skipping h={h}: insufficient observations")
            continue

        results[h] = run_dma_dms(
            y=y_h_valid, X=X_valid,
            n_insample=n_is_h,
            lam=lam, alpha=alpha, kappa=kappa,
            h=h,
            dates=dates_h,
            verbose=verbose
        )

    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_evaluation_table(result: DMAResult,
                            horizon: int = 1,
                            include_ols: bool = False) -> pd.DataFrame:
    """
    Print Table 2 from Buncic & Moretto (2015) — out-of-sample forecast results.

    Columns: Model | MSFE | Relative MSFE | R2_oos (%) | CW-stat | p-value

    Parameters
    ----------
    result       : DMAResult from run_dma_dms
    horizon      : int       forecast horizon (for display only)
    include_ols  : bool      placeholder for adding OLS rows from 03_ml_models

    Returns
    -------
    pd.DataFrame   formatted results table
    """
    rw_msfe = result.msfe['RW']
    rows = []

    # Random walk benchmark
    rows.append({
        'Model':         'Random Walk (RW)',
        'MSFE':          round(rw_msfe, 4),
        'Relative MSFE': 1.0000,
        'R2_oos (%)':    '—',
        'CW-stat':       '—',
        'p-value':       '—',
    })

    # Other models
    for name in ['HA_exp', 'TVP', 'DMS', 'DMA']:
        label = {
            'HA_exp': 'HA (expanding)',
            'TVP':    f'TVP (lambda={result.params["lambda"]})',
            'DMS':    f'DMS (alpha={result.params["alpha"]}, lambda={result.params["lambda"]})',
            'DMA': f'DMA (h={result.params.get("h", 1)}, '
                   f'alpha={result.params["alpha"]}, lambda={result.params["lambda"]})',
        }[name]
        rows.append({
            'Model':         label,
            'MSFE':          round(result.msfe[name], 4),
            'Relative MSFE': round(result.msfe[name] / rw_msfe, 4),
            'R2_oos (%)':    round(result.r2_oos[name] * 100, 4),
            'CW-stat':       round(result.cw_stat[name], 4),
            'p-value':       round(result.cw_pval[name], 4),
        })

    df = pd.DataFrame(rows).set_index('Model')

    print(f"\nTable: {horizon}-step-ahead out-of-sample forecast results")
    print(f"Out-of-sample period: {result.dates[0].strftime('%b %Y') if result.dates is not None else 'N/A'}"
          f" to {result.dates[-1].strftime('%b %Y') if result.dates is not None else 'N/A'}")
    print(f"T_oos = {result.params['T_oos']}, T_is = {result.params['n_insample']}")
    print()
    print(df.to_string())
    return df


def print_pip_summary(result: DMAResult,
                      predictor_names: list[str],
                      top_n: int = 10) -> pd.DataFrame:
    """
    Print summary of time-averaged posterior inclusion probabilities.

    Parameters
    ----------
    result          : DMAResult
    predictor_names : list of predictor variable names (length k)
    top_n           : show top N predictors by mean PIP

    Returns
    -------
    pd.DataFrame with columns: Predictor | Mean PIP | Min PIP | Max PIP
    """
    mean_pip = result.pip_path.mean(axis=0)
    min_pip  = result.pip_path.min(axis=0)
    max_pip  = result.pip_path.max(axis=0)

    df = pd.DataFrame({
        'Predictor': predictor_names,
        'Mean PIP':  np.round(mean_pip, 4),
        'Min PIP':   np.round(min_pip,  4),
        'Max PIP':   np.round(max_pip,  4),
    }).sort_values('Mean PIP', ascending=False)

    print(f"\nPosterior Inclusion Probabilities (time-averaged, top {top_n}):")
    print(df.head(top_n).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Quick validation — run as script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline import build_dataset, PREDICTOR_COLS

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'

    print("=" * 60)
    print("02_dma_dms.py  —  DMA/DMS estimation")
    print("=" * 60)

    # Load data
    df, meta = build_dataset(data_dir=data_dir)

    y = df['r_copper'].values
    X = df[PREDICTOR_COLS].values
    dates = pd.DatetimeIndex(df['date'])
    k = X.shape[1]  # 18

    # In-sample / out-of-sample split
    # Use ~120 months (10 years) as burn-in, consistent with 1/3 of original sample
    # Actual split: first 120 obs in-sample, remaining 216 OOS
    n_insample = 120
    print(f"\nSplit: {n_insample} in-sample, {len(y)-n_insample} out-of-sample")
    print(f"IS period:  {dates[0].strftime('%b %Y')} - {dates[n_insample-1].strftime('%b %Y')}")
    print(f"OOS period: {dates[n_insample].strftime('%b %Y')} - {dates[-1].strftime('%b %Y')}")

    # Run DMA/DMS (h=1)
    print(f"\nRunning DMA/DMS (h=1)...")
    result = run_dma_dms(
        y=y, X=X,
        n_insample=n_insample,
        lam=DEFAULT_LAMBDA,
        alpha=DEFAULT_ALPHA,
        kappa=DEFAULT_KAPPA,
        dates=dates,
        verbose=True,
    )

    # Print evaluation table
    print_evaluation_table(result, horizon=1)

    # Print PIP summary
    pred_names = [c.replace('x_', '') for c in PREDICTOR_COLS]
    print_pip_summary(result, predictor_names=pred_names)

    print(f"\nRuntime: {result.runtime_seconds:.1f}s")