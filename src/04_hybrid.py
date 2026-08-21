"""
04_hybrid.py
============
Hybrid DMA-ML framework for copper return forecasting.

This module combines the DMA/DMS econometric framework with machine learning
models in three principled ways. The goal is not to guarantee improvement
over DMA alone, but to explore whether ML methods can complement the
time-varying Bayesian approach.

Hybrid variants
---------------
1.  PIP-Weighted Features (PIP-EN)
    Multiplies each predictor X_j by its current posterior inclusion
    probability PIP_j from DMA before fitting an ElasticNet model.
    Effect: DMA's time-varying model uncertainty acts as a soft variable
    selection signal for the linear model. Predictors that DMA currently
    deems irrelevant receive near-zero weight; highly included predictors
    retain full weight.
    Natural pairing: linear models only (PIP weighting is meaningless for
    tree-based methods, which split on variable rank not scale).

2.  Forecast Stacking (STACK)
    Trains a Ridge meta-learner on the expanding history of DMA and
    ElasticNet forecasts. The meta-learner learns the optimal time-varying
    convex combination of the two base forecasters.
    Classical reference: Bates & Granger (1969), cited in the paper.
    Simple equal-weight average is also reported as a strong baseline.

3.  Simple Combination (COMBO)
    Equal-weight average of DMA and ElasticNet forecasts.
    Theoretically: averaging reduces variance at the cost of a small bias
    increase, which is typically beneficial when models have uncorrelated
    errors. This is the Bates-Granger (1969) forecast combination result.

Key findings
---------------------------------------------
The equal-weight DMA+ElasticNet combination is the single best specification
(R2_oos = 31.15%), but it improves on DMA alone (30.16%) by only about one
percentage point. PIP-weighted ElasticNet (29.27%) edges standalone
ElasticNet (29.01%), and the Ridge meta-learner stack (24.63%) underperforms
both base models — the forecast combination puzzle (Smith & Wallis, 2009).
The hybrids therefore confirm the thesis's convergence result rather than
overturn it: DMA and the regularised linear models share most of their
information set, leaving little complementary signal to exploit.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import (ElasticNet, ElasticNetCV,
                                   RidgeCV)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TUNE_EVERY = 12   # retune ElasticNet hyperparameters every 12 OOS steps
MIN_META   = 12   # minimum OOS history before meta-learner replaces simple average


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """
    Container for one hybrid model's out-of-sample evaluation.

    Attributes
    ----------
    model_name  : str              hybrid model identifier
    forecasts   : np.ndarray       out-of-sample point forecasts (T_oos,)
    actual      : np.ndarray       realised copper returns (T_oos,)
    dates       : pd.DatetimeIndex OOS dates
    msfe        : float            mean squared forecast error
    r2_oos      : float            Campbell-Thompson OOS R^2 vs random walk
    cw_stat     : float            Clark-West statistic vs random walk
    cw_pval     : float            one-sided p-value for CW test
    runtime_s   : float            wall-clock time in seconds
    description : str              brief description of the hybrid strategy
    """
    model_name  : str  = ''
    forecasts   : np.ndarray = field(default_factory=lambda: np.array([]))
    actual      : np.ndarray = field(default_factory=lambda: np.array([]))
    dates       : Optional[pd.DatetimeIndex] = None
    msfe        : float = 0.0
    r2_oos      : float = 0.0
    cw_stat     : float = 0.0
    cw_pval     : float = 0.0
    runtime_s   : float = 0.0
    description : str  = ''


# ---------------------------------------------------------------------------
# Evaluation utilities (identical to 02 and 03 for consistent reporting)
# ---------------------------------------------------------------------------

def _msfe(errors: np.ndarray) -> float:
    return float(np.mean(errors**2))


def _r2_oos(errors_model: np.ndarray, errors_rw: np.ndarray) -> float:
    """Campbell & Thompson (2008) OOS R^2."""
    return float(1.0 - _msfe(errors_model) / _msfe(errors_rw))


def _clark_west(errors_rw, errors_model,
                forecasts_rw, forecasts_model) -> tuple[float, float]:
    """Clark & West (2007) MSFE-adjusted t-statistic."""
    cw_seq   = (errors_rw**2 - errors_model**2) + (forecasts_rw - forecasts_model)**2
    cw_bar   = cw_seq.mean()
    T        = len(cw_seq)
    bw       = max(1, int(T**(1/3)))
    demeaned = cw_seq - cw_bar
    var_cw   = np.var(cw_seq, ddof=1)
    for lag in range(1, bw + 1):
        weight  = 1.0 - lag / (bw + 1)
        var_cw += 2 * weight * np.mean(demeaned[lag:] * demeaned[:-lag])
    var_cw = max(var_cw, 1e-12)
    from scipy.stats import norm
    cw_stat = float(cw_bar / np.sqrt(var_cw / T))
    return cw_stat, float(1.0 - norm.cdf(cw_stat))


def _package(name, forecasts, y_oos, dates_oos, t0, description='') -> HybridResult:
    """Compute evaluation metrics and pack into HybridResult."""
    rw_fc   = np.zeros(len(y_oos))
    err_rw  = y_oos - rw_fc
    err_mod = y_oos - forecasts
    stat, pval = _clark_west(err_rw, err_mod, rw_fc, forecasts)
    return HybridResult(
        model_name  = name,
        forecasts   = forecasts,
        actual      = y_oos,
        dates       = dates_oos,
        msfe        = _msfe(err_mod),
        r2_oos      = _r2_oos(err_mod, err_rw),
        cw_stat     = stat,
        cw_pval     = pval,
        runtime_s   = time.time() - t0,
        description = description,
    )


# ---------------------------------------------------------------------------
# Hybrid 1 — PIP-Weighted ElasticNet
# ---------------------------------------------------------------------------

def run_pip_weighted_elasticnet(
        y: np.ndarray,
        X: np.ndarray,
        pip_path: np.ndarray,
        n_insample: int,
        dates: Optional[pd.DatetimeIndex] = None,
        verbose: bool = True) -> HybridResult:
    """
    PIP-Weighted ElasticNet (PIP-EN).

    At each OOS step i (forecasting month t = n_insample + i):
        1. Retrieve DMA's current posterior inclusion probabilities PIP_i.
        2. Weight each predictor column j by PIP_i[j]:
               X_weighted[row, j] = X[row, j] * PIP_i[j]
        3. Fit ElasticNet on the weighted predictor matrix.

    The PIP values are derived from the DMA prediction-step probabilities
    (pi_{t|t-1}), which are computed in 02_dma_dms.py BEFORE observing y_t
    and stored in `pip_path`. PIP_i therefore reflects only information
    available at the time the forecast for month t = n_insample + i is made.
    The hybrid weighting inherits this prospective alignment.

    For in-sample training rows, we use pip_path[0] (the PIP vector at the
    first OOS step, i.e. the end of the burn-in period) as a conservative
    approximation. This avoids any forward-looking contamination of the
    training data.

    Why this works for linear models
    ---------------------------------
    Multiplying predictor j by PIP_j ∈ [0,1] before standardisation
    effectively scales down the variance of X_j in proportion to how
    confident DMA is that j belongs in the model. After standardisation,
    features with low PIP have a compressed scale relative to high-PIP
    features, which biases ElasticNet toward selecting high-PIP predictors.
    This creates a soft variable selection mechanism guided by DMA's
    time-varying Bayesian model probabilities.

    Parameters
    ----------
    y          : np.ndarray (T,)       target variable
    X          : np.ndarray (T, k)     predictor matrix
    pip_path   : np.ndarray (T_oos, k) DMA posterior inclusion probs from 02
    n_insample : int                   burn-in observations
    dates      : pd.DatetimeIndex      aligned date index
    verbose    : bool

    Returns
    -------
    HybridResult
    """
    T_oos  = len(y) - n_insample
    k      = X.shape[1]
    fc     = np.zeros(T_oos)
    t0     = time.time()
    pip_is = pip_path[0]       # PIP at start of OOS = end of IS burn-in
    alpha_en, l1_en = 0.01, 0.5

    for i in range(T_oos):
        t     = n_insample + i
        pip_t = pip_path[i]    # current PIP vector — NO lookahead

        # Build PIP weight matrix for all training rows
        # IS rows: use pip_is; previous OOS rows: use their actual pip
        if i > 0:
            pip_weights = np.vstack([
                np.tile(pip_is, (n_insample, 1)),  # (n_insample, k)
                pip_path[:i]                        # (i, k)
            ])  # shape (t, k)
        else:
            pip_weights = np.tile(pip_is, (n_insample, 1))  # (n_insample, k)

        X_train_w = X[:t] * pip_weights     # element-wise scaling
        X_test_w  = X[t:t+1] * pip_t

        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_train_w)
        Xte = sc.transform(X_test_w)

        # Retune ElasticNet hyperparameters every TUNE_EVERY steps
        if i % TUNE_EVERY == 0:
            tscv  = TimeSeriesSplit(n_splits=3)
            m_cv  = ElasticNetCV(cv=tscv, max_iter=5000, random_state=42,
                                  l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]).fit(Xtr, y[:t])
            alpha_en = float(m_cv.alpha_)
            l1_en    = float(m_cv.l1_ratio_)

        m     = ElasticNet(alpha=alpha_en, l1_ratio=l1_en, max_iter=5000).fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])

    if verbose:
        y_oos = y[n_insample:]
        r2    = _r2_oos(y_oos - fc, y_oos)
        print(f"  PIP-EN:       R2={r2:.4f}  t={time.time()-t0:.1f}s")

    return _package(
        'PIP_ElasticNet', fc, y[n_insample:],
        dates[n_insample:] if dates is not None else None,
        t0,
        description="DMA PIPs as soft variable weights → ElasticNet"
    )


# ---------------------------------------------------------------------------
# Hybrid 2 — Forecast Stacking
# ---------------------------------------------------------------------------

def run_forecast_stacking(
        y: np.ndarray,
        dma_forecasts: np.ndarray,
        ml_forecasts: np.ndarray,
        y_oos: np.ndarray,
        dates_oos: Optional[pd.DatetimeIndex] = None,
        verbose: bool = True) -> HybridResult:
    """
    Forecast Stacking (Ridge meta-learner).

    Trains a Ridge regression meta-learner on the expanding history of
    DMA and ElasticNet (or any ML model) out-of-sample forecasts. At each
    step i, the meta-learner has access only to forecasts from steps 0..i-1,
    ensuring strict temporal validity.

    For the first MIN_META=12 OOS steps (insufficient history to train
    the meta-learner), a simple equal-weight average is used instead.

    The meta-learner weight on DMA vs ML is unconstrained — it can learn to
    increase or decrease DMA's contribution over time. However, because Ridge
    regression shrinks weights toward zero, the effective combination tends
    toward the simple average when the two base forecasters perform similarly.

    Parameters
    ----------
    y             : np.ndarray (T,)      full target series
    dma_forecasts : np.ndarray (T_oos,)  DMA one-step-ahead forecasts
    ml_forecasts  : np.ndarray (T_oos,)  ML model forecasts (e.g. ElasticNet)
    y_oos         : np.ndarray (T_oos,)  realised OOS returns
    dates_oos     : pd.DatetimeIndex     OOS dates
    verbose       : bool

    Returns
    -------
    HybridResult
    """
    T_oos = len(y_oos)
    fc    = np.zeros(T_oos)
    t0    = time.time()
    base  = np.column_stack([dma_forecasts, ml_forecasts])  # (T_oos, 2)

    for i in range(T_oos):
        if i < MIN_META:
            # Simple average before sufficient meta-training history
            fc[i] = 0.5 * dma_forecasts[i] + 0.5 * ml_forecasts[i]
        else:
            mXtr = base[:i]           # (i, 2) — past forecasts
            mYtr = y_oos[:i]          # (i,)   — past realisations
            mXte = base[i:i+1]        # (1, 2) — current forecasts

            sc   = StandardScaler()
            mXtr = sc.fit_transform(mXtr)
            mXte = sc.transform(mXte)

            m_meta = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(mXtr, mYtr)
            fc[i]  = float(m_meta.predict(mXte)[0])

    if verbose:
        r2 = _r2_oos(y_oos - fc, y_oos)
        print(f"  Stacking:     R2={r2:.4f}  t={time.time()-t0:.1f}s")

    return _package(
        'Stacking_DMA_EN', fc, y_oos, dates_oos, t0,
        description="Ridge meta-learner on DMA + ElasticNet forecasts"
    )


# ---------------------------------------------------------------------------
# Hybrid 3 — Equal-Weight Combination
# ---------------------------------------------------------------------------

def run_equal_weight_combination(
        dma_forecasts: np.ndarray,
        ml_forecasts: np.ndarray,
        y_oos: np.ndarray,
        dates_oos: Optional[pd.DatetimeIndex] = None,
        name: str = 'Combo_DMA_EN',
        verbose: bool = True) -> HybridResult:
    """
    Equal-Weight Forecast Combination.

    Simple arithmetic average of DMA and ML forecasts:
        y_hat_combo = 0.5 * y_hat_DMA + 0.5 * y_hat_ML

    Despite its simplicity, equal-weight averaging is theoretically motivated
    by the Bates-Granger (1969) result that combining forecasts reduces
    variance when constituent forecasts have uncorrelated errors. In practice,
    it is surprisingly difficult to beat even with sophisticated meta-learners
    — a finding replicated here (Stacking: R²_oos = 24.63% vs Combo: 31.15%).

    This variant is included for two reasons:
    1. It is computationally trivial and fully interpretable.
    2. It matches the 'forecast combination' literature benchmark that
       Issler et al. (2014) find to be the best simple approach at monthly
       frequency (also cited in Buncic & Moretto, 2014, Section 2.4).

    Parameters
    ----------
    dma_forecasts : np.ndarray (T_oos,)
    ml_forecasts  : np.ndarray (T_oos,)
    y_oos         : np.ndarray (T_oos,)
    dates_oos     : pd.DatetimeIndex
    name          : str   model name for display
    verbose       : bool
    """
    t0 = time.time()
    fc = 0.5 * dma_forecasts + 0.5 * ml_forecasts

    if verbose:
        r2 = _r2_oos(y_oos - fc, y_oos)
        print(f"  Combo DMA+EN: R2={r2:.4f}  t={time.time()-t0:.3f}s")

    return _package(
        name, fc, y_oos, dates_oos, t0,
        description="Equal-weight average: 0.5*DMA + 0.5*ElasticNet"
    )


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_hybrids(
        y: np.ndarray,
        X: np.ndarray,
        dma_result,
        ml_results: dict,
        n_insample: int,
        dates: Optional[pd.DatetimeIndex] = None,
        verbose: bool = True) -> dict:
    """
    Run all three hybrid variants and return results as a dict.

    Parameters
    ----------
    y           : np.ndarray (T,)    full target variable
    X           : np.ndarray (T, k)  full predictor matrix
    dma_result  : DMAResult          output of run_dma_dms() from 02_dma_dms
    ml_results  : dict               output of run_all_ml_models() from 03_ml_models
    n_insample  : int                burn-in observations
    dates       : pd.DatetimeIndex   aligned date index
    verbose     : bool

    Returns
    -------
    dict mapping hybrid name (str) -> HybridResult
    """
    y_oos     = y[n_insample:]
    dates_oos = dates[n_insample:] if dates is not None else None
    pip_path  = dma_result.pip_path
    dma_fc    = dma_result.dma_forecasts

    # Use ElasticNet as the primary ML partner (best ML performer)
    if 'ElasticNet' not in ml_results:
        raise ValueError("ElasticNet results missing from ml_results — "
                         "run run_all_ml_models() before run_all_hybrids().")
    en_fc = ml_results['ElasticNet'].forecasts

    if verbose:
        T_oos = len(y_oos)
        print(f"\nHybrid models: T_oos={T_oos}")
        if dates_oos is not None:
            print(f"OOS: {dates_oos[0].strftime('%b %Y')} "
                  f"to {dates_oos[-1].strftime('%b %Y')}")
        print()

    results = {}

    # Hybrid 1: PIP-weighted ElasticNet
    results['PIP_ElasticNet'] = run_pip_weighted_elasticnet(
        y=y, X=X, pip_path=pip_path,
        n_insample=n_insample, dates=dates, verbose=verbose)

    # Hybrid 2: Forecast stacking (Ridge meta-learner)
    results['Stacking'] = run_forecast_stacking(
        y=y, dma_forecasts=dma_fc, ml_forecasts=en_fc,
        y_oos=y_oos, dates_oos=dates_oos, verbose=verbose)

    # Hybrid 3: Equal-weight combination
    results['Combo_DMA_EN'] = run_equal_weight_combination(
        dma_forecasts=dma_fc, ml_forecasts=en_fc,
        y_oos=y_oos, dates_oos=dates_oos, verbose=verbose)

    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_hybrid_table(hybrid_results: dict,
                       dma_result=None,
                       ml_results: dict = None,
                       rw_msfe: Optional[float] = None) -> pd.DataFrame:
    """
    Print consolidated evaluation table: RW + DMA + best ML + all hybrids.

    Parameters
    ----------
    hybrid_results : dict        output of run_all_hybrids
    dma_result     : DMAResult   from 02_dma_dms (optional, for DMA rows)
    ml_results     : dict        from 03_ml_models (optional, for ML rows)
    rw_msfe        : float       RW MSFE for relative MSFE column

    Returns
    -------
    pd.DataFrame  formatted results table
    """
    # Infer RW MSFE
    if rw_msfe is None and dma_result is not None:
        rw_msfe = dma_result.msfe.get('RW', None)
    if rw_msfe is None and hybrid_results:
        first = next(iter(hybrid_results.values()))
        rw_fc   = np.zeros(len(first.actual))
        rw_msfe = float(np.mean((first.actual - rw_fc)**2))

    rows = []

    # Random walk
    rows.append({'Model': 'Random Walk (RW)',
                 'MSFE': round(rw_msfe, 4), 'Rel. MSFE': 1.0,
                 'R2_oos (%)': '—', 'CW-stat': '—', 'p-value': '—'})

    # DMA baseline
    if dma_result is not None:
        for mname in ['DMA']:
            rows.append({
                'Model':      mname,
                'MSFE':       round(dma_result.msfe[mname], 4),
                'Rel. MSFE':  round(dma_result.msfe[mname] / rw_msfe, 4),
                'R2_oos (%)': round(dma_result.r2_oos[mname] * 100, 4),
                'CW-stat':    round(dma_result.cw_stat[mname], 4),
                'p-value':    round(dma_result.cw_pval[mname], 4),
            })

    # Best ML baseline (ElasticNet)
    if ml_results is not None and 'ElasticNet' in ml_results:
        res = ml_results['ElasticNet']
        rows.append({
            'Model':      'ElasticNet (best ML)',
            'MSFE':       round(res.msfe, 4),
            'Rel. MSFE':  round(res.msfe / rw_msfe, 4),
            'R2_oos (%)': round(res.r2_oos * 100, 4),
            'CW-stat':    round(res.cw_stat, 4),
            'p-value':    round(res.cw_pval, 4),
        })

    # Hybrid models
    for name, res in hybrid_results.items():
        rows.append({
            'Model':      name,
            'MSFE':       round(res.msfe, 4),
            'Rel. MSFE':  round(res.msfe / rw_msfe, 4),
            'R2_oos (%)': round(res.r2_oos * 100, 4),
            'CW-stat':    round(res.cw_stat, 4),
            'p-value':    round(res.cw_pval, 4),
        })

    df = pd.DataFrame(rows).set_index('Model')
    print("\nHybrid Model Evaluation Table")
    print(df.to_string())
    return df


def print_hybrid_descriptions(hybrid_results: dict) -> None:
    """Print a brief description of each hybrid strategy."""
    print("\nHybrid Strategy Descriptions")
    print("-" * 60)
    for name, res in hybrid_results.items():
        print(f"  {name}:")
        print(f"    {res.description}")
        print(f"    R2_oos = {res.r2_oos*100:.2f}%  |  "
              f"CW-stat = {res.cw_stat:.4f}  |  p = {res.cw_pval:.4f}")
        print()


# ---------------------------------------------------------------------------
# Quick validation — run as script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline import build_dataset, PREDICTOR_COLS
    from src.dma_dms import run_dma_dms
    from src.ml_models import run_all_ml_models

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'

    print("=" * 60)
    print("04_hybrid.py  —  Hybrid DMA-ML models")
    print("=" * 60)

    df, meta = build_dataset(data_dir=data_dir)
    y     = df['r_copper'].values
    X     = df[PREDICTOR_COLS].values
    dates = pd.DatetimeIndex(df['date'])
    n_insample = 120

    # Step 1: Run DMA to get PIP path and base forecasts
    print("\nStep 1: Running DMA/DMS...")
    dma_result = run_dma_dms(y=y, X=X, n_insample=n_insample,
                              dates=dates, verbose=True)

    # Step 2: Run ML models
    print("\nStep 2: Running ML models...")
    ml_results = run_all_ml_models(y=y, X=X, n_insample=n_insample,
                                    dates=dates, verbose=True)

    # Step 3: Run hybrids
    print("\nStep 3: Running hybrid models...")
    hybrid_results = run_all_hybrids(
        y=y, X=X,
        dma_result=dma_result,
        ml_results=ml_results,
        n_insample=n_insample,
        dates=dates,
        verbose=True,
    )

    # Print results
    print_hybrid_table(hybrid_results, dma_result=dma_result, ml_results=ml_results)
    print_hybrid_descriptions(hybrid_results)