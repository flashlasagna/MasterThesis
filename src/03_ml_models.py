"""
03_ml_models.py
===============
Machine learning models for copper return forecasting.
Companion to the DMA/DMS replication in 02_dma_dms.py.

Models implemented
------------------
1.  OLS (expanding window)    — direct comparator to paper's OLS baseline
2.  Ridge regression          — L2 regularisation; shrinks all coefficients
3.  LASSO                     — L1 regularisation; performs automatic variable selection
4.  Elastic Net               — convex combination of L1 and L2 penalties
5.  Random Forest             — bagging ensemble of decision trees
6.  XGBoost                   — gradient boosting (best performer in weather project)
7.  Bayesian Ridge            — probabilistic ridge with automatic hyperparameter tuning;
                                bridges econometrics and ML (close in spirit to DMA/DMS)

All models use an expanding window walk-forward scheme:
    - At OOS step i (forecasting t = n_insample + i), train on ALL data up to t.
    - Hyperparameters are retuned every TUNE_EVERY=12 steps using TimeSeriesSplit CV.
    - Predictors are standardised (zero mean, unit variance) at each step using
      only in-sample data to prevent information leakage.

This scheme is identical to the "expanding window OLS" benchmark in the paper,
ensuring a fair comparison across all models.

Usage
-----
    from src.ml_models import run_all_ml_models, MLResult
    results = run_all_ml_models(y, X, n_insample=120, dates=dates)

Output
------
    dict mapping model_name -> MLResult dataclass with:
        forecasts, actual, dates, msfe, r2_oos, cw_stat, cw_pval, runtime
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNet, ElasticNetCV,
                                   Lasso, LassoCV, LinearRegression,
                                   Ridge, RidgeCV)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hyperparameter retuning frequency (every N out-of-sample steps)
TUNE_EVERY = 12   # annual retuning — balances adaptability and computation


# Random Forest defaults (shallow trees to prevent overfitting on small sample)
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = 4
RF_MIN_SAMPLES   = 5

# XGBoost defaults
XGB_N_ESTIMATORS  = 100
XGB_MAX_DEPTH     = 3
XGB_LR            = 0.05
XGB_SUBSAMPLE     = 0.8
XGB_COLSAMPLE     = 0.8

# MLP (feed-forward network) -- tuned annually by TimeSeriesSplit CV
MLP_HIDDEN_GRID = [(8,), (16,)]
MLP_ALPHA_GRID  = [10.0, 30.0, 100.0, 300.0, 1000.0]
MLP_N_SEEDS     = 5
MLP_MAX_ITER    = 500


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLResult:
    """
    Container for one ML model's out-of-sample evaluation.

    Attributes
    ----------
    model_name  : str            model identifier
    forecasts   : np.ndarray     out-of-sample point forecasts (T_oos,)
    actual      : np.ndarray     realised copper returns (T_oos,)
    dates       : pd.DatetimeIndex  OOS dates
    msfe        : float          mean squared forecast error
    r2_oos      : float          Campbell-Thompson OOS R^2 vs random walk
    cw_stat     : float          Clark-West statistic vs random walk
    cw_pval     : float          one-sided p-value for CW test
    runtime_s   : float          wall-clock time in seconds
    feature_imp : np.ndarray     feature importances (None for linear models)
    coef_path   : np.ndarray     coefficient path over OOS period (T_oos, k+1)
                                 rows = time steps, cols = [intercept, x_1..x_k]
                                 None for tree-based models
    """
    model_name  : str            = ''
    forecasts   : np.ndarray     = field(default_factory=lambda: np.array([]))
    actual      : np.ndarray     = field(default_factory=lambda: np.array([]))
    dates       : Optional[pd.DatetimeIndex] = None
    msfe        : float          = 0.0
    r2_oos      : float          = 0.0
    cw_stat     : float          = 0.0
    cw_pval     : float          = 0.0
    runtime_s   : float          = 0.0
    feature_imp : Optional[np.ndarray] = None
    coef_path   : Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Evaluation utilities (mirrors 02_dma_dms.py for consistent reporting)
# ---------------------------------------------------------------------------

def _msfe(errors: np.ndarray) -> float:
    return float(np.mean(errors**2))


def _r2_oos(errors_model: np.ndarray, errors_rw: np.ndarray) -> float:
    """Campbell & Thompson (2008) OOS R^2."""
    return float(1.0 - _msfe(errors_model) / _msfe(errors_rw))


def _clark_west(errors_rw, errors_model,
                forecasts_rw, forecasts_model) -> tuple[float, float]:
    """
    Clark & West (2007) MSFE-adjusted t-statistic.
    Identical implementation to 02_dma_dms.py.
    """
    cw_seq  = (errors_rw**2 - errors_model**2) + (forecasts_rw - forecasts_model)**2
    cw_bar  = cw_seq.mean()
    T       = len(cw_seq)
    bw      = max(1, int(T**(1/3)))
    demeaned = cw_seq - cw_bar
    var_cw  = np.var(cw_seq, ddof=1)
    for lag in range(1, bw + 1):
        weight  = 1.0 - lag / (bw + 1)
        var_cw += 2 * weight * np.mean(demeaned[lag:] * demeaned[:-lag])
    var_cw  = max(var_cw, 1e-12)

    cw_stat = float(cw_bar / np.sqrt(var_cw / T))
    from scipy.stats import norm
    p_value = float(1.0 - norm.cdf(cw_stat))
    return cw_stat, p_value


def _package_result(name, forecasts, y_oos, dates_oos, t0,
                    feature_imp=None, coef_path=None) -> MLResult:
    """Compute all evaluation metrics and pack into MLResult."""
    rw_fc    = np.zeros(len(y_oos))
    err_rw   = y_oos - rw_fc
    err_mod  = y_oos - forecasts
    stat, pval = _clark_west(err_rw, err_mod, rw_fc, forecasts)
    return MLResult(
        model_name  = name,
        forecasts   = forecasts,
        actual      = y_oos,
        dates       = dates_oos,
        msfe        = _msfe(err_mod),
        r2_oos      = _r2_oos(err_mod, err_rw),
        cw_stat     = stat,
        cw_pval     = pval,
        runtime_s   = time.time() - t0,
        feature_imp = feature_imp,
        coef_path   = coef_path,
    )


# ---------------------------------------------------------------------------
# Individual model runners
# ---------------------------------------------------------------------------

def run_ols(y: np.ndarray, X: np.ndarray, n_insample: int,
            dates: Optional[pd.DatetimeIndex] = None,
            verbose: bool = True) -> MLResult:
    """
    Expanding window OLS regression.

    Direct comparator to the 'OLS (expanding window)' benchmark in
    Table 2 of Buncic & Moretto (2015). No regularisation, no feature
    selection — all 18 predictors plus intercept at every step.
    Predictors are standardised to make coefficients comparable.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    coef_path = np.zeros((T_oos, X.shape[1] + 1))
    t0    = time.time()

    for i in range(T_oos):
        t  = n_insample + i
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t])
        Xte = sc.transform(X[t:t+1])

        m = LinearRegression().fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])
        coef_path[i] = np.concatenate([[m.intercept_], m.coef_])

    if verbose:
        print(f"  OLS:          R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('OLS', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, coef_path=coef_path)


def run_ridge(y: np.ndarray, X: np.ndarray, n_insample: int,
              dates: Optional[pd.DatetimeIndex] = None,
              verbose: bool = True, gap: int = 0) -> MLResult:
    """
    Ridge regression (L2 regularisation).

    Penalises large coefficients by adding lambda*||beta||_2^2 to the OLS
    objective. All predictors are retained but shrunk toward zero. Best
    suited when many predictors are mildly relevant. Regularisation
    parameter is retuned every TUNE_EVERY steps via TimeSeriesSplit CV.
    """
    T_oos   = len(y) - n_insample
    fc      = np.zeros(T_oos)
    coef_path = np.zeros((T_oos, X.shape[1] + 1))
    alpha   = 1.0
    t0      = time.time()
    alphas  = np.logspace(-3, 4, 40)

    for i in range(T_oos):
        t  = n_insample + i
        t_train = t - gap  # last usable training row (exclusive); gap = h-1 for overlapping targets
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t_train])
        Xte = sc.transform(X[t:t+1])

        if i % TUNE_EVERY == 0:
            tscv  = TimeSeriesSplit(n_splits=3)
            m_cv  = RidgeCV(alphas=alphas, cv=tscv).fit(Xtr, y[:t_train])
            alpha = float(m_cv.alpha_)

        m = Ridge(alpha=alpha).fit(Xtr, y[:t_train])
        fc[i] = float(m.predict(Xte)[0])
        coef_path[i] = np.concatenate([[m.intercept_], m.coef_])

    if verbose:
        print(f"  Ridge:        R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('Ridge', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, coef_path=coef_path)


def run_lasso(y: np.ndarray, X: np.ndarray, n_insample: int,
              dates: Optional[pd.DatetimeIndex] = None,
              verbose: bool = True) -> MLResult:
    """
    LASSO regression (L1 regularisation).

    Penalises coefficients via lambda*||beta||_1. Unlike Ridge, LASSO
    performs automatic variable selection by setting irrelevant coefficients
    to exactly zero. Particularly relevant here as a data-driven counterpart
    to DMA's posterior inclusion probabilities — both select predictors, but
    LASSO does so deterministically while DMA maintains a probability
    distribution over subsets.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    coef_path = np.zeros((T_oos, X.shape[1] + 1))
    alpha = 0.01
    t0    = time.time()

    for i in range(T_oos):
        t  = n_insample + i
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t])
        Xte = sc.transform(X[t:t+1])

        if i % TUNE_EVERY == 0:
            tscv  = TimeSeriesSplit(n_splits=3)
            m_cv  = LassoCV(cv=tscv, max_iter=5000, random_state=42).fit(Xtr, y[:t])
            alpha = float(m_cv.alpha_)

        m = Lasso(alpha=alpha, max_iter=5000).fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])
        coef_path[i] = np.concatenate([[m.intercept_], m.coef_])

    if verbose:
        print(f"  LASSO:        R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('LASSO', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, coef_path=coef_path)


def run_elasticnet(y: np.ndarray, X: np.ndarray, n_insample: int,
                   dates: Optional[pd.DatetimeIndex] = None,
                   verbose: bool = True, gap: int = 0) -> MLResult:
    """
    Elastic Net regression (L1 + L2 regularisation).

    Minimises: OLS_loss + alpha*[rho*||beta||_1 + (1-rho)*||beta||_2^2]
    where rho (l1_ratio) controls the mix between LASSO and Ridge.
    Inherits LASSO's sparsity while retaining Ridge's grouping property
    (correlated predictors tend to be selected/dropped together).
    Both alpha and rho are tuned jointly via TimeSeriesSplit CV.
    gap : int
        Number of most-recent rows excluded from every training set.
        For h-period overlapping targets set gap = h - 1, so that only
        targets whose windows have fully completed by the forecast origin
        enter estimation (prevents overlap-induced lookahead).
    """
    T_oos  = len(y) - n_insample
    fc     = np.zeros(T_oos)
    coef_path = np.zeros((T_oos, X.shape[1] + 1))
    alpha  = 0.01
    l1_r   = 0.5
    t0     = time.time()

    for i in range(T_oos):
        t = n_insample + i
        t_train = t - gap  # last usable training row (exclusive)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t_train])
        Xte = sc.transform(X[t:t + 1])

        if i % TUNE_EVERY == 0:
            tscv = TimeSeriesSplit(n_splits=3)
            m_cv = ElasticNetCV(cv=tscv, max_iter=5000, random_state=42,
                                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]).fit(Xtr, y[:t_train])
            alpha = float(m_cv.alpha_);
            l1_r = float(m_cv.l1_ratio_)

        m = ElasticNet(alpha=alpha, l1_ratio=l1_r, max_iter=5000).fit(Xtr, y[:t_train])
        fc[i] = float(m.predict(Xte)[0])

    if verbose:
        print(f"  ElasticNet:   R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('ElasticNet', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, coef_path=coef_path)


def run_bayesian_ridge(y: np.ndarray, X: np.ndarray, n_insample: int,
                       dates: Optional[pd.DatetimeIndex] = None,
                       verbose: bool = True) -> MLResult:
    """
    Bayesian Ridge regression.

    Places a Gaussian prior over coefficients and estimates both the
    prior precision (alpha) and noise precision (lambda) from the data
    via type-II maximum likelihood (empirical Bayes). No cross-validation
    is needed — hyperparameters are tuned automatically at each step.

    This model is philosophically the closest to DMA/DMS among the ML
    models: both maintain uncertainty over model parameters, both use
    Bayesian updating, and both regularise via prior precision. The key
    difference is that Bayesian Ridge uses a single fixed model structure
    (all 18 predictors) with time-invariant priors, while DMA/DMS averages
    over all 2^18 model subsets and allows coefficients to drift over time.

    The coefficient path is stored for comparison with DMA's beta_dma_path.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    coef_path = np.zeros((T_oos, X.shape[1] + 1))
    t0    = time.time()

    for i in range(T_oos):
        t  = n_insample + i
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t])
        Xte = sc.transform(X[t:t+1])

        m = BayesianRidge(max_iter=300).fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])
        coef_path[i] = np.concatenate([[m.intercept_], m.coef_])

    if verbose:
        print(f"  Bayes Ridge:  R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('BayesianRidge', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, coef_path=coef_path)


def run_random_forest(y: np.ndarray, X: np.ndarray, n_insample: int,
                      dates: Optional[pd.DatetimeIndex] = None,
                      verbose: bool = True) -> MLResult:
    """
    Random Forest regressor.

    Bagging ensemble of decision trees. Each tree is trained on a bootstrap
    sample of the data and a random subset of predictors at each split.
    Hyperparameters are kept conservative (max_depth=4, min_samples_leaf=5)
    to prevent overfitting on the small monthly sample.

    Feature importances (mean decrease in impurity) are stored to compare
    with DMA's posterior inclusion probabilities.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    k     = X.shape[1]
    imp   = np.zeros(k)
    t0    = time.time()

    rf_params = {
        'n_estimators': RF_N_ESTIMATORS,
        'max_depth':    RF_MAX_DEPTH,
        'min_samples_leaf': RF_MIN_SAMPLES,
        'random_state': 42,
        'n_jobs':       -1,
    }

    for i in range(T_oos):
        t  = n_insample + i
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t])
        Xte = sc.transform(X[t:t+1])

        m = RandomForestRegressor(**rf_params).fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])
        imp += m.feature_importances_

    imp /= T_oos  # time-averaged importances

    if verbose:
        print(f"  RandomForest: R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('RandomForest', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, feature_imp=imp)


def run_xgboost(y: np.ndarray, X: np.ndarray, n_insample: int,
                dates: Optional[pd.DatetimeIndex] = None,
                verbose: bool = True) -> MLResult:
    """
    XGBoost gradient boosting regressor.

    Sequentially builds trees where each tree corrects the residuals of the
    ensemble so far. Regularisation via max_depth, subsample, and colsample
    prevents overfitting. Conservative defaults are used given the small
    monthly sample. Typically the strongest ML performer on structured
    tabular data in this sample size range.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    k     = X.shape[1]
    imp   = np.zeros(k)
    t0    = time.time()

    xgb_params = {
        'n_estimators':   XGB_N_ESTIMATORS,
        'max_depth':      XGB_MAX_DEPTH,
        'learning_rate':  XGB_LR,
        'subsample':      XGB_SUBSAMPLE,
        'colsample_bytree': XGB_COLSAMPLE,
        'random_state':   42,
        'verbosity':      0,
    }

    for i in range(T_oos):
        t  = n_insample + i
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:t])
        Xte = sc.transform(X[t:t+1])

        m = xgb.XGBRegressor(**xgb_params).fit(Xtr, y[:t])
        fc[i] = float(m.predict(Xte)[0])
        imp += m.feature_importances_

    imp /= T_oos

    if verbose:
        print(f"  XGBoost:      R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    return _package_result('XGBoost', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None,
                           t0, feature_imp=imp)


def run_mlp(y: np.ndarray, X: np.ndarray, n_insample: int,
            dates: Optional[pd.DatetimeIndex] = None,
            verbose: bool = True) -> MLResult:
    """
    Feed-forward neural network (multilayer perceptron).

    Single hidden layer with ReLU activation, fitted by L-BFGS with an L2
    weight penalty alpha. Two devices make the estimator usable on a sample
    of a few hundred months:
      * Seed ensembling: at every step the forecast is the average of
        MLP_N_SEEDS networks trained from different random initialisations
        (Gu, Kelly & Xiu 2020), which removes initialisation noise.
      * Annual retuning: width and alpha are re-selected every TUNE_EVERY
        steps by 3-fold expanding-window TimeSeriesSplit CV on the training
        window only -- the protocol used for Ridge / LASSO / Elastic Net.
    The alpha grid deliberately extends to very strong shrinkage (1000):
    preliminary runs with alpha <= 10 overfit catastrophically (R2 < -25%),
    and CV selected the grid boundary at every retuning point.

    Preferred over a recurrent network as the neural benchmark because it
    consumes the identical one-month-lagged predictor vector as every
    other model in the comparison; see Section 4.2.4 of the thesis.
    """
    from sklearn.neural_network import MLPRegressor
    import itertools, warnings as _w

    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    t0    = time.time()
    cur   = (MLP_HIDDEN_GRID[0], MLP_ALPHA_GRID[0])
    chosen = []

    def _ensemble_predict(Xtr, ytr, Xte, hidden, alpha):
        preds = []
        for s in range(MLP_N_SEEDS):
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                m = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                                 activation='relu', solver='lbfgs',
                                 max_iter=MLP_MAX_ITER, random_state=s
                                 ).fit(Xtr, ytr)
            preds.append(m.predict(Xte))
        return np.mean(preds, axis=0)

    for i in range(T_oos):
        t   = n_insample + i
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[:t]); Xte = sc.transform(X[t:t+1]); ytr = y[:t]

        if i % TUNE_EVERY == 0:
            best, best_mse = cur, np.inf
            for hidden, alpha in itertools.product(MLP_HIDDEN_GRID, MLP_ALPHA_GRID):
                mse = 0.0
                for tr, va in TimeSeriesSplit(n_splits=3).split(Xtr):
                    p = _ensemble_predict(Xtr[tr], ytr[tr], Xtr[va], hidden, alpha)
                    mse += np.mean((ytr[va] - p)**2)
                if mse < best_mse:
                    best, best_mse = (hidden, alpha), mse
            cur = best
            chosen.append({'oos_step': i, 'hidden': str(cur[0]), 'alpha': cur[1]})

        fc[i] = float(_ensemble_predict(Xtr, ytr, Xte, *cur)[0])

    if verbose:
        print(f"  MLP:          R2={_r2_oos(y[n_insample:]-fc, y[n_insample:]):.4f}"
              f"  t={time.time()-t0:.1f}s")

    res = _package_result('MLP', fc, y[n_insample:],
                          dates[n_insample:] if dates is not None else None, t0)
    res.chosen_params = pd.DataFrame(chosen)
    return res


def run_historical_average(y: np.ndarray, X: np.ndarray, n_insample: int,
                            dates: Optional[pd.DatetimeIndex] = None,
                            verbose: bool = True) -> MLResult:
    """
    Historical average (HA) benchmark — expanding window mean of y.

    Equivalent to OLS regression on an intercept only. Corresponds to the
    'HA (expanding window)' row in Table 2 of Buncic & Moretto (2015).
    Included here for completeness and to anchor the ML comparison table.
    """
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    t0    = time.time()

    for i in range(T_oos):
        t     = n_insample + i
        fc[i] = float(y[:t].mean()) if t > 0 else 0.0

    if verbose:
        y_oos = y[n_insample:]
        rw_fc = np.zeros(T_oos)
        r2    = _r2_oos(y_oos - fc, y_oos - rw_fc)
        print(f"  HA (exp):     R2={r2:.4f}  t={time.time()-t0:.1f}s")

    return _package_result('HA_expanding', fc, y[n_insample:],
                           dates[n_insample:] if dates is not None else None, t0)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_ml_models(y: np.ndarray,
                      X: np.ndarray,
                      n_insample: int,
                      dates: Optional[pd.DatetimeIndex] = None,
                      include_mlp: bool = True,
                      verbose: bool = True) -> dict:
    """
    Run all ML models sequentially and return results as a dict.

    Parameters
    ----------
    y           : np.ndarray (T,)    target variable (copper returns)
    X           : np.ndarray (T, k)  predictor matrix (k=18, no intercept)
    n_insample  : int                burn-in / in-sample observations
    dates       : pd.DatetimeIndex   aligned date index for y
    verbose     : bool               print progress

    Returns
    -------
    dict  mapping model name (str) -> MLResult
    """
    if verbose:
        T_oos = len(y) - n_insample
        print(f"\nML models: T={len(y)}, k={X.shape[1]}, "
              f"T_is={n_insample}, T_oos={T_oos}")
        if dates is not None:
            print(f"OOS: {dates[n_insample].strftime('%b %Y')} "
                  f"to {dates[-1].strftime('%b %Y')}")
        print()

    runners = [
        ('HA_expanding',  run_historical_average),
        ('OLS',           run_ols),
        ('Ridge',         run_ridge),
        ('LASSO',         run_lasso),
        ('ElasticNet',    run_elasticnet),
        ('BayesianRidge', run_bayesian_ridge),
        ('RandomForest',  run_random_forest),
        ('XGBoost',       run_xgboost),
    ]
    if include_mlp:
        runners.append(('MLP', run_mlp))

    results = {}
    for name, fn in runners:
        results[name] = fn(y=y, X=X, n_insample=n_insample,
                           dates=dates, verbose=verbose)

    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_ml_table(ml_results: dict,
                   dma_result=None,
                   rw_msfe: Optional[float] = None) -> pd.DataFrame:
    """
    Print consolidated forecast evaluation table for all ML models,
    optionally appending DMA/DMS results for side-by-side comparison.

    Parameters
    ----------
    ml_results : dict          output of run_all_ml_models
    dma_result : DMAResult     optional DMA result from 02_dma_dms.py
    rw_msfe    : float         if provided, override RW MSFE (e.g. from DMA run)

    Returns
    -------
    pd.DataFrame  formatted results table
    """
    rows = []

    # RW baseline
    if rw_msfe is None and ml_results:
        first = next(iter(ml_results.values()))
        rw_fc   = np.zeros(len(first.actual))
        rw_msfe = float(np.mean((first.actual - rw_fc)**2))

    rows.append({
        'Model':         'Random Walk (RW)',
        'MSFE':          round(rw_msfe, 4),
        'Rel. MSFE':     1.0000,
        'R2_oos (%)':    '—',
        'CW-stat':       '—',
        'p-value':       '—',
    })

    # ML models (sorted by R2)
    ordered = sorted(ml_results.items(),
                     key=lambda x: (x[1].r2_oos if not np.isnan(x[1].r2_oos) else -99),
                     reverse=True)

    for name, res in ordered:
        if np.isnan(res.msfe):
            continue
        rows.append({
            'Model':      name,
            'MSFE':       round(res.msfe, 4),
            'Rel. MSFE':  round(res.msfe / rw_msfe, 4),
            'R2_oos (%)': round(res.r2_oos * 100, 4),
            'CW-stat':    round(res.cw_stat, 4),
            'p-value':    round(res.cw_pval, 4),
        })

    # Append DMA/DMS if provided
    if dma_result is not None:
        for mname in ['TVP', 'DMS', 'DMA']:
            rows.append({
                'Model':      mname,
                'MSFE':       round(dma_result.msfe[mname], 4),
                'Rel. MSFE':  round(dma_result.msfe[mname] / rw_msfe, 4),
                'R2_oos (%)': round(dma_result.r2_oos[mname] * 100, 4),
                'CW-stat':    round(dma_result.cw_stat[mname], 4),
                'p-value':    round(dma_result.cw_pval[mname], 4),
            })

    df = pd.DataFrame(rows).set_index('Model')
    print("\nForecast Evaluation Table — All Models")
    print(df.to_string())
    return df


def print_feature_importance(ml_results: dict,
                              predictor_names: list,
                              models: list = None) -> pd.DataFrame:
    """
    Print feature importance rankings for tree-based models.

    Parameters
    ----------
    ml_results       : dict output of run_all_ml_models
    predictor_names  : list of k predictor names
    models           : list of model names to include (default: RF + XGBoost)
    """
    if models is None:
        models = ['RandomForest', 'XGBoost']

    rows = {'Predictor': predictor_names}
    for name in models:
        if name in ml_results and ml_results[name].feature_imp is not None:
            rows[name] = np.round(ml_results[name].feature_imp, 4)

    df = pd.DataFrame(rows).set_index('Predictor')
    if len(df.columns) > 0:
        df['Mean'] = df.mean(axis=1)
        df = df.sort_values('Mean', ascending=False)
        print(f"\nFeature Importances (time-averaged):")
        print(df.to_string())
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
    print("03_ml_models.py  —  ML model estimation")
    print("=" * 60)

    df, meta = build_dataset(data_dir=data_dir)
    y     = df['r_copper'].values
    X     = df[PREDICTOR_COLS].values
    dates = pd.DatetimeIndex(df['date'])

    n_insample = 120

    results = run_all_ml_models(
        y=y, X=X, n_insample=n_insample,
        dates=dates, verbose=True
    )

    # Print consolidated table
    print_ml_table(results)

    # Feature importance
    pred_names = [c.replace('x_', '') for c in PREDICTOR_COLS]
    print_feature_importance(results, predictor_names=pred_names)