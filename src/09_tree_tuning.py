"""
09_tree_tuning.py
=================
Hyperparameter validation for the tree ensembles (Table T14).

The baseline Random Forest and XGBoost specifications in 03_ml_models.py
are FIXED EX ANTE (RF: 200 trees, max_depth=4, min_samples_leaf=5;
XGB: 100 rounds, max_depth=3, learning_rate=0.05, row/col subsample 0.8).
They were not selected by cross-validation, nor chosen after inspecting
out-of-sample results. This module establishes whether the tree models'
underperformance relative to the penalised regressions is an artefact of
those fixed settings, by

  (A) TUNED variants: hyperparameters re-selected every TUNE_EVERY=12
      steps by 3-fold expanding-window TimeSeriesSplit cross-validation
      on the training window only -- the identical protocol applied to
      Ridge / LASSO / Elastic Net in Section 4.2.1 -- over the grids below;

  (B) SENSITIVITY grids: full walk-forward out-of-sample R^2 for a grid of
      fixed settings spanning the capacity-controlling parameters the
      reviewer identifies (depth and leaf size for RF; rounds, depth and
      learning rate for XGB).

All forecasts use the expanding window, per-step standardisation, and
the one-month predictor lag of the main analysis.

Run from the project root:
    python3 src/09_tree_tuning.py
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline import build_dataset, PREDICTOR_COLS, TARGET_COL

TUNE_EVERY = 12
N_SPLITS   = 3
SEED       = 42

# --- Grids for the tuned variants --------------------------------------
RF_GRID = {
    'max_depth':        [2, 4, 6, None],
    'min_samples_leaf': [2, 5, 20],
}
XGB_GRID = {
    'n_estimators':  [50, 100, 200, 400],
    'max_depth':     [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.10],
}

# --- Grids for the fixed-setting sensitivity surfaces ------------------
RF_SENS  = [(d, l) for d in [2, 4, 6, None] for l in [2, 5, 20]]
XGB_SENS = ([(n, d, 0.05) for n in [25, 50, 100, 200, 400, 800] for d in [2, 3, 4]]
            + [(n, 3, lr) for n in [25, 50, 100, 200, 400, 800] for lr in [0.01, 0.10]])


def _r2(fc, y):  return 100.0 * (1.0 - np.sum((y - fc)**2) / np.sum(y**2))

def _rf(params):
    return RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, **params)

def _xgb(params):
    return xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, verbosity=0, n_jobs=1, **params)


def walk_forward(y, X, n_insample, make_model, params, tune_grid=None, verbose=False):
    """Expanding-window OOS forecasts; optional CV retuning every TUNE_EVERY steps."""
    T_oos = len(y) - n_insample
    fc    = np.zeros(T_oos)
    chosen = []
    cur = dict(params)
    for i in range(T_oos):
        t   = n_insample + i
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[:t]); Xte = sc.transform(X[t:t+1]); ytr = y[:t]
        if tune_grid is not None and i % TUNE_EVERY == 0:
            keys = list(tune_grid)
            best, best_mse = None, np.inf
            for vals in itertools.product(*[tune_grid[k] for k in keys]):
                cand = dict(zip(keys, vals))
                mse = 0.0
                for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(Xtr):
                    m = make_model(cand).fit(Xtr[tr], ytr[tr])
                    mse += np.mean((ytr[va] - m.predict(Xtr[va]))**2)
                if mse < best_mse:
                    best, best_mse = cand, mse
            cur = best
            chosen.append({'oos_step': i, **cur})
        m = make_model(cur).fit(Xtr, ytr)
        fc[i] = float(m.predict(Xte)[0])
    return fc, pd.DataFrame(chosen)


def run_tree_tuning(df, n_insample=120, output_dir='output/', verbose=True):
    y = df[TARGET_COL].to_numpy(float); X = df[PREDICTOR_COLS].to_numpy(float)
    y_oos = y[n_insample:]
    out = Path(output_dir); out.mkdir(exist_ok=True)
    rows = []

    # ---- (A) baseline (fixed ex ante) and tuned variants -----------------
    t0 = time.time()
    fc, _ = walk_forward(y, X, n_insample, _rf, {'max_depth': 4, 'min_samples_leaf': 5})
    rows.append(('RandomForest', 'fixed ex ante (baseline)', 'depth=4, leaf=5, 200 trees', _r2(fc, y_oos)))
    fc, ch_rf = walk_forward(y, X, n_insample, _rf, {}, tune_grid=RF_GRID)
    rows.append(('RandomForest', 'TimeSeriesSplit-tuned', 'depth∈{2,4,6,∞}, leaf∈{2,5,20}', _r2(fc, y_oos)))
    ch_rf.to_csv(out / 'T14_rf_chosen_params.csv', index=False)
    if verbose: print(f"RF done {time.time()-t0:.0f}s"); print(ch_rf.to_string(index=False))

    t0 = time.time()
    fc, _ = walk_forward(y, X, n_insample, _xgb, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05})
    rows.append(('XGBoost', 'fixed ex ante (baseline)', 'rounds=100, depth=3, lr=0.05', _r2(fc, y_oos)))
    fc, ch_xgb = walk_forward(y, X, n_insample, _xgb, {}, tune_grid=XGB_GRID)
    rows.append(('XGBoost', 'TimeSeriesSplit-tuned', 'rounds∈{50..400}, depth∈{2,3,4}, lr∈{.01,.05,.1}', _r2(fc, y_oos)))
    ch_xgb.to_csv(out / 'T14_xgb_chosen_params.csv', index=False)
    if verbose: print(f"XGB done {time.time()-t0:.0f}s"); print(ch_xgb.to_string(index=False))

    tabA = pd.DataFrame(rows, columns=['Model', 'Hyperparameter choice', 'Setting', 'R2_oos (%)'])
    tabA['R2_oos (%)'] = tabA['R2_oos (%)'].round(2)
    tabA.to_csv(out / 'T14_tree_tuned.csv', index=False)
    if verbose: print(tabA.to_string(index=False))

    # ---- (B) sensitivity surfaces ----------------------------------------
    sens = []
    t0 = time.time()
    for d, l in RF_SENS:
        fc, _ = walk_forward(y, X, n_insample, _rf, {'max_depth': d, 'min_samples_leaf': l})
        sens.append({'Model': 'RandomForest', 'max_depth': 'None' if d is None else d,
                     'min_samples_leaf': l, 'n_estimators': 200, 'learning_rate': '',
                     'R2_oos (%)': round(_r2(fc, y_oos), 2)})
        if verbose: print(f"  RF depth={d} leaf={l}: {sens[-1]['R2_oos (%)']:.2f}  ({time.time()-t0:.0f}s)")
    for n, d, lr in XGB_SENS:
        fc, _ = walk_forward(y, X, n_insample, _xgb, {'n_estimators': n, 'max_depth': d, 'learning_rate': lr})
        sens.append({'Model': 'XGBoost', 'max_depth': d, 'min_samples_leaf': '',
                     'n_estimators': n, 'learning_rate': lr, 'R2_oos (%)': round(_r2(fc, y_oos), 2)})
        if verbose: print(f"  XGB rounds={n} depth={d} lr={lr}: {sens[-1]['R2_oos (%)']:.2f}")
    tabB = pd.DataFrame(sens)
    tabB.to_csv(out / 'T14_tree_sensitivity.csv', index=False)
    return tabA, tabB


if __name__ == '__main__':
    df, _ = build_dataset(str(PROJECT_ROOT / 'data'))
    run_tree_tuning(df, output_dir=str(PROJECT_ROOT / 'output'))