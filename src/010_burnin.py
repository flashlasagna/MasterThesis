"""
10_burnin.py
============
Sensitivity to the initial (burn-in) window length, i.e. to the start of
the out-of-sample period (Table T16).

The baseline uses a 120-month burn-in, so the evaluation window opens in
March 2008, six months before the Lehman collapse. Because predictability
is concentrated in crisis episodes, the headline R^2 could depend on this
choice in two distinct ways: (i) through the COMPOSITION of the evaluation
window (whether it contains the GFC), and (ii) through the LENGTH of the
training history available when forecasting begins. This module separates
the two.

Panel A  -- "full OOS": for each burn-in n_is in BURNINS the models are
            re-estimated and evaluated over their own full out-of-sample
            window, from month n_is+1 to February 2026.
Panel B  -- "common window": the same forecasts are evaluated only over
            the window common to all burn-ins (COMMON_START onwards), so
            that differences reflect training length alone.

DMA, Ridge, Elastic Net and the equal-weight DMA+EN combination are run
under the unchanged walk-forward protocol of Section 4.4.

Run from the project root:
    python3 src/10_burnin.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline  import build_dataset, PREDICTOR_COLS, TARGET_COL
from src.dma_dms   import run_dma_dms
from src.ml_models import run_ridge, run_elasticnet

BURNINS      = [60, 84, 96, 108, 120, 132, 144, 168, 180]   # months
COMMON_START = '2013-03-01'   # first OOS month under the longest burn-in


def _r2(fc, y):
    return 100.0 * (1.0 - np.sum((y - fc)**2) / np.sum(y**2))


def _cw(fc, y, bw=5):
    """Clark-West MSFE-adjusted t-stat vs the zero (RW) forecast, Bartlett HAC."""
    cw = y**2 - (y - fc)**2 + fc**2
    T  = len(cw); m = cw.mean(); d = cw - m
    var = np.sum(d**2) / T
    for l in range(1, bw + 1):
        w = 1 - l / (bw + 1)
        var += 2 * w * np.sum(d[l:] * d[:-l]) / T
    return m / np.sqrt(var / T)


def run_burnin_robustness(df, output_dir='output/', verbose=True):
    y     = df[TARGET_COL].to_numpy(float)
    X     = df[PREDICTOR_COLS].to_numpy(float)
    dates = pd.DatetimeIndex(df['date'])
    out   = Path(output_dir); out.mkdir(exist_ok=True)

    rows_full, rows_common, all_fc = [], [], {}
    for n_is in BURNINS:
        t0  = time.time()
        dma = run_dma_dms(y, X, n_is, dates=dates, verbose=False)
        rdg = run_ridge(y, X, n_is, dates=dates, verbose=False)
        en  = run_elasticnet(y, X, n_is, dates=dates, verbose=False)
        d_oos = dates[n_is:]; y_oos = y[n_is:]
        fcs = {'DMA': dma.dma_forecasts, 'Ridge': rdg.forecasts,
               'ElasticNet': en.forecasts,
               'Combo': 0.5 * (dma.dma_forecasts + en.forecasts)}
        all_fc[n_is] = pd.DataFrame(fcs, index=d_oos).assign(actual=y_oos)

        base = {'Burn-in (months)': n_is,
                'OOS start': d_oos[0].strftime('%b %Y'),
                'N_oos': len(y_oos)}
        rows_full.append({**base, **{f'{k} R2': round(_r2(v, y_oos), 2) for k, v in fcs.items()},
                          'DMA CW': round(_cw(fcs['DMA'], y_oos), 2),
                          'Ridge CW': round(_cw(fcs['Ridge'], y_oos), 2)})
        m = d_oos >= COMMON_START
        rows_common.append({**base, 'N_common': int(m.sum()),
                            **{f'{k} R2': round(_r2(v[m], y_oos[m]), 2) for k, v in fcs.items()}})
        if verbose:
            print(f"n_is={n_is:3d} ({base['OOS start']}): "
                  + "  ".join(f"{k} {rows_full[-1][f'{k} R2']:6.2f}" for k in fcs)
                  + f"   | common: DMA {rows_common[-1]['DMA R2']:6.2f} "
                    f"Ridge {rows_common[-1]['Ridge R2']:6.2f}   ({time.time()-t0:.0f}s)",
                  flush=True)
        pd.DataFrame(rows_full).to_csv(out / 'T16_burnin_fullOOS.csv', index=False)
        pd.DataFrame(rows_common).to_csv(out / 'T16_burnin_commonwindow.csv', index=False)

    pd.concat(all_fc, names=['burnin', 'date']).to_csv(out / 'T16_burnin_forecasts.csv')
    return pd.DataFrame(rows_full), pd.DataFrame(rows_common)


if __name__ == '__main__':
    df, _ = build_dataset(str(PROJECT_ROOT / 'data'))
    run_burnin_robustness(df, output_dir=str(PROJECT_ROOT / 'output'))


# ---------------------------------------------------------------------------
# Panel C -- training LENGTH at a fixed evaluation window (Mar 2008 - Feb 2026)
# ---------------------------------------------------------------------------
# Because every model uses an expanding window, changing the burn-in does
# not change the training history available at a given forecast origin:
# the common-window results of Panel B are identical by construction. To
# vary training length proper, we (i) truncate the START of the sample so
# that fewer pre-2008 months are available, keeping the OOS window fixed,
# and (ii) estimate Ridge and Elastic Net on FIXED-LENGTH rolling windows.

SAMPLE_STARTS = ['1998-03-01', '2000-03-01', '2002-03-01', '2004-03-01']
ROLLING       = [60, 96, 120]
OOS_START     = '2008-03-01'


def _rolling_ridge_en(y, X, n_is, window, which='ridge'):
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    T = len(y); fc = np.zeros(T - n_is); alpha, l1 = 1.0, 0.5
    for i in range(T - n_is):
        t = n_is + i; lo = max(0, t - window)
        sc = StandardScaler(); Xtr = sc.fit_transform(X[lo:t]); Xte = sc.transform(X[t:t+1]); ytr = y[lo:t]
        if i % 12 == 0:
            tscv = TimeSeriesSplit(n_splits=3)
            if which == 'ridge':
                alpha = float(RidgeCV(alphas=np.logspace(-3, 4, 40), cv=tscv).fit(Xtr, ytr).alpha_)
            else:
                m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=tscv, max_iter=20000).fit(Xtr, ytr)
                alpha, l1 = float(m.alpha_), float(m.l1_ratio_)
        m = Ridge(alpha=alpha) if which == 'ridge' else ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000)
        fc[i] = float(m.fit(Xtr, ytr).predict(Xte)[0])
    return fc


def run_training_length(df, output_dir='output/', verbose=True):
    dates_all = pd.DatetimeIndex(df['date'])
    out = Path(output_dir); rows = []
    for start in SAMPLE_STARTS:
        sub   = df[dates_all >= start].reset_index(drop=True)
        y     = sub[TARGET_COL].to_numpy(float); X = sub[PREDICTOR_COLS].to_numpy(float)
        dates = pd.DatetimeIndex(sub['date']); n_is = int((dates < OOS_START).sum())
        t0 = time.time()
        dma = run_dma_dms(y, X, n_is, dates=dates, verbose=False)
        rdg = run_ridge(y, X, n_is, dates=dates, verbose=False)
        en  = run_elasticnet(y, X, n_is, dates=dates, verbose=False)
        y_oos = y[n_is:]
        rows.append({'Scheme': 'expanding', 'Sample start': pd.Timestamp(start).strftime('%b %Y'),
                     'Training months at first origin': n_is,
                     'DMA R2': round(_r2(dma.dma_forecasts, y_oos), 2),
                     'Ridge R2': round(_r2(rdg.forecasts, y_oos), 2),
                     'ElasticNet R2': round(_r2(en.forecasts, y_oos), 2),
                     'Combo R2': round(_r2(0.5*(dma.dma_forecasts+en.forecasts), y_oos), 2)})
        if verbose: print(rows[-1], f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / 'T16_training_length.csv', index=False)

    y = df[TARGET_COL].to_numpy(float); X = df[PREDICTOR_COLS].to_numpy(float); n_is = 120; y_oos = y[n_is:]
    for w in ROLLING:
        t0 = time.time()
        r = _rolling_ridge_en(y, X, n_is, w, 'ridge'); e = _rolling_ridge_en(y, X, n_is, w, 'en')
        rows.append({'Scheme': f'rolling {w}m', 'Sample start': 'Mar 1998',
                     'Training months at first origin': w, 'DMA R2': np.nan,
                     'Ridge R2': round(_r2(r, y_oos), 2), 'ElasticNet R2': round(_r2(e, y_oos), 2),
                     'Combo R2': np.nan})
        if verbose: print(rows[-1], f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / 'T16_training_length.csv', index=False)
    return pd.DataFrame(rows)