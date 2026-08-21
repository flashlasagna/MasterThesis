"""
07_realtime.py
==============
Real-time availability robustness (Table T11).

The baseline panel applies a uniform one-month lag so that X_t holds values
dated end of month t-1. For market-determined series (prices, rates, FX,
equities, VIX, BDI, CME warehouse stocks) the value dated t-1 is observable
at the close of t-1 and is never revised. Two predictors are statistical
releases whose month-(t-1) value is NOT in the public domain at the close of
month t-1:

  x_dIP      US Industrial Production (Fed G.17): the month-(t-1) figure is
             released around the 15th-17th of month t, and is revised in the
             following three monthly releases and at annual benchmarks.
  x_dDemand  WBMS world refined consumption and production: the month-(t-1)
             figure is published roughly two months after the reference
             month and is revised in subsequent bulletins.

Two alternative information sets are therefore evaluated:

  'publag'   publication-lag-adjusted panel — x_dIP receives one extra
             month of lag (2 in total) and x_dDemand two extra months
             (3 in total), so that every X_t column is a value that had
             been released by the close of month t-1.
  'market'   market-data-only panel — x_dIP and x_dDemand are dropped
             (k = 16), removing the two series subject to ex-post revision.

Neither variant addresses data revisions (we hold only final vintages); the
'market' variant bounds their possible effect by removing the revised series
altogether. DMA, Ridge and Elastic Net are re-estimated under the identical
walk-forward protocol and compared with the baseline.

Run from the project root:
    python3 src/07_realtime.py
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

# Extra months of lag beyond the baseline one-month shift, by predictor.
EXTRA_PUBLICATION_LAG = {
    'x_dIP':     1,   # G.17 released mid-month t  -> available for origin t
    'x_dDemand': 2,   # WBMS released ~2 months after reference month
}
REVISED_SERIES = list(EXTRA_PUBLICATION_LAG.keys())


def make_variant(df: pd.DataFrame, variant: str):
    """Return (y, X, dates, cols) for a given information-set variant."""
    d = df.copy()
    cols = list(PREDICTOR_COLS)

    if variant == 'baseline':
        pass
    elif variant == 'publag':
        for col, extra in EXTRA_PUBLICATION_LAG.items():
            d[col] = d[col].shift(extra)
    elif variant == 'market':
        cols = [c for c in cols if c not in REVISED_SERIES]
    else:
        raise ValueError(variant)

    # Keep the sample aligned across variants: the extra shifts create NaNs
    # in the first two rows, which fall inside the 120-month burn-in. We
    # back-fill those two rows with the first observed value so that the
    # OOS window (rows 120..335) is identical in every variant.
    d[cols] = d[cols].bfill()
    y     = d[TARGET_COL].to_numpy(dtype=float)
    X     = d[cols].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(d['date'])
    return y, X, dates, cols


def run_realtime_robustness(df: pd.DataFrame,
                            n_insample: int = 120,
                            output_dir: str = 'output/',
                            verbose: bool = True) -> pd.DataFrame:
    rows = []
    forecasts = {}
    for variant in ['baseline', 'publag', 'market']:
        y, X, dates, cols = make_variant(df, variant)
        if verbose:
            print(f"\n=== Variant: {variant}  (k={X.shape[1]}) ===")
        t0 = time.time()
        dma = run_dma_dms(y, X, n_insample, dates=dates, verbose=False)
        rdg = run_ridge(y, X, n_insample, dates=dates, verbose=False)
        en  = run_elasticnet(y, X, n_insample, dates=dates, verbose=False)

        forecasts[variant] = {'DMA': dma.dma_forecasts,
                              'Ridge': rdg.forecasts,
                              'ElasticNet': en.forecasts}
        for name, r2, cw, p, msfe in [
            ('DMA',        dma.r2_oos['DMA'], dma.cw_stat['DMA'],
                           dma.cw_pval['DMA'], dma.msfe['DMA']),
            ('Ridge',      rdg.r2_oos, rdg.cw_stat, rdg.cw_pval, rdg.msfe),
            ('ElasticNet', en.r2_oos,  en.cw_stat,  en.cw_pval,  en.msfe),
        ]:
            rows.append({'Variant': variant, 'k': X.shape[1], 'Model': name,
                         'MSFE': round(msfe, 3), 'R2_oos (%)': round(100*r2, 2),
                         'CW-stat': round(cw, 3), 'p-value': round(p, 4)})
        if verbose:
            print(f"   done in {time.time()-t0:.0f}s")

    tab = pd.DataFrame(rows)
    # Change in R2 relative to baseline, per model
    base = tab[tab.Variant == 'baseline'].set_index('Model')['R2_oos (%)']
    tab['dR2 vs baseline (pp)'] = tab.apply(
        lambda r: round(r['R2_oos (%)'] - base[r['Model']], 2), axis=1)

    out = Path(output_dir); out.mkdir(exist_ok=True)
    tab.to_csv(out / 'T11_realtime_robustness.csv', index=False)
    pd.DataFrame({f"{v}_{m}": fc for v, d in forecasts.items()
                  for m, fc in d.items()},
                 index=dates[n_insample:]).to_csv(
        out / 'T11_realtime_forecasts.csv')
    if verbose:
        print("\n", tab.to_string(index=False))
    return tab


if __name__ == '__main__':
    df, _ = build_dataset(str(PROJECT_ROOT / 'data'))
    run_realtime_robustness(df, output_dir=str(PROJECT_ROOT / 'output'))