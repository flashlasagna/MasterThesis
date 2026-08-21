"""
08_omega.py
===========
Real-time estimation of the forward-price alignment factor omega (Table T12).

The convenience yield (Eq. 2 of the thesis) is

    CY_t = i_t - (omega * F_t - S_t) / S_t ,

where omega rescales the 3-month forward series to the spot series (the two
are quoted in different units by the vendor feeds). The baseline fixes
omega = 1.6428, the mean spot/forward ratio over the full overlapping
sample, which uses data beyond each forecast origin. This module evaluates
two real-time alternatives under the identical walk-forward protocol:

  'full'       baseline: omega fixed at 1.6428 (full-sample mean ratio).
  'insample'   omega fixed at the mean ratio over the 120-month burn-in
               window only (Mar 1998 - Feb 2008); never updated afterwards.
  'recursive'  omega_t = mean ratio over all months up to and including
               the month in which CY is measured. Because every predictor is
               then lagged one month, the omega used at forecast origin t-1
               depends only on spot/forward data through t-1.

For the penalised regressions a FIXED omega is an affine rescaling of CY
and is absorbed exactly by per-step standardisation (Section 3.3.1), so
'insample' must reproduce the baseline ML forecasts to machine precision;
this is verified numerically. A recursive omega_t is time-varying and is
not a pure affine map, so the ML models are re-estimated under it as well.

Run from the project root:
    python3 src/08_omega.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib
_pipe = importlib.import_module('src.pipeline')
from src.pipeline  import PREDICTOR_COLS, TARGET_COL
from src.dma_dms   import run_dma_dms
from src.ml_models import run_ridge, run_elasticnet

BASELINE_OMEGA = 1.6428


def _raw_panel(data_dir: str) -> pd.DataFrame:
    raw      = _pipe._harmonise_units(_pipe._load_all(data_dir))
    ibor, _  = _pipe._build_ibor(raw)
    return _pipe._aggregate_monthly(raw, ibor)


def omega_series(panel: pd.DataFrame, variant: str,
                 n_insample: int = 120) -> pd.Series:
    """Return omega aligned to panel rows (constant or expanding mean)."""
    ratio = panel['spot'] / panel['fwd3m']
    if variant == 'full':
        return pd.Series(BASELINE_OMEGA, index=panel.index)
    # Rows of the raw panel whose date falls in the 120-month burn-in
    # (the final dataset starts at Mar 1998 after the diff/lag drop two rows)
    first_oos_date = panel['date'].iloc[0] + pd.DateOffset(months=2 + n_insample)
    burn = ratio[panel['date'] < first_oos_date]
    if variant == 'insample':
        return pd.Series(float(burn.mean()), index=panel.index)
    if variant == 'recursive':
        return ratio.expanding().mean()
    raise ValueError(variant)


def build_variant(panel: pd.DataFrame, variant: str, n_insample: int = 120):
    """Rebuild the feature panel with CY computed under the given omega."""
    om = omega_series(panel, variant, n_insample)
    p  = panel.copy()
    df = _pipe._build_features(p)          # baseline features (omega=1.6428)
    cy = p['tb3m'] - (om * p['fwd3m'] - p['spot']) / p['spot']
    df['x_ConvYield'] = cy.shift(1)        # same one-month lag as all predictors
    final = ['date', TARGET_COL, 'spot'] + PREDICTOR_COLS
    df = df[final].dropna().reset_index(drop=True)
    y     = df[TARGET_COL].to_numpy(float)
    X     = df[PREDICTOR_COLS].to_numpy(float)
    dates = pd.DatetimeIndex(df['date'])
    return y, X, dates, om


def run_omega_robustness(data_dir: str = 'data/', n_insample: int = 120,
                         output_dir: str = 'output/', verbose: bool = True):
    panel = _raw_panel(data_dir)
    rows, fcs, omegas = [], {}, {}
    for variant in ['full', 'insample', 'recursive']:
        y, X, dates, om = build_variant(panel, variant, n_insample)
        assert len(y) == 336, len(y)
        omegas[variant] = om
        if verbose:
            print(f"\n=== omega variant: {variant} ===")
            if variant == 'recursive':
                o = om.iloc[-len(y):]
                print(f"   omega_t range over OOS: "
                      f"{o.iloc[n_insample]:.4f} -> {o.iloc[-1]:.4f}")
            else:
                print(f"   omega = {om.iloc[0]:.4f}")
        t0  = time.time()
        dma = run_dma_dms(y, X, n_insample, dates=dates, verbose=False)
        rdg = run_ridge(y, X, n_insample, dates=dates, verbose=False)
        en  = run_elasticnet(y, X, n_insample, dates=dates, verbose=False)
        fcs[variant] = {'DMA': dma.dma_forecasts, 'DMS': dma.dms_forecasts,
                        'Ridge': rdg.forecasts, 'ElasticNet': en.forecasts}
        for name, r2, cw, pv, ms in [
            ('DMA',  dma.r2_oos['DMA'], dma.cw_stat['DMA'], dma.cw_pval['DMA'], dma.msfe['DMA']),
            ('DMS',  dma.r2_oos['DMS'], dma.cw_stat['DMS'], dma.cw_pval['DMS'], dma.msfe['DMS']),
            ('Ridge', rdg.r2_oos, rdg.cw_stat, rdg.cw_pval, rdg.msfe),
            ('ElasticNet', en.r2_oos, en.cw_stat, en.cw_pval, en.msfe)]:
            rows.append({'Omega': variant, 'omega_value':
                         'expanding' if variant == 'recursive' else round(float(om.iloc[0]), 4),
                         'Model': name, 'MSFE': round(ms, 3), 'R2_oos (%)': round(100*r2, 2),
                         'CW-stat': round(cw, 3), 'p-value': round(pv, 4)})
        if verbose:
            print(f"   done in {time.time()-t0:.0f}s")

    tab = pd.DataFrame(rows)
    base = tab[tab.Omega == 'full'].set_index('Model')['R2_oos (%)']
    tab['dR2 vs baseline (pp)'] = tab.apply(
        lambda r: round(r['R2_oos (%)'] - base[r['Model']], 2), axis=1)
    # Max abs forecast deviation from baseline, per model (invariance check)
    tab['max |fc - baseline fc|'] = tab.apply(
        lambda r: float(np.max(np.abs(fcs[r['Omega']][r['Model']] - fcs['full'][r['Model']]))),
        axis=1)

    out = Path(output_dir); out.mkdir(exist_ok=True)
    tab.to_csv(out / 'T12_omega_robustness.csv', index=False)
    if verbose:
        print("\n", tab.to_string(index=False))
    return tab, omegas


if __name__ == '__main__':
    run_omega_robustness(str(PROJECT_ROOT / 'data'),
                         output_dir=str(PROJECT_ROOT / 'output'))