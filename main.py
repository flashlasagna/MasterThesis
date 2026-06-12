"""
main.py
=======
Master orchestrator for "Forecasting Copper Prices:
A Replication and ML Extension of Buncic & Moretto (2014)".

Run from the MasterThesis/ root directory:
    python3 main.py

Optional arguments:
    python3 main.py --data data/ --output output/ --insample 120 --skip_mh

Structure
---------
    MasterThesis/
    ├── main.py
    ├── data/          ← all 22 xlsx files
    ├── src/
    │   ├── pipeline.py
    │   ├── dma_dms.py
    │   ├── ml_models.py
    │   ├── hybrid.py
    │   ├── evaluation.py
    │   └── __init__.py
    └── output/        ← all tables (.csv) and figures (.png)

Pipeline steps
--------------
    Step 1  Build dataset          src/pipeline.py
    Step 2  DMA/DMS (h=1)          src/dma_dms.py
    Step 3  DMA/DMS (h=2,3,6)      src/dma_dms.py    [skippable with --skip_mh]
    Step 4  ML models              src/ml_models.py
    Step 5  Multi-horizon ML       src/ml_models.py   [skippable with --skip_mh]
    Step 6  Hybrid models          src/hybrid.py
    Step 7  Full evaluation        src/evaluation.py

Estimated runtime (without LSTM, without multi-horizon):
    ~5-7 minutes on a modern laptop

Estimated runtime (full, including LSTM and multi-horizon):
    ~20-25 minutes
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline    import build_dataset, PREDICTOR_COLS
from src.dma_dms     import (run_dma_dms, run_dma_multihorizon,
                              print_evaluation_table, DEFAULT_LAMBDA,
                              DEFAULT_ALPHA, DEFAULT_KAPPA)
from src.ml_models   import run_all_ml_models, print_ml_table
from src.hybrid      import run_all_hybrids, print_hybrid_table
from src.evaluation  import run_full_evaluation
from src.extensions  import run_all_extensions

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Copper price forecasting — full pipeline')
    p.add_argument('--data',      default='data/',   help='Path to data directory')
    p.add_argument('--output',    default='output/', help='Path to output directory')
    p.add_argument('--insample',  default=120, type=int,
                   help='Number of in-sample (burn-in) observations (default: 120)')
    p.add_argument('--skip_mh',   action='store_true',
                   help='Skip multi-horizon forecasting (h=2,3,6) to save time')
    p.add_argument('--skip_lstm', action='store_true',
                   help='Skip LSTM model (requires PyTorch; slow on CPU)')
    p.add_argument('--skip_sensitivity', action='store_true',
                   help='Skip the T8 forgetting-factor grid (the only heavy extension)')
    p.add_argument('--horizons',  default='1,2,3,6',
                   help='Comma-separated forecast horizons (default: 1,2,3,6)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(n: int, title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  Step {n}: {title}")
    print(f"{'─'*55}")


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs/60:.1f}min"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    data_dir   = args.data
    output_dir = args.output
    n_insample = args.insample
    horizons   = [int(h) for h in args.horizons.split(',')]
    include_lstm = False

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    _header("Copper Price Forecasting — Full Pipeline")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  In-sample  : {n_insample} months")
    print(f"  Horizons   : {horizons}")
    print(f"  LSTM       : {'yes' if include_lstm else 'no (--skip_lstm)'}")
    print(f"  Multi-horiz: {'no (--skip_mh)' if args.skip_mh else 'yes'}")

    # ── Step 1: Build dataset ────────────────────────────────────────────────
    _step(1, "Building dataset")
    t0 = time.time()
    df, meta = build_dataset(data_dir=data_dir)
    y     = df['r_copper'].values
    X     = df[PREDICTOR_COLS].values
    dates = pd.DatetimeIndex(df['date'])
    T, k  = X.shape

    print(f"  Observations : {meta['n_obs']}")
    print(f"  Predictors   : {meta['n_predictors']}")
    print(f"  Sample       : {meta['sample_start']} → {meta['sample_end']}")
    print(f"  LIBOR-SOFR adj : {meta['libor_sofr_adj']*100:.2f} bps")
    print(f"  NaN count    : {df.isnull().sum().sum()}")
    print(f"  In-sample    : {dates[0].strftime('%b %Y')} → "
          f"{dates[n_insample-1].strftime('%b %Y')}  ({n_insample} obs)")
    print(f"  OOS          : {dates[n_insample].strftime('%b %Y')} → "
          f"{dates[-1].strftime('%b %Y')}  ({T-n_insample} obs)")
    print(f"  Elapsed: {_elapsed(t0)}")

    # ── Step 2: DMA/DMS (h=1) ────────────────────────────────────────────────
    _step(2, "DMA/DMS — one-step-ahead (h=1)")
    t0 = time.time()
    dma_result = run_dma_dms(
        y=y, X=X, n_insample=n_insample,
        lam=DEFAULT_LAMBDA, alpha=DEFAULT_ALPHA, kappa=DEFAULT_KAPPA,
        dates=dates, verbose=True,
    )
    print_evaluation_table(dma_result, horizon=1)
    print(f"  Elapsed: {_elapsed(t0)}")

    # ── Step 3: DMA/DMS multi-horizon ────────────────────────────────────────
    dma_mh_results = {}
    if not args.skip_mh and len(horizons) > 1:
        _step(3, f"DMA/DMS — multi-horizon {horizons[1:]}")
        t0 = time.time()
        dma_mh_results = run_dma_multihorizon(
            y=y, X=X, n_insample=n_insample,
            horizons=[h for h in horizons if h > 1],
            lam=DEFAULT_LAMBDA, alpha=DEFAULT_ALPHA, kappa=DEFAULT_KAPPA,
            dates=dates, verbose=True,
        )
        for h, res in dma_mh_results.items():
            print_evaluation_table(res, horizon=h)
        print(f"  Elapsed: {_elapsed(t0)}")
    else:
        _step(3, "DMA/DMS — multi-horizon [SKIPPED]")

    # ── Step 4: ML models (h=1) ──────────────────────────────────────────────
    _step(4, "ML models — one-step-ahead (h=1)")
    t0 = time.time()
    ml_results = run_all_ml_models(
        y=y, X=X, n_insample=n_insample,
        dates=dates, include_lstm=include_lstm, verbose=True,
    )
    print_ml_table(ml_results, dma_result=dma_result,
                   rw_msfe=dma_result.msfe['RW'])
    print(f"  Elapsed: {_elapsed(t0)}")

    # ── Step 5: Multi-horizon ML (ElasticNet only, for Table 3) ──────────────
    if not args.skip_mh and len(horizons) > 1:
        _step(5, f"ML models — multi-horizon ElasticNet {horizons[1:]}")
        t0 = time.time()
        from src.ml_models import run_elasticnet

        for h in [hh for hh in horizons if hh > 1]:
            print(f"\n  h={h}:")
            # Direct method: h-period holding return aligned with X_t,
            # which carries information from t-1 after the pipeline lag.
            y_h = np.full(T, np.nan)
            for t_idx in range(T - h + 1):
                y_h[t_idx] = 100.0 * (
                    np.prod(1.0 + y[t_idx:t_idx+h] / 100.0) - 1.0)
            valid     = ~np.isnan(y_h)
            y_h_valid = y_h[valid]
            X_valid   = X[valid]
            dates_h   = dates[valid]
            if len(y_h_valid) > n_insample:
                res_h = run_elasticnet(
                    y=y_h_valid, X=X_valid,
                    n_insample=n_insample,
                    dates=dates_h, verbose=True, gap=h-1)
                ml_results[f'ElasticNet_h{h}'] = res_h

        print(f"  Elapsed: {_elapsed(t0)}")
    else:
        _step(5, "ML models — multi-horizon [SKIPPED]")

    # ── Step 6: Hybrid models ────────────────────────────────────────────────
    _step(6, "Hybrid DMA-ML models")
    t0 = time.time()
    hybrid_results = run_all_hybrids(
        y=y, X=X,
        dma_result=dma_result,
        ml_results=ml_results,
        n_insample=n_insample,
        dates=dates,
        verbose=True,
    )
    print_hybrid_table(hybrid_results,
                        dma_result=dma_result,
                        ml_results=ml_results,
                        rw_msfe=dma_result.msfe['RW'])
    print(f"  Elapsed: {_elapsed(t0)}")

    # ── Step 7: Full evaluation — all tables and figures ─────────────────────
    _step(7, "Generating all tables and figures")
    t0 = time.time()
    run_full_evaluation(
        df=df,
        predictor_cols=PREDICTOR_COLS,
        dma_result=dma_result,
        ml_results=ml_results,
        hybrid_results=hybrid_results,
        n_insample=n_insample,
        output_dir=output_dir,
        dma_mh_results=dma_mh_results,
    )
    print(f"  Elapsed: {_elapsed(t0)}")

    # -- Step 8: Tier-1 extensions (T7-T10) -----------------------------------
    _step(8, "Tier-1 extensions: DM-HLN, sensitivity, directional, correlation")
    t0 = time.time()
    run_all_extensions(
        df=df,
        predictor_cols=PREDICTOR_COLS,
        dma_result=dma_result,
        ml_results=ml_results,
        hybrid_results=hybrid_results,
        y=y, X=X,
        n_insample=n_insample,
        output_dir=output_dir,
        run_sensitivity=not args.skip_sensitivity,
        dates=dates,
    )
    print(f"  Elapsed: {_elapsed(t0)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    _header("Pipeline complete")
    print(f"  Total runtime : {_elapsed(t_total)}")
    print(f"  Output dir    : {os.path.abspath(output_dir)}")
    print()
    print("  Key results (OOS R², March 2008 – February 2026):")
    print(f"    DMA               :  {dma_result.r2_oos['DMA']*100:.2f}%")
    if 'ElasticNet' in ml_results:
        print(f"    ElasticNet        :  {ml_results['ElasticNet'].r2_oos*100:.2f}%")
    if 'Combo_DMA_EN' in hybrid_results:
        print(f"    Combo DMA+EN      :  {hybrid_results['Combo_DMA_EN'].r2_oos*100:.2f}%")
    print(f"    Random Walk       :   0.00%  (benchmark)")
    print()
    print("  Tables saved  : T1–T10 (.csv)")
    print("  Figures saved : F1–F11 (.pdf)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()