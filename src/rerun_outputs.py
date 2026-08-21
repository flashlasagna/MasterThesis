"""
rerun_outputs.py
================
Regenerate tables and figures from cached estimation results WITHOUT
re-running the DMA filter or the ML models.

main.py writes output/_cache_results.pkl after step 6 (all forecasts).
This script loads it and re-runs only the cheap output stages:

    python3 rerun_outputs.py                 # step 7 (T1-T6, F1-F12) + step 8 (T7, T9, T10, T13, T17)
    python3 rerun_outputs.py --figures       # step 7 only
    python3 rerun_outputs.py --extensions    # step 8 only (no T8 grid)
    python3 rerun_outputs.py --extensions --sensitivity   # include the T8 DMA grid (~12 min)

Edit a plotting or table function in src/05_evaluation.py or
src/06_extensions.py, save, and re-run this script: results appear in
output/ in seconds. Delete the cache (or re-run main.py) whenever the
DATA, the predictor construction, or any MODEL changes -- the cache
holds forecasts, not code.
"""
import argparse, pickle, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_full_evaluation
from src.extensions import run_all_extensions
from src.cache_utils import from_plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='output/')
    ap.add_argument('--figures', action='store_true', help='Run step 7 only')
    ap.add_argument('--extensions', action='store_true', help='Run step 8 only')
    ap.add_argument('--sensitivity', action='store_true',
                    help='Include the T8 forgetting-factor grid in step 8 (slow)')
    args = ap.parse_args()
    run_fig = args.figures or not args.extensions
    run_ext = args.extensions or not args.figures

    cache = Path(args.output) / '_cache_results.pkl'
    if not cache.exists():
        sys.exit(f"No cache at {cache}. Run main.py once first.")
    with open(cache, 'rb') as fh:
        c = from_plain(pickle.load(fh))
    print(f"Loaded cache: {cache}  (DMA R2 = {c['dma_result'].r2_oos['DMA']*100:.2f}%, "
          f"{len(c['ml_results'])} ML models, {len(c['hybrid_results'])} hybrids)")

    y = c['df']['r_copper'].values
    X = c['df'][c['predictor_cols']].values

    if run_fig:
        t0 = time.time(); print("\n=== Step 7: tables T1-T6, figures F1-F12 ===")
        run_full_evaluation(df=c['df'], predictor_cols=c['predictor_cols'],
                            dma_result=c['dma_result'], ml_results=c['ml_results'],
                            hybrid_results=c['hybrid_results'], n_insample=c['n_insample'],
                            output_dir=args.output, dma_mh_results=c['dma_mh_results'])
        print(f"  done in {time.time()-t0:.0f}s")

    if run_ext:
        t0 = time.time(); print("\n=== Step 8: extensions T7, T9, T10, T13, T17"
                                + (", T8" if args.sensitivity else "") + " ===")
        run_all_extensions(df=c['df'], predictor_cols=c['predictor_cols'],
                           dma_result=c['dma_result'], ml_results=c['ml_results'],
                           hybrid_results=c['hybrid_results'], y=y, X=X,
                           n_insample=c['n_insample'], output_dir=args.output,
                           run_sensitivity=args.sensitivity, dates=c['dates'])
        print(f"  done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()