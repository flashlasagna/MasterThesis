# FORECASTING COPPER PRICES: A MACHINE LEARNING APPROACH

**Master of Science in Finance — Master Thesis**
*HEC Lausanne, University of Lausanne*

**Author:** Ruben Mimouni\
**Supervisor:** Prof. Thomas Cho\
**Expert Advisor:** Xavier Marconnet
**Submission:** August 2026

---

## Overview

This thesis replicates and extends the Dynamic Model Averaging and Selection (DMA/DMS) framework of Buncic & Moretto (2015) for forecasting monthly CME copper returns. The original paper uses 18 predictor variables and a Kalman filter-based model averaging approach over 2^18 = 262,144 candidate models. We extend the out-of-sample evaluation period to February 2026, introduce seven machine learning benchmarks (Ridge, LASSO, Elastic Net, Bayesian Ridge, Random Forest, XGBoost, and a feed-forward neural network), propose a hybrid framework combining DMA with ML forecasts, and add a set of robustness extensions addressing real-time data availability, the forward-price alignment factor, tree and network hyperparameters, the burn-in window, test power, forecast horizon, directional accuracy and regime dependence.

**Key findings:**
- DMA achieves out-of-sample R² = 30.16% (March 2008 – February 2026). This full-sample figure is dominated by three crisis episodes: 34 crisis months (16% of the sample) supply 46% of squared returns and two-thirds of the gain over the random walk. Outside them R² is 13–19%; classifying months ex ante by the VIX at the forecast origin separates a high-stress state (R² ≈ 45%) from a low-stress state (R² ≈ 10%).
- Regularised linear ML (Ridge: 29.51%, Elastic Net: 29.01%) is statistically indistinguishable from DMA; the pairwise Diebold-Mariano tests could only have detected a gap of roughly ten R² points.
- Within DMA, averaging over the 2^18 model space beats selecting the single most probable model (DMS: 25.76%): the posterior is nearly flat (effective number of models ≈ 148,000) and the top model changes in 47% of months.
- The equal-weight DMA + Elastic Net combination has the highest point estimate (31.15%) but improves on DMA by only one percentage point (not statistically significant); its merit is robustness across regimes and start dates.
- Non-linear learners lag (Random Forest: 21.23%, MLP: 20.49%, XGBoost: 15.39%) and do not recover under cross-validated re-tuning; at this sample size and with this predictor set the signal is well approximated by a linear map.
- Predictability is a one-month phenomenon: R² falls to single digits at h = 2 and no individual month beyond the next carries any signal.

---

## Repository Structure

```
MasterThesis/
│
├── README.md                        # This file
├── main.py                          # Master orchestrator — runs full pipeline (steps 1–12)
├── rerun_outputs.py                 # Regenerate tables/figures from the cache without re-estimating
│
├── data/                            # Input data (22 xlsx files)
│   ├── Spot.xlsx                    # CME copper spot price
│   ├── 3MFWD.xlsx                   # 3-month forward price
│   ├── 3MTB.xlsx                    # 3-month T-Bill rate
│   ├── 10YTB.xlsx                   # 10-year Treasury yield
│   ├── BDI.xlsx                     # Baltic Dry Index
│   ├── Cons.xlsx                    # World copper consumption
│   ├── Prod.xlsx                    # World copper production
│   ├── Indusprod.xlsx               # US Industrial Production
│   ├── Inv.xlsx                     # CME copper inventories
│   ├── Gold.xlsx                    # Gold spot price
│   ├── SOFR3M.xlsx                  # 3-month SOFR
│   ├── SPX.xlsx                     # S&P 500
│   ├── TEDRATE.xlsx                 # TED spread (to Jan 2022)
│   ├── VIX.xlsx                     # CBOE VIX
│   ├── WTI.xlsx                     # WTI crude oil
│   ├── 3MLIBOR.xlsx                 # 3-month USD LIBOR (to Sep 2020)
│   ├── ALCOA.xlsx                   # Alcoa equity
│   ├── AUDUSD.xlsx                  # AUD/USD exchange rate
│   ├── BHP.xlsx                     # BHP Billiton equity
│   ├── FREEPORT.xlsx                # Freeport-McMoRan equity
│   ├── RIO.xlsx                     # Rio Tinto NYSE ADR
│   └── USDCLP.xlsx                  # USD/CLP exchange rate
│
├── src/                             # Source modules
│   ├── __init__.py
│   ├── 01_pipeline.py               # Data pipeline & feature engineering
│   ├── 02_dma_dms.py                # DMA/DMS Kalman filter engine (+ model-probability diagnostics)
│   ├── 03_ml_models.py              # ML models incl. feed-forward network (run_mlp)
│   ├── 04_hybrid.py                 # Hybrid DMA-ML framework
│   ├── 05_evaluation.py             # Tables T1–T6, figures F1–F13
│   ├── 06_extensions.py             # T7–T10, T13, T17, T20, T21, figures F11, F13
│   ├── 07_realtime.py               # T11 real-time availability of predictors
│   ├── 08_omega.py                  # T12 forward alignment factor (burn-in / recursive)
│   ├── 09_tree_tuning.py            # T14 tree hyperparameter validation & sensitivity
│   ├── 10_burnin.py                 # T16 burn-in / OOS-start and training-history sensitivity
│   ├── cache_utils.py               # Pickle helpers for the results cache
│   └── pipeline.py, dma_dms.py, ml_models.py, hybrid.py, evaluation.py,
│       extensions.py, realtime.py, omega.py, tree_tuning.py, burnin.py
│                                    # Import aliases for the numbered modules
│
└── output/                          # Generated automatically on first run
    ├── _cache_results.pkl           # Forecasts cache used by rerun_outputs.py
    ├── T1_summary_stats.csv         ...  T21_regime_table.csv   (see table below)
    └── F1_copper_price_returns.pdf  ...  F13_rolling_r2_regimes.pdf
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost==2.0.3 matplotlib openpyxl
```

### 2. Place data files

Put all 22 `.xlsx` files in the `data/` folder.

### 3. Run the pipeline

**Fast run** (skips multi-horizon, the forgetting-factor grid, the MLP, and the robustness steps 9–12):
```bash
python main.py --data data/ --output output/ --skip_mh --skip_sensitivity --skip_mlp --skip_robustness
```

**Full run** (everything; ~60 min single-core):
```bash
python main.py --data data/ --output output/
```

### 3b. Regenerate tables and figures without re-estimating

`main.py` caches all forecasts in `output/_cache_results.pkl` after step 6. After editing a plotting or table function in `05_evaluation.py` / `06_extensions.py`:
```bash
python rerun_outputs.py                         # steps 7 + 8 from the cache (seconds to ~1 min)
python rerun_outputs.py --figures               # step 7 only
python rerun_outputs.py --extensions            # step 8 only
python rerun_outputs.py --extensions --sensitivity   # include the T8 DMA grid (~12 min)
```
Re-run `main.py` whenever data, predictors, or any model changes. Steps 9–12 re-estimate models by construction and are not covered by the cache; each numbered module can also be run standalone (e.g. `python src/07_realtime.py`).

### 4. View results

All tables (`.csv`) and figures (`.pdf`) are saved to `output/`.

---

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/` | Path to directory containing xlsx files |
| `--output` | `output/` | Path for saving tables and figures |
| `--insample` | `120` | Burn-in period in months (Mar 1998 – Feb 2008) |
| `--skip_mh` | off | Skip multi-horizon forecasting (h=2,3,6) |
| `--skip_mlp` | off | Skip the feed-forward neural network (~4 min) |
| `--skip_sensitivity` | off | Skip the T8 forgetting-factor grid (~12 min) |
| `--skip_robustness` | off | Skip steps 9–12 (T11, T12, T14, T16) |
| `--skip_tree_tuning` | off | Skip only step 11 (T14, ~25 min) |
| `--horizons` | `1,2,3,6` | Comma-separated forecast horizons |

---

## Pipeline Steps

| Step | Module | Description | Runtime |
|------|--------|-------------|---------|
| 1 | `01_pipeline.py` | Load 22 xlsx files, construct 18 predictors, apply 1-month lag | ~3s |
| 2 | `02_dma_dms.py` | DMA/DMS over 262,144 models via Kalman filter (h=1), with model-probability diagnostics | ~90s |
| 3 | `02_dma_dms.py` | DMA/DMS at h=2,3,6 | ~3.5min |
| 4 | `03_ml_models.py` | OLS, Ridge, LASSO, EN, BayesRidge, RF, XGBoost, MLP | ~5min |
| 5 | `03_ml_models.py` | Multi-horizon Elastic Net | ~15s |
| 6 | `04_hybrid.py` | PIP-EN, Stacking, Equal-weight combo; results cached | ~5s |
| 7 | `05_evaluation.py` | Tables T1–T6, figures F1–F10, F12 (rolling hit rate) | ~15s |
| 8 | `06_extensions.py` | T7 DM-HLN, T17 DM power, T8 forgetting factors, T9 directional, T13 DMA-vs-DMS, T20 horizon decomposition, T21 regimes, T10 correlation; figures F11, F13 | ~13min (T8) / ~1min without |
| 9 | `07_realtime.py` | T11 real-time availability (publication-lag and market-only panels) | ~3min |
| 10 | `08_omega.py` | T12 burn-in and recursive forward alignment factor | ~3min |
| 11 | `09_tree_tuning.py` | T14 CV-tuned RF/XGB and hyperparameter sensitivity surfaces | ~25min |
| 12 | `10_burnin.py` | T16 burn-in / OOS-start sensitivity and training-history variants | ~20min |

---

## Output Tables and Figures

| File | Content | Thesis |
|------|---------|--------|
| T1 | Summary statistics | Table 1 |
| T2 | One-step-ahead OOS evaluation, all models | Table 2 |
| T3 | Multi-horizon results | Table 6 |
| T4 | Hybrid model results | Table 5 |
| T5 | Sub-period R² | Table 7 |
| T6 | Feature importance (DMA PIP, RF, XGB) | Appendix Table 1 |
| T7 | Pairwise Diebold-Mariano (HLN) | Table 3, Panel A |
| T8 | Forgetting-factor grid | Table 8 |
| T9 | Directional accuracy, Pesaran-Timmermann | Table 4 |
| T10 | Predictor correlation matrix | Figure A12 |
| T11 | Real-time availability: publication-lag and market-only panels | Table 9 |
| T12 | Forward alignment factor ω: burn-in and recursive | Table 10 |
| T13 | DMA vs DMS: model-probability diagnostics, sub-periods | Table 11 |
| T14 | Tree hyperparameters: CV-tuned variants and sensitivity surfaces | Table 12 |
| T15 | MLP α sweep and chosen parameters | Table 13 |
| T16 | Burn-in / OOS start (Panels A, B) and training history (Panel C) | Table 14 |
| T17 | Power of the DM tests: minimum detectable difference | Table 3, Panel B |
| T18 | Coefficient-path stability, Ridge vs DMA | quoted in §5.2 |
| T19 | Rolling and sub-period hit rates | §5.4 table, F12 |
| T20 | Horizon decomposition: dilution benchmark, single-month targets | §5.6 table |
| T21 | Predictability by regime: ex-post crisis share, ex-ante VIX / vol classifiers, rolling R² | §5.7 table, F13 |
| F1–F11 | As in the original pipeline | Figures 1–5, A1–A12 |
| F12 | 36-month rolling hit rate and cumulative correct calls | Figure 6 |
| F13 | 36-month rolling R² with crisis shading and ex-ante high-VIX months | Figure 7 |

---

## Methodology Notes

### Predictor alignment and real-time availability
All 18 predictors are lagged by one month (`shift(1)`) so that `X_t` contains only values dated no later than the end of month `t−1`. Sixteen predictors are market-determined and genuinely real-time. Two are statistical releases published after the forecast origin and revised thereafter: US industrial production (released mid-month `t`) and WBMS consumption/production (released ~2 months after the reference month). `07_realtime.py` re-estimates the leading models with these series lagged further (2 and 3 months) and with both dropped; neither change reduces any model's R² (T11).

### TED spread reconstruction
The FRED TED spread was discontinued in January 2022. We reconstruct a continuous series by stitching 3-month LIBOR (to September 2020) with 3-month SOFR, adjusted by a constant LIBOR-SOFR spread estimated over the 534-day overlap window (July 2018 – September 2020). The reconstructed series correlates 0.98 with the original FRED series.

### Target variable and aggregation
Copper returns are computed from monthly-average spot prices following Buncic & Moretto (2015); the convenience yield uses the same aggregation. All other price and index series are sampled at the month-end close to avoid the Working (1960) time-aggregation bias. Dropping the lagged copper return costs only 1–2 points of one-month R² (T20), so the aggregation effect accounts for a small share of the signal.

### Forward-price alignment factor (ω)
The CME spot and 3-month forward series come from different vendor feeds with different quotation conventions, so the forward is rescaled by ω = 1.6428, the mean spot-to-forward ratio over the full overlapping sample. A fixed ω is an affine transformation absorbed by per-step standardisation for the ML models. Because the full-sample estimate uses data beyond each forecast origin, `08_omega.py` re-estimates all models with ω fixed on the burn-in window only (1.6434) and with a recursively estimated ω_t: DMA changes by ≤ 0.01 R² points, Ridge by 0.15 (T12).

### Hyperparameters
Ridge, LASSO, Elastic Net and the MLP are retuned every 12 steps by 3-fold expanding-window `TimeSeriesSplit` CV on the training window. Bayesian Ridge is tuned by marginal likelihood at every step. Random Forest (200 trees, depth 4, min leaf 5) and XGBoost (100 rounds, depth 3, rate 0.05, subsample 0.8) are **fixed ex ante and not tuned**; `09_tree_tuning.py` applies the same CV protocol to both and maps sensitivity surfaces over depth, leaf size, rounds and learning rate (T14). The XGBoost surface is governed by rounds × learning rate and peaks at 20.8% (50 rounds at 0.05); the fixed baseline is slightly over-boosted. Neither tuning nor any fixed setting brings a tree model within six R² points of Ridge.

### Neural network benchmark
The feed-forward network (`run_mlp` in `03_ml_models.py`) is a single-hidden-layer MLP (8 or 16 ReLU units) fitted by L-BFGS, averaged over five random initialisations at every step, with width and L2 penalty re-selected annually by `TimeSeriesSplit` CV. Preliminary runs with weak penalties (α ≤ 10) overfit severely (R² < −25%); the grid therefore runs from 10 to 1000. The fixed-penalty surface (T15) peaks at 25.7% at α = 300 and collapses to the historical-average value at α = 3000; the CV-tuned network attains 20.49%. An LSTM was explored in early development and removed: it changes the information set (12-month sequences) and is not comparable under the common protocol.

### DMA vs DMS diagnostics
`02_dma_dms.py` records, at every forecast origin, the probability of the most probable model, the effective number of models 1/Σπ², the selected model's index and size, and the top-10 probability mass (`DMAResult.diagnostics`). T13 shows the posterior never concentrates (mean π_max = 0.0043%, N_eff ≈ 148,000 of 262,144) and the DMS-selected model changes in 47% of months — the reason averaging beats selection.

### Results cache
`main.py` pickles the estimated results after step 6 to `output/_cache_results.pkl`. Result dataclasses are converted to plain dicts via `src/cache_utils.py` because the numbered modules are loaded through aliases and their classes are not importable by pickle; `rerun_outputs.py` rebuilds them as attribute-style objects.

### XGBoost reproducibility
XGBoost's histogram tree builder is not bit-reproducible across library versions and platforms; the same specification returned 12.54% under an earlier environment and 15.39% under the pinned `xgboost==2.0.3`. All reported figures use the pinned version.

### Import aliases
Python cannot import modules whose filenames start with a digit. The `pipeline.py`, `dma_dms.py`, etc. files in `src/` are thin aliases that dynamically load the corresponding numbered files. Both versions must be present in `src/`.

---

## Results Summary

### One-step-ahead OOS R² (March 2008 – February 2026)

| Model | R²_os |
|-------|-------|
| Combo DMA + EN | 31.15% |
| **DMA** | **30.16%** |
| Ridge | 29.51% |
| PIP-ElasticNet | 29.27% |
| Elastic Net | 29.01% |
| Bayesian Ridge | 27.14% |
| LASSO | 26.46% |
| DMS | 25.76% |
| Stacking | 24.63% |
| Random Forest | 21.23% |
| MLP (seed-ensembled) | 20.49% |
| OLS | 17.93% |
| TVP | 17.43% |
| XGBoost | 15.39% |
| HA (expanding) | −1.31% |

The combination's one-point lead over DMA is not statistically significant (HLN p = 0.61) and is well below the ~5-point difference the test could detect.

### Pairwise Diebold-Mariano (HLN-corrected, top four models)

None of the six pairwise comparisons is significant at any conventional level (smallest p = 0.29). DMA vs Ridge (p = 0.86) and DMA vs Elastic Net (p = 0.77) are far from rejection. The minimum detectable difference at 80% power is 10–11 R² points for these pairs (T17), so the tests establish the absence of a large advantage for either approach, not their equivalence. The MLP is the one model the leaders can distinguish from (DMA vs MLP p = 0.05, Ridge vs MLP p = 0.04).

### Predictability by regime (T21)

| State | N | DMA | Ridge | Combo |
|-------|---|-----|-------|-------|
| Crisis windows (ex post: GFC, 2020, 2022) | 34 | 43.4% | 48.4% | 46.9% |
| Non-crisis | 182 | 18.8% | 13.3% | 17.6% |
| High-stress, ex ante (VIX_{t−1} > expanding 75th pct) | 45 | 45.1% | 47.6% | 47.6% |
| Low-stress, ex ante | 171 | 12.9% | 8.6% | 12.1% |

Crisis months are 15.7% of the sample, 46.3% of squared returns, and 66–76% of the models' total gain over the random walk. Calm-state Clark–West statistics exceed 3.4 in every definition.

### Sub-period R²

| Period | N | DMA | Elastic Net |
|--------|---|-----|-------------|
| Pre-GFC (Mar–Aug 2008) | 6 | −0.11 | 0.07 |
| GFC (Sep 2008–Jun 2009) | 10 | 0.48 | 0.51 |
| Recovery (Jul 2009–Dec 2019) | 126 | 0.24 | 0.15 |
| COVID-19 (Jan–Dec 2020) | 12 | 0.26 | 0.50 |
| Supercycle (Jan–Dec 2021) | 12 | 0.21 | 0.14 |
| Ukraine shock (Jan–Dec 2022) | 12 | 0.37 | 0.27 |
| Post-shock (Jan 2023–Feb 2026) | 38 | 0.03 | 0.08 |

### Burn-in / OOS start (T16)

Opening the evaluation window after the GFC (March 2010 or later) roughly halves every model's full-window R² to 13–18%, with Clark–West significance preserved; the 120-month baseline is the most favourable start date. Under an expanding window, changing the burn-in leaves every forecast unchanged (Panel B); training-history length is varied separately by truncating the sample start and by rolling windows (Panel C), with effects under two R² points for ≥ 72 months of history.

### Horizon (T3, T20)

R² falls from ~30% at h = 1 to 5–9% at h = 2 and zero beyond. A two-month forecast using the h = 1 forecast for month one and zero for month two attains 17–18%, beating the direct h = 2 models; no single month beyond the next has predictive content (R² < 0).

---

## Data Sources

| Series | Source | Coverage |
|--------|--------|----------|
| CME copper spot & forward | Refinitiv Datastream | Jan 1977 – Feb 2026 |
| CME inventories | Refinitiv Datastream | Jan 1970 – Feb 2026 |
| World consumption & production | World Bureau of Metal Statistics | Jan 1998 – Feb 2026 |
| US Industrial Production | FRED | Jan 1970 – May 2026 |
| Treasury rates (3M, 10Y) | FRED | Jan 1970 – Feb 2026 |
| LIBOR 3M | FRED | Jan 1986 – Sep 2020 |
| SOFR 3M | FRED | Jul 2018 – Feb 2026 |
| VIX | FRED / CBOE | Jan 1990 – Feb 2026 |
| S&P 500, Gold, WTI, BDI | Refinitiv Datastream | Various |
| Mining stocks (Alcoa, BHP, FCX, Rio) | Refinitiv Datastream | Various |
| AUD/USD, USD/CLP | Refinitiv Datastream | Various |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
xgboost==2.0.3        # pinned: histogram builder is not bit-reproducible across versions
matplotlib>=3.7.0
openpyxl>=3.1.0
```

---

## Academic Context

This project was developed as a Master's thesis for the MSc Finance programme at HEC Lausanne, University of Lausanne. It replicates and extends:

> Buncic, D. & Moretto, C. (2015). Forecasting copper prices with dynamic
> averaging and selection models. *The North American Journal of Economics
> and Finance*, 33, 1–38. https://doi.org/10.1016/j.najef.2015.03.002

---

## Acknowledgements

- **Prof. Thomas Cho** for supervision and guidance
- **Xavier Marconnet** for his precious advice and improvement suggestions
- **Anthropic Claude**, **Google Gemini** for development assistance

---

## Licence

Submitted in partial fulfilment of the requirements for the Master of Science in Finance degree, HEC Lausanne, August 2026.

---