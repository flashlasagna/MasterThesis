# Forecasting Copper Prices: A Replication and ML Extension of Buncic & Moretto (2014)

**Master of Science in Finance — Master Thesis**
*HEC Lausanne, University of Lausanne*

**Author:** Ruben Mimouni
**Supervisor:** Prof. Thomas Cho
**Submission:** August 2025

---

## Overview

This thesis replicates and extends the Dynamic Model Averaging and Selection (DMA/DMS) framework of Buncic & Moretto (2014) for forecasting monthly CME copper returns. The original paper uses 18 predictor variables and a Kalman filter-based model averaging approach over 2^18 = 262,144 candidate models. We extend the out-of-sample evaluation period to February 2026, introduce five machine learning benchmarks, propose a hybrid framework combining DMA with ML forecasts, and add a set of robustness and value-added extensions (pairwise Diebold-Mariano tests, forgetting-factor sensitivity, directional accuracy, and a predictor correlation matrix).

**Key findings:**
- DMA achieves out-of-sample R² = 30.16% (March 2008 – February 2026)
- Regularised linear ML (Ridge: 29.51%, Elastic Net: 29.01%) converges with DMA once predictors are correctly lagged; pairwise Diebold-Mariano tests cannot distinguish them
- Equal-weight DMA + Elastic Net combination achieves the best R² of 31.15%, improving on DMA by only one percentage point
- Tree-based methods lag (Random Forest: 21.23%, XGBoost: 12.54%)
- Predictability is episodic: R² = 47.95% during the GFC, 37.16% during the Ukraine shock, but only 3.26% post-2023

---

## Repository Structure

```
MasterThesis/
│
├── README.md                        # This file
├── main.py                          # Master orchestrator — runs full pipeline
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
│   ├── 02_dma_dms.py                # DMA/DMS Kalman filter engine
│   ├── 03_ml_models.py              # Machine learning models
│   ├── 04_hybrid.py                 # Hybrid DMA-ML framework
│   ├── 05_evaluation.py             # Tables T1–T6, figures F1–F10
│   ├── 06_extensions.py             # Tier-1 extensions: tables T7–T10, figure F11
│   ├── pipeline.py                  # Import alias for 01_pipeline.py
│   ├── dma_dms.py                   # Import alias for 02_dma_dms.py
│   ├── ml_models.py                 # Import alias for 03_ml_models.py
│   ├── hybrid.py                    # Import alias for 04_hybrid.py
│   ├── evaluation.py                # Import alias for 05_evaluation.py
│   └── extensions.py                # Import alias for 06_extensions.py
│
└── output/                          # Generated automatically on first run
    ├── T1_summary_stats.csv
    ├── T2_oos_results.csv
    ├── T3_multihorizon.csv
    ├── T4_hybrid_results.csv
    ├── T5_subperiod.csv
    ├── T6_feature_importance.csv
    ├── T7_dm_pairwise.csv
    ├── T7_dm_stat_matrix.csv
    ├── T7_dm_pval_matrix.csv
    ├── T8_ff_sensitivity.csv
    ├── T9_directional.csv
    ├── T10_predictor_corr.csv
    ├── F1_copper_price_returns.png
    ├── F2_predictor_grid.png
    ├── F3_actual_predicted.png
    ├── F4_cumulative_sfe.png
    ├── F5_scatter_dma.png
    ├── F6_pip_grid.png
    ├── F7_beta_path.png
    ├── F8_r2_barchart.png
    ├── F9_crisis_zoom.png
    ├── F10_lasso_coef_path.png
    └── F11_predictor_corr_heatmap.png
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib openpyxl
```

For LSTM support (optional, slow on CPU):
```bash
pip install torch
```

### 2. Place data files

Put all 22 `.xlsx` files in the `data/` folder.

### 3. Run the pipeline

**Fast run** (skips LSTM, multi-horizon, and the heavy forgetting-factor grid):
```bash
python main.py --data data/ --output output/ --skip_lstm --skip_mh --skip_sensitivity
```

**Full run** (includes multi-horizon h=2,3,6 and the T8 sensitivity grid):
```bash
python main.py --data data/ --output output/
```

**Full run with LSTM**:
```bash
python main.py --data data/ --output output/ --horizons 1,2,3,6
```

### 4. View results

All tables (`.csv`) and figures (`.png`) are saved to `output/`.

---

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/` | Path to directory containing xlsx files |
| `--output` | `output/` | Path for saving tables and figures |
| `--insample` | `120` | Burn-in period in months (Mar 1998 – Feb 2008) |
| `--skip_mh` | off | Skip multi-horizon forecasting (h=2,3,6) |
| `--skip_lstm` | off | Skip LSTM model (requires PyTorch) |
| `--skip_sensitivity` | off | Skip the T8 forgetting-factor grid (the only compute-heavy extension) |
| `--horizons` | `1,2,3,6` | Comma-separated forecast horizons |

---

## Pipeline Steps

| Step | Module | Description | Runtime |
|------|--------|-------------|---------|
| 1 | `01_pipeline.py` | Load 22 xlsx files, construct 18 predictors, apply 1-month lag | ~3s |
| 2 | `02_dma_dms.py` | DMA/DMS over 262,144 models via Kalman filter (h=1) | ~75s |
| 3 | `02_dma_dms.py` | DMA/DMS at h=2,3,6 | ~3.5min |
| 4 | `03_ml_models.py` | Ridge, LASSO, EN, BayesRidge, RF, XGBoost, (LSTM) | ~75s |
| 5 | `03_ml_models.py` | Multi-horizon Elastic Net | ~15s |
| 6 | `04_hybrid.py` | PIP-EN, Stacking, Equal-weight combo | ~5s |
| 7 | `05_evaluation.py` | Tables T1–T6 and figures F1–F10 | ~12s |
| 8 | `06_extensions.py` | Tables T7–T10 and figure F11; T8 re-runs the DMA grid | ~12min (T8) |

---

## Methodology Notes

### Predictor alignment
All 18 predictors are lagged by one month (`shift(1)`) so that `X_t` contains only information observable at the end of month `t−1`. This prevents contemporaneous bias when forecasting the month-`t` copper return.

### TED spread reconstruction
The FRED TED spread was discontinued in January 2022. We reconstruct a continuous series by stitching 3-month LIBOR (to September 2020) with 3-month SOFR, adjusted by a constant LIBOR-SOFR spread estimated over the 534-day overlap window (July 2018 – September 2020). The reconstructed series correlates 0.98 with the original FRED series.

### Target variable and aggregation
Copper returns are computed from monthly-average spot prices following Buncic & Moretto (2014), who explicitly use monthly average prices for the copper series; the convenience yield uses the same monthly-average aggregation. All other price and index series are sampled at the month-end close to avoid the Working (1960) time-aggregation bias.

### Extensions (T7–T10)
- **T7** — Pairwise Diebold-Mariano tests with the Harvey-Leybourne-Newbold small-sample correction among the four leading models (DMA, Ridge, Elastic Net, Combo). Valid because these models are non-nested, unlike the random-walk comparison which requires Clark-West.
- **T8** — DMA out-of-sample R² across a λ × α forgetting-factor grid. Re-runs the DMA engine per cell (compute-heavy; skippable with `--skip_sensitivity`).
- **T9** — Directional hit-rate and Pesaran-Timmermann test of sign predictability for every model. Exact-zero forecasts scored as misses.
- **T10** — Pearson correlation matrix of the 18 predictors, with a heatmap (F11).

### Import aliases
Python cannot import modules whose filenames start with a digit. The `pipeline.py`, `dma_dms.py`, etc. files in `src/` are thin aliases that dynamically load the corresponding numbered files. Both versions must be present in `src/`.

---

## Results Summary

### One-step-ahead OOS R² (March 2008 – February 2026)

| Model | R²_os |
|-------|-------|
| **Combo DMA + EN** | **31.15%** |
| DMA | 30.16% |
| Ridge | 29.51% |
| PIP-ElasticNet | 29.27% |
| Elastic Net | 29.01% |
| Bayesian Ridge | 27.14% |
| LASSO | 26.46% |
| DMS | 25.76% |
| Stacking | 24.63% |
| Random Forest | 21.23% |
| OLS | 17.93% |
| TVP | 17.43% |
| XGBoost | 12.54% |
| HA (expanding) | −1.31% |

### Pairwise Diebold-Mariano (HLN-corrected, top four models)

None of the six pairwise comparisons is significant at any conventional level (smallest p = 0.29). DMA vs Ridge (p = 0.86) and DMA vs Elastic Net (p = 0.77) are far from rejection: the single-parameter static ridge and the 262,144-model DMA framework produce forecasts of statistically indistinguishable accuracy.

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
xgboost>=2.0.0
matplotlib>=3.7.0
openpyxl>=3.1.0
torch>=2.0.0          # optional, for LSTM
```

---

## Academic Context

This project was developed as a Master's thesis for the MSc Finance programme at HEC Lausanne, University of Lausanne. It replicates and extends:

> Buncic, D. & Moretto, C. (2014). *Forecasting Copper Prices with Dynamic Averaging and Selection Models*. SSRN Working Paper 2482015.

---

## Acknowledgements

- **Prof. Thomas Cho** for supervision and guidance
- **Anthropic Claude**, **Google Gemini** for development assistance

---

## Licence

Submitted in partial fulfilment of the requirements for the Master of Science in Finance degree, HEC Lausanne, August 2025.

---