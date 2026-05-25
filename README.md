# Forecasting Copper Prices: A Replication and ML Extension of Buncic & Moretto (2014)

**Master of Science in Finance — Master Thesis**  
*HEC Lausanne, University of Lausanne*

**Author:** Ruben Mimouni  
**Supervisor:** Prof. Cho  
**Date:** August 2025

---

## Overview

This thesis replicates and extends the Dynamic Model Averaging (DMA/DMS) framework of Buncic & Moretto (2014) for forecasting monthly CME copper returns. The original paper uses 18 predictor variables and a Kalman filter-based model averaging approach over 2^18 = 262,144 candidate models. We extend the out-of-sample evaluation period to February 2026, introduce five machine learning benchmarks, and propose a hybrid framework combining DMA with ML forecasts.

**Key findings:**
- DMA achieves out-of-sample R² = 30.1% (March 2008 – February 2026)
- Regularised linear ML (Ridge: 29.6%, Elastic Net: 29.0%) converges with DMA once predictors are correctly lagged
- Equal-weight DMA + Elastic Net combination achieves the best R² of 31.1%
- Predictability is episodic: R² = 47.5% during the GFC, 37.5% during the Ukraine shock, but only 3.9% post-2023

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
│   ├── 05_evaluation.py             # Tables, figures, evaluation
│   ├── pipeline.py                  # Import alias for 01_pipeline.py
│   ├── dma_dms.py                   # Import alias for 02_dma_dms.py
│   ├── ml_models.py                 # Import alias for 03_ml_models.py
│   ├── hybrid.py                    # Import alias for 04_hybrid.py
│   └── evaluation.py               # Import alias for 05_evaluation.py
│
└── output/                          # Generated automatically on first run
    ├── T1_summary_stats.csv
    ├── T2_oos_results.csv
    ├── T3_multihorizon.csv
    ├── T4_hybrid_results.csv
    ├── T5_subperiod.csv
    ├── T6_feature_importance.csv
    ├── F1_copper_price_returns.png
    ├── F2_predictor_grid.png
    ├── F3_actual_predicted.png
    ├── F4_cumulative_sfe.png
    ├── F5_scatter_dma.png
    ├── F6_pip_grid.png
    ├── F7_beta_path.png
    ├── F8_r2_barchart.png
    ├── F9_crisis_zoom.png
    └── F10_lasso_coef_path.png
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

**Fast run** (~5–7 minutes, skips LSTM and multi-horizon):
```bash
python main.py --data data/ --output output/ --skip_lstm --skip_mh
```

**Full run** (~20–25 minutes, includes multi-horizon h=2,3,6):
```bash
python main.py --data data/ --output output/
```

**Full run with LSTM** (~45–60 minutes):
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
| `--horizons` | `1,2,3,6` | Comma-separated forecast horizons |

---

## Pipeline Steps

| Step | Module | Description | Runtime |
|------|--------|-------------|---------|
| 1 | `01_pipeline.py` | Load 22 xlsx files, construct 18 predictors, apply 1-month lag | ~3s |
| 2 | `02_dma_dms.py` | DMA/DMS over 262,144 models via Kalman filter (h=1) | ~75s |
| 3 | `02_dma_dms.py` | DMA/DMS at h=2,3,6 | ~3.5min |
| 4 | `03_ml_models.py` | Ridge, LASSO, EN, BayesRidge, RF, XGBoost, LSTM | ~75s |
| 5 | `03_ml_models.py` | Multi-horizon Elastic Net | ~15s |
| 6 | `04_hybrid.py` | PIP-EN, Stacking, Equal-weight combo | ~5s |
| 7 | `05_evaluation.py` | All 6 tables and 10 figures | ~12s |

---

## Methodology Notes

### Predictor alignment
All 18 predictors are lagged by one month (`shift(1)`) so that `X_t` contains only information observable at the end of month `t−1`. This prevents contemporaneous bias when forecasting the month-`t` copper return.

### TED spread reconstruction
The FRED TED spread was discontinued in January 2022. We reconstruct a continuous series by stitching 3-month LIBOR (to September 2020) with 3-month SOFR, adjusted by a constant spread of 0.15 bps estimated over the 534-day overlap window. The reconstructed series correlates 0.98 with the original FRED series.

### Target variable
Copper returns are computed as monthly average price changes following Buncic & Moretto (2014), who explicitly use monthly average LME prices. The convenience yield and forward-spot spread use the same monthly average aggregation for internal consistency.

### Import aliases
Python cannot import modules whose filenames start with a digit. The `pipeline.py`, `dma_dms.py`, etc. files in `src/` are thin aliases that dynamically load the corresponding numbered files. Both versions must be present in `src/`.

---

## Results Summary

### One-step-ahead OOS R² (March 2008 – February 2026)

| Model | R²_os |
|-------|-------|
| **Combo DMA + EN** | **31.1%** |
| DMA | 30.1% |
| Ridge | 29.6% |
| PIP-ElasticNet | 29.4% |
| Elastic Net | 29.0% |
| Bayesian Ridge | 27.1% |
| LASSO | 26.3% |
| DMS | 25.4% |
| Stacking | 24.2% |
| Random Forest | 21.3% |
| TVP | 17.5% |
| OLS | 17.2% |
| XGBoost | 14.0% |
| HA (expanding) | −1.3% |

### Sub-period R² — DMA

| Period | N | DMA | Elastic Net |
|--------|---|-----|-------------|
| Pre-GFC (Mar–Aug 2008) | 6 | −0.13 | 0.06 |
| GFC (Sep 2008–Jun 2009) | 10 | 0.48 | 0.51 |
| Recovery (Jul 2009–Dec 2019) | 126 | 0.24 | 0.15 |
| COVID-19 (Jan–Dec 2020) | 12 | 0.25 | 0.50 |
| Supercycle (Jan–Dec 2021) | 12 | 0.21 | 0.14 |
| Ukraine shock (Jan–Dec 2022) | 12 | 0.37 | 0.27 |
| Post-shock (Jan 2023–Feb 2026) | 38 | 0.04 | 0.08 |

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

- **Prof. Cho** for supervision and guidance
- **Anthropic Claude** for development assistance

---

## Licence

Academic use only. Submitted in partial fulfilment of the requirements for the Master of Science in Finance degree, HEC Lausanne, August 2025.

---

<p align="center">
  <i>Built with ☕ in Lausanne, Switzerland</i>
</p>
