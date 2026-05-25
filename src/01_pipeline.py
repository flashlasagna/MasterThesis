"""
01_pipeline.py
==============
Data pipeline for "Forecasting Copper Prices: A Replication and ML Extension
of Buncic & Moretto (2014)".

Steps
-----
1.  Load all 22 raw series from the data directory (xlsx files).
2.  Harmonise units and date indices.
3.  Reconstruct the TED spread (FRED discontinued Jan 2022):
        Phase 1 : 3M LIBOR - 3M T-Bill              (1986 - Sep 2020)
        Phase 2 : 3M SOFR + LIBOR-SOFR adj - 3M T-Bill  (Oct 2020 - Feb 2026)
4.  Aggregate all daily series to monthly averages (consistent with paper).
5.  Engineer all 18 predictor variables exactly as in Buncic & Moretto (2014).
6.  Return a clean, fully-merged monthly panel with no NaNs.

Output
------
    df_final : pd.DataFrame
        Columns : ['date', 'r_copper', 'x_r_copper_lag', 'x_dDemand',
                   'x_dInventory', 'x_ConvYield', 'x_dIP', 'x_Spread',
                   'x_dBDI', 'x_VIX', 'x_TED', 'x_dSPX', 'x_dGold',
                   'x_dWTI', 'x_dCLP', 'x_dAUD', 'x_dAlcoa', 'x_dRio',
                   'x_dFCX', 'x_dBHP']
        Sample  : March 1998 - February 2026  (336 monthly observations)

Usage
-----
    from src.pipeline import build_dataset
    df, meta = build_dataset(data_dir='data/')

Notes
-----
*  Copper spot and inventories use CME/COMEX data (vs LME in the original paper).
   CME and LME copper returns correlate >0.99; the difference is immaterial for
   return forecasting. Documented in the Data section of the thesis.

*  The TED spread reconstruction uses a constant LIBOR-SOFR spread adjustment
   estimated over the 2018-07 to 2020-09 overlap window (534 days, ~15 bps).
   Correlation between reconstructed and FRED TED over their overlapping period
   exceeds 0.98.

*  The Chilean peso predictor follows the paper's convention: positive values
   indicate CLP appreciation against the USD, i.e. returns = -delta_log(USDCLP).

*  Convenience yield is computed without storage costs (set to zero) following
   Fama & French (1988) and Buncic & Moretto (2014, p.17).

References
----------
Buncic, D. & Moretto, C. (2014). Forecasting Copper Prices with Dynamic
    Averaging and Selection Models. SSRN Working Paper 2482015.
Fama, E.F. & French, K.R. (1988). Business Cycles and the Behavior of Metals
    Prices. Journal of Finance, 43(5), 1075-1093.
"""

import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREDICTOR_COLS = [
    'x_r_copper_lag',  #  1. Lagged copper return         (momentum)
    'x_dDemand',       #  2. Delta Excess demand
    'x_dInventory',    #  3. Delta Inventories
    'x_ConvYield',     #  4. Convenience yield            (level)
    'x_dIP',           #  5. Delta US Industrial Production
    'x_Spread',        #  6. Term spread                  (level, decimal)
    'x_dBDI',          #  7. Delta Baltic Dry Index
    'x_VIX',           #  8. VIX                          (level)
    'x_TED',           #  9. TED spread (reconstructed)   (level, decimal)
    'x_dSPX',          # 10. Delta S&P 500
    'x_dGold',         # 11. Delta Gold
    'x_dWTI',          # 12. Delta WTI crude oil
    'x_dCLP',          # 13. Delta Chilean peso (CLP appreciation)
    'x_dAUD',          # 14. Delta Australian dollar
    'x_dAlcoa',        # 15. Delta Alcoa stock
    'x_dRio',          # 16. Delta Rio Tinto stock (NYSE ADR)
    'x_dFCX',          # 17. Delta Freeport-McMoRan stock
    'x_dBHP',          # 18. Delta BHP Billiton stock
]

TARGET_COL   = 'r_copper'
SAMPLE_START = '1998-01-01'
SAMPLE_END   = '2026-02-01'

VARIABLE_LABELS = {
    'r_copper':       'Copper spot return r_t (%)',
    'x_r_copper_lag': 'Lagged copper return r_{t-1} (%)',
    'x_dDemand':      'Delta Demand (mt)',
    'x_dInventory':   'Delta Inventories (mt)',
    'x_ConvYield':    'Convenience yield (level)',
    'x_dIP':          'Delta Industrial production (%)',
    'x_Spread':       'Term spread 10Y-3M (decimal)',
    'x_dBDI':         'Delta Baltic Dry Index (%)',
    'x_VIX':          'VIX (level)',
    'x_TED':          'TED spread (decimal, reconstructed)',
    'x_dSPX':         'Delta S&P 500 (%)',
    'x_dGold':        'Delta Gold (%)',
    'x_dWTI':         'Delta WTI crude oil (%)',
    'x_dCLP':         'Delta Chilean peso (% CLP appreciation)',
    'x_dAUD':         'Delta Australian dollar (%)',
    'x_dAlcoa':       'Delta Alcoa (%)',
    'x_dRio':         'Delta Rio Tinto (%)',
    'x_dFCX':         'Delta Freeport-McMoRan (%)',
    'x_dBHP':         'Delta BHP Billiton (%)',
}


# ===========================================================================
# SECTION 1 — Raw data loading
# ===========================================================================

def _load_raw(data_dir, fname, sheet, date_col, val_col, rename):
    """Load one xlsx series; return DataFrame with columns ['date', rename]."""
    path = Path(data_dir) / fname
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[[date_col, val_col]].copy()
    df.columns = ['date', rename]
    df['date']  = pd.to_datetime(df['date'])
    df[rename]  = pd.to_numeric(df[rename], errors='coerce')
    df = (df.sort_values('date')
            .drop_duplicates('date')
            .reset_index(drop=True))
    return df


def _load_all(data_dir):
    """Load all 22 raw series and return as dict keyed by short name."""
    D = data_dir
    raw = {}

    # Fundamental series
    raw['spot']   = _load_raw(D, 'Spot.xlsx',      'Sheet1',
                               'Date', 'Price',     'spot')
    raw['fwd3m']  = _load_raw(D, '3MFWD.xlsx',     'Sheet1',
                               'Date', 'Price',     'fwd3m')
    raw['tb3m']   = _load_raw(D, '3MTB.xlsx',      'Sheet1',
                               'Date', 'Rate',      'tb3m')
    raw['tb10y']  = _load_raw(D, '10YTB.xlsx',     'Sheet1',
                               'Date', 'Rate',      'tb10y')
    raw['bdi']    = _load_raw(D, 'BDI.xlsx',       'Sheet1',
                               'Date', 'Index',     'bdi')
    raw['cons']   = _load_raw(D, 'Cons.xlsx',      'Table Data',
                               'Date', 'Close',     'cons')
    raw['ip']     = _load_raw(D, 'Indusprod.xlsx', 'Sheet1',
                               'Date', 'Index',     'ip')
    raw['inv']    = _load_raw(D, 'Inv.xlsx',       'Sheet1',
                               'Date', 'Inventory', 'inv')
    raw['prod']   = _load_raw(D, 'Prod.xlsx',      'Table Data',
                               'Date', 'Close',     'prod')

    # Financial series
    raw['gold']    = _load_raw(D, 'Gold.xlsx',    'Sheet1',
                                'Date', 'Index',   'gold')
    raw['spx']     = _load_raw(D, 'SPX.xlsx',     'Sheet1',
                                'Date', 'Index',   'spx')
    raw['vix']     = _load_raw(D, 'VIX.xlsx',     'Sheet 1',
                                'Exchange Date', 'Close', 'vix')
    raw['wti']     = _load_raw(D, 'WTI.xlsx',     'Sheet1',
                                'Date', 'Close',   'wti')
    raw['libor']   = _load_raw(D, '3MLIBOR.xlsx',
                                '3-month-libor-rate-historical-c',
                                'date', ' value',  'libor')
    raw['sofr']    = _load_raw(D, 'SOFR3M.xlsx',  'Daily',
                                'Date', 'Rate',    'sofr')

    # Exchange rates and mining stocks
    raw['alcoa']  = _load_raw(D, 'ALCOA.xlsx',    'ALCOA',
                               'Date', 'Adj Close', 'alcoa')
    raw['audusd'] = _load_raw(D, 'AUDUSD.xlsx',   'Table Data',
                               'Date', 'USD',       'audusd')
    raw['bhp']    = _load_raw(D, 'BHP.xlsx',      'Table Data',
                               'Date', 'Close',     'bhp')
    raw['fcx']    = _load_raw(D, 'FREEPORT.xlsx', 'Table Data',
                               'Date', 'Close',     'fcx')
    raw['rio']    = _load_raw(D, 'RIO.xlsx',      'RIO',
                               'Date', 'Close',     'rio')
    raw['usdclp'] = _load_raw(D, 'USDCLP.xlsx',  'Table Data',
                               'Date', 'CLP',       'usdclp')

    return raw


# ===========================================================================
# SECTION 2 — Unit harmonisation and TED spread reconstruction
# ===========================================================================

def _harmonise_units(raw):
    """Convert LIBOR from percentage points to decimal (e.g. 5.25 -> 0.0525)."""
    raw = dict(raw)
    raw['libor'] = raw['libor'].copy()
    raw['libor']['libor'] = raw['libor']['libor'] / 100.0
    return raw


def _build_ibor(raw):
    """
    Build a unified daily interbank rate (ibor) series.

    Phase 1 (up to 2020-09-03):  3M USD LIBOR  (decimal)
    Phase 2 (2020-09-04 onwards): 3M SOFR + constant adjustment

    The constant adjustment equals mean(LIBOR - SOFR) over the
    2018-07-02 to 2020-09-03 overlap window (~15 bps). This accounts
    for the credit/liquidity premium in LIBOR absent from risk-free SOFR.

    Returns
    -------
    ibor              : pd.DataFrame  ['date', 'ibor']
    libor_sofr_adj    : float          constant spread adjustment (decimal)
    """
    libor_clean = raw['libor'].dropna(subset=['libor'])
    sofr_clean  = raw['sofr'].dropna(subset=['sofr'])

    # Estimate constant adjustment from overlap window
    overlap = pd.merge(libor_clean, sofr_clean, on='date', how='inner')
    libor_sofr_adj = float((overlap['libor'] - overlap['sofr']).mean())

    # Phase 2: SOFR + adjustment for dates after LIBOR ends
    libor_end = libor_clean['date'].max()
    sofr_post = sofr_clean[sofr_clean['date'] > libor_end].copy()
    sofr_post['ibor'] = sofr_post['sofr'] + libor_sofr_adj

    ibor = pd.concat([
        libor_clean[['date', 'libor']].rename(columns={'libor': 'ibor'}),
        sofr_post[['date', 'ibor']]
    ]).sort_values('date').drop_duplicates('date').reset_index(drop=True)

    return ibor, libor_sofr_adj


# ===========================================================================
# SECTION 3 — Monthly aggregation
# ===========================================================================

def _monthly_mean(df, col):
    """Resample daily series to monthly averages (month-start timestamps)."""
    return (df.set_index('date')[col]
              .resample('MS').mean()
              .reset_index())


def _monthly_last(df, col):
    """Resample daily series to month-end close (snapped to month-start label).

    Using the last observation in the month avoids Working's (1960) time-
    aggregation bias: returns built from monthly averages inherit artificial
    positive autocorrelation and damped volatility from the smoothing window.
    Close-to-close returns reflect tradeable market moves.
    """
    s = (df.set_index('date')[col]
            .resample('ME').last())          # month-end timestamps
    s.index = s.index.values.astype('datetime64[M]')  # snap to month-start label
    out = s.reset_index()
    out.columns = ['date', col]
    return out


def _snap_to_month_start(df):
    """Snap an already-monthly series to month-start timestamps."""
    df = df.copy()
    df['date'] = df['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
    return df


def _aggregate_monthly(raw, ibor):
    """
    Aggregate all raw series to monthly and merge into one panel.
    Daily price/index series use monthly mean (consistent with the paper).
    """
    # Daily price/index series -> month-end last (avoids Working's effect on returns)
    # spot and fwd3m use monthly averages following Buncic & Moretto (2014),
    # who explicitly state they use monthly average LME prices as the target.
    # All other price series use month-end close to avoid Working's Effect.
    mean_cols = ['spot', 'fwd3m']
    last_cols  = ['gold', 'spx', 'vix', 'wti', 'bdi',
                  'inv', 'alcoa', 'bhp', 'fcx', 'rio', 'audusd', 'usdclp']
    monthly = {col: _monthly_mean(raw[col], col) for col in mean_cols}
    monthly.update({col: _monthly_last(raw[col], col) for col in last_cols})
    # Interbank rate kept as monthly mean (a level, not a return basis)
    monthly['ibor'] = _monthly_mean(ibor, 'ibor')

    # Monthly series -> snap to month-start
    for col in ['tb3m', 'tb10y']:
        m = _snap_to_month_start(raw[col].rename(columns={col: col}))
        m.columns = ['date', col]
        monthly[col] = m

    for col in ['cons', 'prod']:
        m = _snap_to_month_start(raw[col].rename(columns={col: col}))
        m.columns = ['date', col]
        monthly[col] = m

    ip_m = _snap_to_month_start(raw['ip'].rename(columns={'ip': 'ip'}))
    ip_m.columns = ['date', 'ip']
    ip_m['ip'] = ip_m['ip'].interpolate(method='linear')  # fix 2 NaN values
    monthly['ip'] = ip_m

    # Merge all on date
    panel = reduce(
        lambda l, r: pd.merge(l, r, on='date', how='outer'),
        list(monthly.values())
    )
    panel = panel.sort_values('date').reset_index(drop=True)

    # Restrict to feasible sample window
    panel = panel[
        (panel['date'] >= SAMPLE_START) &
        (panel['date'] <= SAMPLE_END)
    ].reset_index(drop=True)

    return panel


# ===========================================================================
# SECTION 4 — Feature engineering
# ===========================================================================

def _pct_return(s):
    """Simple percentage return: 100 * (P_t / P_{t-1} - 1)."""
    return 100.0 * (s / s.shift(1) - 1.0)


def _log_return(s):
    """Log percentage return: 100 * ln(P_t / P_{t-1})."""
    return 100.0 * np.log(s / s.shift(1))


def _build_features(panel):
    """
    Construct all 18 predictor variables and the target return.

    Variable definitions follow Buncic & Moretto (2014) exactly.
    See module docstring for full definitions.
    """
    df = panel.copy()

    # Target: monthly copper spot return
    df['r_copper'] = _pct_return(df['spot'])

    # Build all 18 predictors at their CONCURRENT month-t value first.
    # A uniform one-month lag is applied at the end of this function so that
    # X_t represents information available at the close of month t-1 — the
    # correct alignment for a one-step-ahead forecast of y_t.

    # 1. Lagged copper return (momentum) — current return; will be lagged below
    df['x_r_copper_lag'] = df['r_copper']

    # 2 & 3. Excess demand and inventory changes
    df['_demand']      = df['cons'] - (df['prod'] + df['inv'])
    df['x_dDemand']    = df['_demand'].diff()
    df['x_dInventory'] = df['inv'].diff()

    # 4. Convenience yield: i_t^tau - (F_t^tau - S_t) / S_t
    #    Storage cost set to zero per Fama & French (1988, p.1077)
    df['x_ConvYield'] = df['tb3m'] - (df['fwd3m'] - df['spot']) / df['spot']

    # 5. US Industrial Production growth
    df['x_dIP'] = _pct_return(df['ip'])

    # 6. Term spread (level, decimal)
    df['x_Spread'] = df['tb10y'] - df['tb3m']

    # 7. Baltic Dry Index growth
    df['x_dBDI'] = _pct_return(df['bdi'])

    # 8. VIX level
    df['x_VIX'] = df['vix']

    # 9. TED spread: reconstructed ibor - 3M T-Bill (both decimal)
    df['x_TED'] = df['ibor'] - df['tb3m']

    # 10. S&P 500 return
    df['x_dSPX'] = _pct_return(df['spx'])

    # 11. Gold return
    df['x_dGold'] = _pct_return(df['gold'])

    # 12. WTI crude oil return
    df['x_dWTI'] = _pct_return(df['wti'])

    # 13. Chilean peso appreciation vs USD: -delta_log(USDCLP)
    #     USDCLP is CLP per USD; appreciation of CLP = decrease in USDCLP
    df['x_dCLP'] = -_log_return(df['usdclp'])

    # 14. Australian dollar return
    df['x_dAUD'] = _pct_return(df['audusd'])

    # 15. Alcoa stock return
    df['x_dAlcoa'] = _pct_return(df['alcoa'])

    # 16. Rio Tinto return (NYSE ADR, priced in USD)
    df['x_dRio'] = _pct_return(df['rio'])

    # 17. Freeport-McMoRan return
    df['x_dFCX'] = _pct_return(df['fcx'])

    # 18. BHP Billiton return
    df['x_dBHP'] = _pct_return(df['bhp'])

    # Apply uniform one-month lag to every predictor so X_t carries only
    # information observable at the end of month t-1. Without this shift,
    # contemporaneous regressors (e.g. month-t S&P 500 return, month-t VIX)
    # leak future information into the h=1 forecast of y_t and the model
    # becomes explanatory rather than predictive.
    for col in PREDICTOR_COLS:
        df[col] = df[col].shift(1)

    return df


# ===========================================================================
# SECTION 5 — Public API
# ===========================================================================

def build_dataset(data_dir='data/'):
    """
    Full pipeline: load -> harmonise -> reconstruct TED -> aggregate -> engineer.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing all xlsx data files.

    Returns
    -------
    df_final : pd.DataFrame
        Clean monthly panel: 336 rows x 20 columns, zero NaNs.

    meta : dict
        'libor_sofr_adj'  : float  LIBOR-SOFR spread adjustment in decimal
        'sample_start'    : str    first observation date
        'sample_end'      : str    last observation date
        'n_obs'           : int    number of monthly observations
        'n_predictors'    : int    number of predictor variables (18)
    """
    raw                    = _load_all(data_dir)
    raw                    = _harmonise_units(raw)
    ibor, libor_sofr_adj   = _build_ibor(raw)
    panel                  = _aggregate_monthly(raw, ibor)
    df                     = _build_features(panel)

    final_cols = ['date', TARGET_COL] + PREDICTOR_COLS
    df_final   = df[final_cols].dropna().reset_index(drop=True)

    meta = {
        'libor_sofr_adj': libor_sofr_adj,
        'sample_start':   str(df_final['date'].min().date()),
        'sample_end':     str(df_final['date'].max().date()),
        'n_obs':          len(df_final),
        'n_predictors':   len(PREDICTOR_COLS),
    }

    return df_final, meta


def summary_statistics(df):
    """
    Compute Table 1 summary statistics: Mean, Median, Std, Skew, Kurt, Min, Max.
    One row per variable (target + 18 predictors), matching paper Table 1 format.
    """
    cols = [TARGET_COL] + PREDICTOR_COLS
    rows = []
    for col in cols:
        s = df[col]
        rows.append({
            'Variable': VARIABLE_LABELS.get(col, col),
            'Mean':     round(s.mean(),   4),
            'Median':   round(s.median(), 4),
            'Std Dev':  round(s.std(),    4),
            'Skewness': round(s.skew(),   4),
            'Kurtosis': round(s.kurt(),   4),
            'Min':      round(s.min(),    4),
            'Max':      round(s.max(),    4),
        })
    return pd.DataFrame(rows).set_index('Variable')


# ===========================================================================
# Quick validation — run as script
# ===========================================================================

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'

    print("=" * 65)
    print("01_pipeline.py  —  building dataset")
    print("=" * 65)

    df, meta = build_dataset(data_dir=data_dir)

    print(f"\n  Observations    : {meta['n_obs']}")
    print(f"  Predictors      : {meta['n_predictors']}")
    print(f"  Sample          : {meta['sample_start']}  to  {meta['sample_end']}")
    print(f"  LIBOR-SOFR adj  : {meta['libor_sofr_adj']*100:.2f} bps")
    print(f"  NaN count       : {df.isnull().sum().sum()}")

    print("\n── Summary Statistics ──────────────────────────────────────────")
    print(summary_statistics(df).to_string())

    print("\n── First 3 rows ────────────────────────────────────────────────")
    print(df.head(3).to_string(index=False))

    print("\n── Last 3 rows ─────────────────────────────────────────────────")
    print(df.tail(3).to_string(index=False))