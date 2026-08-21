"""
05_evaluation.py
================
Comprehensive evaluation, visualisation, and reporting module.

Produces all tables and figures for the thesis from pre-computed model
results. Designed to be run after 02_dma_dms.py, 03_ml_models.py, and
04_hybrid.py have completed.

Outputs
-------
Tables (saved as CSV + formatted string):
    T1  Summary statistics                    → output/T1_summary_stats.csv
    T2  One-step OOS forecast evaluation      → output/T2_oos_results.csv
    T3  Multi-horizon forecast evaluation     → output/T3_multihorizon.csv
    T4  Hybrid model evaluation               → output/T4_hybrid_results.csv
    T5  Sub-period crisis analysis            → output/T5_subperiod.csv
    T6  Feature importance comparison         → output/T6_feature_importance.csv

Figures (saved as PNG at 150 dpi, thesis quality):
    F1  Copper spot price and returns         → output/F1_copper_price_returns.png
    F2  All 18 predictor time series          → output/F2_predictor_grid.png
    F3  Actual vs predicted (DMA + EN)        → output/F3_actual_predicted.png
    F4  Cumulative squared forecast error     → output/F4_cumulative_sfe.png
    F5  Scatter: DMA predicted vs actual      → output/F5_scatter_dma.png
    F6  Posterior inclusion probabilities     → output/F6_pip_grid.png
    F7  DMA time-varying coefficients         → output/F7_beta_path.png
    F8  R² bar chart — all models             → output/F8_r2_barchart.png
    F9  Crisis period zoom (3 events)         → output/F9_crisis_zoom.png
    F10 LASSO coefficient path                → output/F10_lasso_coef_path.png

Usage
-----
    from src.evaluation import run_full_evaluation
    run_full_evaluation(
        df=df, dma_result=dma_result, ml_results=ml_results,
        hybrid_results=hybrid_results, n_insample=120,
        output_dir='output/'
    )
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import norm
import math

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Thesis visual style
# ---------------------------------------------------------------------------

THESIS_STYLE = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         150,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linewidth':     0.5,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.0,
}

# Colour palette — accessible, professional
COLOURS = {
    'actual':  '#2c3e50',   # dark navy  — actual returns
    'dma':     '#e74c3c',   # red        — DMA model
    'en':      '#27ae60',   # green      — ElasticNet
    'rw':      '#7f8c8d',   # grey       — random walk / zero line
    'combo':   '#8e44ad',   # purple     — hybrid / combo
    'xgb':     '#2980b9',   # blue       — XGBoost
    'ml_bar':  '#bdc3c7',   # light grey — ML bar chart
    'hyb_bar': '#8e44ad',   # purple     — hybrid bar chart
    'dma_bar': '#e74c3c',   # red        — DMA bar chart
    'crisis':  '#f39c12',   # orange     — event shading
}

# Crisis / event periods for shading
EVENTS = [
    ('2008-09-01', '2009-06-01', 'GFC'),
    ('2020-01-01', '2020-12-01', 'COVID-19'),
    ('2022-01-01', '2022-12-01', 'Ukraine'),
]

# Predictor short labels (same order as PREDICTOR_COLS)
PRED_SHORT = [
    'Lag r', 'ΔDemand', 'ΔInv', 'Conv. Yield', 'ΔIP', 'Spread',
    'ΔBDI',  'VIX',     'TED',  'ΔSPX',        'ΔGold', 'ΔWTI',
    'ΔCLP',  'ΔAUD',    'ΔAlcoa', 'ΔRio',      'ΔFCX', 'ΔBHP',
]

# Sub-period definitions for crisis analysis
SUBPERIODS = [
    ('Pre-GFC',       '2008-03-01', '2008-08-01'),
    ('GFC',           '2008-09-01', '2009-06-01'),
    ('Recovery',      '2009-07-01', '2019-12-01'),
    ('COVID-19',      '2020-01-01', '2020-12-01'),
    ('Supercycle',    '2021-01-01', '2021-12-01'),
    ('Ukraine shock', '2022-01-01', '2022-12-01'),
    ('Post-shock',    '2023-01-01', '2026-02-01'),
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig, path: str, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    path = os.path.splitext(path)[0] + '.pdf'   # force PDF regardless of caller
    fig.savefig(path, bbox_inches='tight')      # dpi irrelevant for vector PDF
    plt.close(fig)
    print(f"    Saved: {path}")


def _add_event_shading(ax, dates_oos: pd.DatetimeIndex,
                        alpha: float = 0.12,
                        label: bool = True) -> None:
    """Shade GFC, COVID, and Ukraine shock periods."""
    for start, end, name in EVENTS:
        ts, te = pd.Timestamp(start), pd.Timestamp(end)
        # Only shade if period overlaps with the plot x-range
        if te >= dates_oos[0] and ts <= dates_oos[-1]:
            ax.axvspan(ts, te, alpha=alpha, color=COLOURS['crisis'],
                       zorder=0, label=name if label else None)
            label = False  # only label the first one in legend


def _r2_oos_val(fc: np.ndarray, actual: np.ndarray) -> float:
    rw = np.zeros(len(actual))
    return float(1.0 - np.mean((actual - fc)**2) / np.mean((actual - rw)**2))


def _cw_stat(fc: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Clark-West statistic vs driftless random walk."""
    rw  = np.zeros(len(actual))
    e_rw  = actual - rw
    e_mod = actual - fc
    cw_seq = e_rw**2 - e_mod**2 + (rw - fc)**2
    cw_bar = cw_seq.mean()
    T       = len(cw_seq)
    bw      = max(1, int(T**(1/3)))
    dm      = cw_seq - cw_bar
    var_cw  = np.var(cw_seq, ddof=1)
    for lag in range(1, bw + 1):
        var_cw += 2*(1-lag/(bw+1)) * np.mean(dm[lag:]*dm[:-lag])
    var_cw = max(var_cw, 1e-12)
    stat   = float(cw_bar / np.sqrt(var_cw / T))
    return stat, float(1.0 - norm.cdf(stat))


def _format_table_row(name, msfe, rw_msfe, r2, cw, pval) -> dict:
    return {
        'Model':         name,
        'MSFE':          round(msfe, 4),
        'Rel. MSFE':     round(msfe / rw_msfe, 4),
        'R²_oos (%)':    round(r2 * 100, 4),
        'CW-stat':       round(cw, 4),
        'p-value':       round(pval, 4),
    }


# ---------------------------------------------------------------------------
# TABLE GENERATORS
# ---------------------------------------------------------------------------

def make_table1(df: pd.DataFrame, predictor_cols: list,
                output_dir: str) -> pd.DataFrame:
    """
    Table 1: Summary statistics.
    Matches Table 1 format of Buncic & Moretto (2015).
    """
    cols   = ['r_copper'] + predictor_cols
    labels = {
        'r_copper':       'Copper return r_t (%)',
        'x_r_copper_lag': 'Lagged return r_{t-1} (%)',
        'x_dDemand':      'Δ Demand (mt)',
        'x_dInventory':   'Δ Inventories (mt)',
        'x_ConvYield':    'Convenience yield',
        'x_dIP':          'Δ Industrial production (%)',
        'x_Spread':       'Term spread (10Y–3M)',
        'x_dBDI':         'Δ Baltic Dry Index (%)',
        'x_VIX':          'VIX (level)',
        'x_TED':          'TED spread (reconstructed)',
        'x_dSPX':         'Δ S&P 500 (%)',
        'x_dGold':        'Δ Gold (%)',
        'x_dWTI':         'Δ WTI crude oil (%)',
        'x_dCLP':         'Δ CLP (% appreciation)',
        'x_dAUD':         'Δ AUD (%)',
        'x_dAlcoa':       'Δ Alcoa (%)',
        'x_dRio':         'Δ Rio Tinto (%)',
        'x_dFCX':         'Δ Freeport-McMoRan (%)',
        'x_dBHP':         'Δ BHP Billiton (%)',
    }
    rows = []
    for col in cols:
        s = df[col]
        rows.append({
            'Variable': labels.get(col, col),
            'Mean':     round(s.mean(),   4),
            'Median':   round(s.median(), 4),
            'Std Dev':  round(s.std(),    4),
            'Skewness': round(s.skew(),   4),
            'Kurtosis': round(s.kurt(),   4),
            'Min':      round(s.min(),    4),
            'Max':      round(s.max(),    4),
        })
    t1 = pd.DataFrame(rows).set_index('Variable')
    t1.to_csv(os.path.join(output_dir, 'T1_summary_stats.csv'))
    print("  T1 saved")
    return t1


def make_table2(dma_result, ml_results: dict, hybrid_results: dict,
                output_dir: str) -> pd.DataFrame:
    """
    Table 2: One-step-ahead OOS forecast evaluation — all models.
    """
    y_oos   = dma_result.actual
    rw_msfe = dma_result.msfe['RW']
    rows    = []

    # Random walk
    rows.append({'Model':'Random Walk (RW)', 'MSFE':round(rw_msfe,4),
                 'Rel. MSFE':1.0, 'R²_oos (%)':'—', 'CW-stat':'—', 'p-value':'—'})

    # DMA benchmarks from 02
    for name in ['HA_exp', 'TVP', 'DMS', 'DMA']:
        label = {'HA_exp':'HA (expanding)', 'TVP':'TVP', 'DMS':'DMS', 'DMA':'DMA'}[name]
        msfe  = dma_result.msfe[name]
        r2    = dma_result.r2_oos[name]
        cw    = dma_result.cw_stat[name]
        pval  = dma_result.cw_pval[name]
        rows.append(_format_table_row(label, msfe, rw_msfe, r2, cw, pval))

    # ML models from 03
    ml_order = ['OLS','Ridge','LASSO','ElasticNet','BayesianRidge',
                'RandomForest','XGBoost','MLP']
    for name in ml_order:
        if name not in ml_results or np.isnan(ml_results[name].msfe):
            continue
        res = ml_results[name]
        rows.append(_format_table_row(name, res.msfe, rw_msfe,
                                       res.r2_oos, res.cw_stat, res.cw_pval))

    # Hybrid models from 04
    for name, res in hybrid_results.items():
        rows.append(_format_table_row(name, res.msfe, rw_msfe,
                                       res.r2_oos, res.cw_stat, res.cw_pval))

    t2 = pd.DataFrame(rows).set_index('Model')
    t2.to_csv(os.path.join(output_dir, 'T2_oos_results.csv'))
    print("  T2 saved")
    return t2


def make_table3(dma_mh_results: dict, ml_results: dict,
                output_dir: str) -> pd.DataFrame:
    """
    Table 3: Multi-horizon (h=2,3,6) forecast evaluation.
    Mirrors Table 3 format of Buncic & Moretto (2015).
    """
    rows = []
    for h, dma_h in sorted(dma_mh_results.items()):
        rw_msfe = dma_h.msfe['RW']
        # DMA at this horizon
        rows.append({
            'Horizon': h,
            **_format_table_row(f'DMA (h={h})',
                                 dma_h.msfe['DMA'], rw_msfe,
                                 dma_h.r2_oos['DMA'],
                                 dma_h.cw_stat['DMA'],
                                 dma_h.cw_pval['DMA'])
        })
        # ElasticNet at this horizon (if available)
        key = f'ElasticNet_h{h}'
        if key in ml_results:
            res = ml_results[key]
            rows.append({
                'Horizon': h,
                **_format_table_row(f'ElasticNet (h={h})',
                                     res.msfe, rw_msfe,
                                     res.r2_oos, res.cw_stat, res.cw_pval)
            })

    if rows:
        t3 = pd.DataFrame(rows).set_index(['Horizon', 'Model'])
        t3.to_csv(os.path.join(output_dir, 'T3_multihorizon.csv'))
        print("  T3 saved")
        return t3
    else:
        print("  T3 skipped (no multi-horizon results)")
        return pd.DataFrame()


def make_table4(hybrid_results: dict, dma_result, ml_results: dict,
                output_dir: str) -> pd.DataFrame:
    """
    Table 4: Hybrid model evaluation — focused comparison.
    """
    y_oos   = dma_result.actual
    rw_msfe = dma_result.msfe['RW']
    rows    = []

    rows.append({'Model':'Random Walk (RW)', 'MSFE':round(rw_msfe,4),
                 'Rel. MSFE':1.0, 'R²_oos (%)':'—', 'CW-stat':'—',
                 'p-value':'—', 'Description':'Driftless RW benchmark'})

    rows.append({**_format_table_row('DMA', dma_result.msfe['DMA'], rw_msfe,
                                      dma_result.r2_oos['DMA'],
                                      dma_result.cw_stat['DMA'],
                                      dma_result.cw_pval['DMA']),
                  'Description': 'Dynamic Model Averaging'})

    if 'ElasticNet' in ml_results:
        res = ml_results['ElasticNet']
        rows.append({**_format_table_row('ElasticNet', res.msfe, rw_msfe,
                                          res.r2_oos, res.cw_stat, res.cw_pval),
                      'Description': 'Best standalone ML model'})

    descs = {
        'PIP_ElasticNet': 'DMA PIPs as soft variable weights → ElasticNet',
        'Stacking':       'Ridge meta-learner on DMA + ElasticNet forecasts',
        'Combo_DMA_EN':   'Equal-weight average: 0.5×DMA + 0.5×ElasticNet',
    }
    for name, res in hybrid_results.items():
        rows.append({**_format_table_row(name, res.msfe, rw_msfe,
                                          res.r2_oos, res.cw_stat, res.cw_pval),
                      'Description': descs.get(name, '')})

    t4 = pd.DataFrame(rows).set_index('Model')
    t4.to_csv(os.path.join(output_dir, 'T4_hybrid_results.csv'))
    print("  T4 saved")
    return t4


def make_table5(dma_result, ml_results: dict, hybrid_results: dict,
                output_dir: str) -> pd.DataFrame:
    """
    Table 5: Sub-period crisis analysis.
    Reports OOS R² for each model across GFC, COVID, Ukraine and other periods.
    """
    dates_oos = dma_result.dates
    y_oos     = dma_result.actual

    # Collect all forecasts
    all_fcs = {
        'DMA':         dma_result.dma_forecasts,
        'ElasticNet':  ml_results['ElasticNet'].forecasts if 'ElasticNet' in ml_results else None,
        'Combo_DMA_EN': hybrid_results['Combo_DMA_EN'].forecasts if 'Combo_DMA_EN' in hybrid_results else None,
    }

    rows = []
    for period_name, start, end in SUBPERIODS:
        mask = (dates_oos >= start) & (dates_oos <= end)
        n    = mask.sum()
        if n < 3:
            continue
        y_p = y_oos[mask]
        row = {'Period': period_name, 'N': n}
        for mname, fc in all_fcs.items():
            if fc is None:
                continue
            fc_p     = fc[mask]
            row[mname] = round(_r2_oos_val(fc_p, y_p), 4)
        rows.append(row)

    t5 = pd.DataFrame(rows).set_index('Period')
    t5.to_csv(os.path.join(output_dir, 'T5_subperiod.csv'))
    print("  T5 saved")
    return t5


def make_table6(dma_result, ml_results: dict,
                predictor_cols: list, output_dir: str) -> pd.DataFrame:
    """
    Table 6: Feature importance comparison — DMA PIPs vs tree importances.
    """
    mean_pip = dma_result.pip_path.mean(axis=0)
    rows = []
    for j, col in enumerate(predictor_cols):
        label = col.replace('x_', '')
        row   = {'Predictor': label, 'DMA Mean PIP': round(mean_pip[j], 4)}
        if 'RandomForest' in ml_results and ml_results['RandomForest'].feature_imp is not None:
            row['RF Importance'] = round(ml_results['RandomForest'].feature_imp[j], 4)
        if 'XGBoost' in ml_results and ml_results['XGBoost'].feature_imp is not None:
            row['XGB Importance'] = round(ml_results['XGBoost'].feature_imp[j], 4)
        rows.append(row)

    t6 = pd.DataFrame(rows).set_index('Predictor')
    if 'RF Importance' in t6.columns and 'XGB Importance' in t6.columns:
        t6['Mean Importance'] = t6[['RF Importance','XGB Importance']].mean(axis=1).round(4)
        t6 = t6.sort_values('DMA Mean PIP', ascending=False)
    t6.to_csv(os.path.join(output_dir, 'T6_feature_importance.csv'))
    print("  T6 saved")
    return t6


# ---------------------------------------------------------------------------
# FIGURE GENERATORS
# ---------------------------------------------------------------------------

PREDICTOR_GROUPS = [
    # (suffix, group title, indices, ncols)
    ('a', 'Fundamental Variables',            list(range(0, 7)),   4),
    ('b', 'Financial Variables',              list(range(7, 12)),  3),
    ('c', 'FX & Mining Stock Variables',      list(range(12, 18)), 3),
]


def _grouped_grid(n_panels: int, ncols: int, panel_w=4.0, panel_h=2.2):
    """Create a fig/axes grid sized to the number of panels, hiding unused axes."""
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes_flat, nrows

def plot_f1_copper(df: pd.DataFrame, dates: pd.DatetimeIndex,
                   output_dir: str) -> None:
    """F1: Copper spot price and monthly returns."""
    y = df['r_copper'].values
    # Get spot price from raw levels if available in df
    spot_col = next((c for c in df.columns if 'spot' in c.lower() and 'r_' not in c), None)

    with plt.rc_context(THESIS_STYLE):
        if spot_col:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.plot(dates, df[spot_col], color=COLOURS['actual'], lw=1.2)
            ax1.set_ylabel('Price (USD/tonne)')
            ax1.set_title('(a) CME Copper Spot Price')
            ax1_use = ax2
        else:
            fig, ax1_use = plt.subplots(figsize=(12, 4))

        ax1_use.plot(dates, y, color=COLOURS['actual'], lw=0.8, alpha=0.85)
        ax1_use.axhline(0, color=COLOURS['rw'], lw=0.5, ls='--')
        ax1_use.set_ylabel('Return (%)')
        ax1_use.set_title('(b) Monthly Copper Return' if spot_col else 'Monthly Copper Return')
        ax1_use.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1_use.xaxis.set_major_locator(mdates.YearLocator(4))
        plt.xticks(rotation=0)

        _save_fig(fig, os.path.join(output_dir, 'F1_copper_price_returns.png'))


def plot_f2_predictor_grid(df: pd.DataFrame, dates: pd.DatetimeIndex,
                           predictor_cols: list, output_dir: str) -> None:
    """F2a–c: Predictor time series, one figure per predictor category."""
    with plt.rc_context(THESIS_STYLE):
        for suffix, group_title, idxs, ncols in PREDICTOR_GROUPS:
            fig, axes_flat, nrows = _grouped_grid(len(idxs), ncols)

            for k, j in enumerate(idxs):
                ax = axes_flat[k]
                ax.plot(dates, df[predictor_cols[j]].values,
                        color=COLOURS['actual'], lw=0.7, alpha=0.85)
                ax.axhline(0, color=COLOURS['rw'], lw=0.4, ls='--')
                ax.set_title(PRED_SHORT[j], fontsize=8, pad=3)
                ax.tick_params(labelsize=7)
                # bottom row of *visible* panels gets date labels
                if k >= len(idxs) - ncols:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
                    ax.xaxis.set_major_locator(mdates.YearLocator(8))
                    ax.tick_params(labelbottom=True)


            _save_fig(fig, os.path.join(
                output_dir, f'F2{suffix}_predictors_{suffix}.pdf'))


def plot_f3_actual_predicted(dma_result, ml_results: dict,
                              output_dir: str) -> None:
    """F3: Actual copper returns vs DMA and ElasticNet forecasts."""
    dates_oos = dma_result.dates
    y_oos     = dma_result.actual
    dma_fc    = dma_result.dma_forecasts
    en_fc     = ml_results['ElasticNet'].forecasts if 'ElasticNet' in ml_results else None

    with plt.rc_context(THESIS_STYLE):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(dates_oos, y_oos,  color=COLOURS['actual'], lw=1.0,
                alpha=0.9, label='Actual', zorder=3)
        ax.plot(dates_oos, dma_fc, color=COLOURS['dma'], lw=1.0,
                alpha=0.85, label='DMA', zorder=4)
        if en_fc is not None:
            ax.plot(dates_oos, en_fc, color=COLOURS['en'], lw=0.9,
                    alpha=0.75, label='ElasticNet', ls='--', zorder=2)
        ax.axhline(0, color=COLOURS['rw'], lw=0.5, ls='--')
        _add_event_shading(ax, dates_oos)
        ax.set_ylabel('Return (%)')
        ax.legend(loc='upper right', framealpha=0.8, ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Annotate crisis labels
        for start, _, label in EVENTS:
            ts = pd.Timestamp(start)
            if dates_oos[0] <= ts <= dates_oos[-1]:
                ax.annotate(label, xy=(ts, ax.get_ylim()[1]*0.85),
                            fontsize=7, color='#7f8c8d', ha='left')
        _save_fig(fig, os.path.join(output_dir, 'F3_actual_predicted.png'))


def plot_f4_cumulative_sfe(dma_result, ml_results: dict, hybrid_results: dict,
                            output_dir: str) -> None:
    """F4: Cumulative difference in squared forecast errors vs random walk."""
    dates_oos = dma_result.dates
    y_oos     = dma_result.actual
    rw_fc     = np.zeros(len(y_oos))

    def csd(fc): return np.cumsum((y_oos-rw_fc)**2 - (y_oos-fc)**2)

    with plt.rc_context(THESIS_STYLE):
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(dates_oos, csd(dma_result.dma_forecasts),
                color=COLOURS['dma'], lw=1.5, label='DMA', zorder=4)

        if 'ElasticNet' in ml_results:
            ax.plot(dates_oos, csd(ml_results['ElasticNet'].forecasts),
                    color=COLOURS['en'], lw=1.2, ls='--', label='ElasticNet', zorder=3)

        if 'Combo_DMA_EN' in hybrid_results:
            ax.plot(dates_oos, csd(hybrid_results['Combo_DMA_EN'].forecasts),
                    color=COLOURS['combo'], lw=1.0, ls=':', label='Combo DMA+EN', zorder=2)

        ax.axhline(0, color=COLOURS['rw'], lw=0.8)
        _add_event_shading(ax, dates_oos)
        ax.set_ylabel('Cumulative ΔSFE vs Random Walk')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        _save_fig(fig, os.path.join(output_dir, 'F4_cumulative_sfe.png'))



def plot_f12_rolling_hit_rate(dma_result, ml_results: dict, hybrid_results: dict,
                              output_dir: str, window: int = 36) -> pd.DataFrame:
    """
    F12: directional accuracy over time.
      Top panel   -- rolling `window`-month hit rate for DMA, ElasticNet and
                     the equal-weight combo, against the 50% line and the
                     rolling unconditional frequency of positive returns
                     (the hit rate a constant 'always up' call would earn).
      Bottom panel-- cumulative net correct calls (correct minus incorrect),
                     the sign-only analogue of Figure 4.
    Exact-zero forecasts count as incorrect, as in Table 4. Also writes
    T19_rolling_hitrate.csv with the plotted series and sub-period hit rates.
    """
    y_oos     = dma_result.actual
    dates_oos = dma_result.dates
    series = {'DMA': dma_result.dma_forecasts}
    if 'ElasticNet' in ml_results:
        series['ElasticNet'] = ml_results['ElasticNet'].forecasts
    if 'Combo_DMA_EN' in hybrid_results:
        series['Combo DMA+EN'] = hybrid_results['Combo_DMA_EN'].forecasts
    styles = {'DMA': (COLOURS['dma'], '-', 1.5), 'ElasticNet': (COLOURS['en'], '--', 1.2),
              'Combo DMA+EN': (COLOURS['combo'], ':', 1.2)}

    sign_y = np.sign(y_oos)
    hits = {k: ((np.sign(v) == sign_y) & (v != 0)).astype(float) for k, v in series.items()}
    up   = (y_oos > 0).astype(float)
    roll = pd.DataFrame({**{k: pd.Series(v).rolling(window).mean().values for k, v in hits.items()},
                         'Always up': pd.Series(up).rolling(window).mean().values}, index=dates_oos)
    cum  = pd.DataFrame({k: np.cumsum(2*v - 1) for k, v in hits.items()}, index=dates_oos)

    with plt.rc_context(THESIS_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [1.3, 1]})
        for k in series:
            c, ls, lw = styles[k]
            ax1.plot(dates_oos, 100*roll[k], color=c, ls=ls, lw=lw, label=k, zorder=3)
        ax1.plot(dates_oos, 100*roll['Always up'], color=COLOURS['actual'], lw=0.9, ls='-.',
                 label='Unconditional P(return > 0)', zorder=2)
        ax1.axhline(50, color=COLOURS['rw'], lw=0.8)
        _add_event_shading(ax1, dates_oos)
        ax1.set_ylabel(f'{window}-month rolling hit rate (%)')
        ax1.set_ylim(30, 100)
        ax1.legend(loc='lower left', framealpha=0.8, ncol=2)

        for k in series:
            c, ls, lw = styles[k]
            ax2.plot(dates_oos, cum[k], color=c, ls=ls, lw=lw, label=k, zorder=3)
        ax2.axhline(0, color=COLOURS['rw'], lw=0.8)
        _add_event_shading(ax2, dates_oos, label=False)
        ax2.set_ylabel('Cumulative correct minus incorrect calls')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        _save_fig(fig, os.path.join(output_dir, 'F12_rolling_hit_rate.png'))

    # Sub-period hit rates (same windows as Table 7)
    periods = [('Pre-GFC', '2008-03-01', '2008-08-31'), ('GFC', '2008-09-01', '2009-06-30'),
               ('Recovery', '2009-07-01', '2019-12-31'), ('COVID-19', '2020-01-01', '2020-12-31'),
               ('Supercycle', '2021-01-01', '2021-12-31'), ('Ukraine shock', '2022-01-01', '2022-12-31'),
               ('Post-shock', '2023-01-01', '2026-02-28'), ('Full OOS', '2008-03-01', '2026-02-28')]
    rows = []
    for name, a, b in periods:
        m = (dates_oos >= a) & (dates_oos <= b)
        rows.append({'Period': name, 'N': int(m.sum()),
                     **{k: round(100*v[m].mean(), 1) for k, v in hits.items()},
                     'Always up': round(100*up[m].mean(), 1)})
    sub = pd.DataFrame(rows)
    roll.assign(**{f'cum_{k}': cum[k] for k in cum}).to_csv(
        os.path.join(output_dir, 'T19_rolling_hitrate.csv'))
    sub.to_csv(os.path.join(output_dir, 'T19_subperiod_hitrate.csv'), index=False)
    return sub


def plot_f5_scatter_dma(dma_result, output_dir: str) -> None:
    """F5: Scatter plot of DMA predicted vs actual returns."""
    dma_fc = dma_result.dma_forecasts
    y_oos  = dma_result.actual

    with plt.rc_context(THESIS_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(dma_fc, y_oos, alpha=0.45, s=20,
                   color=COLOURS['dma'], label='Observations', zorder=2)
        # OLS fit line
        coef   = np.polyfit(dma_fc, y_oos, 1)
        x_line = np.linspace(dma_fc.min(), dma_fc.max(), 200)
        ax.plot(x_line, np.polyval(coef, x_line),
                color=COLOURS['actual'], lw=1.5, label=f'OLS fit (slope={coef[0]:.2f})')
        ax.axhline(0, color=COLOURS['rw'], lw=0.4, ls='--')
        ax.axvline(0, color=COLOURS['rw'], lw=0.4, ls='--')
        ax.set_xlabel('DMA Forecast (%)')
        ax.set_ylabel('Actual Return (%)')
        ax.legend(framealpha=0.8)
        _save_fig(fig, os.path.join(output_dir, 'F5_scatter_dma.png'))


def plot_f6_pip_grid(dma_result, output_dir: str) -> None:
    """F6a–c: Posterior inclusion probabilities, one figure per category."""
    dates_oos = dma_result.dates
    pip       = dma_result.pip_path  # (T_oos, 18)

    with plt.rc_context(THESIS_STYLE):
        for suffix, group_title, idxs, ncols in PREDICTOR_GROUPS:
            fig, axes_flat, nrows = _grouped_grid(len(idxs), ncols,
                                                  panel_h=2.0)

            for k, j in enumerate(idxs):
                ax = axes_flat[k]
                ax.plot(dates_oos, pip[:, j], color=COLOURS['dma'], lw=0.9)
                ax.axhline(0.5, color=COLOURS['rw'], lw=0.5, ls='--', alpha=0.7)
                _add_event_shading(ax, dates_oos, alpha=0.10, label=False)
                ax.set_title(PRED_SHORT[j], fontsize=8, pad=3)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=7)
                if k >= len(idxs) - ncols:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
                    ax.xaxis.set_major_locator(mdates.YearLocator(6))
                    ax.tick_params(labelbottom=True)


            _save_fig(fig, os.path.join(
                output_dir, f'F6{suffix}_pip_{suffix}.pdf'))


def plot_f7_beta_path(dma_result, output_dir: str) -> None:
    """F7a–c: DMA time-varying coefficient estimates, one figure per category."""
    dates_oos = dma_result.dates
    beta      = dma_result.beta_dma_path  # (T_oos, 19); col 0 = intercept

    with plt.rc_context(THESIS_STYLE):
        for suffix, group_title, idxs, ncols in PREDICTOR_GROUPS:
            fig, axes_flat, nrows = _grouped_grid(len(idxs), ncols,
                                                  panel_h=2.0)

            for k, j in enumerate(idxs):
                ax = axes_flat[k]
                ax.plot(dates_oos, beta[:, j + 1],   # +1 skips intercept
                        color=COLOURS['en'], lw=0.9)
                ax.axhline(0, color=COLOURS['rw'], lw=0.5, ls='--', alpha=0.7)
                _add_event_shading(ax, dates_oos, alpha=0.10, label=False)
                ax.set_title(PRED_SHORT[j], fontsize=8, pad=3)
                ax.tick_params(labelsize=7)
                if k >= len(idxs) - ncols:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
                    ax.xaxis.set_major_locator(mdates.YearLocator(6))
                    ax.tick_params(labelbottom=True)


            _save_fig(fig, os.path.join(
                output_dir, f'F7{suffix}_beta_path_{suffix}.pdf'))


def plot_f8_r2_barchart(dma_result, ml_results: dict,
                         hybrid_results: dict, output_dir: str) -> None:
    """F8: OOS R² bar chart for all models."""
    rw_msfe = dma_result.msfe['RW']

    # Collect model names, R² values, and bar colours
    entries = []

    # ML baselines
    for name in ['OLS', 'Ridge', 'LASSO', 'ElasticNet',
                 'BayesianRidge', 'RandomForest', 'XGBoost', 'MLP']:
        if name in ml_results and not np.isnan(ml_results[name].r2_oos):
            entries.append((name, ml_results[name].r2_oos, COLOURS['ml_bar']))

    # Hybrid models
    for name in ['PIP_ElasticNet', 'Stacking', 'Combo_DMA_EN']:
        if name in hybrid_results:
            label = {'PIP_ElasticNet':'PIP-EN','Stacking':'Stacking',
                     'Combo_DMA_EN':'Combo'}[name]
            entries.append((label, hybrid_results[name].r2_oos, COLOURS['hyb_bar']))

    # DMA
    entries.append(('DMA', dma_result.r2_oos['DMA'], COLOURS['dma_bar']))

    names  = [e[0] for e in entries]
    r2vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    with plt.rc_context(THESIS_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        # Draw bars with their true sign so negative R² extends below zero
        ax.bar(names, r2vals,
               color=colors, edgecolor='white', linewidth=0.5, zorder=2)

        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Out-of-Sample R²')
        lo = min(min(r2vals), 0.0) - 0.05
        hi = max(max(r2vals), 0.0) + 0.08
        ax.set_ylim(lo, hi)

        # Value labels: above the bar for non-negative R², below it otherwise
        for i, (name, v) in enumerate(zip(names, r2vals)):
            if v >= 0:
                ypos, va = v + 0.01, 'bottom'
            else:
                ypos, va = v - 0.01, 'top'
            ax.text(i, ypos, f'{v:.2f}', ha='center', va=va,
                    fontsize=8, fontweight='bold' if name == 'DMA' else 'normal')

        # Legend
        legend_elems = [
            mpatches.Patch(facecolor=COLOURS['ml_bar'],  label='ML models'),
            mpatches.Patch(facecolor=COLOURS['hyb_bar'], label='Hybrid models'),
            mpatches.Patch(facecolor=COLOURS['dma_bar'], label='DMA'),
        ]
        ax.legend(handles=legend_elems, loc='upper left', framealpha=0.8)
        plt.xticks(rotation=30, ha='right')
        _save_fig(fig, os.path.join(output_dir, 'F8_r2_barchart.png'))


def plot_f9_crisis_zoom(dma_result, ml_results: dict,
                         output_dir: str) -> None:
    """
    F9: Crisis period zoom — three side-by-side panels for GFC, COVID, Ukraine.
    Shows actual returns, DMA forecasts, and ElasticNet forecasts in each window.
    """
    dates_oos = dma_result.dates
    y_oos     = dma_result.actual
    dma_fc    = dma_result.dma_forecasts
    en_fc     = ml_results['ElasticNet'].forecasts if 'ElasticNet' in ml_results else None

    crisis_windows = [
        ('Global Financial Crisis\n(Sep 2008 – Jun 2009)',
         '2008-06-01', '2009-12-01', '2008-09-01', '2009-06-30'),
        ('COVID-19\n(Jan 2020 – Dec 2020)',
         '2019-10-01', '2021-06-01', '2020-01-01', '2020-12-31'),
        ('Ukraine War Shock\n(Jan 2022 – Dec 2022)',
         '2021-10-01', '2023-06-01', '2022-01-01', '2022-12-31'),
    ]

    with plt.rc_context(THESIS_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (title, start, end, c_start, c_end) in zip(axes, crisis_windows):
            mask = (dates_oos >= start) & (dates_oos <= end)  # plot
            c_mask = (dates_oos >= c_start) & (dates_oos <= c_end)  # score
            d    = dates_oos[mask]
            ya   = y_oos[mask]
            yd   = dma_fc[mask]

            ax.plot(d, ya, color=COLOURS['actual'], lw=1.2,
                    label='Actual', zorder=3)
            ax.plot(d, yd, color=COLOURS['dma'], lw=1.2,
                    label='DMA', zorder=4)
            if en_fc is not None:
                ax.plot(d, en_fc[mask], color=COLOURS['en'], lw=1.0,
                        ls='--', label='ElasticNet', zorder=2)

            ax.axhline(0, color=COLOURS['rw'], lw=0.5, ls='--')
            ax.set_title(title, fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

            # R² annotation
            if c_mask.sum() > 2:
                r2_d = _r2_oos_val(dma_fc[c_mask], y_oos[c_mask])
                r2_e = _r2_oos_val(en_fc[c_mask], y_oos[c_mask]) if en_fc is not None else None
                txt  = f'DMA R²={r2_d:.2f}'
                if r2_e is not None:
                    txt += f'\nEN R²={r2_e:.2f}'
                ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                        fontsize=8, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        axes[0].legend(loc='lower left', fontsize=8, framealpha=0.8)

        _save_fig(fig, os.path.join(output_dir, 'F9_crisis_zoom.png'))


def plot_f10_lasso_coef(ml_results: dict, dma_result,
                         predictor_cols: list, output_dir: str) -> None:
    """
    F10: LASSO coefficient path over OOS period.
    Compares with DMA PIP path to show convergence of predictor selection.
    """
    if 'LASSO' not in ml_results or ml_results['LASSO'].coef_path is None:
        print("  F10 skipped: LASSO coef_path not available")
        return

    dates_oos  = dma_result.dates
    coef_path  = ml_results['LASSO'].coef_path[:, 1:]  # drop intercept: (T_oos, 18)
    pip_path   = dma_result.pip_path  # (T_oos, 18)

    # Select top 6 predictors by mean absolute LASSO coefficient
    mean_abs   = np.abs(coef_path).mean(axis=0)
    top6_idx   = np.argsort(mean_abs)[::-1][:6]

    with plt.rc_context(THESIS_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        axes_flat = axes.flatten()

        for plot_i, j in enumerate(top6_idx):
            ax = axes_flat[plot_i]
            ax2 = ax.twinx()

            ax.plot(dates_oos, coef_path[:, j],
                    color=COLOURS['en'], lw=1.1, label='LASSO coef')
            ax2.plot(dates_oos, pip_path[:, j],
                     color=COLOURS['dma'], lw=0.9, ls='--', alpha=0.8,
                     label='DMA PIP')
            ax.axhline(0, color=COLOURS['rw'], lw=0.4, ls='--')
            _add_event_shading(ax, dates_oos, alpha=0.08, label=False)

            ax.set_title(PRED_SHORT[j], fontsize=10)
            ax.set_ylabel('LASSO coef.', fontsize=8, color=COLOURS['en'])
            ax2.set_ylabel('DMA PIP', fontsize=8, color=COLOURS['dma'])
            ax2.set_ylim(0, 1)
            ax.tick_params(labelsize=8)
            ax2.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Shared legend
        handles = [
            plt.Line2D([0],[0], color=COLOURS['en'],  lw=1.2, label='LASSO coefficient'),
            plt.Line2D([0],[0], color=COLOURS['dma'], lw=1.0, ls='--', label='DMA PIP'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2,
                   fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

        _save_fig(fig, os.path.join(output_dir, 'F10_lasso_coef_path.png'))


# ---------------------------------------------------------------------------
# Master evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(df: pd.DataFrame,
                         predictor_cols: list,
                         dma_result,
                         ml_results: dict,
                         hybrid_results: dict,
                         n_insample: int,
                         output_dir: str = 'output/',
                         dma_mh_results: dict = None) -> dict:
    """
    Run the complete evaluation pipeline: all tables and all figures.

    Parameters
    ----------
    df              : pd.DataFrame    clean monthly panel from pipeline
    predictor_cols  : list            18 predictor column names
    dma_result      : DMAResult       from run_dma_dms()
    ml_results      : dict            from run_all_ml_models()
    hybrid_results  : dict            from run_all_hybrids()
    n_insample      : int             burn-in observations
    output_dir      : str             directory for all output files
    dma_mh_results  : dict            optional multi-horizon DMA results

    Returns
    -------
    dict  with keys 'tables' and 'figures' mapping to saved file paths
    """
    _ensure_dir(output_dir)
    dates = pd.DatetimeIndex(df['date'])
    saved = {'tables': [], 'figures': []}

    print("\n" + "="*55)
    print("05_evaluation.py  —  generating all outputs")
    print("="*55)

    # ── Tables ──────────────────────────────────────────────────────────────
    print("\nTables:")
    t1 = make_table1(df, predictor_cols, output_dir)
    t2 = make_table2(dma_result, ml_results, hybrid_results, output_dir)
    t3 = make_table3(dma_mh_results or {}, ml_results, output_dir)
    t4 = make_table4(hybrid_results, dma_result, ml_results, output_dir)
    t5 = make_table5(dma_result, ml_results, hybrid_results, output_dir)
    t6 = make_table6(dma_result, ml_results, predictor_cols, output_dir)

    print("\n  ── Table 2 (Main OOS Results) ─────────────────────────────")
    print(t2.to_string())
    print("\n  ── Table 5 (Sub-period Analysis) ──────────────────────────")
    print(t5.to_string())

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nFigures:")
    plot_f1_copper(df, dates, output_dir)
    plot_f2_predictor_grid(df, dates, predictor_cols, output_dir)
    plot_f3_actual_predicted(dma_result, ml_results, output_dir)
    plot_f4_cumulative_sfe(dma_result, ml_results, hybrid_results, output_dir)
    plot_f5_scatter_dma(dma_result, output_dir)
    plot_f12_rolling_hit_rate(dma_result, ml_results, hybrid_results, output_dir)
    plot_f6_pip_grid(dma_result, output_dir)
    plot_f7_beta_path(dma_result, output_dir)
    plot_f8_r2_barchart(dma_result, ml_results, hybrid_results, output_dir)
    plot_f9_crisis_zoom(dma_result, ml_results, output_dir)
    plot_f10_lasso_coef(ml_results, dma_result, predictor_cols, output_dir)

    print("\nAll outputs saved to:", output_dir)
    return saved


# ---------------------------------------------------------------------------
# Quick validation — run as script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline    import build_dataset, PREDICTOR_COLS
    from src.dma_dms     import run_dma_dms
    from src.ml_models   import run_all_ml_models
    from src.hybrid      import run_all_hybrids

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/thesis/output/'

    print("="*60)
    print("05_evaluation.py  —  full evaluation pipeline")
    print("="*60)

    # Build dataset
    df, meta = build_dataset(data_dir=data_dir)
    y        = df['r_copper'].values
    X        = df[PREDICTOR_COLS].values
    dates    = pd.DatetimeIndex(df['date'])
    n_is     = 120

    # Run all models
    print("\nRunning DMA/DMS...")
    dma_result = run_dma_dms(y=y, X=X, n_insample=n_is, dates=dates, verbose=True)

    print("\nRunning ML models...")
    ml_results = run_all_ml_models(y=y, X=X, n_insample=n_is,
                                    dates=dates, verbose=True)

    print("\nRunning hybrid models...")
    hybrid_results = run_all_hybrids(y=y, X=X, dma_result=dma_result,
                                      ml_results=ml_results, n_insample=n_is,
                                      dates=dates, verbose=True)

    # Generate all outputs
    run_full_evaluation(
        df=df, predictor_cols=PREDICTOR_COLS,
        dma_result=dma_result, ml_results=ml_results,
        hybrid_results=hybrid_results, n_insample=n_is,
        output_dir=output_dir,
    )