#!/usr/bin/env python3
"""
plot_spectrum.py
----------------
Two-panel figure:
  Top    : Sky-averaged brightness temperature vs frequency (historical data)
  Bottom : Residual after smooth foreground subtraction, with 21cm signal models

Usage:
    python plot_spectrum.py [--data-dir DIR] [--output FILE] [--unit K|mK]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit("Need tomllib (Python >=3.11) or tomli (pip install tomli)")

# ── Physical constants ────────────────────────────────────────────────────────
C_MS  = 2.998e8
K_B   = 1.381e-23
NU21  = 1420.405752e6   # Hz, 21cm rest frequency

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path, comment='#')
    df.columns = df.columns.str.strip()
    return df

def load_toml(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)

def apply_filter(df, filter_list):
    if not filter_list:
        return df.copy()
    masks = []
    for clause in filter_list:
        mask = pd.Series(True, index=df.index)
        for col, val in clause.items():
            if col not in df.columns:
                raise KeyError(f"Filter column '{col}' not in {list(df.columns)}")
            mask &= (df[col] == val)
        masks.append(mask)
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return df[combined].copy()

def brightness_to_Tb_K(brightness_Wm2HzSr, freq_MHz):
    """Rayleigh-Jeans brightness -> brightness temperature in K."""
    nu = freq_MHz * 1e6
    return brightness_Wm2HzSr * C_MS**2 / (2.0 * K_B * nu**2)

def z_to_freq_MHz(z):
    """Redshift -> observed frequency in MHz for 21cm line."""
    return NU21 / (1.0 + z) / 1e6

def get_xy(df, ds, unit):
    """
    Return (x_MHz, y) for a dataset config dict.
    y is in K or mK depending on --unit argument.
    If ds['y_unit'] == 'K', data are already in Kelvin (skip R-J conversion).
    Otherwise treats y column as W m^-2 Hz^-1 sr^-1 and converts via R-J.
    """
    x = df[ds['x_col']].values
    raw_y = df[ds['y_col']].values

    yerr_col  = ds.get('yerr_col', '')
    yerr_type = ds.get('yerr_type', 'absolute')
    y_unit_in = ds.get('y_unit', 'W_m2_Hz_sr')   # default: brightness units

    if y_unit_in == 'K':
        y_K = raw_y.astype(float)
    else:
        y_K = brightness_to_Tb_K(raw_y, x)

    y = y_K * 1e3 if unit == 'mK' else y_K

    if yerr_col and yerr_col in df.columns:
        raw_err = df[yerr_col].values
        if yerr_type == 'percent':
            yerr_K = y_K * raw_err / 100.0
        elif y_unit_in == 'K':
            yerr_K = raw_err.astype(float)
        else:
            yerr_K = brightness_to_Tb_K(raw_err, x)
        yerr = yerr_K * 1e3 if unit == 'mK' else yerr_K
    else:
        yerr = None

    return x, y, yerr

_OWN_STYLE_KEYS = {'enabled', 'legend_group', 'resid_only', 'top_only'}

def style_to_mpl(style):
    return {k: v for k, v in style.items() if k not in _OWN_STYLE_KEYS}

# ── Foreground model ──────────────────────────────────────────────────────────

def loglog_poly(log_nu, *coeffs):
    """Polynomial in log-log space: sum_i c_i * (log_nu)^i."""
    log_nu = np.asarray(log_nu)
    return sum(c * log_nu**i for i, c in enumerate(coeffs))

def fit_foreground(x_all, y_all, degree=5, freq_min=1.0, freq_max=300.0):
    """
    Fit a degree-N log-log polynomial to all data in [freq_min, freq_max].
    x_all, y_all: arrays of frequency (MHz) and brightness temp (K or mK).
    Returns a callable f(freq_MHz) -> model_value.
    """
    mask = (x_all >= freq_min) & (x_all <= freq_max) & np.isfinite(y_all) & (y_all > 0)
    x_fit = x_all[mask]
    y_fit = y_all[mask]

    if len(x_fit) < degree + 1:
        raise ValueError(f"Only {len(x_fit)} valid points for degree-{degree} fit")

    log_x = np.log(x_fit)
    log_y = np.log(y_fit)

    # Weighted least squares in log-log space
    # Weight by 1/log_y to downweight very bright low-freq points
    coeffs = np.polyfit(log_x, log_y, degree)[::-1]  # ascending order

    def model(freq_MHz):
        return np.exp(loglog_poly(np.log(np.asarray(freq_MHz)), *coeffs))

    return model, coeffs


# ── Mozdzen 2019 sky model (eq. 5, 3-param + ionosphere) ─────────────────
# Parameters from Table 3, 3-param fit with tau=0.005 ionospheric correction
# Four LST-averaged values span the range of sky brightness across 24h LST.
# We plot the full LST envelope as a shaded band on both panels.

_MOZDZEN_PARAMS = [
    # (T75_K, beta, gamma, LST_label)
    (1816., -2.603, -0.042, 'LST 0h'),
    (1682., -2.595, -0.034, 'LST 6h'),
    (2580., -2.578, -0.086, 'LST 12h'),
    (4776., -2.499, -0.076, 'LST 18h'),
]
_MOZDZEN_TAU  = 0.005    # fixed night-time ionospheric absorption
_MOZDZEN_NU0  = 75.0     # MHz reference frequency
_MOZDZEN_TCMB = 2.725    # K

def mozdzen_T(nu_MHz, T75, beta, gamma, tau=_MOZDZEN_TAU):
    """Mozdzen+ 2019 eq. 5: 3-param power law with ionospheric absorption."""
    x = np.asarray(nu_MHz) / _MOZDZEN_NU0
    return T75 * x**(beta + gamma*np.log(x)) * (1.0 - tau * x**(-2)) + _MOZDZEN_TCMB

def mozdzen_envelope(nu_MHz):
    """Return (T_min, T_max) envelope across all LST values."""
    curves = np.array([mozdzen_T(nu_MHz, T75, b, g)
                       for T75, b, g, _ in _MOZDZEN_PARAMS])
    return curves.min(axis=0), curves.max(axis=0)

# ── Main plot ─────────────────────────────────────────────────────────────────

def make_plot(data_dir, output_path, unit='K'):

    datasets_cfg = load_toml(data_dir / 'datasets.toml')['datasets']
    styles_toml  = load_toml(data_dir / 'styles.toml')
    styles_cfg   = styles_toml['styles']
    rp           = styles_toml.get('residual_panel', {})
    resid_fmin   = rp.get('freq_min',         1.0)
    resid_fmax   = rp.get('freq_max',       300.0)
    resid_ylim   = rp.get('ylim_pos',     50000.0)
    resid_linth  = rp.get('symlog_linthresh', 1.0)
    signals_cfg  = load_toml(data_dir / 'signals.toml')['signals']

    csv_cache = {}

    unit_label = 'K' if unit == 'K' else 'mK'
    unit_scale = 1.0 if unit == 'K' else 1e3

    # ── Collect all data points for fitting ───────────────────────────────────
    fit_keys   = {'rae2', 'imp6', 'ground_dipole'}   # datasets used for fg fit
    fit_x_all, fit_y_all = [], []

    # Also collect everything for the top panel
    panel_data = {}   # key -> (x, y, yerr, style, label)

    for key, ds in datasets_cfg.items():
        style = styles_cfg.get(key, {})
        if not style.get('enabled', True):
            continue

        fname = ds['file']
        if fname not in csv_cache:
            csv_cache[fname] = load_csv(data_dir / fname)
        df_raw = csv_cache[fname]
        df = apply_filter(df_raw, ds.get('filter', []))
        if df.empty:
            print(f"  [{key}] WARNING: 0 rows after filter")
            continue

        x, y, yerr = get_xy(df, ds, unit)
        panel_data[key] = (x, y, yerr, style, ds.get('label', key))

        if key in fit_keys:
            fit_x_all.append(x)
            fit_y_all.append(y)

        print(f"  [{key}] {len(x)} pts, "
              f"{x.min():.2f}–{x.max():.2f} MHz, "
              f"T_b {y.min():.1f}–{y.max():.1f} {unit_label}")

    # ── Fit foreground model ──────────────────────────────────────────────────
    fx = np.concatenate(fit_x_all)
    fy = np.concatenate(fit_y_all)
    fg_model, fg_coeffs = fit_foreground(fx, fy, degree=5,
                                          freq_min=resid_fmin, freq_max=200.0)
    print(f"\n  Foreground fit: degree-5 log-log polynomial")
    print(f"  Fit range: {resid_fmin}–200 MHz using: {fit_keys}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                            hspace=0.0, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    freq_plot = np.geomspace(0.1, 300, 500)

    # ── TOP PANEL: data + foreground model ────────────────────────────────────
    group_order  = ['space', 'ground']
    group_labels = {'space': 'Space-based', 'ground': 'Ground-based (Cane 1979)'}

    for key, (x, y, yerr, style, label) in panel_data.items():
        if style.get('resid_only', False):
            continue          # residual-only datasets skip the top panel
        mpl_style = style_to_mpl(style)
        ax_top.errorbar(x, y, yerr=yerr, label=label, **mpl_style)

    # ── Mozdzen 2019: single cold-sky pointing (LST=6h) on top panel ────────
    moz_nu  = np.linspace(50, 100, 300)
    _T75_6h, _b6h, _g6h, _ = _MOZDZEN_PARAMS[1]   # LST=6h
    moz_lst6 = mozdzen_T(moz_nu, _T75_6h, _b6h, _g6h)
    unit_s = 1e3 if unit == 'mK' else 1.0
    ax_top.plot(moz_nu, moz_lst6 * unit_s,
                lw=1.4, color='#cc0077', alpha=0.85, zorder=5,
                label='EDGES low-band (Mozdzen et al. 2019)')

    # Foreground model line on top panel — only within fit range
    freq_fg_plot = freq_plot[freq_plot >= resid_fmin]
    fg_line = fg_model(freq_fg_plot)
    ax_top.plot(freq_fg_plot, fg_line, color='gray', lw=1.2,
                ls='-', alpha=0.7, label='Foreground model (deg-5 poly)',
                zorder=2)

    # Ionospheric cutoff shading
    ax_top.axvspan(0.1, 10, alpha=0.04, color='purple', zorder=0)
    ax_top.axvline(10, color='purple', lw=1, ls='--', alpha=0.4, zorder=1)
    ax_top.annotate('Ionospheric cutoff', xy=(10, 1),
                    xycoords=('data','axes fraction'),
                    xytext=(10.5, 0.08), textcoords=('data','axes fraction'),
                    fontsize=7.5, color='purple', alpha=0.8,
                    #arrowprops=dict(arrowstyle='->', color='purple',
                    #               alpha=0.6, lw=0.8))
                    )
    # ionospheric distortion significant
    ax_top.axvspan(10, 40, alpha=0.04, color='blue', zorder=0)
    ax_top.axvline(40, color='blue', lw=1, ls='--', alpha=0.4, zorder=1)
    ax_top.annotate('Ionospheric distortion', xy=(40, 1),
                    xycoords=('data','axes fraction'),
                    xytext=(40.5, 0.08), textcoords=('data','axes fraction'),
                    fontsize=7.5, color='blue', alpha=0.8)


    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(f'Brightness temperature ({unit_label})', fontsize=11)
    ax_top.set_xlim(0.1, 300)

    # ── Top panel legend ──────────────────────────────────────────────────────
    # Build entirely from config so we control exactly what appears and in what
    # order. Rules:
    #   - resid_only datasets: residual panel only, never top panel
    #   - top_only datasets:   top panel only (but Bowman Tsky is disabled, so
    #                          nothing to show — skip)
    #   - normal datasets:     top panel only
    # Mozdzen model line is added explicitly as it comes from code, not a dataset.
    # Foreground model line is added last.

    # Harvest the matplotlib handles that were actually plotted on ax_top
    mpl_handles, mpl_labels = ax_top.get_legend_handles_labels()
    handle_map = dict(zip(mpl_labels, mpl_handles))

    leg_h, leg_l = [], []
    for grp in group_order:
        grp_entries = []
        for key, ds in datasets_cfg.items():
            style = styles_cfg.get(key, {})
            if not style.get('enabled', True):
                continue
            if style.get('resid_only', False):
                continue        # never in top panel legend
            if style.get('top_only', False):
                continue        # disabled anyway; skip
            if style.get('legend_group', 'other') != grp:
                continue
            lbl = ds.get('label', key)
            if lbl in handle_map:
                grp_entries.append((lbl, handle_map[lbl]))

        # Add Mozdzen model line under 'ground' group
        if grp == 'ground':
            moz_h = plt.Line2D([0], [0], color='#cc0077', lw=1.4, alpha=0.85)
            grp_entries.append(('EDGES low-band (Mozdzen+ 2019)', moz_h))

        if grp_entries:
            leg_h.append(plt.Line2D([], [], linestyle='none'))
            leg_l.append(f'— {group_labels[grp]} —')
            for lbl, h in grp_entries:
                leg_h.append(h)
                leg_l.append(lbl)

    # Foreground model line at the end
    leg_h.append(plt.Line2D([0], [0], color='gray', lw=1.2, ls='-', alpha=0.7))
    leg_l.append('Foreground model (5th-order log-log poly)')

    ax_top.legend(leg_h, leg_l, loc='lower left', fontsize=7,
                  framealpha=0.88, ncol=1, handlelength=1.5)
    ax_top.grid(True, which='major', ls='-',  alpha=0.2)
    ax_top.grid(True, which='minor', ls='--', alpha=0.1)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ── BOTTOM PANEL: residual ────────────────────────────────────────────────
    # Compute residuals for each dataset
    resid_keys = set(datasets_cfg.keys())   # residual for all datasets
    resid_plotted = False

    for key, (x, y, yerr, style, label) in panel_data.items():
        if style.get('top_only', False):
            continue          # top-panel-only datasets skip the residual panel
        ds_cfg = datasets_cfg.get(key, {})
        resid_ready = ds_cfg.get('residual_ready', False)

        # Clip to fit range — no residual outside where the model is valid
        # (residual_ready datasets are already a residual, skip fg subtraction
        #  but still clip to the same x range for visual consistency)
        clip = (x >= resid_fmin) & (x <= resid_fmax)
        if not clip.any():
            continue
        xc    = x[clip]
        yc    = y[clip]
        yerrc = yerr[clip] if yerr is not None else None

        if resid_ready:
            resid = yc          # already a residual; use directly
        else:
            resid = yc - fg_model(xc)

        mpl_style   = style_to_mpl(style)
        # resid_only datasets keep their own markersize/alpha as specified
        if not style.get('resid_only', False):
            mpl_style = {**mpl_style,
                         'markersize': mpl_style.get('markersize', 4) * 0.7,
                         'alpha':      mpl_style.get('alpha', 0.7) * 0.85}
        lab = label if style.get('resid_only', False) else None
        ax_bot.errorbar(xc, resid, yerr=yerrc, label=lab, **mpl_style)

    # Data scatter band: RMS of residuals across all datasets per freq bin
    # Exclude top_only (e.g. Bowman Tsky) and resid_only (pre-computed residuals)
    # so the band reflects only measurements that went through our fg subtraction
    scatter_keys = {k for k, (x,y,yerr,style,label) in panel_data.items()
                    if not style.get('top_only', False)
                    and not style.get('resid_only', False)}
    all_x_raw = np.concatenate([panel_data[k][0] for k in scatter_keys])
    all_y_raw = np.concatenate([panel_data[k][1] for k in scatter_keys])
    clip_all  = (all_x_raw >= resid_fmin) & (all_x_raw <= resid_fmax)
    all_x = all_x_raw[clip_all]
    all_r = (all_y_raw - fg_model(all_x_raw))[clip_all]
    # Compute RMS in log-spaced bins
    bins = np.geomspace(resid_fmin, resid_fmax, 30)
    bin_rms = []
    bin_cen = []
    for i in range(len(bins)-1):
        mask = (all_x >= bins[i]) & (all_x < bins[i+1])
        if mask.sum() >= 2:
            bin_rms.append(np.std(all_r[mask]))
            bin_cen.append(np.sqrt(bins[i]*bins[i+1]))
    bin_rms = np.array(bin_rms)
    bin_cen = np.array(bin_cen)

    # Shaded current data scatter
    #ax_bot.fill_between(bin_cen, -bin_rms, bin_rms,
    #                     alpha=0.15, color='gray',
    #                     label='Current data scatter (±1σ RMS)')

    # ── 21cm signal models ────────────────────────────────────────────────────
    sig_cache = {}
    for sig_key, sig in signals_cfg.items():
        fname = sig['file']
        if fname not in sig_cache:
            sig_cache[fname] = load_csv(data_dir / fname)
        df_sig = sig_cache[fname]

        z_col  = sig['z_col']
        tb_col = sig['tb_col']
        sign   = sig.get('tb_sign', 1)

        # Drop NaN rows for this pair of columns
        valid  = df_sig[[z_col, tb_col]].dropna()
        z_vals = valid[z_col].values
        tb_mK  = valid[tb_col].values * sign   # now absorption = negative

        freq_sig = z_to_freq_MHz(z_vals)       # MHz

        # Convert mK -> plot units
        tb_plot = tb_mK * unit_scale / 1e3     # mK -> K if unit='K', stays mK if 'mK'

        # Only plot where in x range
        mask = (freq_sig >= 0.1) & (freq_sig <= 300)
        if mask.sum() < 2:
            print(f"  [{sig_key}] no points in plot range")
            continue

        # Sort by frequency
        order = np.argsort(freq_sig[mask])
        fx_s  = freq_sig[mask][order]
        fy_s  = tb_plot[mask][order]

        sig_style = {k: v for k, v in sig.items()
                     if k in ('color','linestyle','linewidth','zorder','alpha')}
        ax_bot.plot(fx_s, fy_s, label=sig.get('label', sig_key), **sig_style)

    # ── Goal accuracy lines ───────────────────────────────────────────────────
    goal_unit_scale = unit_scale / 1e3   # mK -> plot units
    goals = [
        (10.0,   '#ff9900', '--',  'Pathfinder goal: 10 K'),
        (0.01,    '#009900', '--',  'Science goal: 10 mK'),
    ]
    freq_goal = np.array([1.0, 200.0])
    freqs = np.logspace(np.log10(freq_goal[0]),np.log10(freq_goal[1]))
    for val_K, col, ls, lbl in goals:
        val_plot = val_K * unit_scale   # K -> plot units
        ax_bot.axhline( val_plot, color=col, ls=ls, lw=1.2, alpha=0.8, label=lbl)
        ax_bot.axhline(-val_plot, color=col, ls=ls, lw=1.2, alpha=0.8)

        ax_bot.fill_between(freqs,-1*val_plot*np.ones_like(freqs),
            y2=val_plot*np.ones_like(freqs),alpha=0.3,color=col)
        ax_bot.text(freq_goal[0]*4,val_plot/4,lbl,fontsize=8)

    # ── Residual panel formatting ─────────────────────────────────────────────
    ax_bot.set_xscale('log')
    ax_bot.set_yscale('symlog',
                      linthresh=resid_linth * unit_scale,
                      linscale=0.5)
    ax_bot.set_ylim(-resid_ylim * unit_scale, resid_ylim * unit_scale)
    ax_bot.set_xlabel('Frequency (MHz)', fontsize=11)
    ax_bot.set_ylabel(f'Residual ({unit_label})', fontsize=11)
    ax_bot.axhline(0, color='black', lw=0.7, alpha=0.5)
    ax_bot.axvspan(0.1, 10, alpha=0.04, color='purple', zorder=0)
    ax_bot.axvline(10, color='purple', lw=1, ls='--', alpha=0.4, zorder=1)
    # ionospheric distortion significant
    ax_bot.axvspan(10, 40, alpha=0.04, color='blue', zorder=0)
    ax_bot.axvline(40, color='blue', lw=1, ls='--', alpha=0.4, zorder=1)


    ax_bot.grid(True, which='major', ls='-',  alpha=0.2)
    ax_bot.grid(True, which='minor', ls='--', alpha=0.1)
    ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))

    leg_bot = ax_bot.legend(loc='lower left', fontsize=7, framealpha=0.88,
                  ncol=2, handlelength=2.0)
    leg_bot.set_zorder(10)
    # (Ionospheric cutoff line extends from top panel — no label needed here)

    # ── Redshift axis on top of top panel ────────────────────────────────────
    ax_top2 = ax_top.twiny()
    ax_top2.set_xscale('log')
    ax_top2.set_xlim(ax_top.get_xlim())
    z_ticks = np.array([10, 30, 50, 100, 200, 500, 1000])
    freq_ticks = z_to_freq_MHz(z_ticks)
    # Only show ticks within plot range
    in_range = (freq_ticks >= 0.1) & (freq_ticks <= 300)
    ax_top2.set_xticks(freq_ticks[in_range])
    ax_top2.set_xticklabels([str(z) for z in z_ticks[in_range]], fontsize=8)
    ax_top2.set_xlabel('← Redshift (21 cm)', fontsize=9, labelpad=4)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax_top.set_title(
        'Sky-averaged radio background: historical measurements & 21 cm signal targets',
        fontsize=11, pad=28)   # pad to clear the redshift axis

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output',   default='spectrum.png')
    parser.add_argument('--unit',     default='K', choices=['K', 'mK'])
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)
    print(f"Data dir: {data_dir.resolve()}")
    print(f"Output  : {output_path.resolve()}")
    print(f"Unit    : {args.unit}\n")
    make_plot(data_dir, output_path, unit=args.unit)

if __name__ == '__main__':
    main()
