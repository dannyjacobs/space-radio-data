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

_OWN_STYLE_KEYS = {'legend_group', 'fit_poly', 'show_temp', 'show_resid'}

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
    resid_ytmin  = rp.get('ytick_min', resid_linth)   # suppress decade ticks below this
    signals_cfg  = load_toml(data_dir / 'signals.toml')['signals']

    csv_cache = {}

    unit_label = 'K' if unit == 'K' else 'mK'
    unit_scale = 1.0 if unit == 'K' else 1e3

    # ── Load every dataset and read its three independent flags ───────────────
    #   fit_poly   : include in the foreground polynomial fit (default False)
    #   show_temp  : show in the upper brightness-temperature panel (default True)
    #   show_resid : show in the lower residual panel (default True)
    panel_data = {}   # key -> (x, y, yerr, style, label, flags)
    fit_x_all, fit_y_all, fit_keys = [], [], []

    for key, ds in datasets_cfg.items():
        style = styles_cfg.get(key, {})

        flags = {
            'fit_poly':   style.get('fit_poly',   False),
            'show_temp':  style.get('show_temp',  True),
            'show_resid': style.get('show_resid', True),
        }

        # A dataset with no panel role and not in the fit contributes nothing
        if not (flags['fit_poly'] or flags['show_temp'] or flags['show_resid']):
            continue

        fname = ds['file']
        if fname not in csv_cache:
            csv_cache[fname] = load_csv(data_dir / fname)
        df = apply_filter(csv_cache[fname], ds.get('filter', []))
        if df.empty:
            print(f"  [{key}] WARNING: 0 rows after filter")
            continue

        x, y, yerr = get_xy(df, ds, unit)
        panel_data[key] = (x, y, yerr, style, ds.get('label', key), flags)

        # Warn on the confusing combination: drives the fit but hidden from residual
        if flags['fit_poly'] and not flags['show_resid']:
            print(f"  [{key}] WARNING: fit_poly=true but show_resid=false — "
                  f"this dataset shapes the fit yet is hidden from the residual "
                  f"panel, which can be confusing to interpret.")

        if flags['fit_poly']:
            fit_x_all.append(x)
            fit_y_all.append(y)
            fit_keys.append(key)

        with np.errstate(invalid='ignore'):
            print(f"  [{key}] {len(x)} pts, "
                  f"{np.nanmin(x):.2f}–{np.nanmax(x):.2f} MHz, "
                  f"T {np.nanmin(y):.1f}–{np.nanmax(y):.1f} {unit_label}"
                  f"  [fit={flags['fit_poly']}, temp={flags['show_temp']}, "
                  f"resid={flags['show_resid']}]")

    # ── Fit foreground model to all fit_poly=true datasets ────────────────────
    if not fit_x_all:
        sys.exit("No datasets have fit_poly=true; cannot build foreground model.")
    fx = np.concatenate(fit_x_all)
    fy = np.concatenate(fit_y_all)
    fg_model, fg_coeffs = fit_foreground(fx, fy, degree=5,
                                          freq_min=resid_fmin, freq_max=400.0)
    print(f"\n  Foreground fit: degree-5 log-log polynomial")
    print(f"  Fit range: {resid_fmin}+ MHz using: {fit_keys}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                            hspace=0.0, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    freq_plot = np.geomspace(0.1, 350, 600)

    # ── TOP PANEL: data + foreground model ────────────────────────────────────
    group_order  = ['space', 'ground']
    group_labels = {'space': 'Space-based', 'ground': 'Ground-based'}

    unit_s = 1e3 if unit == 'mK' else 1.0
    for key, (x, y, yerr, style, label, flags) in panel_data.items():
        if not flags['show_temp']:
            continue          # dataset opted out of the temperature panel
        mpl_style = style_to_mpl(style)
        # Continuous spectrum datasets (marker="none") use ax.plot so that
        # NaN-flagged channels appear as breaks rather than interpolated lines
        if mpl_style.get('marker', 'o') == 'none':
            lw    = mpl_style.get('linewidth', mpl_style.get('lw', 1.2))
            ls    = mpl_style.get('linestyle', mpl_style.get('ls', '-'))
            color = mpl_style.get('color', 'black')
            alpha = mpl_style.get('alpha', 1.0)
            zord  = mpl_style.get('zorder', 3)
            ax_top.plot(x, y, lw=lw, ls=ls, color=color,
                        alpha=alpha, zorder=zord, label=label)
        else:
            ax_top.errorbar(x, y, yerr=yerr, label=label, **mpl_style)

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
    ax_top.set_xlim(0.1, 350)

    # ── Top panel legend ──────────────────────────────────────────────────────
    # Built from config, grouped by legend_group. Only datasets with
    # show_temp=true appear here (they are the ones plotted on this panel).
    # The foreground model line is appended at the end.

    # Harvest the matplotlib handles that were actually plotted on ax_top
    mpl_handles, mpl_labels = ax_top.get_legend_handles_labels()
    handle_map = dict(zip(mpl_labels, mpl_handles))

    leg_h, leg_l = [], []
    for grp in group_order:
        grp_entries = []
        for key, ds in datasets_cfg.items():
            style = styles_cfg.get(key, {})
            if not style.get('show_temp', True):
                continue        # only datasets shown in the temp panel appear here
            if style.get('legend_group', 'other') != grp:
                continue
            lbl = ds.get('label', key)
            if lbl in handle_map:
                grp_entries.append((lbl, handle_map[lbl]))

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
    for key, (x, y, yerr, style, label, flags) in panel_data.items():
        if not flags['show_resid']:
            continue          # dataset opted out of the residual panel
        ds_cfg = datasets_cfg.get(key, {})
        resid_ready = ds_cfg.get('residual_ready', False)

        # Clip to fit range — no residual outside where the model is valid.
        # residual_ready datasets are already a residual (skip fg subtraction)
        # but are still clipped to the same x range for visual consistency.
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

        mpl_style = style_to_mpl(style)
        # Point-marker datasets get shrunk slightly in the denser residual panel;
        # line datasets (marker='none') and residual_ready sets keep their style.
        is_line = mpl_style.get('marker', 'o') == 'none'
        if not is_line and not resid_ready:
            mpl_style = {**mpl_style,
                         'markersize': mpl_style.get('markersize', 4) * 0.7,
                         'alpha':      mpl_style.get('alpha', 0.7) * 0.85}
        # Datasets shown ONLY in the residual panel carry their own label there
        lab = label if not flags['show_temp'] else None
        if is_line:
            lw    = mpl_style.get('linewidth', mpl_style.get('lw', 1.2))
            ls    = mpl_style.get('linestyle', mpl_style.get('ls', '-'))
            color = mpl_style.get('color', 'black')
            alpha = mpl_style.get('alpha', 0.7) * 0.85
            zord  = mpl_style.get('zorder', 3)
            ax_bot.plot(xc, resid, lw=lw, ls=ls, color=color,
                        alpha=alpha, zorder=zord, label=lab)
        else:
            ax_bot.errorbar(xc, resid, yerr=yerrc, label=lab, **mpl_style)

    # Data scatter band: RMS of residuals across datasets that went through the
    # fg subtraction (i.e. shown in residual AND not a pre-computed residual).
    scatter_keys = {k for k, (x, y, yerr, style, label, flags) in panel_data.items()
                    if flags['show_resid']
                    and not datasets_cfg.get(k, {}).get('residual_ready', False)}
    all_x_raw = np.concatenate([panel_data[k][0] for k in scatter_keys])
    all_y_raw = np.concatenate([panel_data[k][1] for k in scatter_keys])
    clip_all  = (all_x_raw >= resid_fmin) & (all_x_raw <= resid_fmax) & np.isfinite(all_y_raw)
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
    freq_goal = np.array([2.0, 200.0])
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

    # Explicit symmetric decade ticks outside ±ytick_min (no zero label, no
    # crowded near-threshold ticks). Suppress symlog's minor ticks entirely,
    # since those are what pile up near the linear/log boundary.
    def symlog_decade_ticks(ymax, ymin_tick):
        hi = int(np.floor(np.log10(ymax)))
        lo = int(np.ceil(np.log10(ymin_tick)))
        pos = [10.0**e for e in range(lo, hi + 1)]
        return [-v for v in reversed(pos)] + pos      # note: no 0.0

    yticks = symlog_decade_ticks(resid_ylim * unit_scale,
                                 resid_ytmin * unit_scale)
    ax_bot.set_yticks(yticks)
    ax_bot.yaxis.set_minor_locator(ticker.NullLocator())
    ax_bot.set_xlabel('Frequency (MHz)', fontsize=11)
    ax_bot.set_ylabel(f'Residual ({unit_label})', fontsize=11)
    ax_bot.axhline(0, color='black', lw=0.7, alpha=0.5)
    ax_bot.axvspan(0.1, 10, alpha=0.04, color='purple', zorder=0)
    ax_bot.axvline(10, color='purple', lw=1, ls='--', alpha=0.4, zorder=1)
    # ionospheric distortion significant
    ax_bot.axvspan(10, 40, alpha=0.04, color='blue', zorder=0)
    ax_bot.axvline(40, color='blue', lw=1, ls='--', alpha=0.4, zorder=1)


    ax_bot.grid(True, which='major', ls='-',  alpha=0.2)
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


    # add a citation
    fig.text(.6,.06,"github.com/dannyjacobs/space-radio-data/",color='lightsteelblue')


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
