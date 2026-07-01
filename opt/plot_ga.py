import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# --- BATCH CONFIGURATION ---
# Add filenames (without path) below.
# Score is read directly from the 'score' column in the CSV.
# If no 'score' column is present, plots are skipped for that file.
# ==============================================================================
 
RUN_LIST = [
    # ("1-LLC_GA_20260508_1849_Gen50_Pop250.csv"),
    # ("2-LLC_GA_20260509_0110_Gen50_Pop250.csv"),
    # ("3-LLC_GA_20260509_1000_Gen50_Pop250.csv"),
    # ("4-LLC_GA_20260510_0015_Gen50_Pop250.csv"),
    # ("5-LLC_GA_20260510_1128_Gen50_Pop250.csv"),
    # ("6-LLC_GA_20260510_1600_Gen50_Pop250.csv"),
    # ("7-LLC_GA_20260510_2053_Gen50_Pop250.csv"),
    # ("8-LLC_GA_20260511_0805_Gen50_Pop250.csv"),
    # ("9-LLC_GA_20260511_2054_Gen50_Pop250.csv"),
    # ("10-LLC_GA_20260512_0805_Gen50_Pop250.csv"),
    # ("11-LLC_GA_20260512_2017_Gen50_Pop250.csv"),
    # ("12-LLC_GA_20260513_2032_Gen50_Pop250.csv"),
    # ("13-LLC_GA_20260514_0708_Gen50_Pop250.csv"),
    #("14-LLC_GA_20260514_1330_Gen50_Pop250.csv"),
    # ("15-LLC_GA_20260514_2038_Gen50_Pop250.csv"),
    # ("16-LLC_GA_20260515_0737_Gen50_Pop250.csv"),
    # ("17-LLC_GA_20260515_1155_Gen50_Pop250.csv"),
    # ("18-LLC_GA_20260515_1640_Gen50_Pop250.csv"),
    # ("19-LLC_GA_20260515_2058_Gen50_Pop250.csv"),
    #("20-LLC_GA_20260516_1710_Gen50_Pop250.csv"),
    #("21-LLC_GA_20260529_2207_Gen50_Pop250.csv"),
    #("22-LLC_GA_20260530_0756_Gen50_Pop250.csv"),
    #("23-LLC_GA_20260530_1342_Gen50_Pop250.csv"),
    #("LLC_GA_20260613_0803_Gen30_Pop250.csv"),
    #("LLC_GA_20260613_1551_Gen50_Pop250.csv"),
    #("LLC_GA_20260613_2057_Gen50_Pop250.csv"),
    #("LLC_GA_20260614_0847_Gen50_Pop250.csv"),
    ("LLC_GA_20260630_1237_Gen10_Pop10.csv")
]

# Design frequency limits
FSW_MIN_HZ = 50e3
FSW_MAX_HZ = 500e3

# Set to True to display each figure interactively; False to only save to PDF
SHOW_PLOTS = False

_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def op_label(df_row, prefix, short=False):
    try:
        vin = int(df_row[f'{prefix}Vin_target'])
        vo  = int(df_row[f'{prefix}Vo_target'])
        io  = df_row[f'{prefix}Io_target']
        if short:
            return f"→{vo}V@{io}A"
        return f"{vin}V $\\rightarrow$ {vo}V @ {io}A"
    except (KeyError, ValueError):
        return prefix.rstrip('_')


def calc_physical_constraints(row, n_pts):
    """Returns real physical quantities, not the G constraint expressions."""
    fsw_vals = [row.get(f'P{i}_fsw_khz', np.nan) for i in range(1, n_pts + 1)]
    fsw_min_op = np.nanmin(fsw_vals)
    fsw_max_op = np.nanmax(fsw_vals)

    ids_off_vals = [row.get(f'P{i}_iDSoff', np.nan) for i in range(1, n_pts + 1)]
    ids_off_min = np.nanmin(ids_off_vals)

    zvs_list, dt_list, reg_list = [], [], []
    t_dis_list, dt_max_list = [], []

    for i in range(1, n_pts + 1):
        t_dis  = row.get(f'P{i}_t_discharge_req', np.nan)
        dt_max = row.get(f'P{i}_max_dt_win_sim', np.nan)
        fsw_i  = row.get(f'P{i}_fsw_khz', np.nan)
        vo_sim = row.get(f'P{i}_voutAVG', np.nan)
        vo_tgt = row.get(f'P{i}_Vo_target', np.nan)

        if not np.isnan(t_dis):
            t_dis_list.append(t_dis)
        if not np.isnan(dt_max):
            dt_max_list.append(dt_max)
        if not any(np.isnan(v) for v in [t_dis, dt_max]):
            zvs_list.append((t_dis - dt_max) * 1e9)
        if not any(np.isnan(v) for v in [dt_max, fsw_i]):
            dt_list.append((dt_max * (fsw_i * 1e3)) * 200.0)
        if not any(np.isnan(v) for v in [vo_sim, vo_tgt]) and vo_tgt > 0:
            reg_list.append(abs(vo_sim - vo_tgt) / vo_tgt * 100.0)

    return {
        'fsw_min_op':         fsw_min_op,
        'fsw_max_op':         fsw_max_op,
        'zvs_tdiff_ns':       max(zvs_list)    if zvs_list    else np.nan,
        'ids_off_min':        ids_off_min,
        'min_dt_win_sim_pct': min(dt_list)     if dt_list     else np.nan,
        'reg_err_max':        max(reg_list)    if reg_list    else np.nan,
        't_dis_worst_ns':     max(t_dis_list)  * 1e9 if t_dis_list  else np.nan,
        'dt_max_worst_ns':    min(dt_max_list) * 1e9 if dt_max_list else np.nan,
    }


def _safe_fmt(val, fmt):
    """Format a numeric value; return '—' on NaN or error."""
    try:
        v = float(val)
        return '—' if np.isnan(v) else f"{v:{fmt}}"
    except (TypeError, ValueError):
        return '—'


# ==============================================================================
# --- MAIN PROCESSING FUNCTION ---
# ==============================================================================

def process_and_plot(filename):
    filepath = os.path.join(_OUTPUTS_DIR, filename)
    print(f"\n{'='*70}")
    print(f"[INFO] Processing: {filename}")

    try:
        df = pd.read_csv(filepath)
        valid_df = df[df['status'] == 'Valid'].copy()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return

    if valid_df.empty:
        print("[WARNING] No valid solutions found — skipping.")
        return

    # Dynamically identify operating points (P1, P2, P3...)
    irrms_cols = [col for col in valid_df.columns if col.endswith('_iRRMS')]
    num_points = len(irrms_cols)
    print(f"Found {num_points} operating points in history.")

    # Use score directly from CSV
    if 'score' not in valid_df.columns:
        print("[WARNING] No 'score' column found in CSV — skipping all plots.")
        return
    valid_df['score_robusto'] = valid_df['score']
    _score_label = r'Score [A]'
    print("Score: read directly from 'score' column in CSV.")

    best_overall       = valid_df.loc[valid_df['score_robusto'].idxmin()]
    best_per_gen_idx   = valid_df.groupby('generation')['score_robusto'].idxmin()
    best_per_gen_df    = valid_df.loc[best_per_gen_idx].sort_values('generation').copy()
    best_per_gen_score = valid_df.groupby('generation')['score_robusto'].min().reset_index()

    generations = sorted(valid_df['generation'].unique())
    markers     = ['o', 's', '^', 'D', 'v', '>']

    _gen_max        = int(valid_df['generation'].max())
    _major_ticks    = sorted({0, 1} | set(range(5, _gen_max + 1, 5)))
    _major_tick_set = set(_major_ticks)
    _cat_tick_pos   = [i for i, g in enumerate(generations) if g in _major_tick_set]
    _cat_tick_labs  = [str(g) for g in generations          if g in _major_tick_set]

    csv_base = os.path.splitext(filename)[0]
    pdf_path = os.path.join(_OUTPUTS_DIR, csv_base + '.pdf')

    all_figs = []

    # ==========================================================================
    # COVER PAGE: Best Overall Individual — Parameters & Key Metrics
    # ==========================================================================
    _best_constr = calc_physical_constraints(best_overall, num_points)

    fig_cover = plt.figure(figsize=(11, 8.5))
    ax_cv = fig_cover.add_axes([0, 0, 1, 1])
    ax_cv.set_xlim(0, 1)
    ax_cv.set_ylim(0, 1)
    ax_cv.axis('off')

    def _cv(x, y, txt, size=10, bold=False, color='black', ha='left'):
        ax_cv.text(x, y, txt, ha=ha, va='top', fontsize=size,
                   fontweight='bold' if bold else 'normal', color=color,
                   transform=ax_cv.transAxes)

    _cv(0.5, 0.975, "LLC GA Optimization — Best Overall Individual Summary",
        size=15, bold=True, ha='center')
    _cv(0.5, 0.930, f"File: {filename}", size=9, color='gray', ha='center')
    ax_cv.axhline(0.910, color='darkgray', linewidth=1.5, xmin=0.04, xmax=0.96)

    _lh  = 0.042
    _y0  = 0.875

    # --- Left: Design Parameters ---
    _yl = _y0
    _cv(0.04, _yl, "Design Parameters", size=11, bold=True); _yl -= _lh
    _dp_fields = [
        ('n',       'Turns ratio  n',          1,    '',    '.4f'),
        ('Cs',      'Series capacitor  Cs',    1e9,  'nF',  '.3f'),
        ('fo_res',  'Resonant frequency  fo',  1e-3, 'kHz', '.2f'),
        ('k_ratio', 'Inductance ratio  k',     1,    '',    '.4f'),
    ]
    for _col, _lbl, _sc, _un, _fmt in _dp_fields:
        if _col in best_overall.index and not pd.isna(best_overall[_col]):
            _v = best_overall[_col] * _sc
            _cv(0.06, _yl, f"{_lbl}:   {_v:{_fmt}} {_un}".rstrip(), size=10)
            _yl -= _lh
    # Ls = 1/(4π²·fo²·Cs);  Lm = k·Ls
    try:
        _fo_f = float(best_overall.get('fo_res', np.nan))
        _Cs_f = float(best_overall.get('Cs', np.nan))
        _k_f  = float(best_overall.get('k_ratio', np.nan))
        if not (np.isnan(_fo_f) or np.isnan(_Cs_f)) and _fo_f > 0 and _Cs_f > 0:
            _Ls_calc = 1.0 / (4.0 * np.pi**2 * _fo_f**2 * _Cs_f)
            _cv(0.06, _yl, f"Series inductance  Ls:   {_Ls_calc*1e6:.3f} µH", size=10)
            _yl -= _lh
            if not np.isnan(_k_f):
                _Lm_calc = _k_f * _Ls_calc
                _cv(0.06, _yl, f"Mag. inductance  Lm:   {_Lm_calc*1e6:.3f} µH", size=10)
                _yl -= _lh
    except (TypeError, ValueError):
        pass

    # --- Right: Performance & Constraints ---
    _yr = _y0
    _cv(0.52, _yr, "Performance & Constraints", size=11, bold=True); _yr -= _lh

    _sv   = best_overall['score_robusto']
    _genv = int(best_overall['generation'])
    _cv(0.54, _yr, f"Score:                      {_sv:.5f} A", size=10); _yr -= _lh
    _cv(0.54, _yr, f"Found at generation:    {_genv}", size=10); _yr -= _lh
    _yr -= _lh * 0.3
    _cv(0.54, _yr, "Constraint Summary:", size=10, bold=True); _yr -= _lh
    _cv(0.56, _yr,
        f"fsw range:   {_safe_fmt(_best_constr['fsw_min_op'],'.1f')} – "
        f"{_safe_fmt(_best_constr['fsw_max_op'],'.1f')} kHz", size=10); _yr -= _lh
    _cv(0.56, _yr,
        f"Min dt window (worst):  {_safe_fmt(_best_constr['min_dt_win_sim_pct'],'.1f')} % of Tsw/2",
        size=10); _yr -= _lh
    _cv(0.56, _yr,
        f"Max regulation error:  {_safe_fmt(_best_constr['reg_err_max'],'.3f')} %",
        size=10); _yr -= _lh

    # Divider before table
    _y_div = min(_yl, _yr) - 0.015
    ax_cv.axhline(_y_div, color='darkgray', linewidth=1.0, xmin=0.04, xmax=0.96)

    # Per-operating-point table
    _yt = _y_div - 0.02
    _cv(0.5, _yt, "Per Operating Point — Best Individual",
        size=11, bold=True, ha='center'); _yt -= _lh

    _col_x = [0.08, 0.20, 0.33, 0.46, 0.59, 0.72, 0.85]
    _hdrs  = ["Op. Point", "fsw (kHz)", "iRRMS (A)", "iDS_off (A)",
               "VCs_rms (V)", "VCs_pk (V)", "iD1_rms (A)"]
    for _h, _cx in zip(_hdrs, _col_x):
        _cv(_cx, _yt, _h, size=8.5, bold=True)
    _yt -= -0.006
    ax_cv.axhline(_yt, color='gray', linewidth=0.6, xmin=0.03, xmax=0.97)
    _yt -= _lh * 0.85

    for _i in range(1, num_points + 1):
        _pfx = f"P{_i}_"
        def _gop(col, pfx=_pfx):
            return best_overall.get(f'{pfx}{col}', np.nan)
        try:
            _row_lbl = op_label(best_overall, _pfx, short=True)
        except Exception:
            _row_lbl = f"P{_i}"
        _row_vals = [
            _row_lbl,
            _safe_fmt(_gop('fsw_khz'), '.1f'),
            _safe_fmt(_gop('iRRMS'),   '.4f'),
            _safe_fmt(_gop('iDSoff'),  '.4f'),
            _safe_fmt(_gop('vCsRMS'),  '.2f'),
            _safe_fmt(_gop('vCsPK'),   '.2f'),
            _safe_fmt(_gop('iD1RMS'),  '.4f'),
        ]
        for _rv, _cx in zip(_row_vals, _col_x):
            _cv(_cx, _yt, str(_rv), size=9)
        _yt -= _lh
        ax_cv.axhline(_yt + 0.005, color='lightgray', linewidth=0.4, xmin=0.03, xmax=0.97)

    fig_cover.tight_layout()
    all_figs.append(fig_cover)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 01: Population status per generation — Valid / Fail / Invalid
    # ==========================================================================
    df['_plot_score'] = np.nan
    df.loc[valid_df.index, '_plot_score'] = valid_df['score_robusto']

    _valid_mask   = df['status'] == 'Valid'
    _fail_mask    = df['status'].str.startswith('Fail',    na=False)
    _invalid_mask = df['status'].str.startswith('Invalid', na=False)

    _rng = np.random.default_rng(42)
    def _jx(s):
        return s + _rng.uniform(-0.35, 0.35, size=len(s))

    _v_scores  = df.loc[_valid_mask, '_plot_score'].dropna()
    _y_vmin    = _v_scores.min() if len(_v_scores) else 0.0
    _span      = max((_v_scores.max() - _y_vmin) if len(_v_scores) else 1.0, 0.01)
    _gap       = _span * 0.12
    _half_band = _gap * 0.4
    _y_sep     = _y_vmin - _gap
    _y_inv_c   = _y_vmin - _gap * 2.5
    _y_fail_c  = _y_vmin - _gap * 4.0

    fig0, ax0 = plt.subplots(figsize=(12, 7))

    _gv = df.loc[_valid_mask,   'generation']
    _gi = df.loc[_invalid_mask, 'generation']
    _gf = df.loc[_fail_mask,    'generation']

    ax0.scatter(_jx(_gv), df.loc[_valid_mask, '_plot_score'],
                color='steelblue', alpha=0.6, s=18, label='Valid', zorder=3)

    if len(_gi):
        _yi = _y_inv_c + _rng.uniform(-_half_band, _half_band, size=len(_gi))
        ax0.scatter(_jx(_gi), _yi, color='black', alpha=0.7, s=18, label='Invalid', zorder=3)

    if len(_gf):
        _yf = _y_fail_c + _rng.uniform(-_half_band, _half_band, size=len(_gf))
        ax0.scatter(_jx(_gf), _yf, color='red', alpha=0.7, s=18, label='Fail', zorder=3)

    ax0.axhline(_y_sep, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    _gen_max = int(df['generation'].max())
    if len(_gi) or len(_gf):
        _y_bot = (_y_fail_c if len(_gf) else _y_inv_c) - _half_band * 2
        ax0.axhspan(_y_bot, _y_sep, alpha=0.05, color='gray')
        if len(_gi):
            ax0.text(_gen_max + 0.3, _y_inv_c, 'Invalid zone',
                     va='center', ha='left', color='black', fontsize=8)
        if len(_gf):
            ax0.text(_gen_max + 0.3, _y_fail_c, 'Fail zone',
                     va='center', ha='left', color='red', fontsize=8)

    _all_gens  = sorted(df['generation'].unique())
    _all_mtick = sorted({0, 1} | set(range(5, int(df['generation'].max()) + 1, 5)))
    ax0.set_xticks([g for g in _all_mtick if g in set(_all_gens)])
    ax0.set_title("Population Status per Generation — All Individuals", fontsize=14)
    ax0.set_xlabel("Generation", fontsize=12)
    ax0.set_ylabel(_score_label + "  (Valid zone)  /  status zone (Fail · Invalid)", fontsize=11)
    ax0.legend(loc='upper right', fontsize=10)
    ax0.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig0.tight_layout()
    all_figs.append(fig0)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 02: Individual count per generation — Valid / Fail / Invalid
    # ==========================================================================
    _status_cat = pd.Categorical(
        np.where(_valid_mask,   'Valid',
        np.where(_fail_mask,    'Fail',
        np.where(_invalid_mask, 'Invalid', 'Other'))),
        categories=['Valid', 'Invalid', 'Fail']
    )
    _cnt = (df.assign(_cat=_status_cat)
              .groupby(['generation', '_cat'], observed=True)
              .size()
              .unstack(fill_value=0)
              .reindex(columns=['Valid', 'Invalid', 'Fail'], fill_value=0))

    _gens_cnt  = _cnt.index.values
    _bar_w     = max(0.55, 0.8 * min(np.diff(_gens_cnt)) if len(_gens_cnt) > 1 else 0.8)
    _colors_cb = {'Valid': 'steelblue', 'Invalid': 'black', 'Fail': 'red'}

    fig0b, (ax0b_bar, ax0b_pct) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig0b.suptitle("Individual Count per Generation — Status Breakdown", fontsize=14)

    _bottom = np.zeros(len(_gens_cnt))
    for _cat in ['Valid', 'Invalid', 'Fail']:
        if _cat in _cnt.columns:
            _vals = _cnt[_cat].values.astype(float)
            ax0b_bar.bar(_gens_cnt, _vals, bottom=_bottom, width=_bar_w,
                         color=_colors_cb[_cat], label=_cat, alpha=0.85, edgecolor='white', linewidth=0.4)
            _bottom += _vals

    ax0b_bar.set_ylabel("Count", fontsize=11)
    ax0b_bar.legend(loc='upper right', fontsize=10)
    ax0b_bar.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax0b_bar.set_ylim(0, _bottom.max() * 1.12)

    _total = _cnt.sum(axis=1).replace(0, np.nan)
    _bottom_pct = np.zeros(len(_gens_cnt))
    for _cat in ['Valid', 'Invalid', 'Fail']:
        if _cat in _cnt.columns:
            _pct = (_cnt[_cat] / _total * 100).fillna(0).values
            ax0b_pct.bar(_gens_cnt, _pct, bottom=_bottom_pct, width=_bar_w,
                         color=_colors_cb[_cat], label=_cat, alpha=0.85, edgecolor='white', linewidth=0.4)
            _bottom_pct += _pct

    _valid_pct = (_cnt['Valid'] / _total * 100).fillna(0)
    ax0b_pct.plot(_gens_cnt, _valid_pct.values, color='steelblue',
                  linewidth=1.8, marker='o', markersize=4, zorder=5, label='% Valid (line)')
    ax0b_pct.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax0b_pct.set_ylabel("Percentage (%)", fontsize=11)
    ax0b_pct.set_xlabel("Generation", fontsize=11)
    ax0b_pct.set_ylim(0, 112)
    ax0b_pct.legend(loc='lower right', fontsize=9)
    ax0b_pct.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax0b_pct.set_xticks([g for g in _all_mtick if g in set(_all_gens)])

    fig0b.tight_layout()
    all_figs.append(fig0b)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 03: Score Evolution — individuals (blue), best per gen (red, small),
    #           best overall (gold star)
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.stripplot(data=valid_df, x='generation', y='score_robusto',
                  color='steelblue', alpha=0.5, jitter=0.2, ax=ax1, legend=False)

    gen_offset = best_per_gen_score['generation'].min()
    ax1.plot(best_per_gen_score['generation'] - gen_offset,
             best_per_gen_score['score_robusto'],
             color='red', marker='D', markersize=4, label='Best per Generation')
    ax1.scatter(best_overall['generation'] - gen_offset, best_overall['score_robusto'],
                color='gold', edgecolor='black', s=200, marker='*', zorder=5, label='Best Overall')

    blue_patch1 = mpatches.Patch(color='steelblue', alpha=0.5, label='Individuals')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[blue_patch1] + handles, labels=['Individuals'] + labels)
    ax1.set_title(f"Score Evolution — {_score_label}")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel(_score_label)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.set_xticks(_cat_tick_pos)
    ax1.set_xticklabels(_cat_tick_labs)
    fig1.tight_layout()
    all_figs.append(fig1)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 06: Population average score per generation (mean only)
    # ==========================================================================
    pop_stats = valid_df.groupby('generation').agg(
        pop_score_mean = ('score_robusto', 'mean'),
        pop_score_min  = ('score_robusto', 'min'),
    ).reset_index()

    fig1d, ax1d = plt.subplots(figsize=(11, 6))
    ax1d.plot(pop_stats['generation'], pop_stats['pop_score_mean'],
              color='steelblue', marker='o', linewidth=2, markersize=6,
              label='Mean score per generation')
    ax1d.plot(pop_stats['generation'], pop_stats['pop_score_min'],
              color='tomato', marker='D', linewidth=1.5, markersize=5, linestyle='--',
              label='Best score per generation')
    ax1d.set_title("Population Score per Generation\n"
                   "(mean and best score over all valid individuals)", fontsize=13)
    ax1d.set_xlabel("Generation", fontsize=12)
    ax1d.set_ylabel(_score_label, fontsize=12)
    ax1d.set_xticks(_major_ticks)
    ax1d.legend(fontsize=10)
    ax1d.grid(True, linestyle='--', alpha=0.6)
    fig1d.tight_layout()
    all_figs.append(fig1d)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 07: Design Parameter Convergence — all generations
    # Outliers shown in blue (valid individuals); best-per-gen marker reduced.
    # ==========================================================================
    _flier_kw = dict(marker='.', alpha=0.6,
                     markerfacecolor='steelblue', markeredgecolor='steelblue')

    params_to_plot = [
        ('n',       'Turns Ratio $n$',                 1),
        ('Cs',      'Series Capacitor $C_s$ (nF)',     1e9),
        ('k_ratio', 'Inductance Ratio $k = L_m/L_s$', 1),
    ]

    # ------------------------------------------------------------------
    # Local helpers — build violin and percentile-band convergence figs.
    # Both helpers close over params_to_plot and _major_ticks.
    # ------------------------------------------------------------------
    def _violin_figs(df_sub, gens_list, champ_df, best_row, title_suffix, tick_pos, tick_labs):
        """One violin figure per parameter — parameter distribution across generations."""
        figs = []
        for col, title, scale in params_to_plot:
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.suptitle(f"Design Parameter Convergence — Violin{title_suffix}", fontsize=15)
            sns.violinplot(data=df_sub, x='generation', y=df_sub[col] * scale,
                           ax=ax, color='lightsteelblue', inner='quartile', cut=0,
                           linewidth=0.8)
            cx, cy = [], []
            for gi, g in enumerate(gens_list):
                r = champ_df[champ_df['generation'] == g]
                if not r.empty:
                    cx.append(gi)
                    cy.append(float(r[col].iloc[0]) * scale)
            ax.plot(cx, cy, color='red', marker='D', markersize=3,
                    linestyle='--', linewidth=1.5, label='Best per Generation', zorder=5)
            _bv = float(best_row[col]) * scale
            ax.axhline(_bv, color='green', linestyle='-', linewidth=2.0,
                       label=f'Best Overall: {_bv:.4g}', zorder=4)
            ax.set_title(title, fontsize=13)
            ax.set_ylabel(title)
            ax.set_xlabel('Generation')
            if tick_pos:
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_labs)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=9)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            figs.append(fig)
        return figs

    def _pctile_figs(df_sub, gens_list, champ_df, best_row, title_suffix, xticks):
        """One percentile-band figure per parameter.

        Outer band: 10th–90th percentile (light fill).
        Inner band: 25th–75th percentile (IQR, darker fill).
        Centre line: median.
        Red dashed line: best individual per generation.
        Green line: best overall individual.
        """
        figs = []
        for col, title, scale in params_to_plot:
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.suptitle(f"Design Parameter Convergence — Percentile Bands{title_suffix}", fontsize=15)
            _scaled = df_sub[col] * scale
            _pct = (_scaled.groupby(df_sub['generation'])
                           .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                           .unstack())
            gp = _pct.index.values
            ax.fill_between(gp, _pct[0.10], _pct[0.90],
                            alpha=0.12, color='steelblue', label='10th–90th %ile')
            ax.fill_between(gp, _pct[0.25], _pct[0.75],
                            alpha=0.30, color='steelblue', label='25th–75th %ile (IQR)')
            ax.plot(gp, _pct[0.50],
                    color='steelblue', linewidth=2, marker='o', markersize=3,
                    label='Median')
            cg, cy = [], []
            for g in gens_list:
                r = champ_df[champ_df['generation'] == g]
                if not r.empty:
                    cg.append(g)
                    cy.append(float(r[col].iloc[0]) * scale)
            ax.plot(cg, cy, color='red', marker='D', markersize=3,
                    linestyle='--', linewidth=1.5, label='Best per Generation', zorder=5)
            _bv = float(best_row[col]) * scale
            ax.axhline(_bv, color='green', linestyle='-', linewidth=2.0,
                       label=f'Best Overall: {_bv:.4g}', zorder=4)
            ax.set_title(title, fontsize=13)
            ax.set_ylabel(title)
            ax.set_xlabel('Generation')
            ax.set_xticks(xticks)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=8, loc='upper right')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            figs.append(fig)
        return figs

    # --- Plot 07: boxplot — one figure per parameter ---
    for col, title, scale in params_to_plot:
        fig_bp, ax_bp = plt.subplots(figsize=(11, 7))
        fig_bp.suptitle("Design Parameter Convergence — Boxplot", fontsize=15)
        sns.boxplot(data=valid_df, x='generation', y=valid_df[col] * scale,
                    ax=ax_bp, color='skyblue', flierprops=_flier_kw)
        champ_x, champ_y = [], []
        for gi, g in enumerate(generations):
            row = best_per_gen_df[best_per_gen_df['generation'] == g]
            if not row.empty:
                champ_x.append(gi)
                champ_y.append(float(row[col].iloc[0]) * scale)
        ax_bp.plot(champ_x, champ_y, color='red', marker='D', markersize=4,
                   linestyle='--', linewidth=1.5, label='Best per Generation', zorder=5)
        _bv = float(best_overall[col]) * scale
        ax_bp.axhline(_bv, color='green', linestyle='-', linewidth=2.0,
                      label=f'Best Overall: {_bv:.4g}', zorder=4)
        ax_bp.set_title(title, fontsize=13)
        ax_bp.set_ylabel(title)
        ax_bp.set_xlabel('Generation')
        ax_bp.set_xticks(_cat_tick_pos)
        ax_bp.set_xticklabels(_cat_tick_labs)
        ax_bp.grid(True, linestyle=':', alpha=0.5)
        ax_bp.legend(fontsize=10)
        fig_bp.tight_layout(rect=[0, 0.03, 1, 0.95])
        all_figs.append(fig_bp)
        if SHOW_PLOTS:
            plt.show()

    # --- Plot 07 alt-A: violin ---
    for _f in _violin_figs(valid_df, generations, best_per_gen_df, best_overall,
                            '', _cat_tick_pos, _cat_tick_labs):
        all_figs.append(_f)
        if SHOW_PLOTS:
            plt.show()

    # --- Plot 07 alt-B: percentile bands ---
    for _f in _pctile_figs(valid_df, generations, best_per_gen_df, best_overall,
                            '', _major_ticks):
        all_figs.append(_f)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # PLOT 08: Design Parameter Convergence — last 15 generations
    # ==========================================================================
    _last_n     = min(15, len(generations))
    _last_gens  = generations[-_last_n:]
    _last_valid = valid_df[valid_df['generation'].isin(_last_gens)].copy()
    _last_champ = best_per_gen_df[best_per_gen_df['generation'].isin(_last_gens)]
    _local_gens = sorted(_last_valid['generation'].unique())
    _local_major = [t for t in _major_ticks if t in set(_local_gens)]

    _local_cat_tick_pos  = [i for i, g in enumerate(_local_gens) if g in _major_tick_set]
    _local_cat_tick_labs = [str(g) for g in _local_gens          if g in _major_tick_set]

    # --- Plot 08: boxplot — one figure per parameter ---
    for col, title, scale in params_to_plot:
        fig_bp8, ax_bp8 = plt.subplots(figsize=(11, 7))
        fig_bp8.suptitle(f"Design Parameter Convergence — Boxplot — Last {_last_n} Generations",
                         fontsize=15)
        sns.boxplot(data=_last_valid, x='generation', y=_last_valid[col] * scale,
                    ax=ax_bp8, color='skyblue', flierprops=_flier_kw)
        champ_x, champ_y = [], []
        for gi, g in enumerate(_local_gens):
            row = _last_champ[_last_champ['generation'] == g]
            if not row.empty:
                champ_x.append(gi)
                champ_y.append(float(row[col].iloc[0]) * scale)
        ax_bp8.plot(champ_x, champ_y, color='red', marker='D', markersize=4,
                    linestyle='--', linewidth=1.5, label='Best per Generation', zorder=5)
        _bv = float(best_overall[col]) * scale
        ax_bp8.axhline(_bv, color='green', linestyle='-', linewidth=2.0,
                       label=f'Best Overall: {_bv:.4g}', zorder=4)
        ax_bp8.set_title(title, fontsize=13)
        ax_bp8.set_ylabel(title)
        ax_bp8.set_xlabel('Generation')
        ax_bp8.grid(True, linestyle=':', alpha=0.5)
        ax_bp8.legend(fontsize=10)
        fig_bp8.tight_layout(rect=[0, 0.03, 1, 0.95])
        all_figs.append(fig_bp8)
        if SHOW_PLOTS:
            plt.show()

    # --- Plot 08 alt-A: violin ---
    for _f in _violin_figs(_last_valid, _local_gens, _last_champ, best_overall,
                            f' — Last {_last_n} Generations',
                            _local_cat_tick_pos, _local_cat_tick_labs):
        all_figs.append(_f)
        if SHOW_PLOTS:
            plt.show()

    # --- Plot 08 alt-B: percentile bands ---
    for _f in _pctile_figs(_last_valid, _local_gens, _last_champ, best_overall,
                            f' — Last {_last_n} Generations', _local_major):
        all_figs.append(_f)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # PLOT 09: iRRMS currents + robust score (best individual per generation)
    # ==========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    for i in range(1, num_points + 1):
        prefix = f"P{i}_"
        label  = f"P{i}: {op_label(best_per_gen_df.iloc[0], prefix)}"
        mkr    = markers[(i - 1) % len(markers)]
        col    = f'{prefix}iRRMS'
        if col in best_per_gen_df.columns:
            ax4.plot(best_per_gen_df['generation'], best_per_gen_df[col],
                     marker=mkr, linestyle='-', linewidth=2, markersize=8, label=label)

    ax4.plot(best_per_gen_df['generation'], best_per_gen_df['score_robusto'],
             color='black', linestyle=':', linewidth=2.5, marker='x', markersize=9,
             label=f'Score ({_score_label})')
    ax4.set_xlabel("Generation", fontsize=12)
    ax4.set_ylabel("Current (A)", fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(title="Operating Points", loc='upper right', fontsize=10)
    ax4.set_title(f"$iR_{{rms}}$ and Score Evolution — {_score_label}\n(Best Individual per Generation)",
                  fontsize=14)
    ax4.set_xticks(_major_ticks)
    fig4.tight_layout()
    all_figs.append(fig4)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 10: iDS_off per operating point (best individual per generation)
    # ==========================================================================
    fig5, ax5 = plt.subplots(figsize=(10, 5))

    for i in range(1, num_points + 1):
        prefix = f"P{i}_"
        col    = f'{prefix}iDSoff'
        if col not in best_per_gen_df.columns:
            continue
        label = f"P{i}: {op_label(best_per_gen_df.iloc[0], prefix, short=True)}"
        mkr   = markers[(i - 1) % len(markers)]
        ax5.plot(best_per_gen_df['generation'], best_per_gen_df[col],
                 marker=mkr, linestyle='-', linewidth=2, markersize=8, label=label)

    ax5.set_title("$I_{DS,off}$ Evolution per Operating Point\n(Best Individual per Generation)",
                  fontsize=13)
    ax5.set_xlabel("Generation", fontsize=12)
    ax5.set_ylabel("$I_{DS,off}$ (A)", fontsize=12)
    ax5.legend(title="Operating Points")
    ax5.set_xticks(_major_ticks)
    ax5.grid(True, linestyle='--', alpha=0.6)
    fig5.tight_layout()
    all_figs.append(fig5)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 11: Series capacitor voltage VCs_rms and VCs_peak (best per generation)
    # ==========================================================================
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5))
    fig6.suptitle("Series Capacitor $C_s$ Voltage — Best Individual per Generation", fontsize=14)

    for i in range(1, num_points + 1):
        prefix  = f"P{i}_"
        label   = f"P{i}: {op_label(best_per_gen_df.iloc[0], prefix, short=True)}"
        mkr     = markers[(i - 1) % len(markers)]
        rms_col = f'{prefix}vCsRMS'
        pk_col  = f'{prefix}vCsPK'
        if rms_col in best_per_gen_df.columns:
            ax6a.plot(best_per_gen_df['generation'], best_per_gen_df[rms_col],
                      marker=mkr, linestyle='-', linewidth=2, markersize=8, label=label)
        if pk_col in best_per_gen_df.columns:
            ax6b.plot(best_per_gen_df['generation'], best_per_gen_df[pk_col],
                      marker=mkr, linestyle='-', linewidth=2, markersize=8, label=label)

    ax6a.set_title("$V_{Cs,rms}$")
    ax6a.set_xlabel("Generation")
    ax6a.set_ylabel("RMS Voltage (V)")
    ax6a.set_xticks(_major_ticks)
    ax6a.legend()
    ax6a.grid(True, linestyle='--', alpha=0.6)

    ax6b.set_title("$V_{Cs,peak}$")
    ax6b.set_xlabel("Generation")
    ax6b.set_ylabel("Peak Voltage (V)")
    ax6b.set_xticks(_major_ticks)
    ax6b.legend()
    ax6b.grid(True, linestyle='--', alpha=0.6)

    fig6.tight_layout()
    all_figs.append(fig6)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 12: Best individual parameters per generation (champion trace)
    # ==========================================================================
    fig7, axes7 = plt.subplots(2, 2, figsize=(12, 9))
    fig7.suptitle("Best Individual Parameters per Generation", fontsize=15)

    champ_params = [
        ('n',       'Turns Ratio $n$',                   1,    ''),
        ('Cs',      'Series Capacitor $C_s$ (nF)',       1e9,  'nF'),
        ('fo_res',  'Resonant Frequency $f_o$ (kHz)',    1e-3, 'kHz'),
        ('k_ratio', 'Inductance Ratio $k = L_m / L_s$', 1,    ''),
    ]

    for i, (col, title, scale, unit) in enumerate(champ_params):
        ax = axes7[i // 2, i % 2]
        ax.plot(best_per_gen_df['generation'], best_per_gen_df[col] * scale,
                color='steelblue', marker='o', linewidth=2, markersize=7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Generation")
        ax.set_ylabel(unit)
        ax.set_xticks(_major_ticks)
        ax.grid(True, linestyle=':', alpha=0.5)

    fig7.tight_layout(rect=[0, 0.03, 1, 0.95])
    all_figs.append(fig7)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 14: Switching frequency range — bar chart with per-generation limits
    # ==========================================================================
    constr_rows = []
    for _, row in best_per_gen_df.iterrows():
        c = calc_physical_constraints(row, num_points)
        c['generation'] = int(row['generation'])
        constr_rows.append(c)
    constr_df = pd.DataFrame(constr_rows).set_index('generation')

    gens_c    = constr_df.index.values
    fsw_min_c = constr_df['fsw_min_op'].values
    fsw_max_c = constr_df['fsw_max_op'].values
    bar_width = max(0.6, (gens_c[-1] - gens_c[0]) / len(gens_c) * 0.6) if len(gens_c) > 1 else 0.6

    fig9a, ax_fsw = plt.subplots(figsize=(12, 5))
    fsw_colors = [
        'mediumseagreen' if (mn >= FSW_MIN_HZ / 1e3 and mx <= FSW_MAX_HZ / 1e3) else 'tomato'
        for mn, mx in zip(fsw_min_c, fsw_max_c)
    ]
    ax_fsw.bar(gens_c, fsw_max_c - fsw_min_c, bottom=fsw_min_c,
               width=bar_width, color=fsw_colors, alpha=0.75,
               edgecolor='black', linewidth=0.6,
               label='Range $[f_{sw,min},\\ f_{sw,max}]$')
    ax_fsw.plot(gens_c, fsw_min_c, color='navy',    marker='v',
                linestyle='none', markersize=7, zorder=5, label='$f_{sw,min}$')
    ax_fsw.plot(gens_c, fsw_max_c, color='crimson', marker='^',
                linestyle='none', markersize=7, zorder=5, label='$f_{sw,max}$')
    ax_fsw.axhline(FSW_MIN_HZ / 1e3, color='red', linestyle='--', linewidth=1.5, alpha=0.85,
                   label=f'Min limit ({FSW_MIN_HZ/1e3:.0f} kHz)')
    ax_fsw.axhline(FSW_MAX_HZ / 1e3, color='red', linestyle='--', linewidth=1.5, alpha=0.85,
                   label=f'Max limit ({FSW_MAX_HZ/1e3:.0f} kHz)')

    TXT_OFFSET = 12
    for x, mn, mx in zip(gens_c, fsw_min_c, fsw_max_c):
        if not np.isnan(mx):
            ax_fsw.text(x, mx + TXT_OFFSET, f"{mx:.0f}",
                        ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
        if not np.isnan(mn):
            ax_fsw.text(x, mn - TXT_OFFSET, f"{mn:.0f}",
                        ha='center', va='top', fontsize=7, color='black', fontweight='bold')

    ax_fsw.set_ylim(0, 600)
    ax_fsw.set_title("Switching Frequency Range — Best Individual per Generation\n"
                     "(green = within limits  |  red = limit violated)", fontsize=12)
    ax_fsw.set_xlabel("Generation", fontsize=11)
    ax_fsw.set_ylabel("Frequency (kHz)", fontsize=11)
    ax_fsw.set_xticks(_major_ticks)
    ax_fsw.legend(fontsize=9, loc='upper right', ncol=3)
    ax_fsw.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig9a.tight_layout()
    all_figs.append(fig9a)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 15: ZVS timing — worst case across operating points
    #           (best individual per generation)
    # ==========================================================================
    fig9b, ax_zvs = plt.subplots(figsize=(10, 5))
    t_dis_vals  = constr_df['t_dis_worst_ns'].values
    dt_max_vals = constr_df['dt_max_worst_ns'].values
    gens_z = constr_df.index.values
    bw     = bar_width * 0.4

    zvs_ok     = [d >= t for d, t in zip(dt_max_vals, t_dis_vals)]
    dis_colors = ['mediumseagreen' if ok else 'tomato' for ok in zvs_ok]
    dt_colors  = ['mediumseagreen' if ok else 'tomato' for ok in zvs_ok]

    ax_zvs.bar(gens_z - bw / 2, t_dis_vals,  bw,
               label='Required discharge time $t_{disch}$',
               color=dis_colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax_zvs.bar(gens_z + bw / 2, dt_max_vals, bw,
               label='Available dead-time $t_{dead}$',
               color=dt_colors, edgecolor='black', linewidth=0.5, alpha=0.55, hatch='//')
    ax_zvs.set_title("ZVS Timing: Required Discharge Time vs Available Dead-Time (ns)\n"
                     "Worst case across operating points — Best Individual per Generation\n"
                     "green: ZVS guaranteed  |  red: ZVS at risk",
                     fontsize=11)
    ax_zvs.set_xlabel("Generation", fontsize=11)
    ax_zvs.set_ylabel("Time (ns)", fontsize=11)
    ax_zvs.set_xticks(_major_ticks)
    ax_zvs.legend(fontsize=9)
    ax_zvs.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig9b.tight_layout()
    all_figs.append(fig9b)
    if SHOW_PLOTS:
        plt.show()

    # ==========================================================================
    # PLOT 17a: Min simulated dead-time window — worst case (% of Tsw/2)
    # ==========================================================================
    if 'min_dt_win_sim_pct' in constr_df.columns:
        fig_dt, ax_dt = plt.subplots(figsize=(10, 5))
        vals_dt   = constr_df['min_dt_win_sim_pct'].values
        gens_dt   = constr_df.index.values
        colors_dt = ['mediumseagreen' if v >= 15.0 else 'tomato' for v in vals_dt]
        ax_dt.bar(gens_dt, vals_dt, color=colors_dt, edgecolor='black', linewidth=0.5, width=bar_width)
        ax_dt.axhline(15.0, color='black', linestyle='--', linewidth=1.8,
                      label='Min required: 15 % (allowed_min_dt_win_pct)')
        ax_dt.set_title("Min Simulated Dead-Time Window / $(T_{sw}/2)$ — Worst Case\n"
                        "Best Individual per Generation  |  green = OK  |  red = violated",
                        fontsize=12)
        ax_dt.set_xlabel("Generation", fontsize=11)
        ax_dt.set_ylabel("% of $T_{sw}/2$", fontsize=11)
        ax_dt.set_xticks(_major_ticks)
        ax_dt.legend(fontsize=9)
        ax_dt.grid(True, axis='y', linestyle=':', alpha=0.5)
        fig_dt.tight_layout()
        all_figs.append(fig_dt)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # PLOT 17b: max_dt_win_sim and t_discharge_req per operating point
    #           — best individual per generation
    # ==========================================================================
    _dt_op_cols = [(f'P{i}_', f'P{i}_max_dt_win_sim', f'P{i}_t_discharge_req')
                   for i in range(1, num_points + 1)
                   if f'P{i}_max_dt_win_sim' in best_per_gen_df.columns]
    if _dt_op_cols:
        _prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig_dt2, ax_dt2 = plt.subplots(figsize=(11, 6))
        for _idx, (_pfx, _col_dt, _col_dis) in enumerate(_dt_op_cols):
            _c   = _prop_colors[_idx % len(_prop_colors)]
            _mkr = markers[_idx % len(markers)]
            _lbl = f"{_pfx.rstrip('_')}: {op_label(best_per_gen_df.iloc[0], _pfx, short=True)}"
            ax_dt2.plot(best_per_gen_df['generation'],
                        best_per_gen_df[_col_dt] * 1e9,
                        color=_c, marker=_mkr, linestyle='-', linewidth=2, markersize=7,
                        label=f'{_lbl} — $t_{{dead,max}}$')
            if _col_dis in best_per_gen_df.columns:
                ax_dt2.plot(best_per_gen_df['generation'],
                            best_per_gen_df[_col_dis] * 1e9,
                            color=_c, marker=_mkr, linestyle='--', linewidth=1.5, markersize=5,
                            alpha=0.75, label=f'{_lbl} — $t_{{dis,req}}$')
        ax_dt2.set_title(
            "Dead-Time Window vs. Required Discharge Time per Operating Point\n"
            "(Best Individual per Generation  —  solid: $t_{dead,max}$  |  dashed: $t_{dis,req}$)",
            fontsize=12)
        ax_dt2.set_xlabel("Generation", fontsize=11)
        ax_dt2.set_ylabel("Time (ns)", fontsize=11)
        ax_dt2.set_xticks(_major_ticks)
        ax_dt2.legend(title="Operating Points", fontsize=8, ncol=2)
        ax_dt2.grid(True, linestyle='--', alpha=0.6)
        fig_dt2.tight_layout()
        all_figs.append(fig_dt2)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # PLOT 18: Regulation error worst case (best individual per generation)
    # ==========================================================================
    if 'reg_err_max' in constr_df.columns:
        fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
        vals_reg   = constr_df['reg_err_max'].values
        gens_reg   = constr_df.index.values
        colors_reg = ['mediumseagreen' if v <= 5.0 else 'tomato' for v in vals_reg]
        ax_reg.bar(gens_reg, vals_reg, color=colors_reg, edgecolor='black', linewidth=0.5, width=bar_width)
        ax_reg.axhline(5.0, color='black', linestyle='--', linewidth=1.8, label='Limit: 5 %')
        ax_reg.set_title("Regulation Error Worst Case — Best Individual per Generation\n"
                         "(green = within limit  |  red = out of limit)", fontsize=12)
        ax_reg.set_xlabel("Generation", fontsize=11)
        ax_reg.set_ylabel("%", fontsize=11)
        ax_reg.set_xticks(_major_ticks)
        ax_reg.legend(fontsize=9)
        ax_reg.grid(True, axis='y', linestyle=':', alpha=0.5)
        fig_reg.tight_layout()
        all_figs.append(fig_reg)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # PLOT 19: Diode D1 — RMS and peak current per operating point
    #           (best individual per generation)
    # ==========================================================================
    _d1_rms = [(f'P{i}_', f'P{i}_iD1RMS') for i in range(1, num_points + 1)
               if f'P{i}_iD1RMS' in best_per_gen_df.columns]
    _d1_pk  = [(f'P{i}_', f'P{i}_iD1PK')  for i in range(1, num_points + 1)
               if f'P{i}_iD1PK'  in best_per_gen_df.columns]

    if _d1_rms or _d1_pk:
        fig_d1, (ax_d1r, ax_d1p) = plt.subplots(1, 2, figsize=(13, 5))
        fig_d1.suptitle("Diode $D_1$ Current — Best Individual per Generation", fontsize=14)

        for _idx, (_pfx, _col) in enumerate(_d1_rms):
            _lbl = f"{_pfx.rstrip('_')}: {op_label(best_per_gen_df.iloc[0], _pfx, short=True)}"
            _mkr = markers[_idx % len(markers)]
            ax_d1r.plot(best_per_gen_df['generation'], best_per_gen_df[_col],
                        marker=_mkr, linestyle='-', linewidth=2, markersize=7, label=_lbl)

        for _idx, (_pfx, _col) in enumerate(_d1_pk):
            _lbl = f"{_pfx.rstrip('_')}: {op_label(best_per_gen_df.iloc[0], _pfx, short=True)}"
            _mkr = markers[_idx % len(markers)]
            ax_d1p.plot(best_per_gen_df['generation'], best_per_gen_df[_col],
                        marker=_mkr, linestyle='-', linewidth=2, markersize=7, label=_lbl)

        ax_d1r.set_title("$I_{D1,rms}$")
        ax_d1r.set_xlabel("Generation")
        ax_d1r.set_ylabel("RMS Current (A)")
        ax_d1r.set_xticks(_major_ticks)
        ax_d1r.legend(title="Operating Points")
        ax_d1r.grid(True, linestyle='--', alpha=0.6)

        ax_d1p.set_title("$I_{D1,peak}$")
        ax_d1p.set_xlabel("Generation")
        ax_d1p.set_ylabel("Peak Current (A)")
        ax_d1p.set_xticks(_major_ticks)
        ax_d1p.legend(title="Operating Points")
        ax_d1p.grid(True, linestyle='--', alpha=0.6)

        fig_d1.tight_layout()
        all_figs.append(fig_d1)
        if SHOW_PLOTS:
            plt.show()

    # ==========================================================================
    # Save all figures to PDF
    # ==========================================================================
    with PdfPages(pdf_path) as pdf:
        for fig in all_figs:
            pdf.savefig(fig, bbox_inches='tight')
    plt.close('all')
    print(f"[INFO] PDF saved: {pdf_path}")


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================
if __name__ == '__main__':
    for _fname in RUN_LIST:
        process_and_plot(_fname)
