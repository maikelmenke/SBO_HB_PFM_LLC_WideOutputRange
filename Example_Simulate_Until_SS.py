import time
import numpy as np
import matplotlib.pyplot as plt

# IEEE double-column full text width [in]
IEEE_W = 7.16

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',      # Times-like math, consistent with body font
    'font.size':          14,
    'axes.titlesize':     14,
    'axes.labelsize':     8,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'legend.fontsize':    8,
    'figure.dpi':         150,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':     0.6,
    'grid.linewidth':     0.4,
    'lines.linewidth':    1.0,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
})

import os

from llc_sim.sim_lib.circuit_builder import build_llc_circuit
from llc_sim.sim_lib.analysis import analyze
from llc_sim.sim_lib.simulation import simulate
from llc_sim.tools.simulate_until_ss import simulate_until_steady_state, SIMULATOR_OPTIONS

# ==========================================
# 1. Circuit parameters
# ==========================================
params = {
    'Rload': 170.21,
    'Cs': 4.4e-9,
    'Ls': 250e-6,
    'Lm': 893e-6,
    'n': 3.4,
    'Co': 20e-6,
    'Vbus': 420,
    'fsw': 85.5e3,
    'NPTsw': 250,
}

SAVE_PLOTS = True
SHOW_PLOTS = False

# Steady-state detection parameters
ss_params = {
    'MaxCycles':              4000,
    'cycles_per_block':       30,
    'SteadyStateTol':         1e-3,
    'avg_cycles':             20,
    'stable_blocks_required': 2,
    'ripple_tol_factor':      2,
}

# ==========================================
# 2. Derived quantities
# ==========================================
fsw      = params['fsw']
Tsw      = 1.0 / fsw
TimeStep = Tsw / params['NPTsw']
cpb      = ss_params['cycles_per_block']

if __name__ == '__main__':
    print("=" * 60)
    print(f"  LLC — Simulation Until Steady State")
    print(f"  fsw       = {fsw/1e3:.1f} kHz")
    print(f"  TimeStep  = {TimeStep*1e9:.1f} ns  ({params['NPTsw']} pts/period)")
    print(f"  block     = {cpb} cycles  |  avg = {ss_params['avg_cycles']} cycles")
    print("=" * 60)

    # ==========================================
    # 3. Run simulate_until_steady_state
    # ==========================================
    circuit_ss = build_llc_circuit(params, config=1)

    t0 = time.perf_counter()

    analysis_last, cycles_to_stable, block_history, block_waveforms = simulate_until_steady_state(
        circuit=circuit_ss,
        fsw=fsw,
        TimeStep=TimeStep,
        MaxCycles=ss_params['MaxCycles'],
        cycles_per_block=cpb,
        SteadyStateTol=ss_params['SteadyStateTol'],
        avg_cycles=ss_params['avg_cycles'],
        stable_blocks_required=ss_params['stable_blocks_required'],
        SteadyStateNode='vo',
        save_all=True,
        ripple_tol_factor=ss_params['ripple_tol_factor'],
        return_block_history=True,
        return_waveforms=True,
    )

    elapsed = time.perf_counter() - t0

    if analysis_last is None:
        print("\n[ERROR] Simulation failed or did not converge within MaxCycles.")
        exit(1)

    if cycles_to_stable >= ss_params['MaxCycles']:
        print(f"\n[WARNING] Limit of {ss_params['MaxCycles']} cycles reached.")
    else:
        print(f"\n[OK] Steady state reached at {cycles_to_stable} cycles ({elapsed*1e3:.1f} ms)")

    # ==========================================
    # 4. Metrics
    # ==========================================
    results = analyze(
        analysis=analysis_last,
        fsw=fsw,
        TimeStep=TimeStep,
        SimCycles=cycles_to_stable,
        return_arrays=False,
        cycles_to_analyze=ss_params['avg_cycles'],
    )

    print("\n--- Steady-State Metrics ---")
    print(f"  Vout (avg)        : {results['voutAVG']:.4f} V")
    print(f"  Iout (avg)        : {results['ioutAVG']:.4f} A")
    print(f"  Pout              : {results['voutAVG'] * results['ioutAVG']:.2f} W")
    print(f"  Ir  (RMS / Peak)  : {results['iRRMS']:.4f} A / {results['iRPK']:.4f} A")
    print(f"  Vcs (Peak)        : {results['vCsPK']:.4f} V")
    print(f"  Max dead time     : {results['max_dt_win_sim']*1e9:.1f} ns")

    # ==========================================
    # 5. Waveforms from block-by-block simulation
    # ==========================================
    def _bw(key):
        for k in (key, key.lower()):
            v = block_waveforms.get(k)
            if v is not None:
                return np.asarray(v, dtype=float)
        return None

    t_bw   = np.asarray(block_waveforms.get('time', []), dtype=float)
    vo_bw  = _bw('vo')
    ils_bw = _bw('VLs_plus')

    # ==========================================
    # 6. IC=0 reference simulation
    # ==========================================
    # IC=0 runs for the same number of cycles as block-by-block so both
    # waveforms share the same time axis.
    ic0_cycles = cycles_to_stable
    print(f"Running IC=0 for {ic0_cycles} cycles ({ic0_cycles*Tsw*1e3:.1f} ms)  …")

    circuit_full = build_llc_circuit(params, config=1)
    # TSTEP-resolution output (plotwinsize omitted): block boundaries land at
    # exact multiples of TSTEP so np.interp gives accurate Poincaré values.
    ref_solver_opts = {k: v for k, v in SIMULATOR_OPTIONS.items() if k != 'plotwinsize'}
    ref_solver_opts['maxord'] = 2
    analysis_full = simulate(circuit_full, fsw, TimeStep, SimCycles=ic0_cycles,
                             solver_options=ref_solver_opts, max_time=2 * TimeStep)

    def _safe(analysis, key):
        try:
            return np.asarray(analysis[key], dtype=float)
        except Exception:
            return None

    t_full   = np.asarray(analysis_full.time, dtype=float)
    vo_full  = _safe(analysis_full, 'vo')
    ils_full = _safe(analysis_full, 'VLs_plus')

    # ── Block boundaries and per-block averages ───────────────────────────
    n_blocks    = len(block_history)
    block_times = [0.0] + [b['t_end_s'] for b in block_history]

    block_avg_vo   = [b['avg_vo'] for b in block_history]
    avg_half_win   = ss_params['avg_cycles'] / (2.0 * fsw)
    bh_marker_t_ms = [(b['t_end_s'] - avg_half_win) * 1e3 for b in block_history]

    # ==========================================
    # FIGURE 1 — Transient overview: block-by-block vs IC=0
    # ==========================================
    fig1, (ax_vo, ax_crit) = plt.subplots(2, 1, figsize=(IEEE_W, 5.5), sharex=True)

    BG = ['#eef2ff', '#fff7ee']
    for k in range(n_blocks):
        t0_k = block_times[k]     * 1e3
        t1_k = block_times[k + 1] * 1e3
        ax_vo.axvspan(t0_k, t1_k, alpha=0.45, color=BG[k % 2], zorder=0)
        ax_vo.axvline(t1_k, color='#888888', lw=0.6, ls='--', alpha=0.5, zorder=1)

    t_bw_ms = t_bw * 1e3 if t_bw.size > 0 else None
    if vo_bw is not None and t_bw_ms is not None:
        ax_vo.plot(t_bw_ms, vo_bw, color='#1f77b4', lw=0.9, zorder=3,
                   label=r'$V_{\mathrm{o}}$ — block-by-block')
    if vo_full is not None:
        ax_vo.plot(t_full * 1e3, vo_full, color='#d62728', lw=0.7, ls='--',
                   alpha=0.75, zorder=2, label=r'$V_{\mathrm{o}}$ — IC=0 (reference)')
    ax_vo.axhline(results['voutAVG'], color='black', lw=1.1, ls=':', alpha=0.75,
                  label=f"SS avg: {results['voutAVG']:.2f} V")

    for t_m, avg in zip(bh_marker_t_ms, block_avg_vo):
        ax_vo.plot(t_m, avg, marker='x', color='#1f4e8c', ms=5, zorder=5)
        ax_vo.annotate(f'{avg:.2f} V',
                       xy=(t_m, avg), xytext=(0, 5), textcoords='offset points',
                       fontsize=8.5, ha='center', color='#1f4e8c')

    ax_vo.set_ylabel(r'$V_{\mathrm{o}}$ (V)')
    ax_vo.legend(loc='lower right', fontsize=9)
    ax_vo.grid(True, ls=':', alpha=0.6)

    tol1 = ss_params['SteadyStateTol']
    tol3 = ss_params['SteadyStateTol'] * ss_params['ripple_tol_factor']

    bh_t   = [b['t_end_s'] * 1e3 for b in block_history]
    bh_avg = [b['avg_error']    for b in block_history]
    bh_end = [b['end_error']    for b in block_history]
    bh_rip = [b['ripple_error'] for b in block_history]

    def _nn(v): return [x if x is not None else float('nan') for x in v]

    ax_crit.semilogy(bh_t, _nn(bh_avg), 'o-', color='#1f77b4', lw=1.0, ms=4,
                     label=r'Criterion 1 — mean $|\Delta\overline{V}|/\overline{V}$')
    ax_crit.semilogy(bh_t, _nn(bh_end), 's-', color='#ff7f0e', lw=1.0, ms=4,
                     label=r'Criterion 2 — Poincaré $|\Delta V_{\mathrm{end}}|/V_{\mathrm{end}}$')
    ax_crit.semilogy(bh_t, _nn(bh_rip), '^-', color='#2ca02c', lw=1.0, ms=4,
                     label=r'Criterion 3 — ripple $|\Delta V_{\mathrm{pp}}|/\overline{V}$')

    ax_crit.axhline(tol1, color='#1f77b4', ls='--', lw=0.9, alpha=0.6,
                    label=f'Tol 1&2 = {tol1:.0e}')
    ax_crit.axhline(tol3, color='#2ca02c', ls='--', lw=0.9, alpha=0.6,
                    label=f'Tol 3 = {tol3:.0e}')

    for b in block_history:
        if b['ok']:
            ax_crit.axvspan((b['t_end_s'] - cpb * Tsw) * 1e3, b['t_end_s'] * 1e3,
                            alpha=0.15, color='green', zorder=0)

    ax_crit.set_ylabel('Relative error')
    ax_crit.set_xlabel('Time (ms)')
    ax_crit.legend(loc='upper right', fontsize=7.5, ncol=2)
    ax_crit.grid(True, ls=':', alpha=0.6)
    ax_crit.set_ylim(bottom=1e-6)
    ax_crit.set_xlim(0, cycles_to_stable * Tsw * 1e3)

    fig1.suptitle(
        f'LLC — Block-by-block vs. IC=0  |  '
        f'$f_{{\\mathrm{{sw}}}}$ = {fsw/1e3:.1f} kHz  |  '
        f'{cycles_to_stable} cycles  |  block = {cpb} cycles',
        fontsize=12,
    )
    plt.tight_layout()
    plot_dir = os.path.join('outputs', 'Find_steady_state')
    os.makedirs(plot_dir, exist_ok=True)
    if SAVE_PLOTS:
        fig1.savefig(os.path.join(plot_dir, 'transient_overview.pdf'))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig1)

    # ==========================================
    # FIGURE 2 — Last cycles: detailed waveforms at steady state
    # ==========================================
    N_LAST = ss_params['avg_cycles']
    t_last    = np.asarray(analysis_last.time, dtype=float)
    mask_last = t_last >= (t_last[-1] - N_LAST * Tsw)
    t_us      = t_last[mask_last] * 1e6

    vo_last  = _safe(analysis_last, 'vo')
    ils_last = _safe(analysis_last, 'VLs_plus')
    ilm_last = _safe(analysis_last, 'VLm_plus')

    vlm_last = None
    if ilm_last is not None:
        vlm_full_arr = params['Lm'] * np.gradient(ilm_last, t_last)
        vlm_last     = vlm_full_arr[mask_last]

    sigs2 = [
        (vo_last[mask_last]  if vo_last  is not None else None, '#d62728', r'$V_{\mathrm{o}}$',  'V (V)'),
        (ils_last[mask_last] if ils_last is not None else None, '#9467bd', r'$I_{\mathrm{Ls}}$', 'A'),
        (ilm_last[mask_last] if ilm_last is not None else None, '#2ca02c', r'$I_{\mathrm{Lm}}$', 'A'),
        (vlm_last,                                               '#ff7f0e', r'$V_{\mathrm{Lm}}$', 'V (V)'),
    ]
    sigs2 = [(y, c, lbl, ylbl) for y, c, lbl, ylbl in sigs2 if y is not None]

    fig2, axes2 = plt.subplots(len(sigs2), 1, figsize=(IEEE_W, 2.5 * len(sigs2)), sharex=True)
    if len(sigs2) == 1:
        axes2 = [axes2]

    fig2.suptitle(
        f'LLC — Last {N_LAST} cycles  |  '
        f'$f_{{\\mathrm{{sw}}}}$ = {fsw/1e3:.1f} kHz  |  '
        f'$V_{{\\mathrm{{o}}}}$ = {results["voutAVG"]:.2f} V',
        fontsize=12,
    )

    for ax, (y, color, label, ylabel) in zip(axes2, sigs2):
        ax.plot(t_us, y, color=color, lw=0.9, label=label)
        ax.axhline(0, color='black', lw=0.5, alpha=0.35)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, ls=':', alpha=0.6)

    axes2[-1].set_xlabel('Time (µs)')
    axes2[-1].set_xlim(t_us[0], t_us[-1])
    plt.tight_layout()
    if SAVE_PLOTS:
        fig2.savefig(os.path.join(plot_dir, 'steady_state_waveforms.pdf'))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig2)
