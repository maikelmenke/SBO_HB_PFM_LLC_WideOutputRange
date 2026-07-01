"""
Example_find_Vo_target_single.py — Convergence visualization for find_vo_target.

Runs the brentq frequency search on a single operating point with full
iteration tracking, then:
  - Prints a per-iteration table (fsw, Vo_sim, error) to the terminal.
  - Saves a 3-panel convergence plot (PDF) showing how Vo_simulated
    converges to Vo_target across optimizer iterations.

Edit OPERATING_POINT to try different conditions, including edge cases.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'mathtext.fontset': 'stix',
})
from scipy.optimize import brentq, minimize_scalar

from llc_sim.tools.find_vo_target_fsw import voltage_residual, voltage_error, ToleranceReached
from llc_sim.sim_lib.circuit_builder import build_llc_circuit
from llc_sim.sim_lib.simulation import simulate
from llc_sim.sim_lib.analysis import analyze
from llc_sim.tools.simulate_until_ss import simulate_until_steady_state

# ── Converter parameters ──────────────────────────────────────────────────────
params = {
    'Vbus':  420,
    'Cs':    4.4e-9,
    'Ls':    250e-6,
    'Lm':    893e-6,
    'n':     3.4,
    'Co':    20e-6,
    'NPTsw': 250,
}

# ── Single operating point ────────────────────────────────────────────────────
# Format: "VbusV/VoV@IoA"
# Edge case examples:
#   "420V/160V@0.94A"  -> VoMax  (above resonance, high gain)
#   "420V/70V@1.5A"    -> VoMin  (above resonance, full load)
#   "420V/70V@0.1A"    -> VoMin, light load
OPERATING_POINT = "420V/70V@1.5A"

# ── Steady-state search settings ──────────────────────────────────────────────
TOL                = 0.1    # voltage tolerance [V]
CONFIG             = 1
CYCLES_PER_BLOCK   = 15
STEADY_STATE_TOL   = 1e-3
AVG_CYCLES         = 10
STABLE_BLOCKS_REQ  = 3
RIPPLE_TOL_FACTOR  = 2.0
SIM_CYCLES         = 500    # used only for final high-fidelity sim
OUTPUT_DIR         = os.path.join("outputs", "Find_Vo_target")
SAVE_PLOTS         = True
SHOW_PLOTS         = False


# ── Parse operating-point string ──────────────────────────────────────────────
def _parse_op(op_str, base_params):
    vbus_s, rest  = op_str.split("/")
    vout_s, iout_s = rest.split("@")
    vbus    = float(vbus_s.replace("V", ""))
    vtarget = float(vout_s.replace("V", ""))
    iout    = float(iout_s.replace("A", ""))
    rload   = vtarget / iout
    return {**base_params, "Vbus": vbus, "Rload": rload}, vtarget


# ── Iteration tracker ─────────────────────────────────────────────────────────
class IterationTracker:
    """
    Wraps a residual function and logs every valid evaluation so the
    convergence history can be plotted.
    """

    def __init__(self, vtarget):
        self.vtarget = vtarget
        self.history = []   # list of {'iter', 'fsw_khz', 'vout', 'error'}

    def wrap(self, residual_fn):
        def _tracked(fsw_khz):
            try:
                res = residual_fn(fsw_khz)
                if np.isfinite(res) and abs(res) < 1e5:
                    self.history.append({
                        'iter':    len(self.history) + 1,
                        'fsw_khz': fsw_khz,
                        'vout':    self.vtarget + res,
                        'error':   abs(res),
                    })
                return res
            except ToleranceReached as e:
                self.history.append({
                    'iter':    len(self.history) + 1,
                    'fsw_khz': e.fsw_khz,
                    'vout':    e.target_val,
                    'error':   e.error,
                })
                raise
        return _tracked


# ── Terminal table printer ────────────────────────────────────────────────────
def _print_table(history, tol, vtarget, fsw_final, elapsed):
    COL = (6, 14, 14, 14)
    hdr = (
        f"{'Iter':>{COL[0]}}  "
        f"{'fsw [kHz]':>{COL[1]}}  "
        f"{'Vo_sim [V]':>{COL[2]}}  "
        f"{'Error [V]':>{COL[3]}}"
    )
    sep = "-" * len(hdr)

    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in history:
        flag = "  <-- converged" if r['error'] <= tol else ""
        print(
            f"{r['iter']:>{COL[0]}}  "
            f"{r['fsw_khz']:>{COL[1]}.4f}  "
            f"{r['vout']:>{COL[2]}.4f}  "
            f"{r['error']:>{COL[3]}.6f}"
            f"{flag}"
        )
    print(sep)
    if fsw_final is not None:
        print(f"\n  Vo_target : {vtarget:.4f} V")
        print(f"  fsw_found : {fsw_final:.6f} kHz")
        print(f"  Iterations: {len(history)}")
        print(f"  Wall time : {elapsed:.2f} s")


# ── Convergence plot ──────────────────────────────────────────────────────────
def _save_plot(history, vtarget, tol, fsw_final, op_str, out_dir):
    iters  = [r['iter']    for r in history]
    vouts  = [r['vout']    for r in history]
    errors = [r['error']   for r in history]
    freqs  = [r['fsw_khz'] for r in history]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"find_vo_target — Convergence History\n"
        f"{op_str}   |   Vtarget = {vtarget:.2f} V"
        + (f"   |   fsw = {fsw_final:.4f} kHz" if fsw_final else ""),
        fontsize=11,
    )

    # ── Panel 1: Vo_sim evolving towards Vo_target ────────────────────────────
    ax = axes[0]
    ax.plot(iters, vouts, 'o-', color='royalblue', markersize=5, linewidth=1.4,
            label="Vo_sim")
    ax.axhline(vtarget, color='crimson', linestyle='--', linewidth=1.5,
               label=f"Vo_target = {vtarget:.2f} V")
    ax.fill_between(iters, vtarget - tol, vtarget + tol,
                    color='crimson', alpha=0.10,
                    label=f"±{tol} V tolerance band")
    ax.set_ylabel("Output Voltage (V)")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_title("Simulated Vo converging to Vo_target")

    # ── Panel 2: |error| on log scale ─────────────────────────────────────────
    ax = axes[1]
    ax.semilogy(iters, errors, 's-', color='darkorange', markersize=5,
                linewidth=1.4, label="|Vo_sim - Vo_target|")
    ax.axhline(tol, color='forestgreen', linestyle=':', linewidth=1.5,
               label=f"Tolerance = {tol} V")
    ax.set_ylabel("|Error| (V)  [log scale]")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.set_title("Convergence Error")

    # ── Panel 3: switching frequency history ──────────────────────────────────
    ax = axes[2]
    ax.plot(iters, freqs, '^-', color='mediumseagreen', markersize=5,
            linewidth=1.4, label="fsw (each iter)")
    if fsw_final is not None:
        ax.axhline(fsw_final, color='purple', linestyle='--', linewidth=1.5,
                   label=f"fsw_final = {fsw_final:.4f} kHz")
    ax.set_xlabel("Optimizer Iteration")
    ax.set_ylabel("Switching Frequency (kHz)")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_title("Frequency Search History")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    safe = op_str.replace("@", "_at_").replace("/", "_out_")
    path = os.path.join(out_dir, f"convergence_{safe}.pdf")
    if SAVE_PLOTS:
        fig.savefig(path, bbox_inches='tight')
        print(f"\n[INFO] Convergence plot saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_params, Vtarget = _parse_op(OPERATING_POINT, params)
    Rload = run_params["Rload"]
    Iout  = Vtarget / Rload

    # Initial TimeStep (overridden per-frequency inside voltage_residual via NPTsw)
    _fo_hz   = 1.0 / (2 * np.pi * np.sqrt(run_params['Cs'] * run_params['Ls']))
    TimeStep = (1.0 / _fo_hz) / run_params['NPTsw']

    # Search bounds (mirrors find_vo_target logic)
    Voafo       = run_params["Vbus"] / (2 * run_params["n"])
    f1_khz      = (1 / (2 * np.pi * np.sqrt(run_params["Cs"] *
                    (run_params["Ls"] + run_params["Lm"])))) / 1e3
    fo_khz      = (1 / (2 * np.pi * np.sqrt(run_params["Cs"] *
                    run_params["Ls"]))) / 1e3
    fsw_max_khz = run_params.get('fsw_max', 10 * fo_khz * 1e3) / 1e3

    if Vtarget >= Voafo:
        low, high = f1_khz + 1.0, fo_khz + 1.0
    else:
        low, high = fo_khz - 1.0, fsw_max_khz

    print(f"\n{'='*65}")
    print(f"  Operating point : {OPERATING_POINT}")
    print(f"  Vtarget         : {Vtarget:.2f} V   Rload = {Rload:.3f} Ohm   Io = {Iout:.3f} A")
    print(f"  Voafo           : {Voafo:.2f} V   (Vbus / 2n)")
    print(f"  f1              : {f1_khz:.3f} kHz   fo = {fo_khz:.3f} kHz")
    print(f"  Search bounds   : [{low:.3f}, {high:.3f}] kHz")
    print(f"  Tolerance       : {TOL} V")
    print(f"{'='*65}\n")

    # Build residual function with tracking
    _res_kwargs = dict(
        params=run_params,
        Vtarget=Vtarget,
        tol=TOL,
        SimCycles=SIM_CYCLES,
        TimeStep=TimeStep,
        Full_Arr=False,
        cycles_per_block=CYCLES_PER_BLOCK,
        steady_state_tol=STEADY_STATE_TOL,
        avg_cycles=AVG_CYCLES,
        stable_blocks_required=STABLE_BLOCKS_REQ,
        config=CONFIG,
        ripple_tol_factor=RIPPLE_TOL_FACTOR,
    )
    residual_fn = partial(voltage_residual, **_res_kwargs)

    tracker  = IterationTracker(Vtarget)
    tracked  = tracker.wrap(residual_fn)

    # brentq search with iteration logging
    fsw_khz = None
    t0 = time.perf_counter()

    try:
        fsw_khz, _ = brentq(tracked, low, high, xtol=1e-3, maxiter=100,
                             full_output=True)
        print(f"[INFO] brentq converged  ->  fsw = {fsw_khz:.6f} kHz")

    except ToleranceReached as e:
        fsw_khz = e.fsw_khz
        print(f"[INFO] Tolerance met     ->  fsw = {fsw_khz:.6f} kHz  "
              f"(Vout = {e.target_val:.4f} V, err = {e.error:.4g} V)")

    except ValueError as exc:
        # Vtarget outside the achievable gain range -> fall back to minimize_scalar
        print(f"[WARNING] brentq could not bracket root ({exc}). "
              f"Falling back to minimize_scalar.")
        abs_kwargs = {**_res_kwargs}
        abs_fn = partial(voltage_error, **{k: v for k, v in abs_kwargs.items()})
        abs_tracker  = IterationTracker(Vtarget)

        def _abs_tracked(fsw_khz_):
            try:
                err = abs_fn(fsw_khz_)
                if np.isfinite(err) and err < 1e5:
                    abs_tracker.history.append({
                        'iter':    len(abs_tracker.history) + 1,
                        'fsw_khz': fsw_khz_,
                        'vout':    Vtarget + err,   # err is |residual|, sign unknown
                        'error':   err,
                    })
                return err
            except ToleranceReached as e2:
                abs_tracker.history.append({
                    'iter':    len(abs_tracker.history) + 1,
                    'fsw_khz': e2.fsw_khz,
                    'vout':    e2.target_val,
                    'error':   e2.error,
                })
                raise

        try:
            res = minimize_scalar(_abs_tracked, bounds=(low, high),
                                  method='bounded',
                                  options={'xatol': 1e-3, 'maxiter': 100})
            fsw_khz = float(res.x)
            tracker.history.extend(abs_tracker.history)
            print(f"[INFO] minimize_scalar best  ->  fsw = {fsw_khz:.6f} kHz  "
                  f"|err| ~ {res.fun:.4g} V")
        except ToleranceReached as e2:
            fsw_khz = e2.fsw_khz
            tracker.history.extend(abs_tracker.history)
            print(f"[INFO] Tolerance met (fallback)  ->  fsw = {fsw_khz:.6f} kHz")

    elapsed = time.perf_counter() - t0

    # Print per-iteration table
    if tracker.history:
        _print_table(tracker.history, TOL, Vtarget, fsw_khz, elapsed)
    else:
        print("[WARNING] No iteration data was captured.")

    # Save convergence plot
    if tracker.history:
        _save_plot(tracker.history, Vtarget, TOL, fsw_khz,
                   OPERATING_POINT, OUTPUT_DIR)
    else:
        print("[WARNING] No data to plot.")
