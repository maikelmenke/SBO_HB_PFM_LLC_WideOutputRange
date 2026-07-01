import sys
import os
import time
from datetime import datetime

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from LLC_problem_op_iR import LLC_Problem_OP_iR, SS_CONFIG
from llc_sim.tools.simulate_until_ss import SIMULATOR_OPTIONS


class _Tee:
    """Mirrors stdout to a log file. Use silent=True to suppress terminal output."""
    def __init__(self, filepath, encoding='utf-8', silent=False):
        self._terminal = sys.stdout
        self._file = open(filepath, 'w', encoding=encoding, buffering=1)
        self._silent = silent

    def write(self, message):
        if not self._silent:
            self._terminal.write(message)
        self._file.write(message)

    def flush(self):
        if not self._silent:
            self._terminal.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._terminal
        self._file.close()

    def fileno(self):
        return self._terminal.fileno()


# ==============================================================================
# Optimization parameters
# ==============================================================================
POP_SIZE = 10   # population size
N_GEN    = 10    # number of generations

# High-power operating points – used for fitness evaluation.
# Fitness = mean(iR_rms across all OPHP) + std(iR_rms across all OPHP)
op_hp_points = [
    "420V/160V@0.95A",
    "420V/100V@1.5A",
    "420V/70V@1.5A",
]

# Light-load operating point – used to verify fsw_max regulation margin.
# The LLC must produce Vo < Vo_LL when operating at fsw_max with this load.
op_ll = "420V/70V@0.1A"

base_params = {
    'Vbus':                   420,    # Bus voltage [V]
    'Co':                     10e-6,  # Output capacitance [F]
    'dead_time':              100e-9, # Design dead time [s] — NOT USED
    'NPTsw':                  300,    # Simulation points per switching period
    'fsw_min':                75e3,   # Minimum switching frequency [Hz]
    'fsw_max':                500e3,  # Maximum switching frequency [Hz]
    'Bmax':                   0.22,   # Maximum flux density [T] — NOT USED
    'dead_time_min_ic':       100e-9, # Minimum dead time the IC can program [s]  (G[1] lower bound)
    'dt_max_pct':             5.0,   # Maximum dead time as % of Ts/2 — sets the dead-time window (G[1])
    'allowed_min_dt_win_pct': 7.5,   # Min required dead-time window as % of Ts/2 (G[2] user limit)
}

# --- RESONANT FREQUENCY (optional) ---
# Set fo_fixed to a value in Hz to fix fo during optimization.
# Set fo_fixed = None to leave fo as a free variable.
fo_fixed = 150e3   # e.g. 150e3  →  fo = 150 kHz fixed
# --------------------------------------


if __name__ == '__main__':
    log_dir  = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'log_ga_{datetime.now().strftime("%Y%m%d_%H%M")}.txt')
    tee = _Tee(log_path, silent=True)
    sys.stdout = tee

    try:
        problem = LLC_Problem_OP_iR(
            base_params, op_hp_points, op_ll, POP_SIZE,
            fo_fixed=fo_fixed,
            verbose=False,
        )

        # ------------------------------------------------------------------
        # Classic GA operators for continuous variables
        #   Selection : binary tournament (pressure=2)
        #   Crossover : SBX eta=15, prob=0.9  — tighter than NSGA-II default,
        #               keeps offspring close to parents (good for expensive eval)
        #   Mutation  : polynomial mutation eta=20, prob=1/n_var
        #   Elitism   : implicit — pymoo GA keeps the best individual across
        #               generations via survivor selection
        # ------------------------------------------------------------------
        n_var          = problem.n_var
        sbx_eta        = 15
        sbx_prob       = 0.9
        pm_eta         = 15
        pm_prob        = 1.0 / n_var
        tourn_pressure = 2

        algorithm = GA(
            pop_size=POP_SIZE,
            selection=TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=tourn_pressure),
            crossover=SBX(eta=sbx_eta, prob=sbx_prob),
            mutation=PM(eta=pm_eta, prob=pm_prob),
            eliminate_duplicates=True,
        )

        print(f"\n=============================================")
        print(f" --- Starting GA Optimization of iR RMS ---")
        print(f" -> Population size     : {POP_SIZE} individuals")
        print(f" -> Generations         : {N_GEN}")
        print(f" -> Crossover           : SBX (eta={sbx_eta}, prob={sbx_prob})")
        print(f" -> Mutation            : Polynomial (eta={pm_eta}, prob=1/{n_var})")
        print(f" -> Selection           : Tournament (pressure={tourn_pressure})")
        print(f" -> HP Operating Points : {op_hp_points}")
        print(f" -> LL Operating Point  : {op_ll}")
        print(f" -> fsw range           : {base_params['fsw_min']/1e3:.0f} – {base_params['fsw_max']/1e3:.0f} kHz")
        print(f" -> fo_fixed            : {'free' if fo_fixed is None else f'{fo_fixed/1e3:.0f} kHz'}")
        print(f" -> Total evaluations   : {POP_SIZE * N_GEN} (approx)")
        print(f"=============================================\n")

        print(f"=============================================")
        print(f" [CONFIG] Run parameters – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Log file : {log_path}")
        print(f"---------------------------------------------")
        def _fmt_param(k, v):
            """Human-readable value for known unit types."""
            _time_keys = {'dead_time', 'dead_time_min_ic'}
            _freq_keys = {'fsw_min', 'fsw_max'}
            _pct_keys  = {'dt_max_pct', 'allowed_min_dt_win_pct'}
            if k in _time_keys:   return f"{v*1e9:.0f} ns"
            if k in _freq_keys:   return f"{v/1e3:.0f} kHz"
            if k in _pct_keys:    return f"{v:.1f} %"
            return str(v)

        print(f" Circuit / design space")
        for k, v in base_params.items():
            print(f"   {k:<25} = {_fmt_param(k, v)}")
        print(f"   {'fo_fixed':<25} = {'free' if fo_fixed is None else f'{fo_fixed/1e3:.0f} kHz'}")
        print(f"---------------------------------------------")
        print(f" Steady-state detection (SS_CONFIG)")
        for k, v in SS_CONFIG.items():
            print(f"   {k:<25} = {v}")
        print(f"---------------------------------------------")
        print(f" Ngspice solver (SIMULATOR_OPTIONS)")
        for k, v in SIMULATOR_OPTIONS.items():
            print(f"   {k:<12} = {v}")
        print(f"=============================================\n")

        start_time = time.time()

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", N_GEN),
            seed=1,
            verbose=False,
        )

        elapsed = time.time() - start_time
        horas, rem = divmod(elapsed, 3600)
        minutos, segundos = divmod(rem, 60)

        print(f"\n========================================")
        print(f"TOTAL TIME: {int(horas)}h {int(minutos)}m {segundos:.1f}s")
        print(f"========================================")

        csv_filename = "LLC_GA_OPT.csv"
        problem.save_results_to_csv(csv_filename)

        backup_path = os.path.join(log_dir, 'LLC_OP_IR_backup.csv')
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print("-> Temporary backup file successfully deleted.")

        if res.X is None:
            print(f"\n[WARNING] No feasible solution found after {N_GEN} generations.")
            print(f"          All individuals were infeasible or penalised.")
        else:
            if fo_fixed is not None:
                n, cs, k = res.X
                fo = fo_fixed
            else:
                n, cs, fo, k = res.X
            ls = 1 / ((2 * np.pi * fo) ** 2 * cs)
            lm = k * ls

            print(f"\n========================================")
            print(f"BEST SOLUTION FOUND:")
            print(f"========================================")
            print(f"Robust Score (iR_mean + iR_std): {res.F[0]:.4f} A")
            print(f"Components: n={n:.1f}, Ls={ls*1e6:.1f}uH, "
                  f"Cs={cs*1e9:.1f}nF, Lm={lm*1e6:.1f}uH")

            # Active constraints: G[0]=ids_off, G[1]=t_dis vs dt_lower,
            #                     G[2]=max_dt_win_sim_pct vs allowed_min_dt_win_pct.
            ids_off_min_best = -res.G[0]   # [A], positive = ZVS OK

            print(f"\n--- Active Constraints (normalised G in brackets) ---")
            print(f"Min ids_off (ZVS)                  : {ids_off_min_best:.3f} A  (>0 OK  | G[0]={res.G[0]:.4f})")
            print(f"t_dis vs dt_lower (norm)           : G[1]={res.G[1]:.4f}  (<=0 OK)")
            print(f"max_dt_win_sim_pct >= min required (norm): G[2]={res.G[2]:.4f}  (<=0 OK)")
            print(f"[Note] fsw range, regulation error and OPLL margin are logged to CSV (not penalised).")
            print(f"========================================\n")

    finally:
        tee.close()
