import os
import sys
import io
import tempfile
import logging
import time
import numpy as np
import pandas as pd
import re
import concurrent.futures
from pymoo.core.problem import Problem

from llc_sim.tools.edge_cases import test_edge_cases
from llc_sim.tools.find_peak_ss import find_maximum_output_voltage_ss
from llc_sim.sim_lib.circuit_builder import build_llc_circuit
from llc_sim.tools.simulate_until_ss import simulate_until_steady_state

# ==============================================================================
# GLOBAL PYSPICE LOGGER CONFIGURATION
# Suppress only the "ngspice version not supported" warning from PySpice,
# while allowing actual ngspice error/debug messages to pass through.
# ==============================================================================
class _SuppressVersionWarning(logging.Filter):
    def filter(self, record):
        return 'version' not in record.getMessage().lower()

logging.getLogger('PySpice').addFilter(_SuppressVersionWarning())

# Steady-state detection parameters shared by all simulations in the GA.
# Defined at module level so main_ga can import and log them for audit.
SS_CONFIG = {
    'MaxCycles':              4000,
    'cycles_per_block':       30,
    'SteadyStateTol':         5e-3,
    'avg_cycles':             20,
    'stable_blocks_required': 2,
    'SteadyStateNode':        'vo',
}


# ==============================================================================
# GLOBAL EVALUATION FUNCTION (Runs on an isolated processor core)
# ==============================================================================
def evaluate_single_individual(args):
    (x, base_params, op_hp_points, op_ll, fsw_min, fsw_max, fo_fixed,
     allowed_min_dt_win_pct, curr_gen, curr_ind, eval_id, verbose) = args

    # ------------------------------------------------------------------
    # Capture all output produced by this worker (Python prints + C-level).
    # Python-level prints (f_vtarget, fast_ss, etc.) → io.StringIO.
    # C-level writes (ngspice DLL to fd 1/2)           → temp file via os.dup2.
    # Both are collected in the finally block and appended to log[].
    #
    # IMPORTANT: set PYTHONUTF8=1 before os.dup2 so that any sub-processes
    # spawned by test_edge_cases inherit UTF-8 as their default encoding.
    # Without this, sub-processes on Windows inherit the binary temp file on
    # fd 1 and Python opens it with 'charmap', causing UnicodeEncodeError on
    # characters like ≤ in f_vtarget prints.
    # ------------------------------------------------------------------
    os.environ['PYTHONUTF8'] = '1'

    _py_cap   = io.StringIO()
    _tmp_path = None
    _saved_fd1 = _saved_fd2 = None
    _saved_stdout, _saved_stderr = sys.stdout, sys.stderr
    sys.stdout = _py_cap
    sys.stderr = _py_cap
    try:
        _tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wlog')
        _tmp_path = _tmp.name
        _saved_fd1 = os.dup(1)
        _saved_fd2 = os.dup(2)
        os.dup2(_tmp.fileno(), 1)
        os.dup2(_tmp.fileno(), 2)
        _tmp.close()
    except Exception:
        pass

    if fo_fixed is not None:
        n_val, cs_val, k_val = x[0], x[1], x[2]
        fo_val = fo_fixed
    else:
        n_val, cs_val, fo_val, k_val = x[0], x[1], x[2], x[3]
    ls_val = 1 / ((2 * np.pi * fo_val)**2 * cs_val)
    lm_val = k_val * ls_val

    params = base_params.copy()
    params.update({'n': n_val, 'Cs': cs_val, 'Ls': ls_val, 'Lm': lm_val})

    entry = {
        'generation': curr_gen, 'individual': curr_ind, 'eval_id': eval_id, 'status': "Processing...",
        'n': n_val, 'Cs': cs_val, 'fo_res': fo_val, 'k_ratio': k_val, 'Ls_calc': ls_val, 'Lm_calc': lm_val
    }

    t0_total = time.perf_counter()
    log = []

    try:
        # ======================================================================
        # Shared resonant parameters
        # ======================================================================
        f1 = (1 / (2 * np.pi * np.sqrt(cs_val * (ls_val + lm_val))))
        ss_config = SS_CONFIG

        # ======================================================================
        # 1.1 – Analytical gain check: Vofo = Vbus / (2·n)
        # If Vofo > max(Vo for all OPHP), all operating points sit in the
        # above-resonance region (gain < 1) and can always be supplied.
        # Otherwise, per-OPHP peak search (step 1.2) is required.
        # ======================================================================
        Vofo = params['Vbus'] / (2 * n_val)
        max_vo_hp = max(
            float([p for p in re.split(r'[V/@A]', op) if p][1])
            for op in op_hp_points
        )
        entry['Vofo'] = Vofo
        log.append(f"[INFO]    [1.1 Vofo      ] Vofo={Vofo:.1f}V, max(Vo_HP)={max_vo_hp:.1f}V"
                   f" -> {'skip 1.2' if Vofo > max_vo_hp else 'run 1.2'}")

        # Default minimum search frequency for every HP OP: fo (series resonance).
        # For OPs that require below-resonance gain, this is updated in step 1.2
        # to f_peak (the simulated peak-gain frequency), ensuring test_edge_cases
        # only searches in the inductive region where ZVS is achievable.
        fsw_min_per_case = {op: fo_val for op in op_hp_points}

        if Vofo <= max_vo_hp:
            # ======================================================================
            # 1.2 – Per-OPHP peak voltage search
            # Only executed for operating points that require gain > 1 (Vo > Vofo).
            # For each such OPHP:
            #   a) Find the peak output voltage and its frequency via simulation.
            #   b) If v_peak < Vo_required → penalise immediately.
            #   c) If the peak sits below fsw_min, simulate at fsw_min to check
            #      whether the converter can still reach Vo_required within the
            #      allowed frequency range.
            # ======================================================================
            for i_hp, op_str in enumerate(op_hp_points):
                parts = [p for p in re.split(r'[V/@A]', op_str) if p]
                vo_op = float(parts[1])
                io_op = float(parts[2])
                rload_op = vo_op / io_op

                if vo_op <= Vofo:
                    log.append(f"[INFO]    [1.2 OPHP {i_hp+1}   ] {op_str}: Vo={vo_op:.1f}V <= Vofo={Vofo:.1f}V -> skip")
                    continue

                t0_12x = time.perf_counter()
                params_hp = params.copy()
                params_hp['Rload'] = rload_op

                f_peak, v_peak, _, _, _, _ = find_maximum_output_voltage_ss(
                    params=params_hp, f_min=f1 + 1e3, f_max=fo_val + 1e3, config=1,
                    ss_params=ss_config, verbose=False
                )
                t_12x = time.perf_counter() - t0_12x
                log.append(f"[INFO]    [1.2 OPHP {i_hp+1}   ] {op_str}: Vpk={v_peak:.1f}V @ fpk={f_peak/1e3:.1f}kHz"
                           f" (need>{vo_op:.1f}V) | t={t_12x:.1f}s")

                # Use the simulated peak-gain frequency as the minimum search
                # bound for this OP in step 3.1, restricting the solver to the
                # inductive (ZVS) side of the gain curve.
                fsw_min_per_case[op_str] = f_peak

                if v_peak < vo_op * 1.05:
                    raise ValueError(
                        f"Insufficient peak gain for {op_str} (Vpk={v_peak:.1f}V < {vo_op:.1f}V × 1.05)"
                    )

                if f_peak < fsw_min:
                    # Peak is in the forbidden zone (below fsw_min).
                    # The highest achievable voltage within the allowed range
                    # is Vo @ fsw_min; verify it still meets the requirement.
                    t0_fmin = time.perf_counter()
                    params_fmin = params_hp.copy()
                    params_fmin['fsw'] = fsw_min
                    time_step_fmin = (1.0 / fsw_min) / params_fmin.get('NPTsw', 400)

                    circuit_fmin = build_llc_circuit(params_fmin, config=1)
                    analysis_fmin, _ = simulate_until_steady_state(
                        circuit_fmin, fsw_min, time_step_fmin,
                        MaxCycles=ss_config['MaxCycles'],
                        cycles_per_block=ss_config['cycles_per_block'],
                        SteadyStateTol=ss_config['SteadyStateTol'],
                        avg_cycles=ss_config['avg_cycles'],
                        stable_blocks_required=ss_config['stable_blocks_required'],
                        SteadyStateNode='vo',
                    )
                    if analysis_fmin is None:
                        raise ValueError(f"Simulation failure at fsw_min for {op_str}")

                    try:
                        _vo_arr = np.array(analysis_fmin['vo'])
                        _t_arr  = np.array(analysis_fmin.time)
                        _periodo = 1.0 / fsw_min
                        _idx = np.searchsorted(_t_arr, _t_arr[-1] - ss_config['avg_cycles'] * _periodo)
                        vo_at_fsw_min = float(np.mean(_vo_arr[_idx:]))
                    except Exception:
                        raise ValueError(f"Vo extraction failure at fsw_min for {op_str}")

                    t_fmin = time.perf_counter() - t0_fmin
                    log.append(f"[INFO]    [1.2 OPHP {i_hp+1}   ] fsw_min check: Vo={vo_at_fsw_min:.1f}V"
                               f" @ {fsw_min/1e3:.0f}kHz (need>{vo_op:.1f}V) | t={t_fmin:.1f}s")

                    if vo_at_fsw_min < vo_op:
                        raise ValueError(
                            f"Below fsw_min needed for {op_str} "
                            f"(Vo_fmin={vo_at_fsw_min:.1f}V < {vo_op:.1f}V)"
                        )

        # ======================================================================
        # 2.1 – OPLL: verify LLC can regulate down to minimum voltage at fsw_max
        # Simulate at fsw_max with the OPLL load. If Vo @ fsw_max < Vo_OPLL, the
        # converter can reach this voltage within the allowed frequency range.
        # ======================================================================
        t0_21 = time.perf_counter()
        parts_ll = [p for p in re.split(r'[V/@A]', op_ll) if p]
        vo_ll = float(parts_ll[1])
        io_ll = float(parts_ll[2])
        rload_ll = vo_ll / io_ll

        params_21 = params.copy()
        params_21['Rload'] = rload_ll
        params_21['fsw']   = fsw_max

        time_step_21 = (1.0 / fsw_max) / params_21.get('NPTsw', 400)
        _maxord_21 = 2

        circuit_21 = build_llc_circuit(params_21, config=1)
        analysis_21, _ = simulate_until_steady_state(
            circuit_21, fsw_max, time_step_21,
            MaxCycles=ss_config['MaxCycles'], cycles_per_block=ss_config['cycles_per_block'],
            SteadyStateTol=ss_config['SteadyStateTol'], avg_cycles=ss_config['avg_cycles'],
            stable_blocks_required=ss_config['stable_blocks_required'], SteadyStateNode='vo',
            maxord=_maxord_21,
        )

        if analysis_21 is None:
            raise ValueError(f"Convergence failure in OPLL test ({fsw_max/1e3:.1f}kHz)")

        try:
            _vo_arr_21 = np.array(analysis_21['vo'])
            _t_arr_21  = np.array(analysis_21.time)
            _periodo_21 = 1.0 / fsw_max
            _t_janela_21 = ss_config['avg_cycles'] * _periodo_21
            _idx_21 = (np.searchsorted(_t_arr_21, _t_arr_21[-1] - _t_janela_21)
                       if (_t_arr_21[-1] - _t_janela_21) > _t_arr_21[0] else 0)
            _seg_t_21 = _t_arr_21[_idx_21:]
            _seg_v_21 = _vo_arr_21[_idx_21:]
            _delta_t_21 = _seg_t_21[-1] - _seg_t_21[0]
            vo_ll_sim = (float(np.trapezoid(_seg_v_21, _seg_t_21) / _delta_t_21)
                         if _delta_t_21 > 0 else float(_seg_v_21[-1]))
        except Exception:
            raise ValueError(f"Vo extraction error in OPLL test ({fsw_max/1e3:.1f}kHz)")

        entry['vo_ll_test']      = vo_ll_sim
        entry['fsw_ll_test_khz'] = fsw_max / 1e3
        entry['t_21_s'] = time.perf_counter() - t0_21
        log.append(f"[INFO]    [2.1 OPLL      ] Vo@fmax={vo_ll_sim:.1f}V @ fsw={fsw_max/1e3:.1f}kHz,"
                   f" Rload={rload_ll:.1f}Ohm (need<{vo_ll:.1f}V) | t={entry['t_21_s']:.1f}s")

        # Hard constraint: LLC must be able to regulate down to Vo_LL at fsw_max.
        # If Vo @ fsw_max >= Vo_LL the converter cannot reach the light-load target
        # within the allowed frequency range — kill immediately with 1e6 penalty.
        if vo_ll_sim >= vo_ll:
            raise ValueError(
                f"OPLL hard constraint: Vo@fsw_max={vo_ll_sim:.1f}V >= Vo_LL={vo_ll:.1f}V"
                f" (margin={vo_ll_sim - vo_ll:.1f}V)"
            )

        # OPLL voltage margin (log only — hard constraint above already kills infeasible
        # individuals; this value is negative by construction and is not used in n_constr).
        opll_voltage_margin = vo_ll_sim - vo_ll
        entry['opll_voltage_margin'] = opll_voltage_margin

        # ======================================================================
        # 3.1 – Operating Point Simulation (all OPHP points via test_edge_cases)
        # ======================================================================
        t0_op = time.perf_counter()
        # TimeStep derived from NPTsw so resolution scales with frequency.
        # fsw is unknown here (it varies per OP); use fsw_max as the conservative
        # reference — highest frequency → smallest period → tightest timestep needed.
        _npts   = params.get('NPTsw', 400)
        _ts_op  = 1.0 / (fsw_max * _npts)
        res_list = test_edge_cases(op_hp_points, params, SimCycles=800, TimeStep=_ts_op,
                                   tol=0.1, max_workers=1, verbose=False,
                                   fsw_min_per_case=fsw_min_per_case)
        entry['t_op_total_s'] = time.perf_counter() - t0_op

        if not res_list or len(res_list) != len(op_hp_points):
            raise ValueError("OPHP Multi-point simulation failed or incomplete")

        # Index results by case name for safe per-point lookup
        res_dict = {pt['case']: pt for pt in res_list}

        # ======================================================================
        # 4.1 – Score   (mean iR_rms + std iR_rms across all OPHP)
        # 4.2 – Constraints: G[0]=ids_off, G[1]=t_dis vs dt_lower, G[2]=max_dt_win_sim_pct vs allowed
        # ======================================================================
        coss_val             = 100e-12  # MOSFET output capacitance (assumed constant) [F]
        vbus_val             = params.get('Vbus', 420)
        dead_time_min_ic     = params.get('dead_time_min_ic', 100e-9)  # min IC dead time [s]
        dt_max_pct           = params.get('dt_max_pct', 10.0)          # max dead time as % of Ts/2 — sets the dead-time window
        max_reg_error        = 0.0     # worst-case regulation error across OPHP
        ids_off_list         = []   # raw ids_off per HP point [A]
        g1_list              = []   # (t_dis - dt_lower) / dt_lower per HP — G[1]
        max_dt_win_sim_pct_list = []   # max_dt_win_sim as % of Ts/2 per HP — G[2] source

        # --- Per-point extraction -------------------------------------------
        # For each high-power operating point, extract:
        #   - regulation error: |Vo_sim - Vo_target| / Vo_target
        #   - ZVS margin: time required to discharge Coss vs. available dead-time window
        #   - all scalar simulation results stored in entry[] for logging/CSV
        for i, op_str in enumerate(op_hp_points):
            pt = res_dict.get(op_str)
            if pt is None:
                raise ValueError(f"Missing simulation result for {op_str}")

            prefix = f"P{i+1}_"
            parts = [p for p in re.split(r'[V/@A]', op_str) if p]
            vin_target, vo_target, io_target = float(parts[0]), float(parts[1]), float(parts[2])

            # Regulation error: relative deviation of simulated Vo from target
            simulated_vo = pt.get('voutAVG', 0.0)
            if vo_target > 0:
                error_pct = abs(simulated_vo - vo_target) / vo_target
                if error_pct > max_reg_error:
                    max_reg_error = error_pct

            # ZVS: collect ids_off and compute discharge time for soft constraint G[1].
            # G[0] checks the ids_off sign directly; t_discharge is only meaningful when > 0.
            ids_off_raw = pt.get('iDSoff', 0.0)
            ids_off_list.append(ids_off_raw)

            fsw_i      = max(pt.get('fsw_khz', 100.0) * 1e3, 1e3)
            allowed_max_t_dis = max(dead_time_min_ic, (dt_max_pct / 100.0) / (2.0 * fsw_i)) 

            if ids_off_raw > 0:
                t_dis_i  = (2.0 * coss_val * vbus_val) / ids_off_raw
                t_dis_g1 = t_dis_i
            else:
                t_dis_i  = np.nan              # ids_off ≤ 0: no physical discharge time
                t_dis_g1 = allowed_max_t_dis * 10.0  # sentinel for G[1] only; G[0] already flags ZVS loss

            g1_list.append((t_dis_g1 - allowed_max_t_dis) / allowed_max_t_dis)

            max_dt_win_sim_pct = pt.get('max_dt_win_sim', 0.0) * fsw_i * 200.0  # [%] of Ts/2
            max_dt_win_sim_pct_list.append(max_dt_win_sim_pct)

            pt['t_discharge_req']    = t_dis_i  # NaN when ids_off ≤ 0
            pt['allowed_max_t_dis']  = allowed_max_t_dis
            pt['max_dt_win_sim_pct'] = max_dt_win_sim_pct

            # Store nominal targets and all scalar simulation outputs in entry
            entry[f"{prefix}Vin_target"] = vin_target
            entry[f"{prefix}Vo_target"]  = vo_target
            entry[f"{prefix}Io_target"]  = io_target

            for k, v in pt.items():
                if not isinstance(v, (list, tuple, np.ndarray)):
                    entry[f"{prefix}{k}"] = v

        ids_off_min              = min(ids_off_list)
        g1_worst                 = max(g1_list)
        min_max_dt_win_sim_pct_worst = min(max_dt_win_sim_pct_list)
        g2_worst                 = (allowed_min_dt_win_pct - min_max_dt_win_sim_pct_worst) / max(allowed_min_dt_win_pct, 1.0)

        # --- Log summary per point ------------------------------------------
        for i_op, op_str in enumerate(op_hp_points):
            pt = res_dict[op_str]
            log.append(f"[INFO]    [OP {i_op+1}] {op_str} -> fsw={pt['fsw_khz']:.1f}kHz,"
                       f" Vo={pt.get('voutAVG', 0):.1f}V, iR_rms={pt.get('iRRMS', 0):.3f}A")

        # --- Score (objective function F) -----------------------------------
        # Generalised (power) mean of iR_rms across all HP operating points.
        # Higher exponent p penalises worst-case points more than the arithmetic mean.
        ir_values = [res_dict[op].get('iRRMS', 0) for op in op_hp_points]

        p = 4  # power-mean exponent
        score = float(np.mean(np.power(ir_values, p)) ** (1 / p))

        entry['score'] = score
        entry['ir_std'] = float(np.std(ir_values))

        # --- Constraint metrics ---------------------------------------------
        fsw_list     = [res_dict[op]['fsw_khz'] * 1e3 for op in op_hp_points]
        _t_dis_vals  = [res_dict[op].get('t_discharge_req', np.nan) for op in op_hp_points]
        t_dis_max_ns = float(np.nanmax(_t_dis_vals)) * 1e9 if any(not np.isnan(v) for v in _t_dis_vals) else np.nan

        entry['fsw_min_sim_kHz']       = min(fsw_list) / 1e3
        entry['fsw_max_sim_kHz']       = max(fsw_list) / 1e3
        entry['ids_off_min_A']         = ids_off_min
        entry['t_dis_max_ns']          = t_dis_max_ns
        entry['max_dt_win_sim_pct']    = min_max_dt_win_sim_pct_worst  # worst-case (minimum) across HP points
        entry['reg_error_max_pct']     = max_reg_error * 100
        entry['opll_margin_V']         = opll_voltage_margin

        F = [score]
        G = [
            -ids_off_min,  # G[0]: violated when ids_off_min < 0 (ZVS lost); direct [A]
            g1_worst,      # G[1]: t_dis vs max(dead_time_min_ic, dt_max_pct%·Ts/2) — normalised
            g2_worst,      # G[2]: allowed_min_dt_win_pct - max_dt_win_sim_pct (violated when too small) — normalised
        ]

        # ======================================================================
        # 5. CONSTRAINT AUDIT AND CSV CLASSIFICATION
        # ======================================================================
        # Active constraints: G[0]=ids_off, G[1]=t_dis vs dt_lower,
        #                     G[2]=max_dt_win_sim_pct >= allowed_min_dt_win_pct (violated when too small).
        # fsw range, reg error and OPLL margin are logged to CSV only (not penalised).
        if min(fsw_list) < fsw_min:
            log.append(f"[INFO]    >> fsw below limit: {min(fsw_list)/1e3:.1f}kHz < {fsw_min/1e3:.0f}kHz (CSV log only)")
        if max(fsw_list) > fsw_max:
            log.append(f"[INFO]    >> fsw above limit: {max(fsw_list)/1e3:.1f}kHz > {fsw_max/1e3:.0f}kHz (CSV log only)")
        if max_reg_error > 0.05:
            log.append(f"[INFO]    >> Reg error {max_reg_error*100:.1f}% > 5% (CSV log only)")

        reasons = []
        if G[0] > 0: reasons.append(f"ids_off<0 -> min={ids_off_min:.3f}A (G[0]={G[0]:.3f})")
        _t_dis_str = f"{t_dis_max_ns:.0f}ns" if not np.isnan(t_dis_max_ns) else "N/A"
        if G[1] > 0: reasons.append(f"t_dis={_t_dis_str} > dt_lower (G[1]={G[1]:.3f})")
        if G[2] > 0: reasons.append(f"max_dt_win_sim={min_max_dt_win_sim_pct_worst:.1f}% < {allowed_min_dt_win_pct:.1f}% min (G[2]={G[2]:.3f})")

        if reasons:
            entry['status'] = "Invalid: " + " | ".join(reasons)
            log.append(f"[WARNING] >> Score: {score:.3f}A | {entry['status']}")
        else:
            entry['status'] = "Valid"
            log.append(f"[INFO]    >> Score: {score:.3f}A | Valid")

        entry['t_total_s'] = time.perf_counter() - t0_total
        return F, G, entry, log

    except Exception as e:
        entry['status'] = f"Fail: {str(e)}"
        entry['t_total_s'] = time.perf_counter() - t0_total
        log.append(f"[ERROR]   >> {str(e)}")
        return [1e6], [1e6]*3, entry, log

    finally:
        # ------------------------------------------------------------------
        # Restore stdout/stderr and collect all output produced by this worker.
        # Since log[] is a mutable list already referenced in the return value,
        # appending here makes the captured output visible to the caller.
        # ------------------------------------------------------------------
        sys.stdout = _saved_stdout
        sys.stderr = _saved_stderr

        # Restore OS-level file descriptors (for ngspice DLL)
        if _saved_fd1 is not None:
            try:
                os.dup2(_saved_fd1, 1)
                os.dup2(_saved_fd2, 2)
                os.close(_saved_fd1)
                os.close(_saved_fd2)
            except Exception:
                pass

        # Collect Python-level prints (f_vtarget, fast_ss, edge_cases, etc.)
        py_output = _py_cap.getvalue()

        # Collect C-level prints (ngspice DLL writes to fd 1/2)
        c_output = ''
        if _tmp_path:
            try:
                with open(_tmp_path, 'rb') as _f:
                    c_output = _f.read().decode('utf-8', errors='replace')
                os.unlink(_tmp_path)
            except Exception:
                pass

        combined = (py_output + c_output).strip()
        if combined:
            log.append("[DEBUG]   [--- worker output ---]")
            for _line in combined.splitlines():
                if _line.strip():
                    log.append(f"[DEBUG]   | {_line}")


# ==============================================================================
# CLASSE DO PROBLEMA
# ==============================================================================
class LLC_Problem_OP_iR(Problem):

    def __init__(self, base_params, op_hp_points, op_ll, pop_size,
                 fo_fixed=None, verbose=True):
        """
        Parameters
        ----------
        base_params   : dict  – fixed circuit parameters (Vbus, Co, dead_time,
                                NPTsw, fsw_min, fsw_max, Bmax,
                                dead_time_min_ic, dt_max_pct,
                                allowed_min_dt_win_pct, …)
        op_hp_points  : list  – high-power operating points used for fitness
                                evaluation, format "VinV/VoV@IoA"
        op_ll         : str   – light-load operating point used to verify the
                                maximum switching frequency, same format
        pop_size      : int   – GA population size (used to track generation)
        fo_fixed      : float or None – if set, fo is fixed and not optimised
        verbose       : bool  – enable per-individual console output
        """
        fsw_min = base_params.get('fsw_min', 50e3)
        fsw_max = base_params.get('fsw_max', 500e3)

        if fo_fixed is not None:
            # fo is fixed: optimise only [n, Cs, k]
            super().__init__(n_var=3, n_obj=1, n_constr=3,
                             xl=np.array([1.0,  1e-9,   1.0]),
                             xu=np.array([5.0, 100e-9, 10.0]))
        else:
            # fo is free: optimise [n, Cs, fo, k]
            # fo bounds follow fsw range from base_params so they stay consistent
            super().__init__(n_var=4, n_obj=1, n_constr=3,
                             xl=np.array([1.0,  1e-9,  fsw_min,  1.0]),
                             xu=np.array([5.0, 100e-9, fsw_max, 10.0]))

        self.base_params         = base_params
        self.op_hp_points        = op_hp_points
        self.op_ll               = op_ll
        self.pop_size            = pop_size
        self.fsw_min_allowed     = base_params.get('fsw_min', 50e3)
        self.fsw_max_allowed     = base_params.get('fsw_max', 500e3)
        self.fo_fixed            = fo_fixed
        self.allowed_min_dt_win_pct = base_params.get('allowed_min_dt_win_pct', 15.0)  # min required dead-time window [%]
        self.n_eval              = 0
        self.history             = []
        self.verbose             = verbose

    def _evaluate(self, X, out, *args, **kwargs):
        num_individuals = len(X)
        F_out = np.zeros((num_individuals, self.n_obj))
        G_out = np.zeros((num_individuals, self.n_constr))

        # Use pymoo's own generation counter — more reliable than estimating from n_eval,
        # which breaks when pymoo evaluates extra individuals (e.g. initial duplicates).
        algorithm = kwargs.get('algorithm')
        curr_gen  = algorithm.n_gen if algorithm is not None else (self.n_eval // self.pop_size + 1)

        tasks = []
        for i in range(num_individuals):
            self.n_eval += 1
            curr_ind = i + 1

            tasks.append((
                X[i], self.base_params, self.op_hp_points, self.op_ll,
                self.fsw_min_allowed, self.fsw_max_allowed,
                self.fo_fixed, self.allowed_min_dt_win_pct,
                curr_gen, curr_ind, self.n_eval, self.verbose
            ))

        cores = max(1, min((os.cpu_count() or 1) - 2, num_individuals))
        print(f"\n==============================================", flush=True)
        print(f"[INFO] [GA] Generation {curr_gen} | {num_individuals} individuals | {cores} cores"
              f" -> evaluating in parallel...", flush=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = list(executor.map(evaluate_single_individual, tasks))

        for i, (f, g, entry, ind_log) in enumerate(results):
            F_out[i, :] = f
            G_out[i, :] = g
            self.history.append(entry)

            gen     = entry.get('generation', 0)
            ind     = entry.get('individual', 0)
            t_total = entry.get('t_total_s', 0)
            n_v     = entry.get('n', 0)
            cs_v    = entry.get('Cs', 0)
            ls_v    = entry.get('Ls_calc', 0)
            lm_v    = entry.get('Lm_calc', 0)
            fo_v    = entry.get('fo_res', 0)
            k_v     = entry.get('k_ratio', 0)
            hdr = (f"n={n_v:.2f}, Cs={cs_v*1e9:.1f}nF, Ls={ls_v*1e6:.1f}uH,"
                   f" Lm={lm_v*1e6:.1f}uH, fo={fo_v/1e3:.1f}kHz, k={k_v:.2f}")
            print(f"\n[INFO] ── [Gen {gen:02d} | Ind {ind:02d}] {hdr} | t={t_total:.1f}s", flush=True)
            for line in ind_log:
                print(line, flush=True)
            _f_str = f"F={f[0]:.4f}" if len(f) == 1 else f"F={f}"
            _g_labels    = ["G0:ids_off",    "G1:t_dis_lo",  "G2:dt_win_pct"]
            _g_raw_keys  = ["ids_off_min_A", "t_dis_max_ns", "max_dt_win_sim_pct"]
            _g_raw_units = ["A",              "ns",           "%"]
            _g_log_keys  = ["fsw_min_sim_kHz", "fsw_max_sim_kHz", "reg_error_max_pct", "opll_margin_V"]
            _g_log_lbls  = ["fsw_min",          "fsw_max",         "reg_err",           "opll"]
            _g_log_units = ["kHz",              "kHz",             "%",                 "V"]
            _g_norm_str = "  ".join(f"{lbl}={gv:+.4f}" for lbl, gv in zip(_g_labels, g))
            _g_raw_str  = "  ".join(
                f"{lbl}={entry.get(rk, float('nan')):+.4f}{u}"
                for lbl, rk, u in zip(_g_labels, _g_raw_keys, _g_raw_units)
            )
            _g_log_str  = "  ".join(
                f"{lbl}={entry.get(rk, float('nan')):+.4f}{u}"
                for lbl, rk, u in zip(_g_log_lbls, _g_log_keys, _g_log_units)
            )
            _feasible = all(gv <= 0 for gv in g)
            print(f"[INFO]    {_f_str}  |  {'FEASIBLE' if _feasible else 'INFEASIBLE'}", flush=True)
            print(f"[INFO]    G(norm) : {_g_norm_str}", flush=True)
            print(f"[INFO]    G(raw)  : {_g_raw_str}", flush=True)
            print(f"[INFO]    G(log)  : {_g_log_str}", flush=True)

        out["F"] = F_out
        out["G"] = G_out

        self.save_results_to_csv("LLC_OP_IR_backup.csv", quiet=True)

    def save_results_to_csv(self, filename="LLC_OP_IR.csv", quiet=False):
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        pd.DataFrame(self.history).to_csv(full_path, index=False)
        if not quiet:
            print(f"\n[INFO] Record saved to: {full_path}")
