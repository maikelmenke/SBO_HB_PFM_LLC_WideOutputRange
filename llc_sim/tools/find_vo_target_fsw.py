import numpy as np
from functools import partial
from scipy.optimize import minimize_scalar, brentq
import traceback

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# Direct imports to avoid Circular Import error
from llc_sim.sim_lib.circuit_builder import build_llc_circuit
from llc_sim.sim_lib.analysis import analyze
from llc_sim.sim_lib.simulation import simulate
from llc_sim.tools.simulate_until_ss import simulate_until_steady_state


class ToleranceReached(Exception):
    """Exception raised when voltage or current tolerance is met during optimization."""
    def __init__(self, fsw_khz, results, target_val, error):
        self.fsw_khz = fsw_khz
        self.results = results
        self.target_val = target_val
        self.error = error

def voltage_error(fsw_khz, params, Vtarget, tol, SimCycles, TimeStep, Full_Arr,
                  cycles_per_block, steady_state_tol,
                  avg_cycles, stable_blocks_required, config,
                  ripple_tol_factor=3.0):
    """
    Objective function used by ``find_vtarget``: simulate at ``fsw_khz`` and
    return |Vout_avg - Vtarget|.

    Runs ``simulate_until_steady_state`` then computes the trapezoidal average
    of the last ``avg_cycles`` of the output voltage.  If the error is already
    within ``tol``, raises :exc:`ToleranceReached` so that
    ``minimize_scalar`` can be interrupted early.

    Returns a large sentinel value (1e6 or 2e6) instead of raising on
    simulation failures or out-of-bounds frequencies, so that the optimizer
    treats those points as infeasible.

    Parameters
    ----------
    fsw_khz : float
        Switching frequency to evaluate [kHz].
    params : dict
        Converter parameters (same format as ``build_llc_circuit``).
    Vtarget : float
        Target output voltage [V].
    tol : float
        Voltage tolerance for early exit [V].  Set 0 to disable.
    SimCycles : int
        Passed to the final high-fidelity simulation (unused in the fast
        steady-state search leg).
    TimeStep : float
        Simulation time step [s].  Overridden by ``NPTsw`` if present in
        ``params``.
    Full_Arr : bool
        Passed through to ``analyze()`` (not used during search).
    cycles_per_block : int
        Block size for ``simulate_until_steady_state``.
    steady_state_tol : float
        Convergence tolerance for ``simulate_until_steady_state``.
    avg_cycles : int
        Number of cycles used for the output voltage average.
    stable_blocks_required : int
        Consecutive stable blocks required to declare steady state.
    config : int
        Circuit configuration (1 or 2) passed to ``build_llc_circuit``.

    Returns
    -------
    float
        |Vout_avg - Vtarget| [V], or a large sentinel on failure.

    Raises
    ------
    ToleranceReached
        If the error falls within ``tol`` before the optimizer converges.
    """

    # Parameter formatting for display
    n_val = params.get('n', 0)
    cs_val = params.get('Cs', 0) * 1e9
    ls_val = params.get('Ls', 0) * 1e6
    lm_val = params.get('Lm', 0) * 1e6
    vbus_val = params.get('Vbus', 0)
    p_str = f"[n={n_val:.2f}, Cs={cs_val:.1f}nF, Ls={ls_val:.1f}uH, Lm={lm_val:.1f}uH, Vin={vbus_val:.0f}V, Vo_tgt={Vtarget:.1f}V]"
    
    current_params = params.copy()
    current_params["fsw"] = fsw_khz * 1e3  # Convert kHz to Hz

    if 'NPTsw' in current_params:
        TimeStep = (1.0 / current_params["fsw"]) / current_params['NPTsw']

    fsw_min = params.get('fsw_min', 10e3)
    fsw_max = params.get('fsw_max', 2000e3)
    if not (fsw_min <= current_params["fsw"] <= fsw_max):
        return 2e6

    try:
        circuit = build_llc_circuit(current_params, Vtarget, config=config)

        analysis_fast, _cycles_to_stable = simulate_until_steady_state(
            circuit=circuit,
            fsw=current_params["fsw"],
            TimeStep=TimeStep,
            MaxCycles=20e3,
            cycles_per_block=cycles_per_block,
            SteadyStateNode='vo',
            SteadyStateTol=steady_state_tol,
            avg_cycles=avg_cycles,
            stable_blocks_required=stable_blocks_required,
            ripple_tol_factor=ripple_tol_factor,
            save_all=False, # Fast mode during search
        )

        if analysis_fast is None:
            print(f"[ERROR]   Find SS - simulation failed for fsw = {fsw_khz:.3f} kHz, retrying with TimeStep/2 | {p_str}")
            circuit_retry = build_llc_circuit(current_params, Vtarget, config=config)
            analysis_fast, _cycles_to_stable = simulate_until_steady_state(
                circuit=circuit_retry,
                fsw=current_params["fsw"],
                TimeStep=TimeStep * 0.5,
                MaxCycles=20e3,
                cycles_per_block=cycles_per_block,
                SteadyStateNode='vo',
                SteadyStateTol=steady_state_tol,
                avg_cycles=avg_cycles,
                stable_blocks_required=stable_blocks_required,
                ripple_tol_factor=ripple_tol_factor,
                save_all=False,
            )
        if analysis_fast is None:
            print(f"[ERROR]   Find SS - retry also failed for fsw = {fsw_khz:.3f} kHz | {p_str}")
            return 1e6

        time_vec = analysis_fast.time
        vo_vec = analysis_fast['vo']

        block_end_time = float(time_vec[-1])
        avg_start_time = max(0.0, block_end_time - avg_cycles * (1.0/current_params["fsw"]))
        avg_indices = time_vec >= (avg_start_time @u_s)

        final_segment = np.asarray(vo_vec[avg_indices], dtype=float)
        time_segment = np.asarray(time_vec[avg_indices], dtype=float)

        if final_segment.size < 2 or not np.all(np.isfinite(final_segment)):
            final_segment = np.asarray(vo_vec, dtype=float)
            time_segment = np.asarray(time_vec, dtype=float)

        integral_area = np.trapezoid(y=final_segment, x=time_segment)
        delta_t = time_segment[-1] - time_segment[0]
        if delta_t > 0:
            vout_fast = float(integral_area / delta_t)
        else:
            vout_fast = float(np.mean(final_segment))

        if not np.isfinite(vout_fast):
            return 1e6

        error = abs(vout_fast - Vtarget)

        # Early exit when voltage tolerance is met
        eps = max(1e-6, 1e-6 * abs(Vtarget))
        if tol > 0 and error <= tol + eps:
            raise ToleranceReached(fsw_khz, results=None, target_val=vout_fast, error=error)

        return error

    except ToleranceReached:
        raise
    except Exception:
        print(f"[ERROR]   Find SS - Unexpected failure at fsw = {fsw_khz:.3f} kHz. Details below: | {p_str}")
        traceback.print_exc()
        return 1e6

def voltage_residual(fsw_khz, params, Vtarget, tol, SimCycles, TimeStep, Full_Arr,
                     cycles_per_block, steady_state_tol,
                     avg_cycles, stable_blocks_required, config,
                     ripple_tol_factor=3.0):
    """
    Signed objective function for ``find_vtarget``: simulate at ``fsw_khz``
    and return ``Vout_avg - Vtarget``.

    Positive when Vout > Vtarget (frequency too low),
    negative when Vout < Vtarget (frequency too high).
    This sign convention is consistent with the LLC gain monotonicity above
    resonance, allowing ``brentq`` to bracket the root reliably.

    Raises :exc:`ToleranceReached` when ``|Vout - Vtarget| <= tol``, same as
    ``voltage_error``, so the early-exit mechanism is preserved.

    On simulation failure returns ``+1e6`` (large positive sentinel).
    On frequency out-of-bounds returns ``+2e6`` (below fsw_min) or
    ``-2e6`` (above fsw_max) to guide ``brentq`` toward the valid region.

    Parameters
    ----------
    Same as ``voltage_error``.

    Returns
    -------
    float
        ``Vout_avg - Vtarget`` [V], or a signed sentinel on failure.

    Raises
    ------
    ToleranceReached
        If ``|Vout_avg - Vtarget| <= tol``.
    """
    n_val = params.get('n', 0)
    cs_val = params.get('Cs', 0) * 1e9
    ls_val = params.get('Ls', 0) * 1e6
    lm_val = params.get('Lm', 0) * 1e6
    vbus_val = params.get('Vbus', 0)
    p_str = f"[n={n_val:.2f}, Cs={cs_val:.1f}nF, Ls={ls_val:.1f}uH, Lm={lm_val:.1f}uH, Vin={vbus_val:.0f}V, Vo_tgt={Vtarget:.1f}V]"

    current_params = params.copy()
    current_params["fsw"] = fsw_khz * 1e3

    if 'NPTsw' in current_params:
        TimeStep = (1.0 / current_params["fsw"]) / current_params['NPTsw']

    fsw_min = params.get('fsw_min', 10e3)
    fsw_max = params.get('fsw_max', 2000e3)
    if current_params["fsw"] < fsw_min:
        return 2e6   # below minimum → high gain expected → Vout > Vtarget
    if current_params["fsw"] > fsw_max:
        return -2e6  # above maximum → low gain expected → Vout < Vtarget

    try:
        circuit = build_llc_circuit(current_params, Vtarget, config=config)

        analysis_fast, _cycles_to_stable = simulate_until_steady_state(
            circuit=circuit,
            fsw=current_params["fsw"],
            TimeStep=TimeStep,
            MaxCycles=20e3,
            cycles_per_block=cycles_per_block,
            SteadyStateNode='vo',
            SteadyStateTol=steady_state_tol,
            avg_cycles=avg_cycles,
            stable_blocks_required=stable_blocks_required,
            ripple_tol_factor=ripple_tol_factor,
            save_all=False,
        )

        if analysis_fast is None:
            print(f"[ERROR]   Find SS - simulation failed for fsw = {fsw_khz:.3f} kHz, retrying with TimeStep/2 | {p_str}")
            circuit_retry = build_llc_circuit(current_params, Vtarget, config=config)
            analysis_fast, _cycles_to_stable = simulate_until_steady_state(
                circuit=circuit_retry,
                fsw=current_params["fsw"],
                TimeStep=TimeStep * 0.5,
                MaxCycles=20e3,
                cycles_per_block=cycles_per_block,
                SteadyStateNode='vo',
                SteadyStateTol=steady_state_tol,
                avg_cycles=avg_cycles,
                stable_blocks_required=stable_blocks_required,
                ripple_tol_factor=ripple_tol_factor,
                save_all=False,
            )
        if analysis_fast is None:
            print(f"[ERROR]   Find SS - retry also failed for fsw = {fsw_khz:.3f} kHz | {p_str}")
            return 1e6

        time_vec = analysis_fast.time
        vo_vec = analysis_fast['vo']

        block_end_time = float(time_vec[-1])
        avg_start_time = max(0.0, block_end_time - avg_cycles * (1.0 / current_params["fsw"]))
        avg_indices = time_vec >= (avg_start_time @u_s)

        final_segment = np.asarray(vo_vec[avg_indices], dtype=float)
        time_segment = np.asarray(time_vec[avg_indices], dtype=float)

        if final_segment.size < 2 or not np.all(np.isfinite(final_segment)):
            final_segment = np.asarray(vo_vec, dtype=float)
            time_segment = np.asarray(time_vec, dtype=float)

        integral_area = np.trapezoid(y=final_segment, x=time_segment)
        delta_t = time_segment[-1] - time_segment[0]
        vout_fast = float(integral_area / delta_t) if delta_t > 0 else float(np.mean(final_segment))

        if not np.isfinite(vout_fast):
            return 1e6

        residual = vout_fast - Vtarget
        abs_error = abs(residual)

        eps = max(1e-6, 1e-6 * abs(Vtarget))
        if tol > 0 and abs_error <= tol + eps:
            raise ToleranceReached(fsw_khz, results=None, target_val=vout_fast, error=abs_error)

        return residual

    except ToleranceReached:
        raise
    except Exception:
        print(f"[ERROR]   Find SS - Unexpected failure at fsw = {fsw_khz:.3f} kHz. Details below: | {p_str}")
        traceback.print_exc()
        return 1e6

def find_vo_target(params, Vtarget, tol, SimCycles, TimeStep,
                 cycles_per_block, steady_state_tol,
                 avg_cycles, stable_blocks_required, Full_Arr=False, config=1, verbose=True,
                 fsw_min_override=None, ripple_tol_factor=3.0):
    """
    Find the switching frequency that produces ``Vtarget`` at the output.

    Uses SciPy's ``brentq`` root-finder on the signed residual
    ``Vout(fsw) - Vtarget``, exploiting the monotonically decreasing LLC
    gain curve. 

    Search bounds are derived from the resonant frequencies:

    - If Vtarget ≥ Vbus/(2n) (above-resonance gain): search between f1
      (parallel resonance) and fo (series resonance).
    - Otherwise (below resonance): search between fo and 10·fo.

    If ``brentq`` cannot bracket the root (``Vtarget`` outside the
    achievable gain range), it falls back to ``minimize_scalar`` on the
    absolute error to return the closest reachable frequency.

    After the search converges (or ``ToleranceReached`` is raised), a
    final high-fidelity simulation is run at the found frequency using the
    full ``SimCycles`` (or the actual number of cycles needed to reach
    steady state) and ``return_arrays=True`` to return complete waveforms.

    Parameters
    ----------
    params : dict
        Converter parameters.  Must include ``Vbus``, ``n``, ``Cs``,
        ``Ls``, ``Lm``.  Optional: ``fsw_min``, ``fsw_max``, ``NPTsw``.
    Vtarget : float
        Target output voltage [V].
    tol : float
        Voltage tolerance for early exit [V].
    SimCycles : int
        Number of cycles for the final high-fidelity simulation.
    TimeStep : float
        Simulation time step [s].
    cycles_per_block : int
        Block size for ``simulate_until_steady_state``.
    steady_state_tol : float
        Convergence tolerance for ``simulate_until_steady_state``.
    avg_cycles : int
        Cycles used for the output voltage average.
    stable_blocks_required : int
        Consecutive stable blocks required to declare steady state.
    Full_Arr : bool, optional
        Passed to ``voltage_error`` (unused during search). Default False.
    config : int, optional
        Circuit configuration for ``build_llc_circuit``. Default 2.
    verbose : bool, optional
        Reserved for future use. Default True.

    Returns
    -------
    tuple[dict, float]
        ``(final_results, fsw_khz)`` where ``final_results`` is the dict
        returned by ``analyze(..., return_arrays=True)`` and ``fsw_khz`` is
        the found switching frequency [kHz].
    """

    # Parameter formatting for display
    n_val = params.get('n', 0)
    cs_val = params.get('Cs', 0) * 1e9
    ls_val = params.get('Ls', 0) * 1e6
    lm_val = params.get('Lm', 0) * 1e6
    vbus_val = params.get('Vbus', 0)
    p_str = f"[n={n_val:.2f}, Cs={cs_val:.1f}nF, Ls={ls_val:.1f}uH, Lm={lm_val:.1f}uH, Vin={vbus_val:.0f}V, Vo_tgt={Vtarget:.1f}V]"
    
    Voafo = params["Vbus"] / (2 * params["n"]) # Vo at resonance
    f1 = (1 / (2 * np.pi * np.sqrt(params["Cs"] * (params["Ls"] + params["Lm"])))) / 1e3
    fo = (1 / (2 * np.pi * np.sqrt(params["Cs"] * params["Ls"]))) / 1e3

    fsw_max_khz = params.get('fsw_max', 10 * fo * 1e3) / 1e3

    if Vtarget >= Voafo:
        low  = (fsw_min_override / 1e3) if fsw_min_override is not None else f1 + 1
        high = fo + 1
    else:
        low  = (fsw_min_override / 1e3) if fsw_min_override is not None else fo - 1
        high = fsw_max_khz

    _residual_kwargs = dict(
        params=params,
        Vtarget=Vtarget,
        tol=tol,
        SimCycles=SimCycles,
        TimeStep=TimeStep,
        Full_Arr=Full_Arr,
        cycles_per_block=cycles_per_block,
        steady_state_tol=steady_state_tol,
        avg_cycles=avg_cycles,
        stable_blocks_required=stable_blocks_required,
        config=config,
        ripple_tol_factor=ripple_tol_factor,
    )
    residual_fn = partial(voltage_residual, **_residual_kwargs)

    # Track the last valid Vout seen by brentq (RootResults has no 'residual' attribute)
    _last_valid = {}
    def _tracked_residual(fsw):
        res = residual_fn(fsw)
        if np.isfinite(res) and abs(res) < 1e5:
            _last_valid['vout'] = Vtarget + res
        return res

    fsw_khz = None
    try:
        fsw_khz, _r = brentq(_tracked_residual, low, high, xtol=1e-3, maxiter=100, full_output=True)
        _ts = (1.0 / (fsw_khz * 1e3)) / params['NPTsw'] if 'NPTsw' in params else TimeStep
        _vout = _last_valid.get('vout')
        _vout_str = f"Vout≈{_vout:.4f} V, |err|≈{abs(_vout - Vtarget):.6g} V" if _vout is not None else "Vout≈N/A"
        print(f"[INFO]    Find SS - brentq converged. fsw = {fsw_khz:.6f} kHz | {_vout_str} | Vtarget={Vtarget:.4f} V | TimeStep={_ts*1e9:.2f}ns | {p_str}")
    except ToleranceReached as e:
        fsw_khz = e.fsw_khz
        _ts = (1.0 / (fsw_khz * 1e3)) / params['NPTsw'] if 'NPTsw' in params else TimeStep
        print(f"[INFO]    Find SS - Tolerance met. fsw = {fsw_khz:.3f} kHz (Vout={e.target_val:.4f} V, |err|={e.error:.6g} V ≤ tol={tol}) | TimeStep={_ts*1e9:.2f}ns | {p_str}")
    except ValueError:
        # f(low) and f(high) have the same sign: Vtarget is outside the
        # achievable gain range for these bounds. Fall back to minimize_scalar
        # on the absolute error to get the closest reachable frequency.
        print(f"[WARNING] Find SS - Vtarget outside achievable gain range "
              f"(bounds=[{low:.2f}, {high:.2f}] kHz). Falling back to minimize_scalar. | {p_str}")
        abs_fn = partial(voltage_error, **_residual_kwargs)
        try:
            result = minimize_scalar(
                abs_fn, bounds=(low, high), method='bounded',
                options={'xatol': 1e-3, 'maxiter': 100}
            )
            fsw_khz = float(result.x)
            _ts = (1.0 / (fsw_khz * 1e3)) / params['NPTsw'] if 'NPTsw' in params else TimeStep
            print(f"[INFO]    Find SS - Best fsw = {fsw_khz:.6f} kHz, |err| ≈ {result.fun:.6g} V | TimeStep={_ts*1e9:.2f}ns | {p_str}")
        except ToleranceReached as e:
            fsw_khz = e.fsw_khz
            _ts = (1.0 / (fsw_khz * 1e3)) / params['NPTsw'] if 'NPTsw' in params else TimeStep
            print(f"[INFO]    Find SS - Tolerance met. fsw = {fsw_khz:.3f} kHz (Vout={e.target_val:.4f} V, |err|={e.error:.6g} V ≤ tol={tol}) | TimeStep={_ts*1e9:.2f}ns | {p_str}")

    # --- Final high-fidelity simulation ---
    try:
        current_params = params.copy()
        current_params["fsw"] = fsw_khz * 1e3  # Hz

        if 'NPTsw' in current_params:
            TimeStep = (1.0 / current_params["fsw"]) / current_params['NPTsw']

        final_ts = TimeStep

        circuit_probe = build_llc_circuit(current_params, Vtarget, config=config)
        _, cycles_to_stable = simulate_until_steady_state(
            circuit=circuit_probe,
            fsw=current_params["fsw"],
            TimeStep=final_ts,
            MaxCycles=20e3,
            cycles_per_block=cycles_per_block,
            SteadyStateNode='vo',
            SteadyStateTol=steady_state_tol,
            avg_cycles=avg_cycles,
            stable_blocks_required=stable_blocks_required,
            ripple_tol_factor=ripple_tol_factor,
            save_all=False
        )

        if cycles_to_stable is None:
            print(f"[ERROR]   Find SS - Final probe failed, retrying with TimeStep/2 | {p_str}")
            final_ts = TimeStep * 0.5
            circuit_probe = build_llc_circuit(current_params, Vtarget, config=config)
            _, cycles_to_stable = simulate_until_steady_state(
                circuit=circuit_probe,
                fsw=current_params["fsw"],
                TimeStep=final_ts,
                MaxCycles=20e3,
                cycles_per_block=cycles_per_block,
                SteadyStateNode='vo',
                SteadyStateTol=steady_state_tol,
                avg_cycles=avg_cycles,
                stable_blocks_required=stable_blocks_required,
                ripple_tol_factor=ripple_tol_factor,
                save_all=False
            )

        final_sim_cycles = cycles_to_stable if cycles_to_stable else cycles_per_block

        circuit_final = build_llc_circuit(current_params, Vtarget, config=config)
        final_analysis = simulate(
            circuit_final,
            current_params["fsw"],
            final_ts,
            SimCycles=final_sim_cycles
        )

        final_results = analyze(
            final_analysis,
            current_params["fsw"],
            final_ts,
            final_sim_cycles,
            return_arrays=True
        )

    except Exception as exc:
        print(f"[ERROR]   Find SS - Final simulation failed: {exc} | {p_str}")
        final_results = {
            'note': 'final dynamic-length sim failed or was skipped',
            'error': str(exc)
        }

    return final_results, fsw_khz

