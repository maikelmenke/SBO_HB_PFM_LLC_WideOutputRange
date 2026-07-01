import numpy as np
from scipy.optimize import minimize_scalar

from llc_sim.sim_lib.circuit_builder import build_llc_circuit
from llc_sim.sim_lib.analysis import analyze
from llc_sim.tools.simulate_until_ss import simulate_until_steady_state

simulation_cache = {}

def _simulate_single_freq_fast_ss(args):
    freq, params_base, time_step, config, ss_params, verbose = args
    freq_khz = freq / 1e3
    
    # Extract and format the filter parameters, Vin and Rload for printing
    n_val = params_base.get('n', 0)
    cs_val = params_base.get('Cs', 0) * 1e9
    ls_val = params_base.get('Ls', 0) * 1e6
    lm_val = params_base.get('Lm', 0) * 1e6
    vbus_val = params_base.get('Vbus', 0)
    rload_val = params_base.get('Rload', 0)
    p_str = f"[n={n_val:.2f}, Cs={cs_val:.1f}nF, Ls={ls_val:.1f}uH, Lm={lm_val:.1f}uH, Vin={vbus_val:.0f}V, Rload={rload_val:.1f}Ω]"
    
    params = params_base.copy()
    params['fsw'] = freq
    
    if 'NPTsw' in params:
        time_step = (1/freq) / params['NPTsw']

    try:
        circuit = build_llc_circuit(params, config=config)
        
        analysis, cycles_to_stable = simulate_until_steady_state(
            circuit=circuit,
            fsw=freq,
            TimeStep=time_step,
            MaxCycles=ss_params.get('MaxCycles', 4000),
            cycles_per_block=ss_params.get('cycles_per_block', 50),
            SteadyStateNode='vo',
            SteadyStateTol=ss_params.get('SteadyStateTol', 1e-3),
            avg_cycles=ss_params.get('avg_cycles', 20),
            stable_blocks_required=ss_params.get('stable_blocks_required', 2)
        )
        
        if analysis is None:
            if verbose: print(f"[ERROR]   Find PK - Simulation failed or returned error for: {freq_khz:.2f} kHz | {p_str}")
            return freq, 0.0

        if cycles_to_stable >= ss_params.get('MaxCycles', 3000):
            if verbose: print(f"[WARNING] Find PK - {freq_khz:.2f} kHz reached the limit of {cycles_to_stable} cycles without perfect stabilization! | {p_str}")

        try:
            vout_array = np.array(analysis['vo'])
            time_array = np.array(analysis.time)
        except IndexError:
            if verbose: print(f"[ERROR]   Find PK - Node 'vo' missing for: {freq_khz:.2f} kHz | {p_str}")
            return freq, 0.0

        period = 1.0 / freq
        cycles_for_avg = ss_params.get('avg_cycles', 20)
        window_time = cycles_for_avg * period
        final_time = time_array[-1]
        slice_start_time = final_time - window_time

        if slice_start_time > time_array[0]:
            idx_inicio = np.searchsorted(time_array, slice_start_time)
        else:
            idx_inicio = 0
            
        vout_avg = float(np.mean(vout_array[idx_inicio:]))
        
        return freq, vout_avg
        
    except Exception as e:
        if verbose: print(f"[ERROR]   Find PK - Simulation aborted for: {freq_khz:.2f} kHz | Reason: {repr(e)} | {p_str}")
        return freq, 0.0

def _get_vout_cached(freq, params_base, time_step, config, ss_params, verbose):
    freq_rounded = round(freq, 2)
    if freq_rounded in simulation_cache:
        return simulation_cache[freq_rounded]
    
    _, vout = _simulate_single_freq_fast_ss((freq, params_base, time_step, config, ss_params, verbose))
    
    simulation_cache[freq_rounded] = vout
    return vout

def find_maximum_output_voltage_ss(params, f_min, f_max, time_step=100e-9,
                                   config=1, ss_params=None, output_dir="outputs", verbose=False):
    global simulation_cache
    simulation_cache = {} 

    # Extract and format parameters for the main function as well
    n_val = params.get('n', 0)
    cs_val = params.get('Cs', 0) * 1e9
    ls_val = params.get('Ls', 0) * 1e6
    lm_val = params.get('Lm', 0) * 1e6
    vbus_val = params.get('Vbus', 0)
    rload_val = params.get('Rload', 0)
    p_str = f"[n={n_val:.2f}, Cs={cs_val:.1f}nF, Ls={ls_val:.1f}uH, Lm={lm_val:.1f}uH, Vin={vbus_val:.0f}V, Rload={rload_val:.1f}Ω]"

    if ss_params is None:
        ss_params = {
            'MaxCycles': 4000, 'cycles_per_block': 50, 'SteadyStateTol': 1e-3,
            'avg_cycles': 20, 'stable_blocks_required': 2
        }

    convergence_history = []

    if verbose:
        print(f"[INFO]    Find PK - Starting peak search via Scalar Optimization (Brent's method) | {p_str}")

    def objective(fsw):
        vo = _get_vout_cached(fsw, params, time_step, config, ss_params, verbose)
        convergence_history.append((fsw, vo))
        if verbose: print(f"[INFO]    Find PK - fsw={fsw/1e3:.3f} kHz -> Vo={vo:.2f} V | {p_str}")
        return -vo

    res = minimize_scalar(
        objective, bounds=(f_min, f_max), method='bounded', options={'xatol': 500, 'maxiter': 20}
    )

    best_freq = res.x
    max_vout = -res.fun

    if verbose:
        print(f"[INFO]    Find PK - Peak found at {best_freq/1e3:.3f} kHz with {len(simulation_cache)} unique simulations. | {p_str}")
        
    best_params = params.copy()
    best_params['fsw'] = best_freq
    best_results = None
    
    try:
        circuit_best = build_llc_circuit(best_params, config=config)
        analysis_best, final_cycles = simulate_until_steady_state(
            circuit_best, best_freq, time_step, 
            MaxCycles=ss_params['MaxCycles'], cycles_per_block=ss_params['cycles_per_block'],
            SteadyStateTol=ss_params['SteadyStateTol'], avg_cycles=ss_params['avg_cycles'],
            stable_blocks_required=ss_params['stable_blocks_required'], SteadyStateNode='vo',
            save_all=True
        )
        
        if analysis_best is not None:
            best_results = analyze(analysis_best, best_freq, time_step, SimCycles=final_cycles)
    except Exception as e:
        if verbose: print(f"[WARNING] Find PK - Full analysis of optimal point failed: {e} | {p_str}")

    sorted_freqs = sorted(simulation_cache.keys())
    sorted_vouts = [simulation_cache[f] for f in sorted_freqs]

    return best_freq, max_vout, best_results, (sorted_freqs, sorted_vouts), output_dir, convergence_history