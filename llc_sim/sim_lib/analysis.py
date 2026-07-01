import numpy as np
import warnings 
from typing import Dict, Any, Optional


def compute_dead_time(time: np.ndarray, iDS: np.ndarray, fsw: float,
                     threshold: float = 0.001) -> float:
    """
    Compute maximum allowed dead time in the last switching cycle.
           
    Parameters
    ----------
    time : np.ndarray
        Simulation time array
    iDS : np.ndarray
        Switch current array
    fsw : float
        Switching frequency in Hz
    threshold : float, optional
        Current threshold to detect switch turn-on (A), default 0.001
        
    Returns
    -------
    float
        Dead time in seconds, or np.nan if computation fails
    """
    try:
        if len(time) < 2 or len(iDS) < 2:
            warnings.warn("Insufficient data points for dead time calculation", RuntimeWarning)
            return np.nan
        
        last_cycle_start_time = time[-1] - (1.0 / fsw)

        if last_cycle_start_time < time[0]:
            warnings.warn(
                f"Last cycle start time ({last_cycle_start_time:.6f}s) is before "
                f"simulation start ({time[0]:.6f}s). Using full simulation.",
                RuntimeWarning
            )
            last_cycle_start_time = time[0]
        
        start_idx = np.argmin(np.abs(time - last_cycle_start_time))

        midpoint_time = time[-1] - (0.5 / fsw)
        midpoint_idx = np.argmin(np.abs(time - midpoint_time))

        turn_on_indices = np.where(iDS[midpoint_idx:] > threshold)[0]
        if len(turn_on_indices) == 0:
            warnings.warn(
                f"No switch turn-on detected after t={time[midpoint_idx]:.6f}s "
                f"(threshold={threshold}A). May indicate insufficient dead time or ZVS loss.",
                RuntimeWarning
            )
            return np.nan
        
        next_on_idx = turn_on_indices[0]
        dead_time = time[midpoint_idx + next_on_idx] - time[midpoint_idx]
        
        # Sanity check: maximum dead time shouldn't be > half cycle period
        half_period = 0.5 / fsw
        if dead_time > half_period:
            warnings.warn(
                f"Calculated dead time ({dead_time*1e6:.2f}µs) exceeds half period "
                f"({half_period*1e6:.2f}µs). Result may be invalid.",
                RuntimeWarning
            )
        
        return dead_time
        
    except (IndexError, ValueError) as e:
        warnings.warn(
            f"Could not compute dead time: {str(e)}. "
            f"Check simulation length and iDS waveform.",
            RuntimeWarning
        )
        return np.nan


def validate_analysis_signals(analysis: Any, required_signals: list) -> None:
    """
    Validate that all required signals are present in analysis results.
    
    Parameters
    ----------
    analysis : PySpice waveform object
        The simulation result object
    required_signals : list
        List of required signal names
        
    Raises
    ------
    ValueError
        If any required signal is missing
    """
    missing = []
    for signal in required_signals:
        if signal == 'time':
            if not hasattr(analysis, 'time'):
                missing.append(signal)
        else:
            try:
                _ = analysis[signal]
            except (KeyError, TypeError):
                missing.append(signal)
    
    if missing:
        raise ValueError(
            f"Missing required signals in analysis results: {', '.join(missing)}"
        )

def analyze(analysis: Any,
           fsw: float, 
           TimeStep: float, 
           SimCycles: int, 
           return_arrays: bool = False,
           cycles_to_analyze: int = 20,
           vab_threshold: float = 0.1,
           dead_time_threshold: float = 0.001) -> Dict[str, Any]:
    """
    Analyze the results of a transient simulation for an LLC converter.

    Extracts key waveforms and calculates performance indicators.
    Optionally returns the full waveform arrays.

    Parameters
    ----------
    analysis : PySpice waveform object
        The result object returned by `simulator.transient(...)`.
    fsw : float
        Switching frequency in Hz.
    TimeStep : float
        Simulation time step in seconds.
    SimCycles : int
        Number of full switching cycles simulated.
    return_arrays : bool, optional
        If True, includes the full NumPy waveform arrays in the returned
        dictionary. Defaults to False (to save memory).
    cycles_to_analyze : int, optional
        Number of cycles at end of simulation to analyze. Default 20.
    vab_threshold : float, optional
        Voltage threshold for switch conduction detection (V). Default 0.1.
    dead_time_threshold : float, optional
        Current threshold for dead time detection (A). Default 0.001.

    Returns
    -------
    dict
        A dictionary containing calculated scalar metrics.
        If `return_arrays` is True, the dictionary also includes the full
        waveform arrays, keyed by their variable names (e.g., 'time', 'vab').

        Scalar Metrics (typically calculated over the last N cycles):
        - 'voutAVG': Average output voltage (V)
        - 'iSecRMS': RMS secondary side current (A)
        - 'iSecPK': Peak secondary current (A)
        - 'iD1PK': Peak diode D1 current (A)
        - 'iD1RMS': RMS diode D1 current (A)
        - 'iRRMS': RMS resonant current (A)
        - 'iRPK': Peak resonant current (A)
        - 'iDSRMS': RMS switch current (A)
        - 'iDSPK': Peak switch current (A)
        - 'iDSoff': Switch current at simulation end (A)
        - 'vCsRMS': RMS capacitor voltage (V)
        - 'vCsPK': Peak capacitor voltage (V)
        - 'ioutAVG': Average output current (A)
        - 'iLmAVG': Average magnetizing current (A)
        - 'vLmPK': Peak magnetizing voltage (V)
        - 'iCoRMS': RMS output capacitor current (A)
        - 'max_dt_win_sim': Estimated dead-time window available before switch turn-on (s)

        Waveform Arrays (only included if return_arrays=True):
        - 'time': Simulation time array (s)
        - 'vab': Voltage across the switching bridge (V)
        - 'vpri': Primary side voltage (V)
        - 'vout': Output voltage (V)
        - 'iR': Resonant current (A)
        - 'iLm': Magnetizing current (A)
        - 'iD1', 'iD2': Diode currents (A)
        - 'iSec': Total secondary current (A)
        - 'iout': Output current (A)
        - 'vcs': Capacitor voltage (V)
        - 'ico': Output capacitor current (A)
        - 'iDS': Approximated switch current (A)
    """
    
    # Validate input parameters
    if fsw <= 0:
        raise ValueError(f"Switching frequency must be positive, got {fsw}")
    if TimeStep <= 0:
        raise ValueError(f"Time step must be positive, got {TimeStep}")
    if SimCycles < 1:
        raise ValueError(f"SimCycles must be >= 1, got {SimCycles}")
    if cycles_to_analyze < 1:
        raise ValueError(f"cycles_to_analyze must be >= 1, got {cycles_to_analyze}")
    
    # Validate required signals are present
    required_signals = ['time', 'vab', 'pri', 'vo', 'VLs_plus', 'VLm_plus',
                       'VD1_cathode', 'VD2_cathode', 'VRload_plus', 'cs']
    validate_analysis_signals(analysis, required_signals)
    
    # Extract and Derive Waveforms 
    time = np.array(analysis.time)
    vab = np.array(analysis['vab'])
    vpri = np.array(analysis['pri'])
    vout = np.array(analysis['vo'])
    iR = np.array(analysis['VLs_plus'])
    iLm = np.array(analysis['VLm_plus'])
    iD1 = np.array(analysis['VD1_cathode'])
    iD2 = np.array(analysis['VD2_cathode'])
    iRec = iD1 + iD2
    iout = np.array(analysis['VRload_plus'])
    vcs = np.array(analysis['vab'] - analysis['cs'])
    # i_Co derived from KCL at node vo: i_Co = i_Rec - i_load
    # (avoids a series voltage source on Co that causes convergence issues)
    ico = iRec - iout

    try:
        # Config 1: extract switch current from series voltage source VSw1.
        MOSVds = np.array(analysis['N001']) - analysis['vab']
        IMOS = np.array(analysis['VSw1'])
        iDS = -IMOS

    except (KeyError, IndexError, AttributeError):
        MOSVds = vab
        iDS = np.where(vab < vab_threshold, -iR, 0)
        IMOS = iDS

    analysis_start_time = time[-1] - (cycles_to_analyze / fsw)

    if analysis_start_time < time[0] - 1e-9 * (time[-1] - time[0]):
        analysis_slice = slice(None, None)
        actual_cycles = (time[-1] - time[0]) * fsw
        warnings.warn(
            f"Requested {cycles_to_analyze} cycles for analysis, but simulation "
            f"only contains ~{actual_cycles:.1f} cycles. Using full simulation data.",
            RuntimeWarning
        )
    else:
        start_idx = np.argmin(np.abs(time - analysis_start_time))
        analysis_slice = slice(start_idx, None)
    
    def safe_max(arr):
        """Peak absolute value."""
        return np.max(np.abs(arr)).astype(float) if arr.size > 0 else np.nan

    def safe_rms2(arr, t_arr):
        """Exact RMS via trapezoidal integration (handles variable time-step)."""
        if arr.size > 1:
            squared_integral = np.trapezoid(y=arr**2, x=t_arr)
            delta_t = t_arr[-1] - t_arr[0]
            return np.sqrt(squared_integral / delta_t).astype(float)
        return np.nan

    def safe_mean2(arr, t_arr):
        """Exact average via trapezoidal integration over a variable-timestep array."""
        if arr.size > 1:
            delta_t = t_arr[-1] - t_arr[0]
            if delta_t > 0:
                integral = np.trapezoid(y=arr, x=t_arr)
                return float(integral / delta_t)
        return np.nan


    metrics = {
        'voutAVG': safe_mean2(vout[analysis_slice],time[analysis_slice]),
        'ioutAVG': safe_mean2(iout[analysis_slice],time[analysis_slice]),

        'iRecRMS': safe_rms2(iRec[analysis_slice],time[analysis_slice]),
        'iRecPK': safe_max(iRec[analysis_slice]),

        'iD1PK': safe_max(iD1[analysis_slice]),
        'iD1RMS': safe_rms2(iD1[analysis_slice],time[analysis_slice]),
        'iD1AVG': safe_mean2(iD1[analysis_slice],time[analysis_slice]),

        'iD2PK': safe_max(iD2[analysis_slice]),
        'iD2RMS': safe_rms2(iD2[analysis_slice],time[analysis_slice]),
        'iD2AVG': safe_mean2(iD2[analysis_slice],time[analysis_slice]),

        'iRRMS': safe_rms2(iR[analysis_slice],time[analysis_slice]),
        'iRPK': safe_max(iR[analysis_slice]),

        'iDSRMS': safe_rms2(iDS[analysis_slice],time[analysis_slice]),
        'iDSPK': safe_max(iDS[analysis_slice]),
        'iDSoff': iDS[-1].astype(float) if len(iDS) > 0 else np.nan,
       
        'vCsRMS': safe_rms2(vcs[analysis_slice],time[analysis_slice]),
        'vCsPK': safe_max(vcs[analysis_slice]),
        
        'iLmAVG': safe_mean2(iLm[analysis_slice],time[analysis_slice]),
        'vLmPK': safe_max(vpri[analysis_slice]),

        'iCoRMS': safe_rms2(ico[analysis_slice],time[analysis_slice]),

        'max_dt_win_sim': float(compute_dead_time(time, iDS, fsw,    # dead-time window available for ZVS [s]
                                               dead_time_threshold)),
    }

    if return_arrays:
        arrays = {
            'time': time,
            'vab': vab,
            'vpri': vpri,
            'vout': vout,
            'iR': iR,
            'iLm': iLm,
            'iD1': iD1,
            'iD2': iD2,
            'iRec': iRec,
            'iout': iout,
            'vcs': vcs,
            'ico': ico,
            'iDS': iDS
        }
        return {**metrics, **arrays}
    else:
        return metrics

