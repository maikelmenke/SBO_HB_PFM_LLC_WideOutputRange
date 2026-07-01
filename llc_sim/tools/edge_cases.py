import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from llc_sim.tools.find_vo_target_fsw import find_vo_target


def create_comparison_table(test_cases: dict):
    """Creates a summary DataFrame and a formatted table from results."""
    import pandas as pd
    from tabulate import tabulate

    rows = {
        "Parameter": [
            "Po", "VoutAVG", "IoutAVG","fsw", "dt_max", "dt_max(%)", "IDSpk", "IDS_off", "IDSrms",
            "ID1rms", "ID1pk", "ID1avg", "ID2rms", "ID2pk", "ID2avg", 
            "IRrms", "IRpk", "VLmpk", "IRecRMS", "IRecPK", "VCspk", "VCsrms", "ICorms"
        ]
    }
    vos = {}
    req = {"time", "vout", "iR", "vab", "iDS", "iD1", "iD2", "iout"}

    for case_name, case_data in test_cases.items():
        results = case_data["results"]
        fsw_khz = case_data["fsw_khz"]

        try:
            _, rest = case_name.split("/")
            vout_str, iout_str = rest.split("@")
            voltage = float(vout_str.replace("V", "").strip())
            current = float(iout_str.replace("A", "").strip())
        except ValueError:
            print(f"[WARNING] Could not parse '{case_name}' for Po calculation. Setting Po to 0.")
            voltage, current = 0.0, 0.0

        po = voltage * current
        dt_max = results.get("max_dt_win_sim", 0.0)
        dt_pct = (dt_max * (fsw_khz * 1000.0)) * 200.0 if fsw_khz else 0.0

        rows[case_name] = [
            f"{po:.1f} W",
            f"{results.get('voutAVG', 0.0):.4f} V",
            f"{results.get('ioutAVG', 0):.4f} A",
            f"{fsw_khz:.4f} kHz",
            f"{dt_max*1e9:.4f} ns",
            f"{dt_pct:.4f}%",
            f"{results.get('iDSPK', 0):.4f} A",
            f"{results.get('iDSoff', 0):.4f} A",
            f"{results.get('iDSRMS', 0):.4f} A",
            f"{results.get('iD1RMS', 0):.4f} A",
            f"{results.get('iD1PK', 0):.4f} A",
            f"{results.get('iD1AVG', 0):.4f} A",
            f"{results.get('iD2RMS', 0):.4f} A",
            f"{results.get('iD2PK', 0):.4f} A",
            f"{results.get('iD2AVG', 0):.4f} A",
            f"{results.get('iRRMS', 0):.4f} A",
            f"{results.get('iRPK', 0):.4f} A",
            f"{results.get('vLmPK', 0):.4f} V",
            f"{results.get('iRecRMS', 0):.4f} A",
            f"{results.get('iRecPK', 0):.4f} A",
            f"{results.get('vCsPK', 0):.4f} V",
            f"{results.get('vCsRMS', 0):.4f} V",
            f"{results.get('iCoRMS', 0):.4f} A",
        ]

        if req.issubset(results):
            vos[case_name] = {k: results[k] for k in req}

    df = pd.DataFrame(rows).set_index("Parameter")
    return tabulate(df, headers="keys", tablefmt="fancy_grid"), df, vos


_SS_DEFAULTS = {
    'cycles_per_block':       30,
    'steady_state_tol':       1e-3,
    'avg_cycles':             20,
    'stable_blocks_required': 3,
    'ripple_tol_factor':      3.0,
}

def get_case_data(params, vtarget, SimCycles, TimeStep, tol, plot=False, verbose=True,
                  fsw_min=None, ss_config=None):
    """Wrapper to call the frequency finding function."""
    ss = {**_SS_DEFAULTS, **(ss_config or {})}
    result, fsw_khz = find_vo_target(
        params=params, Vtarget=vtarget, tol=tol,
        SimCycles=SimCycles, TimeStep=TimeStep, Full_Arr=plot,
        cycles_per_block=ss['cycles_per_block'],
        steady_state_tol=ss['steady_state_tol'],
        avg_cycles=ss['avg_cycles'],
        stable_blocks_required=ss['stable_blocks_required'],
        ripple_tol_factor=ss['ripple_tol_factor'],
        verbose=verbose,
        fsw_min_override=fsw_min,
    )
    return {"results": result, "fsw_khz": fsw_khz}


def process_case(case_name, base_params, SimCycles, TimeStep, tol, plot, verbose=True,
                 fsw_min=None, ss_config=None):
    try:
        ss = {**_SS_DEFAULTS, **(ss_config or {})}
        # Formato convencional: "400V/12V@10A"
        vbus_str, rest = case_name.split("/")
        vout_str, iout_str = rest.split("@")
        vbus = float(vbus_str.replace("V", "").strip())
        vtarget = float(vout_str.replace("V", "").strip())
        current = float(iout_str.replace("A", "").strip())
        if vtarget < 0 or current < 0 or vbus <= 0:
            raise ValueError("Voltages and current must be positive.")
        rload = 1e12 if abs(current) < 1e-9 else vtarget / current
        params = {**base_params, "Rload": rload, "Vbus": vbus}

        start_time = time.perf_counter()
        case_data = get_case_data(params, vtarget, SimCycles, TimeStep,
                                  tol=tol, plot=plot, verbose=verbose,
                                  fsw_min=fsw_min, ss_config=ss_config)
        duration = time.perf_counter() - start_time
        if verbose:
            fsw_found = case_data["fsw_khz"]
            _ts = (1.0 / (fsw_found * 1e3)) / params['NPTsw'] if 'NPTsw' in params else TimeStep
            print(f"[INFO]    Finished {case_name} in {duration:.2f}s"
                  f" [Vbus={vbus:.1f}V, Vout={vtarget:.1f}V, I={current:.3f}A, "
                  f"R={rload:.3f}Ω, Cycles={SimCycles}, Time-Step={_ts*1e9:.2f}ns]", flush=True)
        return case_name, case_data

    except Exception as e:
        print(f"[ERROR]   Skipping {case_name}: Invalid format or error during processing. ({e})", flush=True)
        return None, None


def plot_detailed_case_waveforms(case_name, case_data, fsw_khz,
                                 base_params, num_cycles_zoom=4,
                                 output_dir="outputs/plots", plot_dpi=300,
                                 file_format='pdf'):
    """Saves plots of the key waveforms for a given simulation result."""
    # Lazy import: avoids loading matplotlib unless plots are explicitly requested.
    import matplotlib.pyplot as plt

    # Updated with the new required keys
    req = ["time", "vout", "iR", "iDS", "iD1", "iD2", "iout", "vab"]

    if not all(k in case_data for k in req):
        print(f"[WARNING] Waveforms incomplete for {case_name}, skipping plot.")
        return

    t, vout, Ir, Ids, iD1, iD2, iout, vab = (np.asarray(case_data[k]) for k in req)
    if t.size < 2:
        print(f"[WARNING] Not enough data points for {case_name}, skipping plot.")
        return
        
    tm = t * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    fig.suptitle(f'LLC {case_name} (fsw={fsw_khz:.3f} kHz)', fontsize=14)
    
    # -------------------------------------------------------------------------
    # PLOT 1: Vout and Resonant Current (Full Transient)
    # -------------------------------------------------------------------------
    axes[0].plot(tm, vout, color='blue', label="Vout")
    axes[0].set(title="Output Voltage (Vout) & Resonator Current (Ir) - Full Transient", 
                xlabel="Time (ms)", ylabel="Voltage (V)")
    axes[0].tick_params(axis='y', labelcolor='blue')
    
    ax0b = axes[0].twinx()
    ax0b.plot(tm, Ir, color='red', label="Ir", linewidth=0.8)
    ax0b.set_ylabel("Current (A)", color='red')
    ax0b.tick_params(axis='y', labelcolor='red')
    
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax0b.get_legend_handles_labels()
    ax0b.legend(lines + lines2, labels + labels2, loc='upper right')
    axes[0].grid(True)
    
    # -------------------------------------------------------------------------
    # ZOOM MASK (For the next two plots)
    # -------------------------------------------------------------------------
    if fsw_khz > 0:
        Tms = 1 / fsw_khz
        zoom_mask = tm >= tm[-1] - num_cycles_zoom * Tms
    else:
        zoom_mask = slice(-500, None)
        
    # -------------------------------------------------------------------------
    # PLOT 2: Primary Vab and IDS (Zoom)
    # -------------------------------------------------------------------------
    axes[1].plot(tm[zoom_mask], Ids[zoom_mask], color='green', label="Ids", linewidth=1.5)
    axes[1].set(title="Primary Current (Ids) & Bridge Voltage (Vab) - Zoom", 
                xlabel="Time (ms)", ylabel="Current (A)")
    axes[1].tick_params(axis='y', labelcolor='green')
    
    ax1b = axes[1].twinx()
    ax1b.plot(tm[zoom_mask], vab[zoom_mask], color='purple', linestyle='--', label="Vab", alpha=0.7)
    ax1b.set_ylabel("Voltage (V)", color='purple')
    ax1b.tick_params(axis='y', labelcolor='purple')
    
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper right')
    axes[1].grid(True)
    
    # -------------------------------------------------------------------------
    # PLOT 3: Diode and Output Currents (Zoom)
    # -------------------------------------------------------------------------
    axes[2].plot(tm[zoom_mask], iD1[zoom_mask], color='orange', label="iD1")
    axes[2].plot(tm[zoom_mask], iD2[zoom_mask], color='deepskyblue', label="iD2")
    axes[2].plot(tm[zoom_mask], iout[zoom_mask], color='black', label="iout", linestyle='--', linewidth=2)
    
    axes[2].set(title="Secondary Currents (iD1, iD2) & Output Current (iout) - Zoom", 
                xlabel="Time (ms)", ylabel="Current (A)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    # -------------------------------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    safe_name = case_name.replace("@", "_at_").replace("/", "_out_")
    file_name = f"waves_{safe_name}.{file_format}"
    file_path = os.path.join(output_dir, file_name)
    
    fig.savefig(file_path, dpi=plot_dpi)
    plt.close(fig)
    print(f"[INFO]    Plot saved: {file_name}")


def test_edge_cases(desired_cases_identifiers, base_params,
                             SimCycles=500, TimeStep=10e-9,
                             plot=False, show_table=False, save_csv=False,
                             max_workers=None, tol=0.1,
                             return_arrays=False, verbose=False,
                             fsw_min_per_case=None, ss_config=None,
                             plot_dir=None):
    """
    Simulates a list of LLC operating points IN PARALLEL and prints total runtime.
    """
    # Validate input
    if not isinstance(desired_cases_identifiers, (list, set)):
        print("[ERROR]   'desired_cases_identifiers' must be a list or set.")
        return []
    identifiers = [c for c in desired_cases_identifiers if isinstance(c, str)]
    if not identifiers:
        print("[WARNING] No valid string identifiers found in the input list.")
        return []

    generate_full_arrays = plot or return_arrays
    test_cases = {}

    # --- Parallel execution with timing ---
    workers = max_workers or os.cpu_count() or 1
    if verbose:
        print(f"[INFO]    Running in PARALLEL with max_workers={workers}...", flush=True)
    start_total = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                process_case,
                c,
                base_params,
                SimCycles,
                TimeStep,
                tol,
                generate_full_arrays,
                verbose,
                fsw_min_per_case.get(c) if fsw_min_per_case else None,
                ss_config,
            ): c for c in identifiers
        }

        for fut in as_completed(futs):
            name, data = fut.result()
            if name and data:
                test_cases[name] = data

    total_runtime = time.perf_counter() - start_total
    if verbose:
        print(f"[INFO]    Total simulation runtime (parallel): {total_runtime:.2f} seconds", flush=True)

    if not test_cases:
        if verbose:
            print("[WARNING] No cases were successfully processed.")
        return []

    if verbose:
        print(f"[INFO]    {len(test_cases)} cases were successfully processed.")

    # Build table (only if requested)
    df = None
    if show_table or save_csv:
        table, df, vos = create_comparison_table(test_cases)
        if show_table:
            print("\n" + "="*80)
            print("SIMULATION SUMMARY")
            print("="*80)
            print(table)
            print("="*80 + "\n")

    # Plots (only if requested)
    if plot:
        _, _, vos = create_comparison_table(test_cases)
        if vos:
            out_dir = plot_dir if plot_dir else os.path.join(os.curdir, "outputs", "plots")
            os.makedirs(out_dir, exist_ok=True)
            if verbose: print(f"[INFO]    Saving plots to '{out_dir}'...")
            for cname, data in vos.items():
                plot_detailed_case_waveforms(
                    cname,
                    data,
                    test_cases[cname]["fsw_khz"],
                    base_params,
                    num_cycles_zoom=5,
                    output_dir=out_dir
                )
        else:
            if verbose: print("[WARNING] Plotting was requested, but no waveform data was generated.")

    # CSV (only if requested)
    if save_csv and df is not None and not df.empty:
        out_dir = os.path.join(os.curdir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        fname = f'test_cases_{datetime.now():%Y%m%d_%H%M%S}.csv'
        full_path = os.path.join(out_dir, fname)
        df.to_csv(full_path)
        if verbose: print(f"[INFO]    CSV file saved to {full_path}")

    # Return compact list of results (order by case name for determinism)
    return [{"case": n, "fsw_khz": d["fsw_khz"], **d["results"]}
            for n, d in sorted(test_cases.items(), key=lambda x: x[0])]
