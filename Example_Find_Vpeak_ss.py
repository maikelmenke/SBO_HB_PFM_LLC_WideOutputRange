import os
import time
import numpy as np
import matplotlib.pyplot as plt

from llc_sim.tools.find_peak_ss import find_maximum_output_voltage_ss

# ==========================================
# Global font configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# Circuit Parameter Configuration
# ==========================================

params = {
    'Rload': 170,
    'Cs': 4.4e-9,
    'Ls': 250e-6,
    'Lm': 893e-6,
    'n': 3.4,
    'Co': 20e-6,
    'Vbus': 420,
    'fsw': 100e3,
    'NPTsw': 250
}

SAVE_PLOTS = True
SHOW_PLOTS = False

# Steady-State Detection routine configuration
ss_params = {
    'MaxCycles': 4000,              # Absolute maximum cycle limit before aborting
    'cycles_per_block': 30,         # Every X cycles the simulator checks if it has stabilized
    'SteadyStateTol': 5e-3,        # Error tolerance between one block and the next
    'avg_cycles': 20,               # Cycles used to compute the average at the end of the block
    'stable_blocks_required': 2     # How many consecutive blocks must pass the tolerance
}

if __name__ == '__main__':

    f1 = (1 / (2 * np.pi * np.sqrt(params["Cs"] * (params["Ls"] + params["Lm"]))))
    fo = (1 / (2 * np.pi * np.sqrt(params["Cs"] * params["Ls"])))

    f_min = f1
    f_max = fo + 5e3

    print(f"--- Starting Dynamic Frequency Sweep ---")
    print(f"Range: {f_min/1e3:.1f} kHz to {f_max/1e3:.1f} kHz.")

    start_time = time.time()

    best_freq, max_vout, best_results, curve_data, output_dir, convergence_history = find_maximum_output_voltage_ss(
        params=params,
        f_min=f_min,
        f_max=f_max,
        config=1,
        verbose=False,
        ss_params=ss_params,
        output_dir="outputs"
    )

    elapsed_time = time.time() - start_time

    print(f"\n--- Simulation Results ---")
    print(f"Execution time:     {elapsed_time:.2f} seconds")
    print(f"Maximum Voltage (Vo): {max_vout:.2f} V")
    print(f"Frequency:    {best_freq/1e3:.2f} kHz")

    if best_results:
        print(f"Output Current:     {best_results.get('ioutAVG', 0.0):.2f} A")

    plot_dir = os.path.join(output_dir, 'Find_Vpk')
    os.makedirs(plot_dir, exist_ok=True)

    # ==========================================
    # Voltage vs Frequency Plot
    # ==========================================
    freqs, vouts = curve_data
    freqs_khz = np.array(freqs) / 1000

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freqs_khz, vouts, color='#1f77b4', linestyle='--', marker='o',
            markersize=5, linewidth=1.2, label='Voltage Gain Curve ($V_o$)')

    best_freq_khz = best_freq / 1000
    ax.plot(best_freq_khz, max_vout, color='#d62728', marker='*',
            markersize=18, zorder=5, label=f'Peak: {max_vout:.1f} V @ {best_freq_khz:.1f} kHz')

    ax.axvline(x=best_freq_khz, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.axhline(y=max_vout, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.2)

    ax.set_title('Output Voltage vs. Switching Frequency (Dynamic SS)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Switching Frequency, $f_{sw}$ (kHz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Output Voltage, $V_o$ (V)', fontsize=13, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    legend = ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 11})
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    vin_val   = params.get('Vbus', 400)
    rload_val = params.get('Rload', 10)
    ls_val    = params.get('Ls', 50e-6) * 1e6
    lm_val    = params.get('Lm', 250e-6) * 1e6
    cs_val    = params.get('Cs', 22e-9) * 1e9
    n_val     = params.get('n', 1)

    param_text = (
        "Circuit Parameters:\n"
        f"$V_{{in}}$ = {vin_val} V\n"
        f"$R_{{load}}$ = {rload_val} $\\Omega$\n"
        f"$L_s$ = {ls_val:.1f} $\\mu$H\n"
        f"$L_m$ = {lm_val:.1f} $\\mu$H\n"
        f"$C_s$ = {cs_val:.1f} nF\n"
        f"$n$ = {n_val}"
    )

    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.0)
    ax.text(0.96, 0.05, param_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(plot_dir, 'voltage_vs_frequency_curve_ss.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # ==========================================
    # Convergence Plots (Vo and Fsw vs Iteration)
    # ==========================================
    if convergence_history:
        conv_freqs_khz = [f / 1e3 for f, _ in convergence_history]
        conv_vouts     = [v       for _, v in convergence_history]
        iterations     = list(range(1, len(convergence_history) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        fig.suptitle("Optimization Convergence (Brent's Method)", fontsize=14, fontweight='bold')

        ax1.plot(iterations, conv_vouts, color='#ff7f0e', marker='o', markersize=6,
                 linewidth=1.5, label='$V_o$ per iteration')
        ax1.axhline(y=max_vout, color='#d62728', linestyle='--', linewidth=1.2,
                    label=f'Peak: {max_vout:.2f} V')
        ax1.set_ylabel('Average Output Voltage, $V_o$ (V)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=11)
        ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        ax1.legend(prop={'size': 11})
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2.plot(iterations, conv_freqs_khz, color='#1f77b4', marker='s', markersize=6,
                 linewidth=1.5, label='$f_{sw}$ per iteration')
        ax2.axhline(y=best_freq / 1e3, color='#d62728', linestyle='--', linewidth=1.2,
                    label=f'Optimal: {best_freq/1e3:.2f} kHz')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Switching Frequency, $f_{sw}$ (kHz)', fontsize=12, fontweight='bold')
        ax2.set_xticks(iterations)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        ax2.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        ax2.legend(prop={'size': 11})
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(plot_dir, 'convergence_plot.pdf'), bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        plt.close()
