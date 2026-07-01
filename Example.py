"""
Example.py — Basic LLC simulation example.

Builds an LLC resonant converter circuit, runs a single transient simulation for a fixed number of
cycles, extracts the average output voltage, and plots the output waveform.

Use this script to verify that the simulation environment (PySpice + ngspice)
is installed and working correctly end-to-end.

"""

from llc_sim import build_llc_circuit, simulate, analyze, test_edge_cases, find_vo_target
import matplotlib.pyplot as plt
import numpy as np
import time


params = {
    'Rload': 100,
    'Cs': 4.4e-9,
    'Ls': 250e-6,
    'Lm': 893e-6,
    'n': 3.4,
    'Co': 20e-6,
    'Vbus': 420,
    'fsw': 120e3,
    'NPTsw': 250,
}

SimCycles = 500   # Number of switching cycles to simulate

fsw      = params['fsw']
Tsw      = 1.0 / fsw
TimeStep = Tsw / params['NPTsw']

circuit  = build_llc_circuit(params, config=1)

start_time = time.perf_counter()
analysis = simulate(circuit, params['fsw'], TimeStep, SimCycles)
end_time = time.perf_counter()
duration = end_time - start_time

results  = analyze(analysis, params['fsw'], TimeStep, SimCycles)

print(f"The simulation took {duration*1e3:.1f} ms to execute.")
print(f"Average output voltage: {results['voutAVG']:.2f} V")

plt.figure(figsize=(10, 6))
plt.plot(np.array(analysis.time) * 1000, np.array(analysis['vo']), label='Output Voltage ($V_o$)', linewidth=2)
plt.xlim(0, Tsw * SimCycles * 1e3)
plt.xlabel('Time (ms)')
plt.ylabel('Output Voltage (V)')
plt.title('Transient Analysis - Output Voltage')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()