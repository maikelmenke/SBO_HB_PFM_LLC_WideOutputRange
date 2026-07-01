"""
Example_Find_Vo_target.py — Operating-point evaluation for LLC resonant converters.

Workflow:
  1. Define converter parameters and operating points.
  2. Simulate all operating points in parallel with ``test_edge_cases()``, which
     automatically finds the switching frequency for each target voltage and
     load using ``find_vo_target()``.

Output:
  - Console table: simulation metrics per operating point.
  - PDF plots: waveforms saved in the ``outputs/`` directory.
  - CSV file: raw simulation metrics for all operating points (``outputs/*.csv``).

Edit the ``params`` dict and ``OperatingPoints`` list to match your design.
"""

import os

from llc_sim import test_edge_cases

# Base converter parameters
params = {
    'Vbus': 420,    # Nominal input voltage
    'VoMin':70,     # Minimum output voltage
    'VoMax':160,    # Maximum output Voltage
    'Vox':100,      # Maximum voltage for full power at Io.Max
    'Po':150,       # Nominal output power
    'Io':1.5,       # Maximum output current
    'Iox':0.94,     # Maximum current for full power at Vo.Max
    'IoMin':0.1,    # Minimum output current
    'Cs': 4.4e-9,
    'Ls': 250e-6,
    'Lm': 893e-6,
    'n':3.4,
    'Co': 20e-6,
    'NPTsw': 250,
    'dead_time':100e-9 # not used
}

# Generate the operating points to evaluate converter performance >>> USER DUTY <<<
OperatingPoints = [
    f"{params['Vbus']}V/{params['VoMin']}V@{params['Io']}A",
    f"{params['Vbus']}V/{params['Vox']}V@{params['Io']}A",
    f"{params['Vbus']}V/{params['VoMax']}V@{params['Iox']}A",
    #f"{params['Vbus']}V/{params['VoMin']}V@{params['IoMin']}A",
    #f"{params['Vbus']}V/{params['Vox']}V@{params['IoMin']}A",
    #f"{params['Vbus']}V/{params['VoMax']}V@{params['IoMin']}A",
]

if __name__ == "__main__":

    # Steady-state convergence settings for simulate_until_steady_state
    ss_config = {
        'cycles_per_block':       15,    # cycles simulated per evaluation block
        'steady_state_tol':       1e-3,  # relative convergence tolerance
        'avg_cycles':             10,    # cycles averaged for Vo metric
        'stable_blocks_required': 3,     # consecutive stable blocks to declare SS
        'ripple_tol_factor':      2.0,   # ripple criterion = steady_state_tol × this factor
    }

    test_edge_cases(
        OperatingPoints,
        params,
        show_table=True,
        plot=True,
        tol=0.1,
        save_csv=True,
        ss_config=ss_config,
        plot_dir=os.path.join("outputs", "Find_Vo_target"),
    )
