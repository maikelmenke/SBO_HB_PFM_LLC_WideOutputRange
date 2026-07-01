# LLC Resonant Converter Analysis & Design Framework

A Python-based framework for analysis and design of LLC resonant converters (Half-Bridge with full-wave rectifier) using PySpice/ngspice.

📄 **[Documentation (PDF)](Documentation__LLC_Resonant_Converter_Simulation_Based_Analysis_Framework.pdf)** — full methodology, circuit model, steady-state detection algorithm, and optimisation formulation.

## 🗂️ Project Structure

```text
SBO_HB_PFM_LLC_WideOutputRange/
├── llc_sim/                                    # Simulation engine (PySpice/ngspice wrapper)
│   ├── sim_lib/
│   │   ├── analysis.py                         # Waveform metrics: RMS, Mean, Peak, ZVS detection
│   │   ├── circuit_builder.py                  # Netlist builder
│   │   ├── simulation.py                       # Transient solver configuration
│   │   └── transformer.py                      # Transformer subcircuit
│   └── tools/
│       ├── edge_cases.py                       # Parallel batch testing and comparative tables
│       ├── find_peak_ss.py                     # Peak-gain mapping
│       ├── find_vo_target_fsw.py               # Find fsw for target Vo
│       └── simulate_until_ss.py                # Simulate until steady-state convergence
├── opt/                                        # Optimisation (GA via pymoo)
│   ├── LLC_problem_op_iR.py                    # Problem definition
│   ├── main_ga.py                              # Main optimisation script
│   └── plot_ga.py                              # Design result plots
│
├── Example.py                                  # Basic single transient simulation
├── Example_Simulate_Until_SS.py                # Simulate until steady state (block-by-block)
├── Example_Find_Vpeak_ss.py                    # Peak-gain frequency mapping (max Vo)
├── Example_find_Vo_target_single.py            # Single operating point with convergence plots
├── Example_Find_Vo_target.py                   # Batch evaluation of multiple operating points
├── Documentation__LLC_Resonant_Converter_Simulation_Based_Analysis_Framework.pdf
├── requirements.txt                            # Runtime dependencies
├── setup.py                                    # Package installation script
├── LICENSE                                     # GNU General Public License v3
└── README.md
```

---

## 🚀 Example Scripts

| Script | Purpose |
|---|---|
| [Example.py](#1-examplepy) | Single transient simulation — installation sanity check |
| [Example_Simulate_Until_SS.py](#2-example_simulate_until_sspy) | Simulate until steady state; extract waveforms and metrics |
| [Example_Find_Vpeak_ss.py](#3-example_find_vpeak_sspy) | Map the peak-gain curve (max Vo vs fsw) via Brent's method |
| [Example_find_Vo_target_single.py](#4-example_find_vo_target_singlepy) | Single operating point with full convergence visualization |
| [Example_Find_Vo_target.py](#5-example_find_vo_targetpy) | Batch evaluation of multiple operating points in parallel |

---

## ⚙️ Installation

### Prerequisites

- **Python ≥ 3.10** (tested with Python 3.13.1)
- **ngspice** — the circuit simulation backend. **Must be installed separately.**

> ⚠️ **ngspice version note:** PySpice 1.5 officially supports ngspice up to version 34. This project has been tested and confirmed working with **ngspice 45.2** on Windows. A version warning (`Unsupported Ngspice version 45`) is printed at startup but does **not** affect simulation correctness.

---

### Step 1 — Download ngspice

Download the **64-bit Windows DLL package** directly:

🔗 [ngspice-45.2\_dll\_64.7z (SourceForge)](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/45.2/)

Download the **ngspice-45.2**:

🔗 [ngspice-45.2\_64.7z (SourceForge)](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/45.2/)

> **Note:** The `_dll_64` package is the correct one for PySpice on Windows 64-bit. The plain `_64` package contains only the standalone executable and **does not include the DLL** required by PySpice.

Extract the downloaded archive to a permanent location, for example:
```
D:\ngspice-45.2_64\
├── Spice64\          ← contains ngspice.exe (standalone executable)
└── Spice64_dll\      ← contains ngspice.dll (shared library used by PySpice)
```
> ⚠️ **The `Spice64_dll` folder is mandatory.** PySpice communicates with ngspice through a shared library (DLL), **not** through the executable. If only `Spice64` is present, PySpice will not work.

The DLL is located at:

```
Spice64_dll\dll-vs\ngspice.dll
```

---

### Step 2 — (Optional) Add ngspice to the system PATH

To use the `ngspice` command from any terminal, add the `bin` folder to your system PATH permanently:

**Windows (PowerShell — permanent):**
```powershell
[System.Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";D:\ngspice-45.2_64\Spice64\bin",
    [System.EnvironmentVariableTarget]::User
)
```

Or, to add it only for the current terminal session:
```powershell
$env:PATH += ";D:\ngspice-45.2_64\Spice64\bin"
```

Verify it works:
```powershell
ngspice --version
```

> **Note:** This step is optional for running the framework. PySpice loads the DLL directly from inside the venv and does not require ngspice to be on the PATH.

---

### Step 3 — Clone the repository

```bash
git clone https://github.com/maikelmenke/SBO_HB_PFM_LLC_WideOutputRange.git
cd SBO_HB_PFM_LLC_WideOutputRange
```

---

### Step 4 — Create and activate a virtual environment (recommended)

Open PowerShell in the project folder. If script execution is blocked, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then create and activate the environment:

```bash
# Create
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Linux / macOS (not tested)
source .venv/bin/activate
```

After activating, your prompt changes to `(.venv) PS ...`.

---

### Step 5 — Install Python dependencies

**Option A — install as editable package** (recommended for development):

```bash
pip install -e .
```

**Option B — install from requirements file only**:

```bash
pip install -r requirements.txt
```

---

### Step 6 — Register the ngspice shared library (DLL) with PySpice

Run the post-installation script. This will **automatically download ngspice 34** and place its DLL inside the virtual environment:

```bash
pyspice-post-installation --install-ngspice-dll
```

This places the DLL at:
```
.venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll\dll-vs\ngspice.dll
```

> ⚠️ **This installs ngspice 34 by default.** If you downloaded ngspice 45 in Step 1, proceed to Step 7 to replace it.

---

### Step 7 — Replace the DLL with your ngspice version

Copy your ngspice DLL over the one installed by PySpice:

```powershell
copy "D:\ngspice-45.2_64\Spice64_dll\dll-vs\ngspice.dll" `
     ".venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll\dll-vs\ngspice.dll"
```

Verify the replacement was successful by checking the file size and date:

```powershell
dir ".venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll\dll-vs\"
```

The `ngspice.dll` file should now show the size and date matching your downloaded version (ngspice 45.2 ≈ 8.5 MB, dated Sep 2025).

---

### Step 8 — Verify the installation

```bash
pyspice-post-installation --check-install
```

A successful output ends with:

```
PySpice should work as expected
```

You will also see the version warning and the ngspice build info:

```
WARNING - Unsupported Ngspice version 45
...
** ngspice-45.2 : Circuit level simulation program
...
PySpice should work as expected
```

---

## ▶️ Quick Test

Run the basic example to confirm everything works end-to-end:

```bash
python Example.py
```

Expected output (approximately):

```
The simulation took X ms to execute.
Average output voltage: XX.XX V
```

---


## 🧬 Running the Multi-Objective Optimisation (NSGA-II)

The optimisation script (`opt/main_ga.py`) runs GA via pymoo and can take a long time. It is recommended to run it with the terminal output redirected to a log file rather than printed to the console.

## 📄 License

This project is licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

Copyright (c) 2025 Maikel Fernando Menke, Eduardo Bayona Blanco, Pedro Pappis, Joshua R. Neusser, Guirguis Zaki Guirguis Abdelmessih, Marco Antonio Dalla Costa, Jose Marcos Alonso

You are free to use, study, modify, and distribute this software under the terms of the GPL v3.
Any derivative work or software distributed together with this project must also be released
under the GPL v3 (or a compatible license).

> **Note on tool use:** Using this framework internally to design power electronics products
> (e.g., LLC converters, LED drivers) does **not** trigger GPL obligations. The GPL applies
> only when you **distribute** the simulator software itself or a modified version of it.

See the [LICENSE](LICENSE) file for the full license text, or visit
<https://www.gnu.org/licenses/gpl-3.0.html>.
