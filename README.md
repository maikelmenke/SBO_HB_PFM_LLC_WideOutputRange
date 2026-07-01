# LLC Resonant Converter — Analysis & Design Framework

A Python-based framework for simulation and design of LLC resonant converters (Half-Bridge, Full-Wave Rectifier) using PySpice/ngspice.

📄 **[Documentation (PDF)](Documentation__LLC_Resonant_Converter_Simulation_Based_Analysis_Framework.pdf)** — full methodology, circuit model, steady-state detection algorithm, and optimisation formulation.

---

## ▶️ How to Run

Open `code/main.py` and set each flag to `True` or `False` to choose which examples to execute:

```python
RUN_01_BASIC     = True    # Basic single transient simulation
RUN_02_SS        = True    # Simulate until steady state (block-by-block)
RUN_03_PEAK      = True    # Peak-gain frequency mapping
RUN_04_VO_SINGLE = True    # Single operating point — convergence plot
RUN_05_VO_BATCH  = True    # Batch operating-point evaluation
RUN_GA           = False   # GA optimisation — long-running, off by default
RUN_PLOT_GA      = False   # Plot GA results (reads CSVs from /results/ga/)
```

Then click **Reproducible Run**. Results are saved to the **Files → results/** panel.

---

## 📋 Examples

| Flag | Script | Description |
|---|---|---|
| `RUN_01_BASIC` | `Example.py` | Single transient simulation — sanity check |
| `RUN_02_SS` | `Example_Simulate_Until_SS.py` | Simulates cycle-by-cycle until steady state is detected; saves waveform and metrics plots |
| `RUN_03_PEAK` | `Example_Find_Vpeak_ss.py` | Maps the peak-gain curve (maximum achievable Vo vs. fsw) |
| `RUN_04_VO_SINGLE` | `Example_find_Vo_target_single.py` | Finds the switching frequency for a target output voltage at a single operating point; saves convergence plot |
| `RUN_05_VO_BATCH` | `Example_Find_Vo_target.py` | Evaluates multiple operating points in parallel; saves a comparison table (CSV) and waveform PDFs |
| `RUN_GA` | `opt/main_ga.py` | Multi-objective GA optimisation (NSGA-II via pymoo) — can take hours; results in `results/ga/` |
| `RUN_PLOT_GA` | `opt/plot_ga.py` | Generates PDF plots from GA CSV results already in `results/ga/` |

> **Tip:** To run only one example, set all flags to `False` except the one you want.

---

## 📁 Output Files

All outputs are written to `/results/` and appear in the **Files → results/** panel after the run completes.

| Example | Output location | Contents |
|---|---|---|
| Example 1 | `results/output/` | Voltage/current waveform PDF |
| Example 2 | `results/output/Find_steady_state/` | Steady-state waveform and metrics plots |
| Example 3 | `results/output/Find_Vpeak_ss/` | Peak-gain curve PDF |
| Example 4 | `results/output/Find_Vo_target/` | Convergence and waveform PDFs |
| Example 5 | `results/output/Find_Vo_target/` | Per-operating-point PDFs + `results.csv` |
| GA | `results/ga/` | Pareto front CSV + convergence log |
| Plot GA | `results/ga/` | PDF plots of GA Pareto front and design variables |

---

## 📄 License

GNU General Public License v3.0 or later.
Copyright (c) 2025 Maikel Fernando Menke, Eduardo Bayona Blanco, Pedro Pappis, Joshua R. Neusser, Guirguis Zaki Guirguis Abdelmessih, Marco Antonio Dalla Costa, Jose Marcos Alonso.
