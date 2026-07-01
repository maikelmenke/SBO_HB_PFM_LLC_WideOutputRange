import numpy as np

# Ngspice solver options used in every block simulation.
# Defined here so main_ga / any caller can import and log them for audit.
SIMULATOR_OPTIONS = {
    'method':      'gear',
    'maxord':      3,       # overridden per call via the maxord parameter
    'reltol':      1e-3,
    'abstol':      1e-6,
    'vntol':       1e-4,
    'chgtol':      1e-12,
    'cshunt':      1e-15,
    'rshunt':      1e9,
    'itl4':        200,
    'plotwinsize': 0,
}


def simulate_until_steady_state(
    circuit, fsw, TimeStep, MaxCycles,
    cycles_per_block, SteadyStateTol,
    avg_cycles, stable_blocks_required,
    SteadyStateNode='vo',
    save_all=False,
    ripple_tol_factor=3.0,
    maxord=2,
    return_block_history=False,
    return_waveforms=False,
):
    """
    Simulate circuit until steady state is reached by monitoring THREE criteria
    over the last `avg_cycles` cycles of each block:

      1. Mean criterion    : relative change in trapezoidal average < SteadyStateTol
      2. Poincare criterion: relative change in the interpolated value at the exact
                             cycle boundary < SteadyStateTol
                             (phase-corrected via linear interpolation)
      3. Ripple criterion  : relative change in peak-to-peak ripple < SteadyStateTol
                             × ripple_tol_factor
                             Catches slow envelope oscillations (e.g. LC resonance
                             between Co and output impedance) that fool criteria 1 & 2
                             when consecutive blocks land at similar oscillation phases.

    All three criteria must pass in `stable_blocks_required` consecutive blocks.

    Parameters
    ----------
    circuit : PySpice Circuit
    fsw : float — switching frequency [Hz]
    TimeStep : float — simulation time step [s]
    MaxCycles : int — hard limit on total cycles
    cycles_per_block : int — cycles per evaluation block
    SteadyStateTol : float — relative convergence tolerance (e.g. 1e-3)
    avg_cycles : int — cycles used for metrics at end of block
    stable_blocks_required : int — consecutive blocks that must pass all criteria
    SteadyStateNode : str — monitored node (default: 'vo')
    save_all : bool — if True saves all signals (needed for full analysis)
    ripple_tol_factor : float — ripple criterion tolerance multiplier

    Returns
    -------
    Tuple whose elements depend on the flags set:
      (final_analysis, cycles_to_stable)
      (final_analysis, cycles_to_stable, block_history)          if return_block_history
      (final_analysis, cycles_to_stable, block_history, waveforms) if both flags set
    Returns (None, None, ...) on failure.
    """

    # ---- helpers --------------------
    def get_node(analysis, name):
        for key in (f'v({name})', name, f'V({name})'):
            try:
                return np.asarray(analysis[key], dtype=float)
            except Exception:
                pass
        try:
            return np.asarray(analysis.nodes[name], dtype=float)
        except Exception:
            pass
        raise KeyError(f"Node vector not found for {name}")

    def get_ind_current(analysis, lname):
        # PySpice add_current_probe() inserts a 0 V source named V{lname}_plus
        # in series with the inductor; that is the correct access path for
        # circuits built with circuit_builder (which calls add_current_probe).
        lname_lo  = lname.lower()
        probe_key = f'V{lname}_plus'
        for key in (
            probe_key, probe_key.lower(),
            f'@{lname_lo}[i]', f'@{lname}[i]',
            f'i({lname_lo})', f'i({lname})',
            f'{lname_lo}#branch', f'{lname}#branch',
        ):
            try:
                return np.asarray(analysis[key], dtype=float)
            except Exception:
                pass
        for key in (lname_lo, lname):
            try:
                return np.asarray(analysis.branches[key], dtype=float)
            except Exception:
                pass
        raise KeyError(f"Inductor current not found for {lname}")

    def apply_element_ic(circ, cap_ic=None, ind_ic=None):
        if cap_ic:
            for name, V0 in cap_ic.items():
                try:
                    circ.element(name).initial_condition = float(V0)
                except Exception:
                    pass
        if ind_ic:
            for name, I0 in ind_ic.items():
                try:
                    circ.element(name).initial_condition = float(I0)
                except Exception:
                    pass

    def _extract_element_ics(analysis, idx):
        """Extract cap/ind ICs at a given array index from the analysis."""
        cap_ic = {}
        ind_ic = {}
        try:
            v_vo = get_node(analysis, 'vo')
            cap_ic['Co'] = float(v_vo[idx])
        except Exception:
            pass
        try:
            v_cs = get_node(analysis, 'cs')
            try:
                v_pos = get_node(analysis, 'vab')
            except Exception:
                v_pos = get_node(analysis, 'va')
            cap_ic['Cs'] = float(v_pos[idx] - v_cs[idx])
        except Exception:
            pass
        for lname in ('Ls', 'Lm'):
            try:
                iL = get_ind_current(analysis, lname)
                ind_ic[lname] = float(iL[idx])
            except Exception:
                pass
        return cap_ic, ind_ic

    # ----------------------------------------------------------------
    Tsw            = 1.0 / fsw
    block_duration = cycles_per_block * Tsw

    current_cycle        = 0
    previous_avg         = None
    prev_end_value       = None
    prev_ripple          = None
    final_analysis       = None
    stable_block_counter = 0
    initial_conditions   = {}
    block_history        = []
    _wf_times            = []
    _wf_nodes            = {}

    def _get_stitched():
        if not _wf_times:
            return {}
        stitched = {'time': np.concatenate(_wf_times)}
        for k, v_list in _wf_nodes.items():
            try:
                stitched[k] = np.concatenate(v_list)
            except Exception:
                pass
        return stitched

    def _ret(analysis, cycles):
        result = [analysis, cycles]
        if return_block_history:
            result.append(block_history)
        if return_waveforms:
            result.append(_get_stitched())
        return tuple(result)

    while current_cycle < MaxCycles:
        simulator = circuit.simulator()
        opts = {**SIMULATOR_OPTIONS, 'maxord': maxord}
        simulator.options(**opts)

        # Save only what's needed (robust spellings).
        # Always include the add_current_probe nodes (V{name}_plus) so that
        # get_ind_current can extract ILs / ILm for IC transfer.
        save_list = []
        for n in {SteadyStateNode, 'cs', 'vab', 'va', 'snb1', 'snb2'}:
            save_list += [f'v({n})', n]
        for lname in ('Ls', 'Lm'):
            save_list += [
                f'V{lname}_plus', f'v{lname}_plus',   # current probe (primary path)
                f'@{lname.lower()}[i]', f'@{lname}[i]',
                f'i({lname.lower()})', f'i({lname})',
                f'{lname.lower()}#branch', f'{lname}#branch',
            ]

        if not save_all:
            simulator.save(save_list)

        # ── Apply initial conditions ─────────────────────────────────────────
        if initial_conditions:
            simulator.initial_condition(**initial_conditions)
        # element ICs already set by apply_element_ic at end of previous block

        # ── Transient call ───────────────────────────────────────────────────
        try:
            analysis = simulator.transient(
                step_time=TimeStep,
                end_time=block_duration,
                use_initial_condition=True,
            )
        except Exception:
            return _ret(None, None)

        try:
            time_vector = np.asarray(analysis.time, dtype=float)
        except Exception:
            return _ret(None, None)
        if time_vector.size < 2:
            return _ret(None, None)

        t_block = time_vector   # ∈ [0, block_duration]

        final_analysis = analysis
        current_cycle += cycles_per_block

        # ── Waveform collection (block-by-block) ─────────────────────────────
        if return_waveforms:
            skip = 1 if _wf_times else 0
            block_offset = (current_cycle - cycles_per_block) * Tsw
            t_global = t_block + block_offset
            _wf_times.append(t_global[skip:])
            for nd in analysis.nodes.values():
                try:
                    arr = np.asarray(nd, dtype=float)
                    _wf_nodes.setdefault(str(nd.name), []).append(arr[skip:])
                except Exception:
                    pass

        try:
            node_vector = get_node(analysis, SteadyStateNode)
        except Exception:
            return _ret(None, None)

        # ── Metrics window: last avg_cycles cycles ────────────────────────────
        time_to_avg    = avg_cycles / fsw
        block_end_time = float(t_block[-1])          # ≈ block_duration
        avg_start_time = max(0.0, block_end_time - time_to_avg)
        avg_mask       = t_block >= avg_start_time

        seg_t = t_block[avg_mask]
        seg_v = node_vector[avg_mask]

        # Criterion 1 — trapezoidal average
        if seg_t.size >= 2:
            delta_t     = seg_t[-1] - seg_t[0]
            current_avg = float(np.trapezoid(seg_v, seg_t) / delta_t) if delta_t > 0 else float(seg_v[-1])
        elif seg_t.size == 1:
            current_avg = float(seg_v[-1])
        else:
            current_avg = float(node_vector[-1])

        # Criterion 2 — Poincaré (exact block boundary via interpolation)
        end_value = float(np.interp(block_duration, t_block, node_vector))

        # Criterion 3 — Ripple (peak-to-peak in avg window)
        if seg_v.size >= 2:
            current_ripple = float(np.max(seg_v) - np.min(seg_v))
        else:
            current_ripple = 0.0

        # ── Criteria evaluation ───────────────────────────────────────────────
        ok_avg       = False
        ok_poincare  = False
        ok_ripple    = True
        avg_error    = None
        end_error    = None
        ripple_error = None

        if previous_avg is not None:
            denom_mu  = max(abs(previous_avg), 1e-9)
            avg_error = abs((current_avg - previous_avg) / denom_mu)
            ok_avg    = avg_error < SteadyStateTol

        if prev_end_value is not None:
            denom_end   = max(abs(prev_end_value), 1e-9)
            end_error   = abs((end_value - prev_end_value) / denom_end)
            ok_poincare = end_error < SteadyStateTol

        if prev_ripple is not None:
            denom_rip    = max(abs(current_avg), 1e-9)
            ripple_error = abs((current_ripple - prev_ripple) / denom_rip)
            ok_ripple    = ripple_error < (SteadyStateTol * ripple_tol_factor)

        ok = ok_avg and ok_poincare and ok_ripple

        stable_block_counter = (stable_block_counter + 1) if ok else 0

        if return_block_history:
            block_history.append({
                'block':         current_cycle // cycles_per_block,
                't_end_s':       current_cycle * Tsw,
                'avg_vo':        current_avg,
                'end_vo':        end_value,
                'ripple':        current_ripple,
                'avg_error':     avg_error,
                'end_error':     end_error,
                'ripple_error':  ripple_error,
                'ok_avg':        ok_avg,
                'ok_poincare':   ok_poincare,
                'ok_ripple':     ok_ripple,
                'ok':            ok,
                'stable_count':  stable_block_counter,
            })

        if stable_block_counter >= stable_blocks_required:
            return _ret(final_analysis, current_cycle)

        # ── State handoff ─────────────────────────────────────────────────────
        previous_avg   = current_avg
        prev_end_value = end_value
        prev_ripple    = current_ripple

        # Boundary ICs: state at the very last sample, passed to the next block.
        initial_conditions.clear()
        try:
            for nd in analysis.nodes.values():
                initial_conditions[str(nd.name)] = float(nd[-1])
        except Exception:
            initial_conditions = {}

        boundary_cap_ic, boundary_ind_ic = _extract_element_ics(analysis, -1)
        apply_element_ic(circuit, cap_ic=boundary_cap_ic, ind_ic=boundary_ind_ic)

    return _ret(final_analysis, current_cycle)
