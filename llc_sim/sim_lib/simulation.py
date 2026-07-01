_UNSET = object()


def simulate(circuit, fsw, TimeStep, SimCycles, TimeStartSave=0,
             show_node_names=False, solver_options=None, max_time=_UNSET):
    """
    Run a transient simulation of an LLC converter circuit.

    Parameters
    ----------
    circuit : Circuit
        PySpice Circuit object to be simulated.
    fsw : float
        Switching frequency in Hz.
    TimeStep : float
        Simulation time step in seconds.
    SimCycles : int
        Number of switching cycles to simulate.
    TimeStartSave : float, optional
        Time to start saving results in simulation cycles (default is 0).
    show_node_names : bool, optional
        If True, prints the available node names (default is False).
    solver_options : dict, optional
        Override solver options passed to simulator.options(). If None, uses
        the built-in defaults. Pass SIMULATOR_OPTIONS from simulate_until_ss
        to ensure identical solver settings between both simulations.
    max_time : float or None, optional
        Maximum internal ngspice time step (TMAX in .tran statement).
        Default: 2*TimeStep. Pass None to let ngspice use TMAX=TSTEP.

    Returns
    -------
    analysis : SimulationResult
        Result object containing all node voltages and currents from the simulation.
    """

    TimeEnd = SimCycles * 1 / fsw
    TimeStartSave = TimeStartSave * 1 / fsw

    if max_time is _UNSET:
        max_time = 2 * TimeStep

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    default_opts = dict(
        method='gear',
        maxord=3,
        reltol=1e-3,
        abstol=1e-6,
        vntol=1e-4,
        chgtol=1e-12,
        cshunt=1e-15,
        rshunt=1e9,
        itl4=200,
        plotwinsize=0,
    )

    opts = {**default_opts, **(solver_options or {})}
    simulator.options(**opts)

    simulator.initial_condition(vo=0)

    transient_kwargs = dict(step_time=TimeStep, end_time=TimeEnd, start_time=TimeStartSave)
    if max_time is not None:
        transient_kwargs['max_time'] = max_time
    analysis = simulator.transient(**transient_kwargs)

    if show_node_names:
        print("Available Node Names:")
        for key in analysis.nodes:
            print(f"{key}")

    return analysis
