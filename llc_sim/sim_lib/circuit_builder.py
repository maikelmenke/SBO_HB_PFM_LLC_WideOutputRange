import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from .transformer import TrxSubCir_CT
from PySpice.Spice.Library import SpiceLibrary


def build_llc_circuit(params: dict, Vtarget=None, config=1) -> Circuit:
    """
    Builds the Half-Bridge LLC resonant circuit using PySpice.

    Parameters
    ----------
    params : dict
        Dictionary with circuit parameters:
          fsw       : switching frequency [Hz]
          Vbus      : bus voltage [V]
          Cs        : resonant capacitor [F]
          Ls        : resonant inductor [H]
          Lm        : magnetizing inductance [H]
          Co        : output capacitor [F]
          Rload     : load resistance [Ω]
          n         : turns ratio (primary/secondary)
          L1        : transformer primary inductance (optional, default 1000 H)
          Rc1       : primary copper resistance (optional, default 0.01 Ω)
          Rc2       : secondary copper resistance (optional, default 0.01 Ω)
    Vtarget : float, optional
        Initial voltage of the output capacitor [V].
    config : int
        1 = Ideal square voltage source (default)

    Returns
    -------
    Circuit
        PySpice Circuit object configured with current probes on
        the main components.

    Numerical Stability Notes
    -------------------------
    All nodes that may float receive 1 MΩ bleed resistors to GND.
    This prevents "Timestep too small" at frequencies near f1 where
    the resonant tank impedance is very high.
    Treated nodes: cs, pri, secn1, secn2, vo, and the internal nodes
    of the transformer subcircuit (see transformer.py).
    """
    fsw     = params['fsw']
    Vbus    = params['Vbus']
    CsValue = params['Cs']
    LsValue = params['Ls']
    LmValue = params['Lm']
    CoValue = params['Co']
    RL      = params['Rload']
    n       = params['n']
    L1      = params.get('L1',  1000)
    Rc1     = params.get('Rc1', 0.01)
    Rc2     = params.get('Rc2', 0.01)

    circuit = Circuit('LLC')

    # ── Device models ────────────────────────────────────────────────────────
    circuit.model('MyDiode', 'D', IS=1e-7, N=1.0, RS=0.1)

    # ── Config 1: Ideal square source ────────────────────────────────────────
    if config == 1:
        circuit.PulseVoltageSource("Ideal", 'vab', circuit.gnd,
            initial_value=0, pulsed_value=Vbus,
            pulse_width=1/(2*fsw) - 10e-9, delay_time=0,
            period=1/fsw, rise_time=20e-9, fall_time=20e-9)

        circuit.C('s', 'vab', 'cs', CsValue)
        circuit.L('s', 'cs',  'pri', LsValue)
        circuit.L('m', 'pri', circuit.gnd, LmValue)

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                            copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', circuit.gnd, 'secn1', circuit.gnd, 'secn2')

        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')

        # RC snubbers in parallel with D1 and D2 (R=100 Ω, C=470 pF in series)
        #circuit.R('snub_d1', 'secn1', 'snb1', 100)
        #circuit.C('snub_d1', 'snb1',  'vo',   470e-12)
        #circuit.R('snub_d2', 'secn2', 'snb2', 100)
        #circuit.C('snub_d2', 'snb2',  'vo',   470e-12)

        circuit.C('o', 'vo', circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)

    else:
        raise ValueError(f"config={config} not recognized. Use 1.")

    # ── Current probes ────────────────────────────────────────────────────────
    circuit.Ls.plus.add_current_probe(circuit)
    circuit.Lm.plus.add_current_probe(circuit)
    circuit.Rload.plus.add_current_probe(circuit)
    circuit.D1.plus.add_current_probe(circuit)
    circuit.D2.plus.add_current_probe(circuit)
    circuit.Co.plus.add_current_probe(circuit)

    return circuit
