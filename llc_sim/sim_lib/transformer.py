from PySpice.Spice.Netlist import SubCircuit


class TrxSubCir_CT(SubCircuit):
    """
    Center-tap transformer subcircuit with ideal magnetic coupling (K=1).

    Topology:
        Primary   : input_plus → Rprimary → node_1 → Lprimary → input_minus
        Secondary S1: output_gnd → Lsecondary_s1 → node_2 → Rsecondary_s1 → output_s1
        Secondary S2: output_gnd → Lsecondary_s2 → node_3 → Rsecondary_s2 → output_s2
    """
    __nodes__ = ('input_plus', 'input_minus', 'output_s1', 'output_gnd', 'output_s2')

    def __init__(self, name, turn_ratio,
                 primary_inductance=100,
                 copper_resistance_primary=0.001,
                 copper_resistance_secondary=0.001):

        SubCircuit.__init__(self, name, *self.__nodes__)

        secondary_inductance = primary_inductance / float(turn_ratio ** 2)

        # ── Primary ───────────────────────────────────────────────────────────
        self.R('primary',    'input_plus', 1, copper_resistance_primary)
        self.L('primary',    1, 'input_minus', primary_inductance)

        # ── Secondary S1 ──────────────────────────────────────────────────────
        self.R('secondary_s1', 2, 'output_s1', copper_resistance_secondary)
        self.L('secondary_s1', 2, 'output_gnd', secondary_inductance)

        # ── Secondary S2 ──────────────────────────────────────────────────────
        self.R('secondary_s2', 3, 'output_s2', copper_resistance_secondary)
        self.L('secondary_s2', 'output_gnd', 3, secondary_inductance)

        # ── Magnetic coupling (K=1) ───────────────────────────────────────────
        self.raw_spice = (
            'Kcoupling  Lprimary Lsecondary_s1 1\n'
            'Kcoupling2 Lprimary Lsecondary_s2 1\n'
            'Kcoupling3 Lsecondary_s1 Lsecondary_s2 1'
        )