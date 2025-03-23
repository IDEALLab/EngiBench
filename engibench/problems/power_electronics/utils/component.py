"""@Author: Naga Siva Srinivas Putta <nagasiva@umd.edu>."""


class SimulationParams:
    """Simulation parameters for power electronics components. LTSpice syntax."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.Tstart = params.get("Tstart")
        self.Tstop = params.get("Tstop")
        self.Tstep = params.get("Tstep")
        self.dTmax = params.get("dTmax")
        self.modifiers = params.get("modifiers")


class Resistor:
    """Resistor parameters for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.R = params.get("Resistance")


class Capacitor:
    """Capacitor parameters for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.C = params.get("Capacitance")
        self.ESR = params.get("ESR")
        self.Rp = params.get("Parallel Resistance")


class Diode:
    """Diode parameters for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.Rd = params.get("Dynamic Resistance")
        self.Vt0 = params.get("Threshold Voltage")
        self.Qrr = params.get("Qrr")


class Inductor:
    """Inductor parameters for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.L = params.get("Inductance")
        self.DCR = params.get("DCR")


class MOSFET:
    """MOSFET parameters for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters.

        t_rise = td_on + tr
        t_fall = td_off + tf
        """
        self.Ron = params.get("R_DS_on")
        self.Coss_vs_Vds = params.get("Coss vs Vds")
        self.td_on = params.get("Turn-on delay time")
        self.tr = params.get("Rise time")
        self.td_off = params.get("Turn-off delay time")
        self.tf = params.get("Fall time")
        self.Q_g_tot = params.get("Gate charge total")


class DCVoltageSource:
    """DC Voltage source for power electronics simulation."""

    def __init__(self, params: dict[str, float]):
        """Initialize simulation parameters."""
        self.V_dc = params.get("DC Voltage")
