class Simulation_params:
    def __init__(self, params):
        self.Tstart = params.get("Tstart")
        self.Tstop = params.get("Tstop")
        self.Tstep = params.get("Tstep")
        self.dTmax = params.get("dTmax")
        self.modifiers = params.get("modifiers")


class Resistor:
    def __init__(self, params):
        self.R = params.get("Resistance")


class Capacitor:
    def __init__(self, params):
        self.C = params.get("Capacitance")
        self.ESR = params.get("ESR")
        self.Rp = params.get("Parallel Resistance")


class Diode:
    def __init__(self, params):
        self.Rd = params.get("Dynamic Resistance")
        self.Vt0 = params.get("Threshold Voltage")
        self.Qrr = params.get("Qrr")


class Inductor:
    def __init__(self, params):
        self.L = params.get("Inductance")
        self.DCR = params.get("DCR")


class MOSFET:
    def __init__(self, params):
        self.Ron = params.get("R_DS_on")
        # self.Qrr = params.get("MOSFET_Qrr")
        # self.trr = params.get("MOSFET_trr")
        self.Coss_vs_Vds = params.get("Coss vs Vds")
        self.td_on = params.get("Turn-on delay time")
        self.tr = params.get("Rise time")
        self.td_off = params.get("Turn-off delay time")
        self.tf = params.get("Fall time")
        self.Q_g_tot = params.get("Gate charge total")


class DC_Voltage_Source:
    def __init__(self, params):
        self.V_dc = params.get("DC Voltage")
