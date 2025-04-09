# ruff: noqa: N806 # Upper case
# ruff: noqa: PLR0912, PLR0915 # Too many branches. Too many statements
# ruff: noqa: FIX002 # for TODO

"""Power Electronics problem."""

from __future__ import annotations

import os
import subprocess
from typing import Any, NoReturn

from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import Problem
from engibench.problems.power_electronics.utils import component as cmpt
from engibench.problems.power_electronics.utils import data_sheet as ds
from engibench.problems.power_electronics.utils import dc_dc_efficiency_ngspice as dc_lib_ng
from engibench.problems.power_electronics.utils.ngspice import NgSpice
from engibench.problems.power_electronics.utils.str_to_value import str_to_value


class PowerElectronics(Problem[npt.NDArray]):
    r"""Power Electronics parameter optimization problem.

    ## Problem Description
    This problem simulates a power converter circuit which has a fixed circuit topology. There are 5 switches, 4 diodes, 3 inductors and 6 capacitors.
    The circuit topology is fixed. It is defined in the netlist file `5_4_3_6_10-dcdc_converter_1.net`.
    By changing circuit parameters such as capacitance, we rewrite the netlist file and use ngSpice to simulate the circuits to get the performance metrics, which are defined as the objectives of this problem.
    You can use this problem to train your regression model. You can also try to find the optimal circuit parameters that minimize objectives.

    ## Design space
    The design space is represented by a 20-dimensional vector that defines the cicuit parameters.
    - `C0`, `C1`, `C2`, `C3`, `C4`, `C5`: Capacitor values in Farads for each capacitor. Range: [1e-6, 2e-5].
    - `L0`, `L1`, `L2`: Inductor values in Henries for each inductance. Range: [1e-6, 1e-3].
    - `T1`: Duty cycle, the fraction of "on" time. Range: [0.1, 0.9]. Because all the 5 switches change their on/off state at the same time, we only need to set one `T1` value.
        For example, `T1 = 0.1` means that all the switches are first turned "on" for 10% of the time, then "off" for the remaining 90%. This on-off pattern repeats at a high frequency until the simulation is over.
    - `GS0_L1`, `GS1_L1`, `GS2_L1`, `GS3_L1`, `GS4_L1`: Switches `L1`. Binary values (0 or 1).
    - `GS0_L2`, `GS1_L2`, `GS2_L2`, `GS3_L2`, `GS4_L2`: Switches `L2`. Binary values (0 or 1).
        Each switch is a voltage-controlled switch. For example, `S0` is controlled by `V_GS0`, whose voltage is defined by `GS0_L1` and `GS0_L2`.
        In short, 0 means `S0` is off and 1 means `S0` is on.
        For example, When `GS0_L1 = 0` and `GS0_L2 = 1`, `S0` is first turned off for time `T1 * Ts`, and then turned on for `(1 - T1) * Ts`, where `Ts` is set to 5e-6.
        As a result, each switch can be on -> off, on -> on, off -> on, or off -> off independently.

    ## Objectives
    The objectives are defined by the following parameters:
    - `DcGain-0.25`: The ratio of load vs. input voltage. It's desired to be as close to a preset constant, such as 0.25, as possible.
    - `Voltage Ripple`: Fluctuation of voltage on the load `R0`. The lower the better.

    ## Conditions
    There is no condition for this problem.

    ## Simulator
    The simulator is ngSpice circuit simulator. You can download it based on your operating system:
    - Windows: https://sourceforge.net/projects/ngspice/files/ng-spice-rework/44.2/
    - MacOS: `brew install ngspice`
    - Linux: `sudo apt-get install ngspice`

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/power_electronics).

    ### v0

    #### Fields
    The dataset contains 3 fields:
    - `initial_design`: The 20-dimensional design variable defined above.
    - `DcGain`: The ratio of load vs. input voltage.
    - `Voltage_Ripple`: The fluctuation of voltage on the load `R0`.

    #### Creation Method
    We created this dataset in 3 parts. All the 3 parts are simulated with {`GS0_L1`, `GS1_L1`, `GS2_L1`, `GS3_L1`, `GS4_L1`} = {1, 0, 0, 1, 1} and {`GS0_L2`, `GS1_L2`, `GS2_L2`, `GS3_L2`, `GS4_L2`} = {1, 0, 1, 1, 0}.
    Here are the 3 parts:
    1. 6 capacitors and 3 inductors only take their min and max values. `T1` ranges {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. There are 2^6 * 2^3 * 9 = 4608 samples.
    2. Random sample 4608 points in the 6 + 3 + 1 = 10 dimensional space. Min and max values in each dimension will not be sampled.
    3. Latin hypercube sample 4608 points in the 6 + 3 + 1 = 10 dimensional space. Each dimension is split into 10 intervals. Min and max values in each dimension will not be sampled.

    ## References
    If you use this problem in your research, please cite the following paper:

    ## Lead
    Xuliang Dong @ liangXD523
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("DcGain", ObjectiveDirection.MINIMIZE),
        ("Voltage_Ripple", ObjectiveDirection.MAXIMIZE),
    )
    conditions: frozenset[tuple[str, Any]] = frozenset()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32)
    dataset_id = "IDEALLab/power_electronics_v0"
    container_id = None
    _dataset = None

    def __init__(self, ngspice_path: str | None = None) -> None:
        """Initializes the Power Electronics problem.

        Args:
            ngspice_path: The path to the ngspice executable for Windows.
        """
        super().__init__()

        source_dir = os.path.dirname(os.path.abspath(__file__))  # The absolute path of power_electronics/
        # TODO: check if this works from another repo like EngiOpt

        # Initialize ngspice wrapper
        self.ngspice = NgSpice(ngspice_path)

        self.netlist_dir = os.path.normpath(os.path.join(source_dir, "./data/netlist"))
        if not os.path.exists(self.netlist_dir):
            os.makedirs(self.netlist_dir)

        self.raw_file_dir = os.path.normpath(os.path.join(source_dir, "./data/raw_file"))
        if not os.path.exists(self.raw_file_dir):
            os.makedirs(self.raw_file_dir)

        self.log_file_dir = os.path.normpath(os.path.join(source_dir, "./data/log_file"))
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

        self.components = {"V": 0, "R": 1, "C": 2, "S": 3, "L": 4, "D": 5}

        self.original_netlist_path: str = ""  # Use absolute path. E.g. ".... engibench/problems/power_electronics/data/netlist/5_4_3_6_10-dcdc_converter_1.net"
        self.rewrite_netlist_path: str = (
            ""  # It will be modified in __rewrite_netlist() depending on mode='batch' or mode='control'.
        )
        self.bucket_id: str = ""

        self.netlist_name: str = ""
        self.log_file_path: str = ""
        self.raw_file_path: str = ""

        self.edge_map: dict[str, list[int]] = {"V0": [0, 0], "R0": [0, 0]}
        self.cmp_edg_str: str = ""

        # components of the design variable
        self.capacitor_val: list[float] = []  # range: [1e-6, 2e-5]
        self.inductor_val: list[float] = []  # range: [1e-6, 1e-3]
        self.switch_T1: list[float] = []  # range: [0.1, 0.9]
        self.switch_T2: list[float] = []  # Constant. All 1 for now
        self.switch_L1: list[float] = []  # Binary.
        self.switch_L2: list[float] = []  # Binary.

        self.cmp_cnt: dict[str, int] = {"V": 1, "R": 1, "C": 0, "S": 0, "L": 0, "D": 0}

        self.simulation_results: np.ndarray = np.array([])

        self.G: nx.Graph = nx.Graph()
        self.color_dict: dict[str, str] = {"R": "b", "L": "g", "C": "r", "D": "yellow", "V": "orange", "S": "purple"}

    def load_netlist(self, original_netlist_path: str, bucket_id: str) -> None:
        """Save the original netlist path and bucket id. Create paths for log and raw files."""
        self.original_netlist_path = os.path.abspath(original_netlist_path)
        self.bucket_id = bucket_id  # Alternatively, we can get this from self.original_netlist_path.

        self.netlist_name = (
            self.original_netlist_path.replace("\\", "/").split("/")[-1].removesuffix(".net")
        )  # python 3.9 and newer
        self.rewrite_netlist_path = os.path.join(self.netlist_dir, f"rewrite_{self.netlist_name}.net")
        self.log_file_path = os.path.normpath(os.path.join(self.log_file_dir, f"{self.netlist_name}.log"))
        self.raw_file_path = os.path.normpath(os.path.join(self.raw_file_dir, f"{self.netlist_name}.raw"))

    def __process_topology(self, sweep_data: list[float]) -> None:
        """Read from self.original_netlist_path to get the topology. With the argument sweep_dict (C, L, T1, T2, L1, L2), set variables for rewriting.

        It only keeps component lines and .PARAM lines. So the following lines are discarded in this process: .model, .tran, .save etc.
        ----------
        sweep_dict : dict. variable.
        """
        self.cmp_edg_str = ""  # reset
        self.G = nx.Graph()  # reset

        calc_comp_count = {"V": 0, "R": 0, "C": 0, "S": 0, "L": 0, "D": 0}
        ref_comp_count = {"V": 1, "R": 1, "C": 0, "S": 0, "L": 0, "D": 0}
        num_comp = self.bucket_id.split("_")
        ref_comp_count["S"] = int(num_comp[0])
        ref_comp_count["D"] = int(num_comp[1])
        ref_comp_count["L"] = int(num_comp[2])
        ref_comp_count["C"] = int(num_comp[3])

        self.capacitor_val = sweep_data[: ref_comp_count["C"]]  # 6 capacitors
        self.inductor_val = sweep_data[ref_comp_count["C"] : ref_comp_count["C"] + ref_comp_count["L"]]  # 3 inductors
        self.switch_T1 = [sweep_data[ref_comp_count["C"] + ref_comp_count["L"]]] * ref_comp_count["S"]  # 5 switches (T1)
        self.switch_T2 = [1.0] * ref_comp_count["S"]  # 5 switches (T2), all set to 1.0 for now
        self.switch_L1 = sweep_data[
            ref_comp_count["C"] + ref_comp_count["L"] + 1 : ref_comp_count["C"]
            + ref_comp_count["L"]
            + 1
            + ref_comp_count["S"]
        ]  # 5 switches (L1), binary values (0 or 1)
        self.switch_L2 = sweep_data[
            ref_comp_count["C"] + ref_comp_count["L"] + 1 + ref_comp_count["S"] : ref_comp_count["C"]
            + ref_comp_count["L"]
            + 1
            + ref_comp_count["S"] * 2
        ]  # 5 switches (L2), binary values (0 or 1)

        with open(self.original_netlist_path) as file:
            for line in file:
                if line.strip() != "":
                    line_ = line.replace("=", " ")  # to deal with .PARAM problems. See the comments below.
                    # liang: Note that line still contains \n at the end of it!
                    line_spl = line_.split()
                    if line_spl[0] == ".PARAM" and sweep_data is None:
                        # e.g. .PARAM V0_value = 10
                        # e.g. .PARAM C0_value = 10u
                        # e.g. .PARAM L2_value=0.001. Note the whitespace! It appears in new files provided by RTRC.
                        # e.g. .PARAM GS2_L2=0
                        value = str_to_value(line_spl[-1])
                        if line_spl[1][0] == "C":
                            self.capacitor_val.append(value)
                        elif line_spl[1][0] == "L":
                            self.inductor_val.append(value)
                        elif "T1" in line_spl[1]:
                            self.switch_T1.append(value)
                        elif "T2" in line_spl[1]:
                            self.switch_T2.append(value)
                        elif "L1" in line_spl[1]:
                            self.switch_L1.append(value)
                        elif "L2" in line_spl[1]:
                            self.switch_L2.append(value)
                    elif line[0] in self.components:
                        line_spl = line.split(" ")[:3]

                        if "RC" in line_spl[0] or "_GS" in line_spl[0]:
                            continue  # pass this line

                        self.edge_map[line_spl[0]] = [int(line_spl[1]), int(line_spl[2])]
                        self.cmp_edg_str = self.cmp_edg_str + line  # Liang: do not add a '\n' at the end of this line.
                        calc_comp_count[line[0]] = calc_comp_count[line[0]] + 1

                        self.G.add_node(line_spl[0], bipartite=0, color=self.color_dict[line[0]])
                        self.G.add_node(line_spl[1], bipartite=1, color="gray")
                        self.G.add_node(line_spl[2], bipartite=1, color="gray")
                        self.G.add_edge(line_spl[0], line_spl[1])
                        self.G.add_edge(line_spl[0], line_spl[2])

        if ref_comp_count != calc_comp_count:
            print("Error - process_topology component check")
        self.cmp_cnt = ref_comp_count

    def __rewrite_netlist(self, mode: str = "control") -> None:
        """Rewrite the netlist based on the topology and the sweep data.

        It creates the direct input file sent to ngSpice.
        The main difference between this rewrite.netlist and the original netlist is the control section that contains simulation parameters.
        """
        self.rewrite_netlist_path = os.path.join(
            self.netlist_dir, f"rewrite_{mode}_{self.netlist_name}.net"
        )  # add {mode} compared to that in __load_netlist()
        print(f"rewriting netlist to: {self.rewrite_netlist_path}")

        n_C = len(self.capacitor_val)
        n_L = len(self.inductor_val)
        n_S = len(self.switch_T1)

        with open(self.rewrite_netlist_path, "w") as file:
            self.cmp_edg_str = "* rewrite netlist\n" + self.cmp_edg_str

            RC_str = ""
            for i in range(n_C):
                RC_str += f"RC{i} {self.edge_map[f'C{i}'][0]} {self.edge_map[f'C{i}'][1]} 100meg\n"

            self.cmp_edg_str += f"{RC_str}\n.PARAM V0_value=1000\n"

            for i in range(n_C):
                self.cmp_edg_str += f".PARAM C{i}_value = {self.capacitor_val[i]}\n"

            for i in range(n_L):
                self.cmp_edg_str += f".PARAM L{i}_value = {self.inductor_val[i]}\n"

            self.cmp_edg_str += ".PARAM R0_value = 10\n\n.model Ideal_switch SW (Ron=1m Roff=10Meg Vt=0.5 Vh=0 Lser=0 Vser=0)\n.model Ideal_D D\n\n"

            self.cmp_edg_str += "V_GSref_D  gs_ref_D 0 pwl(0 1 {GS0_T1-10e-9} 1 {GS0_T1} 0 {GS0_T2-10e-9} 0 {GS0_Ts} 1) r=0\nV_GSref_Dc  gs_ref_Dc 0 pwl(0 0 {GS0_T1-10e-9} 0 {GS0_T1} 1 {GS0_T2-10e-9} 1 {GS0_Ts} 0) r=0\n"

            for i in range(n_S):
                self.cmp_edg_str += f"V_GS{i} GS{i} 0 pwl(0 {{GS{i}_L1}} {{GS{i}_T1-10e-9}} {{GS{i}_L1}} {{GS{i}_T1}} {{GS{i}_L2}} {{GS{i}_T2-10e-9}} {{GS{i}_L2}} {{GS{i}_Ts}} {{GS{i}_L1}}) r=0\n"

            for i in range(n_S):
                self.cmp_edg_str += f".PARAM GS{i}_Ts = {self.switch_T2[i] * 5}e-06\n"
                self.cmp_edg_str += f".PARAM GS{i}_T1 = {self.switch_T1[i] * 5}e-06\n"
                self.cmp_edg_str += f".PARAM GS{i}_T2 = {self.switch_T2[i] * 5}e-06\n"
                self.cmp_edg_str += f".PARAM GS{i}_L1 = {self.switch_L1[i]}\n"
                self.cmp_edg_str += f".PARAM GS{i}_L2 = {self.switch_L2[i]}\n"

            if mode == "batch":
                """
                Here is an example:
                .save @C0[i] @RC0[i] @C1[i] @RC1[i] @C2[i] @RC2[i] @C3[i] @RC3[i] @C4[i] @RC4[i] @C5[i] @RC5[i] @L0[i] @L1[i] @L2[i] @S0[i] @S1[i] @S2[i] @S3[i] @S4[i] @D0[i] @D1[i] @D2[i] @D3[i] @R0[i]
                .save all
                """
                self.cmp_edg_str += "\n.save "
                for i in range(n_C):
                    self.cmp_edg_str += f"@C{i}[i] @RC{i}[i] "
                for i in range(n_L):
                    self.cmp_edg_str += f"@L{i}[i] "
                for i in range(n_S):
                    self.cmp_edg_str += f"@S{i}[i] "
                for i in range(self.cmp_cnt["D"]):
                    self.cmp_edg_str += f"@D{i}[i] "
                self.cmp_edg_str += "@R0[i]\n"  # This line should be outside the for loop, otherwise it will be duplicated.
                self.cmp_edg_str += ".save all\n"

                self.cmp_edg_str += ".tran 5n 1.06m 1m 5n uic\n"
                # The following .meas(ure) can be replaced by .print, although the former is tested and preferred, while the latter has not been tested in all cases.
                self.cmp_edg_str += f".meas TRAN Vo_mean avg par('V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})') from = 1m to = 1.06m\n"
                self.cmp_edg_str = self.cmp_edg_str + ".meas TRAN gain param='Vo_mean/1000'\n"
                self.cmp_edg_str += f".meas TRAN Vpp pp par('V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})') from = 1m to = 1.06m\n"
                self.cmp_edg_str += ".meas TRAN Vpp_ratio param = 'Vpp / Vo_mean'\n"  # Ripple voltage
                self.cmp_edg_str += "\n.end"

            elif mode == "control":
                self.cmp_edg_str += "\n.control\nsave "
                for i in range(n_C):
                    self.cmp_edg_str += f"@C{i}[i] @RC{i}[i] "
                for i in range(n_L):
                    self.cmp_edg_str += f"@L{i}[i] "
                for i in range(n_S):
                    self.cmp_edg_str += f"@S{i}[i] "
                for i in range(self.cmp_cnt["D"]):
                    self.cmp_edg_str += f"@D{i}[i] "
                self.cmp_edg_str += "@R0[i]\n"  # This line should be outside the for loop, otherwise it will be duplicated.
                self.cmp_edg_str += "save all\n"

                self.cmp_edg_str += "tran 5n 1.06m 1m 5n uic\n"
                self.cmp_edg_str += f"let Vdiff = V({self.edge_map['R0'][0]}) - V({self.edge_map['R0'][1]})\n"
                self.cmp_edg_str += "meas TRAN Vo_mean avg Vdiff from = 1m to = 1.06m\n"
                self.cmp_edg_str += "meas TRAN Vpp pp Vdiff from = 1m to = 1.06m\n"
                self.cmp_edg_str += "let Gain = Vo_mean / 1000\nlet Vpp_ratio = Vpp / Vo_mean\nprint Gain, Vpp_ratio\nrun\nset filetype = binary\n"
                self.cmp_edg_str += f"write {self.raw_file_path}\n"

                self.cmp_edg_str += "quit\n.endc\n\n.end"

            file.write(self.cmp_edg_str)
            file.close()  # explicitly added this line to ensure that the rewritten netlist file is closed

    def __calculate_efficiency(self) -> tuple[float, int, float, float]:
        capacitor_model, inductor_model, switch_model, diode_model = [], [], [], []
        max_comp_size = 12
        Ts = 5e-6
        Fs = 1 / Ts

        switch_model.extend([cmpt.MOSFET(dict(zip(ds.MOSFET_properties, ds.APT40SM120B))) for _ in range(max_comp_size)])
        diode_model.extend([cmpt.Diode(dict(zip(ds.diode_properties, ds.APT30SCD65B))) for _ in range(max_comp_size)])

        try:
            # Use the ngspice wrapper to run the simulation
            self.ngspice.run(self.rewrite_netlist_path, self.log_file_path)

            for i in range(self.cmp_cnt["C"]):
                # Assuming dissipiation factor = 5 at 200Khz freq;  ESR = Disspiation_factor/ 2*pi*f*C
                # Currently we are using Parallel resistance (component "RC") as 100Meg
                # Using https://www.farnell.com/datasheets/2167237.pdf for ESR
                cap_model_values = [self.capacitor_val[i], 1 / np.sqrt(10), 1e8]
                capacitor_model.append(cmpt.Capacitor(dict(zip(ds.capacitor_properties, cap_model_values))))
            for i in range(self.cmp_cnt["L"]):
                # Using this inductor data sheet for now: https://www.eaton.com/content/dam/eaton/products/electronic-components/resources/data-sheet/eaton-fp1206-high-current-power-inductors-data-sheet.pdf
                ind_model_values = [self.inductor_val[i], 0.43e-3]
                inductor_model.append(cmpt.Inductor(dict(zip(ds.inductor_properties, ind_model_values))))

            err, P_loss, P_src = dc_lib_ng.metric_compute_DC_DC_efficiency_ngspice(
                self.raw_file_path,
                0.001,
                self.cmp_cnt,
                self.edge_map,
                capacitor_model,
                inductor_model,
                switch_model,
                diode_model,
                10,
                0,
                self.switch_L1,
                self.switch_L2,
                [float(ind) * Ts for ind in self.switch_T1],
                [float(ind) * Ts for ind in self.switch_T2],
                Fs,
            )
            efficiency = (P_src - P_loss) / P_src

            error_report = err
            if efficiency < 0 or efficiency > 1:
                error_report = 1  # report invalid efficiency calculation

        except subprocess.CalledProcessError as err:
            efficiency = np.nan
            error_report = 2  # bit 1 will be 1 to report Process error such as invalid circuit
            P_loss = np.nan
            P_src = np.nan

        except subprocess.TimeoutExpired:
            efficiency = np.nan
            error_report = 4  # bit 2 will be 1 to report Timeout
            P_loss = np.nan
            P_src = np.nan

        return efficiency, error_report, P_loss, P_src

    def __parse_log_file(self) -> tuple[float, float]:
        DcGain, VoltageRipple = np.nan, np.nan
        with open(self.log_file_path) as log:
            lines = log.readlines()
            for line in lines:
                if line.strip() != "":
                    parts = line.split()
                    if parts[0] == "gain":
                        DcGain = float(parts[2])
                    elif parts[0] == "vpp_ratio":
                        VoltageRipple = float(parts[2])
        return DcGain, VoltageRipple  # type: ignore

    def simulate(self, design_variable: list[float]) -> npt.NDArray:
        """Simulates the performance of a Power Electronics design.

        Args:
            design_variable: sweep data. It is a list of floats representing the design parameters for the simulation.
                    - Capacitor values (C0, C1, C2, ...) in Farads
                    - Inductor values (L0, L1, L2, ...) in Henries
                    - Switch parameter T1 duty cycle. Fraction [0.1, 0.9]. All switches have the same T1.
                    - Switch parameter T2 is not included. Set to constant 1.0 for all switches.
                    - Switch parameter (L1_1, L1_2, L1_3, ...). Binary (0 or 1).
                    - Switch parameter (L2_1, L2_2, L2_3, ...). Binary (0 or 1).

        Returns:
            simulation_results: a numpy array containing the simulation results [DcGain, VoltageRipple, Efficiency].
        """
        self.__process_topology(sweep_data=design_variable)
        self.__rewrite_netlist(mode="control")
        Efficiency, error_report, _, _ = self.__calculate_efficiency()
        print(f"Error report from _calculate_efficiency: {error_report}")
        DcGain, VoltageRipple = self.__parse_log_file()
        self.simulation_results = np.array([DcGain, VoltageRipple, Efficiency])

        return self.simulation_results

    def optimize(self) -> NoReturn:
        """Optimize the design variable. Not applicable for this problem."""
        return NotImplementedError

    def render(self) -> None:
        """Render the circuit topology using NetworkX.

        It displays the Graph of the circuit topology rather than the circuit diagram.
        Each circuit element (V, L, C, etc.) is a node. Each wire/port is also a node.
        """
        plt.figure()
        node_colors = [self.G.nodes[n]["color"] for n in self.G.nodes()]
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=200, font_size=10)
        plt.show()


if __name__ == "__main__":
    # Create an empty problem
    problem = PowerElectronics()

    # Load the netlist and set the bucket_id
    original_netlist_path = (
        os.path.dirname(os.path.abspath(__file__)) + "./data/netlist/5_4_3_6_10-dcdc_converter_1.net"
    )  # sweep 141
    bucket_id = "5_4_3_6_10"
    problem.load_netlist(original_netlist_path, bucket_id)

    # Manually add the sweep data
    sweep_data = [
        0.000015600,
        0.00001948,
        0.000015185,
        0.000002442,
        0.000009287,
        0.000015377,  # C values
        0.000354659,
        0.000706596,
        0.000195361,  # L values
        0.867615857,  # T1
        1,
        0,
        0,
        1,
        1,  # GS_L1 values
        1,
        0,
        1,
        1,
        0,  # GS_L2 values
    ]

    # Simulate the problem with the provided design variable
    problem.simulate(design_variable=sweep_data)
    print(problem.simulation_results)  # [0.01244983 0.9094711  0.74045004]

    # Another set of sweep data. C0 value and GS_L1, GS_L2 values are changed.
    sweep_data = [
        1.5485e-05,
        0.00001948,
        0.000015185,
        0.000002442,
        0.000009287,
        0.000015377,  # C values
        0.000354659,
        0.000706596,
        0.000195361,  # L values
        0.867615857,  # T1
        1,
        1,
        0,
        0,
        1,  # GS_L1 values
        1,
        1,
        1,
        0,
        0,  # GS_L2 values
    ]

    # Simulate the problem with the provided design variable
    problem.simulate(design_variable=sweep_data)
    print(problem.simulation_results)  # [-1.27858   -0.025081   0.7827396]

    problem.render()
