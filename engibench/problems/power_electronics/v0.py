# ruff: noqa: N806, N815 # Upper case
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
from engibench.problems.power_electronics.utils.config import Config
from engibench.problems.power_electronics.utils.ngspice import NgSpice
from engibench.problems.power_electronics.utils.parse_topology import parse_topology
from engibench.problems.power_electronics.utils.process_log_file import process_log_file
from engibench.problems.power_electronics.utils.process_sweep_data import process_sweep_data
from engibench.problems.power_electronics.utils.rewrite_netlist import rewrite_netlist


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

    def __init__(
        self,
        original_netlist_path: str = "./data/netlist/5_4_3_6_10-dcdc_converter_1.net",
        mode: str = "control",
        bucket_id: str = "5_4_3_6_10",
        ngspice_path: str | None = None,
    ) -> None:
        """Initializes the Power Electronics problem.

        Args:
            original_netlist_path: The path to the original netlist file. Accepts both relative and absolute paths.
            bucket_id: The bucket ID for the netlist file. E.g. "5_4_3_6_10".
            mode: The mode for the simulation. Default to "control". mode = "batch" is for development.
            ngspice_path: The path to the ngspice executable for Windows.
        """
        super().__init__()

        self.config = Config(
            original_netlist_path=original_netlist_path,
            mode=mode,
            bucket_id=bucket_id,
        )
        print(self.config)

        # Initialize ngspice wrapper
        self.ngspice = NgSpice(ngspice_path)

        # Each row contains DcGain and Voltage Ripple
        self.simulation_results: np.ndarray = np.array([])

        # For render()
        self.G: nx.Graph | None = None

    def __calculate_efficiency(self) -> tuple[float, int, float, float]:
        capacitor_model, inductor_model, switch_model, diode_model = [], [], [], []
        max_comp_size = 12
        Ts = 5e-6
        Fs = 1 / Ts

        switch_model.extend([cmpt.MOSFET(dict(zip(ds.MOSFET_properties, ds.APT40SM120B))) for _ in range(max_comp_size)])
        diode_model.extend([cmpt.Diode(dict(zip(ds.diode_properties, ds.APT30SCD65B))) for _ in range(max_comp_size)])

        try:
            # Use the ngspice wrapper to run the simulation
            self.ngspice.run(self.config.rewrite_netlist_path, self.config.log_file_path)

            for i in range(self.config.n_C):
                # Assuming dissipiation factor = 5 at 200Khz freq;  ESR = Disspiation_factor/ 2*pi*f*C
                # Currently we are using Parallel resistance (component "RC") as 100Meg
                # Using https://www.farnell.com/datasheets/2167237.pdf for ESR
                cap_model_values = [self.config.capacitor_val[i], 1 / np.sqrt(10), 1e8]
                capacitor_model.append(cmpt.Capacitor(dict(zip(ds.capacitor_properties, cap_model_values))))
            for i in range(self.config.n_L):
                # Using this inductor data sheet for now: https://www.eaton.com/content/dam/eaton/products/electronic-components/resources/data-sheet/eaton-fp1206-high-current-power-inductors-data-sheet.pdf
                ind_model_values = [self.config.inductor_val[i], 0.43e-3]
                inductor_model.append(cmpt.Inductor(dict(zip(ds.inductor_properties, ind_model_values))))

            err, P_loss, P_src = dc_lib_ng.metric_compute_DC_DC_efficiency_ngspice(
                self.config.raw_file_path,
                0.001,
                self.config.n_S,
                self.config.n_C,
                self.config.n_L,
                self.config.n_D,
                self.config.edge_map if self.config.edge_map is not None else {},
                capacitor_model,
                inductor_model,
                switch_model,
                diode_model,
                10,
                0,
                self.config.switch_L1 if self.config.switch_L1 is not None else [],
                self.config.switch_L2 if self.config.switch_L2 is not None else [],
                [float(ind) * Ts for ind in self.config.switch_T1] if self.config.switch_T1 is not None else [],
                [float(ind) * Ts for ind in self.config.switch_T2]
                if self.config.switch_T2 is not None
                else [],  # TODO: any fix for these if?
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

    def simulate(self, design_variable: list[float]) -> npt.NDArray:
        """Simulates the performance of a Power Electronics design.

        Args:
            design_variable: sweep data. It is a list of floats representing the design parameters for the simulation.
                    In general, they are:
                    - Capacitor values (C0, C1, C2, ...) in Farads.
                    - Inductor values (L0, L1, L2, ...) in Henries.
                    - Switch parameter T1 duty cycle. Fraction [0.1, 0.9]. All switches have the same T1.
                    - Switch parameter T2 is not included. Set to constant 1.0 for all switches.
                    - Switch parameter (L1_1, L1_2, L1_3, ...). Binary (0 or 1).
                    - Switch parameter (L2_1, L2_2, L2_3, ...). Binary (0 or 1).

        Returns:
            simulation_results: a numpy array containing the simulation results [DcGain, VoltageRipple, Efficiency].
        """
        self.config, self.G = parse_topology(self.config)
        self.config = process_sweep_data(config=self.config, sweep_data=design_variable)
        rewrite_netlist(self.config)
        Efficiency, error_report, _, _ = self.__calculate_efficiency()
        print(f"Error report from _calculate_efficiency: {error_report}")
        DcGain, VoltageRipple = process_log_file(self.config.log_file_path)
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
        assert self.G is not None, "Graph is not initialized. Call parse_topology() from simulate() first."
        plt.figure()
        node_colors = [self.G.nodes[n]["color"] for n in self.G.nodes()]
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=200, font_size=10)
        plt.show()


if __name__ == "__main__":
    # Test with absolute path and a different bucket_id
    original_netlist_path = os.path.abspath("./data/netlist/5_4_3_6_10-dcdc_converter_1.net")  # sweep 141
    bucket_id = "1_1_1_1_1"
    problem = PowerElectronics(original_netlist_path=original_netlist_path, bucket_id=bucket_id, mode="batch")

    # Initialize the problem with default values
    problem = PowerElectronics()

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
    problem.simulate(design_variable=sweep_data)
    print(problem.simulation_results)  # [-1.27858   -0.025081   0.7827396]

    problem.render()
