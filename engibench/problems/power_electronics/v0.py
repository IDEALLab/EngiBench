# ruff: noqa: N806


"""Power Electronics problem."""

from dataclasses import dataclass
import os
from typing import Any, NoReturn

from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import Problem
from engibench.problems.power_electronics.utils.config import Config
from engibench.problems.power_electronics.utils.netlist_handler import parse_topology
from engibench.problems.power_electronics.utils.netlist_handler import rewrite_netlist
from engibench.problems.power_electronics.utils.ngspice import NgSpice
from engibench.problems.power_electronics.utils.process_log_file import process_log_file
from engibench.problems.power_electronics.utils.process_sweep_data import process_sweep_data


class PowerElectronics(Problem[npt.NDArray]):
    r"""Power Electronics parameter optimization problem.

    ```{note}
    This problem requires `ngspice` to be installed. See the simulator section for more details.
    ```

    ## Problem Description
    This problem simulates a power converter circuit which has a fixed circuit topology. There are 5 switches, 4 diodes, 3 inductors and 6 capacitors.
    The circuit topology is fixed. It is defined in the netlist file `5_4_3_6_10-dcdc_converter_1.net`.
    By changing circuit parameters such as capacitance, we rewrite the netlist file and use ngSpice to simulate the circuits to get the performance metrics, which are defined as the objectives of this problem.
    You can use this problem to train your regression model. You can also try to find the optimal circuit parameters that minimize objectives.

    ## Motivation
    Optimizing circuit parameters is a critical aspect of circuit design but remains challenging, particularly for power converter circuits that contain diodes and switches,
    which introduce significant nonlinearity and discontinuity. These characteristics make key objectives such as *DcGain* and *Voltage Ripple* highly sensitive to even small parameter variations.

    Because the circuit simulator NgSpice operates as a black box and is non-differentiable, gradient-based optimization methods are not suitable.
    Bayesian optimization is commonly employed for parameter tuning, while surrogate models offer a promising alternative.
    Even under the constraint of a fixed topology, optimizing circuit parameters to minimize the objectives remains a difficult problem for surrogate models.

    NgSpice applies transient analysis by formulating the system as a set of differential equations based on Kirchhoff's laws.
    These equations are discretized using numerical integration methods such as the Backward Euler or trapezoidal rule and solved iteratively at each time step to compute performance metrics.
    To ensure stable simulations, a specific on-off switching pattern is chosen for the circuit.
    Despite this simplification, determining the optimal parameter values remains highly challenging.

    ## Design Space
    The design space for this problem is represented as a 10-dimensional bounded box, where each dimension corresponds to a specific circuit parameter. These parameters include values for capacitors, inductors, and a shared duty cycle for all switches. Each design can be expressed as a vector **x** of the form:

    $$
    x = \begin{bmatrix} C_1,\dots,C_6,L_1,L_2,L_3,T_1 \end{bmatrix}^{\top} \in \mathcal{X},
    \quad
    \mathcal{X} = [1\text{e}{-6}, 2\text{e}{-5}]^6 \times [1\text{e}{-6}, 1\text{e}{-3}]^3 \times [0.1, 0.9]
    $$

    Here, $C_1,\dots,C_6$ are the capacitance values (in Farads), $L_1,L_2,L_3$ are the inductance values (in Henries), and $T_1$ is the duty cycle shared across all 5 switches. The duty cycle $T_1$ denotes the fraction of time during which the switches are in the “on” state and governs a periodic on-off pattern repeated at high frequency throughout the simulation.

    ## Objectives
    The simulation outputs two scalar values: *DcGain* and *Voltage Ripple*. The former represents the ratio of load to input voltage and should ideally approximate a predefined constant, such as $0.25$, as closely as possible. Meanwhile, the latter quantifies the voltage fluctuation at the load.

    **DcGain objective:**

    $$
    \min_{\mathbf{x} \in \mathcal{X}} \; \bigg|\frac{\overline{V_{load}(t)}}{V_{source}} - 0.25\bigg|
    = \bigg|\frac{1}{V_{source}} \cdot \frac{1}{T} \sum_{i=1}^{N-1} \frac{V_{load}(t_{i+1}) + V_{load}(t_i)}{2} \cdot (t_{i+1} - t_i) - 0.25\bigg|
    $$

    where $\overline{V_{load}(t)}$ is the average load voltage, $V_{source} = 1000$ volts, and $T = t_N - t_1$ is the simulation duration.

    **Voltage Ripple objective:**

    $$
    \min_{\mathbf{x} \in \mathcal{X}} \; \text{Voltage Ripple}
    = \frac{V_{pp}(t)}{\overline{V_{load}(t)}}
    = \frac{\max_{i \in [1, N]} V_{load}(t_i) - \min_{i \in [1, N]} V_{load}(t_i)}{\overline{V_{load}(t)}}
    $$

    where $V_{pp}$ is the peak-to-peak load voltage calculated during transient analysis.

    ## Conditions
    This problem does not include environmental or operational conditions as part of its input specification. Unlike other domains where the simulation setup may vary based on conditions (e.g., load configurations or external temperatures), the circuit is simulated under fixed source voltage and switching behavior. As a result, the design optimization task focuses solely on tuning internal circuit parameters, with no external conditions to vary. More complex variants of this problem — involving multiple topologies or variable source voltages — may be considered in future releases.

    ## Simulator
    The simulator is ngSpice circuit simulator. You can download it based on your operating system:
    - Windows: [https://sourceforge.net/projects/ngspice/files/ng-spice-rework/45.2/](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/45.2/)
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

    @dataclass
    class Conditions:
        """Conditions."""

    conditions = Conditions()
    design_space = spaces.Box(
        low=np.array([1e-6] * 6 + [1e-6] * 3 + [0.1] + [0] * 10),
        high=np.array([2e-5] * 6 + [1e-3] * 3 + [0.9] + [1] * 10),
        shape=(20,),
        dtype=np.float32,
    )
    dataset_id = "IDEALLab/power_electronics_v0"
    container_id = None
    config: Config

    def __init__(
        self,
        seed: int = 0,
        target_dir: str = os.getcwd(),
        original_netlist_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/5_4_3_6_10-dcdc_converter_1.net"
        ),
        mode: str = "control",
        ngspice_path: str | None = None,
    ) -> None:
        """Initializes the Power Electronics problem.

        Args:
            seed (int): The random seed for the problem.
            target_dir: The target directory for the rewritten netlist, log and raw files. Default to os.getcwd().
            original_netlist_path: The path to the original netlist file. Accepts both relative and absolute paths.
            bucket_id: The bucket ID for the netlist file. E.g. "5_4_3_6_10".
            mode: The mode for the simulation. Default to "control". mode = "batch" is for development.
            ngspice_path: The path to the ngspice executable for Windows.
        """
        super().__init__(seed=seed)

        self.config = Config(
            target_dir=target_dir,
            original_netlist_path=original_netlist_path,
            mode=mode,
        )
        self.ngspice_path = ngspice_path

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> npt.NDArray:  # noqa: ARG002
        """Simulates the performance of a Power Electronics design.

        Args:
            design: sweep data. It is a list of floats representing the design parameters for the simulation.
                    In general, they are:
                    - Capacitor values (C0, C1, C2, ...) in Farads.
                    - Inductor values (L0, L1, L2, ...) in Henries.
                    - Switch parameter T1 duty cycle. Fraction [0.1, 0.9]. All switches have the same T1.
                    - Switch parameter T2 is not included. Set to constant 1.0 for all switches.
                    - Switch parameter (L1_1, L1_2, L1_3, ...). Binary (0 or 1).
                    - Switch parameter (L2_1, L2_2, L2_3, ...). Binary (0 or 1).
            config: ignored

        Returns:
            simulation_results: a numpy array containing the simulation results [DcGain, VoltageRipple, Efficiency].
        """
        self.config, rewrite_netlist_str, edge_map, _ = parse_topology(self.config)
        self.config = process_sweep_data(config=self.config, sweep_data=design.tolist())
        rewrite_netlist(self.config, rewrite_netlist_str, edge_map)
        # Use the ngspice wrapper to run the simulation
        ngspice = NgSpice(ngspice_windows_path=self.ngspice_path)
        ngspice.run(self.config.rewrite_netlist_path, self.config.log_file_path)
        DcGain, VoltageRipple = process_log_file(self.config.log_file_path)
        return np.array([DcGain, VoltageRipple])

    def optimize(self, _starting_point: npt.NDArray, _config: dict[str, Any] | None = None) -> NoReturn:
        """Optimize the design variable. Not applicable for this problem."""
        raise NotImplementedError("Not yet implemented")

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> None:  # noqa: ARG002
        """Render the circuit topology using NetworkX.

        It displays the Graph of the circuit topology rather than the circuit diagram.
        Each circuit element (V, L, C, etc.) is a node. Each wire/port is also a node.
        """
        _, _, _, G = parse_topology(self.config)
        plt.figure()
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=200, font_size=10)
        plt.show()

    def random_design(self, dataset_split: str = "train", design_key: str = "initial_design") -> tuple[npt.NDArray, int]:
        """Samples a valid random initial design.

        Args:
            dataset_split (str): The key for the dataset to sample from.
            design_key (str): The key for the design to sample from.

        Returns:
            DesignType: The valid random design.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split][design_key]), dtype=int)

        return np.array(self.dataset[dataset_split][design_key][rnd]), rnd

    def reset(self, seed: int | None = None) -> None:
        """Reset the problem.

        Args:
            seed: The seed for the random number generator.
        """
        return super().reset(seed)


if __name__ == "__main__":
    # Test with absolute path and a different bucket_id
    problem = PowerElectronics(seed=0, mode="batch")

    # Initialize the problem with default values
    problem = PowerElectronics(seed=0)

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
    simulation_results = problem.simulate(design=np.array(sweep_data))
    print(simulation_results)  # [0.01244983 0.9094711  0.74045004]

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
    problem.reset(seed=0)
    simulation_results = problem.simulate(design=np.array(sweep_data))
    print(simulation_results)  # [-1.27858   -0.025081   0.7827396]

    problem.render(np.array([]))
