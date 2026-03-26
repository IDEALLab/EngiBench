# PowerElectronics

``` {problem:table}
:lead: Xuliang Dong @ liangXD523
```

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
