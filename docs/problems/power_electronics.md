# PowerElectronics

``` {problem:table}
:lead: Xuliang Dong @ liangXD523
```

```{warning}
For new work, use v1:
`from engibench.problems.power_electronics.v1 import PowerElectronics`.
The plain `engibench.problems.power_electronics` import and problem registry
still select v0 for compatibility. v0 has historical objective directions and
labels whose original simulator environment is unknown; do not treat it as the
corrected v1 benchmark. Interactive v1 simulations on a backend other than the
published Linux x86_64/ngspice 44.2 binary warn because their numerical results
may differ from the v1 labels.
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
The design space is a 20-dimensional bounded box. Its first ten entries are six
capacitors, three inductors, and one shared duty cycle. The final ten entries
are the two binary control levels for each of five switches. The v0 and v1
datasets use the same designs and split membership.

$$
x = \begin{bmatrix} C_1,\dots,C_6,L_1,L_2,L_3,T_1,
G_{1,1},\dots,G_{5,1},G_{1,2},\dots,G_{5,2} \end{bmatrix}^{\top} \in \mathcal{X},
\quad
\mathcal{X} = [1\text{e}{-6}, 2\text{e}{-5}]^6 \times [1\text{e}{-6}, 1\text{e}{-3}]^3
\times [0.1, 0.9] \times \{0,1\}^{10}
$$

Here, $C_1,\dots,C_6$ are the capacitance values (in Farads), $L_1,L_2,L_3$ are the inductance values (in Henries), and $T_1$ is the duty cycle shared across all 5 switches. The duty cycle $T_1$ denotes the fraction of time during which the switches are in the “on” state and governs a periodic on-off pattern repeated at high frequency throughout the simulation.

## Objectives

### v1

The simulator first retains two signed/raw physical measurements:

- the mean output voltage $\overline{V_{load}}$;
- the peak-to-peak output voltage $V_{pp}$.

With $V_{source}=1000$ V, v1 derives a signed DC gain and two minimization
objectives:

**DcGain objective:**

$$
\operatorname{gain}(\mathbf{x}) = \frac{\overline{V_{load}(t)}}{V_{source}},
\qquad
\min_{\mathbf{x} \in \mathcal{X}} \; \big|\operatorname{gain}(\mathbf{x}) - 0.25\big|
= \bigg|\frac{1}{V_{source}} \cdot \frac{1}{T} \sum_{i=1}^{N-1} \frac{V_{load}(t_{i+1}) + V_{load}(t_i)}{2} \cdot (t_{i+1} - t_i) - 0.25\bigg|
$$

where $\overline{V_{load}(t)}$ is the average load voltage, $V_{source} = 1000$ volts, and $T = t_N - t_1$ is the simulation duration.

**Voltage Ripple objective:**

$$
\min_{\mathbf{x} \in \mathcal{X}} \; \text{Relative Voltage Ripple}
= \frac{V_{pp}(t)}{|\overline{V_{load}(t)}|}
= \frac{\max_{i \in [1, N]} V_{load}(t_i) - \min_{i \in [1, N]} V_{load}(t_i)}{|\overline{V_{load}(t)}|}
$$

where $V_{pp}$ is the peak-to-peak load voltage calculated during transient
analysis. The absolute value appears only in the ripple denominator and the
gain-error calculation; it does not discard the sign of the stored mean voltage
or gain. A zero mean voltage makes relative ripple undefined, which is stored
as a null objective with an explicit status instead of being silently replaced.

### v0 compatibility

v0 is intentionally unchanged. Its Python contract declares `DcGain` as a
minimization objective and `Voltage_Ripple` as a maximization objective, while
the older documentation described gain error and ripple minimization. Its
netlist computes signed `Gain = Vo_mean / 1000` and
`Vpp_ratio = Vpp / Vo_mean`; the archived dataset-generation notebook then
applied an absolute-value transformation before saving the published labels.
These semantics are preserved for reproducibility, not recommended as the new
objective definition.

## Conditions
This problem does not include environmental or operational conditions as part of its input specification. Unlike other domains where the simulation setup may vary based on conditions (e.g., load configurations or external temperatures), the circuit is simulated under fixed source voltage and switching behavior. As a result, the design optimization task focuses solely on tuning internal circuit parameters, with no external conditions to vary. More complex variants of this problem — involving multiple topologies or variable source voltages — may be considered in future releases.

## Simulator
For interactive use, install ngspice separately from the Python package:

- Linux: install your distribution's `ngspice` package (for example,
  `sudo apt-get install ngspice` on Ubuntu).
- macOS: build the CI-tested ngspice 44.2 binary with
  `scripts/install_ngspice_macos.sh /path/to/install` and set `NGSPICE_PATH` to
  its `bin/ngspice` executable. On Apple Silicon this installer builds an
  x86_64 binary for Rosetta 2.
- Windows: install [ngspice 45.2](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/45.2/)
  and put `ngspice.exe` on `PATH`, or point `NGSPICE_PATH` to it.

`PowerElectronics(ngspice_path=...)` takes precedence over `NGSPICE_PATH`,
which takes precedence over `PATH`. The wrapper accepts ngspice major versions
42 through 45 for interactive use. v0 warns once when created because its
historical labels have unknown simulator provenance. v1 warns once for each
noncanonical simulator identity; using a supported ngspice version does **not**
guarantee matching the published v1 labels. In particular, an ordinary Ubuntu
package or an ngspice 44.2 build for another architecture may give different
results. See [ngspice bug #622](https://sourceforge.net/p/ngspice/bugs/622/)
for a reported AArch64/x86_64 discrepancy on another circuit.

### Reproducing the v1 dataset

The canonical v1 dataset backend is ngspice 44.2 in a provenance-frozen Linux
x86_64 Apptainer image. Dataset generation rejects a different ngspice version,
operating system, CPU architecture, simulator checksum, or container checksum
unless the caller explicitly selects the noncanonical development override.
The image recipe is
[`containers/power_electronics_v1.def`](../../containers/power_electronics_v1.def);
it records the amd64 base-image digest, the ngspice source archive checksum,
and the Python package versions used by the generation entry point. Because
Debian package repositories change over time, the recipe alone is not a claim
of a byte-identical rebuild. The published SIF is the canonical runtime
artifact. Its SHA-256 is
`40816f203b7e1c68ae37f4d9353bd302486d77988b98733c021f1ff71f48ae02`,
and its ngspice binary SHA-256 is
`11a4334ee90509f5edfdceef541711a34a1943d26a14cf0928ac8d5947b72374`.
Both fingerprints are enforced by every canonical shard; an arbitrary
caller-provided fingerprint cannot redefine the canonical backend.

The runtime publication also includes an SPDX SBOM, package manifests, the
exact ngspice 44.2 source archive, and
[`third-party notices`](source:containers/power_electronics_v1.NOTICES.md).
Pull and verification instructions, including the immutable OCI digest, are in
the [`runtime publication record`](source:containers/power_electronics_v1.PUBLICATION.md).

This policy is narrower than the versions accepted by the interactive v0
wrapper. It exists because transient results have differed materially across
ngspice versions, builds, and CPU architectures.

## Dataset

### v0

The immutable source dataset is
[`IDEALLab/power_electronics_v0`](https://huggingface.co/datasets/IDEALLab/power_electronics_v0)
at revision `5c4adb2ec5cfc71794988b1297a7ff8ffe59daa5`. It contains 9,676
training rows, 2,765 validation rows, and 1,383 test rows.

The exact original runtime can no longer be confirmed. An archived generation
notebook refers to a Windows `ngspice.exe` and an ngspice 36 manual, which is
evidence for Windows/x86_64. On the other hand, a small reproduction with
ngspice 44.2 on an ARM64 Mac is much closer to the published values than the
corresponding ngspice 44.2 x86_64 run, which is evidence consistent with the
dataset having been generated on ARM. Because the reproduction also changes
the ngspice version, it cannot determine the original architecture. Treat the
v0 architecture as unknown and retain both pieces of evidence.

#### Fields
The dataset contains 3 fields:
- `initial_design`: The 20-dimensional design variable defined above.
- `DcGain`: The ratio of load vs. input voltage.
- `Voltage_Ripple`: The fluctuation of voltage on the load `R0`.

The published `DcGain` and `Voltage_Ripple` values are absolute-valued outputs
from the archived generation path. v0 has no complete simulator fingerprint in
its stored metadata.

#### Creation Method
We created this dataset in 3 parts. All the 3 parts are simulated with {`GS0_L1`, `GS1_L1`, `GS2_L1`, `GS3_L1`, `GS4_L1`} = {1, 0, 0, 1, 1} and {`GS0_L2`, `GS1_L2`, `GS2_L2`, `GS3_L2`, `GS4_L2`} = {1, 0, 1, 1, 0}.
Here are the 3 parts:
1. 6 capacitors and 3 inductors only take their min and max values. `T1` ranges {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. There are 2^6 * 2^3 * 9 = 4608 samples.
2. Random sample 4608 points in the 6 + 3 + 1 = 10 dimensional space. Min and max values in each dimension will not be sampled.
3. Latin hypercube sample 4608 points in the 6 + 3 + 1 = 10 dimensional space. Each dimension is split into 10 intervals. Min and max values in each dimension will not be sampled.

### v1

v1 re-simulates the exact v0 `initial_design` rows without changing their
train/validation/test membership. Each output row retains:

- `split`, `source_index`, and a SHA-256 identifier for the design;
- the original v0 labels for direct comparison;
- signed `output_voltage_mean` and signed `dc_gain`;
- `output_voltage_peak_to_peak`;
- `dc_gain_error` and `relative_voltage_ripple`;
- simulation/objective validity flags, a status, and any simulator error;
- the exact v0 dataset revision, EngiBench commit, ngspice version, and ngspice
  executable checksum.

The audited dataset is published as
[`IDEALLab/power_electronics_v1`](https://huggingface.co/datasets/IDEALLab/power_electronics_v1)
at immutable tag
[`v1.0.0`](https://huggingface.co/datasets/IDEALLab/power_electronics_v1/tree/v1.0.0)
and commit `eefad7d727ea1e5bef5e1b7088dea20a9b0cd67f`. It contains all
13,824 source rows in their original split membership. No values or rows were
clipped, removed, or imputed. The two invalid training rows (source indices
4,309 and 5,960) and one invalid validation row (source index 223) remain in
the dataset with null measurements/objectives and explicit status fields.
Consumers should retain those rows and mask them from objective losses.

`simulation_status` describes why objectives may be null. `null` here means a
missing field in the dataset, not a clipped or imputed value:

| Status | Measurements and objectives | Validity flags |
| --- | --- | --- |
| `ok` | Signed measurements and both objectives are finite. | `simulation_valid=true`, `objectives_valid=true` |
| `invalid_measurements` | At least one raw measurement is non-finite or peak-to-peak voltage is negative. Non-finite measurements and derived objectives are null; other raw measurements may remain. | Both false |
| `undefined_relative_voltage_ripple` | Mean voltage is zero. Raw measurements, gain, and gain error remain; relative ripple is null. | `simulation_valid=true`, `objectives_valid=false` |
| `simulation_error` | Simulation raised an error. All measurements and objectives are null; `error_type` and `error_message` are populated. | Both false |

For an objective loss, use `objectives_valid` as the mask. Retain every source
design and split row, including invalid rows, for auditing and reproducibility.

For backward compatibility,
`engibench.problems.power_electronics.PowerElectronics` continues to select
v0. New code selects v1 explicitly with
`from engibench.problems.power_electronics.v1 import PowerElectronics`.

Each atomically written JSONL shard has a companion manifest containing the
selected source indices, Hugging Face dataset fingerprint, complete ngspice
version banner, platform, container image checksum, output checksum, and status
counts. The generator refuses dirty EngiBench code and an uncontainerized or
noncanonical backend by default. Development overrides are explicit and must
not be used for the published dataset.

The generation entry point is:

```bash
python -m engibench.problems.power_electronics.dataset_generation \
  --split test \
  --indices 0,1,2 \
  --output /path/to/results/test-pilot.jsonl \
  --work-dir /path/to/work/test-pilot \
  --ngspice-path /usr/local/bin/ngspice \
  --container-image /path/to/power-electronics-v1.sif
```

For a full run, use `--num-shards` and `--shard-index` for deterministic,
non-overlapping source indices. Do not use `--limit` for the published run.

## Citation
This problem is an original contribution to EngiBench and was not refactored from any prior publication. If you use this problem in your research, please cite the EngiBench paper:

```bibtex
@inproceedings{felten_engibench_2025,
    title = {{EngiBench}: {A} {Framework} for {Data}-{Driven} {Engineering} {Design} {Research}},
    url = {https://openreview.net/forum?id=YowD33Q89V},
    author = {Felten, Florian and Apaza, Gabriel and B\"{a}unlich, Gerhard and Diniz, Cashen and Dong, Xuliang and Drake, Arthur and Habibi, Milad and Hoffman, Nathaniel J. and Keeler, Matthew and Massoudi, Soheyl and VanGessel, Francis G. and Fuge, Mark},
    booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems ({NeurIPS} 2025)},
    year = {2025},
}
```
