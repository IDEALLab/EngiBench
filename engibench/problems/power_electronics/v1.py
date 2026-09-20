"""Power Electronics v1 with explicit physical measurements and corrected objectives.

Unlike v0, v1 preserves signed measurements and minimizes gain error and
relative voltage ripple. Its dataset is regenerated from the exact v0 designs
and splits with a frozen ngspice 44.2 Linux x86_64 backend.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import SimulationResult
from engibench.problems.power_electronics.utils.netlist_handler import parse_topology
from engibench.problems.power_electronics.utils.netlist_handler import rewrite_netlist
from engibench.problems.power_electronics.utils.ngspice import NgSpice
from engibench.problems.power_electronics.utils.ngspice import NgSpiceIdentity
from engibench.problems.power_electronics.utils.process_log_file import process_measurements
from engibench.problems.power_electronics.utils.process_sweep_data import process_sweep_data
from engibench.problems.power_electronics.v0 import PowerElectronics as PowerElectronicsV0

SOURCE_VOLTAGE = 1000.0
TARGET_DC_GAIN = 0.25


@dataclass(frozen=True)
class PowerElectronicsMetrics:
    """Raw signed measurements, derived objectives, and validity information."""

    output_voltage_mean: float
    output_voltage_peak_to_peak: float
    dc_gain: float
    dc_gain_error: float
    relative_voltage_ripple: float
    simulation_valid: bool
    objectives_valid: bool
    status: str

    @property
    def objective_values(self) -> npt.NDArray[np.float64]:
        """Return objectives in the order declared by :class:`PowerElectronics`."""
        return np.array([self.dc_gain_error, self.relative_voltage_ripple], dtype=np.float64)


@dataclass
class PowerElectronicsSimulationResult(SimulationResult):
    """v1 simulation result with raw measurements and backend provenance."""

    output_voltage_mean: float
    output_voltage_peak_to_peak: float
    dc_gain: float
    dc_gain_error: float
    relative_voltage_ripple: float
    simulation_valid: bool
    objectives_valid: bool
    status: str
    simulator_identity: NgSpiceIdentity


def derive_metrics(output_voltage_mean: float, output_voltage_peak_to_peak: float) -> PowerElectronicsMetrics:
    """Derive the v1 objectives without discarding the sign of the DC gain."""
    dc_gain = output_voltage_mean / SOURCE_VOLTAGE
    simulation_valid = bool(
        np.all(np.isfinite((output_voltage_mean, output_voltage_peak_to_peak))) and output_voltage_peak_to_peak >= 0.0
    )
    if not simulation_valid:
        return PowerElectronicsMetrics(
            output_voltage_mean=output_voltage_mean,
            output_voltage_peak_to_peak=output_voltage_peak_to_peak,
            dc_gain=dc_gain,
            dc_gain_error=np.nan,
            relative_voltage_ripple=np.nan,
            simulation_valid=False,
            objectives_valid=False,
            status="invalid_measurements",
        )

    dc_gain_error = abs(dc_gain - TARGET_DC_GAIN)
    if output_voltage_mean == 0.0:
        return PowerElectronicsMetrics(
            output_voltage_mean=output_voltage_mean,
            output_voltage_peak_to_peak=output_voltage_peak_to_peak,
            dc_gain=dc_gain,
            dc_gain_error=dc_gain_error,
            relative_voltage_ripple=np.nan,
            simulation_valid=True,
            objectives_valid=False,
            status="undefined_relative_voltage_ripple",
        )

    return PowerElectronicsMetrics(
        output_voltage_mean=output_voltage_mean,
        output_voltage_peak_to_peak=output_voltage_peak_to_peak,
        dc_gain=dc_gain,
        dc_gain_error=dc_gain_error,
        relative_voltage_ripple=output_voltage_peak_to_peak / abs(output_voltage_mean),
        simulation_valid=True,
        objectives_valid=True,
        status="ok",
    )


class PowerElectronics(PowerElectronicsV0):
    """Power Electronics v1 with corrected objective semantics and provenance."""

    version = 1
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("dc_gain_error", ObjectiveDirection.MINIMIZE),
        ("relative_voltage_ripple", ObjectiveDirection.MINIMIZE),
    )
    dataset_id = "IDEALLab/power_electronics_v1"

    _ngspice_backend: NgSpice | None = None

    def _backend(self) -> NgSpice:
        """Resolve ngspice lazily and reuse its immutable identity across simulations."""
        if self._ngspice_backend is None:
            self._ngspice_backend = NgSpice(ngspice_path=self.ngspice_path)
        return self._ngspice_backend

    @property
    def simulator_identity(self) -> NgSpiceIdentity:
        """Return the exact ngspice binary and host identity used by this problem."""
        return self._backend().identity

    def simulate_verbose(
        self, design: npt.NDArray, config: dict[str, Any] | None = None
    ) -> PowerElectronicsSimulationResult:
        """Simulate a design and return signed measurements plus v1 objectives."""
        del config
        self.config, rewrite_netlist_str, edge_map, _ = parse_topology(self.config)
        self.config = process_sweep_data(config=self.config, sweep_data=design.tolist())
        rewrite_netlist(
            self.config,
            rewrite_netlist_str,
            edge_map,
            include_raw_measurements=True,
        )
        ngspice = self._backend()
        ngspice.run(self.config.rewrite_netlist_path, self.config.log_file_path)
        measurements = process_measurements(self.config.log_file_path)
        metrics = derive_metrics(
            measurements.output_voltage_mean,
            measurements.output_voltage_peak_to_peak,
        )
        return PowerElectronicsSimulationResult(
            objective_values=metrics.objective_values,
            output_voltage_mean=metrics.output_voltage_mean,
            output_voltage_peak_to_peak=metrics.output_voltage_peak_to_peak,
            dc_gain=metrics.dc_gain,
            dc_gain_error=metrics.dc_gain_error,
            relative_voltage_ripple=metrics.relative_voltage_ripple,
            simulation_valid=metrics.simulation_valid,
            objectives_valid=metrics.objectives_valid,
            status=metrics.status,
            simulator_identity=self.simulator_identity,
        )
