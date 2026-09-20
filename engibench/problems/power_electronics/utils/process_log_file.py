"""Read from the log file to get the DcGain and Voltage Ripple values."""
# ruff: noqa: N806 # Upper case

from dataclasses import dataclass
import warnings

import numpy as np

MEASUREMENT_VALUE_INDEX = 2


class InvalidNgSpiceOutputWarning(RuntimeWarning):
    """Warn that ngspice did not produce finite objective measurements."""


@dataclass(frozen=True)
class PowerElectronicsMeasurements:
    """Signed physical measurements emitted by the v1 ngspice netlist."""

    output_voltage_mean: float
    output_voltage_peak_to_peak: float


def process_log_file(log_file_path: str) -> tuple[float, float]:
    """Read from log_file_path to get the DcGain and Voltage Ripple values."""
    DcGain, VoltageRipple = np.nan, np.nan
    with open(log_file_path) as log:
        for line in log:
            parts = line.split()
            if len(parts) <= MEASUREMENT_VALUE_INDEX:
                continue
            try:
                if parts[0] == "gain":
                    DcGain = float(parts[MEASUREMENT_VALUE_INDEX])
                elif parts[0] == "vpp_ratio":
                    VoltageRipple = float(parts[MEASUREMENT_VALUE_INDEX])
            except ValueError:
                continue

    if not np.all(np.isfinite((DcGain, VoltageRipple))):
        warnings.warn(
            f"ngspice did not produce finite gain and voltage-ripple measurements; see {log_file_path}.",
            InvalidNgSpiceOutputWarning,
            stacklevel=2,
        )
    return DcGain, VoltageRipple


def process_measurements(log_file_path: str) -> PowerElectronicsMeasurements:
    """Read signed mean voltage and peak-to-peak voltage from an ngspice log."""
    values = {"vo_mean": np.nan, "vpp": np.nan}
    with open(log_file_path) as log:
        for line in log:
            parts = line.split()
            if len(parts) <= MEASUREMENT_VALUE_INDEX:
                continue
            name = parts[0].lower()
            if name not in values:
                continue
            try:
                values[name] = float(parts[MEASUREMENT_VALUE_INDEX])
            except ValueError:
                continue

    measurements = PowerElectronicsMeasurements(
        output_voltage_mean=values["vo_mean"],
        output_voltage_peak_to_peak=values["vpp"],
    )
    if not np.all(np.isfinite((measurements.output_voltage_mean, measurements.output_voltage_peak_to_peak))):
        warnings.warn(
            f"ngspice did not produce finite mean-voltage and peak-to-peak measurements; see {log_file_path}.",
            InvalidNgSpiceOutputWarning,
            stacklevel=2,
        )
    return measurements
