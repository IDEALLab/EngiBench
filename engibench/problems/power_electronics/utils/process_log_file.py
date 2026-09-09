"""Read from the log file to get the DcGain and Voltage Ripple values."""
# ruff: noqa: N806 # Upper case

import warnings

import numpy as np


class InvalidNgSpiceOutputWarning(RuntimeWarning):
    """Warn that ngspice did not produce finite objective measurements."""


def process_log_file(log_file_path: str) -> tuple[float, float]:
    """Read from log_file_path to get the DcGain and Voltage Ripple values."""
    DcGain, VoltageRipple = np.nan, np.nan
    with open(log_file_path) as log:
        lines = log.readlines()
        for line in lines:
            if line.strip() != "":
                parts = line.split()
                try:
                    if parts[0] == "gain":
                        DcGain = float(parts[2])
                    elif parts[0] == "vpp_ratio":
                        VoltageRipple = float(parts[2])
                except (IndexError, ValueError):
                    continue

    if not np.all(np.isfinite((DcGain, VoltageRipple))):
        warnings.warn(
            f"ngspice did not produce finite gain and voltage-ripple measurements; see {log_file_path}.",
            InvalidNgSpiceOutputWarning,
            stacklevel=2,
        )
    return DcGain, VoltageRipple
