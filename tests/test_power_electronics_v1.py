"""Tests for Power Electronics v1 objective semantics and provenance."""

from pathlib import Path

import numpy as np
import pytest

from engibench.core import ObjectiveDirection
from engibench.problems.power_electronics import PowerElectronics as PublicPowerElectronics
from engibench.problems.power_electronics.utils.ngspice import NgSpiceIdentity
from engibench.problems.power_electronics.v0 import PowerElectronics as PowerElectronicsV0
from engibench.problems.power_electronics.v1 import derive_metrics
from engibench.problems.power_electronics.v1 import PowerElectronics
from tests.test_power_electronics import VALID_DESIGN


class FakeNgSpice:
    """Backend stub that emits deterministic signed physical measurements."""

    identity = NgSpiceIdentity(
        version="44.2",
        major_version=44,
        executable_path="/opt/ngspice/bin/ngspice",
        executable_sha256="abc123",
        platform_system="Linux",
        platform_machine="x86_64",
        version_output="ngspice-44.2 : Circuit level simulation program",
    )

    def run(self, _netlist_path: str, log_file_path: str, timeout: int = 30) -> None:
        del timeout
        Path(log_file_path).write_text("vo_mean = -250.0\nvpp = 25.0\n")


def test_v1_is_explicit_while_v0_remains_the_public_default() -> None:
    assert PublicPowerElectronics is PowerElectronicsV0
    assert PowerElectronics.version == 1
    assert PowerElectronics.dataset_id == "IDEALLab/power_electronics_v1"
    assert PowerElectronicsV0.version == 0
    assert PowerElectronicsV0.dataset_id == "IDEALLab/power_electronics_v0"
    assert PowerElectronicsV0.objectives == (
        ("DcGain", ObjectiveDirection.MINIMIZE),
        ("Voltage_Ripple", ObjectiveDirection.MAXIMIZE),
    )


def test_v1_declares_both_corrected_objectives_as_minimize() -> None:
    assert PowerElectronics.objectives == (
        ("dc_gain_error", ObjectiveDirection.MINIMIZE),
        ("relative_voltage_ripple", ObjectiveDirection.MINIMIZE),
    )


def test_derive_metrics_preserves_negative_gain() -> None:
    metrics = derive_metrics(output_voltage_mean=-250.0, output_voltage_peak_to_peak=25.0)

    np.testing.assert_allclose(
        (metrics.dc_gain, metrics.dc_gain_error, metrics.relative_voltage_ripple),
        (-0.25, 0.5, 0.1),
    )
    assert metrics.simulation_valid
    assert metrics.objectives_valid
    assert metrics.status == "ok"
    np.testing.assert_allclose(metrics.objective_values, [0.5, 0.1])


def test_zero_mean_voltage_has_undefined_ripple_without_hiding_measurement() -> None:
    metrics = derive_metrics(output_voltage_mean=0.0, output_voltage_peak_to_peak=25.0)

    assert metrics.dc_gain == 0.0
    np.testing.assert_allclose(metrics.dc_gain_error, 0.25)
    assert np.isnan(metrics.relative_voltage_ripple)
    assert metrics.simulation_valid
    assert not metrics.objectives_valid
    assert metrics.status == "undefined_relative_voltage_ripple"


@pytest.mark.parametrize(
    ("mean_voltage", "peak_to_peak_voltage"),
    [(np.nan, 1.0), (1.0, np.inf), (1.0, -1.0)],
)
def test_invalid_measurements_produce_invalid_objectives(mean_voltage: float, peak_to_peak_voltage: float) -> None:
    metrics = derive_metrics(mean_voltage, peak_to_peak_voltage)

    assert not metrics.simulation_valid
    assert not metrics.objectives_valid
    assert metrics.status == "invalid_measurements"
    assert np.all(np.isnan(metrics.objective_values))


def test_simulate_verbose_returns_raw_measurements_and_backend_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    problem = PowerElectronics(target_dir=str(tmp_path))
    backend = FakeNgSpice()
    monkeypatch.setattr(problem, "_backend", lambda: backend)

    result = problem.simulate_verbose(VALID_DESIGN)

    np.testing.assert_allclose(result.objective_values, [0.5, 0.1])
    np.testing.assert_allclose(
        (
            result.output_voltage_mean,
            result.output_voltage_peak_to_peak,
            result.dc_gain,
            result.dc_gain_error,
            result.relative_voltage_ripple,
        ),
        (-250.0, 25.0, -0.25, 0.5, 0.1),
    )
    assert result.simulation_valid
    assert result.objectives_valid
    assert result.status == "ok"
    assert result.simulator_identity == backend.identity
    netlist = Path(problem.config.rewrite_netlist_path).read_text()
    assert "print Vo_mean, Vpp, Gain, Vpp_ratio" in netlist
