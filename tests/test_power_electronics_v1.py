"""Tests for Power Electronics v1 objective semantics and provenance."""

import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import pytest

from engibench.core import ObjectiveDirection
from engibench.problems.power_electronics import PowerElectronics as PublicPowerElectronics
from engibench.problems.power_electronics import v1 as v1_module
from engibench.problems.power_electronics.utils.ngspice import NgSpiceIdentity
from engibench.problems.power_electronics.v0 import _warn_if_v0
from engibench.problems.power_electronics.v0 import HistoricalPowerElectronicsWarning
from engibench.problems.power_electronics.v0 import PowerElectronics as PowerElectronicsV0
from engibench.problems.power_electronics.v1 import _warn_if_noncanonical_backend
from engibench.problems.power_electronics.v1 import CANONICAL_NGSPICE_SHA256
from engibench.problems.power_electronics.v1 import DATASET_REVISION
from engibench.problems.power_electronics.v1 import derive_metrics
from engibench.problems.power_electronics.v1 import NoncanonicalPowerElectronicsBackendWarning
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


def test_v0_creation_warns_once_but_v1_creation_does_not(tmp_path: Path) -> None:
    """The compatibility default must direct new users to v1 without spamming."""
    _warn_if_v0.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PowerElectronicsV0(target_dir=str(tmp_path))
        PowerElectronicsV0(target_dir=str(tmp_path))
        PowerElectronics(target_dir=str(tmp_path))
    _warn_if_v0.cache_clear()

    historical = [warning for warning in caught if warning.category is HistoricalPowerElectronicsWarning]
    assert len(historical) == 1
    assert "from engibench.problems.power_electronics.v1 import PowerElectronics" in str(historical[0].message)


def test_v1_dataset_is_pinned_to_published_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Later dataset main-branch edits must not change the v1 problem data."""
    calls: list[tuple[str, str]] = []
    dataset = object()

    def fake_load_dataset(dataset_id: str, *, revision: str) -> object:
        calls.append((dataset_id, revision))
        return dataset

    monkeypatch.setattr(v1_module, "load_dataset", fake_load_dataset)
    problem = PowerElectronics(target_dir=str(tmp_path))

    assert problem.dataset is dataset
    assert problem.dataset is dataset
    assert calls == [("IDEALLab/power_electronics_v1", DATASET_REVISION)]


def test_v1_backend_warning_is_version_specific_and_once_per_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive v1 warns for noncanonical platforms, including x86 with another binary."""
    _warn_if_noncanonical_backend.cache_clear()
    backend = FakeNgSpice()
    monkeypatch.setattr(v1_module, "NgSpice", lambda *, ngspice_path: backend)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            problem = PowerElectronics(target_dir=str(tmp_path))
            assert problem.simulator_identity == FakeNgSpice.identity
        _warn_if_noncanonical_backend("44.2", CANONICAL_NGSPICE_SHA256, "Linux", "x86_64")
    _warn_if_noncanonical_backend.cache_clear()

    backend_warnings = [warning for warning in caught if warning.category is NoncanonicalPowerElectronicsBackendWarning]
    assert len(backend_warnings) == 1
    assert "different from the published dataset" in str(backend_warnings[0].message)
    assert "Linux/x86_64" in str(backend_warnings[0].message)


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


def test_real_ngspice_v1_parses_finite_signed_measurements(tmp_path: Path) -> None:
    """Exercise the actual rewritten netlist and log parser on the CI ngspice."""
    if not os.environ.get("NGSPICE_PATH") and shutil.which("ngspice") is None:
        pytest.skip("ngspice is not installed")
    problem = PowerElectronics(target_dir=str(tmp_path))

    result = problem.simulate_verbose(VALID_DESIGN)

    assert result.status == "ok"
    assert result.simulation_valid
    assert result.objectives_valid
    measurements = (
        result.output_voltage_mean,
        result.output_voltage_peak_to_peak,
        result.dc_gain,
        result.dc_gain_error,
        result.relative_voltage_ripple,
    )
    assert np.all(np.isfinite(measurements))
    assert result.output_voltage_peak_to_peak >= 0.0
    np.testing.assert_allclose(result.dc_gain, result.output_voltage_mean / 1000.0)
    np.testing.assert_allclose(result.dc_gain_error, abs(result.dc_gain - 0.25))
    np.testing.assert_allclose(
        result.relative_voltage_ripple,
        result.output_voltage_peak_to_peak / abs(result.output_voltage_mean),
    )
