from pathlib import Path

import numpy as np
import pytest

from engibench.constraint import Criticality
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.constraint import Violations
from engibench.problems.power_electronics.utils.process_log_file import InvalidNgSpiceOutputWarning
from engibench.problems.power_electronics.utils.process_log_file import process_log_file
from engibench.problems.power_electronics.v0 import PowerElectronics

VALID_DESIGN = np.array(
    [
        15.600e-6,
        19.480e-6,
        15.185e-6,
        2.442e-6,
        9.287e-6,
        15.377e-6,
        354.659e-6,
        706.596e-6,
        195.361e-6,
        0.5,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    dtype=np.float32,
)


def constraint_names(violations: Violations) -> set[str]:
    """Return the callback names responsible for a collection of violations."""
    return {violation.constraint.check.__name__ for violation in violations.violations}


@pytest.fixture
def problem(tmp_path: Path) -> PowerElectronics:
    return PowerElectronics(target_dir=str(tmp_path))


def test_valid_design_satisfies_constraints(problem: PowerElectronics) -> None:
    assert not problem.check_constraints(VALID_DESIGN, {})


def test_passive_components_must_be_positive(problem: PowerElectronics) -> None:
    design = VALID_DESIGN.copy()
    design[0] = 0.0

    violations = problem.check_constraints(design, {}).by_category(THEORY)

    assert constraint_names(violations) == {"passive_components_are_positive"}
    assert all(violation.constraint.criticality is Criticality.Error for violation in violations.violations)


def test_duty_cycle_must_be_a_fraction(problem: PowerElectronics) -> None:
    design = VALID_DESIGN.copy()
    design[9] = -0.1

    violations = problem.check_constraints(design, {})

    assert constraint_names(violations.by_category(THEORY)) == {"duty_cycle_is_fraction"}
    assert constraint_names(violations.by_category(IMPL)) == {"duty_cycle_has_valid_pwl_timing"}


@pytest.mark.parametrize("duty_cycle", [0.001, 0.998])
def test_duty_cycle_must_produce_increasing_pwl_times(problem: PowerElectronics, duty_cycle: float) -> None:
    design = VALID_DESIGN.copy()
    design[9] = duty_cycle

    violations = problem.check_constraints(design, {}).by_category(IMPL)

    assert constraint_names(violations) == {"duty_cycle_has_valid_pwl_timing"}


def test_switch_levels_must_be_binary(problem: PowerElectronics) -> None:
    design = VALID_DESIGN.copy()
    design[10] = 0.5

    violations = problem.check_constraints(design, {}).by_category(THEORY)

    assert constraint_names(violations) == {"switch_levels_are_binary"}


def test_wrong_design_shape_only_violates_design_space(problem: PowerElectronics) -> None:
    violations = problem.check_constraints(np.zeros(2, dtype=np.float32), {})

    assert constraint_names(violations) == {"design_constraint"}


def test_process_log_file_reads_finite_objectives(tmp_path: Path) -> None:
    log_path = tmp_path / "simulation.log"
    log_path.write_text("gain = 1.25\nvpp_ratio = 0.125\n")

    assert process_log_file(str(log_path)) == (1.25, 0.125)


@pytest.mark.parametrize(
    "contents",
    [
        "Error: transient analysis failed\n",
        "gain = invalid\nvpp_ratio = 0.125\n",
        "gain = nan\nvpp_ratio = 0.125\n",
        "gain = 1.25\nvpp_ratio = inf\n",
    ],
)
def test_process_log_file_warns_about_nonfinite_objectives(tmp_path: Path, contents: str) -> None:
    log_path = tmp_path / "simulation.log"
    log_path.write_text(contents)

    with pytest.warns(InvalidNgSpiceOutputWarning, match="did not produce finite"):
        objectives = process_log_file(str(log_path))

    assert not np.all(np.isfinite(objectives))
