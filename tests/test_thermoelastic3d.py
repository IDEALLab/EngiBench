import pytest

from engibench.problems.thermoelastic3d import ThermoElastic3D


def test_thermoelastic3d_config_builds_matching_non_cubic_default_masks() -> None:
    problem = ThermoElastic3D(config={"nelx": 2, "nely": 4, "nelz": 3})

    assert problem.config is not None
    assert problem.design_space.shape == (4, 2, 3)
    assert problem.config.fixed_elements.shape == (3, 5, 4)
    assert problem.config.force_elements_x.shape == (3, 5, 4)
    assert problem.config.force_elements_y.shape == (3, 5, 4)
    assert problem.config.force_elements_z.shape == (3, 5, 4)


def test_thermoelastic3d_unsupported_dataset_configs_fail_before_loading() -> None:
    problem = ThermoElastic3D(config={"nelx": 2, "nely": 4, "nelz": 3})

    assert problem.dataset_id == ""
    with pytest.raises(ValueError, match="dataset access is implemented only for cubic grids"):
        _ = problem.dataset
    with pytest.raises(ValueError, match="dataset access is implemented only for cubic grids"):
        problem.random_design()
