import numpy as np
import numpy.typing as npt
import pytest

from engibench.problems.thermoelastic3d import ThermoElastic3D
from engibench.problems.thermoelastic3d.model.fem_model import FeaModel3D


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


def reference_cone(nely: int, nelx: int, nelz: int, rmin: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The correct cone filter on the (nely, nelx, nelz) grid the design actually uses."""
    coords = np.array([(y, x, z) for y in range(nely) for x in range(nelx) for z in range(nelz)], float)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    h = np.maximum(0.0, rmin - dist)
    return h, h.sum(axis=1)


def test_filter_matches_reference_on_non_cubic_grid() -> None:
    nely, nelx, nelz, rmin = 4, 2, 3, 1.5
    h, hs = FeaModel3D().get_filter(nelx, nely, nelz, rmin)
    rh, rhs = reference_cone(nely, nelx, nelz, rmin)
    v = np.arange(nely * nelx * nelz)
    np.testing.assert_allclose((h @ v) / hs, (rh @ v) / rhs, rtol=1e-12)
