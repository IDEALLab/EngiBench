import numpy as np

from engibench.problems.airfoil.v0 import self_intersect


def test_self_intersect_finds_no_intersections_for_nonintersecting_curve() -> None:
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    assert self_intersect(curve) is None


def test_self_intersect_finds_no_intersections_for_longer_nonintersecting_curve() -> None:
    curve = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0], [2.0, 1.0], [2.0, 0.0], [0.0, 0.0]])
    assert self_intersect(curve) is None


def test_self_intersect_finds_intersections_for_intersecting_curve() -> None:
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    intersection = self_intersect(curve)
    assert intersection is not None
    index, a, b = intersection
    assert index == 1
    np.testing.assert_equal(a, np.array([1.0, 0.0]))
    np.testing.assert_equal(b, np.array([0.0, 1.0]))
