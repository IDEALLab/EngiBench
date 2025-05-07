import os
import pickle
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils import slurm


def test_pickle_callable_works_for_a_function() -> None:
    serialized = pickle.dumps(slurm.MemorizeModule(a_function))
    deserialized = pickle.loads(serialized)
    assert deserialized()


def test_pickle_callable_works_for_a_class() -> None:
    serialized = pickle.dumps(slurm.MemorizeModule(AClass))
    deserialized = pickle.loads(serialized)
    assert deserialized(1.0).x == 1.0


def test_pickle_callable_works_for_a_method() -> None:
    a_method = AClass(1.0).a_method
    serialized = pickle.dumps(slurm.MemorizeModule(a_method))
    deserialized = pickle.loads(serialized)
    assert deserialized() == 1.0


class FakeDesign:
    """It's only a model."""

    def __init__(self, design_id: int) -> None:
        self.design_id = design_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.design_id == other.design_id


class FakePlot:
    """It's only a model."""

    def __init__(self, design: FakeDesign) -> None:
        self.design = design

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.design == other.design


class FakeProblem(Problem[FakeDesign]):
    def __init__(self, problem_id: int, *, some_arg: bool) -> None:
        self.problem_id = problem_id
        self.some_arg = some_arg

    def simulate(self, design: FakeDesign, config: dict[str, Any] | None = None, **kwargs) -> NDArray[np.float64]:
        offset = (config or {})["offset"]
        return np.array([design.design_id + offset], dtype=np.float64)

    def optimize(
        self, starting_point: FakeDesign, config: dict[str, Any] | None = None
    ) -> tuple[FakeDesign, list[OptiStep]]:
        return starting_point, []

    def render(self, design: FakeDesign, *, open_window: bool = False) -> FakePlot:
        return FakePlot(design)


FAKE_SBATCH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tools",
    "fake_sbatch.py",
)


def find_real_sbatch() -> list[str]:
    try:
        if (
            subprocess.run(
                ["sbatch", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            ).returncode
            == 0
        ):
            return ["sbatch"]
    except FileNotFoundError:
        pass
    return []


def job_render(problem_id: int, design_id: int) -> FakePlot:
    p = FakeProblem(problem_id, some_arg=True)
    design, _ = p.optimize(FakeDesign(design_id))
    return p.render(design)


def job_simulate(problem_id: int, design_id: int) -> NDArray[np.float64]:
    p = FakeProblem(problem_id, some_arg=True)
    design, _ = p.optimize(FakeDesign(design_id))
    return p.simulate(design, {"offset": 10})


def a_function() -> bool:
    return True


class AClass:
    def __init__(self, x: float) -> None:
        self.x = x

    def a_method(self) -> float:
        return self.x


@pytest.mark.parametrize("sbatch_exec", [FAKE_SBATCH, *find_real_sbatch()])
def test_sbatch_map_save(sbatch_exec: str) -> None:
    """Test if a fake slurm can process FakeProblem."""

    slurm_args = slurm.SlurmConfig(sbatch_executable=sbatch_exec)
    slurm.sbatch_map(
        job_render,
        args=[{"problem_id": 1, "design_id": -1}, {"problem_id": 2, "design_id": -2}, {"problem_id": 3, "design_id": -3}],
        slurm_args=slurm_args,
    ).save("results.pkl", slurm_args=slurm_args)
    results = slurm.load_results()
    os.remove("results.pkl")
    for result in results:
        if isinstance(result, slurm.JobError):
            raise result
    assert results == [FakePlot(FakeDesign(-1)), FakePlot(FakeDesign(-2)), FakePlot(FakeDesign(-3))]


def f_reduce(results: list[NDArray[np.float64]]) -> float:
    return sum(v[0] for v in results)


@pytest.mark.parametrize("sbatch_exec", [FAKE_SBATCH, *find_real_sbatch()])
def test_sbatch_map_reduce(sbatch_exec: str) -> None:
    """Test if a fake slurm can process FakeProblem."""

    slurm_args = slurm.SlurmConfig(sbatch_executable=sbatch_exec)
    result = slurm.sbatch_map(
        job_simulate,
        args=[{"problem_id": 1, "design_id": -1}, {"problem_id": 2, "design_id": -2}, {"problem_id": 3, "design_id": -3}],
        slurm_args=slurm_args,
    ).reduce(f_reduce, slurm_args=slurm_args)
    expected_result = 24.0
    assert result == expected_result
