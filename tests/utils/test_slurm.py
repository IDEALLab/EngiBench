# ruff: noqa
# TODO(https://github.com/IDEALLab/EngiBench/issues/107): Rework slurm utils

# from __future__ import annotations

# import os
# import pickle
# import subprocess
# from typing import Any

# import numpy as np
# import numpy.typing as npt
# import pytest

# from engibench.core import Problem
# from engibench.utils import slurm


# class FakeDesign:
#     """It's only a model."""

#     def __init__(self, design_id: int) -> None:
#         self.design_id = design_id


# class FakeProblem(Problem[FakeDesign]):
#     def __init__(self, problem_id: int, *, some_arg: bool) -> None:
#         self.problem_id = problem_id
#         self.some_arg = some_arg

#     def simulate(self, design: FakeDesign, config: dict[str, Any] | None = None, **kwargs) -> npt.NDArray:
#         offset = (config or {})["offset"]
#         return np.array([design.design_id + offset])


# FAKE_SBATCH = os.path.join(
#     os.path.dirname(__file__),
#     "..",
#     "tools",
#     "fake_sbatch.py",
# )


# def find_real_sbatch() -> list[str]:
#     try:
#         if (
#             subprocess.run(
#                 ["sbatch", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
#             ).returncode
#             == 0
#         ):
#             return ["sbatch"]
#     except FileNotFoundError:
#         pass
#     return []


# @pytest.mark.parametrize("sbatch_exec", [FAKE_SBATCH, *find_real_sbatch()])
# def test_run_slurm(sbatch_exec: str) -> None:
#     """Test if a fake slurm can process FakeProblem."""

#     static_args = slurm.Args(simulate_args={"config": {"offset": 10}}, problem_args={"some_arg": True})
#     parameter_space = [
#         slurm.Args(problem_args={"problem_id": 1}, design_args={"design_id": -1}),
#         slurm.Args(problem_args={"problem_id": 2}, design_args={"design_id": -2}),
#         slurm.Args(problem_args={"problem_id": 3}, design_args={"design_id": -3}),
#     ]
#     slurm.submit(
#         problem=FakeProblem,
#         static_args=static_args,
#         parameter_space=parameter_space,
#         config=slurm.SlurmConfig(sbatch_executable=sbatch_exec),
#     )
#     with open("results.pkl", "rb") as stream:
#         results = pickle.load(stream)
#     os.remove("results.pkl")
#     assert results == [
#         {
#             "problem_args": {"some_arg": True, "problem_id": 1},
#             "simulate_args": {"config": {"offset": 10}},
#             "design_args": {"design_id": -1},
#             "results": np.array([9]),
#         },
#         {
#             "problem_args": {"some_arg": True, "problem_id": 2},
#             "simulate_args": {"config": {"offset": 10}},
#             "design_args": {"design_id": -2},
#             "results": np.array([8]),
#         },
#         {
#             "problem_args": {"some_arg": True, "problem_id": 3},
#             "simulate_args": {"config": {"offset": 10}},
#             "design_args": {"design_id": -3},
#             "results": np.array([7]),
#         },
#     ]
