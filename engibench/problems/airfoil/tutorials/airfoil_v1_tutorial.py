"""Tutorial: the airfoil v1 problem and its features.

This script walks through every feature added in airfoil v1, end to end:

1. Instantiating the v1 problem and inspecting its design space / objectives / conditions
   (note the new sampled ``temperature`` condition).
2. Running a single CFD analysis with ``simulate`` (returns drag, lift).
3. Running an IPOPT shape optimization with ``optimize`` (drag-min at a target lift,
   lift enforced as a one-sided inequality, +/-0.10 FFD shape bounds).
4. Reading the scalar objective trajectory (``optisteps``) and the rich geometry/surface
   trajectory of every intermediate design (``optimization_trajectory``).
5. Using the advanced overrides: choosing which surface fields are written
   (``surface_variables``) and overriding the internal ADflow solver schedule
   (``solver_options``).
6. Locating the debugging artifacts each run produces (``opt.hst``, ``IPOPT.out``,
   ``final_abs_volume.npy``, tarred per-iteration surface/section files).
7. Generating a dataset at scale on Slurm.

The v1 backend runs MACH-Aero (ADflow + pyOptSparse) inside the ``mdolab/public``
container, so this needs a working container runtime (Docker/Podman/Apptainer) and the
container image. A full optimization takes several minutes; pass ``--skip-optimize`` to
only exercise the cheap parts, or ``--from-study <dir>`` to inspect a finished run.

Run it, e.g.:

    export CONTAINER_RUNTIME=apptainer
    python -m engibench.problems.airfoil.tutorials.airfoil_v1_tutorial \
        --coords-file /path/to/baseline_airfoils.npy --mpicores 4
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.v1 import Airfoil


def inspect_problem(problem: Airfoil) -> None:
    """Print the v1 design space, objectives and conditions (incl. the new temperature)."""
    print("\n=== 1. Problem definition (v1) ===")
    print("version:", problem.version, "| dataset:", problem.dataset_id)
    print("objectives:", problem.objectives)
    print("design space:", problem.design_space)
    # Conditions now include `temperature` alongside mach/reynolds/cl_target/area_*.
    print("conditions:", problem.conditions)


def load_baseline(problem: Airfoil, coords_file: str | None, idx: int) -> dict[str, Any]:
    """Return a baseline design dict from a local coords .npy, or from the dataset."""
    if coords_file is not None:
        raw = np.load(coords_file, allow_pickle=True)
        coords = np.asarray(raw[idx], dtype=float)
        if coords.shape[0] != 2:  # noqa: PLR2004  -- coords are (2, P); transpose (P, 2) inputs
            coords = coords.T
        return {"coords": coords, "angle_of_attack": 2.5}
    design, _ = problem.random_design()  # needs the published dataset
    return design


def run_simulate(problem: Airfoil, design: dict[str, Any], mpicores: int) -> None:
    """Run one CFD analysis. The flow condition now carries a temperature."""
    print("\n=== 2. simulate: one CFD analysis ===")
    config = {"mach": 0.5, "reynolds": 5.0e6, "temperature": 288.15}
    drag, lift = problem.simulate(design, config=config, mpicores=mpicores)
    print(f"drag={drag:.6f}  lift={lift:.6f}  (L/D={lift / drag:.2f})")


def run_optimize(problem: Airfoil, design: dict[str, Any], mpicores: int) -> tuple[dict[str, Any], list]:
    """Run an IPOPT shape optimization (drag-min at a target lift)."""
    print("\n=== 3. optimize: IPOPT shape optimization ===")
    # area_initial is the baseline area the (baseline-scaled) area constraint references.
    config = {
        "mach": 0.5,
        "reynolds": 5.0e6,
        "temperature": 288.15,
        "cl_target": 0.5,  # enforced as cl_target <= cl <= 1.2*cl_target
        "area_ratio_min": 0.8,
        "area_initial": calc_area(design["coords"]),
    }
    opt_design, optisteps = problem.optimize(design, config=config, mpicores=mpicores)
    print("optimized alpha:", round(float(opt_design["angle_of_attack"]), 4))
    print("optimized coords shape:", np.asarray(opt_design["coords"]).shape)

    print("\n=== 4a. objective trajectory (optisteps) ===")
    print(f"{len(optisteps)} steps recorded")
    if optisteps:
        print("first objective (cd):", optisteps[0].obj_values, "-> last:", optisteps[-1].obj_values)
    return opt_design, optisteps


def show_trajectory(problem: Airfoil) -> None:
    """Read the full geometry/surface trajectory of every intermediate design."""
    print("\n=== 4b. geometry + surface trajectory (optimization_trajectory) ===")
    # Geometry only (cheap): one entry per evaluated design, ordered by index.
    trajectory = problem.optimization_trajectory()
    print(f"{len(trajectory)} designs in the trajectory")
    if trajectory:
        first, last = trajectory[0], trajectory[-1]
        print(f"design {first['index']}: coords {first['coords'].shape}")
        print(f"design {last['index']}: coords {last['coords'].shape}")

    # With surface fields: each entry gains a DataFrame of every surface variable
    # (cp, Mach, skin friction, separation sensors, ...) per node.
    rich = problem.optimization_trajectory(include_surface=True)
    if rich:
        surf = rich[-1]["surface"]
        print("surface fields available:", list(surf.columns))
        if "CoefPressure" in surf.columns:
            cp = surf["CoefPressure"].to_numpy()
            print(f"final design cp range: [{cp.min():.3f}, {cp.max():.3f}]")


def advanced_overrides(problem: Airfoil, design: dict[str, Any], mpicores: int) -> None:
    """Show the advanced knobs: choose written surface fields and override the solver schedule."""
    print("\n=== 5. advanced overrides (surface_variables, solver_options) ===")
    config = {
        "mach": 0.5,
        "reynolds": 5.0e6,
        "temperature": 288.15,
        # Only write a small set of surface fields instead of the full default set.
        "surface_variables": ["cp", "mach", "yplus"],
        # Override the internal per-Mach ADflow schedule (merged last into aeroOptions).
        "solver_options": {"nCycles": 5000, "L2Convergence": 1e-9},
    }
    drag, lift = problem.simulate(design, config=config, mpicores=mpicores)
    print(f"(custom surface set + solver options) drag={drag:.6f} lift={lift:.6f}")


def show_artifacts(problem: Airfoil) -> None:
    """Point at the per-run debugging artifacts."""
    print("\n=== 6. debugging artifacts ===")
    output_dir = problem.study_output_dir
    if os.path.isdir(output_dir):
        wanted = ("opt.hst", "IPOPT.out", "final_abs_volume.npy")
        present = [f for f in sorted(os.listdir(output_dir)) if f in wanted or "intermediate" in f]
        print("artifacts:", present)
        abs_vol_path = os.path.join(output_dir, "final_abs_volume.npy")
        if os.path.exists(abs_vol_path):
            print("final absolute area:", np.load(abs_vol_path))


def show_dataset_generation() -> None:
    """Print the command that generates a v1 dataset at scale on Slurm."""
    print("\n=== 7. dataset generation on Slurm ===")
    print(
        "python engibench/problems/airfoil/dataset_slurm_airfoil_optimize.py \\\n"
        "    -account <hpc_account> -n_designs 100 -n_flows 1 -group_size 4 \\\n"
        "    -min_ma 0.4 -max_ma 1.2 -min_re 1e6 -max_re 1e8 \\\n"
        "    -min_temp 220 -max_temp 310 -return_history \\\n"
        "    --coords_file /path/to/baseline_airfoils.npy"
    )


def main() -> None:
    """Run the airfoil v1 feature tour."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords-file", type=str, default=None, help="Local .npy of baseline coordinates.")
    parser.add_argument("--design-index", type=int, default=0, help="Which baseline design to use.")
    parser.add_argument("--mpicores", type=int, default=4, help="MPI cores per CFD run.")
    parser.add_argument("--skip-optimize", action="store_true", help="Skip the (slow) optimization.")
    args = parser.parse_args()

    os.environ.setdefault("CONTAINER_RUNTIME", "apptainer")

    problem = Airfoil(seed=0)
    inspect_problem(problem)

    design = load_baseline(problem, args.coords_file, args.design_index)
    run_simulate(problem, design, args.mpicores)
    advanced_overrides(problem, design, args.mpicores)

    if not args.skip_optimize:
        run_optimize(problem, design, args.mpicores)
        show_trajectory(problem)
        show_artifacts(problem)

    show_dataset_generation()


if __name__ == "__main__":
    main()
