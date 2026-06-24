# mypy: ignore-errors
"""2D airfoil analysis (v1).

Based on the MACH-Aero tutorials. The aeroOptions are brought in line with the v1
optimization template (RANS + QCR, internal per-Mach ANK/NK schedule, full surface
variables) so the cl/cd from ``simulate`` are self-consistent with ``optimize``.

The solver schedule is applied internally (the user does not tune it); advanced users
can override it via ``solver_options`` or restrict the written fields via
``surface_variables``.
"""

import itertools
import json
import os
import sys
from typing import Any

from adflow import ADFLOW
from baseclasses import AeroProblem
from cli_interface import AnalysisParameters
from cli_interface import DEFAULT_SURFACE_VARIABLES
from cli_interface import Task
from mpi4py import MPI
import numpy as np

# Mach-regime thresholds for the ANK/NK solver schedule.
_MACH_TRANSONIC_LO = 0.65
_MACH_TRANSONIC_HI = 0.8
_MACH_SONIC = 1.0


def ank_schedule(mach: float) -> dict:
    """Per-Mach ANK/NK schedule (kept in sync with airfoil_opt.py)."""
    ank_physical_ls_tol = 0.2
    cfl = 1.0
    ank_second_ord_switch_tol = 5e-8
    if _MACH_TRANSONIC_LO < mach < _MACH_TRANSONIC_HI:
        ank_second_ord_switch_tol = 1e-6
        ank_physical_ls_tol = 0.3
    elif _MACH_TRANSONIC_HI <= mach < _MACH_SONIC:
        ank_second_ord_switch_tol = 1e-5
        ank_physical_ls_tol = 0.4
    elif mach >= _MACH_SONIC:
        ank_second_ord_switch_tol = 1e-5
        ank_physical_ls_tol = 0.4
        cfl = 0.75
    return {
        "nCycles": 10000,
        "CFL": cfl,
        "ANKPhysicalLSTol": ank_physical_ls_tol,
        "ANKCFLCutback": 0.05,
        "ANKSecondOrdSwitchTol": ank_second_ord_switch_tol,
        "useDissContinuation": False,
        "dissContMagnitude": 0.2 * mach,
    }


def main() -> None:  # noqa: C901, PLR0915
    """Entry point of the script."""
    args = AnalysisParameters.from_dict(json.loads(sys.argv[1]))
    mesh_fname = args.mesh_fname
    output_dir = args.output_dir
    task = args.task

    mach = args.mach
    reynolds = args.reynolds
    altitude = args.altitude
    temperature = args.temperature
    use_altitude = args.use_altitude
    reynolds_length = 1.0
    surface_variables = args.surface_variables if args.surface_variables is not None else DEFAULT_SURFACE_VARIABLES

    comm = MPI.COMM_WORLD
    print(f"Processor {comm.rank} of {comm.size} is running")
    if not os.path.exists(output_dir) and comm.rank == 0:
        os.mkdir(output_dir)

    aero_options = {
        # I/O Parameters
        "gridFile": mesh_fname,
        "outputDirectory": output_dir,
        "writeVolumeSolution": False,
        "surfaceVariables": surface_variables,
        "writeTecplotSurfaceSolution": True,
        "monitorvariables": ["cpu", "cl", "cd", "yplus"],
        # Physics Parameters
        "equationType": "RANS",
        "smoother": "DADI",
        "rkreset": True,
        "nrkreset": 20,
        "useQCR": True,
        # ANK / NK
        "useanksolver": True,
        "ankswitchtol": 1.0,
        "useNKSolver": True,
        "nkswitchtol": 1e-5,
        "liftIndex": 2,
        "infchangecorrection": True,
        "nsubiterturb": 10,
        # Termination Criteria
        "L2Convergence": 1e-8,
        "L2ConvergenceCoarse": 1e-4,
    }
    aero_options.update(ank_schedule(mach))
    if args.solver_options:
        aero_options.update(args.solver_options)

    cfd_solver = ADFLOW(options=aero_options)

    # Slice the section so fc_*_slices.dat is written.
    cfd_solver.addSlices("z", np.array([0.5]), sliceType="absolute")

    alpha = args.alpha
    if use_altitude:
        ap = AeroProblem(
            name="fc", alpha=alpha, mach=mach, altitude=altitude, areaRef=1.0, chordRef=1.0, evalFuncs=["cl", "cd"]
        )
    else:
        ap = AeroProblem(
            name="fc",
            alpha=alpha,
            mach=mach,
            T=temperature,
            reynolds=reynolds,
            reynoldsLength=reynolds_length,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd"],
        )

    if task == Task.ANALYSIS:
        print("Running analysis")
        cfd_solver(ap)
        funcs: dict[str, Any] = {}
        cfd_solver.evalFunctions(ap, funcs)
        cfd_solver.checkSolutionFailure(ap, funcs)
        if comm.rank == 0:
            cl = funcs[f"{ap.name}_cl"]
            cd = funcs[f"{ap.name}_cd"]
            outputs = np.array([mach, reynolds, alpha, cl, cd])
            np.save(os.path.join(output_dir, "outputs.npy"), outputs)

    elif task == Task.POLAR:
        print("Running polar")
        alphas = np.linspace(0, 20, 50)
        alphas.sort()
        cl_list = []
        cd_list = []
        reslist = []
        for alpha_v in alphas:
            ap.name = f"fc_{alpha_v:4.2f}"
            ap.alpha = alpha_v
            if comm.rank == 0:
                print(f"current alpha: {ap.alpha}")
            cfd_solver(ap)
            funcs = {}
            cfd_solver.evalFunctions(ap, funcs)
            cl_list.append(funcs[f"{ap.name}_cl"])
            cd_list.append(funcs[f"{ap.name}_cd"])
            reslist.append(cfd_solver.getFreeStreamResidual(ap))
            if comm.rank == 0:
                print(f"CL: {cl_list[-1]}, CD: {cd_list[-1]}")
        if comm.rank == 0:
            outputs = np.array(
                list(
                    zip(itertools.repeat(mach), itertools.repeat(reynolds), alphas, cl_list, cd_list, reslist, strict=False)
                )
            )
            np.save(os.path.join(output_dir, "M_Re_alpha_CL_CD_res.npy"), outputs)

    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
