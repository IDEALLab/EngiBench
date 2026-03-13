"""This file is based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/
"""

import itertools
import json
import os
import sys
from typing import Any

from adflow import ADFLOW
from baseclasses import AeroProblem
from cli_interface import AnalysisParameters  # type:ignore[import-not-found]
from cli_interface import Task
from mpi4py import MPI
import numpy as np


def main() -> None:  # noqa: C901, PLR0915
    """Entry point of the script."""
    args = AnalysisParameters.from_dict(json.loads(sys.argv[1]))
    mesh_fname = args.mesh_fname
    output_dir = args.output_dir
    task = args.task

    # mach number
    mach = args.mach
    # Reynolds number
    reynolds = args.reynolds
    # altitude
    altitude = args.altitude
    # temperature
    temperature = args.temperature
    # Whether to use altitude
    use_altitude = args.use_altitude
    # Reynold's Length
    reynolds_length = 1.0

    comm = MPI.COMM_WORLD
    print(f"Processor {comm.rank} of {comm.size} is running")
    if not os.path.exists(output_dir) and comm.rank == 0:
        os.mkdir(output_dir)

    # rst ADflow options
    aero_options = {
        # I/O Parameters
        "gridFile": mesh_fname,
        "outputDirectory": output_dir,
        "monitorvariables": ["cl", "cd", "resrho", "resrhoe"],
        "writeTecplotSurfaceSolution": True,
        # Physics Parameters
        "equationType": "RANS",
        "smoother": "DADI",
        "rkreset": True,
        "nrkreset": 10,
        # Solver Parameters
        "MGCycle": "sg",
        # ANK Solver Parameters
        "useANKSolver": True,
        "ankswitchtol": 1e-1,
        "liftIndex": 2,
        "nsubiterturb": 10,
        # NK Solver Parameters
        "useNKSolver": True,
        "NKSwitchTol": 1e-4,
        # Termination Criteria
        "L2Convergence": 1e-9,
        "L2ConvergenceCoarse": 1e-4,
        "nCycles": 5000,
    }
    print("ADflow options:")
    # rst Start ADflow
    # Create solver
    cfd_solver = ADFLOW(options=aero_options)

    # Add features
    span = 1.0
    pos = np.array([0.5]) * span
    cfd_solver.addSlices("z", pos, sliceType="absolute")

    # rst Create AeroProblem
    alpha = args.alpha

    if use_altitude:
        ap = AeroProblem(
            name="fc",
            alpha=alpha,
            mach=mach,
            altitude=altitude,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd"],
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

    # rst Run ADflow
    if task == Task.ANALYSIS:
        print("Running analysis")
        # Solve
        cfd_solver(ap)
        # rst Evaluate and print
        funcs: dict[str, Any] = {}
        cfd_solver.evalFunctions(ap, funcs)
        # Print the evaluated functions
        if comm.rank == 0:
            cl = funcs[f"{ap.name}_cl"]
            cd = funcs[f"{ap.name}_cd"]
            # Save the lift and drag coefficients to a file
            outputs = np.array([mach, reynolds, alpha, cl, cd])
            np.save(os.path.join(output_dir, "outputs.npy"), outputs)

    # rst Create polar arrays
    elif task == Task.POLAR:
        print("Running polar")
        # Create an array of alpha values.
        # In this case we create 5 random alpha values between 0 and 10
        alphas = np.linspace(0, 20, 50)
        # Sort the alpha values
        alphas.sort()

        # Create storage for the evaluated lift and drag coefficients
        cl_list = []
        cd_list = []
        reslist = []
        # rst Start loop
        # Loop over the alpha values and evaluate the polar
        for alpha in alphas:
            # rst update AP
            # Update the name in the AeroProblem. This allows us to modify the
            # output file names with the current alpha.
            ap.name = f"fc_{alpha:4.2f}"

            # Update the alpha in aero problem and print it to the screen.
            ap.alpha = alpha
            if comm.rank == 0:
                print(f"current alpha: {ap.alpha}")

            # rst Run ADflow polar
            # Solve the flow
            cfd_solver(ap)

            # Evaluate functions
            funcs = {}
            cfd_solver.evalFunctions(ap, funcs)

            # Store the function values in the output list
            cl_list.append(funcs[f"{ap.name}_cl"])
            cd_list.append(funcs[f"{ap.name}_cd"])
            reslist.append(cfd_solver.getFreeStreamResidual(ap))
            if comm.rank == 0:
                print(f"CL: {cl_list[-1]}, CD: {cd_list[-1]}")

        # rst Print polar
        # Print the evaluated functions in a table
        if comm.rank == 0:
            outputs = np.array(
                list(
                    zip(itertools.repeat(mach), itertools.repeat(reynolds), alphas, cl_list, cd_list, reslist, strict=False)
                )
            )
            for *_, alpha_v, cl, cd, _res in outputs:
                print(f"{alpha_v:6.1f} {cl:8.4f} {cd:8.4f}")
            # Save the lift and drag coefficients to a file
            np.save(os.path.join(output_dir, "M_Re_alpha_CL_CD_res.npy"), outputs)

    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
