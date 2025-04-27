# mypy: ignore-errors
"""This file is based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/

TEMPLATED VARS:
- $mesh_fname: Path to the mesh file.
- $output_dir: Path to the output directory.
- $task: The task to perform: "analysis" or "polar".
- $mach: The Mach number (float).
- $reynolds: The Reynolds number (float).
- $temperature: The temperature (float).
- $alpha: The angle of attack (float).
"""

import numpy as np
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI
import signal
import atexit
import sys

_exit_code = 0

def set_exit_code(code):
    global _exit_code
    _exit_code = code

def cleanup_handler(signum=None, frame=None):
    # Clean termination code for MPI
    global _exit_code
    try:
        from mpi4py import MPI
        if MPI.Is_initialized():
            if signum is not None:
                _exit_code = 1
            MPI.COMM_WORLD.Abort(_exit_code)
    except:
        pass
    sys.exit(_exit_code)

if __name__ == "__main__":

    # Register signal handlers
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    atexit.register(cleanup_handler)

    try:
        mesh_fname = $mesh_fname
        output_dir = $output_dir
        task = $task

        # mach number
        mach = $mach
        # Reynolds number
        reynolds = $reynolds
        # altitude
        altitude = $altitude
        # temperature
        T = $temperature
        # Whether to use altitude
        use_altitude = $use_altitude
        # Reynold's Length
        reynoldsLength = 1.0

        comm = MPI.COMM_WORLD
        print(f"Processor {comm.rank} of {comm.size} is running")
        if not os.path.exists(output_dir):
            if comm.rank == 0:
                os.mkdir(output_dir)

        # rst ADflow options
        aeroOptions = {
            # I/O Parameters
            "gridFile": mesh_fname,
            "outputDirectory": output_dir,
            "monitorvariables": ["cl", "cd","resrho","resrhoe"],
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
        CFDSolver = ADFLOW(options=aeroOptions)

        # Add features
        # CFDSolver.addLiftDistribution(150, "z")
        span = 1.0
        pos = np.array([0.5]) * span
        CFDSolver.addSlices("z", pos, sliceType="absolute")

        # rst Create AeroProblem
        alpha = $alpha

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
                T=T,
                reynolds=reynolds,
                reynoldsLength=reynoldsLength,
                areaRef=1.0,
                chordRef=1.0,
                evalFuncs=["cl", "cd"],
            )

        # rst Run ADflow
        if task == "analysis":
            print("Running analysis")
            # Solve
            CFDSolver(ap)
            # rst Evaluate and print
            funcs = {}
            CFDSolver.evalFunctions(ap, funcs)
            # Print the evaluated functions
            if comm.rank == 0:
                CL = funcs[f"{ap.name}_cl"]
                CD = funcs[f"{ap.name}_cd"]
                # Save the lift and drag coefficients to a file
                outputs = np.array([mach, reynolds, alpha, CL, CD])
                np.save(os.path.join(output_dir, "outputs.npy"), outputs)

        # rst Create polar arrays
        elif task == "polar":
            print("Running polar")
            # Create an array of alpha values.
            # In this case we create 5 random alpha values between 0 and 10
            # from numpy.random import uniform
            alphaList = np.linspace(0, 20, 50)
            # Sort the alpha values
            alphaList.sort()

            # Create storage for the evaluated lift and drag coefficients
            CLList = []
            CDList = []
            reslist = []
            # rst Start loop
            # Loop over the alpha values and evaluate the polar
            for alpha in alphaList:
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
                CFDSolver(ap)

                # Evaluate functions
                funcs = {}
                CFDSolver.evalFunctions(ap, funcs)

                # Store the function values in the output list
                CLList.append(funcs[f"{ap.name}_cl"])
                CDList.append(funcs[f"{ap.name}_cd"])
                reslist.append(CFDSolver.getFreeStreamResidual(ap))
                if comm.rank == 0:
                    print(f"CL: {CLList[-1]}, CD: {CDList[-1]}")

            # rst Print polar
            # Print the evaluated functions in a table
            if comm.rank == 0:
                outputs = []
                for alpha_v, cl, cd, res in zip(alphaList, CLList, CDList, reslist):
                    print(f"{alpha_v:6.1f} {cl:8.4f} {cd:8.4f}")
                    outputs.append([mach, reynolds, alpha_v, cl, cd, res])
                # Save the lift and drag coefficients to a file
                np.save(os.path.join(output_dir, "M_Re_alpha_CL_CD_res.npy"), outputs)

        MPI.COMM_WORLD.Barrier()
        set_exit_code(0)

    except Exception as e:
        print(f"Error: {e}")
        set_exit_code(1)
