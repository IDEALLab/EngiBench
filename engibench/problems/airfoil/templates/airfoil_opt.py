# mypy: ignore-errors
"""This file is largely based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/

TEMPLATED VARS:
- $cl_target: The lift coefficient constraint (float).
- $alpha: The angle of attack (float).
- $mach: The Mach number (float).
- $altitude: The cruising altitude (int).
- $ffd_fname: Path to the FFD file.
- $mesh_fname: Path to the mesh file.
- $output_dir: Path to the output directory.
- $opt: The optimization algorithm: SLSQP or SNOPT.
- $opt_options: The optimization options (dict).
"""

# ======================================================================
#         Import modules
# ======================================================================
import os

from adflow import ADFLOW
from baseclasses import AeroProblem
from idwarp import USMesh
from mpi4py import MPI
from multipoint import multiPointSparse
import numpy as np
from pygeo import DVConstraints
from pygeo import DVGeometry
from pyoptsparse import OPT
from pyoptsparse import Optimization

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

# ======================================================================
#         Functions:
# ======================================================================
def cruiseFuncs(x):
    if MPI.COMM_WORLD.rank == 0:
        print(x)
    # Set design vars
    DVGeo.setDesignVars(x)
    ap.setDesignVars(x)
    # Run CFD
    CFDSolver(ap)
    # Evaluate functions
    funcs = {}
    DVCon.evalFunctions(funcs)
    CFDSolver.evalFunctions(ap, funcs)
    CFDSolver.checkSolutionFailure(ap, funcs)
    if MPI.COMM_WORLD.rank == 0:
        print(funcs)
    return funcs


def cruiseFuncsSens(x, funcs):
    funcsSens = {}
    DVCon.evalFunctionsSens(funcsSens)
    CFDSolver.evalFunctionsSens(ap, funcsSens)
    CFDSolver.checkAdjointFailure(ap, funcsSens)
    if MPI.COMM_WORLD.rank == 0:
        print(funcsSens)
    return funcsSens


def objCon(funcs, printOK):
    # Assemble the objective and any additional constraints:
    funcs["obj"] = funcs[ap["cd"]]
    funcs["cl_con_" + ap.name] = funcs[ap["cl"]] - mycl
    if printOK:
        print("funcs in obj:", funcs)
    return funcs


if __name__ == "__main__":

    # Register signal handlers
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    atexit.register(cleanup_handler)

    try:
        # ======================================================================
        #         Specify parameters for optimization
        # ======================================================================
        # cL constraint
        mycl = $cl_target
        # angle of attack
        alpha = $alpha
        # mach number
        mach = $mach
        # Reynolds number
        reynolds = $reynolds
        # cruising altitude
        altitude = $altitude
        # temperature
        T = $temperature
        # Whether to use altitude
        use_altitude = $use_altitude
        # Reynold's Length
        reynoldsLength = 1.0
        # volume constraint ratio
        area_ratio_min = $area_ratio_min
        area_initial = $area_initial
        area_input_design = $area_input_design

        # Optimization parameters
        opt = $opt
        opt_options = $opt_options
        # ======================================================================
        #         Create multipoint communication object
        # ======================================================================
        MP = multiPointSparse(MPI.COMM_WORLD)
        MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
        comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
        if not os.path.exists($output_dir):
            if comm.rank == 0:
                os.mkdir($output_dir)

        # ======================================================================
        #         ADflow Set-up
        # ======================================================================
        aeroOptions = {
            # Common Parameters
            "gridFile": $mesh_fname,
            "outputDirectory": $output_dir,
            #"writeSurfaceSolution": False,
            "writeVolumeSolution": False,
            "writeTecplotSurfaceSolution": True,
            "monitorvariables": ["cl", "cd", "yplus"],
            # Physics Parameters
            "equationType": "RANS",
            "smoother": "DADI",
            "nCycles": 5000,
            "rkreset": True,
            "nrkreset": 10,
            # NK Options
            "useNKSolver": True,
            "nkswitchtol": 1e-8,
            # ANK Options
            "useanksolver": True,
            "ankswitchtol": 1e-1,
            # "ANKCoupledSwitchTol": 1e-6,
            # "ANKSecondOrdSwitchTol": 1e-5,
            "liftIndex": 2,
            "infchangecorrection": True,
            "nsubiterturb": 10,
            # Convergence Parameters
            "L2Convergence": 1e-8,
            "L2ConvergenceCoarse": 1e-4,
            # Adjoint Parameters
            "adjointSolver": "GMRES",
            "adjointL2Convergence": 1e-8,
            "ADPC": True,
            "adjointMaxIter": 1000,
            "adjointSubspaceSize": 200,
        }

        # Create solver
        CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
        # ======================================================================
        #         Set up flow conditions with AeroProblem
        # ======================================================================

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

        # Add angle of attack variable
        ap.addDV("alpha", value=alpha, lower=0.0, upper=10.0, scale=1.0)
        # ======================================================================
        #         Geometric Design Variable Set-up
        # ======================================================================
        # Create DVGeometry object
        DVGeo = DVGeometry($ffd_fname)
        DVGeo.addLocalDV("shape", lower=-0.025, upper=0.025, axis="y", scale=1.0)

        span = 1.0
        pos = np.array([0.5]) * span
        CFDSolver.addSlices("z", pos, sliceType="absolute")

        # Add DVGeo object to CFD solver
        CFDSolver.setDVGeo(DVGeo)
        # ======================================================================
        #         DVConstraint Setup
        # ======================================================================

        DVCon = DVConstraints()
        DVCon.setDVGeo(DVGeo)

        # Only ADflow has the getTriangulatedSurface Function
        DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())

        # Le/Te constraints
        lIndex = DVGeo.getLocalIndex(0)
        indSetA = []
        indSetB = []
        for k in range(0, 1):
            indSetA.append(lIndex[0, 0, k])  # all DV for upper and lower should be same but different sign
            indSetB.append(lIndex[0, 1, k])
        for k in range(0, 1):
            indSetA.append(lIndex[-1, 0, k])
            indSetB.append(lIndex[-1, 1, k])
        DVCon.addLeTeConstraints(0, indSetA=indSetA, indSetB=indSetB)

        # DV should be same along spanwise
        lIndex = DVGeo.getLocalIndex(0)
        indSetA = []
        indSetB = []
        for i in range(lIndex.shape[0]):
            indSetA.append(lIndex[i, 0, 0])
            indSetB.append(lIndex[i, 0, 1])
        for i in range(lIndex.shape[0]):
            indSetA.append(lIndex[i, 1, 0])
            indSetB.append(lIndex[i, 1, 1])
        DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0)

        le = 0.010001
        leList = [[le, 0, le], [le, 0, 1.0 - le]]
        teList = [[1.0 - le, 0, le], [1.0 - le, 0, 1.0 - le]]

        DVCon.addVolumeConstraint(leList, teList, 2, 100, lower=area_ratio_min*area_initial/area_input_design, upper=1.2*area_initial/area_input_design, scaled=True)
        DVCon.addThicknessConstraints2D(leList, teList, 2, 100, lower=0.15, upper=3.0)
        # Final constraint to keep TE thickness at original or greater
        DVCon.addThicknessConstraints1D(ptList=teList, nCon=2, axis=[0, 1, 0], lower=1.0, scaled=True)

        if comm.rank == 0:
            fileName = os.path.join($output_dir, "constraints.dat")
            DVCon.writeTecplot(fileName)
        # ======================================================================
        #         Mesh Warping Set-up
        # ======================================================================
        meshOptions = {"gridFile": $mesh_fname}

        mesh = USMesh(options=meshOptions, comm=comm)
        CFDSolver.setMesh(mesh)

        # ======================================================================
        #         Optimization Problem Set-up
        # ======================================================================
        # Create optimization problem
        optProb = Optimization("opt", MP.obj, comm=MPI.COMM_WORLD)

        # Add objective
        optProb.addObj("obj", scale=1e4)

        # Add variables from the AeroProblem
        ap.addVariablesPyOpt(optProb)

        # Add DVGeo variables
        DVGeo.addVariablesPyOpt(optProb)

        # Add constraints
        DVCon.addConstraintsPyOpt(optProb)
        optProb.addCon("cl_con_" + ap.name, lower=0.0, upper=0.0, scale=1.0)

        # The MP object needs the 'obj' and 'sens' function for each proc set,
        # the optimization problem and what the objcon function is:
        MP.setProcSetObjFunc("cruise", cruiseFuncs)
        MP.setProcSetSensFunc("cruise", cruiseFuncsSens)
        MP.setObjCon(objCon)
        MP.setOptProb(optProb)
        optProb.printSparsity()
        # Set up optimizer
        if opt == "SLSQP":
            optOptions = {"IFILE": os.path.join($output_dir, "SLSQP.out")}
        elif opt == "SNOPT":
            optOptions = {
                "Major feasibility tolerance": 1e-4,
                "Major optimality tolerance": 1e-4,
                "Hessian full memory": None,
                "Function precision": 1e-8,
                "Print file": os.path.join($output_dir, "SNOPT_print.out"),
                "Summary file": os.path.join($output_dir, "SNOPT_summary.out"),
            }
        optOptions.update(opt_options)
        opt = OPT(opt, options=optOptions)

        # Run Optimization
        sol = opt(optProb, MP.sens, sensMode='pgc', sensStep=1e-6, storeHistory=os.path.join($output_dir, "opt.hst"))
        if MPI.COMM_WORLD.rank == 0:
            print(sol)

        MPI.COMM_WORLD.Barrier()
        set_exit_code(0)

    except Exception as e:
        print(f"Error: {e}")
        set_exit_code(1)
