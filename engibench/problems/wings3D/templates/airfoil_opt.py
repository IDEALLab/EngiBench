"""This file is largely based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/
"""

# ======================================================================
#         Import modules
# ======================================================================
import json
import os
import sys

from adflow import ADFLOW
from baseclasses import AeroProblem
from cli_interface import Algorithm  # type:ignore[import-not-found]
from cli_interface import OptimizeParameters
from idwarp import USMesh
from mpi4py import MPI
from multipoint import multiPointSparse
import numpy as np
from pygeo import DVConstraints
from pygeo import DVGeometry
from pyoptsparse import OPT
from pyoptsparse import Optimization


def main() -> None:  # noqa: C901, PLR0915
    """Entry point of the script."""
    args = OptimizeParameters.from_dict(json.loads(sys.argv[1]))
    # ======================================================================
    #         Specify parameters for optimization
    # ======================================================================
    # cL constraint
    mycl = args.cl_target
    # angle of attack
    alpha = args.alpha
    # mach number
    mach = args.mach
    # Reynolds number
    reynolds = args.reynolds
    # cruising altitude
    altitude = args.altitude
    # temperature
    temperature = args.temperature
    # Whether to use altitude
    use_altitude = args.use_altitude
    # Reynold's Length
    reynolds_length = 1.0
    # volume constraint ratio
    area_ratio_min = args.area_ratio_min
    area_initial = args.area_initial
    area_input_design = args.area_input_design

    # Optimization parameters
    opt = args.opt
    opt_options = args.opt_options
    # ======================================================================
    #         Create multipoint communication object
    # ======================================================================
    mp = multiPointSparse(MPI.COMM_WORLD)
    mp.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
    comm, *_ = mp.createCommunicators()
    if not os.path.exists(args.output_dir) and comm.rank == 0:
        os.mkdir(args.output_dir)

    # ======================================================================
    #         ADflow Set-up
    # ======================================================================
    aero_options = {
        # Common Parameters
        "gridFile": args.mesh_fname,
        "outputDirectory": args.output_dir,
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
    cfd_solver = ADFLOW(options=aero_options, comm=comm)
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
            T=temperature,
            reynolds=reynolds,
            reynoldsLength=reynolds_length,
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
    dv_geo = DVGeometry(args.ffd_fname)
    dv_geo.addLocalDV("shape", lower=-0.025, upper=0.025, axis="y", scale=1.0)

    span = 1.0
    pos = np.array([0.5]) * span
    cfd_solver.addSlices("z", pos, sliceType="absolute")

    # Add DVGeo object to CFD solver
    cfd_solver.setDVGeo(dv_geo)
    # ======================================================================
    #         DVConstraint Setup
    # ======================================================================

    dv_con = DVConstraints()
    dv_con.setDVGeo(dv_geo)

    # Only ADflow has the getTriangulatedSurface Function
    dv_con.setSurface(cfd_solver.getTriangulatedMeshSurface())

    # Le/Te constraints
    l_index = dv_geo.getLocalIndex(0)
    ind_set_a = []
    ind_set_b = []
    for k in range(1):
        ind_set_a.append(l_index[0, 0, k])  # all DV for upper and lower should be same but different sign
        ind_set_b.append(l_index[0, 1, k])
    for k in range(1):
        ind_set_a.append(l_index[-1, 0, k])
        ind_set_b.append(l_index[-1, 1, k])
    dv_con.addLeTeConstraints(0, indSetA=ind_set_a, indSetB=ind_set_b)

    # DV should be same along spanwise
    l_index = dv_geo.getLocalIndex(0)
    ind_set_a = []
    ind_set_b = []
    for i in range(l_index.shape[0]):
        ind_set_a.append(l_index[i, 0, 0])
        ind_set_b.append(l_index[i, 0, 1])
    for i in range(l_index.shape[0]):
        ind_set_a.append(l_index[i, 1, 0])
        ind_set_b.append(l_index[i, 1, 1])
    dv_con.addLinearConstraintsShape(ind_set_a, ind_set_b, factorA=1.0, factorB=-1.0, lower=0, upper=0)

    le = 0.010001
    le_list = [[le, 0, le], [le, 0, 1.0 - le]]
    te_list = [[1.0 - le, 0, le], [1.0 - le, 0, 1.0 - le]]

    dv_con.addVolumeConstraint(
        le_list,
        te_list,
        2,
        100,
        lower=area_ratio_min * area_initial / area_input_design,
        upper=1.2 * area_initial / area_input_design,
        scaled=True,
    )
    dv_con.addThicknessConstraints2D(le_list, te_list, 2, 100, lower=0.15, upper=3.0)
    # Final constraint to keep TE thickness at original or greater
    dv_con.addThicknessConstraints1D(ptList=te_list, nCon=2, axis=[0, 1, 0], lower=1.0, scaled=True)

    if comm.rank == 0:
        file_name = os.path.join(args.output_dir, "constraints.dat")
        dv_con.writeTecplot(file_name)
    # ======================================================================
    #         Mesh Warping Set-up
    # ======================================================================
    mesh_options = {"gridFile": args.mesh_fname}

    mesh = USMesh(options=mesh_options, comm=comm)
    cfd_solver.setMesh(mesh)

    # ======================================================================
    #         Optimization Problem Set-up
    # ======================================================================
    # Create optimization problem
    opt_prob = Optimization("opt", mp.obj, comm=MPI.COMM_WORLD)

    # Add objective
    opt_prob.addObj("obj", scale=1e4)

    # Add variables from the AeroProblem
    ap.addVariablesPyOpt(opt_prob)

    # Add DVGeo variables
    dv_geo.addVariablesPyOpt(opt_prob)

    # Add constraints
    dv_con.addConstraintsPyOpt(opt_prob)
    opt_prob.addCon("cl_con_" + ap.name, lower=0.0, upper=0.0, scale=1.0)

    # The MP object needs the 'obj' and 'sens' function for each proc set,
    # the optimization problem and what the objcon function is:
    def cruise_funcs(x):
        if MPI.COMM_WORLD.rank == 0:
            print(x)
        # Set design vars
        dv_geo.setDesignVars(x)
        ap.setDesignVars(x)
        # Run CFD
        cfd_solver(ap)
        # Evaluate functions
        funcs = {}
        dv_con.evalFunctions(funcs)
        cfd_solver.evalFunctions(ap, funcs)
        cfd_solver.checkSolutionFailure(ap, funcs)
        if MPI.COMM_WORLD.rank == 0:
            print(funcs)
        return funcs

    def cruise_funcs_sens(_x, _funcs):
        funcs_sens = {}
        dv_con.evalFunctionsSens(funcs_sens)
        cfd_solver.evalFunctionsSens(ap, funcs_sens)
        cfd_solver.checkAdjointFailure(ap, funcs_sens)
        if MPI.COMM_WORLD.rank == 0:
            print(funcs_sens)
        return funcs_sens

    def obj_con(funcs, print_ok):
        # Assemble the objective and any additional constraints:
        funcs["obj"] = funcs[ap["cd"]]
        funcs["cl_con_" + ap.name] = funcs[ap["cl"]] - mycl
        if print_ok:
            print("funcs in obj:", funcs)
        return funcs

    mp.setProcSetObjFunc("cruise", cruise_funcs)
    mp.setProcSetSensFunc("cruise", cruise_funcs_sens)
    mp.setObjCon(obj_con)
    mp.setOptProb(opt_prob)
    opt_prob.printSparsity()
    # Set up optimizer
    if opt == Algorithm.SLSQP:
        opt_options = {"IFILE": os.path.join(args.output_dir, "SLSQP.out")}
    elif opt == Algorithm.SNOPT:
        opt_options = {
            "Major feasibility tolerance": 1e-4,
            "Major optimality tolerance": 1e-4,
            "Hessian full memory": None,
            "Function precision": 1e-8,
            "Print file": os.path.join(args.output_dir, "SNOPT_print.out"),
            "Summary file": os.path.join(args.output_dir, "SNOPT_summary.out"),
        }
    opt_options.update(opt_options)
    opt = OPT(opt.name, options=opt_options)

    # Run Optimization
    sol = opt(opt_prob, mp.sens, sensMode="pgc", sensStep=1e-6, storeHistory=os.path.join(args.output_dir, "opt.hst"))
    if MPI.COMM_WORLD.rank == 0:
        print(sol)

    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
