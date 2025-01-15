from engibench.problems.airfoil2d.airfoil2d import Airfoil2D
problem = Airfoil2D()
problem.reset(seed=0, cleanup=False)

dataset = problem.dataset

# Get design and conditions from the dataset
design = problem.random_design()
problem.optimize(design, mpicores=1)