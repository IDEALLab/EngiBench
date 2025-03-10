# Parameter space discovery via slurm

This module allows submitting a slurm job array where each
individual job runs a simulation with a different set of parameters.

For a given problem `ExampleProblem` with design type `ExampleDesign`,
the following code will submit a slurm job array with 3 jobs:

```py
static_args = slurm.Args(simulate_args={"config": {"sim_arg": 10}}, problem_args={"some_arg": True})
parameter_space = [
     slurm.Args(problem_args={"problem_arg": 1}, design_args={"design_arg": -1}),
     slurm.Args(problem_args={"problem_arg": 2}, design_args={"design_arg": -2}),
     slurm.Args(problem_args={"problem_arg": 3}, design_args={"design_arg": -3}),
]
slurm.submit(
    problem=ExampleProblem,
    static_args=static_args,
    parameter_space=parameter_space),
)
```

For example the first job would execute:

```py
problem = ExampleProblem(some_arg=True, problem_arg=1)
design = ExampleDesign(design_arg=-1)
problem.simulate(design, config={"sim_arg": 10})
```

and so on.

The main steps needed to submit a job is to declare the arguments
as a `Args` object:

```{eval-rst}
.. autoclass:: engibench.utils.slurm.Args
   :members:
```

passing it to `submit()`:

```{eval-rst}
.. automethod:: engibench.utils.slurm.submit
```

To tweak the arguments passed to sbatch, the `config` argument can be passed to `submit()`:

```{eval-rst}
.. autoclass:: engibench.utils.slurm.SlurmConfig
   :members:
```
