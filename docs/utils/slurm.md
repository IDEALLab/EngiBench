# Parameter space discovery via slurm

This module allows submitting a python callback as a slurm job array where each
individual job runs the callback with a different set of parameters.

```{testsetup} slurm
import numpy as np
from engibench.utils import slurm
import fake_sbatch
from examples.beams2d_patched import run_job

slurm_args = slurm.SlurmConfig(sbatch_executable=fake_sbatch.__file__)
np.set_printoptions(edgeitems=1)
```

**Step 1:** Define a callback and the parameter space

```{literalinclude} examples/beams2d.py
```

```{testcode} slurm
parameter_space = [
    {
        "config": {
            "volfrac": volfrac,
            "forcedist": forcedist,
        }
    }
    for forcedist in [0.0, 0.5]
    for volfrac in [0.1, 0.25, 0.8]
]
```

**Step 2:** Specify arguments to be passed to slurm:

```{testcode} slurm-real
from engibench.utils import slurm
slurm_args = slurm.SlurmConfig(runtime="1:00:00")
```

**Step 3:** Submit a slurm job array

```{testcode} slurm

job = slurm.sbatch_map(
    run_job,
    args=parameter_space,
    slurm_args=slurm_args,
)
```

For example this would submit a slurm array with 6 elements running

```py
run_job(config={"volfrac": 0.1, "forcedist": 0.0}) # element 1
run_job(config={"volfrac": 0.25, "forcedist": 0.0}) # element 2
run_job(config={"volfrac": 0.8, "forcedist": 0.0}) # element 3
run_job(config={"volfrac": 0.1, "forcedist": 0.5}) # element 4
run_job(config={"volfrac": 0.25, "forcedist": 0.5}) # element 5
run_job(config={"volfrac": 0.8, "forcedist": 0.5}) # element 6
```

By default, [sbatch_map()](#engibench.utils.slurm.sbatch_map) submits the job array in the background. That means that the execution flow of the python script will continue while the jobs are running.
If the calling python scripts needs to load results from the jobs, `wait=True` can be passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map). In this case the call to [sbatch_map()](#engibench.utils.slurm.sbatch_map) will block until all jobs have completed.

Results of a submitted job also can be either "reduced" or saved automatically:

**Step 4a:** Reduce results from multiple job array elements

```{testcode} slurm

reduced = job.reduce(list, slurm_args=slurm_args) # Collect all results to a list

# To save resources, to render the docs no actual optimization is performed.
# Instead optimize() is replaced by a method returning zeros:
print(reduced) # doctest: +ELLIPSIS
```

```{testoutput} slurm
[array([[0., ..., 0.]], shape=(100, 50)), ...]
```

**Step 4b:** Save all results in one [pickle](https://docs.python.org/3/library/pickle.html) archive

```py
job.save("results.pkl", slurm_args=slurm_args)
```

[job.save()](#engibench.utils.slurm.SubmittedJobArray.save) will submit another slurm job using `slurm_args` (can be chosen independently from the arguments passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map)) which will wait until all parallel jobs submitted by [sbatch_map()](#engibench.utils.slurm.sbatch_map) have completed.


The module contains a convenience wrapper around [pickle.load()](https://docs.python.org/3/library/pickle.html#pickle.load) to load result archives:

```py
results = slurm.load_results("results.pkl")
```

```{note}
In contrast to [sbatch_map()](#engibench.utils.slurm.sbatch_map), [job.save()](#engibench.utils.slurm.SubmittedJobArray.save) and [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce) are blocking.
The only reason that [sbatch_map()](#engibench.utils.slurm.sbatch_map) is non-blocking is to make it possible to chain this call with [job.save()](#engibench.utils.slurm.SubmittedJobArray.save)/ [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce).
If however `wait=True` is passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map), the call will be also blocking, in which case chaining with [job.save()](#engibench.utils.slurm.SubmittedJobArray.save) or [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce) is not possible.
```

## Error handling

Errors occurring during a job will be handled as good as possible.

**[sbatch_map()](#engibench.utils.slurm.sbatch_map)/[job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce) workflow**:
[sbatch_map()](#engibench.utils.slurm.sbatch_map) will try to pass all results collected from [sbatch_map()](#engibench.utils.slurm.sbatch_map) to the reduce callback,
passing a [JobError](#engibench.utils.slurm.JobError) instances for every failed job.
This gives the callback the opportunity to handle failed jobs itself.
If the callback does not handle failed job, [sbatch_map()](#engibench.utils.slurm.sbatch_map) will raise a  [JobError](#engibench.utils.slurm.JobError) containing information about all failed jobs.
If the callback raises an exception on its own, [sbatch_map()](#engibench.utils.slurm.sbatch_map) will raise a [JobError](#engibench.utils.slurm.JobError) wrapping the occurred error.

**[sbatch_map()](#engibench.utils.slurm.sbatch_map)/ [job.save()](#engibench.utils.slurm.SubmittedJobArray.save) workflow**: any failed job will produce a [pickle](https://docs.python.org/3/library/pickle.html) archive containing the exception instead of a result.

## Technical details

The actual slurm job array will run instances of [engibench/utils/slurm/run_job.py](https://github.com/IDEALLab/EngiBench/blob/main/engibench/utils/slurm.py). The data to be processed will be serialized by [sbatch_map()](#engibench.utils.slurm.sbatch_map) and deserialized by [run_job.py](https://github.com/IDEALLab/EngiBench/blob/main/engibench/utils/slurm.py) using [pickle](https://docs.python.org/3/library/pickle.html).
This also includes the callable itself.
The pickle archive will contain the name of the module where the callable is defined together with the name of the callable.
If the callable is a bound method of a class instance, the arguments needed to reconstruct the class instance are serialized as well.
By default pickle only manages to deserialize callables defined in python modules which are reachable via `PYTHONPATH`.
To also allow python modules not reachable via `PYTHONPATH`, `engibench.utils.slurm` has an enhanced serializing mechanism (`engibench.utils.slurm.MemorizeModule`).
```{note}
As deserializing works via module name, callable name and arguments, the callable must be defined at the toplevel of the module and cannot be a python object which is returned by a factory function.
```
The execution flow is depicted in the [following figure](#slurm-execution-flow).
The execution flow of [sbatch_map()](#engibench.utils.slurm.sbatch_map) + [save()](#engibench.utils.slurm.SubmittedJobArray.save) is a bit simpler - it does not need the final deserializing step from slurm back to the calling script.


```{figure} slurm-execution-flow.svg
:alt: slurm execution flow
:name: slurm-execution-flow

Execution flow of [sbatch_map()](#engibench.utils.slurm.sbatch_map) followed by [reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce)
```

## API

```{eval-rst}
.. autofunction:: engibench.utils.slurm::sbatch_map
```

```{eval-rst}
.. autoclass:: engibench.utils.slurm::SubmittedJobArray
   :members:
```

To tweak the arguments passed to sbatch, the `config` argument can be passed to `submit()`:

```{eval-rst}
.. autoclass:: engibench.utils.slurm.SlurmConfig
   :members:
```

```{eval-rst}
.. autoclass:: engibench.utils.slurm.JobError
   :members:
```
