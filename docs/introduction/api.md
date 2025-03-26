# API

The main class defining a problem is `engibench.core.Problem`. It is defined as follows
```{eval-rst}
.. autoclass:: engibench.core.Problem
```


 ## Dataset
The dataset is a HuggingFace `Dataset` object that defines the dataset of the problem. This is typically useful to train ML models for inverse design or surrogate modeling.

A dataset is generally composed of several columns:
- `optimal_design`: The optimal design of the problem.
- All columns listed in `problem.objectives`: The objectives of the problem.
- All columns listed in `problem.conditions`: The conditions of the problem.
- Additional columns which can be useful for advanced usage.

## Methods
```{eval-rst}
.. automethod:: engibench.core.Problem.simulate
.. automethod:: engibench.core.Problem.optimize

    Where an OptiStep is defined as:

    .. autoclass:: engibench.core.OptiStep

.. automethod:: engibench.core.Problem.reset
.. automethod:: engibench.core.Problem.render
.. automethod:: engibench.core.Problem.random_design
```

## Attributes
```{eval-rst}
.. autoattribute:: engibench.core.Problem.objectives

    This attribute is a list of objectives that can be optimized. The objectives are defined as tuples where the first member is the objective name, and the second member is 'maximize' or 'minimize'.

    .. code::

        >>> problem.possible_objectives
        frozenset({('cl_val', ObjectiveDirection.MAXIMIZE), ('cd_val', ObjectiveDirection.MINIMIZE)})

.. autoattribute:: engibench.core.Problem.conditions

    This attribute list the conditions of the problem. The conditions are defined as tuples where the first member is the boundary condition name, and the second member is the value.

    .. code::

        >>> problem.boundary_conditions
        frozenset({('marchDist', 100.0), ('s0', 3e-06)}) # TODO update this

.. autoattribute:: engibench.core.Problem.design_space

    This attribute is a `gymnasium.spaces.Box` object that defines the design space of the problem. This is typically useful to define your neural network input/output layer.

    .. code::

        >>> problem.design_space
        Box(0.0, 1.0, (2, 192), float32)

.. autoattribute:: engibench.core.Problem.dataset_id

    This attribute is a string that defines the dataset id of the problem. This is typically useful to fetch the dataset from our [dataset registry](https://huggingface.co/IDEALLab).

    .. code::

        >>> problem.dataset_id
        'IDEALLab/airfoil_2d_v0'

.. autoattribute:: engibench.core.Problem.dataset

    This attribute is a HuggingFace `Dataset` object that defines the dataset of the problem. This is typically useful to train your neural network.

    .. code::

        >>> problem.dataset
        Dataset({
            features: ['design', 'objective'],
            num_rows: 1000
        })

.. autoattribute:: engibench.core.Problem.container_id

    This attribute is a string that defines the container id of the problem. We use it to fetch the simulator containers (generally from DockerHub).

    .. code::

        >>> problem.dataset_id
        'mdolab/public:u22-gcc-ompi-stable'
```
