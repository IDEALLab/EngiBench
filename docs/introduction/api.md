# API

The main class defining a problem is `engibench.core.Problem`. It is defined as follows
```{eval-rst}
.. autoclass:: engibench.core.Problem
```



## Methods
```{eval-rst}
.. automethod:: engibench.core.Problem.simulate
.. automethod:: engibench.core.Problem.optimize
.. automethod:: engibench.core.Problem.reset
.. automethod:: engibench.core.Problem.render
.. automethod:: engibench.core.Problem.random_design
```

## Attributes
```{eval-rst}
.. autoattribute:: engibench.core.Problem.possible_objectives

    This attribute is a list of possible objectives that can be optimized. The objectives are defined as tuples where the first member is the objective name, and the second member is 'maximize' or 'minimize'.

    .. code::

        >>> problem.possible_objectives
        frozenset({('lift', 'maximize'), ('drag', 'minimize')})

.. autoattribute:: engibench.core.Problem.boundary_conditions

    This attribute list the boundary conditions of the problem. The boundary conditions are defined as tuples where the first member is the boundary condition name, and the second member is the value.

    .. code::

        >>> problem.boundary_conditions
        frozenset({('marchDist', 100.0), ('s0', 3e-06)})

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

.. autoattribute:: engibench.core.Problem.input_space

    This is an internal attribute that defines the input space of the simulator -- we use str for simulators relying on files. Users should not use this attribute.
```
