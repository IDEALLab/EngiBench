# HeatConduction2D

``` {problem:table}
:lead: Milad Habibi @MIladHB
```

## Design space
These problems are a specific subset of topology optimization problems aimed at minimizing thermal compliance within a unit square (2D) subject to: a constraint on the volume of highly conductive material used, and given boundary conditions,
particularly the location of the adiabatic region. The adiabatic region refers to a symmetric length on the bottom side of the 2D problem space . The design space for the 2D problem consists of a 2D array representing solid densities,
which is parametrized by resolution, that is, 'DesignSpace = [0,1] By default, a $101 \times 101$ space is used for the 2D problem.

## Objectives
The objective is defined and indexed as follows:

0. `c`: Thermal compliance coefficient to minimize.

## Conditions
The conditions are defined by the following parameters:

```{problem:conditions}
```

## Simulator
The simulator is a docker container with the dolfin-adjoint software that computes the thermal compliance of the design.
We convert use intermediary files to convert from and to the simulator that is run from a Docker image.

## Dataset
The dataset has been generated the dolfin-adjoint software. It is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/heat_conduction_2d_v0).

### v0

#### Fields
The dataset only contains conditions and optimal designs (no objective).

#### Creation Method
The creation method for the dataset is specified in the reference paper.

## References
If you use this problem in your research, please cite the following paper:
Milad Habibi, Jun Wang, and Mark Fuge, "When Is it Actually Worth Learning Inverse Design?" in IDETC 2023. doi: https://doi.org/10.1115/DETC2023-116678
