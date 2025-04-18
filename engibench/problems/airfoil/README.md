# Airfoil 2D

**Lead**: Cashen Diniz @cashend

Airfoil 2D is a benchmark problem that aims to optimize the shape of an airfoil to maximize the lift-to-drag ratio.
We rely on MACH-Aero for the simulations.

## Side notes

Here is the script I've used to upload the data to HF using the pickle files here: https://github.com/IDEALLab/OptimizingDiffusionSciTech2024/tree/main/data/optimized_data

```python
import pandas as pd
import numpy as np
import datasets
from datasets import Dataset, DatasetDict


opt_train_airfoils, opt_test_airfoils, opt_val_airfoils = pd.read_pickle("train_test_val_opt_airfoils.pkl")
init_train_airfoils, init_test_airfoils, init_val_airfoils = pd.read_pickle("train_test_val_init_airfoils.pkl")
train_params, test_params, val_params = pd.read_pickle("train_test_val_opt_params.pkl")

# For each airfoil, we need one row containing the initial and optimized airfoil, as well as the parameters

dataset_train = []

for o, i, p in zip(opt_train_airfoils, init_train_airfoils, train_params):
    dataset_train.append(
        {
            "initial_design": {"coords": i, "angle_of_attack": np.cast[np.float32](p[4])},
            "optimal_design": {"coords": o, "angle_of_attack": np.cast[np.float32](p[4])},
            "mach": p[0],
            "reynolds": p[1],
            "cl_target": p[2],
            "area_ratio_min": p[3],
            "area_initial": p[5],
            "cd": p[6],
            "cl": p[7],
            "cl_con_violation": p[8],
            "area_ratio": p[9],
        }
    )

dataset_val = []

for o, i, p in zip(opt_test_airfoils, init_test_airfoils, test_params):
    dataset_val.append(
        {
            "initial_design": {"coords": i, "angle_of_attack": np.cast[np.float32](p[4])},
            "optimal_design": {"coords": o, "angle_of_attack": np.cast[np.float32](p[4])},
            "mach": p[0],
            "reynolds": p[1],
            "cl_target": p[2],
            "area_ratio_min": p[3],
            "area_initial": p[5],
            "cd": p[6],
            "cl": p[7],
            "cl_con_violation": p[8],
            "area_ratio": p[9],
        }
    )

dataset_testt = []

for o, i, p in zip(opt_val_airfoils, init_val_airfoils, val_params):
    dataset_testt.append(
        {
            "initial_design": {"coords": i, "angle_of_attack": np.cast[np.float32](p[4])},
            "optimal_design": {"coords": o, "angle_of_attack": np.cast[np.float32](p[4])},
            "mach": p[0],
            "reynolds": p[1],
            "cl_target": p[2],
            "area_ratio_min": p[3],
            "area_initial": p[5],
            "cd": p[6],
            "cl": p[7],
            "cl_con_violation": p[8],
            "area_ratio": p[9],
        }
    )


# Create a huggingface dataset from the three splits above
train_spit = Dataset.from_list(dataset_train)
print(train_spit.shape)
val_spit = Dataset.from_list(dataset_val)
test_spit = Dataset.from_list(dataset_testt)
dataset_dict = DatasetDict({"train": train_spit, "val": val_spit, "test": test_spit})
dataset_dict.push_to_hub("IDEALLab/airfoil_v0")

```
