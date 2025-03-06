# Beams 2D (MBB Beam)

**Lead**: Arthur Drake @arthurdrake1

Beams 2D is a benchmark problem that aims to optimize a 2D MBB beam using the structural topology optimization (TO) approach. This is based on the 88-line code by Andreassen et al. (2011). We allow the user to specify the required set of boundary conditions and other parameters prior to optimization.

## Side notes

Here is the script I've used to upload the data to HF:

```python
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

data_dir = './data_new'

# Step 1: Load the required files
xPrint = np.load(os.path.join(data_dir, "xPrint.npy"), allow_pickle=True)  # Shape: (1188, 100, 200)
compliance = np.load(os.path.join(data_dir, "c.npy"))  # Shape: (1188,)
params = np.load(os.path.join(data_dir, "params.npy"))  # Shape: (1188, 6)

# Step 2: Process xPrint images
# Flatten without transposing (ravel keeps the memory order intact)
# xPrint_flattened = np.array([np.swapaxes(img, -2, -1).ravel() for img in xPrint])  # Shape: (1188, 20000)

# Step 3: Split into train (80%), val (15%), test (5%)
train_data, temp_data, train_params, temp_params, train_compliance, temp_compliance = train_test_split(
    xPrint, params, compliance, test_size=0.20, random_state=42
)
val_data, test_data, val_params, test_params, val_compliance, test_compliance = train_test_split(
    temp_data, temp_params, temp_compliance, test_size=0.25, random_state=42
)  # 15% val, 5% test

# Step 4: Create Dataset Lists
dataset_train = [
    {
        "optimal_design": x,
        "nelx": int(2 * p[0]),
        "nely": int(p[0]),
        "volfrac": p[1],
        "penal": p[2],
        "rmin": p[3],
        "ft": int(p[4]),
        "max_iter": int(200),
        "overhang_constraint": int(p[5]),
        "c": c,
    }
    for x, p, c in zip(train_data, train_params, train_compliance)
]

dataset_val = [
    {
        "optimal_design": x,
        "nelx": int(2 * p[0]),
        "nely": int(p[0]),
        "volfrac": p[1],
        "penal": p[2],
        "rmin": p[3],
        "ft": int(p[4]),
        "max_iter": int(200),
        "overhang_constraint": int(p[5]),
        "c": c,
    }
    for x, p, c in zip(val_data, val_params, val_compliance)
]

dataset_test = [
    {
        "optimal_design": x,
        "nelx": int(2 * p[0]),
        "nely": int(p[0]),
        "volfrac": p[1],
        "penal": p[2],
        "rmin": p[3],
        "ft": int(p[4]),
        "max_iter": int(200),
        "overhang_constraint": int(p[5]),
        "c": c,
    }
    for x, p, c in zip(test_data, test_params, test_compliance)
]

# Step 5: Convert to Hugging Face Dataset
train_split = Dataset.from_list(dataset_train)
val_split = Dataset.from_list(dataset_val)
test_split = Dataset.from_list(dataset_test)

# Step 6: Create DatasetDict and push to Hugging Face
dataset_dict = DatasetDict({"train": train_split, "val": val_split, "test": test_split})

# Define repo name (change "your-username" to your HF username)
repo_name = "IDEALLab/beams_2d_v0"

# Upload to Hugging Face
dataset_dict.push_to_hub(repo_name)

print("Dataset successfully uploaded to Hugging Face!")

```
