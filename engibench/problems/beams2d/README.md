# Beams 2D (MBB Beam)

**Lead**: Arthur Drake @arthurdrake1

Beams 2D is a benchmark problem that aims to optimize a 2D MBB beam using the structural topology optimization (TO) approach. This is based on the 88-line code by Andreassen et al. (2011). We allow the user to specify the required set of boundary conditions and other parameters prior to optimization.

## Side notes

Here is the script I've used to generate the dataset conditions. Please note that `max_iter = 100` and it is assumed that `nelx = 2*nely`. This yields a total of 14553 samples, or 4851 samples for each of the three image resolutions.

```python
all_params = [
    np.array([25, 50, 100]),                        # nely (nelx = 2*nely)
    np.round(np.linspace(0.15, 0.4, 21), 4),        # volfrac
    np.round(np.linspace(1.5, 4.0, 11), 4),         # rmin
    np.round(np.linspace(0, 1, 21), 4)              # forcedist
]

params = np.array(np.meshgrid(*all_params)).T.reshape(-1, len(all_params))

```

Here is the script I've used to upload the data to HF:

```python
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

output_path = os.path.join(data_dir, "_data.pkl")

# Load the Pickle file
with open(output_path, "rb") as f:
    design_dict = pickle.load(f)

# Hugging Face API instance
api = HfApi()

# Loop through each resolution and create a Hugging Face dataset
for resolution, data in design_dict.items():
    print(f"Processing resolution: {resolution}...")

    # Convert dictionary list to Hugging Face format
    dataset_list = data

    # Split dataset into Train (80%), Val (15%), Test (5%)
    train_data, temp_data = train_test_split(dataset_list, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.25, random_state=42)  # 15% val, 5% test

    # Convert to Hugging Face Dataset format
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # Define Hugging Face repository name dynamically based on resolution
    repo_name = f"IDEALLab/beams_2d_{resolution.replace('x', '_')}_v0"

    # Visualize one sample optimal_design
    sample = dataset_dict['train'][0]['optimal_design']
    plt.figure(figsize=(10, 5))
    sns.heatmap(sample)
    plt.title(f"Sample from {resolution} dataset")
    plt.show()

    # Upload dataset to Hugging Face
    dataset_dict.push_to_hub(repo_name)

    print(f"Dataset for {resolution} successfully uploaded to Hugging Face at {repo_name}!")

print("All datasets successfully uploaded to Hugging Face!")

```
